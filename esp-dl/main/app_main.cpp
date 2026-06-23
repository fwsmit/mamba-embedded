#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_timer.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

const char *TAG = "mamba_har";

//
// Embedded model binary symbols
//
extern const uint8_t model_espdl_start[] asm("_binary_model_espdl_start");
extern const uint8_t model_espdl_end[] asm("_binary_model_espdl_end");

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

static dl::Model *load_model(void) {
  auto *model = new dl::Model((const char *)model_espdl_start,
                              (size_t)(model_espdl_end - model_espdl_start));
  ESP_LOGI(TAG, "Model loaded");

  esp_err_t test_err = model->test();
  if (test_err != ESP_OK) {
    ESP_LOGW(TAG, "Model test failed with error 0x%x: %s", test_err,
             esp_err_to_name(test_err));
  }
  ESP_LOGI(TAG, "Model tested");

  return model;
}

// ---------------------------------------------------------------------------
// Dataset loading (from partition)
// ---------------------------------------------------------------------------

struct Dataset {
  const uint8_t *data; // start of sample data (after 8-byte header)
  uint32_t num_samples;
  uint32_t elements_per_sample;
  esp_partition_mmap_handle_t map_handle;
};

static bool load_dataset(Dataset *ds) {
  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_UNDEFINED, "dataset");
  if (!part) {
    ESP_LOGI(TAG, "No dataset partition found");
    return false;
  }
  ESP_LOGI(TAG, "Dataset partition found at 0x%x, size %lu bytes",
           part->address, (unsigned long)part->size);

  const void *map_ptr;
  esp_partition_mmap_handle_t map_handle;
  esp_err_t err = esp_partition_mmap(
      part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map_ptr, &map_handle);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to mmap dataset partition: 0x%x", err);
    return false;
  }

  const uint8_t *bytes = (const uint8_t *)map_ptr;
  ds->num_samples = *(const uint32_t *)bytes;
  ds->elements_per_sample = *(const uint32_t *)(bytes + 4);
  ds->data = bytes + 8;
  ds->map_handle = map_handle;

  ESP_LOGI(TAG, "Dataset: %lu samples, %lu elements per sample",
           (unsigned long)ds->num_samples,
           (unsigned long)ds->elements_per_sample);

  return true;
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

static float run_and_time(dl::Model *model, int num_runs) {
  int64_t total_us = 0;
  for (int i = 0; i < num_runs; ++i) {
    int64_t t_start = esp_timer_get_time();
    model->run();
    total_us += esp_timer_get_time() - t_start;
  }
  return (float)total_us / (float)num_runs;
}

// ---------------------------------------------------------------------------
// Dataset inference
// ---------------------------------------------------------------------------

static bool run_dataset_inference(dl::Model *model, const Dataset &ds,
                                  dl::TensorBase *input_tensor,
                                  dl::TensorBase *output_tensor,
                                  int num_classes,
                                  const std::vector<int> &input_shape,
                                  int input_exponent, dl::dtype_t input_dtype) {
  int n_samples = (int)ds.num_samples;
  ESP_LOGI(TAG, "Running inference on %d samples", n_samples);

  float scores[16]; // supports up to 16 classes
  if (num_classes > 16) {
    ESP_LOGE(TAG, "Too many classes: %d", num_classes);
    return false;
  }

  // Time the first sample separately (after warm-up)
  input_tensor->assign(input_shape, ds.data, input_exponent, input_dtype);
  model->run(); // warm-up
  float avg_us = run_and_time(model, 10);
  ESP_LOGI(TAG, "Average single-inference latency: %.1f us (%.3f ms)", avg_us,
           avg_us / 1000.0f);

  // Run on all dataset samples
  for (int i = 0; i < n_samples; ++i) {
    const void *sample_ptr = ds.data + i * ds.elements_per_sample;

    // Assign quantised sample data directly to the model's input tensor
    input_tensor->assign(input_shape, sample_ptr, input_exponent, input_dtype);

    // Run inference
    model->run();

    // Dequantise and print prediction for a few representative samples
    if (!dequantize_scores(output_tensor, scores, num_classes)) {
      ESP_LOGE(TAG, "Failed to dequantise output on sample %d", i);
      continue;
    }

    int predicted = find_argmax(scores, num_classes);

    if (i < 5 || i == n_samples - 1) {
      ESP_LOGI(TAG, "Sample %4d: prediction %d (confidence %.4f)", i, predicted,
               scores[predicted]);
    } else if (i == 5) {
      ESP_LOGI(TAG, "...");
    }
  }

  ESP_LOGI(TAG, "Inferred %d samples", n_samples);
  return true;
}

// ---------------------------------------------------------------------------
// Output decoding
// ---------------------------------------------------------------------------

static bool dequantize_scores(dl::TensorBase *output, float *scores,
                              int num_classes) {
  int exponent = output->get_exponent();
  float scale = (exponent > 0) ? (float)(1 << exponent)
                               : (1.0f / (float)(1 << -exponent));

  if (output->get_dtype() == dl::DATA_TYPE_INT8) {
    int8_t *raw = (int8_t *)output->data;
    for (int i = 0; i < num_classes; ++i)
      scores[i] = (float)raw[i] * scale;
    return true;
  } else if (output->get_dtype() == dl::DATA_TYPE_FLOAT) {
    float *raw = (float *)output->data;
    for (int i = 0; i < num_classes; ++i)
      scores[i] = raw[i];
    return true;
  }

  ESP_LOGE(TAG, "Unsupported output dtype: %s", output->get_dtype_string());
  return false;
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

static int find_argmax(const float *scores, int num_classes) {
  int idx = 0;
  float best = scores[0];
  for (int i = 1; i < num_classes; ++i) {
    if (scores[i] > best) {
      best = scores[i];
      idx = i;
    }
  }
  return idx;
}

// ---------------------------------------------------------------------------
// Profiling summary
// ---------------------------------------------------------------------------

static void print_grouped_profile(dl::Model *model) {
  ESP_LOGI(TAG, "Profiling model");
  // model->profile(true);
  auto mod_info = model->get_module_info();

  // Aggregate by op type: [type] -> {count, total_latency}
  std::map<std::string, std::pair<int, uint32_t>> grouped;
  for (const auto &kv : mod_info) {
    const std::string &type = kv.second.type;
    if (type.empty())
      continue; // skip "total" entry
    grouped[type].first++;
    grouped[type].second += kv.second.latency;
  }

  // Determine column widths
  size_t col0_w = strlen("type");
  size_t col1_w = strlen("count");
  size_t col2_w = strlen("total latency");
  size_t col3_w = strlen("avg latency");
  char buf[64];
  for (const auto &g : grouped) {
    col0_w = std::max(col0_w, g.first.size());
    snprintf(buf, sizeof(buf), "%d", g.second.first);
    col1_w = std::max(col1_w, strlen(buf));
#if DL_LOG_LATENCY_UNIT
    snprintf(buf, sizeof(buf), "%ldcycle", g.second.second);
#else
    snprintf(buf, sizeof(buf), "%ldus", g.second.second);
#endif
    col2_w = std::max(col2_w, strlen(buf));
    uint32_t avg = g.second.second / g.second.first;
#if DL_LOG_LATENCY_UNIT
    snprintf(buf, sizeof(buf), "%ldcycle", avg);
#else
    snprintf(buf, sizeof(buf), "%ldus", avg);
#endif
    col3_w = std::max(col3_w, strlen(buf));
  }

  // Sort by total latency descending
  std::vector<std::pair<std::string, std::pair<int, uint32_t>>> sorted(
      grouped.begin(), grouped.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    return a.second.second > b.second.second;
  });

  // Print grouped table
  std::string sep_str = "+-" + std::string(col0_w, '-') + "-+-" +
                        std::string(col1_w, '-') + "-+-" +
                        std::string(col2_w, '-') + "-+-" +
                        std::string(col3_w, '-') + "-+";
  ESP_LOGI(TAG, "");
  ESP_LOGI(TAG, "%s", sep_str.c_str());
  int title_pad = (sep_str.size() - 2 - 18) / 2;
  ESP_LOGI(TAG, "|%*s Grouped by type %*s|", title_pad, "",
           (int)(sep_str.size() - 2 - 18 - title_pad), "");
  ESP_LOGI(TAG, "%s", sep_str.c_str());
  ESP_LOGI(TAG, "| %-*s | %-*s | %-*s | %-*s |", (int)col0_w, "type",
           (int)col1_w, "count", (int)col2_w, "total latency", (int)col3_w,
           "avg latency");
  ESP_LOGI(TAG, "%s", sep_str.c_str());
  for (const auto &g : sorted) {
    uint32_t avg = g.second.second / g.second.first;
#if DL_LOG_LATENCY_UNIT
    ESP_LOGI(TAG, "| %-*s | %*d | %*ldcycle | %*ldcycle |", (int)col0_w,
             g.first.c_str(), (int)col1_w, g.second.first, (int)col2_w - 5,
             g.second.second, (int)col3_w - 5, avg);
#else
    ESP_LOGI(TAG, "| %-*s | %*d | %*ldus | %*ldus |", (int)col0_w,
             g.first.c_str(), (int)col1_w, g.second.first, (int)col2_w - 2,
             g.second.second, (int)col3_w - 2, avg);
#endif
  }
  ESP_LOGI(TAG, "%s", sep_str.c_str());
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

static void cleanup_model(dl::Model *model) {
  model->get_fbs_model()->clear_map();
  delete model;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "Starting HAR inference");

  //
  // Load model from embedded binary
  //
  dl::Model *model = load_model();

  //
  // Initialise the flatbuffer maps and get input / output info
  //
  model->get_fbs_model()->load_map();
  auto &graph_inputs = model->get_inputs();
  auto &graph_outputs = model->get_outputs();

  if (graph_inputs.empty() || graph_outputs.empty()) {
    ESP_LOGE(TAG, "Model has no inputs or outputs");
    cleanup_model(model);
    return;
  }

  dl::TensorBase *input_tensor = graph_inputs.begin()->second;
  dl::TensorBase *output_tensor = graph_outputs.begin()->second;

  std::vector<int> input_shape = input_tensor->get_shape();
  std::vector<int> output_shape = output_tensor->get_shape();
  int num_classes = output_shape.empty() ? 0 : output_shape.back();
  int input_exponent = input_tensor->get_exponent();
  dl::dtype_t input_dtype = input_tensor->get_dtype();

  ESP_LOGI(TAG, "Model input shape: [%d,%d,%d], exponent: %d, dtype: %s",
           input_shape[0], input_shape[1], input_shape[2], input_exponent,
           input_tensor->get_dtype_string());
  ESP_LOGI(TAG, "Model output classes: %d", num_classes);

  //
  // Load dataset from partition
  //
  Dataset ds;
  if (!load_dataset(&ds) || ds.num_samples == 0) {
    ESP_LOGE(TAG, "No dataset available");
    cleanup_model(model);
    return;
  }

  //
  // Run inference on dataset samples
  //
  if (!run_dataset_inference(model, ds, input_tensor, output_tensor,
                             num_classes, input_shape, input_exponent,
                             input_dtype)) {
    cleanup_model(model);
    return;
  }

  //
  // Profiling summary (grouped by op type)
  //
  print_grouped_profile(model);

  //
  // Cleanup
  //
  esp_partition_munmap(ds.map_handle);
  cleanup_model(model);

  ESP_LOGI(TAG, "INFERENCE_OK");
}
