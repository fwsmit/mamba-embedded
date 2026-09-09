#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"

#include <cstring>
#include <vector>

const char *TAG = "mamba_acc";

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

  auto &inputs = model->get_inputs();
  bool is_float =
      inputs.empty() || inputs.begin()->second->get_dtype() == dl::DATA_TYPE_FLOAT;
  if (is_float) {
    ESP_LOGI(TAG, "Float model detected, skipping model test");
  } else {
    esp_err_t test_err = model->test();
    if (test_err != ESP_OK) {
      ESP_LOGW(TAG, "Model test failed with error 0x%x: %s", test_err,
               esp_err_to_name(test_err));
    }
    ESP_LOGI(TAG, "Model tested");
  }

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
// Inference helpers
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

static bool predict(dl::Model *model, dl::TensorBase *input_tensor,
                    dl::TensorBase *output_tensor,
                    const std::vector<int> &input_shape, int input_exponent,
                    dl::dtype_t input_dtype, const void *sample_ptr,
                    int num_classes, int *predicted, float *confidence) {
  input_tensor->assign(input_shape, sample_ptr, input_exponent, input_dtype);
  model->run();

  int exponent = output_tensor->get_exponent();
  float scale = (exponent > 0) ? (float)(1 << exponent)
                               : (1.0f / (float)(1 << -exponent));

  float scores[16] = {0.0f}; // supports up to 16 classes
  if (num_classes > 16) {
    ESP_LOGE(TAG, "Too many classes: %d", num_classes);
    return false;
  }

  if (output_tensor->get_dtype() == dl::DATA_TYPE_INT8) {
    int8_t *raw = (int8_t *)output_tensor->data;
    for (int i = 0; i < num_classes; ++i)
      scores[i] = (float)raw[i] * scale;
  } else if (output_tensor->get_dtype() == dl::DATA_TYPE_INT16) {
    int16_t *raw = (int16_t *)output_tensor->data;
    for (int i = 0; i < num_classes; ++i)
      scores[i] = (float)raw[i] * scale;
  } else if (output_tensor->get_dtype() == dl::DATA_TYPE_FLOAT) {
    float *raw = (float *)output_tensor->data;
    for (int i = 0; i < num_classes; ++i)
      scores[i] = raw[i];
  } else {
    ESP_LOGE(TAG, "Unsupported output dtype: %s",
             output_tensor->get_dtype_string());
    return false;
  }

  *predicted = find_argmax(scores, num_classes);
  *confidence = scores[*predicted];
  return true;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "Starting accuracy test");

  dl::Model *model = load_model();

  model->get_fbs_model()->load_map();
  auto &graph_inputs = model->get_inputs();
  auto &graph_outputs = model->get_outputs();

  if (graph_inputs.empty() || graph_outputs.empty()) {
    ESP_LOGE(TAG, "Model failed to load: no inputs or outputs registered");
    printf("INFERENCE_FAIL\n");
    delete model;
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
  ESP_LOGI(TAG, "Model output classes: %d (dtype %s)", num_classes,
           output_tensor->get_dtype_string());

  Dataset ds;
  if (!load_dataset(&ds) || ds.num_samples == 0) {
    ESP_LOGE(TAG, "No dataset available");
    printf("INFERENCE_FAIL\n");
    delete model;
    return;
  }

  // Warm-up + average single-inference latency, printed once as a sanity stat
  input_tensor->assign(input_shape, ds.data, input_exponent, input_dtype);
  model->run();
  int64_t total_us = 0;
  const int num_runs = 10;
  for (int i = 0; i < num_runs; ++i) {
    int64_t t_start = esp_timer_get_time();
    model->run();
    total_us += esp_timer_get_time() - t_start;
  }
  ESP_LOGI(TAG, "Average single-inference latency: %.1f us (%.3f ms)",
           (float)total_us / (float)num_runs,
           (float)total_us / (float)num_runs / 1000.0f);

  //
  // Accuracy probe: run predictions on the first N dataset samples and print
  // them machine-readably. The host scores them against ground-truth labels.
  //
  const int N = 100;
  uint32_t n = ds.num_samples < (uint32_t)N ? ds.num_samples : (uint32_t)N;

  // Prevent the idle-task watchdog from rebooting us mid-loop
  esp_task_wdt_delete(xTaskGetIdleTaskHandleForCore(0));

  printf("===PREDICTIONS_START===\n");
  printf("%lu\n", (unsigned long)n);
  for (uint32_t i = 0; i < n; ++i) {
    int predicted;
    float confidence;
    // Byte offset: elements_per_sample is an element count, so scale by the
    // input element size (2 for int16 datasets, 1 for int8).
    size_t sample_bytes = ds.elements_per_sample *
                          (input_dtype == dl::DATA_TYPE_INT16 ? 2 : 1);
    const void *sample_ptr = ds.data + i * sample_bytes;
    if (!predict(model, input_tensor, output_tensor, input_shape,
                 input_exponent, input_dtype, sample_ptr, num_classes,
                 &predicted, &confidence)) {
      ESP_LOGE(TAG, "Prediction failed on sample %lu", (unsigned long)i);
      printf("%d %f\n", -1, 0.0f);
      continue;
    }
    printf("%d %f\n", predicted, confidence);
  }
  printf("===PREDICTIONS_END===\n");

  esp_task_wdt_add(xTaskGetIdleTaskHandleForCore(0));

  esp_partition_munmap(ds.map_handle);
  delete model;

  ESP_LOGI(TAG, "INFERENCE_OK");
}