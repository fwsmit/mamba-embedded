#include "dl_model_base.hpp"
// #include "dl_variable.hpp"
#include "esp_log.h"
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
extern const uint8_t
    har_mamba_1_espdl_start[] asm("_binary_har_mamba_1_espdl_start");
extern const uint8_t
    har_mamba_1_espdl_end[] asm("_binary_har_mamba_1_espdl_end");

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "Starting HAR inference");

  //
  // HAR input:
  // Shape = [1, 10, 57]
  // Total = 570 floats
  //
  constexpr int SEQ_LEN = 10;
  constexpr int FEATURE_DIM = 57;
  constexpr int INPUT_SIZE = SEQ_LEN * FEATURE_DIM;

  //
  // Create zero-filled input tensor
  //
  std::vector<float> input_data(INPUT_SIZE, 0.0f);

  //
  // Create ESP-DL tensor shape
  //
  std::vector<int> input_shape = {1, SEQ_LEN, FEATURE_DIM};

  //
  // Create input tensor
  //
  // Constructor signature: TensorBase(shape, data, exponent, dtype)
  // exponent = 0 is the identity scale factor for float32
  dl::TensorBase *input = new dl::TensorBase(input_shape, input_data.data(), 0,
                                             dl::DATA_TYPE_FLOAT);

  //
  // Load model from embedded binary
  //
  auto *model =
      new dl::Model((const char *)har_mamba_1_espdl_start,
                    (size_t)(har_mamba_1_espdl_end - har_mamba_1_espdl_start));

  ESP_LOGI(TAG, "Model loaded");

  ESP_ERROR_CHECK(model->test());

  ESP_LOGI(TAG, "Model tested");
  // Run inference — single-tensor overload; outputs retrieved via get_outputs()
  //
  int64_t t_start = esp_timer_get_time();
  model->run(input);
  int64_t t_end = esp_timer_get_time();
  int64_t elapsed_us = t_end - t_start;

  ESP_LOGI(TAG, "Inference completed in %lld us (%.2f ms)", elapsed_us,
           elapsed_us / 1000.0);

  //
  // Read output tensor
  //
  // Expected HAR output shape: [1, 6]
  //
  auto &output_map = model->get_outputs();

  if (output_map.empty()) {
    ESP_LOGE(TAG, "No output tensors");
    delete model;
    delete input;
    return;
  }

  // Take the first (and only) output regardless of its name
  auto *output = output_map.begin()->second;

  //
  // Dequantize output: int8 quantized with exponent scale
  // DL_SCALE(exponent) = (exponent > 0) ? (1 << exponent) : (1.0 / (1 <<
  // -exponent))
  //
  float scores[6];
  int exponent = output->get_exponent();
  float scale = (exponent > 0) ? (float)(1 << exponent)
                               : (1.0f / (float)(1 << -exponent));

  ESP_LOGI(TAG, "Output dtype: %s, exponent: %d, scale: %f",
           output->get_dtype_string(), exponent, scale);

  if (output->get_dtype() == dl::DATA_TYPE_INT8) {
    int8_t *raw = (int8_t *)output->data;
    for (int i = 0; i < 6; ++i) {
      scores[i] = (float)raw[i] * scale;
    }
  } else if (output->get_dtype() == dl::DATA_TYPE_FLOAT) {
    float *raw = (float *)output->data;
    for (int i = 0; i < 6; ++i) {
      scores[i] = raw[i];
    }
  } else {
    ESP_LOGE(TAG, "Unsupported output dtype");
    delete model;
    delete input;
    return;
  }

  //
  // Find argmax
  //
  int predicted_class = 0;
  float max_score = scores[0];

  for (int i = 1; i < 6; ++i) {
    if (scores[i] > max_score) {
      max_score = scores[i];
      predicted_class = i;
    }
  }

  ESP_LOGI(TAG, "Prediction: %d", predicted_class);
  ESP_LOGI(TAG, "Confidence: %f", max_score);

  //
  // Print all logits
  //
  for (int i = 0; i < 6; ++i) {
    ESP_LOGI(TAG, "Class %d score: %f", i, scores[i]);
  }

  ESP_LOGI(TAG, "Profiling model");
  model->profile(true);

  //
  // Grouped-by-type profiling summary
  //
  {
    auto mod_info = model->get_module_info();
    // Aggregate by op type: [type] -> {count, total_latency}
    std::map<std::string, std::pair<int, uint32_t>> grouped;
    for (const auto &kv : mod_info) {
      const std::string &type = kv.second.type;
      if (type.empty()) continue; // skip "total" entry
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
    std::vector<std::pair<std::string, std::pair<int, uint32_t>>> sorted(grouped.begin(), grouped.end());
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
    ESP_LOGI(TAG, "|%*s Grouped by type %*s|", title_pad, "", (int)(sep_str.size() - 2 - 18 - title_pad), "");
    ESP_LOGI(TAG, "%s", sep_str.c_str());
    ESP_LOGI(TAG, "| %-*s | %-*s | %-*s | %-*s |",
             (int)col0_w, "type",
             (int)col1_w, "count",
             (int)col2_w, "total latency",
             (int)col3_w, "avg latency");
    ESP_LOGI(TAG, "%s", sep_str.c_str());
    for (const auto &g : sorted) {
      uint32_t avg = g.second.second / g.second.first;
#if DL_LOG_LATENCY_UNIT
      ESP_LOGI(TAG, "| %-*s | %*d | %*ldcycle | %*ldcycle |",
               (int)col0_w, g.first.c_str(),
               (int)col1_w, g.second.first,
               (int)col2_w - 5, g.second.second,
               (int)col3_w - 5, avg);
#else
      ESP_LOGI(TAG, "| %-*s | %*d | %*ldus | %*ldus |",
               (int)col0_w, g.first.c_str(),
               (int)col1_w, g.second.first,
               (int)col2_w - 2, g.second.second,
               (int)col3_w - 2, avg);
#endif
    }
    ESP_LOGI(TAG, "%s", sep_str.c_str());
  }

  //
  // Cleanup
  //
  // Output tensors are owned by the model; do not delete them manually.
  delete model;
  delete input;

  ESP_LOGI(TAG, "Finished");
  ESP_LOGI(TAG, "INFERENCE_OK");
}
