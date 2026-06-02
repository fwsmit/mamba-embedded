#include "dl_model_base.hpp"
// #include "dl_variable.hpp"
#include "esp_log.h"

#include <cstring>
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

  // ESP_ERROR_CHECK(model->test());
  //
  // ESP_LOGI(TAG, "Model tested");
  //
  // Run inference — single-tensor overload; outputs retrieved via get_outputs()
  //
  model->run(input);

  ESP_LOGI(TAG, "Inference completed");

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

  float *scores = (float *)output->data;

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
  // Optional: print all logits
  //
  for (int i = 0; i < 6; ++i) {
    ESP_LOGI(TAG, "Class %d score: %f", i, scores[i]);
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
