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
    model_espdl_start[] asm("_binary_model_espdl_start");
extern const uint8_t
    model_espdl_end[] asm("_binary_model_espdl_end");

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

static dl::Model *load_model(void) {
    auto *model = new dl::Model((const char *)model_espdl_start,
                                (size_t)(model_espdl_end - model_espdl_start));
    ESP_LOGI(TAG, "Model loaded");

    esp_err_t test_err = model->test();
    if (test_err != ESP_OK) {
        ESP_LOGW(TAG, "Model test failed with error 0x%x: %s",
                 test_err, esp_err_to_name(test_err));
    }
    ESP_LOGI(TAG, "Model tested");

    return model;
}

static dl::TensorBase *load_test_input(dl::Model *model) {
    model->get_fbs_model()->load_map();
    auto &graph_inputs = model->get_inputs();
    std::string input_name = graph_inputs.begin()->first;
    dl::TensorBase *test_input =
        model->get_fbs_model()->get_test_input_tensor(input_name);
    graph_inputs.begin()->second->assign(test_input);
    return test_input;
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
// Output decoding
// ---------------------------------------------------------------------------

static bool dequantize_scores(dl::TensorBase *output,
                               float *scores, int num_classes) {
    int exponent = output->get_exponent();
    float scale = (exponent > 0) ? (float)(1 << exponent)
                                 : (1.0f / (float)(1 << -exponent));

    ESP_LOGI(TAG, "Output dtype: %s, exponent: %d, scale: %f",
             output->get_dtype_string(), exponent, scale);

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

    ESP_LOGE(TAG, "Unsupported output dtype");
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
    model->profile(true);
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
    std::vector<std::pair<std::string, std::pair<int, uint32_t>>> sorted(
        grouped.begin(), grouped.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto &a, const auto &b) {
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
    ESP_LOGI(TAG, "|%*s Grouped by type %*s|",
             title_pad, "",
             (int)(sep_str.size() - 2 - 18 - title_pad), "");
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

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

static void cleanup_model(dl::Model *model, dl::TensorBase *test_input) {
    delete test_input;
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
    // Load the baked-in test input and assign it to the model's input tensor
    //
    dl::TensorBase *test_input = load_test_input(model);

    //
    // Run inference
    //
    float avg_us = run_and_time(model, 10);

    ESP_LOGI(TAG, "Average inference over 10 runs: %.1f us (%.3f ms)",
             avg_us, avg_us / 1000.0f);

    //
    // Read output tensor
    //
    // Expected HAR output shape: [1, 6]
    //
    auto &output_map = model->get_outputs();

    if (output_map.empty()) {
        ESP_LOGE(TAG, "No output tensors");
        cleanup_model(model, test_input);
        return;
    }

    // Take the first (and only) output regardless of its name
    dl::TensorBase *output = output_map.begin()->second;

    //
    // Dequantize output: int8 quantized with exponent scale
    // DL_SCALE(exponent) = (exponent > 0) ? (1 << exponent) : (1.0 / (1 << -exponent))
    //
    float scores[6];
    if (!dequantize_scores(output, scores, 6)) {
        cleanup_model(model, test_input);
        return;
    }

    //
    // Find argmax and print results
    //
    int predicted_class = find_argmax(scores, 6);

    ESP_LOGI(TAG, "Prediction: %d", predicted_class);
    ESP_LOGI(TAG, "Confidence: %f", scores[predicted_class]);

    for (int i = 0; i < 6; ++i) {
        ESP_LOGI(TAG, "Class %d score: %f", i, scores[i]);
    }

    //
    // Profiling summary (grouped by op type)
    //
    print_grouped_profile(model);

    //
    // Cleanup
    //
    cleanup_model(model, test_input);

    ESP_LOGI(TAG, "INFERENCE_OK");
}