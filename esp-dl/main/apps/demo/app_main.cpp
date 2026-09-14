/*
 * I2S microphone demo for the ESP32-S3-EYE: live Speech-Commands keyword
 * classification with the embedded model.espdl (a KWS model, e.g.
 * demo.espdl / experiments/mamba-1-kws-2 trial-55 espdl).
 *
 * The mic is captured at 16 kHz (the same sample rate the Speech-Commands
 * dataset was recorded at; the INMP441 supports 16 kHz) and each second of
 * audio is converted to MFCC features with the exact same processing as the
 * offline dataset (train/kws_dataset_gen.py): torchaudio-style MFCC with
 * n_fft=640 (40 ms), hop_length=320 (20 ms), n_mels=40, center=false,
 * power=2.0, log_mels=true, f_min=0, f_max=8000, followed by PPQ int8
 * quantization (q = clamp(round(x * 2))) matching dataset-trial-*.bin and
 * the model's input tensor (scale 2^-9). See mfcc_kws.c for the port.
 *
 * The 12 output classes are the dataset's label order:
 *   down go left no off on right stop up yes silence unknown
 *
 * Wiring (INMP441 pin on the left, ESP32-S3-EYE GPIO on the right; pin
 * numbers are the BSP_I2S_* definitions in bsp/esp32_s3_eye.h):
 *
 *   INMP441              ESP32-S3-EYE
 *   -------              -------------
 *   SCK      <---------  GPIO 41   (BCLK, driven by the MCU)
 *   WS       <---------  GPIO 42   (word select, driven by the MCU)
 *   SD       --------->  GPIO 2    (serial data, 24-bit audio in 32-bit words)
 *   VDD      <---------  3V3
 *   GND      <---------  GND
 *   L/R      <---------  GND or 3V3
 *   (L/R selects which I2S slot carries the data; both slots are read and
 *    the louder one is used, so either position works)
 *
 * Mic stats (active channel, RMS, peak, waveform stamp) are printed once per
 * inference cycle, over the freshly captured audio. Inference runs on a
 * sliding 1 s window at up to 6 Hz (every ~166.7 ms); models whose MFCC +
 * inference time exceeds the 166.7 ms budget simply infer less often - the
 * mic audio is always consumed at the full 16 kHz rate, so no audio is ever
 * dropped. Each cycle the recognised keyword, softmax confidence and raw
 * (pre-softmax) output logits are printed.
 */

#include "dl_model_base.hpp"
#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "bsp/esp32_s3_eye.h"
#include "driver/i2s_std.h"
#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"
#include "mfcc_kws.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

const char *TAG = "kws_demo";

#define SAMPLE_RATE_HZ 16000 /* Speech-Commands sample rate */
#define FRAME_BYTES    8     /* 2 channels x 32-bit I2S slots */
#define STEP_SAMPLES  2666 /* minimum audio per inference: ~166.7 ms at 16 kHz */
#define DMA_FRAMES    (8 * 511) /* I2S DMA ring: 8 descriptors x 511 stereo frames */
#define FULL_SCALE     8388607.0 /* 2^23 - 1 (24-bit audio in 32-bit word) */

#define WAVE_SAMPLES 24

//
// Embedded model binary symbols
//
extern const uint8_t model_espdl_start[] asm("_binary_model_espdl_start");
extern const uint8_t model_espdl_end[] asm("_binary_model_espdl_end");

//
// Speech-Commands label order (kws_dataset_gen.ALL_LABELS)
//
static const char *KWS_LABELS[12] = {
    "down", "go",   "left",  "no",   "off",  "on",
    "right", "stop", "up",    "yes",  "silence", "unknown",
};

static uint8_t raw_buf[DMA_FRAMES * FRAME_BYTES];
static int16_t ch_l[KWS_CLIP_SAMPLES];
static int16_t ch_r[KWS_CLIP_SAMPLES];
static float feats[KWS_N_FRAMES * KWS_N_MFCC];
static int8_t quant[KWS_N_FRAMES * KWS_N_MFCC];
static int16_t quant16[KWS_N_FRAMES * KWS_N_MFCC];

static void wave_stamp(const int16_t *ch, char *out) {
  for (int i = 0; i < WAVE_SAMPLES; ++i) {
    int v = (int)ch[i] * WAVE_SAMPLES / 32767;
    if (v < -WAVE_SAMPLES) v = -WAVE_SAMPLES;
    if (v > WAVE_SAMPLES) v = WAVE_SAMPLES;
    out[i] = (v >= 0) ? (char)('0' + v) : (char)('A' - v - 1);
  }
  out[WAVE_SAMPLES] = '\0';
}

static dl::Model *load_model(void) {
  auto *model = new dl::Model((const char *)model_espdl_start,
                              (size_t)(model_espdl_end - model_espdl_start));
  ESP_LOGI(TAG, "Model loaded");
  /* model->test() is skipped: the .espdl has no embedded test outputs. */
  return model;
}

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
  } else if (output->get_dtype() == dl::DATA_TYPE_INT16) {
    int16_t *raw = (int16_t *)output->data;
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

extern "C" void app_main(void) {
  ESP_LOGI(TAG, "=== I2S keyword classifier demo ===");
  ESP_LOGI(TAG, "pins: BCLK->INMP441.SCK (GPIO %d), WS->INMP441.WS (GPIO %d), "
                "INMP441.SD->DATA-IN (GPIO %d)",
           (int)BSP_I2S_SCLK, (int)BSP_I2S_LCLK, (int)BSP_I2S_DSIN);
  ESP_LOGI(TAG, "config: %d Hz, 2x32-bit slots (24-bit audio, MSB-justified)",
           SAMPLE_RATE_HZ);

  i2s_std_config_t mic_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
      .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT,
                                                      I2S_SLOT_MODE_STEREO),
      .gpio_cfg =
          {
              .mclk = BSP_I2S_MCLK,
              .bclk = BSP_I2S_SCLK,
              .ws = BSP_I2S_LCLK,
              .dout = BSP_I2S_DOUT,
              .din = BSP_I2S_DSIN,
              .invert_flags =
                  {
                      .mclk_inv = false,
                      .bclk_inv = false,
                      .ws_inv = false,
                  },
          },
  };

  /* Own I2S RX channel with a DMA ring larger than the per-cycle compute
   * time (~150 ms of MFCC + inference), so audio produced while we compute
   * is buffered instead of dropped (the BSP default ring is only ~90 ms).
   * The channel is wrapped in esp_codec_dev exactly like the BSP does. */
  i2s_chan_config_t chan_cfg =
      I2S_CHANNEL_DEFAULT_CONFIG(CONFIG_BSP_I2S_NUM, I2S_ROLE_MASTER);
  chan_cfg.auto_clear = true;
  chan_cfg.dma_desc_num = 8;
  chan_cfg.dma_frame_num = 511; /* 8 x 511 stereo frames ~ 255 ms ring */
  i2s_chan_handle_t rx_chan;
  if (i2s_new_channel(&chan_cfg, NULL, &rx_chan) != ESP_OK ||
      i2s_channel_init_std_mode(rx_chan, &mic_cfg) != ESP_OK) {
    ESP_LOGE(TAG, "I2S RX channel init failed");
    printf("MIC_TEST_FAIL\n");
    return;
  }

  audio_codec_i2s_cfg_t i2s_cfg = {
      .port = CONFIG_BSP_I2S_NUM,
      .rx_handle = rx_chan,
      .tx_handle = NULL,
      .clk_src = I2S_CLK_SRC_DEFAULT,
  };
  const audio_codec_data_if_t *data_if = audio_codec_new_i2s_data(&i2s_cfg);
  esp_codec_dev_cfg_t codec_dev_cfg = {
      .dev_type = ESP_CODEC_DEV_TYPE_IN,
      .codec_if = NULL,
      .data_if = data_if,
  };
  esp_codec_dev_handle_t mic = esp_codec_dev_new(&codec_dev_cfg);
  if (mic == NULL) {
    ESP_LOGE(TAG, "esp_codec_dev_new failed");
    printf("MIC_TEST_FAIL\n");
    return;
  }

  esp_codec_dev_sample_info_t fs = {
      .bits_per_sample = 32,
      .channel = 2, /* capture both I2S slots */
      .channel_mask = ESP_CODEC_DEV_MAKE_CHANNEL_MASK(0) |
                      ESP_CODEC_DEV_MAKE_CHANNEL_MASK(1),
      .sample_rate = SAMPLE_RATE_HZ,
      .mclk_multiple = 0,
  };
  if (esp_codec_dev_open(mic, &fs) != ESP_CODEC_DEV_OK) {
    ESP_LOGE(TAG, "esp_codec_dev_open failed");
    printf("MIC_TEST_FAIL\n");
    return;
  }

  //
  // Load model from embedded binary and wire up input / output tensors
  //
  dl::Model *model = load_model();
  model->get_fbs_model()->load_map();
  auto &graph_inputs = model->get_inputs();
  auto &graph_outputs = model->get_outputs();

  if (graph_inputs.empty() || graph_outputs.empty()) {
    ESP_LOGE(TAG, "Model failed to load: no inputs or outputs registered");
    printf("INFERENCE_FAIL\n");
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

  if (num_classes < 1 || num_classes > 12) {
    ESP_LOGE(TAG, "Unexpected class count: %d (demo supports 1..12)", num_classes);
    printf("INFERENCE_FAIL\n");
    return;
  }

  float scores[12];
  uint32_t total_read = 0;

  for (uint32_t block = 1;; ++block) {
    /* Consume exactly the audio the mic produced since the last cycle (but
     * at least STEP_SAMPLES): a non-blocking read drains the DMA ring, a
     * blocking top-up refills to the 6 Hz step when the ring ran dry. This
     * keeps consumption equal to production, so the ring never overruns and
     * no audio is lost regardless of the model's inference latency - slow
     * models simply infer less often, on always-contiguous audio. */
    size_t got = 0;
    esp_err_t rerr = i2s_channel_read(rx_chan, raw_buf, sizeof(raw_buf), &got, 0);
    if (rerr != ESP_OK && rerr != ESP_ERR_TIMEOUT) {
      ESP_LOGW(TAG, "read failed (0x%x) - mic connected and powered?", rerr);
      vTaskDelay(50 / portTICK_PERIOD_MS);
      continue;
    }
    size_t n = got / FRAME_BYTES;
    if (n < STEP_SAMPLES) {
      if (esp_codec_dev_read(mic, raw_buf + got,
                             (STEP_SAMPLES - n) * FRAME_BYTES) != ESP_CODEC_DEV_OK) {
        ESP_LOGW(TAG, "read failed (timeout?) - mic connected and powered?");
        vTaskDelay(50 / portTICK_PERIOD_MS);
        continue;
      }
      n = STEP_SAMPLES;
    } else if (n > STEP_SAMPLES + SAMPLE_RATE_HZ / 100) {
      /* Overloaded model: the drain never blocks, so the task would spin at
       * 100% CPU and starve the IDLE task (task watchdog). */
      vTaskDelay(1);
    }
    bool filling = total_read < KWS_CLIP_SAMPLES; /* discard startup audio */
    total_read += n;

    /* Slide the 1 s window forward and decode the fresh audio into its tail. */
    memmove(ch_l, ch_l + n, (KWS_CLIP_SAMPLES - n) * sizeof(int16_t));
    memmove(ch_r, ch_r + n, (KWS_CLIP_SAMPLES - n) * sizeof(int16_t));
    int16_t *dst_l = ch_l + (KWS_CLIP_SAMPLES - n);
    int16_t *dst_r = ch_r + (KWS_CLIP_SAMPLES - n);

    //
    // Decode both 24-bit slots (MSB-justified in 32-bit words) to int16 PCM
    //
    int64_t chunk_sq_l = 0, chunk_sq_r = 0;
    int32_t cmin_l = 0, cmax_l = 0, cmin_r = 0, cmax_r = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint8_t *p = raw_buf + i * FRAME_BYTES;
      int32_t wl = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                             ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
      int32_t wr = (int32_t)((uint32_t)p[4] | ((uint32_t)p[5] << 8) |
                             ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24));
      int32_t sl = wl >> 8; /* 24-bit audio, MSB-justified in the word */
      int32_t sr = wr >> 8;
      dst_l[i] = (int16_t)(sl >> 8);
      dst_r[i] = (int16_t)(sr >> 8);
      chunk_sq_l += (int64_t)sl * sl;
      chunk_sq_r += (int64_t)sr * sr;
      if (sl < cmin_l) cmin_l = sl;
      if (sl > cmax_l) cmax_l = sl;
      if (sr < cmin_r) cmin_r = sr;
      if (sr > cmax_r) cmax_r = sr;
    }

    if (filling) continue;

    /* Mic stats over the fresh audio, once per inference cycle (same line
     * format as before so the host dashboard parses it unchanged). */
    bool chunk_r = chunk_sq_r > chunk_sq_l;
    double chunk_rms = sqrt((double)(chunk_r ? chunk_sq_r : chunk_sq_l));
    char wave[WAVE_SAMPLES + 1];
    wave_stamp(chunk_r ? dst_r : dst_l, wave);
    printf("active %s | rms %7.1f dBFS | peak %+5.1f/%5.1f dBFS | wave %s\n",
           chunk_r ? "R" : "L",
           (chunk_rms > 0.0) ? 20.0 * log10(chunk_rms / FULL_SCALE) : -INFINITY,
           20.0 * log10((double)(-(chunk_r ? cmin_r : cmin_l)) / FULL_SCALE),
           20.0 * log10((double)(chunk_r ? cmax_r : cmax_l) / FULL_SCALE), wave);

    /* The INMP441 transmits on only one slot; use the louder one. */
    bool active_r = chunk_sq_r > chunk_sq_l;
    const int16_t *pcm = active_r ? ch_r : ch_l;

    //
    // Extract MFCC features exactly like the offline Speech-Commands
    // pipeline, then quantize to the model's input scale. The input tensor
    // width follows the quantization method: int8 models (PPQ scale 0.5,
    // exponent -1) use q = round(f * 2); int16 models (e.g. exponent -9)
    // use q = round(f * 2^-exponent) = round(f * 512). Both are the PPQ
    // formula q = clip(round(x / scale - offset)) with offset 0.
    //
    int64_t tmfcc = esp_timer_get_time();
    kws_mfcc(pcm, feats);
    float mfcc_ms = (float)(esp_timer_get_time() - tmfcc) / 1000.0f;
    if (input_dtype == dl::DATA_TYPE_INT16) {
      float scale = (input_exponent > 0)
                        ? (float)(1 << input_exponent)
                        : (1.0f / (float)(1 << -input_exponent));
      float inv_scale = 1.0f / scale;
      for (int i = 0; i < KWS_N_FRAMES * KWS_N_MFCC; ++i) {
        float v = feats[i] * inv_scale;
        if (v > 32767.0f) v = 32767.0f;
        if (v < -32768.0f) v = -32768.0f;
        quant16[i] = (int16_t)roundf(v);
      }
      input_tensor->assign(input_shape, quant16, input_exponent, input_dtype);
    } else {
      kws_mfcc_quantize(feats, quant);
      input_tensor->assign(input_shape, quant, input_exponent, input_dtype);
    }

    int64_t t0 = esp_timer_get_time();
    model->run();
    float latency_ms = (float)(esp_timer_get_time() - t0) / 1000.0f;

    int predicted = -1;
    if (dequantize_scores(output_tensor, scores, num_classes)) {
      float best = scores[0];
      predicted = 0;
      for (int k = 1; k < num_classes; ++k) {
        if (scores[k] > best) {
          best = scores[k];
          predicted = k;
        }
      }
    }

    /* Softmax confidence over the raw scores. */
    float conf = -1.0f;
    if (predicted >= 0) {
      float mx = scores[0], sum = 0.0f;
      for (int k = 1; k < num_classes; ++k)
        if (scores[k] > mx) mx = scores[k];
      for (int k = 0; k < num_classes; ++k)
        sum += expf(scores[k] - mx);
      conf = expf(scores[predicted] - mx) / sum;
    }

    printf("=== KWS block %lu (%.3f s, %d frames) ===\n",
           (unsigned long)block, (double)KWS_CLIP_SAMPLES / SAMPLE_RATE_HZ,
           KWS_N_FRAMES);
    printf("mfcc %.1f ms\n", mfcc_ms);
    printf("pred %d %-7s | conf %.2f | latency %.1f ms\n", predicted,
           predicted >= 0 ? KWS_LABELS[predicted] : "?", conf, latency_ms);
    /* Raw (pre-softmax) output scores so the host can tune its own
     * minimum-confidence gate without reflashing. */
    if (predicted >= 0) {
      printf("logits");
      for (int k = 0; k < num_classes; ++k)
        printf(" %+.2f", scores[k]);
      printf("\n");
    }
  }
}