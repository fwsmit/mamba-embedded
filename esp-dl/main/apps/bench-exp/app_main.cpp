/*
 * Benchmark: expf() vs the esp-dl 8-bit exp lookup table.
 *
 * The LUT is built exactly like the softmax / log-softmax modules do it
 * (dl::tool::gen_lut_8bit with expf at the input exponent), then both paths
 * process the same int8 input array many times, timed per pass with
 * esp_timer_get_time(). A volatile sink prevents dead-code elimination.
 */

#include "dl_tool.hpp"
#include "esp_timer.h"
#include "sdkconfig.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define N_ELEMS 4096
#define REPS 2000
#define EXPONENT (-3)

static int8_t in_data[N_ELEMS];
static float out_a[N_ELEMS];
static float out_b[N_ELEMS];
static float exp_table[256] __attribute__((aligned(16)));

static volatile float sink;

static void fill_inputs(void) {
  uint32_t x = 0x12345678u;
  for (int i = 0; i < N_ELEMS; i++) {
    x = x * 1664525u + 1013904223u;
    in_data[i] = (int8_t)(x >> 24);
  }
}

static void run_lut(void) {
  float acc = 0.f;
  for (int i = 0; i < N_ELEMS; i++)
    acc += exp_table[in_data[i] + 128];
  sink = acc;
}

static void run_expf(void) {
  const float scale = DL_SCALE(EXPONENT);
  float acc = 0.f;
  for (int i = 0; i < N_ELEMS; i++)
    acc += expf(scale * (float)in_data[i]);
  sink = acc;
}

static void report(const char *name, float *samples) {
  float sorted[REPS];
  memcpy(sorted, samples, sizeof(sorted));
  for (int i = 1; i < REPS; i++) {
    float v = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > v) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = v;
  }
  float sum = 0.f;
  for (int i = 0; i < REPS; i++)
    sum += samples[i];
  float mean = sum / REPS;
  float var = 0.f;
  for (int i = 0; i < REPS; i++) {
    float d = samples[i] - mean;
    var += d * d;
  }
  float std = sqrtf(var / (REPS - 1));
  printf("%s_us mean %.3f median %.3f std %.3f min %.3f max %.3f p05 %.3f p95 %.3f\n",
         name, mean, sorted[REPS / 2], std, sorted[0], sorted[REPS - 1],
         sorted[REPS / 20], sorted[REPS - 1 - REPS / 20]);
  printf("%s_ns_per_elem mean %.2f median %.2f min %.2f max %.2f\n",
         name, mean * 1000.f / N_ELEMS, sorted[REPS / 2] * 1000.f / N_ELEMS,
         sorted[0] * 1000.f / N_ELEMS, sorted[REPS - 1] * 1000.f / N_ELEMS);
}

extern "C" void app_main(void) {
  printf("cpu_freq_hz %ld\n", (long)CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ * 1000000L);
  printf("elements %d reps %d exponent %d\n", N_ELEMS, REPS, EXPONENT);

  fill_inputs();
  dl::tool::gen_lut_8bit(exp_table, EXPONENT, [](float x) { return expf(x); });

  /* Accuracy of the LUT vs expf over the full int8 range. */
  const float scale = DL_SCALE(EXPONENT);
  float max_abs = 0.f, max_rel = 0.f, sum_abs = 0.f;
  for (int i = -128; i < 128; i++) {
    float ref = expf(scale * (float)i);
    float got = exp_table[i + 128];
    float d = fabsf(got - ref);
    sum_abs += d;
    if (d > max_abs)
      max_abs = d;
    if (ref > 1e-6f && d / ref > max_rel)
      max_rel = d / ref;
  }
  printf("lut_accuracy max_abs %.3g mean_abs %.3g max_rel %.3g\n",
         max_abs, sum_abs / 256.f, max_rel);

  for (int i = 0; i < N_ELEMS; i++) {
    out_a[i] = out_b[i] = 0.f;
  }
  for (int w = 0; w < 50; w++) {
    run_lut();
    run_expf();
  }

  static float t_lut[REPS], t_expf[REPS];
  for (int r = 0; r < REPS; r++) {
    int64_t t0 = esp_timer_get_time();
    run_lut();
    int64_t t1 = esp_timer_get_time();
    run_expf();
    int64_t t2 = esp_timer_get_time();
    t_lut[r] = (float)(t1 - t0);
    t_expf[r] = (float)(t2 - t1);
  }

  report("lut", t_lut);
  report("expf", t_expf);
  printf("speedup_median %.2f\n",
         t_expf[REPS / 2] == 0 ? 0.f : (float)t_expf[REPS / 2] / t_lut[REPS / 2]);
  printf("INFERENCE_OK\n");
}
