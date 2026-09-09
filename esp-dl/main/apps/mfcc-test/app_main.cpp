/*
 * MFCC round-trip test for the demo feature extractor.
 *
 * No microphone, no model: the 16000 int16 samples of one Speech-Commands
 * test utterance (exported by tools/mfcc_test/export_kws_waveform.py) are
 * converted to MFCC by the exact same kws_mfcc() the I2S demo uses, and the
 * result is compared on-device against the reference features from
 * test.pkl (embedded in ref.h, produced by the same script).
 *
 * The full 49x40 float32 array is printed between ===MFCC_START===
 * / ===MFCC_END=== markers so the host can also compare offline.
 */

#include "mfcc_kws.h"
#include "pcm.h"
#include "ref.h"

#include <stdio.h>

extern "C" void app_main(void) {
  /* Static, like the demo: the main task stack is far too small for these. */
  static float feats[KWS_N_FRAMES * KWS_N_MFCC];
  static int8_t q[KWS_N_FRAMES * KWS_N_MFCC];
  kws_mfcc(kws_test_pcm, feats);

  /* On-device comparison against the offline reference features. */
  float max_abs = 0.0f, sum_abs = 0.0f;
  int worst = 0;
  for (int i = 0; i < KWS_N_FRAMES * KWS_N_MFCC; ++i) {
    float d = feats[i] - kws_test_ref_mfcc[i];
    if (d < 0.0f) d = -d;
    sum_abs += d;
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
  }
  float mean_abs = sum_abs / (float)(KWS_N_FRAMES * KWS_N_MFCC);

  /* The int8 model input: how many coefficients quantize differently. */
  kws_mfcc_quantize(feats, q);
  int n_diff_q = 0;
  for (int i = 0; i < KWS_N_FRAMES * KWS_N_MFCC; ++i)
    if (q[i] != kws_test_ref_quant[i]) ++n_diff_q;

  printf("===MFCC_START===\n");
  for (int i = 0; i < KWS_N_FRAMES * KWS_N_MFCC; ++i)
    printf("%.9g\n", feats[i]);
  printf("===MFCC_END===\n");

  printf("DIAG max_abs_diff %.9g mean_abs_diff %.9g quant_diff %d of %d (worst idx %d)\n",
         max_abs, mean_abs, n_diff_q, KWS_N_FRAMES * KWS_N_MFCC, worst);
  printf("INFERENCE_OK\n");
}