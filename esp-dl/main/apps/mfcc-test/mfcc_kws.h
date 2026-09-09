#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Speech-Commands keyword classification feature extractor.
//
// Replicates the offline training pipeline (train/kws_dataset_gen.py:
// build_mfcc_transform()) exactly:
//   sample_rate = 16000, n_mfcc = 40, n_fft = 640, win_length = 640,
//   hop_length = 320, n_mels = 40, center = false, power = 2.0,
//   log_mels = true (f_min = 0, f_max = sample_rate/2 = 8000 defaults)
// so the .espdl keyword classifier sees the same features it was
// trained/calibrated on. Input amplitude follows torchaudio.load(): a
// 16-bit WAV normalised to float by dividing by 32768.
//

#define KWS_SAMPLE_RATE 16000
#define KWS_CLIP_SAMPLES 16000 /* 1 s clip */

#define KWS_N_FFT 640 /* 40 ms frame */
#define KWS_HOP 320   /* 20 ms stride */
#define KWS_N_BINS 321

#define KWS_N_MELS 40
#define KWS_N_MFCC 40
#define KWS_N_FRAMES ((KWS_CLIP_SAMPLES - KWS_N_FFT) / KWS_HOP + 1) /* 49 */

//
// Compute the (KWS_N_FRAMES, KWS_N_MFCC) MFCC features of a 1-second,
// 16 kHz mono clip.
//
// pcm:        raw int16 samples, exactly KWS_CLIP_SAMPLES of them. The
//             int16 values are normalised by /32768 before feature
//             extraction (matching how torchaudio.load() decodes a 16-bit
//             WAV), so clipping-free audio near full scale yields
//             float samples in [-1, 1].
// mfcc_out:   KWS_N_FRAMES * KWS_N_MFCC float32 values, frame-major;
//             each frame holds KWS_N_MFCC coefficients.
//
void kws_mfcc(const int16_t *pcm, float *mfcc_out);

//
// Quantize MFCC features to the int8 input of the model, exactly like the
// dataset .bin files: PPQ linear quantization with scale = 0.5,
// offset = 0, range [-128, 127] (q = clamp(round(x * 2))). The model's
// input tensor carries exponent -9, i.e. the int8 values are interpreted
// as q * 2^-9 = q / 512, which is the scale the quantized graph was
// calibrated with.
//
// mfcc: KWS_N_FRAMES * KWS_N_MFCC float32 features (from kws_mfcc).
// q:    row-major KWS_N_FRAMES * KWS_N_MFCC int8 values, flat.
//
void kws_mfcc_quantize(const float *mfcc, int8_t *q);

#ifdef __cplusplus
}
#endif
