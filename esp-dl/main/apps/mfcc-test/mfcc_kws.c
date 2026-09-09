/*
 * Speech-Commands keyword MFCC front-end for the ESP32-S3-EYE demo.
 *
 * A faithful C port of the feature extractor that produced the offline
 * Speech-Commands dataset (train/kws_dataset_gen.py build_mfcc_transform()):
 * torchaudio.transforms.MFCC with sample_rate=16000, n_mfcc=40,
 * n_fft=640, win_length=640, hop_length=320, n_mels=40, center=false,
 * power=2.0, log_mels=true and the default f_min=0 / f_max=8000.
 *
 * The window, mel filterbank and DCT matrices are static tables generated
 * from torchaudio itself (same float32 values), so the only remaining
 * differences vs. the offline pipeline are float32 FFT / accumulation
 * rounding, which stay far below the int8 input quantization step.
 */

#include "mfcc_kws.h"

#include "kws_tables.h"

#include <math.h>

#define KWS_TWOPI 6.283185307179586f

// Workspace (static so the app task stack stays small)
static float s_ws_in[KWS_N_FFT];  /* real FFT input, one windowed frame */
static float s_ws_x[KWS_N_FFT * 2]; /* 640 complex bins [re, im] pairs */
static float s_ws_a[128][5][2];     /* 5-point DFT of each 128-subsequence */
static float s_ws_fft[128 * 2];     /* 128-point radix-2 work buffer */
static float s_five[5][5][2];       /* (e^{-2*pi*i*a*b/5}) factors */
static int s_init = 0;

static void kws_init_tables(void) {
    if (s_init) return;
    for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) {
            float ang = -KWS_TWOPI * (float)(a * b) / 5.0f;
            s_five[a][b][0] = cosf(ang);
            s_five[a][b][1] = sinf(ang);
        }
    }
    s_init = 1;
}

/* In-place 128-point radix-2 complex FFT, input/output [re, im] pairs. */
static void kws_fft128(float *x) {
    int n = 128;
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = x[2 * i], ti = x[2 * i + 1];
            x[2 * i] = x[2 * j];
            x[2 * i + 1] = x[2 * j + 1];
            x[2 * j] = tr;
            x[2 * j + 1] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float wang = -KWS_TWOPI / len;
        float wpr = cosf(wang), wpi = sinf(wang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int k = 0; k < half; ++k) {
                int a = i + k, b = a + half;
                float xr = x[2 * b] * wr - x[2 * b + 1] * wi;
                float xi = x[2 * b] * wi + x[2 * b + 1] * wr;
                x[2 * b] = x[2 * a] - xr;
                x[2 * b + 1] = x[2 * a + 1] - xi;
                x[2 * a] += xr;
                x[2 * a + 1] += xi;
                float nwr = wr * wpr - wi * wpi;
                wi = wr * wpi + wi * wpr;
                wr = nwr;
            }
        }
    }
}

/*
 * 640-point real-input FFT implemented as a mixed-radix 640 = 5 x 128
 * Cooley-Tukey decomposition:
 *   X[k1 + 5*k2] = Sum_n2 e^{-2*pi*i*n2*k1/640} A[n2][k1] e^{-2*pi*i*n2*k2/128}
 * with A[n2] the 5-point DFT of the decimated subsequence x[128*n1 + n2].
 * Result: all 640 complex bins in s_ws_x ([re, im] pairs); only bins
 * 0..320 (the one-sided spectrum) are used.
 */
static void kws_fft640(const float *x) {
    for (int n2 = 0; n2 < 128; ++n2) {
        float s0 = x[n2], s1 = x[128 + n2], s2 = x[256 + n2];
        float s3 = x[384 + n2], s4 = x[512 + n2];
        for (int k1 = 0; k1 < 5; ++k1) {
            float re = s0 * s_five[0][k1][0] + s1 * s_five[1][k1][0] +
                       s2 * s_five[2][k1][0] + s3 * s_five[3][k1][0] +
                       s4 * s_five[4][k1][0];
            float im = s0 * s_five[0][k1][1] + s1 * s_five[1][k1][1] +
                       s2 * s_five[2][k1][1] + s3 * s_five[3][k1][1] +
                       s4 * s_five[4][k1][1];
            s_ws_a[n2][k1][0] = re;
            s_ws_a[n2][k1][1] = im;
        }
    }
    for (int k1 = 0; k1 < 5; ++k1) {
        for (int n2 = 0; n2 < 128; ++n2) {
            float ang = -KWS_TWOPI * (float)(n2 * k1) / (float)KWS_N_FFT;
            float cr = cosf(ang), si = sinf(ang);
            float re = s_ws_a[n2][k1][0], im = s_ws_a[n2][k1][1];
            s_ws_fft[2 * n2] = re * cr - im * si;
            s_ws_fft[2 * n2 + 1] = re * si + im * cr;
        }
        kws_fft128(s_ws_fft);
        for (int k2 = 0; k2 < 128; ++k2) {
            int k = k1 + 5 * k2;
            s_ws_x[2 * k] = s_ws_fft[2 * k2];
            s_ws_x[2 * k + 1] = s_ws_fft[2 * k2 + 1];
        }
    }
}

void kws_mfcc(const int16_t *pcm, float *mfcc_out) {
    kws_init_tables();

    float power[KWS_N_BINS];
    float logmel[KWS_N_MELS];

    for (int t = 0; t < KWS_N_FRAMES; ++t) {
        const int16_t *chunk = pcm + t * KWS_HOP;

        /* Windowed frame, normalised to float exactly like a 16-bit WAV. */
        for (int i = 0; i < KWS_N_FFT; ++i) {
            s_ws_in[i] = (float)chunk[i] / 32768.0f * kws_window[i];
        }

        kws_fft640(s_ws_in);

        /* One-sided power spectrum, |X[k]|^2. */
        for (int k = 0; k < KWS_N_BINS; ++k) {
            float re = s_ws_x[2 * k], im = s_ws_x[2 * k + 1];
            power[k] = re * re + im * im;
        }

        /* Mel filterbank integration + torchaudio's log(x + 1e-6). */
        for (int m = 0; m < KWS_N_MELS; ++m) {
            float acc = 0.0f;
            const float *col = &kws_fbank[m];
            for (int k = 1; k < KWS_N_BINS; ++k) { /* bin 0 is unused by the bank */
                acc += power[k] * col[k * KWS_N_MELS];
            }
            logmel[m] = logf(acc + 1e-6f);
        }

        /* Orthonormal DCT-II (matching torchaudio's create_dct). */
        for (int k = 0; k < KWS_N_MFCC; ++k) {
            float acc = 0.0f;
            for (int m = 0; m < KWS_N_MELS; ++m) {
                acc += logmel[m] * kws_dct[m * KWS_N_MFCC + k];
            }
            mfcc_out[t * KWS_N_MFCC + k] = acc;
        }
    }
}

void kws_mfcc_quantize(const float *mfcc, int8_t *q) {
    for (int i = 0; i < KWS_N_FRAMES * KWS_N_MFCC; ++i) {
        float v = roundf(mfcc[i] * 2.0f);
        if (v < -128.0f) v = -128.0f;
        else if (v > 127.0f) v = 127.0f;
        q[i] = (int8_t)v;
    }
}