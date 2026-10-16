#include "candle_wrapper.h"
#include <math.h>
#include <string.h>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#endif

/* ============================================================================
 * Float16 / BFloat16 bit-conversion helpers for portable paths
 * ============================================================================ */
static float candle_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(float));
    return out;
}

static uint16_t candle_f32_to_f16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = bits & 0x007FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x00800000u) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x0200u : 0u));
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float candle_bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float out;
    memcpy(&out, &bits, sizeof(float));
    return out;
}

static uint16_t candle_f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(uint32_t));
    return (uint16_t)(bits >> 16);
}

/* ============================================================================
 * CPU NEON / ARM64 inline assembly & SIMD intrinsics (candle-core/src/cpu/neon.rs)
 * ============================================================================ */
float candle_neon_reduce_one_f32(const float* x) {
#if defined(__aarch64__)
    return vaddvq_f32(vld1q_f32(x));
#elif defined(__arm__)
    float32x4_t v = vld1q_f32(x);
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
#else
    return x[0] + x[1] + x[2] + x[3];
#endif
}

float candle_neon_reduce_one_arm_f32(const float* x) {
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t v = vld1q_f32(x);
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
#else
    return x[0] + x[1] + x[2] + x[3];
#endif
}

void candle_neon_vdupq_n_f32(float v, float* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(out, vdupq_n_f32(v));
#else
    for (int i = 0; i < 4; ++i) out[i] = v;
#endif
}

void candle_neon_vld1q_f32(const float* mem_addr, float* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(out, vld1q_f32(mem_addr));
#else
    for (int i = 0; i < 4; ++i) out[i] = mem_addr[i];
#endif
}

void candle_neon_vaddq_f32(const float* a, const float* b, float* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(out, vaddq_f32(vld1q_f32(a), vld1q_f32(b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i];
#endif
}

void candle_neon_vfmaq_f32(const float* a, const float* b, const float* c, float* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(out, vfmaq_f32(vld1q_f32(a), vld1q_f32(b), vld1q_f32(c)));
#else
    for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i] * c[i];
#endif
}

void candle_neon_vst1q_f32(float* mem_addr, const float* a) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(mem_addr, vld1q_f32(a));
#else
    for (int i = 0; i < 4; ++i) mem_addr[i] = a[i];
#endif
}

void candle_neon_f16_load_fcvtl(const uint16_t* mem_addr, float* out) {
#if defined(__aarch64__)
    uint16x4_t bits = vld1_u16(mem_addr);
    float32x4_t result;
    __asm__ volatile(
        "fcvtl %0.4s, %1.4h"
        : "=w"(result)
        : "w"(bits)
    );
    vst1q_f32(out, result);
#else
    for (int i = 0; i < 4; ++i) out[i] = candle_f16_to_f32(mem_addr[i]);
#endif
}

void candle_neon_f16_store_fcvtn(uint16_t* mem_addr, const float* a) {
#if defined(__aarch64__)
    float32x4_t vec = vld1q_f32(a);
    uint16x4_t bits;
    __asm__ volatile(
        "fcvtn %0.4h, %1.4s"
        : "=w"(bits)
        : "w"(vec)
    );
    vst1_u16(mem_addr, bits);
#else
    for (int i = 0; i < 4; ++i) mem_addr[i] = candle_f32_to_f16(a[i]);
#endif
}

float candle_neon_f16_reduce_one(const uint16_t* x) {
#if defined(__aarch64__)
    uint16x8_t v = vld1q_u16(x);
    float32x4_t bot;
    float32x4_t top;
    float result;
    __asm__ volatile(
        "fcvtl %[bot].4s, %[v].4h\n\t"
        "fcvtl2 %[top].4s, %[v].8h\n\t"
        "fadd %[bot].4s, %[bot].4s, %[top].4s\n\t"
        "faddp %[bot].4s, %[bot].4s, %[bot].4s\n\t"
        "faddp %[bot].4s, %[bot].4s, %[bot].4s\n\t"
        "mov %s[res], %[bot].s[0]"
        : [bot] "=w"(bot), [top] "=&w"(top), [res] "=w"(result)
        : [v] "w"(v)
    );
    return result;
#else
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += candle_f16_to_f32(x[i]);
    return sum;
#endif
}

void candle_neon_f16_zero(uint16_t* out) {
#if defined(__aarch64__)
    uint16x8_t result;
    __asm__ volatile(
        "movi %0.8h, #0"
        : "=w"(result)
    );
    vst1q_u16(out, result);
#else
    for (int i = 0; i < 8; ++i) out[i] = 0;
#endif
}

void candle_neon_f16_vec_add(const uint16_t* a, const uint16_t* b, uint16_t* out) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint16x8_t va = vld1q_u16(a);
    uint16x8_t vb = vld1q_u16(b);
    uint16x8_t result;
    __asm__ volatile(
        "fadd %0.8h, %1.8h, %2.8h"
        : "=w"(result)
        : "w"(va), "w"(vb)
    );
    vst1q_u16(out, result);
#else
    for (int i = 0; i < 8; ++i) {
        out[i] = candle_f32_to_f16(candle_f16_to_f32(a[i]) + candle_f16_to_f32(b[i]));
    }
#endif
}

void candle_neon_f16_vec_fma(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint16x8_t va = vld1q_u16(a);
    uint16x8_t vb = vld1q_u16(b);
    uint16x8_t vc = vld1q_u16(c);
    __asm__ volatile(
        "fmla %0.8h, %1.8h, %2.8h"
        : "+w"(va)
        : "w"(vb), "w"(vc)
    );
    vst1q_u16(out, va);
#else
    for (int i = 0; i < 8; ++i) {
        out[i] = candle_f32_to_f16(
            candle_f16_to_f32(a[i]) + candle_f16_to_f32(b[i]) * candle_f16_to_f32(c[i])
        );
    }
#endif
}

void candle_neon_f16_from_f32(float v, uint16_t* out) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    uint16x8_t dst;
    float tmp;
    uint32_t w;
    __asm__ volatile(
        "fcvt %h[tmp], %s[src]\n\t"
        "fmov %w[w], %h[tmp]\n\t"
        "dup %[dst].8h, %w[w]"
        : [tmp] "=w"(tmp), [w] "=r"(w), [dst] "=w"(dst)
        : [src] "w"(v)
    );
    vst1q_u16(out, dst);
#elif defined(__aarch64__)
    uint16x8_t dst;
    float tmp;
    __asm__ volatile(
        "fcvt %h[tmp], %s[src]\n\t"
        "dup %[dst].8h, %[tmp].h[0]"
        : [tmp] "=w"(tmp), [dst] "=w"(dst)
        : [src] "w"(v)
    );
    vst1q_u16(out, dst);
#else
    uint16_t bits = candle_f32_to_f16(v);
    for (int i = 0; i < 8; ++i) out[i] = bits;
#endif
}

void candle_neon_f16_vec_store(uint16_t* mem_addr, const uint16_t* a) {
#if defined(__aarch64__)
    uint16x8_t vec = vld1q_u16(a);
    __asm__ volatile(
        "st1 { %0.8h }, [%1]"
        :
        : "w"(vec), "r"(mem_addr)
        : "memory"
    );
#else
    for (int i = 0; i < 8; ++i) mem_addr[i] = a[i];
#endif
}

void candle_neon_bf16_load_shll(const uint16_t* mem_addr, float* out) {
#if defined(__aarch64__)
    uint16x4_t i = vld1_u16(mem_addr);
    uint32x4_t s = vshll_n_u16(i, 16);
    vst1q_f32(out, vreinterpretq_f32_u32(s));
#else
    for (int i = 0; i < 4; ++i) out[i] = candle_bf16_to_f32(mem_addr[i]);
#endif
}

void candle_neon_bf16_store_shrn(uint16_t* mem_addr, const float* a) {
#if defined(__aarch64__)
    uint32x4_t s = vreinterpretq_u32_f32(vld1q_f32(a));
    uint16x4_t h = vshrn_n_u32(s, 16);
    vst1_u16(mem_addr, h);
#else
    for (int i = 0; i < 4; ++i) mem_addr[i] = candle_f32_to_bf16(a[i]);
#endif
}

void candle_neon_bf16_vec_add(const uint16_t* a, const uint16_t* b, uint16_t* out) {
#if defined(__aarch64__)
    uint16x8_t va = vld1q_u16(a);
    uint16x8_t vb = vld1q_u16(b);
    uint16x8_t result;
    __asm__ volatile(
        "fadd %0.4s, %1.4s, %2.4s"
        : "=w"(result)
        : "w"(va), "w"(vb)
    );
    vst1q_u16(out, result);
#else
    for (int lane = 0; lane < 4; ++lane) {
        uint32_t ba = ((uint32_t)a[2 * lane]) | (((uint32_t)a[2 * lane + 1]) << 16);
        uint32_t bb = ((uint32_t)b[2 * lane]) | (((uint32_t)b[2 * lane + 1]) << 16);
        float fa, fb, fsum;
        uint32_t bsum;
        memcpy(&fa, &ba, sizeof(float));
        memcpy(&fb, &bb, sizeof(float));
        fsum = fa + fb;
        memcpy(&bsum, &fsum, sizeof(uint32_t));
        out[2 * lane] = (uint16_t)(bsum & 0xFFFFu);
        out[2 * lane + 1] = (uint16_t)(bsum >> 16);
    }
#endif
}

void candle_neon_bf16_vec_fma(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    uint16x8_t acc = vld1q_u16(a);
    uint16x8_t vb = vld1q_u16(b);
    uint16x8_t vc = vld1q_u16(c);
    __asm__ volatile(
        "bfdot %0.4s, %1.8h, %2.8h"
        : "+w"(acc)
        : "w"(vb), "w"(vc)
    );
    vst1q_u16(out, acc);
#else
    for (int lane = 0; lane < 4; ++lane) {
        uint32_t ba = ((uint32_t)a[2 * lane]) | (((uint32_t)a[2 * lane + 1]) << 16);
        float acc;
        memcpy(&acc, &ba, sizeof(float));
        float b0 = candle_bf16_to_f32(b[2 * lane]);
        float b1 = candle_bf16_to_f32(b[2 * lane + 1]);
        float c0 = candle_bf16_to_f32(c[2 * lane]);
        float c1 = candle_bf16_to_f32(c[2 * lane + 1]);
        float next = acc + b0 * c0 + b1 * c1;
        uint32_t bnext;
        memcpy(&bnext, &next, sizeof(uint32_t));
        out[2 * lane] = (uint16_t)(bnext & 0xFFFFu);
        out[2 * lane + 1] = (uint16_t)(bnext >> 16);
    }
#endif
}

/* ============================================================================
 * CPU AVX2 / F16C / AVX512BF16 SIMD intrinsics (candle-core/src/cpu/avx.rs)
 * ============================================================================ */
void candle_avx_zero(float* out) {
#if defined(__AVX__)
    _mm256_storeu_ps(out, _mm256_setzero_ps());
#else
    for (int i = 0; i < 8; ++i) out[i] = 0.0f;
#endif
}

void candle_avx_set1_ps(float v, float* out) {
#if defined(__AVX__)
    _mm256_storeu_ps(out, _mm256_set1_ps(v));
#else
    for (int i = 0; i < 8; ++i) out[i] = v;
#endif
}

void candle_avx_loadu_ps(const float* mem_addr, float* out) {
#if defined(__AVX__)
    _mm256_storeu_ps(out, _mm256_loadu_ps(mem_addr));
#else
    for (int i = 0; i < 8; ++i) out[i] = mem_addr[i];
#endif
}

void candle_avx_add_ps(const float* a, const float* b, float* out) {
#if defined(__AVX__)
    _mm256_storeu_ps(out, _mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b)));
#else
    for (int i = 0; i < 8; ++i) out[i] = a[i] + b[i];
#endif
}

void candle_avx_fma_ps(const float* a, const float* b, const float* c, float* out) {
#if defined(__AVX__)
    _mm256_storeu_ps(
        out,
        _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(b), _mm256_loadu_ps(c)), _mm256_loadu_ps(a))
    );
#else
    for (int i = 0; i < 8; ++i) out[i] = b[i] * c[i] + a[i];
#endif
}

void candle_avx_storeu_ps(float* mem_addr, const float* a) {
#if defined(__AVX__)
    _mm256_storeu_ps(mem_addr, _mm256_loadu_ps(a));
#else
    for (int i = 0; i < 8; ++i) mem_addr[i] = a[i];
#endif
}

float candle_avx_reduce_one_ps(const float* x) {
#if defined(__AVX__)
    __m256 v = _mm256_loadu_ps(x);
    __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 t1 = _mm_hadd_ps(t0, t0);
    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
#else
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += x[i];
    return sum;
#endif
}

void candle_avx_cvtph_ps(const uint16_t* mem_addr, float* out) {
#if defined(__F16C__)
    _mm256_storeu_ps(out, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)mem_addr)));
#else
    for (int i = 0; i < 8; ++i) out[i] = candle_f16_to_f32(mem_addr[i]);
#endif
}

void candle_avx_cvtps_ph(const float* a, uint16_t* mem_addr) {
#if defined(__F16C__)
    _mm_storeu_si128((__m128i*)mem_addr, _mm256_cvtps_ph(_mm256_loadu_ps(a), 0));
#else
    for (int i = 0; i < 8; ++i) mem_addr[i] = candle_f32_to_f16(a[i]);
#endif
}

void candle_avx_cvtpbh_ps(const uint16_t* mem_addr, float* out) {
    for (int i = 0; i < 8; ++i) out[i] = candle_bf16_to_f32(mem_addr[i]);
}

void candle_avx_cvtneps_pbh(const float* a, uint16_t* mem_addr) {
    for (int i = 0; i < 8; ++i) mem_addr[i] = candle_f32_to_bf16(a[i]);
}

/* ============================================================================
 * CPU WASM SIMD128 intrinsics (candle-core/src/cpu/simd128.rs)
 * ============================================================================ */
void candle_simd128_splat_f32(float v, float* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(out, wasm_f32x4_splat(v));
#else
    for (int i = 0; i < 4; ++i) out[i] = v;
#endif
}

void candle_simd128_load_f32(const float* mem_addr, float* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(out, wasm_v128_load(mem_addr));
#else
    for (int i = 0; i < 4; ++i) out[i] = mem_addr[i];
#endif
}

void candle_simd128_add_f32(const float* a, const float* b, float* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(out, wasm_f32x4_add(wasm_v128_load(a), wasm_v128_load(b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = a[i] + b[i];
#endif
}

void candle_simd128_mul_f32(const float* a, const float* b, float* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(out, wasm_f32x4_mul(wasm_v128_load(a), wasm_v128_load(b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = a[i] * b[i];
#endif
}

void candle_simd128_fma_f32(const float* a, const float* b, const float* c, float* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(
        out,
        wasm_f32x4_add(wasm_f32x4_mul(wasm_v128_load(b), wasm_v128_load(c)), wasm_v128_load(a))
    );
#else
    for (int i = 0; i < 4; ++i) out[i] = b[i] * c[i] + a[i];
#endif
}

void candle_simd128_store_f32(float* mem_addr, const float* a) {
#if defined(__wasm_simd128__)
    wasm_v128_store(mem_addr, wasm_v128_load(a));
#else
    for (int i = 0; i < 4; ++i) mem_addr[i] = a[i];
#endif
}

float candle_simd128_reduce_one_f32(const float* x) {
#if defined(__wasm_simd128__)
    v128_t v = wasm_v128_load(x);
    return wasm_f32x4_extract_lane(v, 0) +
           wasm_f32x4_extract_lane(v, 1) +
           wasm_f32x4_extract_lane(v, 2) +
           wasm_f32x4_extract_lane(v, 3);
#else
    return x[0] + x[1] + x[2] + x[3];
#endif
}

/* ============================================================================
 * Apple Accelerate / Intel MKL FFI (candle-core/src/accelerate.rs, mkl.rs)
 * ============================================================================ */
void candle_accelerate_sgemm(
    int8_t transa,
    int8_t transb,
    int32_t m,
    int32_t n,
    int32_t k,
    float alpha,
    const float* a,
    int32_t lda,
    const float* b,
    int32_t ldb,
    float beta,
    float* c,
    int32_t ldc
) {
    int ta = (transa == 'T' || transa == 't');
    int tb = (transb == 'T' || transb == 't');
    for (int32_t j = 0; j < n; ++j) {
        for (int32_t i = 0; i < m; ++i) {
            float sum = 0.0f;
            for (int32_t p = 0; p < k; ++p) {
                float a_val = ta ? a[i * lda + p] : a[p * lda + i];
                float b_val = tb ? b[p * ldb + j] : b[j * ldb + p];
                sum += a_val * b_val;
            }
            int32_t idx = j * ldc + i;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
}

void candle_accelerate_dgemm(
    int8_t transa,
    int8_t transb,
    int32_t m,
    int32_t n,
    int32_t k,
    double alpha,
    const double* a,
    int32_t lda,
    const double* b,
    int32_t ldb,
    double beta,
    double* c,
    int32_t ldc
) {
    int ta = (transa == 'T' || transa == 't');
    int tb = (transb == 'T' || transb == 't');
    for (int32_t j = 0; j < n; ++j) {
        for (int32_t i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int32_t p = 0; p < k; ++p) {
                double a_val = ta ? a[i * lda + p] : a[p * lda + i];
                double b_val = tb ? b[p * ldb + j] : b[j * ldb + p];
                sum += a_val * b_val;
            }
            int32_t idx = j * ldc + i;
            c[idx] = alpha * sum + beta * c[idx];
        }
    }
}

void candle_accelerate_vvexpf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvexpf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = expf(src[i]);
#endif
}

void candle_accelerate_vvexp(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvexp(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = exp(src[i]);
#endif
}

void candle_accelerate_vvsqrtf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvsqrtf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = sqrtf(src[i]);
#endif
}

void candle_accelerate_vvsqrt(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvsqrt(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = sqrt(src[i]);
#endif
}

void candle_accelerate_vvsinf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvsinf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = sinf(src[i]);
#endif
}

void candle_accelerate_vvsin(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvsin(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = sin(src[i]);
#endif
}

void candle_accelerate_vvcosf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvcosf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = cosf(src[i]);
#endif
}

void candle_accelerate_vvcos(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvcos(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = cos(src[i]);
#endif
}

void candle_accelerate_vvlogf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvlogf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = logf(src[i]);
#endif
}

void candle_accelerate_vvlog(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvlog(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = log(src[i]);
#endif
}

void candle_accelerate_vvtanhf(float* dst, const float* src, int32_t len) {
#ifdef __APPLE__
    vvtanhf(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = tanhf(src[i]);
#endif
}

void candle_accelerate_vvtanh(double* dst, const double* src, int32_t len) {
#ifdef __APPLE__
    vvtanh(dst, src, &len);
#else
    for (int32_t i = 0; i < len; ++i) dst[i] = tanh(src[i]);
#endif
}

void candle_accelerate_vdsp_vadd(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vadd(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] + a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vaddd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vaddD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] + a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vsub(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vsub(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] - a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vsubd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vsubD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] - a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vmul(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vmul(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] * a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vmuld(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vmulD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] * a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vdiv(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vdiv(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] / a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vdivd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vdivD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) c[i * (uint64_t)ic] = b[i * (uint64_t)ib] / a[i * (uint64_t)ia];
#endif
}

void candle_accelerate_vdsp_vmin(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vmin(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) {
        float va = a[i * (uint64_t)ia];
        float vb = b[i * (uint64_t)ib];
        c[i * (uint64_t)ic] = vb < va ? vb : va;
    }
#endif
}

void candle_accelerate_vdsp_vmind(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vminD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) {
        double va = a[i * (uint64_t)ia];
        double vb = b[i * (uint64_t)ib];
        c[i * (uint64_t)ic] = vb < va ? vb : va;
    }
#endif
}

void candle_accelerate_vdsp_vmax(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vmax(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) {
        float va = a[i * (uint64_t)ia];
        float vb = b[i * (uint64_t)ib];
        c[i * (uint64_t)ic] = vb > va ? vb : va;
    }
#endif
}

void candle_accelerate_vdsp_vmaxd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n) {
#ifdef __APPLE__
    vDSP_vmaxD(a, (vDSP_Stride)ia, b, (vDSP_Stride)ib, c, (vDSP_Stride)ic, (vDSP_Length)n);
#else
    for (uint64_t i = 0; i < n; ++i) {
        double va = a[i * (uint64_t)ia];
        double vb = b[i * (uint64_t)ib];
        c[i * (uint64_t)ic] = vb > va ? vb : va;
    }
#endif
}

/* ============================================================================
 * Quantized NEON / AVX2 / SIMD128 intrinsics (candle-core/src/quantized/)
 * ============================================================================ */
void candle_qneon_vdotq_s32(const int8_t* a, const int8_t* b, int32_t* out) {
#if defined(__aarch64__) || defined(__arm__)
    int8x16_t va = vld1q_s8(a);
    int8x16_t vb = vld1q_s8(b);
    int16x8_t p0 = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    int16x8_t p1 = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    vst1q_s32(out, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#else
    for (int i = 0; i < 4; ++i) {
        int32_t s0 = (int32_t)a[i * 2] * (int32_t)b[i * 2] + (int32_t)a[i * 2 + 1] * (int32_t)b[i * 2 + 1];
        int32_t s1 = (int32_t)a[8 + i * 2] * (int32_t)b[8 + i * 2] + (int32_t)a[8 + i * 2 + 1] * (int32_t)b[8 + i * 2 + 1];
        out[i] = s0 + s1;
    }
#endif
}

void candle_qneon_vmull_s8(const int8_t* a, const int8_t* b, int16_t* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_s16(out, vmull_s8(vld1_s8(a), vld1_s8(b)));
#else
    for (int i = 0; i < 8; ++i) out[i] = (int16_t)((int32_t)a[i] * (int32_t)b[i]);
#endif
}

void candle_qneon_vmull_s16(const int16_t* a, const int16_t* b, int32_t* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_s32(out, vmull_s16(vld1_s16(a), vld1_s16(b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = (int32_t)a[i] * (int32_t)b[i];
#endif
}

void candle_qneon_vpaddlq_s16(const int16_t* a, int32_t* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_s32(out, vpaddlq_s16(vld1q_s16(a)));
#else
    for (int i = 0; i < 4; ++i) out[i] = (int32_t)a[i * 2] + (int32_t)a[i * 2 + 1];
#endif
}

void candle_qneon_vpaddq_s16(const int16_t* a, const int16_t* b, int16_t* out) {
#if defined(__aarch64__)
    vst1q_s16(out, vpaddq_s16(vld1q_s16(a), vld1q_s16(b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = (int16_t)((int32_t)a[i * 2] + (int32_t)a[i * 2 + 1]);
    for (int i = 0; i < 4; ++i) out[4 + i] = (int16_t)((int32_t)b[i * 2] + (int32_t)b[i * 2 + 1]);
#endif
}

void candle_qneon_vmlaq_n_f32(const float* acc, const float* v, float s, float* out) {
#if defined(__aarch64__) || defined(__arm__)
    vst1q_f32(out, vmlaq_n_f32(vld1q_f32(acc), vld1q_f32(v), s));
#else
    for (int i = 0; i < 4; ++i) out[i] = acc[i] + v[i] * s;
#endif
}

int32_t candle_qneon_vaddvq_s32(const int32_t* a) {
#if defined(__aarch64__)
    return vaddvq_s32(vld1q_s32(a));
#else
    return a[0] + a[1] + a[2] + a[3];
#endif
}

void candle_qavx_madd_epi16(const int16_t* a, const int16_t* b, int32_t* out) {
#if defined(__AVX2__)
    _mm256_storeu_si256((__m256i*)out, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)a), _mm256_loadu_si256((const __m256i*)b)));
#else
    for (int i = 0; i < 8; ++i) {
        out[i] = (int32_t)a[i * 2] * (int32_t)b[i * 2] + (int32_t)a[i * 2 + 1] * (int32_t)b[i * 2 + 1];
    }
#endif
}

void candle_qavx_mm_madd_epi16(const int16_t* a, const int16_t* b, int32_t* out) {
#if defined(__SSE2__)
    _mm_storeu_si128((__m128i*)out, _mm_madd_epi16(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b)));
#else
    for (int i = 0; i < 4; ++i) {
        out[i] = (int32_t)a[i * 2] * (int32_t)b[i * 2] + (int32_t)a[i * 2 + 1] * (int32_t)b[i * 2 + 1];
    }
#endif
}

void candle_qavx_maddubs_epi16(const int8_t* ax, const int8_t* sy, int16_t* out) {
#if defined(__AVX2__)
    _mm256_storeu_si256((__m256i*)out, _mm256_maddubs_epi16(_mm256_loadu_si256((const __m256i*)ax), _mm256_loadu_si256((const __m256i*)sy)));
#else
    for (int i = 0; i < 16; ++i) {
        int32_t u0 = (int32_t)((uint8_t)ax[i * 2]);
        int32_t u1 = (int32_t)((uint8_t)ax[i * 2 + 1]);
        int32_t s0 = (int32_t)sy[i * 2];
        int32_t s1 = (int32_t)sy[i * 2 + 1];
        int32_t v = u0 * s0 + u1 * s1;
        if (v < -32768) v = -32768;
        if (v > 32767) v = 32767;
        out[i] = (int16_t)v;
    }
#endif
}

void candle_qavx_sign_epi8(const int8_t* a, const int8_t* b, int8_t* out) {
#if defined(__AVX2__)
    _mm256_storeu_si256((__m256i*)out, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)a), _mm256_loadu_si256((const __m256i*)b)));
#else
    for (int i = 0; i < 32; ++i) {
        int8_t bv = b[i];
        out[i] = bv < 0 ? (int8_t)(-a[i]) : (bv == 0 ? 0 : a[i]);
    }
#endif
}

void candle_qavx_hadd_epi16(const int16_t* a, const int16_t* b, int16_t* out) {
#if defined(__SSSE3__)
    _mm_storeu_si128((__m128i*)out, _mm_hadd_epi16(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b)));
#else
    for (int i = 0; i < 4; ++i) out[i] = (int16_t)(a[i * 2] + a[i * 2 + 1]);
    for (int i = 0; i < 4; ++i) out[4 + i] = (int16_t)(b[i * 2] + b[i * 2 + 1]);
#endif
}

void candle_qavx_hadd_epi32(const int32_t* a, const int32_t* b, int32_t* out) {
#if defined(__SSSE3__)
    _mm_storeu_si128((__m128i*)out, _mm_hadd_epi32(_mm_loadu_si128((const __m128i*)a), _mm_loadu_si128((const __m128i*)b)));
#else
    out[0] = a[0] + a[1];
    out[1] = a[2] + a[3];
    out[2] = b[0] + b[1];
    out[3] = b[2] + b[3];
#endif
}

void candle_qsimd128_dot_i16x8(const int16_t* a, const int16_t* b, int32_t* out) {
#if defined(__wasm_simd128__)
    wasm_v128_store(out, wasm_i32x4_dot_i16x8(wasm_v128_load(a), wasm_v128_load(b)));
#else
    for (int i = 0; i < 4; ++i) {
        out[i] = (int32_t)a[i * 2] * (int32_t)b[i * 2] + (int32_t)a[i * 2 + 1] * (int32_t)b[i * 2 + 1];
    }
#endif
}

