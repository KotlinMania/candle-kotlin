#ifndef CANDLE_WRAPPER_H
#define CANDLE_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CPU NEON / ARM64 inline assembly & SIMD intrinsics (candle-core/src/cpu/neon.rs)
 * ============================================================================ */
float candle_neon_reduce_one_f32(const float* x);
float candle_neon_reduce_one_arm_f32(const float* x);
void candle_neon_vdupq_n_f32(float v, float* out);
void candle_neon_vld1q_f32(const float* mem_addr, float* out);
void candle_neon_vaddq_f32(const float* a, const float* b, float* out);
void candle_neon_vfmaq_f32(const float* a, const float* b, const float* c, float* out);
void candle_neon_vst1q_f32(float* mem_addr, const float* a);

/* fp16 fallback & fp16 hardware assembly (FCVTL / FCVTN / FMLA.8h / FADDP) */
void candle_neon_f16_load_fcvtl(const uint16_t* mem_addr, float* out);
void candle_neon_f16_store_fcvtn(uint16_t* mem_addr, const float* a);
float candle_neon_f16_reduce_one(const uint16_t* x);
void candle_neon_f16_zero(uint16_t* out);
void candle_neon_f16_vec_add(const uint16_t* a, const uint16_t* b, uint16_t* out);
void candle_neon_f16_vec_fma(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out);
void candle_neon_f16_from_f32(float v, uint16_t* out);
void candle_neon_f16_vec_store(uint16_t* mem_addr, const uint16_t* a);

/* bf16 fallback & bf16 BFDOT hardware assembly */
void candle_neon_bf16_load_shll(const uint16_t* mem_addr, float* out);
void candle_neon_bf16_store_shrn(uint16_t* mem_addr, const float* a);
void candle_neon_bf16_vec_add(const uint16_t* a, const uint16_t* b, uint16_t* out);
void candle_neon_bf16_vec_fma(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out);

/* ============================================================================
 * CPU AVX2 / F16C / AVX512BF16 SIMD intrinsics (candle-core/src/cpu/avx.rs)
 * ============================================================================ */
void candle_avx_zero(float* out);
void candle_avx_set1_ps(float v, float* out);
void candle_avx_loadu_ps(const float* mem_addr, float* out);
void candle_avx_add_ps(const float* a, const float* b, float* out);
void candle_avx_fma_ps(const float* a, const float* b, const float* c, float* out);
void candle_avx_storeu_ps(float* mem_addr, const float* a);
float candle_avx_reduce_one_ps(const float* x);
void candle_avx_cvtph_ps(const uint16_t* mem_addr, float* out);
void candle_avx_cvtps_ph(const float* a, uint16_t* mem_addr);
void candle_avx_cvtpbh_ps(const uint16_t* mem_addr, float* out);
void candle_avx_cvtneps_pbh(const float* a, uint16_t* mem_addr);

/* ============================================================================
 * CPU WASM SIMD128 intrinsics (candle-core/src/cpu/simd128.rs)
 * ============================================================================ */
void candle_simd128_splat_f32(float v, float* out);
void candle_simd128_load_f32(const float* mem_addr, float* out);
void candle_simd128_add_f32(const float* a, const float* b, float* out);
void candle_simd128_mul_f32(const float* a, const float* b, float* out);
void candle_simd128_fma_f32(const float* a, const float* b, const float* c, float* out);
void candle_simd128_store_f32(float* mem_addr, const float* a);
float candle_simd128_reduce_one_f32(const float* x);

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
);

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
);

void candle_accelerate_vvexpf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvexp(double* dst, const double* src, int32_t len);
void candle_accelerate_vvsqrtf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvsqrt(double* dst, const double* src, int32_t len);
void candle_accelerate_vvsinf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvsin(double* dst, const double* src, int32_t len);
void candle_accelerate_vvcosf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvcos(double* dst, const double* src, int32_t len);
void candle_accelerate_vvlogf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvlog(double* dst, const double* src, int32_t len);
void candle_accelerate_vvtanhf(float* dst, const float* src, int32_t len);
void candle_accelerate_vvtanh(double* dst, const double* src, int32_t len);

void candle_accelerate_vdsp_vadd(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vaddd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vsub(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vsubd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmul(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmuld(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vdiv(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vdivd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmin(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmind(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmax(const float* a, int64_t ia, const float* b, int64_t ib, float* c, int64_t ic, uint64_t n);
void candle_accelerate_vdsp_vmaxd(const double* a, int64_t ia, const double* b, int64_t ib, double* c, int64_t ic, uint64_t n);

/* ============================================================================
 * Quantized NEON / AVX2 / SIMD128 intrinsics (candle-core/src/quantized/)
 * ============================================================================ */
void candle_qneon_vdotq_s32(const int8_t* a, const int8_t* b, int32_t* out);
void candle_qneon_vmull_s8(const int8_t* a, const int8_t* b, int16_t* out);
void candle_qneon_vmull_s16(const int16_t* a, const int16_t* b, int32_t* out);
void candle_qneon_vpaddlq_s16(const int16_t* a, int32_t* out);
void candle_qneon_vpaddq_s16(const int16_t* a, const int16_t* b, int16_t* out);
void candle_qneon_vmlaq_n_f32(const float* acc, const float* v, float s, float* out);
int32_t candle_qneon_vaddvq_s32(const int32_t* a);

void candle_qavx_madd_epi16(const int16_t* a, const int16_t* b, int32_t* out);
void candle_qavx_mm_madd_epi16(const int16_t* a, const int16_t* b, int32_t* out);
void candle_qavx_maddubs_epi16(const int8_t* ax, const int8_t* sy, int16_t* out);
void candle_qavx_sign_epi8(const int8_t* a, const int8_t* b, int8_t* out);
void candle_qavx_hadd_epi16(const int16_t* a, const int16_t* b, int16_t* out);
void candle_qavx_hadd_epi32(const int32_t* a, const int32_t* b, int32_t* out);

void candle_qsimd128_dot_i16x8(const int16_t* a, const int16_t* b, int32_t* out);

#ifdef __cplusplus
}
#endif

#endif /* CANDLE_WRAPPER_H */
