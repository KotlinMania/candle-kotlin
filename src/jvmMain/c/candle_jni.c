#include <jni.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../../nativeInterop/cinterop/candle_wrapper.c"

/* ============================================================================
 * JNI Bindings for CPU NEON / ARM64 inline assembly & SIMD
 * ============================================================================ */

JNIEXPORT jfloat JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonReduceOneF32(
    JNIEnv* env, jclass clazz, jfloatArray x) {
    (void)clazz;
    jfloat buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    (*env)->GetFloatArrayRegion(env, x, 0, 4, buf);
    return candle_neon_reduce_one_f32(buf);
}

JNIEXPORT jfloat JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonReduceOneArmF32(
    JNIEnv* env, jclass clazz, jfloatArray x, jint offset) {
    (void)clazz;
    jfloat buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    (*env)->GetFloatArrayRegion(env, x, offset, 4, buf);
    return candle_neon_reduce_one_arm_f32(buf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonVdupqNF32(
    JNIEnv* env, jclass clazz, jfloat v, jfloatArray out) {
    (void)clazz;
    jfloat buf[4];
    candle_neon_vdupq_n_f32(v, buf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, buf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonVld1qF32(
    JNIEnv* env, jclass clazz, jfloatArray memAddr, jint offset, jfloatArray out) {
    (void)clazz;
    jfloat inBuf[4];
    jfloat outBuf[4];
    (*env)->GetFloatArrayRegion(env, memAddr, offset, 4, inBuf);
    candle_neon_vld1q_f32(inBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonVaddqF32(
    JNIEnv* env, jclass clazz, jfloatArray a, jfloatArray b, jfloatArray out) {
    (void)clazz;
    jfloat aBuf[4], bBuf[4], outBuf[4];
    (*env)->GetFloatArrayRegion(env, a, 0, 4, aBuf);
    (*env)->GetFloatArrayRegion(env, b, 0, 4, bBuf);
    candle_neon_vaddq_f32(aBuf, bBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonVfmaqF32(
    JNIEnv* env, jclass clazz, jfloatArray a, jfloatArray b, jfloatArray c, jfloatArray out) {
    (void)clazz;
    jfloat aBuf[4], bBuf[4], cBuf[4], outBuf[4];
    (*env)->GetFloatArrayRegion(env, a, 0, 4, aBuf);
    (*env)->GetFloatArrayRegion(env, b, 0, 4, bBuf);
    (*env)->GetFloatArrayRegion(env, c, 0, 4, cBuf);
    candle_neon_vfmaq_f32(aBuf, bBuf, cBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonVst1qF32(
    JNIEnv* env, jclass clazz, jfloatArray memAddr, jint offset, jfloatArray a) {
    (void)clazz;
    jfloat aBuf[4], outBuf[4];
    (*env)->GetFloatArrayRegion(env, a, 0, 4, aBuf);
    candle_neon_vst1q_f32(outBuf, aBuf);
    (*env)->SetFloatArrayRegion(env, memAddr, offset, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16LoadFcvtl(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jfloatArray out) {
    (void)clazz;
    jshort inBuf[4];
    jfloat outBuf[4];
    (*env)->GetShortArrayRegion(env, memAddr, offset, 4, inBuf);
    candle_neon_f16_load_fcvtl((const uint16_t*)inBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16StoreFcvtn(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jfloatArray a) {
    (void)clazz;
    jfloat aBuf[4];
    jshort outBuf[4];
    (*env)->GetFloatArrayRegion(env, a, 0, 4, aBuf);
    candle_neon_f16_store_fcvtn((uint16_t*)outBuf, aBuf);
    (*env)->SetShortArrayRegion(env, memAddr, offset, 4, outBuf);
}

JNIEXPORT jfloat JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16ReduceOne(
    JNIEnv* env, jclass clazz, jshortArray x) {
    (void)clazz;
    jshort buf[8];
    (*env)->GetShortArrayRegion(env, x, 0, 8, buf);
    return candle_neon_f16_reduce_one((const uint16_t*)buf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16Zero(
    JNIEnv* env, jclass clazz, jshortArray out) {
    (void)clazz;
    jshort buf[8];
    candle_neon_f16_zero((uint16_t*)buf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, buf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16VecAdd(
    JNIEnv* env, jclass clazz, jshortArray a, jshortArray b, jshortArray out) {
    (void)clazz;
    jshort aBuf[8], bBuf[8], outBuf[8];
    (*env)->GetShortArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetShortArrayRegion(env, b, 0, 8, bBuf);
    candle_neon_f16_vec_add((const uint16_t*)aBuf, (const uint16_t*)bBuf, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16VecFma(
    JNIEnv* env, jclass clazz, jshortArray a, jshortArray b, jshortArray c, jshortArray out) {
    (void)clazz;
    jshort aBuf[8], bBuf[8], cBuf[8], outBuf[8];
    (*env)->GetShortArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetShortArrayRegion(env, b, 0, 8, bBuf);
    (*env)->GetShortArrayRegion(env, c, 0, 8, cBuf);
    candle_neon_f16_vec_fma((const uint16_t*)aBuf, (const uint16_t*)bBuf, (const uint16_t*)cBuf, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16FromF32(
    JNIEnv* env, jclass clazz, jfloat v, jshortArray out) {
    (void)clazz;
    jshort outBuf[8];
    candle_neon_f16_from_f32(v, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonF16VecStore(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jshortArray a) {
    (void)clazz;
    jshort aBuf[8], outBuf[8];
    (*env)->GetShortArrayRegion(env, a, 0, 8, aBuf);
    candle_neon_f16_vec_store((uint16_t*)outBuf, (const uint16_t*)aBuf);
    (*env)->SetShortArrayRegion(env, memAddr, offset, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonBf16LoadShll(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jfloatArray out) {
    (void)clazz;
    jshort inBuf[4];
    jfloat outBuf[4];
    (*env)->GetShortArrayRegion(env, memAddr, offset, 4, inBuf);
    candle_neon_bf16_load_shll((const uint16_t*)inBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonBf16StoreShrn(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jfloatArray a) {
    (void)clazz;
    jfloat aBuf[4];
    jshort outBuf[4];
    (*env)->GetFloatArrayRegion(env, a, 0, 4, aBuf);
    candle_neon_bf16_store_shrn((uint16_t*)outBuf, aBuf);
    (*env)->SetShortArrayRegion(env, memAddr, offset, 4, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonBf16VecAdd(
    JNIEnv* env, jclass clazz, jshortArray a, jshortArray b, jshortArray out) {
    (void)clazz;
    jshort aBuf[8], bBuf[8], outBuf[8];
    (*env)->GetShortArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetShortArrayRegion(env, b, 0, 8, bBuf);
    candle_neon_bf16_vec_add((const uint16_t*)aBuf, (const uint16_t*)bBuf, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_neonBf16VecFma(
    JNIEnv* env, jclass clazz, jshortArray a, jshortArray b, jshortArray c, jshortArray out) {
    (void)clazz;
    jshort aBuf[8], bBuf[8], cBuf[8], outBuf[8];
    (*env)->GetShortArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetShortArrayRegion(env, b, 0, 8, bBuf);
    (*env)->GetShortArrayRegion(env, c, 0, 8, cBuf);
    candle_neon_bf16_vec_fma((const uint16_t*)aBuf, (const uint16_t*)bBuf, (const uint16_t*)cBuf, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, out, 0, 8, outBuf);
}

/* ============================================================================
 * JNI Bindings for CPU AVX2 / F16C / AVX512BF16
 * ============================================================================ */

JNIEXPORT jfloat JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_avxReduceOnePs(
    JNIEnv* env, jclass clazz, jfloatArray x) {
    (void)clazz;
    jfloat buf[8];
    (*env)->GetFloatArrayRegion(env, x, 0, 8, buf);
    return candle_avx_reduce_one_ps(buf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_avxAddPs(
    JNIEnv* env, jclass clazz, jfloatArray a, jfloatArray b, jfloatArray out) {
    (void)clazz;
    jfloat aBuf[8], bBuf[8], outBuf[8];
    (*env)->GetFloatArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetFloatArrayRegion(env, b, 0, 8, bBuf);
    candle_avx_add_ps(aBuf, bBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_avxFmaPs(
    JNIEnv* env, jclass clazz, jfloatArray a, jfloatArray b, jfloatArray c, jfloatArray out) {
    (void)clazz;
    jfloat aBuf[8], bBuf[8], cBuf[8], outBuf[8];
    (*env)->GetFloatArrayRegion(env, a, 0, 8, aBuf);
    (*env)->GetFloatArrayRegion(env, b, 0, 8, bBuf);
    (*env)->GetFloatArrayRegion(env, c, 0, 8, cBuf);
    candle_avx_fma_ps(aBuf, bBuf, cBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_avxCvtphPs(
    JNIEnv* env, jclass clazz, jshortArray memAddr, jint offset, jfloatArray out) {
    (void)clazz;
    jshort inBuf[8];
    jfloat outBuf[8];
    (*env)->GetShortArrayRegion(env, memAddr, offset, 8, inBuf);
    candle_avx_cvtph_ps((const uint16_t*)inBuf, outBuf);
    (*env)->SetFloatArrayRegion(env, out, 0, 8, outBuf);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_avxCvtpsPh(
    JNIEnv* env, jclass clazz, jfloatArray a, jshortArray memAddr, jint offset) {
    (void)clazz;
    jfloat aBuf[8];
    jshort outBuf[8];
    (*env)->GetFloatArrayRegion(env, a, 0, 8, aBuf);
    candle_avx_cvtps_ph(aBuf, (uint16_t*)outBuf);
    (*env)->SetShortArrayRegion(env, memAddr, offset, 8, outBuf);
}

/* ============================================================================
 * JNI Bindings for Accelerate / MKL
 * ============================================================================ */

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_accelerateVvexpf(
    JNIEnv* env, jclass clazz, jfloatArray dst, jfloatArray src, jint len) {
    (void)clazz;
    if (len <= 0) return;
    jfloat* srcPtr = (*env)->GetFloatArrayElements(env, src, NULL);
    jfloat* dstPtr = (*env)->GetFloatArrayElements(env, dst, NULL);
    if (srcPtr && dstPtr) {
        candle_accelerate_vvexpf(dstPtr, srcPtr, len);
    }
    if (srcPtr) (*env)->ReleaseFloatArrayElements(env, src, srcPtr, JNI_ABORT);
    if (dstPtr) (*env)->ReleaseFloatArrayElements(env, dst, dstPtr, 0);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_accelerateVvsqrtf(
    JNIEnv* env, jclass clazz, jfloatArray dst, jfloatArray src, jint len) {
    (void)clazz;
    if (len <= 0) return;
    jfloat* srcPtr = (*env)->GetFloatArrayElements(env, src, NULL);
    jfloat* dstPtr = (*env)->GetFloatArrayElements(env, dst, NULL);
    if (srcPtr && dstPtr) {
        candle_accelerate_vvsqrtf(dstPtr, srcPtr, len);
    }
    if (srcPtr) (*env)->ReleaseFloatArrayElements(env, src, srcPtr, JNI_ABORT);
    if (dstPtr) (*env)->ReleaseFloatArrayElements(env, dst, dstPtr, 0);
}

JNIEXPORT void JNICALL
Java_io_github_kotlinmania_candle_internal_CandleJni_accelerateVvtanhf(
    JNIEnv* env, jclass clazz, jfloatArray dst, jfloatArray src, jint len) {
    (void)clazz;
    if (len <= 0) return;
    jfloat* srcPtr = (*env)->GetFloatArrayElements(env, src, NULL);
    jfloat* dstPtr = (*env)->GetFloatArrayElements(env, dst, NULL);
    if (srcPtr && dstPtr) {
        candle_accelerate_vvtanhf(dstPtr, srcPtr, len);
    }
    if (srcPtr) (*env)->ReleaseFloatArrayElements(env, src, srcPtr, JNI_ABORT);
    if (dstPtr) (*env)->ReleaseFloatArrayElements(env, dst, dstPtr, 0);
}
