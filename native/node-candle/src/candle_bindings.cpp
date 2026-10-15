// candle_bindings.cpp — N-API module init for candle-kotlin native SIMD/Accelerate addon
// Shared between Kotlin/JS and Kotlin/WASM targets.

#include <napi.h>
#include "candle_wrapper.h"

namespace {

Napi::Value NeonReduceOneF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array x = info[0].As<Napi::Float32Array>();
    return Napi::Number::New(env, candle_neon_reduce_one_f32(x.Data()));
}

Napi::Value NeonReduceOneArmF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array x = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    return Napi::Number::New(env, candle_neon_reduce_one_arm_f32(x.Data() + offset));
}

Napi::Value NeonVdupqNF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    float v = info[0].As<Napi::Number>().FloatValue();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_vdupq_n_f32(v, out.Data());
    return out;
}

Napi::Value NeonVld1qF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_vld1q_f32(mem.Data() + offset, out.Data());
    return out;
}

Napi::Value NeonVaddqF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_vaddq_f32(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value NeonVfmaqF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array c = info[2].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_vfmaq_f32(a.Data(), b.Data(), c.Data(), out.Data());
    return out;
}

Napi::Value NeonVst1qF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[2].As<Napi::Float32Array>();
    candle_neon_vst1q_f32(mem.Data() + offset, a.Data());
    return env.Undefined();
}

Napi::Value NeonF16LoadFcvtl(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_f16_load_fcvtl(reinterpret_cast<const uint16_t*>(mem.Data() + offset), out.Data());
    return out;
}

Napi::Value NeonF16StoreFcvtn(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[2].As<Napi::Float32Array>();
    candle_neon_f16_store_fcvtn(reinterpret_cast<uint16_t*>(mem.Data() + offset), a.Data());
    return env.Undefined();
}

Napi::Value NeonF16ReduceOne(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array x = info[0].As<Napi::Int16Array>();
    return Napi::Number::New(env, candle_neon_f16_reduce_one(reinterpret_cast<const uint16_t*>(x.Data())));
}

Napi::Value NeonF16Zero(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_f16_zero(reinterpret_cast<uint16_t*>(out.Data()));
    return out;
}

Napi::Value NeonF16VecAdd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_f16_vec_add(
        reinterpret_cast<const uint16_t*>(a.Data()),
        reinterpret_cast<const uint16_t*>(b.Data()),
        reinterpret_cast<uint16_t*>(out.Data())
    );
    return out;
}

Napi::Value NeonF16VecFma(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array c = info[2].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_f16_vec_fma(
        reinterpret_cast<const uint16_t*>(a.Data()),
        reinterpret_cast<const uint16_t*>(b.Data()),
        reinterpret_cast<const uint16_t*>(c.Data()),
        reinterpret_cast<uint16_t*>(out.Data())
    );
    return out;
}

Napi::Value NeonF16FromF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    float v = info[0].As<Napi::Number>().FloatValue();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_f16_from_f32(v, reinterpret_cast<uint16_t*>(out.Data()));
    return out;
}

Napi::Value NeonF16VecStore(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Int16Array a = info[2].As<Napi::Int16Array>();
    candle_neon_f16_vec_store(
        reinterpret_cast<uint16_t*>(mem.Data() + offset),
        reinterpret_cast<const uint16_t*>(a.Data())
    );
    return env.Undefined();
}

Napi::Value NeonBf16LoadShll(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_neon_bf16_load_shll(reinterpret_cast<const uint16_t*>(mem.Data() + offset), out.Data());
    return out;
}

Napi::Value NeonBf16StoreShrn(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[2].As<Napi::Float32Array>();
    candle_neon_bf16_store_shrn(reinterpret_cast<uint16_t*>(mem.Data() + offset), a.Data());
    return env.Undefined();
}

Napi::Value NeonBf16VecAdd(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_bf16_vec_add(
        reinterpret_cast<const uint16_t*>(a.Data()),
        reinterpret_cast<const uint16_t*>(b.Data()),
        reinterpret_cast<uint16_t*>(out.Data())
    );
    return out;
}

Napi::Value NeonBf16VecFma(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array c = info[2].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_neon_bf16_vec_fma(
        reinterpret_cast<const uint16_t*>(a.Data()),
        reinterpret_cast<const uint16_t*>(b.Data()),
        reinterpret_cast<const uint16_t*>(c.Data()),
        reinterpret_cast<uint16_t*>(out.Data())
    );
    return out;
}

Napi::Value AvxZero(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_zero(out.Data());
    return out;
}

Napi::Value AvxSet1Ps(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    float v = info[0].As<Napi::Number>().FloatValue();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_set1_ps(v, out.Data());
    return out;
}

Napi::Value AvxLoaduPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_loadu_ps(mem.Data() + offset, out.Data());
    return out;
}

Napi::Value AvxStoreuPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[2].As<Napi::Float32Array>();
    candle_avx_storeu_ps(mem.Data() + offset, a.Data());
    return env.Undefined();
}

Napi::Value AvxReduceOnePs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array x = info[0].As<Napi::Float32Array>();
    return Napi::Number::New(env, candle_avx_reduce_one_ps(x.Data()));
}

Napi::Value AvxAddPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_add_ps(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value AvxFmaPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array c = info[2].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_fma_ps(a.Data(), b.Data(), c.Data(), out.Data());
    return out;
}

Napi::Value AvxCvtphPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_cvtph_ps(reinterpret_cast<const uint16_t*>(mem.Data() + offset), out.Data());
    return out;
}

Napi::Value AvxCvtpsPh(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Int16Array mem = info[1].As<Napi::Int16Array>();
    int32_t offset = info[2].As<Napi::Number>().Int32Value();
    candle_avx_cvtps_ph(a.Data(), reinterpret_cast<uint16_t*>(mem.Data() + offset));
    return env.Undefined();
}

Napi::Value AvxCvtpbhPs(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array mem = info[0].As<Napi::Int16Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 8);
    candle_avx_cvtpbh_ps(reinterpret_cast<const uint16_t*>(mem.Data() + offset), out.Data());
    return out;
}

Napi::Value AvxCvtnepsPbh(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Int16Array mem = info[1].As<Napi::Int16Array>();
    int32_t offset = info[2].As<Napi::Number>().Int32Value();
    candle_avx_cvtneps_pbh(a.Data(), reinterpret_cast<uint16_t*>(mem.Data() + offset));
    return env.Undefined();
}

Napi::Value Simd128SplatF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    float v = info[0].As<Napi::Number>().FloatValue();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_simd128_splat_f32(v, out.Data());
    return out;
}

Napi::Value Simd128LoadF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_simd128_load_f32(mem.Data() + offset, out.Data());
    return out;
}

Napi::Value Simd128AddF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_simd128_add_f32(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value Simd128MulF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_simd128_mul_f32(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value Simd128FmaF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array a = info[0].As<Napi::Float32Array>();
    Napi::Float32Array b = info[1].As<Napi::Float32Array>();
    Napi::Float32Array c = info[2].As<Napi::Float32Array>();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_simd128_fma_f32(a.Data(), b.Data(), c.Data(), out.Data());
    return out;
}

Napi::Value Simd128StoreF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array mem = info[0].As<Napi::Float32Array>();
    int32_t offset = info[1].As<Napi::Number>().Int32Value();
    Napi::Float32Array a = info[2].As<Napi::Float32Array>();
    candle_simd128_store_f32(mem.Data() + offset, a.Data());
    return env.Undefined();
}

Napi::Value Simd128ReduceOneF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array x = info[0].As<Napi::Float32Array>();
    return Napi::Number::New(env, candle_simd128_reduce_one_f32(x.Data()));
}

Napi::Value AccelerateSgemm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int8_t transa = static_cast<int8_t>(info[0].As<Napi::Number>().Int32Value());
    int8_t transb = static_cast<int8_t>(info[1].As<Napi::Number>().Int32Value());
    int32_t m = info[2].As<Napi::Number>().Int32Value();
    int32_t n = info[3].As<Napi::Number>().Int32Value();
    int32_t k = info[4].As<Napi::Number>().Int32Value();
    float alpha = info[5].As<Napi::Number>().FloatValue();
    Napi::Float32Array a = info[6].As<Napi::Float32Array>();
    int32_t lda = info[7].As<Napi::Number>().Int32Value();
    Napi::Float32Array b = info[8].As<Napi::Float32Array>();
    int32_t ldb = info[9].As<Napi::Number>().Int32Value();
    float beta = info[10].As<Napi::Number>().FloatValue();
    Napi::Float32Array c = info[11].As<Napi::Float32Array>();
    int32_t ldc = info[12].As<Napi::Number>().Int32Value();
    candle_accelerate_sgemm(transa, transb, m, n, k, alpha, a.Data(), lda, b.Data(), ldb, beta, c.Data(), ldc);
    return env.Undefined();
}

Napi::Value AccelerateDgemm(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int8_t transa = static_cast<int8_t>(info[0].As<Napi::Number>().Int32Value());
    int8_t transb = static_cast<int8_t>(info[1].As<Napi::Number>().Int32Value());
    int32_t m = info[2].As<Napi::Number>().Int32Value();
    int32_t n = info[3].As<Napi::Number>().Int32Value();
    int32_t k = info[4].As<Napi::Number>().Int32Value();
    double alpha = info[5].As<Napi::Number>().DoubleValue();
    Napi::Float64Array a = info[6].As<Napi::Float64Array>();
    int32_t lda = info[7].As<Napi::Number>().Int32Value();
    Napi::Float64Array b = info[8].As<Napi::Float64Array>();
    int32_t ldb = info[9].As<Napi::Number>().Int32Value();
    double beta = info[10].As<Napi::Number>().DoubleValue();
    Napi::Float64Array c = info[11].As<Napi::Float64Array>();
    int32_t ldc = info[12].As<Napi::Number>().Int32Value();
    candle_accelerate_dgemm(transa, transb, m, n, k, alpha, a.Data(), lda, b.Data(), ldb, beta, c.Data(), ldc);
    return env.Undefined();
}

Napi::Value AccelerateVvexpf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvexpf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvexp(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvexp(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvsqrtf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvsqrtf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvsqrt(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvsqrt(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvsinf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvsinf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvsin(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvsin(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvcosf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvcosf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvcos(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvcos(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvlogf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvlogf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvlog(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvlog(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvtanhf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvtanhf(dst.Data(), src.Data(), len);
    return env.Undefined();
}

Napi::Value AccelerateVvtanh(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float64Array dst = info[0].As<Napi::Float64Array>();
    Napi::Float64Array src = info[1].As<Napi::Float64Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvtanh(dst.Data(), src.Data(), len);
    return env.Undefined();
}

#define DEFINE_VDSP_F32_NAPI(NapiName, CFunc) \
Napi::Value NapiName(const Napi::CallbackInfo& info) { \
    Napi::Env env = info.Env(); \
    Napi::Float32Array a = info[0].As<Napi::Float32Array>(); \
    int64_t ia = info[1].As<Napi::Number>().Int64Value(); \
    Napi::Float32Array b = info[2].As<Napi::Float32Array>(); \
    int64_t ib = info[3].As<Napi::Number>().Int64Value(); \
    Napi::Float32Array c = info[4].As<Napi::Float32Array>(); \
    int64_t ic = info[5].As<Napi::Number>().Int64Value(); \
    uint64_t n = static_cast<uint64_t>(info[6].As<Napi::Number>().Int64Value()); \
    CFunc(a.Data(), ia, b.Data(), ib, c.Data(), ic, n); \
    return env.Undefined(); \
}

#define DEFINE_VDSP_F64_NAPI(NapiName, CFunc) \
Napi::Value NapiName(const Napi::CallbackInfo& info) { \
    Napi::Env env = info.Env(); \
    Napi::Float64Array a = info[0].As<Napi::Float64Array>(); \
    int64_t ia = info[1].As<Napi::Number>().Int64Value(); \
    Napi::Float64Array b = info[2].As<Napi::Float64Array>(); \
    int64_t ib = info[3].As<Napi::Number>().Int64Value(); \
    Napi::Float64Array c = info[4].As<Napi::Float64Array>(); \
    int64_t ic = info[5].As<Napi::Number>().Int64Value(); \
    uint64_t n = static_cast<uint64_t>(info[6].As<Napi::Number>().Int64Value()); \
    CFunc(a.Data(), ia, b.Data(), ib, c.Data(), ic, n); \
    return env.Undefined(); \
}

DEFINE_VDSP_F32_NAPI(AccelerateVdspVadd, candle_accelerate_vdsp_vadd)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVaddD, candle_accelerate_vdsp_vaddd)
DEFINE_VDSP_F32_NAPI(AccelerateVdspVsub, candle_accelerate_vdsp_vsub)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVsubD, candle_accelerate_vdsp_vsubd)
DEFINE_VDSP_F32_NAPI(AccelerateVdspVmul, candle_accelerate_vdsp_vmul)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVmulD, candle_accelerate_vdsp_vmuld)
DEFINE_VDSP_F32_NAPI(AccelerateVdspVdiv, candle_accelerate_vdsp_vdiv)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVdivD, candle_accelerate_vdsp_vdivd)
DEFINE_VDSP_F32_NAPI(AccelerateVdspVmin, candle_accelerate_vdsp_vmin)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVminD, candle_accelerate_vdsp_vmind)
DEFINE_VDSP_F32_NAPI(AccelerateVdspVmax, candle_accelerate_vdsp_vmax)
DEFINE_VDSP_F64_NAPI(AccelerateVdspVmaxD, candle_accelerate_vdsp_vmaxd)

Napi::Value QneonVdotqS32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int8Array a = info[0].As<Napi::Int8Array>();
    Napi::Int8Array b = info[1].As<Napi::Int8Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qneon_vdotq_s32(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QneonVmullS8(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int8Array a = info[0].As<Napi::Int8Array>();
    Napi::Int8Array b = info[1].As<Napi::Int8Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_qneon_vmull_s8(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QneonVmullS16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qneon_vmull_s16(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QneonVpaddlqS16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qneon_vpaddlq_s16(a.Data(), out.Data());
    return out;
}

Napi::Value QneonVpaddqS16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_qneon_vpaddq_s16(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QneonVmlaqNF32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array acc = info[0].As<Napi::Float32Array>();
    Napi::Float32Array v = info[1].As<Napi::Float32Array>();
    float s = info[2].As<Napi::Number>().FloatValue();
    Napi::Float32Array out = Napi::Float32Array::New(env, 4);
    candle_qneon_vmlaq_n_f32(acc.Data(), v.Data(), s, out.Data());
    return out;
}

Napi::Value QneonVaddvqS32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int32Array a = info[0].As<Napi::Int32Array>();
    return Napi::Number::New(env, candle_qneon_vaddvq_s32(a.Data()));
}

Napi::Value QavxMaddEpi16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 8);
    candle_qavx_madd_epi16(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QavxMmMaddEpi16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qavx_mm_madd_epi16(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QavxMaddubsEpi16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int8Array ax = info[0].As<Napi::Int8Array>();
    Napi::Int8Array sy = info[1].As<Napi::Int8Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 16);
    candle_qavx_maddubs_epi16(ax.Data(), sy.Data(), out.Data());
    return out;
}

Napi::Value QavxSignEpi8(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int8Array a = info[0].As<Napi::Int8Array>();
    Napi::Int8Array b = info[1].As<Napi::Int8Array>();
    Napi::Int8Array out = Napi::Int8Array::New(env, 32);
    candle_qavx_sign_epi8(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QavxHaddEpi16(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int16Array out = Napi::Int16Array::New(env, 8);
    candle_qavx_hadd_epi16(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value QavxHaddEpi32(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int32Array a = info[0].As<Napi::Int32Array>();
    Napi::Int32Array b = info[1].As<Napi::Int32Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qavx_hadd_epi32(a.Data(), b.Data(), out.Data());
    return out;
}

Napi::Value Qsimd128DotI16x8(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Int16Array a = info[0].As<Napi::Int16Array>();
    Napi::Int16Array b = info[1].As<Napi::Int16Array>();
    Napi::Int32Array out = Napi::Int32Array::New(env, 4);
    candle_qsimd128_dot_i16x8(a.Data(), b.Data(), out.Data());
    return out;
}

} // namespace

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("neonReduceOneF32", Napi::Function::New(env, NeonReduceOneF32));
    exports.Set("neonReduceOneArmF32", Napi::Function::New(env, NeonReduceOneArmF32));
    exports.Set("neonVdupqNF32", Napi::Function::New(env, NeonVdupqNF32));
    exports.Set("neonVld1qF32", Napi::Function::New(env, NeonVld1qF32));
    exports.Set("neonVaddqF32", Napi::Function::New(env, NeonVaddqF32));
    exports.Set("neonVfmaqF32", Napi::Function::New(env, NeonVfmaqF32));
    exports.Set("neonVst1qF32", Napi::Function::New(env, NeonVst1qF32));
    exports.Set("neonF16LoadFcvtl", Napi::Function::New(env, NeonF16LoadFcvtl));
    exports.Set("neonF16StoreFcvtn", Napi::Function::New(env, NeonF16StoreFcvtn));
    exports.Set("neonF16ReduceOne", Napi::Function::New(env, NeonF16ReduceOne));
    exports.Set("neonF16Zero", Napi::Function::New(env, NeonF16Zero));
    exports.Set("neonF16VecAdd", Napi::Function::New(env, NeonF16VecAdd));
    exports.Set("neonF16VecFma", Napi::Function::New(env, NeonF16VecFma));
    exports.Set("neonF16FromF32", Napi::Function::New(env, NeonF16FromF32));
    exports.Set("neonF16VecStore", Napi::Function::New(env, NeonF16VecStore));
    exports.Set("neonBf16LoadShll", Napi::Function::New(env, NeonBf16LoadShll));
    exports.Set("neonBf16StoreShrn", Napi::Function::New(env, NeonBf16StoreShrn));
    exports.Set("neonBf16VecAdd", Napi::Function::New(env, NeonBf16VecAdd));
    exports.Set("neonBf16VecFma", Napi::Function::New(env, NeonBf16VecFma));
    exports.Set("avxZero", Napi::Function::New(env, AvxZero));
    exports.Set("avxSet1Ps", Napi::Function::New(env, AvxSet1Ps));
    exports.Set("avxLoaduPs", Napi::Function::New(env, AvxLoaduPs));
    exports.Set("avxStoreuPs", Napi::Function::New(env, AvxStoreuPs));
    exports.Set("avxReduceOnePs", Napi::Function::New(env, AvxReduceOnePs));
    exports.Set("avxAddPs", Napi::Function::New(env, AvxAddPs));
    exports.Set("avxFmaPs", Napi::Function::New(env, AvxFmaPs));
    exports.Set("avxCvtphPs", Napi::Function::New(env, AvxCvtphPs));
    exports.Set("avxCvtpsPh", Napi::Function::New(env, AvxCvtpsPh));
    exports.Set("avxCvtpbhPs", Napi::Function::New(env, AvxCvtpbhPs));
    exports.Set("avxCvtnepsPbh", Napi::Function::New(env, AvxCvtnepsPbh));
    exports.Set("simd128SplatF32", Napi::Function::New(env, Simd128SplatF32));
    exports.Set("simd128LoadF32", Napi::Function::New(env, Simd128LoadF32));
    exports.Set("simd128AddF32", Napi::Function::New(env, Simd128AddF32));
    exports.Set("simd128MulF32", Napi::Function::New(env, Simd128MulF32));
    exports.Set("simd128FmaF32", Napi::Function::New(env, Simd128FmaF32));
    exports.Set("simd128StoreF32", Napi::Function::New(env, Simd128StoreF32));
    exports.Set("simd128ReduceOneF32", Napi::Function::New(env, Simd128ReduceOneF32));
    exports.Set("accelerateSgemm", Napi::Function::New(env, AccelerateSgemm));
    exports.Set("accelerateDgemm", Napi::Function::New(env, AccelerateDgemm));
    exports.Set("accelerateVvexpf", Napi::Function::New(env, AccelerateVvexpf));
    exports.Set("accelerateVvexp", Napi::Function::New(env, AccelerateVvexp));
    exports.Set("accelerateVvsqrtf", Napi::Function::New(env, AccelerateVvsqrtf));
    exports.Set("accelerateVvsqrt", Napi::Function::New(env, AccelerateVvsqrt));
    exports.Set("accelerateVvsinf", Napi::Function::New(env, AccelerateVvsinf));
    exports.Set("accelerateVvsin", Napi::Function::New(env, AccelerateVvsin));
    exports.Set("accelerateVvcosf", Napi::Function::New(env, AccelerateVvcosf));
    exports.Set("accelerateVvcos", Napi::Function::New(env, AccelerateVvcos));
    exports.Set("accelerateVvlogf", Napi::Function::New(env, AccelerateVvlogf));
    exports.Set("accelerateVvlog", Napi::Function::New(env, AccelerateVvlog));
    exports.Set("accelerateVvtanhf", Napi::Function::New(env, AccelerateVvtanhf));
    exports.Set("accelerateVvtanh", Napi::Function::New(env, AccelerateVvtanh));
    exports.Set("accelerateVdspVadd", Napi::Function::New(env, AccelerateVdspVadd));
    exports.Set("accelerateVdspVaddD", Napi::Function::New(env, AccelerateVdspVaddD));
    exports.Set("accelerateVdspVsub", Napi::Function::New(env, AccelerateVdspVsub));
    exports.Set("accelerateVdspVsubD", Napi::Function::New(env, AccelerateVdspVsubD));
    exports.Set("accelerateVdspVmul", Napi::Function::New(env, AccelerateVdspVmul));
    exports.Set("accelerateVdspVmulD", Napi::Function::New(env, AccelerateVdspVmulD));
    exports.Set("accelerateVdspVdiv", Napi::Function::New(env, AccelerateVdspVdiv));
    exports.Set("accelerateVdspVdivD", Napi::Function::New(env, AccelerateVdspVdivD));
    exports.Set("accelerateVdspVmin", Napi::Function::New(env, AccelerateVdspVmin));
    exports.Set("accelerateVdspVminD", Napi::Function::New(env, AccelerateVdspVminD));
    exports.Set("accelerateVdspVmax", Napi::Function::New(env, AccelerateVdspVmax));
    exports.Set("accelerateVdspVmaxD", Napi::Function::New(env, AccelerateVdspVmaxD));
    exports.Set("qneonVdotqS32", Napi::Function::New(env, QneonVdotqS32));
    exports.Set("qneonVmullS8", Napi::Function::New(env, QneonVmullS8));
    exports.Set("qneonVmullS16", Napi::Function::New(env, QneonVmullS16));
    exports.Set("qneonVpaddlqS16", Napi::Function::New(env, QneonVpaddlqS16));
    exports.Set("qneonVpaddqS16", Napi::Function::New(env, QneonVpaddqS16));
    exports.Set("qneonVmlaqNF32", Napi::Function::New(env, QneonVmlaqNF32));
    exports.Set("qneonVaddvqS32", Napi::Function::New(env, QneonVaddvqS32));
    exports.Set("qavxMaddEpi16", Napi::Function::New(env, QavxMaddEpi16));
    exports.Set("qavxMmMaddEpi16", Napi::Function::New(env, QavxMmMaddEpi16));
    exports.Set("qavxMaddubsEpi16", Napi::Function::New(env, QavxMaddubsEpi16));
    exports.Set("qavxSignEpi8", Napi::Function::New(env, QavxSignEpi8));
    exports.Set("qavxHaddEpi16", Napi::Function::New(env, QavxHaddEpi16));
    exports.Set("qavxHaddEpi32", Napi::Function::New(env, QavxHaddEpi32));
    exports.Set("qsimd128DotI16x8", Napi::Function::New(env, Qsimd128DotI16x8));
    return exports;
}

NODE_API_MODULE(candle_native, Init)
