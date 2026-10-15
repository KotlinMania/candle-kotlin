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

Napi::Value AccelerateVvexpf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvexpf(dst.Data(), src.Data(), len);
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

Napi::Value AccelerateVvtanhf(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    Napi::Float32Array dst = info[0].As<Napi::Float32Array>();
    Napi::Float32Array src = info[1].As<Napi::Float32Array>();
    int32_t len = info[2].As<Napi::Number>().Int32Value();
    candle_accelerate_vvtanhf(dst.Data(), src.Data(), len);
    return env.Undefined();
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
    exports.Set("avxReduceOnePs", Napi::Function::New(env, AvxReduceOnePs));
    exports.Set("avxAddPs", Napi::Function::New(env, AvxAddPs));
    exports.Set("avxFmaPs", Napi::Function::New(env, AvxFmaPs));
    exports.Set("avxCvtphPs", Napi::Function::New(env, AvxCvtphPs));
    exports.Set("avxCvtpsPh", Napi::Function::New(env, AvxCvtpsPh));
    exports.Set("accelerateVvexpf", Napi::Function::New(env, AccelerateVvexpf));
    exports.Set("accelerateVvsqrtf", Napi::Function::New(env, AccelerateVvsqrtf));
    exports.Set("accelerateVvtanhf", Napi::Function::New(env, AccelerateVvtanhf));
    return exports;
}

NODE_API_MODULE(candle_native, Init)
