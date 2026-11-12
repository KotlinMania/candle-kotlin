#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <napi.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif
#if __has_include("km_core_intrinsics_inline.h")
#include "km_core_intrinsics_inline.h"
#elif __has_include("../../../native/interop/cinterop/km_core_intrinsics_inline.h")
#include "../../../native/interop/cinterop/km_core_intrinsics_inline.h"
#endif

struct KmNapiRegistrarNode {
    void (*register_fn)(Napi::Env, Napi::Object);
    KmNapiRegistrarNode* next;
};
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
extern KmNapiRegistrarNode* km_napi_registrar_head;

namespace {

static int64_t kmCoreNapiArgI64(const Napi::Value& v) {
    if (v.IsNull() || v.IsUndefined()) return 0;
    if (v.IsString()) {
        std::string s = v.As<Napi::String>().Utf8Value();
        return static_cast<int64_t>(std::strtoll(s.c_str(), nullptr, 10));
    }
    if (v.IsNumber()) return v.As<Napi::Number>().Int64Value();
    return 0;
}

static uint64_t kmCoreNapiArgU64(const Napi::Value& v) {
    if (v.IsNull() || v.IsUndefined()) return 0;
    if (v.IsString()) {
        std::string s = v.As<Napi::String>().Utf8Value();
        return static_cast<uint64_t>(std::strtoull(s.c_str(), nullptr, 10));
    }
    if (v.IsNumber()) return static_cast<uint64_t>(v.As<Napi::Number>().Int64Value());
    return 0;
}

static inline void kmCoreCpuSpinHint(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
    _mm_pause();
#else
    __asm__ __volatile__("pause" ::: "memory");
#endif
#endif
}

static inline uint64_t kmCoreCpuBlackBoxU64(uint64_t v) {
#if defined(__clang__) || defined(__GNUC__)
    __asm__ __volatile__("" : "+r"(v) :: "memory");
    return v;
#else
    volatile uint64_t sink = v;
    return sink;
#endif
}

static inline uint64_t kmCoreCpuUmulhU64(uint64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b)) >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
    uint64_t hi = 0;
    (void)_umul128(a, b, &hi);
    return hi;
#else
    uint64_t aLo = a & 0xFFFFFFFFULL, aHi = a >> 32;
    uint64_t bLo = b & 0xFFFFFFFFULL, bHi = b >> 32;
    uint64_t p0 = aLo * bLo, p1 = aLo * bHi, p2 = aHi * bLo, p3 = aHi * bHi;
    uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
    return p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
#endif
}

static inline int64_t kmCoreCpuSmulhI64(int64_t a, int64_t b) {
#if defined(__SIZEOF_INT128__)
    return static_cast<int64_t>((static_cast<__int128>(a) * static_cast<__int128>(b)) >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
    int64_t hi = 0;
    (void)_mul128(a, b, &hi);
    return hi;
#else
    uint64_t hi = kmCoreCpuUmulhU64(static_cast<uint64_t>(a), static_cast<uint64_t>(b));
    if (a < 0) hi -= static_cast<uint64_t>(b);
    if (b < 0) hi -= static_cast<uint64_t>(a);
    return static_cast<int64_t>(hi);
#endif
}

static inline uint8_t kmCoreCpuAtomicCxchgU128(
    int64_t addr, uint64_t expLo, uint64_t expHi, uint64_t desLo, uint64_t desHi, uint64_t* outLo, uint64_t* outHi
) {
#if defined(_MSC_VER) && defined(_M_X64)
    int64_t cmp[2] = { static_cast<int64_t>(expLo), static_cast<int64_t>(expHi) };
    unsigned char ok = _InterlockedCompareExchange128(
        reinterpret_cast<volatile int64_t*>(static_cast<uintptr_t>(addr)),
        static_cast<int64_t>(desHi),
        static_cast<int64_t>(desLo),
        cmp
    );
    *outLo = static_cast<uint64_t>(cmp[0]);
    *outHi = static_cast<uint64_t>(cmp[1]);
    return ok ? 1 : 0;
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    uint64_t curLo = 0, curHi = 0;
    uint32_t status = 0;
    void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
    for (;;) {
        __asm__ __volatile__("ldaxp %0, %1, [%2]" : "=&r"(curLo), "=&r"(curHi) : "r"(ptr) : "memory");
        if (curLo != expLo || curHi != expHi) {
            __asm__ __volatile__("clrex" ::: "memory");
            *outLo = curLo;
            *outHi = curHi;
            return 0;
        }
        __asm__ __volatile__("stlxp %w0, %1, %2, [%3]" : "=&r"(status) : "r"(desLo), "r"(desHi), "r"(ptr) : "memory");
        if (status == 0) {
            *outLo = expLo;
            *outHi = expHi;
            return 1;
        }
    }
#elif defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    uint8_t ok = 0;
    uint64_t prevLo = expLo;
    uint64_t prevHi = expHi;
    __uint128_t* memPtr = reinterpret_cast<__uint128_t*>(static_cast<uintptr_t>(addr));
    __asm__ __volatile__(
        "lock cmpxchg16b %1\n\t"
        "sete %0"
        : "=q"(ok), "+m"(*memPtr), "+a"(prevLo), "+d"(prevHi)
        : "b"(desLo), "c"(desHi)
        : "cc", "memory"
    );
    *outLo = prevLo;
    *outHi = prevHi;
    return ok;
#else
    uint64_t* p = reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(addr));
    *outLo = p[0];
    *outHi = p[1];
    if (p[0] == expLo && p[1] == expHi) {
        p[0] = desLo;
        p[1] = desHi;
        return 1;
    }
    return 0;
#endif
}

static void kmRegisterCoreNapiObject(Napi::Env env) {
    Napi::Object global = env.Global();
    if (global.Has("kmCoreNapi") && global.Get("kmCoreNapi").IsObject()) {
        return;
    }
    Napi::Object coreObj = Napi::Object::New(env);
    coreObj.Set("kmLoadU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmLoadU8(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmLoadU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmLoadU16(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmLoadU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmLoadU32(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmLoadU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmLoadU64(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmLoadF32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<double>(kmLoadF32(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmLoadF64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), kmLoadF64(kmCoreNapiArgI64(info[0])));
    }));
    coreObj.Set("kmStoreU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmStoreU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmStoreU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmStoreU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmStoreF32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreF32(kmCoreNapiArgI64(info[0]), static_cast<float>(info[1].As<Napi::Number>().DoubleValue()));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmStoreF64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmStoreF64(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().DoubleValue());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmVolatileLoadU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmVolatileLoadU8(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmVolatileLoadU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmVolatileLoadU16(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmVolatileLoadU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmVolatileLoadU32(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmVolatileLoadU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmVolatileLoadU64(kmCoreNapiArgI64(info[0]))));
    }));
    coreObj.Set("kmVolatileStoreU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmVolatileStoreU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmVolatileStoreU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmVolatileStoreU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmVolatileStoreU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmVolatileStoreU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmVolatileStoreU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmVolatileStoreU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmCopyNonoverlapping", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmCopyNonoverlapping(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmCopy", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmCopy(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmWriteBytes", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmWriteBytes(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()), kmCoreNapiArgI64(info[2]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmCompareBytes", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), kmCompareBytes(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2])));
    }));
    coreObj.Set("kmAlloc", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmAlloc(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]))));
    }));
    coreObj.Set("kmAllocZeroed", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmAllocZeroed(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]))));
    }));
    coreObj.Set("kmRealloc", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmRealloc(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]), kmCoreNapiArgI64(info[3]))));
    }));
    coreObj.Set("kmDealloc", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmDealloc(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicFence", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicFence(info[0].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicSignalFence", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicSignalFence(info[0].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicLoadU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicLoadU8(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicLoadU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicLoadU16(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicLoadU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmAtomicLoadU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicLoadU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmAtomicLoadU64(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicStoreU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicStoreU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicStoreU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicStoreU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicStoreU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicStoreU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value(), info[2].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicStoreU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmAtomicStoreU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]), info[2].As<Napi::Number>().Int32Value());
        return info.Env().Undefined();
    }));
    coreObj.Set("kmAtomicXchgU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicXchgU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicXchgU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicXchgU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicXchgU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmAtomicXchgU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value(), info[2].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicXchgU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmAtomicXchgU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]), info[2].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicCxchgU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        uint8_t ok = 0;
        uint8_t prev = kmAtomicCxchgU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()), static_cast<uint8_t>(info[2].As<Napi::Number>().Uint32Value()), info[3].As<Napi::Number>().Int32Value(), info[4].As<Napi::Number>().Int32Value(), &ok);
        return Napi::Number::New(info.Env(), static_cast<int32_t>((static_cast<uint32_t>(ok) << 8) | static_cast<uint32_t>(prev)));
    }));
    coreObj.Set("kmAtomicCxchgU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        uint8_t ok = 0;
        uint16_t prev = kmAtomicCxchgU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()), static_cast<uint16_t>(info[2].As<Napi::Number>().Uint32Value()), info[3].As<Napi::Number>().Int32Value(), info[4].As<Napi::Number>().Int32Value(), &ok);
        return Napi::Number::New(info.Env(), static_cast<int32_t>((static_cast<uint32_t>(ok) << 16) | static_cast<uint32_t>(prev)));
    }));
    coreObj.Set("kmAtomicCxchgU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        uint8_t ok = 0;
        uint32_t prev = kmAtomicCxchgU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value(), info[2].As<Napi::Number>().Uint32Value(), info[3].As<Napi::Number>().Int32Value(), info[4].As<Napi::Number>().Int32Value(), &ok);
        return Napi::String::New(info.Env(), std::to_string(prev) + ":" + (ok ? "1" : "0"));
    }));
    coreObj.Set("kmAtomicCxchgU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        uint8_t ok = 0;
        uint64_t prev = kmAtomicCxchgU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]), kmCoreNapiArgU64(info[2]), info[3].As<Napi::Number>().Int32Value(), info[4].As<Napi::Number>().Int32Value(), &ok);
        return Napi::String::New(info.Env(), std::to_string(prev) + ":" + (ok ? "1" : "0"));
    }));
    coreObj.Set("kmAtomicRmwU8", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicRmwU8(kmCoreNapiArgI64(info[0]), static_cast<uint8_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value(), info[3].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicRmwU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmAtomicRmwU16(kmCoreNapiArgI64(info[0]), static_cast<uint16_t>(info[1].As<Napi::Number>().Uint32Value()), info[2].As<Napi::Number>().Int32Value(), info[3].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicRmwU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmAtomicRmwU32(kmCoreNapiArgI64(info[0]), info[1].As<Napi::Number>().Uint32Value(), info[2].As<Napi::Number>().Int32Value(), info[3].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmAtomicRmwU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmAtomicRmwU64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgU64(info[1]), info[2].As<Napi::Number>().Int32Value(), info[3].As<Napi::Number>().Int32Value())));
    }));
    coreObj.Set("kmCtpopU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmCtpopU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmCtlzU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmCtlzU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmCttzU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmCttzU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmBswapU16", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<uint32_t>(kmBswapU16(static_cast<uint16_t>(info[0].As<Napi::Number>().Uint32Value()))));
    }));
    coreObj.Set("kmBswapU32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<int32_t>(kmBswapU32(info[0].As<Napi::Number>().Uint32Value())));
    }));
    coreObj.Set("kmBswapU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmBswapU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmBitreverseU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmBitreverseU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmFmaf32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<double>(kmFmaf32(static_cast<float>(info[0].As<Napi::Number>().DoubleValue()), static_cast<float>(info[1].As<Napi::Number>().DoubleValue()), static_cast<float>(info[2].As<Napi::Number>().DoubleValue()))));
    }));
    coreObj.Set("kmFmaf64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), kmFmaf64(info[0].As<Napi::Number>().DoubleValue(), info[1].As<Napi::Number>().DoubleValue(), info[2].As<Napi::Number>().DoubleValue()));
    }));
    coreObj.Set("kmSqrtf32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<double>(kmSqrtf32(static_cast<float>(info[0].As<Napi::Number>().DoubleValue()))));
    }));
    coreObj.Set("kmSqrtf64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), kmSqrtf64(info[0].As<Napi::Number>().DoubleValue()));
    }));
    coreObj.Set("kmAbort", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        (void)info;
        kmAbort();
        return info.Env().Undefined();
    }));
    coreObj.Set("kmSpinLoopHint", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmCoreCpuSpinHint();
        return info.Env().Undefined();
    }));
    coreObj.Set("kmBlackBoxU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmCoreCpuBlackBoxU64(kmCoreNapiArgU64(info[0]))));
    }));
    coreObj.Set("kmUmulhU64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmCoreCpuUmulhU64(kmCoreNapiArgU64(info[0]), kmCoreNapiArgU64(info[1]))));
    }));
    coreObj.Set("kmSmulhI64", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::String::New(info.Env(), std::to_string(kmCoreCpuSmulhI64(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]))));
    }));
    coreObj.Set("kmAtomicCxchgU128Pair", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        uint64_t outLo = 0, outHi = 0;
        uint8_t ok = kmCoreCpuAtomicCxchgU128(
            kmCoreNapiArgI64(info[0]),
            kmCoreNapiArgU64(info[1]),
            kmCoreNapiArgU64(info[2]),
            kmCoreNapiArgU64(info[3]),
            kmCoreNapiArgU64(info[4]),
            &outLo,
            &outHi
        );
        return Napi::String::New(info.Env(), std::to_string(outLo) + ":" + std::to_string(outHi) + ":" + (ok ? "1" : "0"));
    }));
    coreObj.Set("kmSimdFmaF32x4", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        kmSimdFmaF32x4(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]), kmCoreNapiArgI64(info[3]));
        return info.Env().Undefined();
    }));
    coreObj.Set("kmSimdDotF32", Napi::Function::New(env, [](const Napi::CallbackInfo& info) -> Napi::Value {
        return Napi::Number::New(info.Env(), static_cast<double>(kmSimdDotF32(kmCoreNapiArgI64(info[0]), kmCoreNapiArgI64(info[1]), kmCoreNapiArgI64(info[2]))));
    }));
    global.Set("kmCoreNapi", coreObj);
}


static void register_candle_kmcore_intrinsics_napi(Napi::Env env, Napi::Object exports) {
    kmRegisterCoreNapiObject(env);
    Napi::Value coreVal = env.Global().Get("kmCoreNapi");
    if (coreVal.IsObject()) {
        Napi::Object coreObj = coreVal.As<Napi::Object>();
        Napi::Array keys = coreObj.GetPropertyNames();
        for (uint32_t i = 0; i < keys.Length(); i++) {
            Napi::Value k = keys.Get(i);
            exports.Set(k, coreObj.Get(k));
        }
    }
}

static KmNapiRegistrarNode registrar_node = { &register_candle_kmcore_intrinsics_napi, nullptr };
struct KmNapiAutoRegister {
    KmNapiAutoRegister() {
        registrar_node.next = km_napi_registrar_head;
        km_napi_registrar_head = &registrar_node;
    }
};
static KmNapiAutoRegister auto_register_instance;

} // namespace

#ifdef KM_NAPI_STANDALONE_MODULE
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("hidden")))
#endif
KmNapiRegistrarNode* km_napi_registrar_head = nullptr;
static Napi::Object KmNapiStandaloneInit(Napi::Env env, Napi::Object exports) {
    for (KmNapiRegistrarNode* cur = km_napi_registrar_head; cur != nullptr; cur = cur->next) {
        cur->register_fn(env, exports);
    }
    return exports;
}
NODE_API_MODULE(NODE_GYP_MODULE_NAME, KmNapiStandaloneInit)
#endif
