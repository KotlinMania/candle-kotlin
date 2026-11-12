#ifndef KM_CORE_INTRINSICS_INLINE_H
#define KM_CORE_INTRINSICS_INLINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if !defined(__cplusplus)
#include <stdatomic.h>
#elif defined(_MSC_VER) && !defined(__clang__)
#include <atomic>
#endif

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(__clang__) || defined(__GNUC__)
#define KM_ALWAYS_INLINE static inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define KM_ALWAYS_INLINE static __forceinline
#else
#define KM_ALWAYS_INLINE static inline
#endif

/* Aligned & unaligned scalar memory loads and stores */
KM_ALWAYS_INLINE uint8_t kmLoadU8(int64_t addr) {
    return *(const uint8_t*)(uintptr_t)addr;
}
KM_ALWAYS_INLINE uint16_t kmLoadU16(int64_t addr) {
    uint16_t v;
    memcpy(&v, (const void*)(uintptr_t)addr, 2);
    return v;
}
KM_ALWAYS_INLINE uint32_t kmLoadU32(int64_t addr) {
    uint32_t v;
    memcpy(&v, (const void*)(uintptr_t)addr, 4);
    return v;
}
KM_ALWAYS_INLINE uint64_t kmLoadU64(int64_t addr) {
    uint64_t v;
    memcpy(&v, (const void*)(uintptr_t)addr, 8);
    return v;
}
KM_ALWAYS_INLINE float kmLoadF32(int64_t addr) {
    float v;
    memcpy(&v, (const void*)(uintptr_t)addr, 4);
    return v;
}
KM_ALWAYS_INLINE double kmLoadF64(int64_t addr) {
    double v;
    memcpy(&v, (const void*)(uintptr_t)addr, 8);
    return v;
}

KM_ALWAYS_INLINE void kmStoreU8(int64_t addr, uint8_t val) {
    *(uint8_t*)(uintptr_t)addr = val;
}
KM_ALWAYS_INLINE void kmStoreU16(int64_t addr, uint16_t val) {
    memcpy((void*)(uintptr_t)addr, &val, 2);
}
KM_ALWAYS_INLINE void kmStoreU32(int64_t addr, uint32_t val) {
    memcpy((void*)(uintptr_t)addr, &val, 4);
}
KM_ALWAYS_INLINE void kmStoreU64(int64_t addr, uint64_t val) {
    memcpy((void*)(uintptr_t)addr, &val, 8);
}
KM_ALWAYS_INLINE void kmStoreF32(int64_t addr, float val) {
    memcpy((void*)(uintptr_t)addr, &val, 4);
}
KM_ALWAYS_INLINE void kmStoreF64(int64_t addr, double val) {
    memcpy((void*)(uintptr_t)addr, &val, 8);
}

/* Volatile loads and stores */
KM_ALWAYS_INLINE uint8_t kmVolatileLoadU8(int64_t addr) {
    return *(volatile const uint8_t*)(uintptr_t)addr;
}
KM_ALWAYS_INLINE uint16_t kmVolatileLoadU16(int64_t addr) {
    return *(volatile const uint16_t*)(uintptr_t)addr;
}
KM_ALWAYS_INLINE uint32_t kmVolatileLoadU32(int64_t addr) {
    return *(volatile const uint32_t*)(uintptr_t)addr;
}
KM_ALWAYS_INLINE uint64_t kmVolatileLoadU64(int64_t addr) {
    return *(volatile const uint64_t*)(uintptr_t)addr;
}

KM_ALWAYS_INLINE void kmVolatileStoreU8(int64_t addr, uint8_t val) {
    *(volatile uint8_t*)(uintptr_t)addr = val;
}
KM_ALWAYS_INLINE void kmVolatileStoreU16(int64_t addr, uint16_t val) {
    *(volatile uint16_t*)(uintptr_t)addr = val;
}
KM_ALWAYS_INLINE void kmVolatileStoreU32(int64_t addr, uint32_t val) {
    *(volatile uint32_t*)(uintptr_t)addr = val;
}
KM_ALWAYS_INLINE void kmVolatileStoreU64(int64_t addr, uint64_t val) {
    *(volatile uint64_t*)(uintptr_t)addr = val;
}

/* Bulk memory intrinsics */
KM_ALWAYS_INLINE void kmCopyNonoverlapping(int64_t dst, int64_t src, int64_t bytes) {
    if (bytes > 0) {
        memcpy((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, (size_t)bytes);
    }
}
KM_ALWAYS_INLINE void kmCopy(int64_t dst, int64_t src, int64_t bytes) {
    if (bytes > 0) {
        memmove((void*)(uintptr_t)dst, (const void*)(uintptr_t)src, (size_t)bytes);
    }
}
KM_ALWAYS_INLINE void kmWriteBytes(int64_t dst, uint8_t val, int64_t bytes) {
    if (bytes > 0) {
        memset((void*)(uintptr_t)dst, (int)val, (size_t)bytes);
    }
}
KM_ALWAYS_INLINE int32_t kmCompareBytes(int64_t left, int64_t right, int64_t bytes) {
    if (bytes <= 0) return 0;
    return (int32_t)memcmp((const void*)(uintptr_t)left, (const void*)(uintptr_t)right, (size_t)bytes);
}

/* Aligned system allocator backing GlobalAlloc */
KM_ALWAYS_INLINE int64_t kmAlloc(int64_t size, int64_t align) {
    size_t sz = size <= 0 ? 1 : (size_t)size;
    size_t al = align <= 1 ? 1 : (size_t)align;
#if defined(_WIN32)
    return (int64_t)(uintptr_t)_aligned_malloc(sz, al < sizeof(void*) ? sizeof(void*) : al);
#else
    if (al <= sizeof(void*) * 2) {
        return (int64_t)(uintptr_t)malloc(sz);
    }
    void* out = NULL;
    if (posix_memalign(&out, al, sz) != 0) return 0;
    return (int64_t)(uintptr_t)out;
#endif
}

KM_ALWAYS_INLINE int64_t kmAllocZeroed(int64_t size, int64_t align) {
    size_t sz = size <= 0 ? 1 : (size_t)size;
    size_t al = align <= 1 ? 1 : (size_t)align;
#if defined(_WIN32)
    void* ptr = _aligned_malloc(sz, al < sizeof(void*) ? sizeof(void*) : al);
    if (ptr != NULL) memset(ptr, 0, sz);
    return (int64_t)(uintptr_t)ptr;
#else
    if (al <= sizeof(void*) * 2) {
        return (int64_t)(uintptr_t)calloc(1, sz);
    }
    void* out = NULL;
    if (posix_memalign(&out, al, sz) != 0) return 0;
    memset(out, 0, sz);
    return (int64_t)(uintptr_t)out;
#endif
}

KM_ALWAYS_INLINE int64_t kmRealloc(int64_t ptr, int64_t oldSize, int64_t align, int64_t newSize) {
    if (ptr == 0) return kmAlloc(newSize, align);
    size_t nsz = newSize <= 0 ? 1 : (size_t)newSize;
    size_t al = align <= 1 ? 1 : (size_t)align;
#if defined(_WIN32)
    (void)oldSize;
    return (int64_t)(uintptr_t)_aligned_realloc((void*)(uintptr_t)ptr, nsz, al < sizeof(void*) ? sizeof(void*) : al);
#else
    if (al <= sizeof(void*) * 2) {
        return (int64_t)(uintptr_t)realloc((void*)(uintptr_t)ptr, nsz);
    }
    int64_t fresh = kmAlloc(newSize, align);
    if (fresh != 0) {
        size_t copyLen = (size_t)oldSize < nsz ? (size_t)oldSize : nsz;
        if (copyLen > 0) memcpy((void*)(uintptr_t)fresh, (const void*)(uintptr_t)ptr, copyLen);
        free((void*)(uintptr_t)ptr);
    }
    return fresh;
#endif
}

KM_ALWAYS_INLINE void kmDealloc(int64_t ptr, int64_t size, int64_t align) {
    (void)size;
    (void)align;
    if (ptr == 0) return;
#if defined(_WIN32)
    _aligned_free((void*)(uintptr_t)ptr);
#else
    free((void*)(uintptr_t)ptr);
#endif
}

/* C11 / Clang-GCC __atomic_* operations with explicit Rust ordering constants:
 * 0 = Relaxed, 1 = Release, 2 = Acquire, 3 = AcqRel, 4 = SeqCst
 */
KM_ALWAYS_INLINE void kmAtomicFence(int32_t ord) {
#if defined(__clang__) || defined(__GNUC__)
    switch (ord) {
        case 0: __atomic_thread_fence(__ATOMIC_RELAXED); break;
        case 1: __atomic_thread_fence(__ATOMIC_RELEASE); break;
        case 2: __atomic_thread_fence(__ATOMIC_ACQUIRE); break;
        case 3: __atomic_thread_fence(__ATOMIC_ACQ_REL); break;
        default: __atomic_thread_fence(__ATOMIC_SEQ_CST); break;
    }
#elif !defined(__cplusplus)
    switch (ord) {
        case 0: atomic_thread_fence(memory_order_relaxed); break;
        case 1: atomic_thread_fence(memory_order_release); break;
        case 2: atomic_thread_fence(memory_order_acquire); break;
        case 3: atomic_thread_fence(memory_order_acq_rel); break;
        default: atomic_thread_fence(memory_order_seq_cst); break;
    }
#endif
}

KM_ALWAYS_INLINE void kmAtomicSignalFence(int32_t ord) {
#if defined(__clang__) || defined(__GNUC__)
    switch (ord) {
        case 0: __atomic_signal_fence(__ATOMIC_RELAXED); break;
        case 1: __atomic_signal_fence(__ATOMIC_RELEASE); break;
        case 2: __atomic_signal_fence(__ATOMIC_ACQUIRE); break;
        case 3: __atomic_signal_fence(__ATOMIC_ACQ_REL); break;
        default: __atomic_signal_fence(__ATOMIC_SEQ_CST); break;
    }
#elif !defined(__cplusplus)
    switch (ord) {
        case 0: atomic_signal_fence(memory_order_relaxed); break;
        case 1: atomic_signal_fence(memory_order_release); break;
        case 2: atomic_signal_fence(memory_order_acquire); break;
        case 3: atomic_signal_fence(memory_order_acq_rel); break;
        default: atomic_signal_fence(memory_order_seq_cst); break;
    }
#endif
}

#if defined(__cplusplus) && defined(_MSC_VER) && !defined(__clang__)
#define KM_DEF_ATOMIC_LOAD_STORE(Suffix, CType, SignedCType) \
KM_ALWAYS_INLINE CType kmAtomicLoad##Suffix(int64_t addr, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    std::atomic_ref<CType> ref(*p); \
    switch (ord) { \
        case 0: return ref.load(std::memory_order_relaxed); \
        case 2: return ref.load(std::memory_order_acquire); \
        default: return ref.load(std::memory_order_seq_cst); \
    } \
} \
KM_ALWAYS_INLINE void kmAtomicStore##Suffix(int64_t addr, CType val, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    std::atomic_ref<CType> ref(*p); \
    switch (ord) { \
        case 0: ref.store(val, std::memory_order_relaxed); break; \
        case 1: ref.store(val, std::memory_order_release); break; \
        default: ref.store(val, std::memory_order_seq_cst); break; \
    } \
} \
KM_ALWAYS_INLINE CType kmAtomicXchg##Suffix(int64_t addr, CType val, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    std::atomic_ref<CType> ref(*p); \
    switch (ord) { \
        case 0: return ref.exchange(val, std::memory_order_relaxed); \
        case 1: return ref.exchange(val, std::memory_order_release); \
        case 2: return ref.exchange(val, std::memory_order_acquire); \
        case 3: return ref.exchange(val, std::memory_order_acq_rel); \
        default: return ref.exchange(val, std::memory_order_seq_cst); \
    } \
} \
KM_ALWAYS_INLINE CType kmAtomicCxchg##Suffix(int64_t addr, CType expected, CType desired, int32_t succ, int32_t fail, uint8_t* okOut) { \
    CType* p = (CType*)(uintptr_t)addr; \
    std::atomic_ref<CType> ref(*p); \
    std::memory_order sm = succ == 0 ? std::memory_order_relaxed : (succ == 1 ? std::memory_order_release : (succ == 2 ? std::memory_order_acquire : (succ == 3 ? std::memory_order_acq_rel : std::memory_order_seq_cst))); \
    std::memory_order fm = fail == 0 ? std::memory_order_relaxed : (fail == 2 ? std::memory_order_acquire : std::memory_order_seq_cst); \
    uint8_t ok = ref.compare_exchange_strong(expected, desired, sm, fm) ? 1 : 0; \
    if (okOut != NULL) *okOut = ok; \
    return expected; \
} \
KM_ALWAYS_INLINE CType kmAtomicRmw##Suffix(int64_t addr, CType val, int32_t op, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    std::atomic_ref<CType> ref(*p); \
    std::memory_order model = ord == 0 ? std::memory_order_relaxed : (ord == 1 ? std::memory_order_release : (ord == 2 ? std::memory_order_acquire : (ord == 3 ? std::memory_order_acq_rel : std::memory_order_seq_cst))); \
    switch (op) { \
        case 1: return ref.fetch_add(val, model); \
        case 2: return ref.fetch_sub(val, model); \
        case 3: return ref.fetch_and(val, model); \
        case 5: return ref.fetch_or(val, model); \
        case 6: return ref.fetch_xor(val, model); \
        default: { \
            CType cur = ref.load(std::memory_order_relaxed); \
            for (;;) { \
                CType next = cur; \
                if (op == 4) next = (CType)(~(cur & val)); \
                else if (op == 7) next = ((SignedCType)cur > (SignedCType)val) ? cur : val; \
                else if (op == 8) next = ((SignedCType)cur < (SignedCType)val) ? cur : val; \
                else if (op == 9) next = (cur > val) ? cur : val; \
                else next = (cur < val) ? cur : val; \
                if (ref.compare_exchange_weak(cur, next, model, std::memory_order_relaxed)) return cur; \
            } \
        } \
    } \
}
#else
#define KM_DEF_ATOMIC_LOAD_STORE(Suffix, CType, SignedCType) \
KM_ALWAYS_INLINE CType kmAtomicLoad##Suffix(int64_t addr, int32_t ord) { \
    const CType* p = (const CType*)(uintptr_t)addr; \
    switch (ord) { \
        case 0: return __atomic_load_n(p, __ATOMIC_RELAXED); \
        case 2: return __atomic_load_n(p, __ATOMIC_ACQUIRE); \
        default: return __atomic_load_n(p, __ATOMIC_SEQ_CST); \
    } \
} \
KM_ALWAYS_INLINE void kmAtomicStore##Suffix(int64_t addr, CType val, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    switch (ord) { \
        case 0: __atomic_store_n(p, val, __ATOMIC_RELAXED); break; \
        case 1: __atomic_store_n(p, val, __ATOMIC_RELEASE); break; \
        default: __atomic_store_n(p, val, __ATOMIC_SEQ_CST); break; \
    } \
} \
KM_ALWAYS_INLINE CType kmAtomicXchg##Suffix(int64_t addr, CType val, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    switch (ord) { \
        case 0: return __atomic_exchange_n(p, val, __ATOMIC_RELAXED); \
        case 1: return __atomic_exchange_n(p, val, __ATOMIC_RELEASE); \
        case 2: return __atomic_exchange_n(p, val, __ATOMIC_ACQUIRE); \
        case 3: return __atomic_exchange_n(p, val, __ATOMIC_ACQ_REL); \
        default: return __atomic_exchange_n(p, val, __ATOMIC_SEQ_CST); \
    } \
} \
KM_ALWAYS_INLINE CType kmAtomicCxchg##Suffix(int64_t addr, CType expected, CType desired, int32_t succ, int32_t fail, uint8_t* okOut) { \
    CType* p = (CType*)(uintptr_t)addr; \
    int succModel = succ == 0 ? __ATOMIC_RELAXED : (succ == 1 ? __ATOMIC_RELEASE : (succ == 2 ? __ATOMIC_ACQUIRE : (succ == 3 ? __ATOMIC_ACQ_REL : __ATOMIC_SEQ_CST))); \
    int failModel = fail == 0 ? __ATOMIC_RELAXED : (fail == 2 ? __ATOMIC_ACQUIRE : __ATOMIC_SEQ_CST); \
    if (failModel > succModel) failModel = __ATOMIC_RELAXED; \
    uint8_t ok = __atomic_compare_exchange_n(p, &expected, desired, 0, succModel, failModel) ? 1 : 0; \
    if (okOut != NULL) *okOut = ok; \
    return expected; \
} \
KM_ALWAYS_INLINE CType kmAtomicRmw##Suffix(int64_t addr, CType val, int32_t op, int32_t ord) { \
    CType* p = (CType*)(uintptr_t)addr; \
    int model = ord == 0 ? __ATOMIC_RELAXED : (ord == 1 ? __ATOMIC_RELEASE : (ord == 2 ? __ATOMIC_ACQUIRE : (ord == 3 ? __ATOMIC_ACQ_REL : __ATOMIC_SEQ_CST))); \
    switch (op) { \
        case 1: return __atomic_fetch_add(p, val, model); \
        case 2: return __atomic_fetch_sub(p, val, model); \
        case 3: return __atomic_fetch_and(p, val, model); \
        case 4: return __atomic_fetch_nand(p, val, model); \
        case 5: return __atomic_fetch_or(p, val, model); \
        case 6: return __atomic_fetch_xor(p, val, model); \
        default: { \
            CType cur = __atomic_load_n(p, __ATOMIC_RELAXED); \
            for (;;) { \
                CType next = cur; \
                if (op == 7) next = ((SignedCType)cur > (SignedCType)val) ? cur : val; \
                else if (op == 8) next = ((SignedCType)cur < (SignedCType)val) ? cur : val; \
                else if (op == 9) next = (cur > val) ? cur : val; \
                else next = (cur < val) ? cur : val; \
                if (__atomic_compare_exchange_n(p, &cur, next, 1, model, __ATOMIC_RELAXED)) return cur; \
            } \
        } \
    } \
}
#endif

KM_DEF_ATOMIC_LOAD_STORE(U8, uint8_t, int8_t)
KM_DEF_ATOMIC_LOAD_STORE(U16, uint16_t, int16_t)
KM_DEF_ATOMIC_LOAD_STORE(U32, uint32_t, int32_t)
KM_DEF_ATOMIC_LOAD_STORE(U64, uint64_t, int64_t)

/* Bit, math, and trap compiler builtins */
KM_ALWAYS_INLINE uint32_t kmCtpopU64(uint64_t v) {
#if defined(__clang__) || defined(__GNUC__)
    return (uint32_t)__builtin_popcountll(v);
#else
    uint32_t c = 0;
    while (v) { v &= (v - 1); c++; }
    return c;
#endif
}

KM_ALWAYS_INLINE uint32_t kmCtlzU64(uint64_t v) {
    if (v == 0) return 64;
#if defined(__clang__) || defined(__GNUC__)
    return (uint32_t)__builtin_clzll(v);
#else
    uint32_t n = 0;
    for (int i = 63; i >= 0 && ((v >> i) & 1ULL) == 0; i--) n++;
    return n;
#endif
}

KM_ALWAYS_INLINE uint32_t kmCttzU64(uint64_t v) {
    if (v == 0) return 64;
#if defined(__clang__) || defined(__GNUC__)
    return (uint32_t)__builtin_ctzll(v);
#else
    uint32_t n = 0;
    while ((v & 1ULL) == 0) { v >>= 1; n++; }
    return n;
#endif
}

KM_ALWAYS_INLINE uint16_t kmBswapU16(uint16_t v) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_bswap16(v);
#else
    return (uint16_t)((v << 8) | (v >> 8));
#endif
}

KM_ALWAYS_INLINE uint32_t kmBswapU32(uint32_t v) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_bswap32(v);
#else
    return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
#endif
}

KM_ALWAYS_INLINE uint64_t kmBswapU64(uint64_t v) {
#if defined(__clang__) || defined(__GNUC__)
    return __builtin_bswap64(v);
#else
    return ((uint64_t)kmBswapU32((uint32_t)v) << 32) | (uint64_t)kmBswapU32((uint32_t)(v >> 32));
#endif
}

KM_ALWAYS_INLINE uint64_t kmBitreverseU64(uint64_t v) {
#if defined(__clang__)
    return __builtin_bitreverse64(v);
#else
    v = ((v >> 1) & 0x5555555555555555ULL) | ((v & 0x5555555555555555ULL) << 1);
    v = ((v >> 2) & 0x3333333333333333ULL) | ((v & 0x3333333333333333ULL) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((v & 0x0F0F0F0F0F0F0F0FULL) << 4);
    return kmBswapU64(v);
#endif
}

KM_ALWAYS_INLINE float kmFmaf32(float a, float b, float c) { return fmaf(a, b, c); }
KM_ALWAYS_INLINE double kmFmaf64(double a, double b, double c) { return fma(a, b, c); }
KM_ALWAYS_INLINE float kmSqrtf32(float x) { return sqrtf(x); }
KM_ALWAYS_INLINE double kmSqrtf64(double x) { return sqrt(x); }

KM_ALWAYS_INLINE void kmAbort(void) {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_trap();
#else
    abort();
#endif
}

/* Architecture-guarded SIMD f32x4 / f64x2 fused kernels for core::arch */
#if defined(__aarch64__)
#include <arm_neon.h>
KM_ALWAYS_INLINE void kmSimdFmaF32x4(int64_t dstAddr, int64_t aAddr, int64_t bAddr, int64_t cAddr) {
    float32x4_t a = vld1q_f32((const float*)(uintptr_t)aAddr);
    float32x4_t b = vld1q_f32((const float*)(uintptr_t)bAddr);
    float32x4_t c = vld1q_f32((const float*)(uintptr_t)cAddr);
    vst1q_f32((float*)(uintptr_t)dstAddr, vfmaq_f32(c, a, b));
}
KM_ALWAYS_INLINE float kmSimdDotF32(int64_t aAddr, int64_t bAddr, int64_t len) {
    const float* a = (const float*)(uintptr_t)aAddr;
    const float* b = (const float*)(uintptr_t)bAddr;
    float32x4_t acc = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float sum = vaddvq_f32(acc);
    for (; i < len; i++) sum = fmaf(a[i], b[i], sum);
    return sum;
}
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
KM_ALWAYS_INLINE void kmSimdFmaF32x4(int64_t dstAddr, int64_t aAddr, int64_t bAddr, int64_t cAddr) {
    __m128 a = _mm_loadu_ps((const float*)(uintptr_t)aAddr);
    __m128 b = _mm_loadu_ps((const float*)(uintptr_t)bAddr);
    __m128 c = _mm_loadu_ps((const float*)(uintptr_t)cAddr);
    _mm_storeu_ps((float*)(uintptr_t)dstAddr, _mm_add_ps(_mm_mul_ps(a, b), c));
}
KM_ALWAYS_INLINE float kmSimdDotF32(int64_t aAddr, int64_t bAddr, int64_t len) {
    const float* a = (const float*)(uintptr_t)aAddr;
    const float* b = (const float*)(uintptr_t)bAddr;
    __m128 acc = _mm_setzero_ps();
    int64_t i = 0;
    for (; i + 4 <= len; i += 4) {
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    }
    float lanes[4];
    _mm_storeu_ps(lanes, acc);
    float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < len; i++) sum = fmaf(a[i], b[i], sum);
    return sum;
}
#else
KM_ALWAYS_INLINE void kmSimdFmaF32x4(int64_t dstAddr, int64_t aAddr, int64_t bAddr, int64_t cAddr) {
    const float* a = (const float*)(uintptr_t)aAddr;
    const float* b = (const float*)(uintptr_t)bAddr;
    const float* c = (const float*)(uintptr_t)cAddr;
    float* dst = (float*)(uintptr_t)dstAddr;
    for (int i = 0; i < 4; i++) dst[i] = fmaf(a[i], b[i], c[i]);
}
KM_ALWAYS_INLINE float kmSimdDotF32(int64_t aAddr, int64_t bAddr, int64_t len) {
    const float* a = (const float*)(uintptr_t)aAddr;
    const float* b = (const float*)(uintptr_t)bAddr;
    float sum = 0.0f;
    for (int64_t i = 0; i < len; i++) sum = fmaf(a[i], b[i], sum);
    return sum;
}
#endif

#endif /* KM_CORE_INTRINSICS_INLINE_H */
