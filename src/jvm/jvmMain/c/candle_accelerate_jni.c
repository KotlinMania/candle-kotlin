#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <jni.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#ifndef RTLD_DEFAULT
#define RTLD_DEFAULT ((void*)0)
#endif
#endif

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1sgemmFfi(JNIEnv* km_env, jclass km_clazz, jstring transa, jstring transb, jlong m, jlong n, jlong k, jlong alpha, jlong a, jlong lda, jlong b, jlong ldb, jlong beta, jlong c, jlong ldc) {
    (void)km_env;
    (void)km_clazz;
    const char* transa_utf = (transa != NULL) ? (*km_env)->GetStringUTFChars(km_env, transa, NULL) : NULL;
    const char* transb_utf = (transb != NULL) ? (*km_env)->GetStringUTFChars(km_env, transb, NULL) : NULL;
    typedef void (*km_fn_t)(const char*, const char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "sgemm_");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_sgemm_");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "sgemm_");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_sgemm_");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "sgemm_");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr(transa_utf, transb_utf, (void*)(uintptr_t)m, (void*)(uintptr_t)n, (void*)(uintptr_t)k, (void*)(uintptr_t)alpha, (void*)(uintptr_t)a, (void*)(uintptr_t)lda, (void*)(uintptr_t)b, (void*)(uintptr_t)ldb, (void*)(uintptr_t)beta, (void*)(uintptr_t)c, (void*)(uintptr_t)ldc);
    } else {
        errno = ENOSYS;
    }
    if (transa != NULL && transa_utf != NULL) (*km_env)->ReleaseStringUTFChars(km_env, transa, transa_utf);
    if (transb != NULL && transb_utf != NULL) (*km_env)->ReleaseStringUTFChars(km_env, transb, transb_utf);
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1dgemmFfi(JNIEnv* km_env, jclass km_clazz, jstring transa, jstring transb, jlong m, jlong n, jlong k, jlong alpha, jlong a, jlong lda, jlong b, jlong ldb, jlong beta, jlong c, jlong ldc) {
    (void)km_env;
    (void)km_clazz;
    const char* transa_utf = (transa != NULL) ? (*km_env)->GetStringUTFChars(km_env, transa, NULL) : NULL;
    const char* transb_utf = (transb != NULL) ? (*km_env)->GetStringUTFChars(km_env, transb, NULL) : NULL;
    typedef void (*km_fn_t)(const char*, const char*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "dgemm_");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_dgemm_");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "dgemm_");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_dgemm_");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "dgemm_");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr(transa_utf, transb_utf, (void*)(uintptr_t)m, (void*)(uintptr_t)n, (void*)(uintptr_t)k, (void*)(uintptr_t)alpha, (void*)(uintptr_t)a, (void*)(uintptr_t)lda, (void*)(uintptr_t)b, (void*)(uintptr_t)ldb, (void*)(uintptr_t)beta, (void*)(uintptr_t)c, (void*)(uintptr_t)ldc);
    } else {
        errno = ENOSYS;
    }
    if (transa != NULL && transa_utf != NULL) (*km_env)->ReleaseStringUTFChars(km_env, transa, transa_utf);
    if (transb != NULL && transb_utf != NULL) (*km_env)->ReleaseStringUTFChars(km_env, transb, transb_utf);
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvexpf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvexpf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvexpf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvexpf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvexpf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvexpf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvexp(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvexp");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvexp");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvexp");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvexp");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvexp");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvsqrtf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvsqrtf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvsqrtf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvsqrtf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvsqrtf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvsqrtf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvsqrt(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvsqrt");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvsqrt");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvsqrt");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvsqrt");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvsqrt");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvsinf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvsinf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvsinf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvsinf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvsinf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvsinf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvsin(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvsin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvsin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvsin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvsin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvsin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvcosf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvcosf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvcosf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvcosf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvcosf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvcosf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvcos(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvcos");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvcos");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvcos");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvcos");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvcos");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvlogf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvlogf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvlogf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvlogf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvlogf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvlogf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvlog(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvlog");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvlog");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvlog");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvlog");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvlog");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvtanhf(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvtanhf");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvtanhf");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvtanhf");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvtanhf");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvtanhf");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vvtanh(JNIEnv* km_env, jclass km_clazz, jlong dst, jlong src, jlong len) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vvtanh");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vvtanh");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vvtanh");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vvtanh");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vvtanh");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)dst, (void*)(uintptr_t)src, (void*)(uintptr_t)len);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVaddD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vaddD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vaddD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vaddD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vaddD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vaddD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVadd(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vadd");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vadd");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vadd");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vadd");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vadd");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVsubD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vsubD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vsubD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vsubD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vsubD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vsubD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVsub(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vsub");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vsub");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vsub");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vsub");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vsub");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVmulD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vmulD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vmulD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vmulD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vmulD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vmulD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVmul(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vmul");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vmul");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vmul");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vmul");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vmul");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVdivD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vdivD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vdivD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vdivD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vdivD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vdivD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVdiv(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vdiv");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vdiv");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vdiv");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vdiv");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vdiv");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVminD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vminD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vminD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vminD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vminD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vminD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVmin(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vmin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vmin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vmin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vmin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vmin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVmaxD(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vmaxD");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vmaxD");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vmaxD");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vmaxD");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vmaxD");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_accelerate_AccelerateJniBridge_candle_1accelerate_1vDspVmax(JNIEnv* km_env, jclass km_clazz, jlong arg0, jlong arg1, jlong arg2, jlong arg3, jlong arg4, jlong arg5, jlong arg6) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(void*, int64_t, void*, int64_t, void*, int64_t, uint64_t);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vDSP_vmax");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vDSP_vmax");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vDSP_vmax");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vDSP_vmax");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vDSP_vmax");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((void*)(uintptr_t)arg0, (int64_t)arg1, (void*)(uintptr_t)arg2, (int64_t)arg3, (void*)(uintptr_t)arg4, (int64_t)arg5, (uint64_t)arg6);
    } else {
        errno = ENOSYS;
    }
}

