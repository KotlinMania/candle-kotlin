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

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsTanh(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsTanh");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsTanh");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsTanh");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsTanh");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsTanh");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdTanh(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdTanh");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdTanh");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdTanh");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdTanh");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdTanh");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsExp(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsExp");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsExp");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsExp");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsExp");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsExp");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdExp(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdExp");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdExp");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdExp");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdExp");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdExp");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsLn(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsLn");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsLn");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsLn");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsLn");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsLn");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdLn(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdLn");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdLn");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdLn");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdLn");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdLn");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsSin(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsSin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsSin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsSin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsSin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsSin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdSin(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdSin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdSin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdSin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdSin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdSin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsCos(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsCos");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsCos");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsCos");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsCos");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsCos");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdCos(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdCos");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdCos");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdCos");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdCos");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdCos");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsSqrt(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsSqrt");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsSqrt");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsSqrt");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsSqrt");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsSqrt");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdSqrt(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdSqrt");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdSqrt");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdSqrt");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdSqrt");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdSqrt");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsAdd(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsAdd");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsAdd");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsAdd");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsAdd");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsAdd");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdAdd(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdAdd");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdAdd");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdAdd");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdAdd");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdAdd");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsSub(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsSub");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsSub");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsSub");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsSub");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsSub");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdSub(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdSub");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdSub");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdSub");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdSub");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdSub");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsMul(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsMul");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsMul");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsMul");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsMul");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsMul");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdMul(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdMul");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdMul");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdMul");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdMul");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdMul");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsDiv(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsDiv");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsDiv");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsDiv");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsDiv");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsDiv");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdDiv(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdDiv");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdDiv");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdDiv");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdDiv");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdDiv");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsFmax(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsFmax");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsFmax");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsFmax");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsFmax");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsFmax");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdFmax(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdFmax");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdFmax");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdFmax");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdFmax");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdFmax");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vsFmin(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vsFmin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vsFmin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vsFmin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vsFmin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vsFmin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1vdFmin(JNIEnv* km_env, jclass km_clazz, jint n, jlong a, jlong b, jlong y) {
    (void)km_env;
    (void)km_clazz;
    typedef void (*km_fn_t)(int32_t, void*, void*, void*);
    static km_fn_t km_fn_ptr = NULL;
#ifdef _WIN32
    if (km_fn_ptr == NULL) {
        HMODULE km_h_ucrt = GetModuleHandleA("ucrtbase.dll");
        if (km_h_ucrt == NULL) km_h_ucrt = LoadLibraryA("ucrtbase.dll");
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "vdFmin");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_vdFmin");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "vdFmin");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_vdFmin");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "vdFmin");
    }
#endif
    if (km_fn_ptr != NULL) {
        km_fn_ptr((int32_t)n, (void*)(uintptr_t)a, (void*)(uintptr_t)b, (void*)(uintptr_t)y);
    } else {
        errno = ENOSYS;
    }
}

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1sgemm(JNIEnv* km_env, jclass km_clazz, jstring transa, jstring transb, jlong m, jlong n, jlong k, jlong alpha, jlong a, jlong lda, jlong b, jlong ldb, jlong beta, jlong c, jlong ldc) {
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

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1dgemm(JNIEnv* km_env, jclass km_clazz, jstring transa, jstring transb, jlong m, jlong n, jlong k, jlong alpha, jlong a, jlong lda, jlong b, jlong ldb, jlong beta, jlong c, jlong ldc) {
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

JNIEXPORT void JNICALL Java_io_github_kotlinmania_candle_mkl_MklJniBridge_candle_1mkl_1hgemm(JNIEnv* km_env, jclass km_clazz, jstring transa, jstring transb, jlong m, jlong n, jlong k, jlong alpha, jlong a, jlong lda, jlong b, jlong ldb, jlong beta, jlong c, jlong ldc) {
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
        if (km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "hgemm_");
        if (km_fn_ptr == NULL && km_h_ucrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_ucrt, "_hgemm_");
        if (km_fn_ptr == NULL) {
            HMODULE km_h_msvcrt = GetModuleHandleA("msvcrt.dll");
            if (km_h_msvcrt == NULL) km_h_msvcrt = LoadLibraryA("msvcrt.dll");
            if (km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "hgemm_");
            if (km_fn_ptr == NULL && km_h_msvcrt != NULL) km_fn_ptr = (km_fn_t)(void*)GetProcAddress(km_h_msvcrt, "_hgemm_");
        }
    }
#else
    if (km_fn_ptr == NULL) {
        km_fn_ptr = (km_fn_t)dlsym(RTLD_DEFAULT, "hgemm_");
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

