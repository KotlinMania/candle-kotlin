@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CInt

private object MklJniBridge {
    init {
        System.loadLibrary("candle_jni")
    }

    @JvmStatic
    external fun candle_mkl_vsTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vsFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_vdFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candle_mkl_sgemm(transa: String?, transb: String?, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candle_mkl_dgemm(transa: String?, transb: String?, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candle_mkl_hgemm(transa: String?, transb: String?, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)
}

public actual fun vsTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsTanh(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdTanh(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsExp(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdExp(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsLn(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdLn(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsSin(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdSin(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsCos(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdCos(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsSqrt(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdSqrt(n, (a?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsAdd(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdAdd(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsSub(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdSub(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsMul(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdMul(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsDiv(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdDiv(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsFmax(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdFmax(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vsFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vsFmin(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun vdFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    MklJniBridge.candle_mkl_vdFmin(n, (a?.value ?: 0L), (b?.value ?: 0L), (y?.value ?: 0L))
}

public actual fun sgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    MklJniBridge.candle_mkl_sgemm(transa, transb, (m?.value ?: 0L), (n?.value ?: 0L), (k?.value ?: 0L), (alpha?.value ?: 0L), (a?.value ?: 0L), (lda?.value ?: 0L), (b?.value ?: 0L), (ldb?.value ?: 0L), (beta?.value ?: 0L), (c?.value ?: 0L), (ldc?.value ?: 0L))
}

public actual fun dgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    MklJniBridge.candle_mkl_dgemm(transa, transb, (m?.value ?: 0L), (n?.value ?: 0L), (k?.value ?: 0L), (alpha?.value ?: 0L), (a?.value ?: 0L), (lda?.value ?: 0L), (b?.value ?: 0L), (ldb?.value ?: 0L), (beta?.value ?: 0L), (c?.value ?: 0L), (ldc?.value ?: 0L))
}

public actual fun hgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    MklJniBridge.candle_mkl_hgemm(transa, transb, (m?.value ?: 0L), (n?.value ?: 0L), (k?.value ?: 0L), (alpha?.value ?: 0L), (a?.value ?: 0L), (lda?.value ?: 0L), (b?.value ?: 0L), (ldb?.value ?: 0L), (beta?.value ?: 0L), (c?.value ?: 0L), (ldc?.value ?: 0L))
}
