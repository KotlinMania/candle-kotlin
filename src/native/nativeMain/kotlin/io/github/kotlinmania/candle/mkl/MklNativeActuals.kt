@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.candle.COpaquePointer
import candle.cinterop.candle_mkl_dgemm
import candle.cinterop.candle_mkl_hgemm
import candle.cinterop.candle_mkl_sgemm
import candle.cinterop.candle_mkl_vdAdd
import candle.cinterop.candle_mkl_vdCos
import candle.cinterop.candle_mkl_vdDiv
import candle.cinterop.candle_mkl_vdExp
import candle.cinterop.candle_mkl_vdFmax
import candle.cinterop.candle_mkl_vdFmin
import candle.cinterop.candle_mkl_vdLn
import candle.cinterop.candle_mkl_vdMul
import candle.cinterop.candle_mkl_vdSin
import candle.cinterop.candle_mkl_vdSqrt
import candle.cinterop.candle_mkl_vdSub
import candle.cinterop.candle_mkl_vdTanh
import candle.cinterop.candle_mkl_vsAdd
import candle.cinterop.candle_mkl_vsCos
import candle.cinterop.candle_mkl_vsDiv
import candle.cinterop.candle_mkl_vsExp
import candle.cinterop.candle_mkl_vsFmax
import candle.cinterop.candle_mkl_vsFmin
import candle.cinterop.candle_mkl_vsLn
import candle.cinterop.candle_mkl_vsMul
import candle.cinterop.candle_mkl_vsSin
import candle.cinterop.candle_mkl_vsSqrt
import candle.cinterop.candle_mkl_vsSub
import candle.cinterop.candle_mkl_vsTanh
import io.github.kotlinmania.libc.CInt
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun vsTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsTanh(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdTanh(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsExp(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdExp(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsLn(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdLn(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsSin(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdSin(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsCos(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdCos(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsSqrt(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdSqrt(n, a?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsAdd(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdAdd(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsSub(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdSub(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsMul(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdMul(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsDiv(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdDiv(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsFmax(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdFmax(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vsFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vsFmin(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun vdFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    candle_mkl_vdFmin(n, a?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), y?.value?.toCPointer<ByteVar>())
}

public actual fun sgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    candle_mkl_sgemm(transa, transb, m?.value?.toCPointer<ByteVar>(), n?.value?.toCPointer<ByteVar>(), k?.value?.toCPointer<ByteVar>(), alpha?.value?.toCPointer<ByteVar>(), a?.value?.toCPointer<ByteVar>(), lda?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), ldb?.value?.toCPointer<ByteVar>(), beta?.value?.toCPointer<ByteVar>(), c?.value?.toCPointer<ByteVar>(), ldc?.value?.toCPointer<ByteVar>())
}

public actual fun dgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    candle_mkl_dgemm(transa, transb, m?.value?.toCPointer<ByteVar>(), n?.value?.toCPointer<ByteVar>(), k?.value?.toCPointer<ByteVar>(), alpha?.value?.toCPointer<ByteVar>(), a?.value?.toCPointer<ByteVar>(), lda?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), ldb?.value?.toCPointer<ByteVar>(), beta?.value?.toCPointer<ByteVar>(), c?.value?.toCPointer<ByteVar>(), ldc?.value?.toCPointer<ByteVar>())
}

public actual fun hgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    candle_mkl_hgemm(transa, transb, m?.value?.toCPointer<ByteVar>(), n?.value?.toCPointer<ByteVar>(), k?.value?.toCPointer<ByteVar>(), alpha?.value?.toCPointer<ByteVar>(), a?.value?.toCPointer<ByteVar>(), lda?.value?.toCPointer<ByteVar>(), b?.value?.toCPointer<ByteVar>(), ldb?.value?.toCPointer<ByteVar>(), beta?.value?.toCPointer<ByteVar>(), c?.value?.toCPointer<ByteVar>(), ldc?.value?.toCPointer<ByteVar>())
}
