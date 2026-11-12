@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class, kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.candle.mkl.cinterop.candleMklDgemm
import io.github.kotlinmania.candle.mkl.cinterop.candleMklHgemm
import io.github.kotlinmania.candle.mkl.cinterop.candleMklSgemm
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdAdd
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdCos
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdDiv
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdExp
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdFmax
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdFmin
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdLn
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdMul
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdSin
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdSqrt
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdSub
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVdTanh
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsAdd
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsCos
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsDiv
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsExp
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsFmax
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsFmin
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsLn
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsMul
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsSin
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsSqrt
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsSub
import io.github.kotlinmania.candle.mkl.cinterop.candleMklVsTanh
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun vsTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsTanh(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdTanh(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsExp(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsExp(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdExp(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdExp(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsLn(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsLn(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdLn(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdLn(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSin(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsSin(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSin(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdSin(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsCos(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsCos(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdCos(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdCos(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVsSqrt(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    candleMklVdSqrt(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsAdd(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdAdd(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsSub(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdSub(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsMul(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdMul(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsDiv(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdDiv(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsFmax(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdFmax(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVsFmin(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    candleMklVdFmin(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun sgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    candleMklSgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun dgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    candleMklDgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun hgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    candleMklHgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}
