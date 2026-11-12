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
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.toCPointer

public actual fun vsTanh(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsTanh(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdTanh(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdTanh(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsExp(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsExp(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdExp(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdExp(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsLn(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsLn(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdLn(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdLn(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSin(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsSin(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSin(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdSin(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsCos(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsCos(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdCos(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdCos(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSqrt(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsSqrt(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSqrt(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdSqrt(n, a.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsAdd(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsAdd(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdAdd(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdAdd(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsSub(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsSub(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdSub(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdSub(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsMul(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsMul(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdMul(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdMul(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsDiv(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsDiv(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdDiv(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdDiv(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsFmax(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsFmax(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdFmax(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdFmax(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vsFmin(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    candleMklVsFmin(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun vdFmin(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    candleMklVdFmin(n, a.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), y.rawAddress.toCPointer<ByteVar>())
}

public actual fun sgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    candleMklSgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun dgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    candleMklDgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}

public actual fun hgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<F16>, a: ConstPtr<F16>, lda: ConstPtr<CInt>, b: ConstPtr<F16>, ldb: ConstPtr<CInt>, beta: ConstPtr<F16>, c: MutPtr<F16>, ldc: ConstPtr<CInt>) {
    candleMklHgemm(transa.rawAddress.toCPointer<ByteVar>(), transb.rawAddress.toCPointer<ByteVar>(), m.rawAddress.toCPointer<ByteVar>(), n.rawAddress.toCPointer<ByteVar>(), k.rawAddress.toCPointer<ByteVar>(), alpha.rawAddress.toCPointer<ByteVar>(), a.rawAddress.toCPointer<ByteVar>(), lda.rawAddress.toCPointer<ByteVar>(), b.rawAddress.toCPointer<ByteVar>(), ldb.rawAddress.toCPointer<ByteVar>(), beta.rawAddress.toCPointer<ByteVar>(), c.rawAddress.toCPointer<ByteVar>(), ldc.rawAddress.toCPointer<ByteVar>())
}
