@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.half.F16
import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private object MklJniBridge {
    init {
        System.loadLibrary("candle_jni")
    }

    @JvmStatic
    external fun candleMklVsTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklSgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleMklDgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleMklHgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)
}

public actual fun vsTanh(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsTanh(n, a.rawAddress, y.rawAddress)
}

public actual fun vdTanh(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdTanh(n, a.rawAddress, y.rawAddress)
}

public actual fun vsExp(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsExp(n, a.rawAddress, y.rawAddress)
}

public actual fun vdExp(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdExp(n, a.rawAddress, y.rawAddress)
}

public actual fun vsLn(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsLn(n, a.rawAddress, y.rawAddress)
}

public actual fun vdLn(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdLn(n, a.rawAddress, y.rawAddress)
}

public actual fun vsSin(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsSin(n, a.rawAddress, y.rawAddress)
}

public actual fun vdSin(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdSin(n, a.rawAddress, y.rawAddress)
}

public actual fun vsCos(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsCos(n, a.rawAddress, y.rawAddress)
}

public actual fun vdCos(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdCos(n, a.rawAddress, y.rawAddress)
}

public actual fun vsSqrt(n: CInt, a: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsSqrt(n, a.rawAddress, y.rawAddress)
}

public actual fun vdSqrt(n: CInt, a: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdSqrt(n, a.rawAddress, y.rawAddress)
}

public actual fun vsAdd(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsAdd(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdAdd(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdAdd(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsSub(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsSub(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdSub(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdSub(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsMul(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsMul(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdMul(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdMul(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsDiv(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsDiv(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdDiv(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdDiv(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsFmax(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsFmax(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdFmax(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdFmax(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsFmin(n: CInt, a: ConstPtr<CFloat>, b: ConstPtr<CFloat>, y: MutPtr<CFloat>) {
    MklJniBridge.candleMklVsFmin(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdFmin(n: CInt, a: ConstPtr<CDouble>, b: ConstPtr<CDouble>, y: MutPtr<CDouble>) {
    MklJniBridge.candleMklVdFmin(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun sgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    MklJniBridge.candleMklSgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun dgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    MklJniBridge.candleMklDgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun hgemm(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<F16>, a: ConstPtr<F16>, lda: ConstPtr<CInt>, b: ConstPtr<F16>, ldb: ConstPtr<CInt>, beta: ConstPtr<F16>, c: MutPtr<F16>, ldc: ConstPtr<CInt>) {
    MklJniBridge.candleMklHgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}
