// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.internal.CandleNative

internal actual val vdupqNF32: (Float) -> FloatArray = { x -> CandleNative.neonVdupqNF32(x) }
internal actual val vld1qF32: (FloatArray, Int) -> FloatArray = { mem, off -> CandleNative.neonVld1qF32(mem, off) }
internal actual val vaddqF32: (FloatArray, FloatArray) -> FloatArray = { a, b -> CandleNative.neonVaddqF32(a, b) }
internal actual val vfmaqF32: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleNative.neonVfmaqF32(a, b, c)
}
internal actual val vst1qF32: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleNative.neonVst1qF32(mem, off, a)
}
internal actual val vaddvqF32: (FloatArray) -> Float = { x -> CandleNative.neonReduceOneF32(x) }
internal actual val vgetqLaneF32: (FloatArray, Int) -> Float = { x, lane -> x[lane] }
internal actual val fcvtl4s4h: (ShortArray) -> FloatArray = { bits -> CandleNative.neonF16LoadFcvtl(bits, 0) }
internal actual val fcvtn4h4s: (FloatArray) -> ShortArray = { a ->
    val out = ShortArray(4)
    CandleNative.neonF16StoreFcvtn(out, 0, a)
    out
}
internal actual val asmMovi8hZero: () -> ShortArray = { CandleNative.neonF16Zero() }
internal actual val asmFadd8h: (ShortArray, ShortArray) -> ShortArray = { a, b -> CandleNative.neonF16VecAdd(a, b) }
internal actual val asmFmla8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    CandleNative.neonF16VecFma(a, b, c)
}
internal actual val asmDup8hFromF32: (Float) -> ShortArray = { v -> CandleNative.neonF16FromF32(v) }
internal actual val asmReduceF16x8: (ShortArray) -> Float = { x -> CandleNative.neonF16ReduceOne(x) }
internal actual val asmFadd4sOnU16x8: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    CandleNative.neonBf16VecAdd(a, b)
}
internal actual val asmBfdot4s8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    CandleNative.neonBf16VecFma(a, b, c)
}
internal actual val mm256SetzeroPs: () -> FloatArray = { CandleNative.avxZero() }
internal actual val mm256Set1Ps: (Float) -> FloatArray = { v -> CandleNative.avxSet1Ps(v) }
internal actual val mm256LoaduPs: (FloatArray, Int) -> FloatArray = { mem, off ->
    CandleNative.avxLoaduPs(mem, off)
}
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> CandleNative.avxAddPs(a, b) }
internal actual val mm256FmaPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleNative.avxFmaPs(a, b, c)
}
internal actual val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleNative.avxStoreuPs(mem, off, a)
}
internal actual val mm256ReduceOnePs: (FloatArray) -> Float = { x -> CandleNative.avxReduceOnePs(x) }
internal actual val mm256CvtphPs: (ShortArray, Int) -> FloatArray = { a, off -> CandleNative.avxCvtphPs(a, off) }
internal actual val mm256CvtpsPh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleNative.avxCvtpsPh(a, mem, off)
}
internal actual val mm256CvtpbhPs: (ShortArray, Int) -> FloatArray = { a, off -> CandleNative.avxCvtpbhPs(a, off) }
internal actual val mm256CvtnepsPbh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleNative.avxCvtnepsPbh(a, mem, off)
}

internal actual val f32x4Splat: (Float) -> FloatArray = { v -> CandleNative.simd128SplatF32(v) }
internal actual val v128Load: (FloatArray, Int) -> FloatArray = { mem, off ->
    CandleNative.simd128LoadF32(mem, off)
}
internal actual val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    CandleNative.simd128AddF32(a, b)
}
internal actual val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    CandleNative.simd128MulF32(a, b)
}
internal actual val f32x4Fma: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleNative.simd128FmaF32(a, b, c)
}
internal actual val v128Store: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleNative.simd128StoreF32(mem, off, a)
}
internal actual val f32x4ReduceOne: (FloatArray) -> Float = { x -> CandleNative.simd128ReduceOneF32(x) }

internal actual val accelSgemm: (
    Byte,
    Byte,
    Int,
    Int,
    Int,
    Float,
    FloatArray,
    Int,
    FloatArray,
    Int,
    Float,
    FloatArray,
    Int,
) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
    CandleNative.accelerateSgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
internal actual val accelDgemm: (
    Byte,
    Byte,
    Int,
    Int,
    Int,
    Double,
    DoubleArray,
    Int,
    DoubleArray,
    Int,
    Double,
    DoubleArray,
    Int,
) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
    CandleNative.accelerateDgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
internal actual val accelVvexpf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvexpf(dst, src, len)
}
internal actual val accelVvexp: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvexp(dst, src, len)
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvsqrtf(dst, src, len)
}
internal actual val accelVvsqrt: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvsqrt(dst, src, len)
}
internal actual val accelVvsinf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvsinf(dst, src, len)
}
internal actual val accelVvsin: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvsin(dst, src, len)
}
internal actual val accelVvcosf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvcosf(dst, src, len)
}
internal actual val accelVvcos: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvcos(dst, src, len)
}
internal actual val accelVvlogf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvlogf(dst, src, len)
}
internal actual val accelVvlog: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvlog(dst, src, len)
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvtanhf(dst, src, len)
}
internal actual val accelVvtanh: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleNative.accelerateVvtanh(dst, src, len)
}
internal actual val accelVdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVadd(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVaddD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVsub(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVsubD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVmul(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVmulD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVdiv(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVdivD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVmin(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVminD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVmax(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
internal actual val accelVdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleNative.accelerateVdspVmaxD(a, ia.toInt(), b, ib.toInt(), c, ic.toInt(), n.toInt())
}
