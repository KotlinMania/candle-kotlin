// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.internal.CandleCInterop

internal actual val vdupqNF32: (Float) -> FloatArray = { x -> CandleCInterop.neonVdupqNF32(x) }
internal actual val vld1qF32: (FloatArray, Int) -> FloatArray = { mem, off -> CandleCInterop.neonVld1qF32(mem, off) }
internal actual val vaddqF32: (FloatArray, FloatArray) -> FloatArray = { a, b -> CandleCInterop.neonVaddqF32(a, b) }
internal actual val vfmaqF32: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleCInterop.neonVfmaqF32(a, b, c)
}
internal actual val vst1qF32: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleCInterop.neonVst1qF32(mem, off, a)
}
internal actual val vaddvqF32: (FloatArray) -> Float = { x -> CandleCInterop.neonReduceOneF32(x) }
internal actual val vgetqLaneF32: (FloatArray, Int) -> Float = { x, lane -> x[lane] }
internal actual val fcvtl4s4h: (ShortArray) -> FloatArray = { bits -> CandleCInterop.neonF16LoadFcvtl(bits, 0) }
internal actual val fcvtn4h4s: (FloatArray) -> ShortArray = { a ->
    val out = ShortArray(4)
    CandleCInterop.neonF16StoreFcvtn(out, 0, a)
    out
}
internal actual val asmMovi8hZero: () -> ShortArray = { CandleCInterop.neonF16Zero() }
internal actual val asmFadd8h: (ShortArray, ShortArray) -> ShortArray = { a, b -> CandleCInterop.neonF16VecAdd(a, b) }
internal actual val asmFmla8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    CandleCInterop.neonF16VecFma(a, b, c)
}
internal actual val asmDup8hFromF32: (Float) -> ShortArray = { v -> CandleCInterop.neonF16FromF32(v) }
internal actual val asmReduceF16x8: (ShortArray) -> Float = { x -> CandleCInterop.neonF16ReduceOne(x) }
internal actual val asmFadd4sOnU16x8: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    CandleCInterop.neonBf16VecAdd(a, b)
}
internal actual val asmBfdot4s8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    CandleCInterop.neonBf16VecFma(a, b, c)
}
internal actual val mm256SetzeroPs: () -> FloatArray = { CandleCInterop.avxZero() }
internal actual val mm256Set1Ps: (Float) -> FloatArray = { v -> CandleCInterop.avxSet1Ps(v) }
internal actual val mm256LoaduPs: (FloatArray, Int) -> FloatArray = { mem, off ->
    CandleCInterop.avxLoaduPs(mem, off)
}
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> CandleCInterop.avxAddPs(a, b) }
internal actual val mm256FmaPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleCInterop.avxFmaPs(a, b, c)
}
internal actual val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleCInterop.avxStoreuPs(mem, off, a)
}
internal actual val mm256ReduceOnePs: (FloatArray) -> Float = { x -> CandleCInterop.avxReduceOnePs(x) }
internal actual val mm256CvtphPs: (ShortArray, Int) -> FloatArray = { a, off -> CandleCInterop.avxCvtphPs(a, off) }
internal actual val mm256CvtpsPh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleCInterop.avxCvtpsPh(a, mem, off)
}
internal actual val mm256CvtpbhPs: (ShortArray, Int) -> FloatArray = { a, off -> CandleCInterop.avxCvtpbhPs(a, off) }
internal actual val mm256CvtnepsPbh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleCInterop.avxCvtnepsPbh(a, mem, off)
}

internal actual val f32x4Splat: (Float) -> FloatArray = { v -> CandleCInterop.simd128SplatF32(v) }
internal actual val v128Load: (FloatArray, Int) -> FloatArray = { mem, off ->
    CandleCInterop.simd128LoadF32(mem, off)
}
internal actual val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    CandleCInterop.simd128AddF32(a, b)
}
internal actual val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    CandleCInterop.simd128MulF32(a, b)
}
internal actual val f32x4Fma: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    CandleCInterop.simd128FmaF32(a, b, c)
}
internal actual val v128Store: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleCInterop.simd128StoreF32(mem, off, a)
}
internal actual val f32x4ReduceOne: (FloatArray) -> Float = { x -> CandleCInterop.simd128ReduceOneF32(x) }

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
    CandleCInterop.accelerateSgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
    CandleCInterop.accelerateDgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
internal actual val accelVvexpf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvexpf(dst, src, len)
}
internal actual val accelVvexp: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvexp(dst, src, len)
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvsqrtf(dst, src, len)
}
internal actual val accelVvsqrt: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvsqrt(dst, src, len)
}
internal actual val accelVvsinf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvsinf(dst, src, len)
}
internal actual val accelVvsin: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvsin(dst, src, len)
}
internal actual val accelVvcosf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvcosf(dst, src, len)
}
internal actual val accelVvcos: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvcos(dst, src, len)
}
internal actual val accelVvlogf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvlogf(dst, src, len)
}
internal actual val accelVvlog: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvlog(dst, src, len)
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvtanhf(dst, src, len)
}
internal actual val accelVvtanh: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvtanh(dst, src, len)
}
internal actual val accelVdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVadd(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVaddD(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVsub(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVsubD(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVmul(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVmulD(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVdiv(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVdivD(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVmin(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVminD(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVmax(a, ia, b, ib, c, ic, n)
}
internal actual val accelVdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleCInterop.accelerateVdspVmaxD(a, ia, b, ib, c, ic, n)
}
