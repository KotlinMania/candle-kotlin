// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.internal.CandleJni

internal actual val vdupqNF32: (Float) -> FloatArray = { x ->
    val out = FloatArray(4)
    CandleJni.neonVdupqNF32(x, out)
    out
}
internal actual val vld1qF32: (FloatArray, Int) -> FloatArray = { mem, off ->
    val out = FloatArray(4)
    CandleJni.neonVld1qF32(mem, off, out)
    out
}
internal actual val vaddqF32: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    val out = FloatArray(4)
    CandleJni.neonVaddqF32(a, b, out)
    out
}
internal actual val vfmaqF32: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    val out = FloatArray(4)
    CandleJni.neonVfmaqF32(a, b, c, out)
    out
}
internal actual val vst1qF32: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleJni.neonVst1qF32(mem, off, a)
}
internal actual val vaddvqF32: (FloatArray) -> Float = { x -> CandleJni.neonReduceOneF32(x) }
internal actual val vgetqLaneF32: (FloatArray, Int) -> Float = { x, lane -> x[lane] }
internal actual val fcvtl4s4h: (ShortArray) -> FloatArray = { bits ->
    val out = FloatArray(4)
    CandleJni.neonF16LoadFcvtl(bits, 0, out)
    out
}
internal actual val fcvtn4h4s: (FloatArray) -> ShortArray = { a ->
    val out = ShortArray(4)
    CandleJni.neonF16StoreFcvtn(out, 0, a)
    out
}
internal actual val asmMovi8hZero: () -> ShortArray = {
    val out = ShortArray(8)
    CandleJni.neonF16Zero(out)
    out
}
internal actual val asmFadd8h: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    CandleJni.neonF16VecAdd(a, b, out)
    out
}
internal actual val asmFmla8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    val out = ShortArray(8)
    CandleJni.neonF16VecFma(a, b, c, out)
    out
}
internal actual val asmDup8hFromF32: (Float) -> ShortArray = { v ->
    val out = ShortArray(8)
    CandleJni.neonF16FromF32(v, out)
    out
}
internal actual val asmReduceF16x8: (ShortArray) -> Float = { x -> CandleJni.neonF16ReduceOne(x) }
internal actual val asmFadd4sOnU16x8: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    CandleJni.neonBf16VecAdd(a, b, out)
    out
}
internal actual val asmBfdot4s8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    val out = ShortArray(8)
    CandleJni.neonBf16VecFma(a, b, c, out)
    out
}
internal actual val mm256SetzeroPs: () -> FloatArray = {
    val out = FloatArray(8)
    CandleJni.avxZero(out)
    out
}
internal actual val mm256Set1Ps: (Float) -> FloatArray = { v ->
    val out = FloatArray(8)
    CandleJni.avxSet1Ps(v, out)
    out
}
internal actual val mm256LoaduPs: (FloatArray, Int) -> FloatArray = { mem, off ->
    val out = FloatArray(8)
    CandleJni.avxLoaduPs(mem, off, out)
    out
}
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    val out = FloatArray(8)
    CandleJni.avxAddPs(a, b, out)
    out
}
internal actual val mm256FmaPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    val out = FloatArray(8)
    CandleJni.avxFmaPs(a, b, c, out)
    out
}
internal actual val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleJni.avxStoreuPs(mem, off, a)
}
internal actual val mm256ReduceOnePs: (FloatArray) -> Float = { x -> CandleJni.avxReduceOnePs(x) }
internal actual val mm256CvtphPs: (ShortArray, Int) -> FloatArray = { a, off ->
    val out = FloatArray(8)
    CandleJni.avxCvtphPs(a, off, out)
    out
}
internal actual val mm256CvtpsPh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleJni.avxCvtpsPh(a, mem, off)
}
internal actual val mm256CvtpbhPs: (ShortArray, Int) -> FloatArray = { a, off ->
    val out = FloatArray(8)
    CandleJni.avxCvtpbhPs(a, off, out)
    out
}
internal actual val mm256CvtnepsPbh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    CandleJni.avxCvtnepsPbh(a, mem, off)
}

internal actual val f32x4Splat: (Float) -> FloatArray = { v ->
    val out = FloatArray(4)
    CandleJni.simd128SplatF32(v, out)
    out
}
internal actual val v128Load: (FloatArray, Int) -> FloatArray = { mem, off ->
    val out = FloatArray(4)
    CandleJni.simd128LoadF32(mem, off, out)
    out
}
internal actual val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    val out = FloatArray(4)
    CandleJni.simd128AddF32(a, b, out)
    out
}
internal actual val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    val out = FloatArray(4)
    CandleJni.simd128MulF32(a, b, out)
    out
}
internal actual val f32x4Fma: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    val out = FloatArray(4)
    CandleJni.simd128FmaF32(a, b, c, out)
    out
}
internal actual val v128Store: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    CandleJni.simd128StoreF32(mem, off, a)
}
internal actual val f32x4ReduceOne: (FloatArray) -> Float = { x -> CandleJni.simd128ReduceOneF32(x) }

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
    CandleJni.accelerateSgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
    CandleJni.accelerateDgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
internal actual val accelVvexpf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvexpf(dst, src, len)
}
internal actual val accelVvexp: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvexp(dst, src, len)
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvsqrtf(dst, src, len)
}
internal actual val accelVvsqrt: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvsqrt(dst, src, len)
}
internal actual val accelVvsinf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvsinf(dst, src, len)
}
internal actual val accelVvsin: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvsin(dst, src, len)
}
internal actual val accelVvcosf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvcosf(dst, src, len)
}
internal actual val accelVvcos: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvcos(dst, src, len)
}
internal actual val accelVvlogf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvlogf(dst, src, len)
}
internal actual val accelVvlog: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvlog(dst, src, len)
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvtanhf(dst, src, len)
}
internal actual val accelVvtanh: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvtanh(dst, src, len)
}
internal actual val accelVdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVadd(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVaddD(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVsub(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVsubD(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVmul(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVmulD(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVdiv(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVdivD(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVmin(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVminD(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVmax(a, ia, b, ib, c, ic, n.toLong())
}
internal actual val accelVdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    CandleJni.accelerateVdspVmaxD(a, ia, b, ib, c, ic, n.toLong())
}
