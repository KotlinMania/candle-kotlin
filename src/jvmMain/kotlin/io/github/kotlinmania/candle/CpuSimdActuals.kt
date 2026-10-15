// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.internal.CandleJni
import io.github.kotlinmania.libc.CInt

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
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    val out = FloatArray(8)
    CandleJni.avxAddPs(a, b, out)
    out
}
internal actual val mm256CvtphPs: (ShortArray) -> FloatArray = { a ->
    val out = FloatArray(8)
    CandleJni.avxCvtphPs(a, 0, out)
    out
}
internal actual val accelVvexpf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvexpf(dst, src, len)
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvsqrtf(dst, src, len)
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleJni.accelerateVvtanhf(dst, src, len)
}
