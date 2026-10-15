// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.internal.CandleCInterop
import io.github.kotlinmania.libc.CInt

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
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> CandleCInterop.avxAddPs(a, b) }
internal actual val mm256CvtphPs: (ShortArray) -> FloatArray = { a -> CandleCInterop.avxCvtphPs(a, 0) }
internal actual val accelVvexpf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvexpf(dst, src, len)
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvsqrtf(dst, src, len)
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    CandleCInterop.accelerateVvtanhf(dst, src, len)
}
