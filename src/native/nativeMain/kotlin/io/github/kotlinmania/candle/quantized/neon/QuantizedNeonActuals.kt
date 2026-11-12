// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.candle.internal.CandleCInterop

internal actual val vmullS8: (ByteArray, ByteArray) -> ShortArray = { a, b -> CandleCInterop.qneonVmullS8(a, b) }
internal actual val vmullS16: (ShortArray, ShortArray) -> IntArray = { a, b -> CandleCInterop.qneonVmullS16(a, b) }
internal actual val vpaddlqS16: (ShortArray) -> IntArray = { a -> CandleCInterop.qneonVpaddlqS16(a) }
internal actual val vpaddqS16: (ShortArray, ShortArray) -> ShortArray = { a, b -> CandleCInterop.qneonVpaddqS16(a, b) }
internal actual val vmlaqNF32: (FloatArray, FloatArray, Float) -> FloatArray = { acc, v, s ->
    CandleCInterop.qneonVmlaqNF32(acc, v, s)
}
internal actual val vaddvqS32: (IntArray) -> Int = { a -> CandleCInterop.qneonVaddvqS32(a) }
