// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.candle.internal.CandleNative

internal actual val vmullS8: (ByteArray, ByteArray) -> ShortArray = { a, b -> CandleNative.qneonVmullS8(a, b) }
internal actual val vmullS16: (ShortArray, ShortArray) -> IntArray = { a, b -> CandleNative.qneonVmullS16(a, b) }
internal actual val vpaddlqS16: (ShortArray) -> IntArray = { a -> CandleNative.qneonVpaddlqS16(a) }
internal actual val vpaddqS16: (ShortArray, ShortArray) -> ShortArray = { a, b -> CandleNative.qneonVpaddqS16(a, b) }
internal actual val vmlaqNF32: (FloatArray, FloatArray, Float) -> FloatArray = { acc, v, s ->
    CandleNative.qneonVmlaqNF32(acc, v, s)
}
internal actual val vaddvqS32: (IntArray) -> Int = { a -> CandleNative.qneonVaddvqS32(a) }
