// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.candle.internal.CandleJni

internal actual val vmullS8: (ByteArray, ByteArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    CandleJni.qneonVmullS8(a, b, out)
    out
}
internal actual val vmullS16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    val out = IntArray(4)
    CandleJni.qneonVmullS16(a, b, out)
    out
}
internal actual val vpaddlqS16: (ShortArray) -> IntArray = { a ->
    val out = IntArray(4)
    CandleJni.qneonVpaddlqS16(a, out)
    out
}
internal actual val vpaddqS16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    CandleJni.qneonVpaddqS16(a, b, out)
    out
}
internal actual val vmlaqNF32: (FloatArray, FloatArray, Float) -> FloatArray = { acc, v, s ->
    val out = FloatArray(4)
    CandleJni.qneonVmlaqNF32(acc, v, s, out)
    out
}
internal actual val vaddvqS32: (IntArray) -> Int = { a -> CandleJni.qneonVaddvqS32(a) }
