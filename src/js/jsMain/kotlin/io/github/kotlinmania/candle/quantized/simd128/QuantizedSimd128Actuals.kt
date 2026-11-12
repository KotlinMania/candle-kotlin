// port-lint: source quantized/simd128.rs
package io.github.kotlinmania.candle.quantized.simd128

import io.github.kotlinmania.candle.internal.CandleNative

internal actual val i32x4DotI16x8: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleNative.qsimd128DotI16x8(a, b)
}
