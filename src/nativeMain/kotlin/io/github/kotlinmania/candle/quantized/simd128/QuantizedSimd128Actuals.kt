// port-lint: source quantized/simd128.rs
package io.github.kotlinmania.candle.quantized.simd128

import io.github.kotlinmania.candle.internal.CandleCInterop

internal actual val i32x4DotI16x8: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleCInterop.qsimd128DotI16x8(a, b)
}
