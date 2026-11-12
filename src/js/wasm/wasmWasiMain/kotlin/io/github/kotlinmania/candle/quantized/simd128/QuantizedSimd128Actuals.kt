// port-lint: source quantized/simd128.rs
package io.github.kotlinmania.candle.quantized.simd128

internal actual val i32x4DotI16x8: (ShortArray, ShortArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i * 2].toInt() * b[i * 2].toInt() + a[i * 2 + 1].toInt() * b[i * 2 + 1].toInt() }
}
