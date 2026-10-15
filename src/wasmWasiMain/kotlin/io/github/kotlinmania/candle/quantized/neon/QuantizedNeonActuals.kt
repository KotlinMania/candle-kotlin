// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

internal actual val vmullS8: (ByteArray, ByteArray) -> ShortArray = { a, b ->
    ShortArray(8) { i -> (a[i].toInt() * b[i].toInt()).toShort() }
}
internal actual val vmullS16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i].toInt() * b[i].toInt() }
}
internal actual val vpaddlqS16: (ShortArray) -> IntArray = { a ->
    IntArray(4) { i -> a[i * 2].toInt() + a[i * 2 + 1].toInt() }
}
internal actual val vpaddqS16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i ->
        if (i < 4) (a[i * 2].toInt() + a[i * 2 + 1].toInt()).toShort()
        else (b[(i - 4) * 2].toInt() + b[(i - 4) * 2 + 1].toInt()).toShort()
    }
}
internal actual val vmlaqNF32: (FloatArray, FloatArray, Float) -> FloatArray = { acc, v, s ->
    FloatArray(4) { i -> acc[i] + v[i] * s }
}
internal actual val vaddvqS32: (IntArray) -> Int = { a -> a[0] + a[1] + a[2] + a[3] }
