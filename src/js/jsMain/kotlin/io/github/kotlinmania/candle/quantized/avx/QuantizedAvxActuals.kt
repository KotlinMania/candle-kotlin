// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

import io.github.kotlinmania.candle.internal.CandleNative

internal actual val mm256MaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleNative.qavxMaddEpi16(a, b)
}
internal actual val mmMaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleNative.qavxMmMaddEpi16(a, b)
}
internal actual val mm256MaddubsEpi16: (ByteArray, ByteArray) -> ShortArray = { ax, sy ->
    CandleNative.qavxMaddubsEpi16(ax, sy)
}
internal actual val mm256SignEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    CandleNative.qavxSignEpi8(a, b)
}
internal actual val mmHaddEpi16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    CandleNative.qavxHaddEpi16(a, b)
}
internal actual val mmHaddEpi32: (IntArray, IntArray) -> IntArray = { a, b ->
    CandleNative.qavxHaddEpi32(a, b)
}
