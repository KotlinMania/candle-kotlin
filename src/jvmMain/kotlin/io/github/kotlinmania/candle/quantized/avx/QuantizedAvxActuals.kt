// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

import io.github.kotlinmania.candle.internal.CandleJni

internal actual val mm256MaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    val out = IntArray(8)
    CandleJni.qavxMaddEpi16(a, b, out)
    out
}
internal actual val mmMaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    val out = IntArray(4)
    CandleJni.qavxMmMaddEpi16(a, b, out)
    out
}
internal actual val mm256MaddubsEpi16: (ByteArray, ByteArray) -> ShortArray = { ax, sy ->
    val out = ShortArray(16)
    CandleJni.qavxMaddubsEpi16(ax, sy, out)
    out
}
internal actual val mm256SignEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    val out = ByteArray(32)
    CandleJni.qavxSignEpi8(a, b, out)
    out
}
internal actual val mmHaddEpi16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    CandleJni.qavxHaddEpi16(a, b, out)
    out
}
internal actual val mmHaddEpi32: (IntArray, IntArray) -> IntArray = { a, b ->
    val out = IntArray(4)
    CandleJni.qavxHaddEpi32(a, b, out)
    out
}
