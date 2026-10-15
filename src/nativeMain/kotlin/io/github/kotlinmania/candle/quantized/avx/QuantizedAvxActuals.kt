// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

import io.github.kotlinmania.candle.internal.CandleCInterop

internal actual val mm256MaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleCInterop.qavxMaddEpi16(a, b)
}
internal actual val mmMaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    CandleCInterop.qavxMmMaddEpi16(a, b)
}
internal actual val mm256MaddubsEpi16: (ByteArray, ByteArray) -> ShortArray = { ax, sy ->
    CandleCInterop.qavxMaddubsEpi16(ax, sy)
}
internal actual val mm256SignEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    CandleCInterop.qavxSignEpi8(a, b)
}
internal actual val mmHaddEpi16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    CandleCInterop.qavxHaddEpi16(a, b)
}
internal actual val mmHaddEpi32: (IntArray, IntArray) -> IntArray = { a, b ->
    CandleCInterop.qavxHaddEpi32(a, b)
}
