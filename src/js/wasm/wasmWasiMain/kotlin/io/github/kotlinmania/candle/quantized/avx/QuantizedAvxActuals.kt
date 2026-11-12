// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

internal actual val mm256MaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    IntArray(8) { i -> a[i * 2].toInt() * b[i * 2].toInt() + a[i * 2 + 1].toInt() * b[i * 2 + 1].toInt() }
}
internal actual val mmMaddEpi16: (ShortArray, ShortArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i * 2].toInt() * b[i * 2].toInt() + a[i * 2 + 1].toInt() * b[i * 2 + 1].toInt() }
}
internal actual val mm256MaddubsEpi16: (ByteArray, ByteArray) -> ShortArray = { ax, sy ->
    ShortArray(16) { i ->
        val u0 = ax[i * 2].toInt() and 0xFF
        val u1 = ax[i * 2 + 1].toInt() and 0xFF
        val s0 = sy[i * 2].toInt()
        val s1 = sy[i * 2 + 1].toInt()
        (u0 * s0 + u1 * s1).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
    }
}
internal actual val mm256SignEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i ->
        val bv = b[i].toInt()
        when {
            bv < 0 -> (-a[i].toInt()).toByte()
            bv == 0 -> 0
            else -> a[i]
        }
    }
}
internal actual val mmHaddEpi16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i ->
        if (i < 4) {
            (a[i * 2].toInt() + a[i * 2 + 1].toInt()).toShort()
        } else {
            (b[(i - 4) * 2].toInt() + b[(i - 4) * 2 + 1].toInt()).toShort()
        }
    }
}
internal actual val mmHaddEpi32: (IntArray, IntArray) -> IntArray = { a, b ->
    intArrayOf(a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3])
}
