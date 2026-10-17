// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.BlockQ8K
import io.github.kotlinmania.candle.QK80
import io.github.kotlinmania.candle.QKK

private val mm256Set1Epi16: (Int) -> ShortArray = { v -> ShortArray(16) { v.toShort() } }
private val mm256Set1Epi8: (Int) -> ByteArray = { v -> ByteArray(32) { v.toByte() } }
private val mmSet1Epi8: (Int) -> ByteArray = { v -> ByteArray(16) { v.toByte() } }
private val mm256SetzeroPs: () -> FloatArray = { FloatArray(8) }
private val mmSetzeroPs: () -> FloatArray = { FloatArray(4) }
private val mm256SetzeroSi256: () -> IntArray = { IntArray(8) }
private val mmSetzeroSi128: () -> IntArray = { IntArray(4) }
private val mm256Set1Ps: (Float) -> FloatArray = { v -> FloatArray(8) { v } }
private val mmSet1Ps: (Float) -> FloatArray = { v -> FloatArray(4) { v } }
private val mm256BroadcastSs: (Float) -> FloatArray = { v -> FloatArray(8) { v } }
internal expect val mm256MaddEpi16: (ShortArray, ShortArray) -> IntArray
internal expect val mmMaddEpi16: (ShortArray, ShortArray) -> IntArray
internal expect val mm256MaddubsEpi16: (ByteArray, ByteArray) -> ShortArray
private val mm256Cvtepi32Ps: (IntArray) -> FloatArray = { a -> FloatArray(8) { i -> a[i].toFloat() } }
private val mmCvtepi32Ps: (IntArray) -> FloatArray = { a -> FloatArray(4) { i -> a[i].toFloat() } }
private val mm256Extractf128Ps: (FloatArray, Int) -> FloatArray = { x, imm ->
    FloatArray(4) { i -> x[imm * 4 + i] }
}
private val mm256Castps256Ps128: (FloatArray) -> FloatArray = { x ->
    FloatArray(4) { i -> x[i] }
}
private val mmAddPs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    FloatArray(4) { i -> a[i] + b[i] }
}
private val mmMovehlPs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    floatArrayOf(b[2], b[3], a[2], a[3])
}
private val mmMovehdupPs: (FloatArray) -> FloatArray = { a ->
    floatArrayOf(a[1], a[1], a[3], a[3])
}
private val mmAddSs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    floatArrayOf(a[0] + b[0], a[1], a[2], a[3])
}
private val mmCvtssF32: (FloatArray) -> Float = { a -> a[0] }
private val mmLoaduSi128: (ByteArray, Int) -> ByteArray = { src, off ->
    ByteArray(16) { i -> src[off + i] }
}
private val mmLoaduSi128U: (ByteArray, Int) -> ByteArray = { src, off ->
    ByteArray(16) { i -> src[off + i] }
}
private val mm256LoaduSi256: (ByteArray, Int) -> ByteArray = { src, off ->
    ByteArray(32) { i -> src[off + i] }
}
private val mm256LoaduSi256U: (ByteArray, Int) -> ByteArray = { src, off ->
    ByteArray(32) { i -> src[off + i] }
}
private val mm256LoaduSi256S: (ShortArray, Int) -> ShortArray = { src, off ->
    ShortArray(16) { i -> src[off + i] }
}
private val mm256Castsi128Si256: (ByteArray) -> ByteArray = { a ->
    ByteArray(32) { i -> if (i < 16) a[i] else 0 }
}
private val mm256Insertf128Si256: (ByteArray, ByteArray, Int) -> ByteArray = { a, b, imm ->
    val out = a.copyOf()
    val base = imm * 16
    for (i in 0 until 16) out[base + i] = b[i]
    out
}
private val mm256Insertf128Si256S: (ShortArray, ShortArray, Int) -> ShortArray = { a, b, imm ->
    val out = a.copyOf()
    val base = imm * 8
    for (i in 0 until 8) out[base + i] = b[i]
    out
}
private val mmSrliEpi16: (ByteArray, Int) -> ByteArray = { a, shift ->
    val out = ByteArray(16)
    for (i in 0 until 8) {
        val w = ((a[i * 2].toInt() and 0xFF) or ((a[i * 2 + 1].toInt() and 0xFF) shl 8)) ushr shift
        out[i * 2] = (w and 0xFF).toByte()
        out[i * 2 + 1] = ((w ushr 8) and 0xFF).toByte()
    }
    out
}
private val mm256SrliEpi16: (ByteArray, Int) -> ByteArray = { a, shift ->
    val out = ByteArray(32)
    for (i in 0 until 16) {
        val w = ((a[i * 2].toInt() and 0xFF) or ((a[i * 2 + 1].toInt() and 0xFF) shl 8)) ushr shift
        out[i * 2] = (w and 0xFF).toByte()
        out[i * 2 + 1] = ((w ushr 8) and 0xFF).toByte()
    }
    out
}
private val mm256SlliEpi16: (ByteArray, Int) -> ByteArray = { a, shift ->
    val out = ByteArray(32)
    for (i in 0 until 16) {
        val w = (((a[i * 2].toInt() and 0xFF) or ((a[i * 2 + 1].toInt() and 0xFF) shl 8)) shl shift) and 0xFFFF
        out[i * 2] = (w and 0xFF).toByte()
        out[i * 2 + 1] = ((w ushr 8) and 0xFF).toByte()
    }
    out
}
private val mm256AndSi256: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i -> (a[i].toInt() and b[i].toInt()).toByte() }
}
private val mm256AndnotSi256: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i -> (a[i].toInt().inv() and b[i].toInt()).toByte() }
}
private val mm256OrSi256: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i -> (a[i].toInt() or b[i].toInt()).toByte() }
}
private val mmAndSi128: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() and b[i].toInt()).toByte() }
}
private val mm256SubEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i -> (a[i].toInt() - b[i].toInt()).toByte() }
}
private val mmSubEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() - b[i].toInt()).toByte() }
}
private val mm256AddEpi8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(32) { i -> (a[i].toInt() + b[i].toInt()).toByte() }
}
private val mm256SubEpi16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(16) { i -> (a[i].toInt() - b[i].toInt()).toShort() }
}
private val mm256AddEpi32: (IntArray, IntArray) -> IntArray = { a, b ->
    IntArray(8) { i -> a[i] + b[i] }
}
internal expect val mm256SignEpi8: (ByteArray, ByteArray) -> ByteArray
private val mm256FmaddPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    FloatArray(8) { i -> a[i] * b[i] + c[i] }
}
private val mmFmaddPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    FloatArray(4) { i -> a[i] * b[i] + c[i] }
}
private val mmShuffleEpi8: (ByteArray, ByteArray) -> ByteArray = { a, shuf ->
    ByteArray(16) { i ->
        val idx = shuf[i].toInt()
        if (idx < 0) 0 else a[idx and 0xF]
    }
}
private val mm256ShuffleEpi8S: (ShortArray, ByteArray) -> ShortArray = { a, shuf ->
    val bytes = ByteArray(32)
    for (i in 0 until 16) {
        bytes[i * 2] = (a[i].toInt() and 0xFF).toByte()
        bytes[i * 2 + 1] = ((a[i].toInt() ushr 8) and 0xFF).toByte()
    }
    val outBytes = ByteArray(32)
    for (lane in 0 until 2) {
        val base = lane * 16
        for (i in 0 until 16) {
            val idx = shuf[base + i].toInt()
            outBytes[base + i] = if (idx < 0) 0 else bytes[base + (idx and 0xF)]
        }
    }
    ShortArray(16) { i ->
        ((outBytes[i * 2].toInt() and 0xFF) or (outBytes[i * 2 + 1].toInt() shl 8)).toShort()
    }
}
private val mm256Cvtepi8Epi16: (ByteArray) -> ShortArray = { a ->
    ShortArray(16) { i -> a[i].toShort() }
}
private val mm256Cvtepu8Epi16: (ByteArray) -> ShortArray = { a ->
    ShortArray(16) { i -> (a[i].toInt() and 0xFF).toShort() }
}
private val mm256Extracti128Si256S: (ShortArray, Int) -> ShortArray = { a, imm ->
    ShortArray(8) { i -> a[imm * 8 + i] }
}
private val mm256Extracti128Si256B: (ByteArray, Int) -> ByteArray = { a, imm ->
    ByteArray(16) { i -> a[imm * 16 + i] }
}
internal expect val mmHaddEpi16: (ShortArray, ShortArray) -> ShortArray
internal expect val mmHaddEpi32: (IntArray, IntArray) -> IntArray
private val mmExtractEpi32: (IntArray, Int) -> Int = { a, imm -> a[imm] }
private val mmSetEpi32: (Int, Int, Int, Int) -> ByteArray = { e3, e2, e1, e0 ->
    val words = intArrayOf(e0, e1, e2, e3)
    val out = ByteArray(16)
    for (w in 0 until 4) {
        out[w * 4] = (words[w] and 0xFF).toByte()
        out[w * 4 + 1] = ((words[w] ushr 8) and 0xFF).toByte()
        out[w * 4 + 2] = ((words[w] ushr 16) and 0xFF).toByte()
        out[w * 4 + 3] = ((words[w] ushr 24) and 0xFF).toByte()
    }
    out
}
private val readU32Into: (ByteArray, IntArray, Int) -> Unit = { src, dst, count ->
    for (w in 0 until count) {
        val b0 = src[w * 4].toInt() and 0xFF
        val b1 = src[w * 4 + 1].toInt() and 0xFF
        val b2 = src[w * 4 + 2].toInt() and 0xFF
        val b3 = src[w * 4 + 3].toInt() and 0xFF
        dst[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
}
private val kShuffle128: ByteArray = ByteArray(128) { idx -> (idx / 8).toByte() }
private val kShuffleK4: ByteArray =
    ByteArray(256) { idx ->
        val group = idx / 32
        val pair = idx % 2
        (group * 2 + pair).toByte()
    }
private val kShuffleQ3k: ByteArray =
    ByteArray(128) { idx ->
        val group = idx / 16
        val pair = idx % 2
        (group * 2 + pair).toByte()
    }

internal fun sumI16PairsFloat(x: ShortArray): FloatArray {
    val ones = mm256Set1Epi16(1)
    val summedPairs = mm256MaddEpi16(ones, x)
    return mm256Cvtepi32Ps(summedPairs)
}

internal fun mulSumUs8PairsFloat(
    ax: ByteArray,
    sy: ByteArray,
): FloatArray {
    val dot = mm256MaddubsEpi16(ax, sy)
    return sumI16PairsFloat(dot)
}

internal fun hsumFloat8(x: FloatArray): Float {
    var res = mm256Extractf128Ps(x, 1)
    res = mmAddPs(res, mm256Castps256Ps128(x))
    res = mmAddPs(res, mmMovehlPs(res, res))
    res = mmAddSs(res, mmMovehdupPs(res))
    return mmCvtssF32(res)
}

internal fun bytesFromNibbles32(rsi: ByteArray): ByteArray {
    val tmp = mmLoaduSi128(rsi, 0)
    val bytes = mm256Insertf128Si256(mm256Castsi128Si256(tmp), mmSrliEpi16(tmp, 4), 1)
    val lowMask = mm256Set1Epi8(0xF)
    return mm256AndSi256(lowMask, bytes)
}

internal fun mulSumI8PairsFloat(
    x: ByteArray,
    y: ByteArray,
): FloatArray {
    val ax = mm256SignEpi8(x, x)
    val sy = mm256SignEpi8(y, x)
    return mulSumUs8PairsFloat(ax, sy)
}

internal fun vecDotQ40Q80(
    n: Int,
    xs: List<BlockQ40>,
    ys: List<BlockQ80>,
): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    var acc = mm256SetzeroPs()
    for ((x, y) in xs.zip(ys)) {
        val d = mm256Set1Ps(x.d.toFloat() * y.d.toFloat())
        var bx = bytesFromNibbles32(mmLoaduSi128U(x.qs, 0))
        val off = mm256Set1Epi8(8)
        bx = mm256SubEpi8(bx, off)
        val by = mm256LoaduSi256(y.qs, 0)
        val q = mulSumI8PairsFloat(bx, by)
        acc = mm256FmaddPs(d, q, acc)
    }
    return hsumFloat8(acc)
}

internal fun vecDotQ80Q80(
    n: Int,
    xs: List<BlockQ80>,
    ys: List<BlockQ80>,
): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    var acc = mm256SetzeroPs()
    for ((x, y) in xs.zip(ys)) {
        val d = mm256Set1Ps(x.d.toFloat() * y.d.toFloat())
        val bx = mm256LoaduSi256(x.qs, 0)
        val by = mm256LoaduSi256(y.qs, 0)
        val q = mulSumI8PairsFloat(bx, by)
        acc = mm256FmaddPs(d, q, acc)
    }
    return hsumFloat8(acc)
}

private fun getScaleShuffle(i: Int): ByteArray = mmLoaduSi128(kShuffle128, i * 16)

private fun getScaleShuffleK4(i: Int): ByteArray = mm256LoaduSi256(kShuffleK4, i * 32)

private fun getScaleShuffleQ3k(i: Int): ByteArray = mm256LoaduSi256(kShuffleQ3k, i * 32)

internal fun vecDotQ6kQ8k(
    n: Int,
    xs: List<BlockQ6K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    val m4 = mm256Set1Epi8(0xF)
    val m2 = mm256Set1Epi8(3)
    val m32s = mm256Set1Epi8(32)
    var acc = mm256SetzeroPs()
    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        var q4 = 0
        var qh = 0
        var q8 = 0

        val scales = mmLoaduSi128(x.scales, 0)
        var sumi = mm256SetzeroSi256()

        for (j in 0 until QKK / 128) {
            val isIdx = j * 4
            val scale0 = mmShuffleEpi8(scales, getScaleShuffle(isIdx))
            val scale1 = mmShuffleEpi8(scales, getScaleShuffle(isIdx + 1))
            val scale2 = mmShuffleEpi8(scales, getScaleShuffle(isIdx + 2))
            val scale3 = mmShuffleEpi8(scales, getScaleShuffle(isIdx + 3))

            val q4bits1 = mm256LoaduSi256U(x.ql, q4)
            q4 += 32
            val q4bits2 = mm256LoaduSi256U(x.ql, q4)
            q4 += 32
            val q4bitsH = mm256LoaduSi256U(x.qh, qh)
            qh += 32

            val q4h0 = mm256SlliEpi16(mm256AndSi256(q4bitsH, m2), 4)
            val q4h1 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 2), m2), 4)
            val q4h2 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 4), m2), 4)
            val q4h3 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 6), m2), 4)

            val q40 = mm256OrSi256(mm256AndSi256(q4bits1, m4), q4h0)
            val q41 = mm256OrSi256(mm256AndSi256(q4bits2, m4), q4h1)
            val q42 = mm256OrSi256(mm256AndSi256(mm256SrliEpi16(q4bits1, 4), m4), q4h2)
            val q43 = mm256OrSi256(mm256AndSi256(mm256SrliEpi16(q4bits2, 4), m4), q4h3)

            val q80 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q81 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q82 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q83 = mm256LoaduSi256(y.qs, q8)
            q8 += 32

            val q8s0 = mm256MaddubsEpi16(m32s, q80)
            val q8s1 = mm256MaddubsEpi16(m32s, q81)
            val q8s2 = mm256MaddubsEpi16(m32s, q82)
            val q8s3 = mm256MaddubsEpi16(m32s, q83)

            val p160 = mm256SubEpi16(mm256MaddubsEpi16(q40, q80), q8s0)
            val p161 = mm256SubEpi16(mm256MaddubsEpi16(q41, q81), q8s1)
            val p162 = mm256SubEpi16(mm256MaddubsEpi16(q42, q82), q8s2)
            val p163 = mm256SubEpi16(mm256MaddubsEpi16(q43, q83), q8s3)

            val m0 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale0), p160)
            val m1 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale1), p161)
            val m2v = mm256MaddEpi16(mm256Cvtepi8Epi16(scale2), p162)
            val m3 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale3), p163)

            sumi = mm256AddEpi32(sumi, mm256AddEpi32(m0, m1))
            sumi = mm256AddEpi32(sumi, mm256AddEpi32(m2v, m3))
        }
        acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
    }
    return hsumFloat8(acc)
}

private fun mm256SetM128i(
    a: ShortArray,
    b: ShortArray,
): ShortArray = mm256Insertf128Si256S(ShortArray(16) { i -> if (i < 8) b[i] else 0 }, a, 1)

internal fun vecDotQ2kQ8k(
    n: Int,
    xs: List<BlockQ2K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    val m3 = mm256Set1Epi8(3)
    val m4 = mmSet1Epi8(0xF)
    var acc = mm256SetzeroPs()

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        var q2 = 0
        var q8 = 0

        val minsAndScales = mmLoaduSi128U(x.scales, 0)
        val scales8 = mmAndSi128(minsAndScales, m4)
        val mins8 = mmAndSi128(mmSrliEpi16(minsAndScales, 4), m4)
        val mins = mm256Cvtepi8Epi16(mins8)
        val prod = mm256MaddEpi16(mins, mm256LoaduSi256S(y.bsums, 0))

        acc = mm256FmaddPs(mm256BroadcastSs(dmin), mm256Cvtepi32Ps(prod), acc)

        val allScales = mm256Cvtepi8Epi16(scales8)
        val lScales = mm256Extracti128Si256S(allScales, 0)
        val hScales = mm256Extracti128Si256S(allScales, 1)
        val scales =
            arrayOf(
                mm256SetM128i(lScales, lScales),
                mm256SetM128i(hScales, hScales),
            )

        var sumi = mm256SetzeroSi256()

        for (scale in scales) {
            val q2bits = mm256LoaduSi256U(x.qs, q2)
            q2 += 32

            val q80 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q81 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q82 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q83 = mm256LoaduSi256(y.qs, q8)
            q8 += 32

            val q20 = mm256AndSi256(q2bits, m3)
            val q21 = mm256AndSi256(mm256SrliEpi16(q2bits, 2), m3)
            val q22 = mm256AndSi256(mm256SrliEpi16(q2bits, 4), m3)
            val q23 = mm256AndSi256(mm256SrliEpi16(q2bits, 6), m3)

            var p0 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(0)), mm256MaddubsEpi16(q20, q80))
            val p1 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(1)), mm256MaddubsEpi16(q21, q81))
            var p2 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(2)), mm256MaddubsEpi16(q22, q82))
            val p3 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(3)), mm256MaddubsEpi16(q23, q83))

            p0 = mm256AddEpi32(p0, p1)
            p2 = mm256AddEpi32(p2, p3)

            sumi = mm256AddEpi32(sumi, mm256AddEpi32(p0, p2))
        }
        acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
    }

    return hsumFloat8(acc)
}

internal fun vecDotQ3kQ8k(
    n: Int,
    xs: List<BlockQ3K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ3kQ8k: $n is not divisible by $QKK"
    }
    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f
    val aux = IntArray(3)

    val m3 = mm256Set1Epi8(3)
    val mone = mm256Set1Epi8(1)
    val m32 = mmSet1Epi8(32)

    var acc = mm256SetzeroPs()
    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()

        var q3 = 0
        var q8 = 0

        readU32Into(x.scales, aux, 3)
        var scales128 =
            mmSetEpi32(
                ((aux[1] ushr 4) and kmask2) or (((aux[2] ushr 6) and kmask1) shl 4),
                ((aux[0] ushr 4) and kmask2) or (((aux[2] ushr 4) and kmask1) shl 4),
                (aux[1] and kmask2) or (((aux[2] ushr 2) and kmask1) shl 4),
                (aux[0] and kmask2) or ((aux[2] and kmask1) shl 4),
            )
        scales128 = mmSubEpi8(scales128, m32)
        val allScales = mm256Cvtepi8Epi16(scales128)
        val lScales = mm256Extracti128Si256S(allScales, 0)
        val hScales = mm256Extracti128Si256S(allScales, 1)
        val scales =
            arrayOf(
                mm256SetM128i(lScales, lScales),
                mm256SetM128i(hScales, hScales),
            )

        // high bit
        val hbits = mm256LoaduSi256U(x.hmask, 0)

        var sumi = mm256SetzeroSi256()

        for ((j, scale) in scales.withIndex()) {
            // load low 2 bits
            val q3bits = mm256LoaduSi256U(x.qs, q3)
            q3 += 32

            // Prepare low and high bits
            val q3l0 = mm256AndSi256(q3bits, m3)
            val q3h0 =
                mm256SlliEpi16(
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 0)), 0)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 4)), 4)
                    },
                    2,
                )

            val q3l1 = mm256AndSi256(mm256SrliEpi16(q3bits, 2), m3)
            val q3h1 =
                mm256SlliEpi16(
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 1)), 1)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 5)), 5)
                    },
                    2,
                )

            val q3l2 = mm256AndSi256(mm256SrliEpi16(q3bits, 4), m3)
            val q3h2 =
                mm256SlliEpi16(
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 2)), 2)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 6)), 6)
                    },
                    2,
                )

            val q3l3 = mm256AndSi256(mm256SrliEpi16(q3bits, 6), m3)
            val q3h3 =
                mm256SlliEpi16(
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 3)), 3)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 7)), 7)
                    },
                    2,
                )

            // load Q8 quants
            val q80 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q81 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q82 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q83 = mm256LoaduSi256(y.qs, q8)
            q8 += 32

            val q8s0 = mm256MaddubsEpi16(q3h0, q80)
            val q8s1 = mm256MaddubsEpi16(q3h1, q81)
            val q8s2 = mm256MaddubsEpi16(q3h2, q82)
            val q8s3 = mm256MaddubsEpi16(q3h3, q83)

            val p160 = mm256SubEpi16(mm256MaddubsEpi16(q3l0, q80), q8s0)
            val p161 = mm256SubEpi16(mm256MaddubsEpi16(q3l1, q81), q8s1)
            val p162 = mm256SubEpi16(mm256MaddubsEpi16(q3l2, q82), q8s2)
            val p163 = mm256SubEpi16(mm256MaddubsEpi16(q3l3, q83), q8s3)

            // multiply with scales
            var m0 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(0)), p160)
            val m1 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(1)), p161)
            var m2 = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(2)), p162)
            val m3v = mm256MaddEpi16(mm256ShuffleEpi8S(scale, getScaleShuffleQ3k(3)), p163)

            // accumulate
            m0 = mm256AddEpi32(m0, m1)
            m2 = mm256AddEpi32(m2, m3v)
            sumi = mm256AddEpi32(sumi, mm256AddEpi32(m0, m2))
        }

        // multiply with block scale and accumulate
        acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
    }
    return hsumFloat8(acc)
}

internal fun vecDotQ4kQ8k(
    n: Int,
    xs: List<BlockQ4K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ4kQ8k: $n is not divisible by $QKK"
    }
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val m4 = mm256Set1Epi8(0xF)

    var acc = mm256SetzeroPs()
    var accM = mmSetzeroPs()

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        readU32Into(x.scales, utmp, 3)

        utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        var q4 = 0
        var q8 = 0

        val minsAndScales =
            mm256Cvtepu8Epi16(
                mmSetEpi32(
                    utmp[3],
                    utmp[2],
                    utmp[1],
                    utmp[0],
                ),
            )

        val q8sums = mm256LoaduSi256S(y.bsums, 0)
        val q8s =
            mmHaddEpi16(
                mm256Extracti128Si256S(q8sums, 0),
                mm256Extracti128Si256S(q8sums, 1),
            )
        val prod = mmMaddEpi16(mm256Extracti128Si256S(minsAndScales, 1), q8s)
        accM = mmFmaddPs(mmSet1Ps(dmin), mmCvtepi32Ps(prod), accM)

        val sc128 = mm256Extracti128Si256S(minsAndScales, 0)
        val scales = mm256SetM128i(sc128, sc128)

        var sumi = mm256SetzeroSi256()

        for (j in 0 until QKK / 64) {
            val scaleL = mm256ShuffleEpi8S(scales, getScaleShuffleK4(2 * j))
            val scaleH = mm256ShuffleEpi8S(scales, getScaleShuffleK4(2 * j + 1))

            val q4bits = mm256LoaduSi256U(x.qs, q4)
            q4 += 32
            val q4l = mm256AndSi256(q4bits, m4)
            val q4h = mm256AndSi256(mm256SrliEpi16(q4bits, 4), m4)

            val q8l = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val p16l = mm256MaddEpi16(scaleL, mm256MaddubsEpi16(q4l, q8l))
            sumi = mm256AddEpi32(sumi, p16l)

            val q8h = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val p16h = mm256MaddEpi16(scaleH, mm256MaddubsEpi16(q4h, q8h))
            sumi = mm256AddEpi32(sumi, p16h)
        }

        val vd = mm256Set1Ps(d)
        acc = mm256FmaddPs(vd, mm256Cvtepi32Ps(sumi), acc)
    }

    accM = mmAddPs(accM, mmMovehlPs(accM, accM))
    accM = mmAddSs(accM, mmMovehdupPs(accM))

    return hsumFloat8(acc) + mmCvtssF32(accM)
}

internal fun vecDotQ5kQ8k(
    n: Int,
    xs: List<BlockQ5K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ5kQ8k: $n is not divisible by $QKK"
    }
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val m4 = mm256Set1Epi8(0xF)
    val mzero = mmSetzeroSi128()
    val mone = mm256Set1Epi8(1)

    var acc = mm256SetzeroPs()
    var summs = 0.0f

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        readU32Into(x.scales, utmp, 3)

        utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        var q5 = 0
        var q8 = 0

        val minsAndScales =
            mm256Cvtepu8Epi16(
                mmSetEpi32(
                    utmp[3],
                    utmp[2],
                    utmp[1],
                    utmp[0],
                ),
            )

        val q8sums = mm256LoaduSi256S(y.bsums, 0)
        val q8s =
            mmHaddEpi16(
                mm256Extracti128Si256S(q8sums, 0),
                mm256Extracti128Si256S(q8sums, 1),
            )
        val prod = mmMaddEpi16(mm256Extracti128Si256S(minsAndScales, 1), q8s)
        val hsum = mmHaddEpi32(mmHaddEpi32(prod, mzero), mzero)
        summs += dmin * mmExtractEpi32(hsum, 0).toFloat()

        val sc128 = mm256Extracti128Si256S(minsAndScales, 0)
        val scales = mm256SetM128i(sc128, sc128)

        val hbits = mm256LoaduSi256U(x.qh, 0)
        var hmask = mone

        var sumi = mm256SetzeroSi256()

        for (j in 0 until QKK / 64) {
            val scale0 = mm256ShuffleEpi8S(scales, getScaleShuffleK4(2 * j))
            val scale1 = mm256ShuffleEpi8S(scales, getScaleShuffleK4(2 * j + 1))

            val q5bits = mm256LoaduSi256U(x.qs, q5)
            q5 += 32

            // Similar to q3k we hardcode the shifts here to avoid loading them into a separate register
            val q5l0 = mm256AndSi256(q5bits, m4)
            val q5l0ShiftInput = mm256AndSi256(hbits, hmask)
            val q5l0RightShift =
                when (j) {
                    0 -> mm256SrliEpi16(q5l0ShiftInput, 0)
                    1 -> mm256SrliEpi16(q5l0ShiftInput, 2)
                    2 -> mm256SrliEpi16(q5l0ShiftInput, 4)
                    3 -> mm256SrliEpi16(q5l0ShiftInput, 6)
                    else -> error("unreachable")
                }
            val q5h0 = mm256SlliEpi16(q5l0RightShift, 4)
            val q50 = mm256AddEpi8(q5l0, q5h0)
            hmask = mm256SlliEpi16(hmask, 1)

            val q5l1 = mm256AndSi256(mm256SrliEpi16(q5bits, 4), m4)
            val q5l1ShiftInput = mm256AndSi256(hbits, hmask)
            val q5l1RightShift =
                when (j) {
                    0 -> mm256SrliEpi16(q5l1ShiftInput, 1)
                    1 -> mm256SrliEpi16(q5l1ShiftInput, 3)
                    2 -> mm256SrliEpi16(q5l1ShiftInput, 5)
                    3 -> mm256SrliEpi16(q5l1ShiftInput, 7)
                    else -> error("unreachable")
                }

            val q5h1 = mm256SlliEpi16(q5l1RightShift, 4)
            val q51 = mm256AddEpi8(q5l1, q5h1)
            hmask = mm256SlliEpi16(hmask, 1)

            val q80 = mm256LoaduSi256(y.qs, q8)
            q8 += 32
            val q81 = mm256LoaduSi256(y.qs, q8)
            q8 += 32

            val p160 = mm256MaddEpi16(scale0, mm256MaddubsEpi16(q50, q80))
            val p161 = mm256MaddEpi16(scale1, mm256MaddubsEpi16(q51, q81))

            sumi = mm256AddEpi32(sumi, mm256AddEpi32(p160, p161))
        }
        val vd = mm256Set1Ps(d)
        acc = mm256FmaddPs(vd, mm256Cvtepi32Ps(sumi), acc)
    }
    return hsumFloat8(acc) + summs
}

internal fun vecDotQ8kQ8k(
    n: Int,
    xs: List<BlockQ8K>,
    ys: List<BlockQ8K>,
): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    var acc = mm256SetzeroPs()
    for ((x, y) in xs.zip(ys)) {
        var sumi = mm256SetzeroSi256()
        val xQs = x.qs
        val yQs = y.qs
        for (j in 0 until QKK step 32) {
            val xsVec = mm256LoaduSi256(xQs, j)
            val ysVec = mm256LoaduSi256(yQs, j)

            val xs0 = mm256Cvtepi8Epi16(mm256Extracti128Si256B(xsVec, 0))
            val ys0 = mm256Cvtepi8Epi16(mm256Extracti128Si256B(ysVec, 0))
            sumi = mm256AddEpi32(sumi, mm256MaddEpi16(xs0, ys0))

            val xs1 = mm256Cvtepi8Epi16(mm256Extracti128Si256B(xsVec, 1))
            val ys1 = mm256Cvtepi8Epi16(mm256Extracti128Si256B(ysVec, 1))
            sumi = mm256AddEpi32(sumi, mm256MaddEpi16(xs1, ys1))
        }
        val d = mm256Set1Ps(x.d * y.d)
        acc = mm256FmaddPs(d, mm256Cvtepi32Ps(sumi), acc)
    }
    return hsumFloat8(acc)
}
