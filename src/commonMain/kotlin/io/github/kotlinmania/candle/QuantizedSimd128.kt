// port-lint: source quantized/simd128.rs
package io.github.kotlinmania.candle.quantized.simd128

import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ8K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.QK80
import io.github.kotlinmania.candle.QKK

private val f32x4Splat: (Float) -> FloatArray = { v -> FloatArray(4) { v } }
private val i32x4Splat: (Int) -> IntArray = { v -> IntArray(4) { v } }
private val i32x4Of: (Int, Int, Int, Int) -> IntArray = { a, b, c, d -> intArrayOf(a, b, c, d) }
private val i16x8Splat: (Int) -> ShortArray = { v -> ShortArray(8) { v.toShort() } }
private val u8x16Splat: (Int) -> ByteArray = { v -> ByteArray(16) { v.toByte() } }
private val i8x16Splat: (Int) -> ByteArray = { v -> ByteArray(16) { v.toByte() } }
private val v128Load: (ByteArray, Int) -> ByteArray = { src, off -> src.copyOfRange(off, off + 16) }
private val v128LoadU: (ByteArray, Int) -> ByteArray = { src, off -> ByteArray(16) { i -> src[off + i] } }
private val v128Store: (ByteArray, Int, ByteArray) -> Unit = { dst, off, v ->
    for (i in 0 until 16) dst[off + i] = v[i]
}
private val v128StoreU: (UByteArray, Int, ByteArray) -> Unit = { dst, off, v ->
    for (i in 0 until 16) dst[off + i] = v[i].toUByte()
}
private val v128And: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() and b[i].toInt()).toByte() }
}
private val v128Or: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() or b[i].toInt()).toByte() }
}
private val v128AndI16: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i -> (a[i].toInt() and b[i].toInt()).toShort() }
}
private val i8x16Sub: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() - b[i].toInt()).toByte() }
}
private val u8x16Shr: (ByteArray, Int) -> ByteArray = { a, shift ->
    ByteArray(16) { i -> ((a[i].toInt() and 0xFF) ushr shift).toByte() }
}
private val u8x16Shl: (ByteArray, Int) -> ByteArray = { a, shift ->
    ByteArray(16) { i -> ((a[i].toInt() and 0xFF) shl shift).toByte() }
}
private val i16x8ExtendLowI8x16: (ByteArray) -> ShortArray = { a ->
    ShortArray(8) { i -> a[i].toShort() }
}
private val i16x8ExtendHighI8x16: (ByteArray) -> ShortArray = { a ->
    ShortArray(8) { i -> a[i + 8].toShort() }
}
private val i16x8ExtendLowU8x16: (ByteArray) -> ShortArray = { a ->
    ShortArray(8) { i -> (a[i].toInt() and 0xFF).toShort() }
}
private val i16x8ExtendHighU8x16: (ByteArray) -> ShortArray = { a ->
    ShortArray(8) { i -> (a[i + 8].toInt() and 0xFF).toShort() }
}
private val i16x8LoadExtendI8x8: (ByteArray, Int) -> ShortArray = { src, off ->
    ShortArray(8) { i -> src[off + i].toShort() }
}
private fun i16x8LoadExtendU8x8(src: UByteArray, off: Int): ShortArray =
    ShortArray(8) { i -> (src[off + i].toInt() and 0xFF).toShort() }
private fun i16x8LoadExtendU8x8(src: ByteArray, off: Int): ShortArray =
    ShortArray(8) { i -> (src[off + i].toInt() and 0xFF).toShort() }
private val i16x8LoadExtendU8x8B: (ByteArray, Int) -> ShortArray = { src, off ->
    ShortArray(8) { i -> (src[off + i].toInt() and 0xFF).toShort() }
}
private val i16x8Shr: (ShortArray, Int) -> ShortArray = { a, shift ->
    ShortArray(8) { i -> (a[i].toInt() shr shift).toShort() }
}
private val i16x8Add: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i -> (a[i].toInt() + b[i].toInt()).toShort() }
}
private val i16x8Sub: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i -> (a[i].toInt() - b[i].toInt()).toShort() }
}
private val i16x8Mul: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) { i -> (a[i].toInt() * b[i].toInt()).toShort() }
}
private val i8x16NarrowI16x8: (ShortArray, ShortArray) -> ByteArray = { low, high ->
    ByteArray(16) { i -> if (i < 8) low[i].toByte() else high[i - 8].toByte() }
}
private val i32x4ExtendLowI16x8: (ShortArray) -> IntArray = { a ->
    IntArray(4) { i -> a[i].toInt() }
}
private val i32x4ExtendHighI16x8: (ShortArray) -> IntArray = { a ->
    IntArray(4) { i -> a[i + 4].toInt() }
}
private val i32x4LoadExtendI16x4: (ShortArray, Int) -> IntArray = { src, off ->
    IntArray(4) { i -> src[off + i].toInt() }
}
internal expect val i32x4DotI16x8: (ShortArray, ShortArray) -> IntArray
private val i32x4Add: (IntArray, IntArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i] + b[i] }
}
private val i32x4Mul: (IntArray, IntArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i] * b[i] }
}
private val i32x4Shr: (IntArray, Int) -> IntArray = { a, shift ->
    IntArray(4) { i -> a[i] shr shift }
}
private val f32x4ConvertI32x4: (IntArray) -> FloatArray = { a ->
    FloatArray(4) { i -> a[i].toFloat() }
}
private val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    FloatArray(4) { i -> a[i] + b[i] }
}
private val f32x4Sub: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    FloatArray(4) { i -> a[i] - b[i] }
}
private val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    FloatArray(4) { i -> a[i] * b[i] }
}
private val f32x4ExtractLane: (FloatArray, Int) -> Float = { a, lane -> a[lane] }
private val readU32Into: (ByteArray, IntArray, Int) -> Unit = { src, dst, count ->
    for (w in 0 until count) {
        val b0 = src[w * 4].toInt() and 0xFF
        val b1 = src[w * 4 + 1].toInt() and 0xFF
        val b2 = src[w * 4 + 2].toInt() and 0xFF
        val b3 = src[w * 4 + 3].toInt() and 0xFF
        dst[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
}
private val writeU32Into: (IntArray, Int, Int, UByteArray) -> Unit = { src, start, end, dst ->
    for (idx in start until end) {
        val w = idx - start
        val word = src[idx]
        dst[w * 4] = (word and 0xFF).toUByte()
        dst[w * 4 + 1] = ((word ushr 8) and 0xFF).toUByte()
        dst[w * 4 + 2] = ((word ushr 16) and 0xFF).toUByte()
        dst[w * 4 + 3] = ((word ushr 24) and 0xFF).toUByte()
    }
}

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    var acc = f32x4Splat(0.0f)
    for ((x, y) in xs.zip(ys)) {
        val x1234 = v128LoadU(x.qs, 0)
        val x12 = i8x16Sub(v128And(x1234, u8x16Splat(0x0F)), i8x16Splat(8))
        val x34 = i8x16Sub(u8x16Shr(x1234, 4), i8x16Splat(8))

        val x1 = i16x8ExtendLowI8x16(x12)
        val y1 = i16x8LoadExtendI8x8(y.qs, 0)
        var sumXy = i32x4DotI16x8(x1, y1)

        val x2 = i16x8ExtendHighI8x16(x12)
        val y2 = i16x8LoadExtendI8x8(y.qs, 8)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x2, y2))

        val x3 = i16x8ExtendLowI8x16(x34)
        val y3 = i16x8LoadExtendI8x8(y.qs, 16)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x3, y3))

        val x4 = i16x8ExtendHighI8x16(x34)
        val y4 = i16x8LoadExtendI8x8(y.qs, 24)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x4, y4))

        val sumXyF = f32x4ConvertI32x4(sumXy)

        // f32x4RelaxedMadd is nightly only.
        val d = f32x4Splat(x.d.toFloat() * y.d.toFloat())
        val scaled = f32x4Mul(sumXyF, d)
        acc = f32x4Add(acc, scaled)
    }
    val res = f32x4ExtractLane(acc, 0) +
        f32x4ExtractLane(acc, 1) +
        f32x4ExtractLane(acc, 2) +
        f32x4ExtractLane(acc, 3)
    return res
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    var acc = f32x4Splat(0.0f)
    for ((x, y) in xs.zip(ys)) {
        val x1 = i16x8LoadExtendI8x8(x.qs, 0)
        val y1 = i16x8LoadExtendI8x8(y.qs, 0)
        var sumXy = i32x4DotI16x8(x1, y1)

        val x2 = i16x8LoadExtendI8x8(x.qs, 8)
        val y2 = i16x8LoadExtendI8x8(y.qs, 8)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x2, y2))

        val x3 = i16x8LoadExtendI8x8(x.qs, 16)
        val y3 = i16x8LoadExtendI8x8(y.qs, 16)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x3, y3))

        val x4 = i16x8LoadExtendI8x8(x.qs, 24)
        val y4 = i16x8LoadExtendI8x8(y.qs, 24)
        sumXy = i32x4Add(sumXy, i32x4DotI16x8(x4, y4))

        val sumXyF = f32x4ConvertI32x4(sumXy)

        // f32x4RelaxedMadd is nightly only.
        val d = f32x4Splat(x.d.toFloat() * y.d.toFloat())
        val scaled = f32x4Mul(sumXyF, d)
        acc = f32x4Add(acc, scaled)
    }
    val res = f32x4ExtractLane(acc, 0) +
        f32x4ExtractLane(acc, 1) +
        f32x4ExtractLane(acc, 2) +
        f32x4ExtractLane(acc, 3)
    return res
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    var sumf = f32x4Splat(0.0f)
    for ((x, y) in xs.zip(ys)) {
        var q2Off = 0
        var q8Off = 0
        val sc = x.scales

        var summs = i32x4Splat(0)
        for (i in 0 until (QKK / 16) step 4) {
            val bsums = i32x4LoadExtendI16x4(y.bsums, i)
            val scales = i32x4Shr(
                i32x4Of(
                    sc[i].toInt() and 0xFF,
                    sc[i + 1].toInt() and 0xFF,
                    sc[i + 2].toInt() and 0xFF,
                    sc[i + 3].toInt() and 0xFF,
                ),
                4,
            )
            summs = i32x4Add(summs, i32x4Mul(bsums, scales))
        }
        val summsF = f32x4ConvertI32x4(summs)

        val dall = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

        var isum = i32x4Splat(0)
        var isIdx = 0
        for (block in 0 until (QKK / 128)) {
            var shift = 0
            for (sub in 0 until 4) {
                val d0 = sc[isIdx].toInt() and 0xF
                isIdx += 1
                var isuml0 = i16x8Splat(0)
                for (l in 0 until 16 step 8) {
                    val q8Vec = i16x8LoadExtendI8x8(y.qs, q8Off + l)
                    val q2Vec = v128AndI16(i16x8Shr(i16x8LoadExtendU8x8(x.qs, q2Off + l), shift), i16x8Splat(3))
                    isuml0 = i16x8Add(isuml0, i16x8Mul(q2Vec, q8Vec))
                }
                val dd0 = i32x4Splat(d0)
                isum = i32x4Add(isum, i32x4Mul(i32x4ExtendLowI16x8(isuml0), dd0))
                isum = i32x4Add(isum, i32x4Mul(i32x4ExtendHighI16x8(isuml0), dd0))

                val d1 = sc[isIdx].toInt() and 0xF
                isIdx += 1
                var isuml1 = i16x8Splat(0)
                for (l in 16 until 32 step 8) {
                    val q8Vec = i16x8LoadExtendI8x8(y.qs, q8Off + l)
                    val q2Vec = v128AndI16(i16x8Shr(i16x8LoadExtendU8x8(x.qs, q2Off + l), shift), i16x8Splat(3))
                    isuml1 = i16x8Add(isuml1, i16x8Mul(q2Vec, q8Vec))
                }
                val dd1 = i32x4Splat(d1)
                isum = i32x4Add(isum, i32x4Mul(i32x4ExtendLowI16x8(isuml1), dd1))
                isum = i32x4Add(isum, i32x4Mul(i32x4ExtendHighI16x8(isuml1), dd1))
                shift += 2
                // adjust the indexing
                q8Off += 32
            }
            // adjust the indexing
            q2Off += 32
        }
        val isumF = f32x4ConvertI32x4(isum)
        sumf = f32x4Add(
            sumf,
            f32x4Sub(
                f32x4Mul(isumF, f32x4Splat(dall)),
                f32x4Mul(summsF, f32x4Splat(dmin)),
            ),
        )
    }
    val res = f32x4ExtractLane(sumf, 0) +
        f32x4ExtractLane(sumf, 1) +
        f32x4ExtractLane(sumf, 2) +
        f32x4ExtractLane(sumf, 3)
    return res
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ4kQ8k: $n is not divisible by $QKK"
    }
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val utmp = IntArray(4)
    val scales = UByteArray(8)
    val mins = UByteArray(8)

    val aux8 = UByteArray(QKK)
    var sums = f32x4Splat(0.0f)
    for ((y, x) in ys.zip(xs)) {
        val q4 = x.qs
        val q8 = y.qs

        for (j in 0 until QKK / 64) {
            val q41 = v128LoadU(q4, 32 * j)
            val q42 = v128LoadU(q4, 32 * j + 16)
            v128StoreU(aux8, 64 * j, v128And(q41, u8x16Splat(0x0F)))
            v128StoreU(aux8, 64 * j + 16, v128And(q42, u8x16Splat(0x0F)))
            v128StoreU(aux8, 64 * j + 32, u8x16Shr(q41, 4))
            v128StoreU(aux8, 64 * j + 48, u8x16Shr(q42, 4))
        }

        readU32Into(x.scales, utmp, 3)

        utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        // extract scales and mins
        writeU32Into(utmp, 0, 2, scales)
        writeU32Into(utmp, 2, 4, mins)

        var sumi = i32x4Splat(0)
        for (j in 0 until QKK / 16 step 4) {
            val bsums = i32x4LoadExtendI16x4(y.bsums, j)
            val m1 = mins[j / 2].toInt()
            val m2 = mins[j / 2 + 1].toInt()
            val minsVec = i32x4Of(m1, m1, m2, m2)
            sumi = i32x4Add(sumi, i32x4Mul(bsums, minsVec))
        }

        var aux32 = i32x4Splat(0)
        for ((scaleI, scaleByte) in scales.withIndex()) {
            val scale = i32x4Splat(scaleByte.toInt())
            for (j in 0 until 4) {
                val i = 32 * scaleI + 8 * j
                val q8Vec = i16x8LoadExtendI8x8(q8, i)
                val aux8Vec = i16x8LoadExtendU8x8(aux8, i)
                val aux16 = i16x8Mul(q8Vec, aux8Vec)
                aux32 = i32x4Add(aux32, i32x4Mul(scale, i32x4ExtendLowI16x8(aux16)))
                aux32 = i32x4Add(aux32, i32x4Mul(scale, i32x4ExtendHighI16x8(aux16)))
            }
        }
        val aux32F = f32x4ConvertI32x4(aux32)
        val d = f32x4Splat(x.d.toFloat() * y.d)
        sums = f32x4Add(sums, f32x4Mul(aux32F, d))
        val dmin = f32x4Splat(x.dmin.toFloat() * y.d)
        val sumiF = f32x4ConvertI32x4(sumi)
        sums = f32x4Sub(sums, f32x4Mul(sumiF, dmin))
    }
    val res = f32x4ExtractLane(sums, 0) +
        f32x4ExtractLane(sums, 1) +
        f32x4ExtractLane(sums, 2) +
        f32x4ExtractLane(sums, 3)
    return res
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    val aux8 = ByteArray(QKK)
    var sums = f32x4Splat(0.0f)

    for ((x, y) in xs.zip(ys)) {
        val q4 = x.ql
        val qh = x.qh
        val q8 = y.qs
        var aux32 = f32x4Splat(0.0f)

        for (j in 0 until QKK step 128) {
            val q4Off = j / 2
            val qhOff = j / 4
            for (l in 0 until 32 step 16) {
                val a80 = v128Or(
                    v128And(v128LoadU(q4, q4Off + l), u8x16Splat(0xF)),
                    u8x16Shl(
                        v128And(v128LoadU(qh, qhOff + l), u8x16Splat(3)),
                        4,
                    ),
                )
                val a8Low0 = i16x8Sub(i16x8ExtendLowU8x16(a80), i16x8Splat(32))
                val a8High0 = i16x8Sub(i16x8ExtendHighU8x16(a80), i16x8Splat(32))
                v128Store(aux8, j + l, i8x16NarrowI16x8(a8Low0, a8High0))

                val a81 = v128Or(
                    v128And(v128LoadU(q4, q4Off + l + 32), u8x16Splat(0xF)),
                    u8x16Shl(
                        v128And(
                            u8x16Shr(v128LoadU(qh, qhOff + l), 2),
                            u8x16Splat(3),
                        ),
                        4,
                    ),
                )
                val a8Low1 = i16x8Sub(i16x8ExtendLowU8x16(a81), i16x8Splat(32))
                val a8High1 = i16x8Sub(i16x8ExtendHighU8x16(a81), i16x8Splat(32))
                v128Store(aux8, j + l + 32, i8x16NarrowI16x8(a8Low1, a8High1))

                val a82 = v128Or(
                    u8x16Shr(v128LoadU(q4, q4Off + l), 4),
                    u8x16Shl(
                        v128And(
                            u8x16Shr(v128LoadU(qh, qhOff + l), 4),
                            u8x16Splat(3),
                        ),
                        4,
                    ),
                )
                val a8Low2 = i16x8Sub(i16x8ExtendLowU8x16(a82), i16x8Splat(32))
                val a8High2 = i16x8Sub(i16x8ExtendHighU8x16(a82), i16x8Splat(32))
                v128Store(aux8, j + l + 64, i8x16NarrowI16x8(a8Low2, a8High2))

                val a83 = v128Or(
                    u8x16Shr(v128LoadU(q4, q4Off + l + 32), 4),
                    u8x16Shl(
                        v128And(
                            u8x16Shr(v128LoadU(qh, qhOff + l), 6),
                            u8x16Splat(3),
                        ),
                        4,
                    ),
                )
                val a8Low3 = i16x8Sub(i16x8ExtendLowU8x16(a83), i16x8Splat(32))
                val a8High3 = i16x8Sub(i16x8ExtendHighU8x16(a83), i16x8Splat(32))
                v128Store(aux8, j + l + 96, i8x16NarrowI16x8(a8Low3, a8High3))
            }
        }

        for ((j, scaleByte) in x.scales.withIndex()) {
            val scale = f32x4Splat(scaleByte.toFloat())
            for (offset in intArrayOf(0, 8)) {
                val aux16 = i16x8Mul(
                    i16x8LoadExtendI8x8(q8, 16 * j + offset),
                    i16x8LoadExtendI8x8(aux8, 16 * j + offset),
                )
                aux32 = f32x4Add(
                    aux32,
                    f32x4Mul(f32x4ConvertI32x4(i32x4ExtendLowI16x8(aux16)), scale),
                )
                aux32 = f32x4Add(
                    aux32,
                    f32x4Mul(f32x4ConvertI32x4(i32x4ExtendHighI16x8(aux16)), scale),
                )
            }
        }

        val d = f32x4Splat(x.d.toFloat() * y.d)
        sums = f32x4Add(sums, f32x4Mul(aux32, d))
    }
    val res = f32x4ExtractLane(sums, 0) +
        f32x4ExtractLane(sums, 1) +
        f32x4ExtractLane(sums, 2) +
        f32x4ExtractLane(sums, 3)
    return res
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    var acc = f32x4Splat(0.0f)
    for ((x, y) in xs.zip(ys)) {
        val xQs = x.qs
        val yQs = y.qs
        var sumi = i32x4Splat(0)
        for (j in 0 until QKK step 8) {
            val xsVec = i16x8LoadExtendI8x8(xQs, j)
            val ysVec = i16x8LoadExtendI8x8(yQs, j)
            val sumXy = i32x4DotI16x8(xsVec, ysVec)
            sumi = i32x4Add(sumi, sumXy)
        }
        val d = f32x4Splat(x.d * y.d)
        acc = f32x4Add(acc, f32x4Mul(f32x4ConvertI32x4(sumi), d))
    }
    val res = f32x4ExtractLane(acc, 0) +
        f32x4ExtractLane(acc, 1) +
        f32x4ExtractLane(acc, 2) +
        f32x4ExtractLane(acc, 3)
    return res
}
