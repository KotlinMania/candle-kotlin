// port-lint: source quantized/avx.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.avx

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.quantized.kquants.BlockQ2K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ3K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ5K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.QK8_0
import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.half.f16
import io.github.kotlinmania.rustlang.core.arch.x8664.*
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr

internal fun sumI16PairsFloat(x: M256i): M256 {
    val ones = mm256Set1Epi16(1)
    val summedPairs = mm256MaddEpi16(ones, x)
    return mm256Cvtepi32Ps(summedPairs)
}

internal fun mulSumUs8PairsFloat(ax: M256i, sy: M256i): M256 {
    val dot = mm256MaddubsEpi16(ax, sy)
    return sumI16PairsFloat(dot)
}

internal fun hsumFloat8(x: M256): Float {
    val res = mm256Extractf128Ps(x, 1)
    val res1 = mmAddPs(res, mm256Castps256Ps128(x))
    val res2 = mmAddPs(res1, mmMovehlPs(res1, res1))
    val res3 = mmAddSs(res2, mmMovehdupPs(res2))
    return mmCvtssF32(res3)
}

/**
 * # Safety
 * `rsi` must point to at least 16 readable bytes.
 */
internal fun bytesFromNibbles32(rsi: ConstPtr): M256i {
    val tmp = mmLoaduSi128(ConstPtr(rsi.rawAddress, rsi.rawMeta))
    val bytes = mm256Insertf128Si256(mm256Castsi128Si256(tmp), mmSrliEpi16(tmp, 4), 1)
    val lowMask = mm256Set1Epi8(0xF)
    return mm256AndSi256(lowMask, bytes)
}

internal fun mulSumI8PairsFloat(x: M256i, y: M256i): M256 {
    val ax = mm256SignEpi8(x, x)
    val sy = mm256SignEpi8(y, x)
    return mulSumUs8PairsFloat(ax, sy)
}

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q4_0_q8_0: $n is not divisible by $qk80" }
    return run {
        var acc = mm256SetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = mm256Set1Ps(f16.toF32(x.d) * f16.toF32(y.d))
            val bx = bytesFromNibbles32(x.qs.asPtr())
            val off = mm256Set1Epi8(8)
            val bx1 = mm256SubEpi8(bx, off)
            val by = mm256LoaduSi256(constPtrFromRustCast(y.qs.asPtr()))
            val q = mulSumI8PairsFloat(bx1, by)
            acc = mm256FmaddPs(d, q, acc)
        }
        hsumFloat8(acc)
    }
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q8_0_q8_0: $n is not divisible by $qk80" }
    return run {
        var acc = mm256SetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = mm256Set1Ps(f16.toF32(x.d) * f16.toF32(y.d))
            val bx = mm256LoaduSi256(constPtrFromRustCast(x.qs.asPtr()))
            val by = mm256LoaduSi256(constPtrFromRustCast(y.qs.asPtr()))
            val q = mulSumI8PairsFloat(bx, by)
            acc = mm256FmaddPs(d, q, acc)
        }
        hsumFloat8(acc)
    }
}

/**
 * # Safety
 * `i` must be less than 8.
 */
public fun getScaleShuffle(i: Int): M128i {
    val K_SHUFFLE: UByteArray = ubyteArrayOf(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 7u, 8u, 8u, 8u, 8u, 8u, 8u, 8u, 8u, 9u, 9u, 9u, 9u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 11u, 11u, 11u, 11u, 12u, 12u, 12u, 12u, 12u, 12u, 12u, 12u, 13u, 13u, 13u, 13u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 14u, 14u, 14u, 14u, 15u, 15u, 15u, 15u, 15u, 15u, 15u, 15u)
    return mmLoaduSi128(ConstPtr(((constPtrFromRustCast(K_SHUFFLE.asPtr()))).rawAddress + ((i).toLong() * (ObjectSlotLayout<M128i>()).strideBytes)))
}

/**
 * # Safety
 * `i` must be less than 8.
 */
public fun getScaleShuffleK4(i: Int): M256i {
    val K_SHUFFLE: UByteArray = ubyteArrayOf(0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u)
    return mm256LoaduSi256(ConstPtr(((constPtrFromRustCast(K_SHUFFLE.asPtr()))).rawAddress + ((i).toLong() * (ObjectSlotLayout<M256i>()).strideBytes)))
}

/**
 * # Safety
 * `i` must be less than 4.
 */
public fun getScaleShuffleQ3k(i: Int): M256i {
    val K_SHUFFLE: UByteArray = ubyteArrayOf(0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 2u, 3u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 4u, 5u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 6u, 7u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 8u, 9u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 10u, 11u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 12u, 13u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u, 14u, 15u)
    return mm256LoaduSi256(ConstPtr(((constPtrFromRustCast(K_SHUFFLE.asPtr()))).rawAddress + ((i).toLong() * (ObjectSlotLayout<M256i>()).strideBytes)))
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q6k_8k: $n is not divisible by $qkK" }
    return run {
        val m4 = mm256Set1Epi8(0xF)
        val m2 = mm256Set1Epi8(3)
        val m32s = mm256Set1Epi8(32)
        var acc = mm256SetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            var q4 = x.ql.asPtr()
            var qh = x.qh.asPtr()
            var q8 = y.qs.asPtr()
            val scales = mmLoaduSi128(constPtrFromRustCast(x.scales.asPtr()))
            var sumi = mm256SetzeroSi256()
            for (j in 0 until QK_K / 128) {
                val isMatch = j * 4
                val scale0 = mmShuffleEpi8(scales, getScaleShuffle(isMatch))
                val scale1 = mmShuffleEpi8(scales, getScaleShuffle(isMatch + 1))
                val scale2 = mmShuffleEpi8(scales, getScaleShuffle(isMatch + 2))
                val scale3 = mmShuffleEpi8(scales, getScaleShuffle(isMatch + 3))
                val q4bits1 = mm256LoaduSi256(constPtrFromRustCast(q4))
                q4 = q4.add(32)
                val q4bits2 = mm256LoaduSi256(constPtrFromRustCast(q4))
                q4 = q4.add(32)
                val q4bitsH = mm256LoaduSi256(constPtrFromRustCast(qh))
                qh = qh.add(32)
                val q4h0 = mm256SlliEpi16(mm256AndSi256(q4bitsH, m2), 4)
                val q4h1 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 2), m2), 4)
                val q4h2 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 4), m2), 4)
                val q4h3 = mm256SlliEpi16(mm256AndSi256(mm256SrliEpi16(q4bitsH, 6), m2), 4)
                val q40 = mm256OrSi256(mm256AndSi256(q4bits1, m4), q4h0)
                val q41 = mm256OrSi256(mm256AndSi256(q4bits2, m4), q4h1)
                val q42 = mm256OrSi256(mm256AndSi256(mm256SrliEpi16(q4bits1, 4), m4), q4h2)
                val q43 = mm256OrSi256(mm256AndSi256(mm256SrliEpi16(q4bits2, 4), m4), q4h3)
                val q80 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q81 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q82 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q83 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q8s0 = mm256MaddubsEpi16(m32s, q80)
                val q8s1 = mm256MaddubsEpi16(m32s, q81)
                val q8s2 = mm256MaddubsEpi16(m32s, q82)
                val q8s3 = mm256MaddubsEpi16(m32s, q83)
                val p160 = mm256MaddubsEpi16(q40, q80)
                val p161 = mm256MaddubsEpi16(q41, q81)
                val p162 = mm256MaddubsEpi16(q42, q82)
                val p163 = mm256MaddubsEpi16(q43, q83)
                val p1601 = mm256SubEpi16(p160, q8s0)
                val p1611 = mm256SubEpi16(p161, q8s1)
                val p1621 = mm256SubEpi16(p162, q8s2)
                val p1631 = mm256SubEpi16(p163, q8s3)
                val p1602 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale0), p1601)
                val p1612 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale1), p1611)
                val p1622 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale2), p1621)
                val p1632 = mm256MaddEpi16(mm256Cvtepi8Epi16(scale3), p1631)
                sumi = mm256AddEpi32(sumi, mm256AddEpi32(p1602, p1612))
                sumi = mm256AddEpi32(sumi, mm256AddEpi32(p1622, p1632))
            }
            acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
        }
        hsumFloat8(acc)
    }
}

public fun mm256SetM128i(a: M128i, b: M128i): M256i = mm256Insertf128Si256(mm256Castsi128Si256(b), a, 1)

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q2k_q8k: $n is not divisible by $qkK" }
    return run {
        val m3 = mm256Set1Epi8(3)
        val m4 = mmSet1Epi8(0xF)
        var acc = mm256SetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = -y.d * x.dmin.toF32()
            var q2 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            val minsAndScales = mmLoaduSi128(constPtrFromRustCast(x.scales.asPtr()))
            val scales8 = mmAndSi128(minsAndScales, m4)
            val mins8 = mmAndSi128(mmSrliEpi16(minsAndScales, 4), m4)
            val mins = mm256Cvtepi8Epi16(mins8)
            val prod = mm256MaddEpi16(mins, mm256LoaduSi256(constPtrFromRustCast(y.bsums.asPtr())))
            acc = mm256FmaddPs(mm256BroadcastSs(dmin), mm256Cvtepi32Ps(prod), acc)
            val allScales = mm256Cvtepi8Epi16(scales8)
            val lScales = mm256Extracti128Si256(allScales, 0)
            val hScales = mm256Extracti128Si256(allScales, 1)
            val scales = listOf(mm256SetM128i(lScales, lScales), mm256SetM128i(hScales, hScales))
            var sumi = mm256SetzeroSi256()
            for (scale in scales) {
                val q2bits = mm256LoaduSi256(constPtrFromRustCast(q2))
                q2 = q2.add(32)
                val q80 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q81 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q82 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q83 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q20 = mm256AndSi256(q2bits, m3)
                val q21 = mm256AndSi256(mm256SrliEpi16(q2bits, 2), m3)
                val q22 = mm256AndSi256(mm256SrliEpi16(q2bits, 4), m3)
                val q23 = mm256AndSi256(mm256SrliEpi16(q2bits, 6), m3)
                val p0 = mm256MaddubsEpi16(q20, q80)
                val p1 = mm256MaddubsEpi16(q21, q81)
                val p2 = mm256MaddubsEpi16(q22, q82)
                val p3 = mm256MaddubsEpi16(q23, q83)
                val p01 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(0)), p0)
                val p11 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(1)), p1)
                val p21 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(2)), p2)
                val p31 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(3)), p3)
                val p02 = mm256AddEpi32(p01, p11)
                val p22 = mm256AddEpi32(p21, p31)
                sumi = mm256AddEpi32(sumi, mm256AddEpi32(p02, p22))
            }
            acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
        }
        hsumFloat8(acc)
    }
}

internal fun vecDotQ3kQ8k(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x03030303u
    val KMASK2: UInt = 0x0f0f0f0fu
    check(n.isMultipleOf(QK_K)) { "vec_dot_q3k_q8k: $n is not divisible by $qkK" }
    val aux = List(3) { 0u }
    return run {
        val m3 = mm256Set1Epi8(3)
        val mone = mm256Set1Epi8(1)
        val m32 = mmSet1Epi8(32)
        var acc = mm256SetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            var q3 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            LittleEndian.readU32Into(x.scales, aux)
            val scales128 = mmSetEpi32((((aux[1] shr 4) and KMASK2) or (((aux[2] shr 6) and KMASK1) shl 4)).toInt(), (((aux[0] shr 4) and KMASK2) or (((aux[2] shr 4) and KMASK1) shl 4)).toInt(), ((aux[1] and KMASK2) or (((aux[2] shr 2) and KMASK1) shl 4)).toInt(), ((aux[0] and KMASK2) or (((aux[2]) and KMASK1) shl 4)).toInt())
            val scales1281 = mmSubEpi8(scales128, m32)
            val allScales = mm256Cvtepi8Epi16(scales1281)
            val lScales = mm256Extracti128Si256(allScales, 0)
            val hScales = mm256Extracti128Si256(allScales, 1)
            val scales = listOf(mm256SetM128i(lScales, lScales), mm256SetM128i(hScales, hScales))
            // high bit
            val hbits = mm256LoaduSi256(constPtrFromRustCast(x.hmask.asPtr()))
            var sumi = mm256SetzeroSi256()
            for ((j, scale) in scales.iter().mapIndexed { index, item -> Pair(index, item) }) {
                // load low 2 bits
                val q3bits = mm256LoaduSi256(constPtrFromRustCast(q3))
                q3 = q3.add(32)
                // Prepare low and high bits
                // We hardcode the shifts here to avoid loading them into a separate register
                val q3l0 = mm256AndSi256(q3bits, m3)
                val q3h0 =
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 0)), 0)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 4)), 4)
                    }
                val q3h01 = mm256SlliEpi16(q3h0, 2)
                val q3l1 = mm256AndSi256(mm256SrliEpi16(q3bits, 2), m3)
                val q3h1 =
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 1)), 1)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 5)), 5)
                    }
                val q3h11 = mm256SlliEpi16(q3h1, 2)
                val q3l2 = mm256AndSi256(mm256SrliEpi16(q3bits, 4), m3)
                val q3h2 =
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 2)), 2)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 6)), 6)
                    }
                val q3h21 = mm256SlliEpi16(q3h2, 2)
                val q3l3 = mm256AndSi256(mm256SrliEpi16(q3bits, 6), m3)
                val q3h3 =
                    if (j == 0) {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 3)), 3)
                    } else {
                        mm256SrliEpi16(mm256AndnotSi256(hbits, mm256SlliEpi16(mone, 7)), 7)
                    }
                val q3h31 = mm256SlliEpi16(q3h3, 2)
                // load Q8 quants
                val q80 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q81 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q82 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q83 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we
                // can use _mm256_maddubs_epi16, and then subtract. The high bit part has the 2
                // already subtracted (and so, it is zero if the high bit was not set, and 2 if the
                // high bit was set)
                val q8s0 = mm256MaddubsEpi16(q3h01, q80)
                val q8s1 = mm256MaddubsEpi16(q3h11, q81)
                val q8s2 = mm256MaddubsEpi16(q3h21, q82)
                val q8s3 = mm256MaddubsEpi16(q3h31, q83)
                val p160 = mm256MaddubsEpi16(q3l0, q80)
                val p161 = mm256MaddubsEpi16(q3l1, q81)
                val p162 = mm256MaddubsEpi16(q3l2, q82)
                val p163 = mm256MaddubsEpi16(q3l3, q83)
                val p1601 = mm256SubEpi16(p160, q8s0)
                val p1611 = mm256SubEpi16(p161, q8s1)
                val p1621 = mm256SubEpi16(p162, q8s2)
                val p1631 = mm256SubEpi16(p163, q8s3)
                // multiply with scales
                val p1602 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(0)), p1601)
                val p1612 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(1)), p1611)
                val p1622 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(2)), p1621)
                val p1632 = mm256MaddEpi16(mm256ShuffleEpi8(scale, getScaleShuffleQ3k(3)), p1631)
                // accumulate
                val p1603 = mm256AddEpi32(p1602, p1612)
                val p1623 = mm256AddEpi32(p1622, p1632)
                sumi = mm256AddEpi32(sumi, mm256AddEpi32(p1603, p1623))
            }
            // multiply with block scale and accumulate
            acc = mm256FmaddPs(mm256BroadcastSs(d), mm256Cvtepi32Ps(sumi), acc)
        }
        hsumFloat8(acc)
    }
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    check(n.isMultipleOf(QK_K)) { "vec_dot_q4k_q8k: $n is not divisible by $qkK" }
    val utmp = List(4) { 0u }
    return run {
        val m4 = mm256Set1Epi8(0xF)
        var acc = mm256SetzeroPs()
        var accM = mmSetzeroPs()
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = -y.d * x.dmin.toF32()
            LittleEndian.readU32Into(x.scales, ((utmp).slice(0 until 3)))
            utmp[3] = ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)
            val uaux = utmp[1] and KMASK1
            utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
            utmp[2] = uaux
            utmp[0] = utmp[0] and KMASK1
            var q4 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            val minsAndScales = mm256Cvtepu8Epi16(mmSetEpi32(utmp[3].toInt(), utmp[2].toInt(), utmp[1].toInt(), utmp[0].toInt()))
            val q8sums = mm256LoaduSi256(constPtrFromRustCast(y.bsums.asPtr()))
            val q8s = mmHaddEpi16(mm256Extracti128Si256(q8sums, 0), mm256Extracti128Si256(q8sums, 1))
            val prod = mmMaddEpi16(mm256Extracti128Si256(minsAndScales, 1), q8s)
            accM = mmFmaddPs(mmSet1Ps(dmin), mmCvtepi32Ps(prod), accM)
            val sc128 = mm256Extracti128Si256(minsAndScales, 0)
            val scales = mm256SetM128i(sc128, sc128)
            var sumi = mm256SetzeroSi256()
            for (j in 0 until QK_K / 64) {
                val scaleL = mm256ShuffleEpi8(scales, getScaleShuffleK4(2 * j))
                val scaleH = mm256ShuffleEpi8(scales, getScaleShuffleK4(2 * j + 1))
                val q4bits = mm256LoaduSi256(constPtrFromRustCast(q4))
                q4 = q4.add(32)
                val q4l = mm256AndSi256(q4bits, m4)
                val q4h = mm256AndSi256(mm256SrliEpi16(q4bits, 4), m4)
                val q8l = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val p16l = mm256MaddubsEpi16(q4l, q8l)
                val p16l1 = mm256MaddEpi16(scaleL, p16l)
                sumi = mm256AddEpi32(sumi, p16l1)
                val q8h = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val p16h = mm256MaddubsEpi16(q4h, q8h)
                val p16h1 = mm256MaddEpi16(scaleH, p16h)
                sumi = mm256AddEpi32(sumi, p16h1)
            }
            val vd = mm256Set1Ps(d)
            acc = mm256FmaddPs(vd, mm256Cvtepi32Ps(sumi), acc)
        }
        val accM1 = mmAddPs(accM, mmMovehlPs(accM, accM))
        val accM2 = mmAddSs(accM1, mmMovehdupPs(accM1))
        hsumFloat8(acc) + mmCvtssF32(accM2)
    }
}

internal fun vecDotQ5kQ8k(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    check(n.isMultipleOf(QK_K)) { "vec_dot_q5k_q8k: $n is not divisible by $qkK" }
    val utmp = List(4) { 0u }
    return run {
        val m4 = mm256Set1Epi8(0xF)
        val mzero = mmSetzeroSi128()
        val mone = mm256Set1Epi8(1)
        var acc = mm256SetzeroPs()
        var summs = 0.0
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = -y.d * x.dmin.toF32()
            LittleEndian.readU32Into(x.scales, ((utmp).slice(0 until 3)))
            utmp[3] = ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)
            val uaux = utmp[1] and KMASK1
            utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
            utmp[2] = uaux
            utmp[0] = utmp[0] and KMASK1
            var q5 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            val minsAndScales = mm256Cvtepu8Epi16(mmSetEpi32(utmp[3].toInt(), utmp[2].toInt(), utmp[1].toInt(), utmp[0].toInt()))
            val q8sums = mm256LoaduSi256(constPtrFromRustCast(y.bsums.asPtr()))
            val q8s = mmHaddEpi16(mm256Extracti128Si256(q8sums, 0), mm256Extracti128Si256(q8sums, 1))
            val prod = mmMaddEpi16(mm256Extracti128Si256(minsAndScales, 1), q8s)
            val hsum = mmHaddEpi32(mmHaddEpi32(prod, mzero), mzero)
            summs += dmin * mmExtractEpi32(hsum, 0).toFloat()
            val sc128 = mm256Extracti128Si256(minsAndScales, 0)
            val scales = mm256SetM128i(sc128, sc128)
            val hbits = mm256LoaduSi256(constPtrFromRustCast(x.qh.asPtr()))
            var hmask = mone
            var sumi = mm256SetzeroSi256()
            for (j in 0 until QK_K / 64) {
                val scale0 = mm256ShuffleEpi8(scales, getScaleShuffleK4(2 * j))
                val scale1 = mm256ShuffleEpi8(scales, getScaleShuffleK4(2 * j + 1))
                val q5bits = mm256LoaduSi256(constPtrFromRustCast(q5))
                q5 = q5.add(32)
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
                val q80 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val q81 = mm256LoaduSi256(constPtrFromRustCast(q8))
                q8 = q8.add(32)
                val p160 = mm256MaddubsEpi16(q50, q80)
                val p161 = mm256MaddubsEpi16(q51, q81)
                val p1601 = mm256MaddEpi16(scale0, p160)
                val p1611 = mm256MaddEpi16(scale1, p161)
                sumi = mm256AddEpi32(sumi, mm256AddEpi32(p1601, p1611))
            }
            val vd = mm256Set1Ps(d)
            acc = mm256FmaddPs(vd, mm256Cvtepi32Ps(sumi), acc)
        }
        hsumFloat8(acc) + summs
    }
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q8k_8k: $n is not divisible by $qkK" }
    return run {
        var acc = mm256SetzeroPs()
        for ((xs1, ys1) in (xs.zip(ys))) {
            var sumi = mm256SetzeroSi256()
            val xQs = xs1.qs.asPtr()
            val yQs = ys1.qs.asPtr()
            for (j in (0 until QK_K).stepBy(32)) {
                val xs2 = mm256LoaduSi256(constPtrFromRustCast(xQs.add(j)))
                val ys2 = mm256LoaduSi256(constPtrFromRustCast(yQs.add(j)))
                val xs0 = mm256Cvtepi8Epi16(mm256Extracti128Si256(xs2, 0))
                val ys0 = mm256Cvtepi8Epi16(mm256Extracti128Si256(ys2, 0))
                sumi = mm256AddEpi32(sumi, mm256MaddEpi16(xs0, ys0))
                val xs1 = mm256Cvtepi8Epi16(mm256Extracti128Si256(xs2, 1))
                val ys1 = mm256Cvtepi8Epi16(mm256Extracti128Si256(ys2, 1))
                sumi = mm256AddEpi32(sumi, mm256MaddEpi16(xs1, ys1))
            }
            val d = mm256Set1Ps(xs1.d * ys1.d)
            acc = mm256FmaddPs(d, mm256Cvtepi32Ps(sumi), acc)
        }
        hsumFloat8(acc)
    }
}
