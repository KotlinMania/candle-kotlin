// port-lint: source quantized/simd128.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.simd128

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.quantized.kquants.BlockQ2K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.QK8_0
import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.half.f16
import io.github.kotlinmania.rustlang.core.arch.wasm32.*
import io.github.kotlinmania.rustlang.core.mem.AddressableArrayLocal
import io.github.kotlinmania.rustlang.core.mem.FixedArrayTypeLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q4_0_q8_0: $n is not divisible by $qk80" }
    return run {
        var acc = f32x4Splat(0.0f)
        for ((x, y) in (xs.zip(ys))) {
            val x1234 = v128Load(constPtrFromRustCast(x.qs.asPtr()))
            val x12 = v128And(x1234, u8x16Splat(0x0F))
            val x121 = i8x16Sub(x12, i8x16Splat(8))
            val x34 = u8x16Shr(x1234, 4)
            val x341 = i8x16Sub(x34, i8x16Splat(8))
            val x1 = i16x8ExtendLowI8x16(x121)
            val y1 = i16x8LoadExtendI8x8(y.qs.asPtr())
            val sumXy = i32x4DotI16x8(x1, y1)
            val x2 = i16x8ExtendHighI8x16(x121)
            val y2 = i16x8LoadExtendI8x8(y.qs.asPtr().add(8))
            val sumXy1 = i32x4Add(sumXy, i32x4DotI16x8(x2, y2))
            val x3 = i16x8ExtendLowI8x16(x341)
            val y3 = i16x8LoadExtendI8x8(y.qs.asPtr().add(16))
            val sumXy2 = i32x4Add(sumXy1, i32x4DotI16x8(x3, y3))
            val x4 = i16x8ExtendHighI8x16(x341)
            val y4 = i16x8LoadExtendI8x8(y.qs.asPtr().add(24))
            val sumXy3 = i32x4Add(sumXy2, i32x4DotI16x8(x4, y4))
            val sumXy4 = f32x4ConvertI32x4(sumXy3)
            // f32x4RelaxedMadd is nightly only.
            val d = f32x4Splat(f16.toF32(x.d) * f16.toF32(y.d))
            val scaled = f32x4Mul(sumXy4, d)
            acc = f32x4Add(acc, scaled)
        }
        val res = f32x4ExtractLane(acc, 0) + f32x4ExtractLane(acc, 1) + f32x4ExtractLane(acc, 2) + f32x4ExtractLane(acc, 3)
        res
    }
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q8_0_q8_0: $n is not divisible by $qk80" }
    return run {
        var acc = f32x4Splat(0.0f)
        for ((x, y) in (xs.zip(ys))) {
            val x1 = i16x8LoadExtendI8x8(x.qs.asPtr())
            val y1 = i16x8LoadExtendI8x8(y.qs.asPtr())
            val sumXy = i32x4DotI16x8(x1, y1)
            val x2 = i16x8LoadExtendI8x8(x.qs.asPtr().add(8))
            val y2 = i16x8LoadExtendI8x8(y.qs.asPtr().add(8))
            val sumXy1 = i32x4Add(sumXy, i32x4DotI16x8(x2, y2))
            val x3 = i16x8LoadExtendI8x8(x.qs.asPtr().add(16))
            val y3 = i16x8LoadExtendI8x8(y.qs.asPtr().add(16))
            val sumXy2 = i32x4Add(sumXy1, i32x4DotI16x8(x3, y3))
            val x4 = i16x8LoadExtendI8x8(x.qs.asPtr().add(24))
            val y4 = i16x8LoadExtendI8x8(y.qs.asPtr().add(24))
            val sumXy3 = i32x4Add(sumXy2, i32x4DotI16x8(x4, y4))
            val sumXy4 = f32x4ConvertI32x4(sumXy3)
            // f32x4RelaxedMadd is nightly only.
            val d = f32x4Splat(f16.toF32(x.d) * f16.toF32(y.d))
            val scaled = f32x4Mul(sumXy4, d)
            acc = f32x4Add(acc, scaled)
        }
        val res = f32x4ExtractLane(acc, 0) + f32x4ExtractLane(acc, 1) + f32x4ExtractLane(acc, 2) + f32x4ExtractLane(acc, 3)
        res
    }
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q2k_q8k: $n is not divisible by $qkK" }
    return run {
        var sumf = f32x4Splat(0.0f)
        for ((x, y) in (xs.zip(ys))) {
            var q2: List<Any?> = x.qs
            var q8: List<Any?> = y.qs
            val sc = x.scales
            var summs = i32x4Splat(0)
            for (i in (0 until (QK_K / 16)).stepBy(4)) {
                val bsums = i32x4LoadExtendI16x4(y.bsums.asPtr().add(i))
                val scales = i32x4Shr(i32x4(sc[i].toInt(), sc[i + 1].toInt(), sc[i + 2].toInt(), sc[i + 3].toInt()), 4)
                summs = i32x4Add(summs, i32x4Mul(bsums, scales))
            }
            val summs1 = f32x4ConvertI32x4(summs)
            val dall = y.d * x.d.toF32()
            val dmin = y.d * x.dmin.toF32()
            var isum = i32x4Splat(0)
            var isMatch = 0
            for (unused in 0 until (QK_K / 128)) {
                var shift = 0
                for (unused1 in 0 until 4) {
                    val d = (sc[isMatch] and 0xF).toInt()
                    isMatch += 1
                    var isuml = i16x8Splat(0)
                    for (l in (0 until 16).stepBy(8)) {
                        val q81 = i16x8LoadExtendI8x8(q8.asPtr().add(l))
                        val q21 = i16x8LoadExtendU8x8(q2.asPtr().add(l))
                        val q22 = v128And(i16x8Shr(q21, shift), i16x8Splat(3))
                        isuml = i16x8Add(isuml, i16x8Mul(q22, q81))
                    }
                    val dd = i32x4Splat(d)
                    isum = i32x4Add(isum, i32x4Mul(i32x4ExtendLowI16x8(isuml), dd))
                    isum = i32x4Add(isum, i32x4Mul(i32x4ExtendHighI16x8(isuml), dd))
                    val d1 = (sc[isMatch] and 0xF).toInt()
                    isMatch += 1
                    var isuml1 = i16x8Splat(0)
                    for (l1 in (16 until 32).stepBy(8)) {
                        val q82 = i16x8LoadExtendI8x8(q8.asPtr().add(l1))
                        val q23 = i16x8LoadExtendU8x8(q2.asPtr().add(l1))
                        val q24 = v128And(i16x8Shr(q23, shift), i16x8Splat(3))
                        isuml1 = i16x8Add(isuml1, i16x8Mul(q24, q82))
                    }
                    val dd1 = i32x4Splat(d1)
                    isum = i32x4Add(isum, i32x4Mul(i32x4ExtendLowI16x8(isuml1), dd1))
                    isum = i32x4Add(isum, i32x4Mul(i32x4ExtendHighI16x8(isuml1), dd1))
                    shift += 2
                    // adjust the indexing
                    q8 = ((q8).slice(32 until q8.size))
                }
                // adjust the indexing
                q2 = ((q2).slice(32 until q2.size))
            }
            val isum1 = f32x4ConvertI32x4(isum)
            sumf = f32x4Add(sumf, f32x4Sub(f32x4Mul(isum1, f32x4Splat(dall)), f32x4Mul(summs1, f32x4Splat(dmin))))
        }
        val sumf1 = f32x4ExtractLane(sumf, 0) + f32x4ExtractLane(sumf, 1) + f32x4ExtractLane(sumf, 2) + f32x4ExtractLane(sumf, 3)
        sumf1
    }
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    check(n.isMultipleOf(QK_K)) { "vec_dot_q4k_q8k: $n is not divisible by $qkK" }
    val utmp: UIntArray = UIntArray(4) { 0u }
    val scales: UByteArray = UByteArray(8) { 0u.toUByte() }
    val mins: UByteArray = UByteArray(8) { 0u.toUByte() }
    val localStorageL192C4 = AddressableArrayLocal(UByteArray(QK_K) { 0u.toUByte() }, FixedArrayTypeLayout(PrimitiveLayouts.U8, (QK_K).toInt()))
    var aux8 by localStorageL192C4
    var sums = f32x4Splat(0.0f)
    return run {
        for ((y, x) in (ys.zip(xs))) {
            val q4 = x.qs
            val q8 = y.qs
            for (j in 0 until QK_K / 64) {
                val q41 = v128Load(constPtrFromRustCast(q4.asPtr().add(32 * j)))
                val q42 = v128Load(constPtrFromRustCast(q4.asPtr().add(32 * j + 16)))
                v128Store(MutPtr(MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), v128And(q41, u8x16Splat(0x0F)))
                v128Store(MutPtr(MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 16).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 16).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), v128And(q42, u8x16Splat(0x0F)))
                v128Store(MutPtr(MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 32).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 32).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), u8x16Shr(q41, 4))
                v128Store(MutPtr(MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 48).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, MutPtr((localStorageL192C4.elementMutPtr(0)).rawAddress + ((64 * j + 48).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta), u8x16Shr(q42, 4))
            }
            LittleEndian.readU32Into(x.scales, ((utmp).slice(0 until 3)))
            utmp[3] = ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)
            val uaux = utmp[1] and KMASK1
            utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
            utmp[2] = uaux
            utmp[0] = utmp[0] and KMASK1
            // extract scales and mins
            LittleEndian.writeU32Into(((utmp).slice(0 until 2)), scales)
            LittleEndian.writeU32Into(((utmp).slice(2 until 4)), mins)
            var sumi = i32x4Splat(0)
            for (j1 in (0 until QK_K / 16).stepBy(4)) {
                val bsums = i32x4LoadExtendI16x4(y.bsums.asPtr().add(j1))
                val (m1, m2) = Pair(mins[j1 / 2].toInt(), mins[j1 / 2 + 1].toInt())
                val mins1 = i32x4(m1, m1, m2, m2)
                sumi = i32x4Add(sumi, i32x4Mul(bsums, mins1))
            }
            var aux32 = i32x4Splat(0)
            for ((scaleI, scale) in scales.enumerate()) {
                val scale1 = i32x4Splat(scale.toInt())
                for (j2 in 0 until 4) {
                    val i = 32 * scaleI + 8 * j2
                    val q81 = i16x8LoadExtendI8x8(q8.asPtr().add(i))
                    val aux81 = i16x8LoadExtendU8x8(ConstPtr((localStorageL192C4.elementConstPtr(0)).rawAddress + ((i).toLong() * (PrimitiveLayouts.U8).strideBytes)))
                    val aux16 = i16x8Mul(q81, aux81)
                    aux32 = i32x4Add(aux32, i32x4Mul(scale1, i32x4ExtendLowI16x8(aux16)))
                    aux32 = i32x4Add(aux32, i32x4Mul(scale1, i32x4ExtendHighI16x8(aux16)))
                }
            }
            val aux321 = f32x4ConvertI32x4(aux32)
            val d = f32x4Splat(x.d.toF32() * y.d)
            sums = f32x4Add(sums, f32x4Mul(aux321, d))
            val dmin = x.dmin.toF32() * y.d
            val dmin1 = f32x4Splat(dmin)
            val sumi1 = f32x4ConvertI32x4(sumi)
            sums = f32x4Sub(sums, f32x4Mul(sumi1, dmin1))
        }
        val sums1 = f32x4ExtractLane(sums, 0) + f32x4ExtractLane(sums, 1) + f32x4ExtractLane(sums, 2) + f32x4ExtractLane(sums, 3)
        sums1
    }
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q6k_q8k: $n is not divisible by $qkK" }
    val localStorageL274C4 = AddressableArrayLocal(ByteArray(QK_K) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I8, (QK_K).toInt()))
    var aux8 by localStorageL274C4
    return run {
        var sums = f32x4Splat(0.0f)
        for ((x, y) in (xs.zip(ys))) {
            val q4 = x.ql
            val qh = x.qh
            val q8 = y.qs
            var aux32 = f32x4Splat(0.0f)
            for (j in (0 until QK_K).stepBy(128)) {
                val aux81 = MutPtr((localStorageL274C4.elementMutPtr(0)).rawAddress + ((j).toLong() * (PrimitiveLayouts.I8).strideBytes))
                val q41 = q4.asPtr().add(j / 2)
                val qh1 = qh.asPtr().add(j / 4)
                for (l in (0 until 32).stepBy(16)) {
                    // aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                    val a8 = v128Or(v128And(v128Load(constPtrFromRustCast(q41.add(l))), u8x16Splat(0xF)), u8x16Shl(v128And(v128Load(constPtrFromRustCast(qh1.add(l))), u8x16Splat(3)), 4))
                    val a8Low = i16x8Sub(i16x8ExtendLowU8x16(a8), i16x8Splat(32))
                    val a8High = i16x8Sub(i16x8ExtendHighU8x16(a8), i16x8Splat(32))
                    v128Store(mutPtrFromRustCast(aux81.add(l)), i8x16NarrowI16x8(a8Low, a8High))
                    // aux8[l + 32] =
                    // (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                    val a81 = v128Or(v128And(v128Load(constPtrFromRustCast(q41.add(l + 32))), u8x16Splat(0xF)), u8x16Shl(v128And(u8x16Shr(v128Load(constPtrFromRustCast(qh1.add(l))), 2), u8x16Splat(3)), 4))
                    val a8Low1 = i16x8Sub(i16x8ExtendLowU8x16(a81), i16x8Splat(32))
                    val a8High1 = i16x8Sub(i16x8ExtendHighU8x16(a81), i16x8Splat(32))
                    v128Store(mutPtrFromRustCast(aux81.add(l + 32)), i8x16NarrowI16x8(a8Low1, a8High1))
                    // aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                    val a82 = v128Or(u8x16Shr(v128Load(constPtrFromRustCast(q41.add(l))), 4), u8x16Shl(v128And(u8x16Shr(v128Load(constPtrFromRustCast(qh1.add(l))), 4), u8x16Splat(3)), 4))
                    val a8Low2 = i16x8Sub(i16x8ExtendLowU8x16(a82), i16x8Splat(32))
                    val a8High2 = i16x8Sub(i16x8ExtendHighU8x16(a82), i16x8Splat(32))
                    v128Store(mutPtrFromRustCast(aux81.add(l + 64)), i8x16NarrowI16x8(a8Low2, a8High2))
                    // aux8[l + 96] =
                    // (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                    val a83 = v128Or(u8x16Shr(v128Load(constPtrFromRustCast(q41.add(l + 32))), 4), u8x16Shl(v128And(u8x16Shr(v128Load(constPtrFromRustCast(qh1.add(l))), 6), u8x16Splat(3)), 4))
                    val a8Low3 = i16x8Sub(i16x8ExtendLowU8x16(a83), i16x8Splat(32))
                    val a8High3 = i16x8Sub(i16x8ExtendHighU8x16(a83), i16x8Splat(32))
                    v128Store(mutPtrFromRustCast(aux81.add(l + 96)), i8x16NarrowI16x8(a8Low3, a8High3))
                }
            }
            for ((j1, scale) in x.scales.iter().enumerate()) {
                val scale1 = f32x4Splat(scale.toFloat())
                for (offset in listOf(0, 8)) {
                    val aux16 = i16x8Mul(i16x8LoadExtendI8x8(q8.asPtr().add(16 * j1 + offset)), i16x8LoadExtendI8x8(ConstPtr((localStorageL274C4.elementConstPtr(0)).rawAddress + ((16 * j1 + offset).toLong() * (PrimitiveLayouts.I8).strideBytes))))
                    aux32 = f32x4Add(aux32, f32x4Mul(f32x4ConvertI32x4(i32x4ExtendLowI16x8(aux16)), scale1))
                    aux32 = f32x4Add(aux32, f32x4Mul(f32x4ConvertI32x4(i32x4ExtendHighI16x8(aux16)), scale1))
                }
            }
            val d = f32x4Splat(x.d.toF32() * y.d)
            sums = f32x4Add(sums, f32x4Mul(aux32, d))
        }
        val sums1 = f32x4ExtractLane(sums, 0) + f32x4ExtractLane(sums, 1) + f32x4ExtractLane(sums, 2) + f32x4ExtractLane(sums, 3)
        sums1
    }
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q8k_q8k: $n is not divisible by $qkK" }
    return run {
        var acc = f32x4Splat(0.0f)
        for ((xs1, ys1) in (xs.zip(ys))) {
            val xQs = xs1.qs.asPtr()
            val yQs = ys1.qs.asPtr()
            var sumi = i32x4Splat(0)
            for (j in (0 until QK_K).stepBy(8)) {
                val xs2 = i16x8LoadExtendI8x8(xQs.add(j))
                val ys2 = i16x8LoadExtendI8x8(yQs.add(j))
                val sumXy = i32x4DotI16x8(xs2, ys2)
                sumi = i32x4Add(sumi, sumXy)
            }
            val d = f32x4Splat(xs1.d * ys1.d)
            acc = f32x4Add(acc, f32x4Mul(f32x4ConvertI32x4(sumi), d))
        }
        val res = f32x4ExtractLane(acc, 0) + f32x4ExtractLane(acc, 1) + f32x4ExtractLane(acc, 2) + f32x4ExtractLane(acc, 3)
        res
    }
}
