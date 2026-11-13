// port-lint: source quantized/neon.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.cpu.features.get
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
import io.github.kotlinmania.candle.quantized.kquants.Qk80
import io.github.kotlinmania.candle.quantized.kquants.QkK
import io.github.kotlinmania.candle.quantized.repack.BlockQ40x4
import io.github.kotlinmania.candle.quantized.repack.BlockQ4Kx8
import io.github.kotlinmania.candle.quantized.repack.BlockQ5Kx8
import io.github.kotlinmania.candle.quantized.repack.BlockQ6Kx8
import io.github.kotlinmania.candle.quantized.repack.BlockQ80x4
import io.github.kotlinmania.candle.quantized.repack.BlockQ8Kx4
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.arch.aarch64.*
import io.github.kotlinmania.rustlang.core.cell.RefCell
import io.github.kotlinmania.rustlang.core.mem.AddressableArrayLocal
import io.github.kotlinmania.rustlang.core.mem.FixedArrayTypeLayout
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.Result

public fun vdotqS32(a: Int8x16T, b: Int8x16T): Int32x4T {
    val acc: Int32x4T = vdupqNS32(0)
    asm("sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b", acc, inout, vreg, acc, a, vreg, a, b, vreg, b, options(nostack, nomem))
    return acc
}

/**
 * Two SDOT ops into one accumulator
 */
public fun vdotqS32Pair(a0: Int8x16T, b0: Int8x16T, a1: Int8x16T, b1: Int8x16T): Int32x4T {
    val acc: Int32x4T = vdupqNS32(0)
    asm("sdot {acc:v}.4s, {a0:v}.16b, {b0:v}.16b", "sdot {acc:v}.4s, {a1:v}.16b, {b1:v}.16b", acc, inout, vreg, acc, a0, vreg, a0, b0, vreg, b0, a1, vreg, a1, b1, vreg, b1, options(nostack, nomem))
    return acc
}

/**
 * Accumulating SDOT: acc += dot4(a, b) for each lane.
 */
public fun sdotAcc(acc: Int32x4T, a: Int8x16T, b: Int8x16T): Int32x4T {
    val out = acc
    asm("sdot {out:v}.4s, {a:v}.16b, {b:v}.16b", out, inout, vreg, out, a, vreg, a, b, vreg, b, options(nostack, nomem))
    return out
}

public fun sdotLaneqS32(acc: Int32x4T, a: Int8x16T, b: Int8x16T, lane: Int): Int32x4T = sdotAcc(acc, a, vreinterpretqS8S32(vdupqLaneqS32(vreinterpretqS32S8(b), lane)))

public fun smmlaS32(acc: Int32x4T, a: Int8x16T, b: Int8x16T): Int32x4T {
    val out = acc
    asm("smmla {out:v}.4s, {a:v}.16b, {b:v}.16b", out, inout, vreg, out, a, vreg, a, b, vreg, b, options(nostack, nomem))
    return out
}

public fun decodeQ4kx8Scales(scalesIn: ConstPtr): Pair<Int16x8T, Int16x8T> {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    // Direct unaligned reads avoid the copyNonoverlapping stack round-trip.
    val sm0 = (PrimitiveLayouts.U32).readNative(((ConstPtr(scalesIn.rawAddress, scalesIn.rawMeta))).rawAddress)
    val sm1 = (PrimitiveLayouts.U32).readNative(((ConstPtr(ConstPtr((scalesIn).rawAddress + ((4).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, ConstPtr((scalesIn).rawAddress + ((4).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta))).rawAddress)
    val sm2 = (PrimitiveLayouts.U32).readNative(((ConstPtr(ConstPtr((scalesIn).rawAddress + ((8).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawAddress, ConstPtr((scalesIn).rawAddress + ((8).toLong() * (PrimitiveLayouts.U8).strideBytes)).rawMeta))).rawAddress)
    val mins03 = sm1 and KMASK1
    val mins47 = ((sm2 shr 4) and KMASK2) or (((sm1 shr 6) and KMASK3) shl 4)
    val outMins = vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(vcreateU32((mins03.toULong()) or ((mins47.toULong()) shl 32)))))
    val sc0 = sm0 and KMASK1
    val sc1 = (sm2 and KMASK2) or (((sm0 shr 6) and KMASK3) shl 4)
    val outScales = vmovlS8(vreinterpretS8U8(vreinterpretU8U32(vcreateU32((sc0.toULong()) or ((sc1.toULong()) shl 32)))))
    return Pair(outMins, outScales)
}

public fun mergeSignedMax(absA: Float32x4T, smaxA: Float32x4T, absB: Float32x4T, smaxB: Float32x4T): Pair<Float32x4T, Float32x4T> = Pair(vmaxqF32(absA, absB), vbslqF32(vcgtqF32(absB, absA), smaxB, smaxA))

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q4_0_q8_0: $n is not divisible by $qk80" }
    val nb = n / QK8_0
    return run {
        var sumv0 = vdupqNF32(0.0f)
        for (i in 0 until nb) {
            val x0 = xs[i]
            val y0 = ys[i]
            val m4b = vdupqNU8(0x0F)
            val s8b = vdupqNS8(0x8)
            val v00 = vld1qU8(x0.qs.asPtr())
            // 4-bit -> 8-bit
            val v00l = vreinterpretqS8U8(vandqU8(v00, m4b))
            val v00h = vreinterpretqS8U8(vshrqNU8(v00, 4))
            // sub 8
            val v00ls = vsubqS8(v00l, s8b)
            val v00hs = vsubqS8(v00h, s8b)
            // load y
            val v10l = vld1qS8(y0.qs.asPtr())
            val v10h = vld1qS8(y0.qs.asPtr().add(16))
            val pl0 = vdotqS32(v00ls, v10l)
            val ph0 = vdotqS32(v00hs, v10h)
            sumv0 = vmlaqNF32(sumv0, vcvtqF32S32(vaddqS32(pl0, ph0)), x0.d.toF32() * y0.d.toF32())
        }
        vaddvqF32(sumv0)
    }
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_q8_0_q8_0: $n is not divisible by $qk80" }
    val nb = n / QK8_0
    return run {
        var sumv0 = vdupqNF32(0.0f)
        for (i in 0 until nb) {
            val x0 = xs[i]
            val y0 = ys[i]
            val x00 = vld1qS8(x0.qs.asPtr())
            val x01 = vld1qS8(x0.qs.asPtr().add(16))
            // load y
            val y00 = vld1qS8(y0.qs.asPtr())
            val y01 = vld1qS8(y0.qs.asPtr().add(16))
            val p0 = vdotqS32(x00, y00)
            val p1 = vdotqS32(x01, y01)
            sumv0 = vmlaqNF32(sumv0, vcvtqF32S32(vaddqS32(p0, p1)), x0.d.toF32() * y0.d.toF32())
        }
        vaddvqF32(sumv0)
    }
}

internal fun quantizeQ80(xs: List<Float>, ys: MutableList<BlockQ80>) {
    check(xs.size.isMultipleOf(QK8_0))
    check(ys.size == xs.size / QK8_0)
    val (xChunks, _) = xs.asChunks<Qk80>()
    for ((chunk, y) in (xChunks.iter().zip(ys))) {
        run {
            val p = chunk.asPtr()
            val v0 = vld1qF32(p)
            val v1 = vld1qF32(p.add(4))
            val v2 = vld1qF32(p.add(8))
            val v3 = vld1qF32(p.add(12))
            val v4 = vld1qF32(p.add(16))
            val v5 = vld1qF32(p.add(20))
            val v6 = vld1qF32(p.add(24))
            val v7 = vld1qF32(p.add(28))
            val a01 = vmaxqF32(vabsqF32(v0), vabsqF32(v1))
            val a23 = vmaxqF32(vabsqF32(v2), vabsqF32(v3))
            val a45 = vmaxqF32(vabsqF32(v4), vabsqF32(v5))
            val a67 = vmaxqF32(vabsqF32(v6), vabsqF32(v7))
            val amax = vmaxvqF32(vmaxqF32(vmaxqF32(a01, a23), vmaxqF32(a45, a67)))
            val d = amax / ((1 shl 7) - 1).toFloat()
            val id =
                if (d != 0.0f) {
                    1.0f / d
                } else {
                    0.0
                }
            y.d = F16.fromF32(d)
            val id1 = vdupqNF32(id)
            val s01 = vcombineS16(vqmovnS32(vcvtaqS32F32(vmulqF32(v0, id1))), vqmovnS32(vcvtaqS32F32(vmulqF32(v1, id1))))
            val s23 = vcombineS16(vqmovnS32(vcvtaqS32F32(vmulqF32(v2, id1))), vqmovnS32(vcvtaqS32F32(vmulqF32(v3, id1))))
            val q0 = vcombineS8(vqmovnS16(s01), vqmovnS16(s23))
            vst1qS8(y.qs.asMutPtr(), q0)
            val s45 = vcombineS16(vqmovnS32(vcvtaqS32F32(vmulqF32(v4, id1))), vqmovnS32(vcvtaqS32F32(vmulqF32(v5, id1))))
            val s67 = vcombineS16(vqmovnS32(vcvtaqS32F32(vmulqF32(v6, id1))), vqmovnS32(vcvtaqS32F32(vmulqF32(v7, id1))))
            val q1 = vcombineS8(vqmovnS16(s45), vqmovnS16(s67))
            vst1qS8(y.qs.asMutPtr().add(16), q1)
        }
    }
}

internal fun vecDot4Q80Q80(n: Int, xs0: List<BlockQ80>, xs1: List<BlockQ80>, xs2: List<BlockQ80>, xs3: List<BlockQ80>, ys: List<BlockQ80>): List<Any?> {
    check(n.isMultipleOf(QK8_0)) { "vec_dot_4_q8_0_q8_0: $n is not divisible by $qk80" }
    val nb = n / QK8_0
    return run {
        var sum0 = vdupqNF32(0.0f)
        var sum1 = vdupqNF32(0.0f)
        var sum2 = vdupqNF32(0.0f)
        var sum3 = vdupqNF32(0.0f)
        for (i in 0 until nb) {
            val y = ys[i]
            val y0 = vld1qS8(y.qs.asPtr())
            val y1 = vld1qS8(y.qs.asPtr().add(16))
            val yd = y.d.toF32()
            val x0 = xs0[i]
            val x1 = xs1[i]
            val x2 = xs2[i]
            val x3 = xs3[i]
            // Each column uses 2 SDOTs fused into one accumulator via vdotqS32Pair.
            // All 4 columns share y0/y1 and have independent accumulators.
            val p0 = vdotqS32Pair(vld1qS8(x0.qs.asPtr()), y0, vld1qS8(x0.qs.asPtr().add(16)), y1)
            val p1 = vdotqS32Pair(vld1qS8(x1.qs.asPtr()), y0, vld1qS8(x1.qs.asPtr().add(16)), y1)
            val p2 = vdotqS32Pair(vld1qS8(x2.qs.asPtr()), y0, vld1qS8(x2.qs.asPtr().add(16)), y1)
            val p3 = vdotqS32Pair(vld1qS8(x3.qs.asPtr()), y0, vld1qS8(x3.qs.asPtr().add(16)), y1)
            sum0 = vmlaqNF32(sum0, vcvtqF32S32(p0), x0.d.toF32() * yd)
            sum1 = vmlaqNF32(sum1, vcvtqF32S32(p1), x1.d.toF32() * yd)
            sum2 = vmlaqNF32(sum2, vcvtqF32S32(p2), x2.d.toF32() * yd)
            sum3 = vmlaqNF32(sum3, vcvtqF32S32(p3), x3.d.toF32() * yd)
        }
        listOf(vaddvqF32(sum0), vaddvqF32(sum1), vaddvqF32(sum2), vaddvqF32(sum3))
    }
}

public fun quantizeQ80x4Interleaved(lhs: List<Float>, k: Int, row: Int, out: MutableList<BlockQ80x4>, blockLen: Int) {
    val kInBlocks = k / QK8_0
    for (b in 0 until kInBlocks) {
        val p = BlockQ80x4(d = List(4) { F16.ZERO }, qs = ByteArray(QK8_0 * 4) { 0.toByte() })
        val ids = List(4) { 0.0f }
        for (r in 0 until 4) {
            val xs = ((lhs).slice((row + r) * k + b * QK8_0 until (row + r) * k + (b + 1) * QK8_0))
            var amax = 0.0f
            for (x in xs) {
                amax = amax.max(x.abs())
            }
            val d = amax / ((1 shl 7) - 1).toFloat()
            ids[r] =
                if (d != 0.0f) {
                    1.0f / d
                } else {
                    0.0
                }
            p.d[r] = F16.fromF32(d)
        }
        for (r1 in 0 until 4) {
            val xs1 = ((lhs).slice((row + r1) * k + b * QK8_0 until (row + r1) * k + (b + 1) * QK8_0))
            for ((j, x1) in xs1.mapIndexed { index, item -> Pair(index, item) }) {
                p.qs[(j / blockLen) * blockLen * 4 + r1 * blockLen + j % blockLen] = (x1 * ids[r1]).round().toByte()
            }
        }
        out[b] = p
    }
}

public fun vecDot4Q40x4Q804x4(n: Int, xs: List<BlockQ40x4>, ys: List<BlockQ80>): FloatArray {
    check(n.isMultipleOf(QK8_0))
    val out = List(4) { 0.0f }
    val highMask = vdupqNU8(0xf0)
    var acc = vdupqNF32(0.0)
    for ((x, y) in (xs.zip(ys))) {
        val bd = loadF16x4(x.d.asPtr())
        val ad = vdupqNF32(y.d.toF32())
        val a0 = vld1qS8(y.qs.asPtr())
        val a1 = vld1qS8(y.qs.asPtr().add(16))
        val b0 = vld1qS8(x.qs.asPtr())
        val b1 = vld1qS8(x.qs.asPtr().add(16))
        val b2 = vld1qS8(x.qs.asPtr().add(32))
        val b3 = vld1qS8(x.qs.asPtr().add(48))
        var ret = vdupqNS32(0)
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b0), 4)), a0, 0)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b1), 4)), a0, 1)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b2), 4)), a0, 2)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b3), 4)), a0, 3)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b0), highMask)), a1, 0)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b1), highMask)), a1, 1)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b2), highMask)), a1, 2)
        }
        run {
            ret = sdotLaneqS32(ret, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b3), highMask)), a1, 3)
        }
        acc = vmlaqF32(acc, vmulqNF32(vcvtqF32S32(ret), 1.0 / 16.0), vmulqF32(ad, bd))
    }
    vst1qF32(out.asMutPtr(), acc)
    return out
}

public fun vecDot4Q40x4Q804x8(n: Int, xs: List<BlockQ40x4>, ys: List<BlockQ80>): FloatArray {
    check(n.isMultipleOf(QK8_0))
    val out = List(4) { 0.0f }
    val highMask = vdupqNU8(0xf0)
    var acc = vdupqNF32(0.0)
    for ((x, y) in (xs.zip(ys))) {
        val bd = loadF16x4(x.d.asPtr())
        val ad = vdupqNF32(y.d.toF32())
        val a0 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr())))
        val a1 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(8))))
        val a2 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(16))))
        val a3 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(24))))
        val b0 = vld1qS8(x.qs.asPtr())
        val b1 = vld1qS8(x.qs.asPtr().add(16))
        val b2 = vld1qS8(x.qs.asPtr().add(32))
        val b3 = vld1qS8(x.qs.asPtr().add(48))
        var ret0 = vdupqNS32(0)
        var ret1 = vdupqNS32(0)
        ret0 = sdotAcc(ret0, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b0), 4)), a0)
        ret1 = sdotAcc(ret1, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b1), 4)), a0)
        ret0 = sdotAcc(ret0, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b2), 4)), a1)
        ret1 = sdotAcc(ret1, vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(b3), 4)), a1)
        ret0 = sdotAcc(ret0, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b0), highMask)), a2)
        ret1 = sdotAcc(ret1, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b1), highMask)), a2)
        ret0 = sdotAcc(ret0, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b2), highMask)), a3)
        ret1 = sdotAcc(ret1, vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(b3), highMask)), a3)
        val ret = vpaddqS32(ret0, ret1)
        acc = vmlaqF32(acc, vmulqNF32(vcvtqF32S32(ret), 1.0 / 16.0), vmulqF32(ad, bd))
    }
    vst1qF32(out.asMutPtr(), acc)
    return out
}

public fun vecDot4Q40x4Q80(n: Int, xs: List<BlockQ40x4>, ys: List<BlockQ80>): FloatArray =
    if (get().i8mm) {
        vecDot4Q40x4Q804x8(n, xs, ys)
    } else {
        vecDot4Q40x4Q804x4(n, xs, ys)
    }

public fun matmulQ40X4Gemv(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ40x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    threadLocal(LHS_Q4_0_GEMV_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>())
    val requiredLen = (kInBlocks * elemSize).divCeil(8)
    return LHS_Q4_0_GEMV_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsB: MutableList<BlockQ80> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
        BlockQ80.fromFloat(lhs, lhsB)
        val pool = barrierPool()
        val lhsPtr = lhsB.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        val x4BlockBytes = sizeOf<BlockQ40x4>(typeLayoutT = ObjectSlotLayout<BlockQ40x4>())
        // equal-sized gemv units: a static split skips the shared cursor, which costs
        // several us per small decode matmul
        pool.executeStatic(nGroups, { range ->
            val lhsRow = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (g in range) {
                val xs = fromRawParts(ConstPtr((repackedPtr + g * kInBlocks * x4BlockBytes).toLong()), kInBlocks)
                val results = vecDot4Q40x4Q80(k, xs, lhsRow)
                run {
                    copyNonoverlapping(results.asPtr(), MutPtr((dstPtr1).rawAddress + ((g * 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), 4)
                }
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ40x44x4(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ80x4>, rhs: List<BlockQ40x4>, kInBlocks: Int) {
    var sum0 = vdupqNF32(0.0)
    var sum1 = vdupqNF32(0.0)
    var sum2 = vdupqNF32(0.0)
    var sum3 = vdupqNF32(0.0)
    val highMask = vdupqNU8(0xf0)
    for (i in 0 until kInBlocks) {
        val x = rhs[i]
        val y = lhs[i]
        val bd = listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32())
        val bd1 = vld1qF32(bd.asPtr())
        val ad = listOf(y.d[0].toF32(), y.d[1].toF32(), y.d[2].toF32(), y.d[3].toF32())
        var ret0 = vdupqNS32(0)
        var ret1 = vdupqNS32(0)
        var ret2 = vdupqNS32(0)
        var ret3 = vdupqNS32(0)
        for (j in 0 until QK8_0 / 8) {
            val q4 = vld1qS8(x.qs.asPtr().add(j * 16))
            val q4lo = vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(q4), 4))
            val q4hi = vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(q4), highMask))
            run {
                ret0 = sdotLaneqS32(ret0, q4lo, vld1qS8(y.qs.asPtr().add(j * 16)), 0)
                ret1 = sdotLaneqS32(ret1, q4lo, vld1qS8(y.qs.asPtr().add(j * 16)), 1)
                ret2 = sdotLaneqS32(ret2, q4lo, vld1qS8(y.qs.asPtr().add(j * 16)), 2)
                ret3 = sdotLaneqS32(ret3, q4lo, vld1qS8(y.qs.asPtr().add(j * 16)), 3)
            }
            run {
                ret0 = sdotLaneqS32(ret0, q4hi, vld1qS8(y.qs.asPtr().add((j + 4) * 16)), 0)
                ret1 = sdotLaneqS32(ret1, q4hi, vld1qS8(y.qs.asPtr().add((j + 4) * 16)), 1)
                ret2 = sdotLaneqS32(ret2, q4hi, vld1qS8(y.qs.asPtr().add((j + 4) * 16)), 2)
                ret3 = sdotLaneqS32(ret3, q4hi, vld1qS8(y.qs.asPtr().add((j + 4) * 16)), 3)
            }
        }
        sum0 = vmlaqF32(sum0, vmulqNF32(vcvtqF32S32(ret0), 1.0 / 16.0), vmulqNF32(bd1, ad[0]))
        sum1 = vmlaqF32(sum1, vmulqNF32(vcvtqF32S32(ret1), 1.0 / 16.0), vmulqNF32(bd1, ad[1]))
        sum2 = vmlaqF32(sum2, vmulqNF32(vcvtqF32S32(ret2), 1.0 / 16.0), vmulqNF32(bd1, ad[2]))
        sum3 = vmlaqF32(sum3, vmulqNF32(vcvtqF32S32(ret3), 1.0 / 16.0), vmulqNF32(bd1, ad[3]))
    }
    vst1qF32(MutPtr((dst).rawAddress + ((row * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum0)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum1)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum2)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 3) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum3)
}

internal fun matmulQ40X4(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ40x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    if (m == 1) {
        return matmulQ40X4Gemv(Triple(m, k, n), lhs, repacked, dst)
    }
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    val rowGroups = m / 4
    threadLocal(LHS_Q4_0_X4_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_Q4_0_X4_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ80x4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            quantizeQ80x4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 4)
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 4
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ40x4>()).strideBytes)), kInBlocks)
                storeQ40x44x4(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ40x44x4I8mm(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ80x4>, rhs: List<BlockQ40x4>, kInBlocks: Int) {
    val highMask = vdupqNU8(0xf0)
    var sum0 = vdupqNF32(0.0)
    var sum1 = vdupqNF32(0.0)
    var sum2 = vdupqNF32(0.0)
    var sum3 = vdupqNF32(0.0)
    for (i in 0 until kInBlocks) {
        val x = rhs[i]
        val y = lhs[i]
        val bd = listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32())
        val bd1 = vld1qF32(bd.asPtr())
        val ad = listOf(y.d[0].toF32(), y.d[1].toF32(), y.d[2].toF32(), y.d[3].toF32())
        var acc00 = vdupqNS32(0)
        var acc01 = vdupqNS32(0)
        var acc10 = vdupqNS32(0)
        var acc11 = vdupqNS32(0)
        for (chunk in 0 until 2) {
            val q4Offset = chunk * 32
            val q8LoOffset = chunk * 32
            val q8HiOffset = (chunk + 2) * 32
            val q401 = vld1qS8(x.qs.asPtr().add(q4Offset))
            val q423 = vld1qS8(x.qs.asPtr().add(q4Offset + 16))
            val q4lo01 = vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(q401), 4))
            val q4lo23 = vreinterpretqS8U8(vshlqNU8(vreinterpretqU8S8(q423), 4))
            val q4hi01 = vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(q401), highMask))
            val q4hi23 = vreinterpretqS8U8(vandqU8(vreinterpretqU8S8(q423), highMask))
            val a01 = vld1qS8(y.qs.asPtr().add(q8LoOffset))
            val a23 = vld1qS8(y.qs.asPtr().add(q8LoOffset + 16))
            acc00 = smmlaS32(acc00, a01, q4lo01)
            acc01 = smmlaS32(acc01, a01, q4lo23)
            acc10 = smmlaS32(acc10, a23, q4lo01)
            acc11 = smmlaS32(acc11, a23, q4lo23)
            val a011 = vld1qS8(y.qs.asPtr().add(q8HiOffset))
            val a231 = vld1qS8(y.qs.asPtr().add(q8HiOffset + 16))
            acc00 = smmlaS32(acc00, a011, q4hi01)
            acc01 = smmlaS32(acc01, a011, q4hi23)
            acc10 = smmlaS32(acc10, a231, q4hi01)
            acc11 = smmlaS32(acc11, a231, q4hi23)
        }
        val row0 = vcombineS32(vgetLowS32(acc00), vgetLowS32(acc01))
        val row1 = vcombineS32(vgetHighS32(acc00), vgetHighS32(acc01))
        val row2 = vcombineS32(vgetLowS32(acc10), vgetLowS32(acc11))
        val row3 = vcombineS32(vgetHighS32(acc10), vgetHighS32(acc11))
        sum0 = vmlaqF32(sum0, vmulqNF32(vcvtqF32S32(row0), 1.0 / 16.0), vmulqNF32(bd1, ad[0]))
        sum1 = vmlaqF32(sum1, vmulqNF32(vcvtqF32S32(row1), 1.0 / 16.0), vmulqNF32(bd1, ad[1]))
        sum2 = vmlaqF32(sum2, vmulqNF32(vcvtqF32S32(row2), 1.0 / 16.0), vmulqNF32(bd1, ad[2]))
        sum3 = vmlaqF32(sum3, vmulqNF32(vcvtqF32S32(row3), 1.0 / 16.0), vmulqNF32(bd1, ad[3]))
    }
    vst1qF32(MutPtr((dst).rawAddress + ((row * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum0)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum1)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum2)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 3) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum3)
}

internal fun matmulQ40X4I8mm(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ40x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    val rowGroups = m / 4
    threadLocal(LHS_Q4_0_X4_I8MM_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_Q4_0_X4_I8MM_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ80x4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            quantizeQ80x4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 8)
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 4
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ40x4>()).strideBytes)), kInBlocks)
                storeQ40x44x4I8mm(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
            }
        })
        Result.Ok(Unit)
    })
}

public fun quantizeQ8kX4Interleaved(lhs: List<Float>, k: Int, row: Int, out: MutableList<BlockQ8Kx4>, blockLen: Int) {
    val kInBlocks = k / QK_K
    for (b in 0 until kInBlocks) {
        val p = BlockQ8Kx4(d = FloatArray(4) { 0.0f }, qs = ByteArray(QK_K * 4) { 0.toByte() }, bsums = ShortArray(QK_K / 4) { 0.toShort() })
        val src = List(4) { List(QK_K) { 0.0f } }
        val iscale = List(4) { 0.0f }
        for (r in 0 until 4) {
            val xs = ((lhs).slice((row + r) * k + b * QK_K until (row + r) * k + (b + 1) * QK_K))
            var max = 0.0f
            var amax = 0.0f
            for ((j, x) in xs.mapIndexed { index, item -> Pair(index, item) }) {
                src[r][j] = x
                if (amax < x.abs()) {
                    amax = x.abs()
                    max = x
                }
            }
            iscale[r] =
                if (amax != 0.0) {
                    -127.0 / max
                } else {
                    0.0
                }
            p.d[r] =
                if (amax != 0.0) {
                    1.0f / iscale[r]
                } else {
                    0.0f
                }
        }
        for (j1 in 0 until QK_K * 4) {
            val srcOffset = (j1 / (4 * blockLen)) * blockLen + j1 % blockLen
            val srcId = (j1 % (4 * blockLen)) / blockLen
            val q = (src[srcId][srcOffset] * iscale[srcId]).round().toByte()
            p.qs[j1] = q
            val index =
                if (blockLen == 4) {
                    (((j1 and 15) shr 2) shl 2) + ((j1 shr 8) shl 4) + ((j1 shr 6) and 3)
                } else {
                    (((j1 and 31) shr 3) shl 2) + ((j1 shr 8) shl 4) + ((j1 shr 6) and 3)
                }
            p.bsums[index] += q.toShort()
        }
        out[b] = p
    }
}

public fun storeQ4kx84x8I8mm(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ8Kx4>, rhs: List<BlockQ4Kx8>, kInBlocks: Int) {
    val m4b = vdupqNU8(0x0f)
    val accF32 = List(8) { vdupqNF32(0.0) }
    for (b in 0 until kInBlocks) {
        val x = rhs[b]
        val y = lhs[b]
        val bsumsArr = List(4) { List(8) { 0 } }
        for ((r, bsumsRow) in bsumsArr.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
            for ((i, bsums) in bsumsRow.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
                // quarter-major interleaved layout: [16 * quarter + 4 * row + half-pair]
                val base = 16 * (i / 2) + 4 * r + (2 * i) % 4
                bsums = y.bsums[base] + y.bsums[base + 1]
            }
        }
        val acc = List(8) { vdupqNS32(0) }
        val biasAcc = List(8) { vdupqNS32(0) }
        for (sb in 0 until QK_K / 64) {
            val (mins0, scales0) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24))
            val (mins1, scales1) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24 + 12))
            val q4sbMins = listOf(mins0, mins1)
            val q4sbScales = List(2) { List(8) { 0 } }
            vst1qS16(q4sbScales[0].asMutPtr(), scales0)
            vst1qS16(q4sbScales[1].asMutPtr(), scales1)
            val q8Base = y.qs.asPtr().add(sb * 256)
            val q8s = List(2) { List(8) { vdupqNS8(0) } }
            val (q8s0, q8s1) = q8s.splitAtMut(1)
            for ((i1, q80, q81) in (q8s0[0].iterMut().zip(q8s1[0].iterMut())).mapIndexed { index, item -> Pair(index, item) }) {
                q80 = vld1qS8(q8Base.add(i1 * 32))
                q81 = vld1qS8(q8Base.add(i1 * 32 + 16))
            }
            for (cp in 0 until 4) {
                val qs = listOf(vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp + 64)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp + 128)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp + 192)))
                var acc0 = vdupqNS32(0)
                var acc1 = vdupqNS32(0)
                var acc2 = vdupqNS32(0)
                var acc3 = vdupqNS32(0)
                for (i2 in 0 until 4) {
                    val qsLo = vreinterpretqS8U8(vandqU8(qs[i2], m4b))
                    val qsHi = vreinterpretqS8U8(vshrqNU8(qs[i2], 4))
                    acc0 = smmlaS32(acc0, qsLo, q8s[0][i2])
                    acc2 = smmlaS32(acc2, qsLo, q8s[1][i2])
                    acc1 = smmlaS32(acc1, qsHi, q8s[0][i2 + 4])
                    acc3 = smmlaS32(acc3, qsHi, q8s[1][i2 + 4])
                }
                val scaleOffset = cp * 2
                val scale0 = vcombineS32(vdupNS32(q4sbScales[0][scaleOffset].toInt()), vdupNS32(q4sbScales[0][scaleOffset + 1].toInt()))
                val scale1 = vcombineS32(vdupNS32(q4sbScales[1][scaleOffset].toInt()), vdupNS32(q4sbScales[1][scaleOffset + 1].toInt()))
                acc[cp] = vmlaqS32(acc[cp], acc0, scale0)
                acc[cp] = vmlaqS32(acc[cp], acc1, scale1)
                acc[cp + 4] = vmlaqS32(acc[cp + 4], acc2, scale0)
                acc[cp + 4] = vmlaqS32(acc[cp + 4], acc3, scale1)
            }
            for (q8Row in 0 until 4) {
                val bsumsVecLo = vdupNS16(bsumsArr[q8Row][sb * 2])
                val bsumsVecHi = vdupNS16(bsumsArr[q8Row][sb * 2 + 1])
                biasAcc[2 * q8Row] = vmlalS16(biasAcc[2 * q8Row], bsumsVecLo, vgetLowS16(q4sbMins[0]))
                biasAcc[2 * q8Row] = vmlalS16(biasAcc[2 * q8Row], bsumsVecHi, vgetLowS16(q4sbMins[1]))
                biasAcc[2 * q8Row + 1] = vmlalS16(biasAcc[2 * q8Row + 1], bsumsVecLo, vgetHighS16(q4sbMins[0]))
                biasAcc[2 * q8Row + 1] = vmlalS16(biasAcc[2 * q8Row + 1], bsumsVecHi, vgetHighS16(q4sbMins[1]))
            }
        }
        for (item in acc) {
            val aux = vzipS32(vgetLowS32(item), vgetHighS32(item))
            item = vcombineS32(aux.first, aux.second)
        }
        val reorderAcc = listOf(vcombineS32(vgetLowS32(acc[0]), vgetLowS32(acc[1])), vcombineS32(vgetLowS32(acc[2]), vgetLowS32(acc[3])), vcombineS32(vgetHighS32(acc[0]), vgetHighS32(acc[1])), vcombineS32(vgetHighS32(acc[2]), vgetHighS32(acc[3])), vcombineS32(vgetLowS32(acc[4]), vgetLowS32(acc[5])), vcombineS32(vgetLowS32(acc[6]), vgetLowS32(acc[7])), vcombineS32(vgetHighS32(acc[4]), vgetHighS32(acc[5])), vcombineS32(vgetHighS32(acc[6]), vgetHighS32(acc[7])))
        for (r1 in 0 until 4) {
            for (g in 0 until 2) {
                val q4D = listOf(x.d[g * 4].toF32(), x.d[g * 4 + 1].toF32(), x.d[g * 4 + 2].toF32(), x.d[g * 4 + 3].toF32())
                val q4Dmin = listOf(x.dmin[g * 4].toF32(), x.dmin[g * 4 + 1].toF32(), x.dmin[g * 4 + 2].toF32(), x.dmin[g * 4 + 3].toF32())
                val q8D = y.d[r1]
                val scale = vmulqNF32(vld1qF32(q4D.asPtr()), q8D)
                val dmins = vmulqNF32(vld1qF32(q4Dmin.asPtr()), q8D)
                val idx = 2 * r1 + g
                accF32[idx] = vsubqF32(accF32[idx], vmulqF32(vcvtqF32S32(biasAcc[idx]), dmins))
                accF32[idx] = vmlaqF32(accF32[idx], vcvtqF32S32(reorderAcc[idx]), scale)
            }
        }
    }
    for (r2 in 0 until 4) {
        vst1qF32(MutPtr((dst).rawAddress + (((row + r2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r2])
        vst1qF32(MutPtr((dst).rawAddress + (((row + r2) * n + col + 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r2 + 1])
    }
}

internal fun matmulQ4kX8I8mm(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ4Kx8>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK_K))
    check(n.isMultipleOf(8))
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    val rowGroups = m / 4
    threadLocal(LHS_Q4K_X8_I8MM_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8Kx4>(typeLayoutT = ObjectSlotLayout<BlockQ8Kx4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_Q4K_X8_I8MM_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ8Kx4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            quantizeQ8kX4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 8)
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 8
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8Kx4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ4Kx8>()).strideBytes)), kInBlocks)
                storeQ4kx84x8I8mm(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ5kx84x8(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ8Kx4>, rhs: List<BlockQ5Kx8>, kInBlocks: Int) {
    val m4b = vdupqNU8(0x0f)
    val mone = vdupqNU8(1)
    val mtwo = vdupqNU8(2)
    val accF32 = List(8) { vdupqNF32(0.0) }
    for (b in 0 until kInBlocks) {
        val x = rhs[b]
        val y = lhs[b]
        val q5D0123 = vld1qF32(listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32()).asPtr())
        val q5D4567 = vld1qF32(listOf(x.d[4].toF32(), x.d[5].toF32(), x.d[6].toF32(), x.d[7].toF32()).asPtr())
        val q5Dmin0123 = vld1qF32(listOf(x.dmin[0].toF32(), x.dmin[1].toF32(), x.dmin[2].toF32(), x.dmin[3].toF32()).asPtr())
        val q5Dmin4567 = vld1qF32(listOf(x.dmin[4].toF32(), x.dmin[5].toF32(), x.dmin[6].toF32(), x.dmin[7].toF32()).asPtr())
        val sbdScale0123 = listOf(vmulqNF32(q5D0123, y.d[0]), vmulqNF32(q5D0123, y.d[1]), vmulqNF32(q5D0123, y.d[2]), vmulqNF32(q5D0123, y.d[3]))
        val sbdScale4567 = listOf(vmulqNF32(q5D4567, y.d[0]), vmulqNF32(q5D4567, y.d[1]), vmulqNF32(q5D4567, y.d[2]), vmulqNF32(q5D4567, y.d[3]))
        val sbdMin0123 = listOf(vmulqNF32(q5Dmin0123, y.d[0]), vmulqNF32(q5Dmin0123, y.d[1]), vmulqNF32(q5Dmin0123, y.d[2]), vmulqNF32(q5Dmin0123, y.d[3]))
        val sbdMin4567 = listOf(vmulqNF32(q5Dmin4567, y.d[0]), vmulqNF32(q5Dmin4567, y.d[1]), vmulqNF32(q5Dmin4567, y.d[2]), vmulqNF32(q5Dmin4567, y.d[3]))
        val bsumsArr = List(4) { List(8) { 0 } }
        for (r in 0 until 4) {
            for (i in 0 until 8) {
                // quarter-major interleaved layout: [16 * quarter + 4 * row + half-pair]
                val base = 16 * (i / 2) + 4 * r + (2 * i) % 4
                bsumsArr[r][i] = y.bsums[base] + y.bsums[base + 1]
            }
        }
        val biasAcc = List(8) { vdupqNS32(0) }
        val qh = List(2) { List(8) { vdupqNU8(0) } }
        for (c in 0 until 2) {
            for (i1 in 0 until 8) {
                qh[c][i1] = vld1qU8(x.qh.asPtr().add(i1 * 32 + 16 * c))
            }
        }
        for (sb in 0 until QK_K / 64) {
            val accLo = List(8) { vdupqNS32(0) }
            val accHi = List(8) { vdupqNS32(0) }
            val (q5sbMins0, q5sbScales0) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24))
            val (q5sbMins1, q5sbScales1) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24 + 12))
            val q5sbMins = listOf(q5sbMins0, q5sbMins1)
            val q5sbScales = listOf(q5sbScales0, q5sbScales1)
            for (kIdx in 0 until 8) {
                val q8Blk0 = vld1qS8(y.qs.asPtr().add(sb * 256 + 16 * kIdx))
                val q8Blk1 = vld1qS8(y.qs.asPtr().add(sb * 256 + 16 * kIdx + 128))
                val q50123 = vld1qU8(x.qs.asPtr().add(sb * QK_K + 32 * kIdx))
                val q54567 = vld1qU8(x.qs.asPtr().add(sb * QK_K + 32 * kIdx + 16))
                val hbitLo0123 = vandqU8(qh[0][kIdx], mone)
                val hbitHi0123 = vshlqNU8(vandqU8(qh[0][kIdx], mtwo), 3)
                qh[0][kIdx] = vshrqNU8(qh[0][kIdx], 2)
                val hbitLo4567 = vandqU8(qh[1][kIdx], mone)
                val hbitHi4567 = vshlqNU8(vandqU8(qh[1][kIdx], mtwo), 3)
                qh[1][kIdx] = vshrqNU8(qh[1][kIdx], 2)
                val q50123Lo = vreinterpretqS8U8(vorrqU8(vandqU8(q50123, m4b), vshlqNU8(hbitLo0123, 4)))
                val q50123Hi = vreinterpretqS8U8(vorrqU8(vshrqNU8(q50123, 4), hbitHi0123))
                accLo[0] = sdotLaneqS32(accLo[0], q50123Lo, q8Blk0, 0)
                accLo[1] = sdotLaneqS32(accLo[1], q50123Lo, q8Blk0, 1)
                accLo[2] = sdotLaneqS32(accLo[2], q50123Lo, q8Blk0, 2)
                accLo[3] = sdotLaneqS32(accLo[3], q50123Lo, q8Blk0, 3)
                accHi[0] = sdotLaneqS32(accHi[0], q50123Hi, q8Blk1, 0)
                accHi[1] = sdotLaneqS32(accHi[1], q50123Hi, q8Blk1, 1)
                accHi[2] = sdotLaneqS32(accHi[2], q50123Hi, q8Blk1, 2)
                accHi[3] = sdotLaneqS32(accHi[3], q50123Hi, q8Blk1, 3)
                val q54567Lo = vreinterpretqS8U8(vorrqU8(vandqU8(q54567, m4b), vshlqNU8(hbitLo4567, 4)))
                val q54567Hi = vreinterpretqS8U8(vorrqU8(vshrqNU8(q54567, 4), hbitHi4567))
                accLo[4] = sdotLaneqS32(accLo[4], q54567Lo, q8Blk0, 0)
                accLo[5] = sdotLaneqS32(accLo[5], q54567Lo, q8Blk0, 1)
                accLo[6] = sdotLaneqS32(accLo[6], q54567Lo, q8Blk0, 2)
                accLo[7] = sdotLaneqS32(accLo[7], q54567Lo, q8Blk0, 3)
                accHi[4] = sdotLaneqS32(accHi[4], q54567Hi, q8Blk1, 0)
                accHi[5] = sdotLaneqS32(accHi[5], q54567Hi, q8Blk1, 1)
                accHi[6] = sdotLaneqS32(accHi[6], q54567Hi, q8Blk1, 2)
                accHi[7] = sdotLaneqS32(accHi[7], q54567Hi, q8Blk1, 3)
            }
            val sc0123Lo = vgetLowS16(q5sbScales[0])
            val sc4567Lo = vgetHighS16(q5sbScales[0])
            val sc0123Hi = vgetLowS16(q5sbScales[1])
            val sc4567Hi = vgetHighS16(q5sbScales[1])
            for (r1 in 0 until 4) {
                val sumf0123 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(sc0123Lo), accLo[r1]), vmulqS32(vmovlS16(sc0123Hi), accHi[r1])))
                accF32[2 * r1] = vmlaqF32(accF32[2 * r1], sbdScale0123[r1], sumf0123)
                val sumf4567 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(sc4567Lo), accLo[r1 + 4]), vmulqS32(vmovlS16(sc4567Hi), accHi[r1 + 4])))
                accF32[2 * r1 + 1] = vmlaqF32(accF32[2 * r1 + 1], sbdScale4567[r1], sumf4567)
                val bsumsVecLo = vdupNS16(bsumsArr[r1][sb * 2])
                val bsumsVecHi = vdupNS16(bsumsArr[r1][sb * 2 + 1])
                biasAcc[2 * r1] = vmlalS16(biasAcc[2 * r1], bsumsVecLo, vgetLowS16(q5sbMins[0]))
                biasAcc[2 * r1] = vmlalS16(biasAcc[2 * r1], bsumsVecHi, vgetLowS16(q5sbMins[1]))
                biasAcc[2 * r1 + 1] = vmlalS16(biasAcc[2 * r1 + 1], bsumsVecLo, vgetHighS16(q5sbMins[0]))
                biasAcc[2 * r1 + 1] = vmlalS16(biasAcc[2 * r1 + 1], bsumsVecHi, vgetHighS16(q5sbMins[1]))
            }
        }
        for (r2 in 0 until 4) {
            accF32[2 * r2] = vsubqF32(accF32[2 * r2], vmulqF32(vcvtqF32S32(biasAcc[2 * r2]), sbdMin0123[r2]))
            accF32[2 * r2 + 1] = vsubqF32(accF32[2 * r2 + 1], vmulqF32(vcvtqF32S32(biasAcc[2 * r2 + 1]), sbdMin4567[r2]))
        }
    }
    for (r3 in 0 until 4) {
        vst1qF32(MutPtr((dst).rawAddress + (((row + r3) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r3])
        vst1qF32(MutPtr((dst).rawAddress + (((row + r3) * n + col + 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r3 + 1])
    }
}

public fun storeQ5kx84x8I8mm(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ8Kx4>, rhs: List<BlockQ5Kx8>, kInBlocks: Int) {
    val m4b = vdupqNU8(0x0f)
    val mone = vdupqNU8(1)
    val mtwo = vdupqNU8(2)
    val accF32 = List(8) { vdupqNF32(0.0) }
    for (b in 0 until kInBlocks) {
        val x = rhs[b]
        val y = lhs[b]
        val bsumsArr = List(4) { List(8) { 0 } }
        for ((r, bsumsRow) in bsumsArr.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
            for ((i, bsums) in bsumsRow.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
                // quarter-major interleaved layout: [16 * quarter + 4 * row + half-pair]
                val base = 16 * (i / 2) + 4 * r + (2 * i) % 4
                bsums = y.bsums[base] + y.bsums[base + 1]
            }
        }
        val acc = List(8) { vdupqNS32(0) }
        val biasAcc = List(8) { vdupqNS32(0) }
        val qh = List(4) { List(4) { vdupqNU8(0) } }
        for ((cp, qhCp) in qh.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
            qhCp[0] = vld1qU8(x.qh.asPtr().add(16 * cp))
            qhCp[1] = vld1qU8(x.qh.asPtr().add(16 * cp + 64))
            qhCp[2] = vld1qU8(x.qh.asPtr().add(16 * cp + 128))
            qhCp[3] = vld1qU8(x.qh.asPtr().add(16 * cp + 192))
        }
        for (sb in 0 until QK_K / 64) {
            val (mins0, scales0) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24))
            val (mins1, scales1) = decodeQ4kx8Scales(x.scales.asPtr().add(sb * 24 + 12))
            val q5sbMins = listOf(mins0, mins1)
            val q5sbScales = List(2) { List(8) { 0 } }
            vst1qS16(q5sbScales[0].asMutPtr(), scales0)
            vst1qS16(q5sbScales[1].asMutPtr(), scales1)
            val q8Base = y.qs.asPtr().add(sb * 256)
            val q8s = List(2) { List(8) { vdupqNS8(0) } }
            val (q8s0, q8s1) = q8s.splitAtMut(1)
            for ((i1, q80, q81) in (q8s0[0].iterMut().zip(q8s1[0].iterMut())).mapIndexed { index, item -> Pair(index, item) }) {
                q80 = vld1qS8(q8Base.add(i1 * 32))
                q81 = vld1qS8(q8Base.add(i1 * 32 + 16))
            }
            for (cp1 in 0 until 4) {
                val qs = listOf(vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp1)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp1 + 64)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp1 + 128)), vld1qU8(x.qs.asPtr().add(sb * QK_K + 16 * cp1 + 192)))
                var acc0 = vdupqNS32(0)
                var acc1 = vdupqNS32(0)
                var acc2 = vdupqNS32(0)
                var acc3 = vdupqNS32(0)
                for (i2 in 0 until 4) {
                    val hbitLo = vandqU8(qh[cp1][i2], mone)
                    val hbitHi = vshlqNU8(vandqU8(qh[cp1][i2], mtwo), 3)
                    qh[cp1][i2] = vshrqNU8(qh[cp1][i2], 2)
                    val qsLo = vreinterpretqS8U8(vorrqU8(vandqU8(qs[i2], m4b), vshlqNU8(hbitLo, 4)))
                    val qsHi = vreinterpretqS8U8(vorrqU8(vshrqNU8(qs[i2], 4), hbitHi))
                    acc0 = smmlaS32(acc0, qsLo, q8s[0][i2])
                    acc2 = smmlaS32(acc2, qsLo, q8s[1][i2])
                    acc1 = smmlaS32(acc1, qsHi, q8s[0][i2 + 4])
                    acc3 = smmlaS32(acc3, qsHi, q8s[1][i2 + 4])
                }
                val scaleOffset = cp1 * 2
                val scale0 = vcombineS32(vdupNS32(q5sbScales[0][scaleOffset].toInt()), vdupNS32(q5sbScales[0][scaleOffset + 1].toInt()))
                val scale1 = vcombineS32(vdupNS32(q5sbScales[1][scaleOffset].toInt()), vdupNS32(q5sbScales[1][scaleOffset + 1].toInt()))
                acc[cp1] = vmlaqS32(acc[cp1], acc0, scale0)
                acc[cp1] = vmlaqS32(acc[cp1], acc1, scale1)
                acc[cp1 + 4] = vmlaqS32(acc[cp1 + 4], acc2, scale0)
                acc[cp1 + 4] = vmlaqS32(acc[cp1 + 4], acc3, scale1)
            }
            for (q8Row in 0 until 4) {
                val bsumsVecLo = vdupNS16(bsumsArr[q8Row][sb * 2])
                val bsumsVecHi = vdupNS16(bsumsArr[q8Row][sb * 2 + 1])
                biasAcc[2 * q8Row] = vmlalS16(biasAcc[2 * q8Row], bsumsVecLo, vgetLowS16(q5sbMins[0]))
                biasAcc[2 * q8Row] = vmlalS16(biasAcc[2 * q8Row], bsumsVecHi, vgetLowS16(q5sbMins[1]))
                biasAcc[2 * q8Row + 1] = vmlalS16(biasAcc[2 * q8Row + 1], bsumsVecLo, vgetHighS16(q5sbMins[0]))
                biasAcc[2 * q8Row + 1] = vmlalS16(biasAcc[2 * q8Row + 1], bsumsVecHi, vgetHighS16(q5sbMins[1]))
            }
        }
        for (item in acc) {
            val aux = vzipS32(vgetLowS32(item), vgetHighS32(item))
            item = vcombineS32(aux.first, aux.second)
        }
        val reorderAcc = listOf(vcombineS32(vgetLowS32(acc[0]), vgetLowS32(acc[1])), vcombineS32(vgetLowS32(acc[2]), vgetLowS32(acc[3])), vcombineS32(vgetHighS32(acc[0]), vgetHighS32(acc[1])), vcombineS32(vgetHighS32(acc[2]), vgetHighS32(acc[3])), vcombineS32(vgetLowS32(acc[4]), vgetLowS32(acc[5])), vcombineS32(vgetLowS32(acc[6]), vgetLowS32(acc[7])), vcombineS32(vgetHighS32(acc[4]), vgetHighS32(acc[5])), vcombineS32(vgetHighS32(acc[6]), vgetHighS32(acc[7])))
        for (r1 in 0 until 4) {
            for (g in 0 until 2) {
                val q5D = listOf(x.d[g * 4].toF32(), x.d[g * 4 + 1].toF32(), x.d[g * 4 + 2].toF32(), x.d[g * 4 + 3].toF32())
                val q5Dmin = listOf(x.dmin[g * 4].toF32(), x.dmin[g * 4 + 1].toF32(), x.dmin[g * 4 + 2].toF32(), x.dmin[g * 4 + 3].toF32())
                val q8D = y.d[r1]
                val scale = vmulqNF32(vld1qF32(q5D.asPtr()), q8D)
                val dmins = vmulqNF32(vld1qF32(q5Dmin.asPtr()), q8D)
                val idx = 2 * r1 + g
                accF32[idx] = vsubqF32(accF32[idx], vmulqF32(vcvtqF32S32(biasAcc[idx]), dmins))
                accF32[idx] = vmlaqF32(accF32[idx], vcvtqF32S32(reorderAcc[idx]), scale)
            }
        }
    }
    for (r2 in 0 until 4) {
        vst1qF32(MutPtr((dst).rawAddress + (((row + r2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r2])
        vst1qF32(MutPtr((dst).rawAddress + (((row + r2) * n + col + 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r2 + 1])
    }
}

public fun vecDot8Q5kx8Q8k8x4(n: Int, xs: List<BlockQ5Kx8>, ys: List<BlockQ8K>): FloatArray {
    check(n.isMultipleOf(QK_K))
    val out = List(8) { 0.0f }
    val m4b = vdupqNU8(0x0f)
    val mone = vdupqNU8(1)
    val mtwo = vdupqNU8(2)
    var accF320 = vdupqNF32(0.0)
    var accF321 = vdupqNF32(0.0)
    for ((q5, q8) in (xs.zip(ys))) {
        val sbScale0 = vmulqF32(loadF16x4(q5.d.asPtr()), vdupqNF32(q8.d))
        val sbScale1 = vmulqF32(loadF16x4(q5.d.asPtr().add(4)), vdupqNF32(q8.d))
        val sbMin0 = vmulqF32(loadF16x4(q5.dmin.asPtr()), vdupqNF32(q8.d))
        val sbMin1 = vmulqF32(loadF16x4(q5.dmin.asPtr().add(4)), vdupqNF32(q8.d))
        val bsums = vpaddqS16(vld1qS16(q8.bsums.asPtr()), vld1qS16(q8.bsums.asPtr().add(8)))
        val localStorageL1596C8 = AddressableArrayLocal(ShortArray(8) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I16, (8).toInt()))
        var bsumsArr by localStorageL1596C8
        vst1qS16(localStorageL1596C8.elementMutPtr(0), bsums)
        val qh = List(2) { List(8) { vdupqNU8(0) } }
        for (c in 0 until 2) {
            for (i in 0 until 8) {
                qh[c][i] = vld1qU8(q5.qh.asPtr().add(i * 32 + 16 * c))
            }
        }
        val biasAcc = List(2) { vdupqNS32(0) }
        for (sb in 0 until QK_K / 64) {
            val accLo = List(2) { vdupqNS32(0) }
            val accHi = List(2) { vdupqNS32(0) }
            val (mins0, scales0) = decodeQ4kx8Scales(q5.scales.asPtr().add(sb * 24))
            val (mins1, scales1) = decodeQ4kx8Scales(q5.scales.asPtr().add(sb * 24 + 12))
            val q5sbMins = listOf(mins0, mins1)
            val q5sbScales = listOf(scales0, scales1)
            val q8Qs = listOf(vld1qS8(q8.qs.asPtr().add(sb * 64)), vld1qS8(q8.qs.asPtr().add(sb * 64 + 16)), vld1qS8(q8.qs.asPtr().add(sb * 64 + 32)), vld1qS8(q8.qs.asPtr().add(sb * 64 + 48)))
            for (c1 in 0 until 2) {
                for (i1 in 0 until 8) {
                    val q5Col = vld1qU8(q5.qs.asPtr().add(sb * QK_K + i1 * 32 + 16 * c1))
                    val hbitLo = vandqU8(qh[c1][i1], mone)
                    val hbitHi = vshlqNU8(vandqU8(qh[c1][i1], mtwo), 3)
                    qh[c1][i1] = vshrqNU8(qh[c1][i1], 2)
                    val q5Lo = vreinterpretqS8U8(vsliqNU8(vandqU8(q5Col, m4b), hbitLo, 4))
                    val q5Hi = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5Col, 4), hbitHi))
                    when (i1) {
                        0 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[0], 0)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[2], 0)
                        }
                        1 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[0], 1)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[2], 1)
                        }
                        2 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[0], 2)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[2], 2)
                        }
                        3 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[0], 3)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[2], 3)
                        }
                        4 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[1], 0)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[3], 0)
                        }
                        5 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[1], 1)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[3], 1)
                        }
                        6 -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[1], 2)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[3], 2)
                        }
                        else -> {
                            accLo[c1] = sdotLaneqS32(accLo[c1], q5Lo, q8Qs[1], 3)
                            accHi[c1] = sdotLaneqS32(accHi[c1], q5Hi, q8Qs[3], 3)
                        }
                    }
                }
            }
            val sumf0 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(vgetLowS16(q5sbScales[0])), accLo[0]), vmulqS32(vmovlS16(vgetLowS16(q5sbScales[1])), accHi[0])))
            accF320 = vfmaqF32(accF320, sbScale0, sumf0)
            val sumf1 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(vgetHighS16(q5sbScales[0])), accLo[1]), vmulqS32(vmovlS16(vgetHighS16(q5sbScales[1])), accHi[1])))
            accF321 = vfmaqF32(accF321, sbScale1, sumf1)
            val bsumsVecLo = vdupNS16(bsumsArr[2 * sb])
            val bsumsVecHi = vdupNS16(bsumsArr[2 * sb + 1])
            biasAcc[0] = vmlalS16(biasAcc[0], bsumsVecLo, vgetLowS16(q5sbMins[0]))
            biasAcc[0] = vmlalS16(biasAcc[0], bsumsVecHi, vgetLowS16(q5sbMins[1]))
            biasAcc[1] = vmlalS16(biasAcc[1], bsumsVecLo, vgetHighS16(q5sbMins[0]))
            biasAcc[1] = vmlalS16(biasAcc[1], bsumsVecHi, vgetHighS16(q5sbMins[1]))
        }
        accF320 = vmlsqF32(accF320, vcvtqF32S32(biasAcc[0]), sbMin0)
        accF321 = vmlsqF32(accF321, vcvtqF32S32(biasAcc[1]), sbMin1)
    }
    vst1qF32(out.asMutPtr(), accF320)
    vst1qF32(out.asMutPtr().add(4), accF321)
    return out
}

public fun vecDot8Q5kx8Q8k8x8(n: Int, xs: List<BlockQ5Kx8>, ys: List<BlockQ8K>): FloatArray {
    check(n.isMultipleOf(QK_K))
    val out = List(8) { 0.0f }
    val m4b = vdupqNU8(0x0f)
    val mone = vdupqNU8(1)
    val mtwo = vdupqNU8(2)
    var accF320 = vdupqNF32(0.0)
    var accF321 = vdupqNF32(0.0)
    for ((q5, q8) in (xs.zip(ys))) {
        val sbScale0 = vmulqF32(loadF16x4(q5.d.asPtr()), vdupqNF32(q8.d))
        val sbScale1 = vmulqF32(loadF16x4(q5.d.asPtr().add(4)), vdupqNF32(q8.d))
        val sbMin0 = vmulqF32(loadF16x4(q5.dmin.asPtr()), vdupqNF32(q8.d))
        val sbMin1 = vmulqF32(loadF16x4(q5.dmin.asPtr().add(4)), vdupqNF32(q8.d))
        val bsums = vpaddqS16(vld1qS16(q8.bsums.asPtr()), vld1qS16(q8.bsums.asPtr().add(8)))
        val localStorageL1713C8 = AddressableArrayLocal(ShortArray(8) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I16, (8).toInt()))
        var bsumsArr by localStorageL1713C8
        vst1qS16(localStorageL1713C8.elementMutPtr(0), bsums)
        val qh = List(4) { List(4) { vdupqNU8(0) } }
        for ((cp, qhCp) in qh.iterMut().mapIndexed { index, item -> Pair(index, item) }) {
            qhCp[0] = vld1qU8(q5.qh.asPtr().add(16 * cp))
            qhCp[1] = vld1qU8(q5.qh.asPtr().add(16 * cp + 64))
            qhCp[2] = vld1qU8(q5.qh.asPtr().add(16 * cp + 128))
            qhCp[3] = vld1qU8(q5.qh.asPtr().add(16 * cp + 192))
        }
        val biasAcc = List(2) { vdupqNS32(0) }
        for (sb in 0 until QK_K / 64) {
            val accLo = List(4) { vdupqNS32(0) }
            val accHi = List(4) { vdupqNS32(0) }
            val (mins0, scales0) = decodeQ4kx8Scales(q5.scales.asPtr().add(sb * 24))
            val (mins1, scales1) = decodeQ4kx8Scales(q5.scales.asPtr().add(sb * 24 + 12))
            val q5sbMins = listOf(mins0, mins1)
            val q5sbScales = listOf(scales0, scales1)
            val q8Base = q8.qs.asPtr().add(sb * 64)
            val q8Qs = listOf(vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(8)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(16)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(24)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(32)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(40)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(48)))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8Base.add(56)))))
            for (cp1 in 0 until 4) {
                for (i in 0 until 4) {
                    val q5Col = vld1qU8(q5.qs.asPtr().add(sb * QK_K + i * 64 + 16 * cp1))
                    val hbitLo = vandqU8(qh[cp1][i], mone)
                    val hbitHi = vshlqNU8(vandqU8(qh[cp1][i], mtwo), 3)
                    qh[cp1][i] = vshrqNU8(qh[cp1][i], 2)
                    val q5Lo = vreinterpretqS8U8(vsliqNU8(vandqU8(q5Col, m4b), hbitLo, 4))
                    val q5Hi = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5Col, 4), hbitHi))
                    accLo[cp1] = sdotAcc(accLo[cp1], q5Lo, q8Qs[i])
                    accHi[cp1] = sdotAcc(accHi[cp1], q5Hi, q8Qs[i + 4])
                }
            }
            val sumf0 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(vgetLowS16(q5sbScales[0])), vpaddqS32(accLo[0], accLo[1])), vmulqS32(vmovlS16(vgetLowS16(q5sbScales[1])), vpaddqS32(accHi[0], accHi[1]))))
            accF320 = vfmaqF32(accF320, sbScale0, sumf0)
            val sumf1 = vcvtqF32S32(vaddqS32(vmulqS32(vmovlS16(vgetHighS16(q5sbScales[0])), vpaddqS32(accLo[2], accLo[3])), vmulqS32(vmovlS16(vgetHighS16(q5sbScales[1])), vpaddqS32(accHi[2], accHi[3]))))
            accF321 = vfmaqF32(accF321, sbScale1, sumf1)
            val bsumsVecLo = vdupNS16(bsumsArr[2 * sb])
            val bsumsVecHi = vdupNS16(bsumsArr[2 * sb + 1])
            biasAcc[0] = vmlalS16(biasAcc[0], bsumsVecLo, vgetLowS16(q5sbMins[0]))
            biasAcc[0] = vmlalS16(biasAcc[0], bsumsVecHi, vgetLowS16(q5sbMins[1]))
            biasAcc[1] = vmlalS16(biasAcc[1], bsumsVecLo, vgetHighS16(q5sbMins[0]))
            biasAcc[1] = vmlalS16(biasAcc[1], bsumsVecHi, vgetHighS16(q5sbMins[1]))
        }
        accF320 = vmlsqF32(accF320, vcvtqF32S32(biasAcc[0]), sbMin0)
        accF321 = vmlsqF32(accF321, vcvtqF32S32(biasAcc[1]), sbMin1)
    }
    vst1qF32(out.asMutPtr(), accF320)
    vst1qF32(out.asMutPtr().add(4), accF321)
    return out
}

public fun vecDot8Q5kx8Q8k(n: Int, xs: List<BlockQ5Kx8>, ys: List<BlockQ8K>): FloatArray =
    if (get().i8mm) {
        vecDot8Q5kx8Q8k8x8(n, xs, ys)
    } else {
        vecDot8Q5kx8Q8k8x4(n, xs, ys)
    }

public fun matmulQ5kX8Gemv(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ5Kx8>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK_K))
    check(n.isMultipleOf(8))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    threadLocal(LHS_Q5K_GEMV_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8K>(typeLayoutT = ObjectSlotLayout<BlockQ8K>())
    val requiredLen = (kInBlocks * elemSize).divCeil(8)
    return LHS_Q5K_GEMV_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsB: MutableList<BlockQ8K> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
        BlockQ8K.fromFloat(lhs, lhsB)
        val pool = barrierPool()
        val lhsPtr = lhsB.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        val x8BlockBytes = sizeOf<BlockQ5Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ5Kx8>())
        // equal-sized gemv units: a static split skips the shared cursor, which costs
        // several us per small decode matmul
        pool.executeStatic(nGroups, { range ->
            val lhsRow = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (g in range) {
                val xs = fromRawParts(ConstPtr((repackedPtr + g * kInBlocks * x8BlockBytes).toLong()), kInBlocks)
                val results = vecDot8Q5kx8Q8k(k, xs, lhsRow)
                run {
                    copyNonoverlapping(results.asPtr(), MutPtr((dstPtr1).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                }
            }
        })
        Result.Ok(Unit)
    })
}

internal fun matmulQ5kX8(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ5Kx8>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK_K))
    check(n.isMultipleOf(8))
    val features = get()
    if (features.dotprod && m == 1) {
        return matmulQ5kX8Gemv(Triple(m, k, n), lhs, repacked, dst)
    }
    val useI8mm = features.i8mm
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    val rowGroups = m / 4
    threadLocal(LHS_Q5K_X8_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8Kx4>(typeLayoutT = ObjectSlotLayout<BlockQ8Kx4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_Q5K_X8_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ8Kx4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            if (useI8mm) {
                quantizeQ8kX4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 8)
            } else {
                quantizeQ8kX4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 4)
            }
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 8
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8Kx4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ5Kx8>()).strideBytes)), kInBlocks)
                if (useI8mm) {
                    storeQ5kx84x8I8mm(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
                } else {
                    storeQ5kx84x8(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
                }
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ6kx84x8(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ8Kx4>, rhs: List<BlockQ6Kx8>, kInBlocks: Int) {
    val m4b = vdupqNU8(0x0f)
    val maskLo = vdupqNU8(0x03)
    val maskHi = vdupqNU8(0x30)
    val m32s = vdupqNS8(32)
    val accF32 = List(8) { vdupqNF32(0.0) }
    for (b in 0 until kInBlocks) {
        val x = rhs[b]
        val y = lhs[b]
        val q6D0123 = vld1qF32(listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32()).asPtr())
        val q6D4567 = vld1qF32(listOf(x.d[4].toF32(), x.d[5].toF32(), x.d[6].toF32(), x.d[7].toF32()).asPtr())
        val sbdScale0123 = listOf(vmulqNF32(q6D0123, y.d[0]), vmulqNF32(q6D0123, y.d[1]), vmulqNF32(q6D0123, y.d[2]), vmulqNF32(q6D0123, y.d[3]))
        val sbdScale4567 = listOf(vmulqNF32(q6D4567, y.d[0]), vmulqNF32(q6D4567, y.d[1]), vmulqNF32(q6D4567, y.d[2]), vmulqNF32(q6D4567, y.d[3]))
        val accS32 = List(8) { vdupqNS32(0) }
        for (half in 0 until 2) {
            val qlBase = x.ql.asPtr().add(half * 512)
            val qhBase = x.qh.asPtr().add(half * 256)
            for (sb in 0 until QK_K / 64) {
                val accLo = List(8) { vdupqNS32(0) }
                val accHi = List(8) { vdupqNS32(0) }
                val q8BaseL = y.qs.asPtr().add(half * 512 + sb * 64)
                val q8BaseH = y.qs.asPtr().add(half * 512 + 256 + sb * 64)
                val qlOffBase = sb * QK_K / 2
                val qhOffBase = qlOffBase and 255
                for (kIdx in 0 until 4) {
                    val q8L = vld1qS8(q8BaseL.add(16 * kIdx))
                    val q8H = vld1qS8(q8BaseH.add(16 * kIdx))
                    val q6Ql0123 = vld1qU8(qlBase.add(qlOffBase + kIdx * 32))
                    val q6Ql4567 = vld1qU8(qlBase.add(qlOffBase + kIdx * 32 + 16))
                    var q6Qh0123 = vld1qU8(qhBase.add(qhOffBase + kIdx * 32))
                    var q6Qh4567 = vld1qU8(qhBase.add(qhOffBase + kIdx * 32 + 16))
                    if (sb > 1) {
                        q6Qh0123 = vshrqNU8(q6Qh0123, 2)
                        q6Qh4567 = vshrqNU8(q6Qh4567, 2)
                    }
                    val hbitLo0123 = vandqU8(q6Qh0123, maskLo)
                    val hbitHi0123 = vandqU8(q6Qh0123, maskHi)
                    val hbitLo4567 = vandqU8(q6Qh4567, maskLo)
                    val hbitHi4567 = vandqU8(q6Qh4567, maskHi)
                    val q60123Lo = vsubqS8(vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql0123, m4b), vshlqNU8(hbitLo0123, 4))), m32s)
                    val q60123Hi = vsubqS8(vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql0123, 4), hbitHi0123)), m32s)
                    accLo[0] = sdotLaneqS32(accLo[0], q60123Lo, q8L, 0)
                    accLo[1] = sdotLaneqS32(accLo[1], q60123Lo, q8L, 1)
                    accLo[2] = sdotLaneqS32(accLo[2], q60123Lo, q8L, 2)
                    accLo[3] = sdotLaneqS32(accLo[3], q60123Lo, q8L, 3)
                    accHi[0] = sdotLaneqS32(accHi[0], q60123Hi, q8H, 0)
                    accHi[1] = sdotLaneqS32(accHi[1], q60123Hi, q8H, 1)
                    accHi[2] = sdotLaneqS32(accHi[2], q60123Hi, q8H, 2)
                    accHi[3] = sdotLaneqS32(accHi[3], q60123Hi, q8H, 3)
                    val q64567Lo = vsubqS8(vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql4567, m4b), vshlqNU8(hbitLo4567, 4))), m32s)
                    val q64567Hi = vsubqS8(vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql4567, 4), hbitHi4567)), m32s)
                    accLo[4] = sdotLaneqS32(accLo[4], q64567Lo, q8L, 0)
                    accLo[5] = sdotLaneqS32(accLo[5], q64567Lo, q8L, 1)
                    accLo[6] = sdotLaneqS32(accLo[6], q64567Lo, q8L, 2)
                    accLo[7] = sdotLaneqS32(accLo[7], q64567Lo, q8L, 3)
                    accHi[4] = sdotLaneqS32(accHi[4], q64567Hi, q8H, 0)
                    accHi[5] = sdotLaneqS32(accHi[5], q64567Hi, q8H, 1)
                    accHi[6] = sdotLaneqS32(accHi[6], q64567Hi, q8H, 2)
                    accHi[7] = sdotLaneqS32(accHi[7], q64567Hi, q8H, 3)
                }
                val scaleIdxL = half * 8 + sb
                val scaleIdxH = half * 8 + sb + 4
                for (g in 0 until 2) {
                    val baseL = scaleIdxL * 8 + g * 4
                    val baseH = scaleIdxH * 8 + g * 4
                    val scaleVecL = vld1qS32(listOf(x.scales[baseL].toInt(), x.scales[baseL + 1].toInt(), x.scales[baseL + 2].toInt(), x.scales[baseL + 3].toInt()).asPtr())
                    val scaleVecH = vld1qS32(listOf(x.scales[baseH].toInt(), x.scales[baseH + 1].toInt(), x.scales[baseH + 2].toInt(), x.scales[baseH + 3].toInt()).asPtr())
                    val accOffset = g * 4
                    for (row1 in 0 until 4) {
                        val idx = row1 * 2 + g
                        accS32[idx] = vmlaqS32(accS32[idx], accLo[accOffset + row1], scaleVecL)
                        accS32[idx] = vmlaqS32(accS32[idx], accHi[accOffset + row1], scaleVecH)
                    }
                }
            }
        }
        for (r in 0 until 4) {
            val idx0 = 2 * r
            val idx1 = 2 * r + 1
            accF32[idx0] = vmlaqF32(accF32[idx0], vcvtqF32S32(accS32[idx0]), sbdScale0123[r])
            accF32[idx1] = vmlaqF32(accF32[idx1], vcvtqF32S32(accS32[idx1]), sbdScale4567[r])
        }
    }
    for (r1 in 0 until 4) {
        vst1qF32(MutPtr((dst).rawAddress + (((row + r1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r1])
        vst1qF32(MutPtr((dst).rawAddress + (((row + r1) * n + col + 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r1 + 1])
    }
}

public fun storeQ6kx84x8I8mm(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ8Kx4>, rhs: List<BlockQ6Kx8>, kInBlocks: Int) {
    val m4b = vdupqNU8(0x0f)
    val maskLo = vdupqNU8(0x03)
    val maskHi = vdupqNU8(0x30)
    val m32s = vdupqNS8(32)
    val accF32 = List(8) { vdupqNF32(0.0) }
    for (b in 0 until kInBlocks) {
        val x = rhs[b]
        val y = lhs[b]
        val acc = List(8) { vdupqNS32(0) }
        for (half in 0 until 2) {
            val qlBase = x.ql.asPtr().add(half * 512)
            val qhBase = x.qh.asPtr().add(half * 256)
            for (sb in 0 until QK_K / 64) {
                val q8BaseL = y.qs.asPtr().add(half * 512 + sb * 64)
                val q8BaseH = y.qs.asPtr().add(half * 512 + 256 + sb * 64)
                val q8L01 = listOf(vld1qS8(q8BaseL), vld1qS8(q8BaseL.add(32)))
                val q8L23 = listOf(vld1qS8(q8BaseL.add(16)), vld1qS8(q8BaseL.add(48)))
                val q8H01 = listOf(vld1qS8(q8BaseH), vld1qS8(q8BaseH.add(32)))
                val q8H23 = listOf(vld1qS8(q8BaseH.add(16)), vld1qS8(q8BaseH.add(48)))
                val qlOffBase = sb * QK_K / 2
                val qhOffBase = qlOffBase and 255
                for (cp in 0 until 4) {
                    var q6Qh0 = vld1qU8(qhBase.add(qhOffBase + 16 * cp))
                    var q6Qh1 = vld1qU8(qhBase.add(qhOffBase + 64 + 16 * cp))
                    if (sb > 1) {
                        q6Qh0 = vshrqNU8(q6Qh0, 2)
                        q6Qh1 = vshrqNU8(q6Qh1, 2)
                    }
                    val q6Ql0 = vld1qU8(qlBase.add(qlOffBase + 16 * cp))
                    val q6Ql1 = vld1qU8(qlBase.add(qlOffBase + 64 + 16 * cp))
                    val q6L0 = vsubqS8(vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql0, m4b), vshlqNU8(vandqU8(q6Qh0, maskLo), 4))), m32s)
                    val q6L1 = vsubqS8(vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql1, m4b), vshlqNU8(vandqU8(q6Qh1, maskLo), 4))), m32s)
                    val q6H0 = vsubqS8(vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql0, 4), vandqU8(q6Qh0, maskHi))), m32s)
                    val q6H1 = vsubqS8(vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql1, 4), vandqU8(q6Qh1, maskHi))), m32s)
                    var sbAcc0l = smmlaS32(vdupqNS32(0), q6L0, q8L01[0])
                    sbAcc0l = smmlaS32(sbAcc0l, q6L1, q8L01[1])
                    var sbAcc0h = smmlaS32(vdupqNS32(0), q6H0, q8H01[0])
                    sbAcc0h = smmlaS32(sbAcc0h, q6H1, q8H01[1])
                    var sbAcc1l = smmlaS32(vdupqNS32(0), q6L0, q8L23[0])
                    sbAcc1l = smmlaS32(sbAcc1l, q6L1, q8L23[1])
                    var sbAcc1h = smmlaS32(vdupqNS32(0), q6H0, q8H23[0])
                    sbAcc1h = smmlaS32(sbAcc1h, q6H1, q8H23[1])
                    val scaleIdxL = half * 8 + sb
                    val scaleIdxH = half * 8 + sb + 4
                    val s0 = x.scales[scaleIdxL * 8 + cp * 2].toInt()
                    val s1 = x.scales[scaleIdxL * 8 + cp * 2 + 1].toInt()
                    val s2 = x.scales[scaleIdxH * 8 + cp * 2].toInt()
                    val s3 = x.scales[scaleIdxH * 8 + cp * 2 + 1].toInt()
                    val scaleVecL = vcombineS32(vdupNS32(s0), vdupNS32(s1))
                    val scaleVecH = vcombineS32(vdupNS32(s2), vdupNS32(s3))
                    acc[cp] = vmlaqS32(acc[cp], sbAcc0l, scaleVecL)
                    acc[cp] = vmlaqS32(acc[cp], sbAcc0h, scaleVecH)
                    acc[cp + 4] = vmlaqS32(acc[cp + 4], sbAcc1l, scaleVecL)
                    acc[cp + 4] = vmlaqS32(acc[cp + 4], sbAcc1h, scaleVecH)
                }
            }
        }
        for (item in acc) {
            val aux = vzipS32(vgetLowS32(item), vgetHighS32(item))
            item = vcombineS32(aux.first, aux.second)
        }
        val reorderAcc = listOf(vcombineS32(vgetLowS32(acc[0]), vgetLowS32(acc[1])), vcombineS32(vgetLowS32(acc[2]), vgetLowS32(acc[3])), vcombineS32(vgetHighS32(acc[0]), vgetHighS32(acc[1])), vcombineS32(vgetHighS32(acc[2]), vgetHighS32(acc[3])), vcombineS32(vgetLowS32(acc[4]), vgetLowS32(acc[5])), vcombineS32(vgetLowS32(acc[6]), vgetLowS32(acc[7])), vcombineS32(vgetHighS32(acc[4]), vgetHighS32(acc[5])), vcombineS32(vgetHighS32(acc[6]), vgetHighS32(acc[7])))
        for (r in 0 until 4) {
            for (g in 0 until 2) {
                val q6D = listOf(x.d[g * 4].toF32(), x.d[g * 4 + 1].toF32(), x.d[g * 4 + 2].toF32(), x.d[g * 4 + 3].toF32())
                val scale = vmulqNF32(vld1qF32(q6D.asPtr()), y.d[r])
                val idx = 2 * r + g
                accF32[idx] = vmlaqF32(accF32[idx], vcvtqF32S32(reorderAcc[idx]), scale)
            }
        }
    }
    for (r1 in 0 until 4) {
        vst1qF32(MutPtr((dst).rawAddress + (((row + r1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r1])
        vst1qF32(MutPtr((dst).rawAddress + (((row + r1) * n + col + 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), accF32[2 * r1 + 1])
    }
}

public fun loadF16x4(ptr: ConstPtr): Float32x4T {
    val raw = vld1U64(ConstPtr(ptr.rawAddress, ptr.rawMeta))
    var result: Float32x4T
    asm("fcvtl {out:v}.4s, {inp:v}.4h", inp, vreg, raw, out, out, vreg, result, options(nostack, nomem))
    return result
}

public fun loadI32x2(a: Int, b: Int): Int32x2T = vld1S32(listOf(a, b).asPtr())

public fun loadI32x4(a: Int, b: Int, c: Int, d: Int): Int32x4T = vld1qS32(listOf(a, b, c, d).asPtr())

public fun vecDot8Q6kx8Q8k8x8(n: Int, xs: List<BlockQ6Kx8>, ys: List<BlockQ8K>): FloatArray {
    check(n.isMultipleOf(QK_K))
    val out = List(8) { 0.0f }
    val m4b = vdupqNU8(0x0f)
    val maskLo = vdupqNU8(0x03)
    val maskHi = vdupqNU8(0x30)
    var accF320 = vdupqNF32(0.0)
    var accF321 = vdupqNF32(0.0)
    for ((q6, q8) in (xs.zip(ys))) {
        val sbScale0 = vmulqF32(loadF16x4(q6.d.asPtr()), vdupqNF32(q8.d))
        val sbScale1 = vmulqF32(loadF16x4(q6.d.asPtr().add(4)), vdupqNF32(q8.d))
        val acc = List(4) { vdupNS32(0) }
        val localStorageL2323C8 = AddressableArrayLocal(ShortArray((16 * 8).toInt()) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I16, (16 * 8).toInt()))
        var q6Scales by localStorageL2323C8
        for (i in 0 until 16) {
            val scales = vmovlS8(vld1S8(q6.scales.asPtr().add(i * 8)))
            vst1qS16(MutPtr((localStorageL2323C8.elementMutPtr(0)).rawAddress + ((i * 8).toLong() * (PrimitiveLayouts.I16).strideBytes)), scales)
        }
        val bias = List(8) { 0 }
        for (i1 in 0 until 16) {
            val bsum = q8.bsums[i1].toInt()
            for (c in 0 until 8) {
                bias[c] += bsum * q6Scales[i1 * 8 + c].toInt()
            }
        }
        for (half in 0 until 2) {
            val qlBase = q6.ql.asPtr().add(half * 512)
            val qhBase = q6.qh.asPtr().add(half * 256)
            for (sb in 0 until QK_K / 64) {
                val q8BaseL = q8.qs.asPtr().add(half * 128 + sb * 16)
                val q8BaseH = q8BaseL.add(64)
                val q8L = listOf(vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8BaseL))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8BaseL.add(8)))))
                val q8H = listOf(vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8BaseH))), vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8BaseH.add(8)))))
                val qlOffBase = sb * QK_K / 2
                val qhOffBase = qlOffBase and 255
                val q6Ql0 = vld1qU8X4(qlBase.add(qlOffBase))
                val q6Ql1 = vld1qU8X4(qlBase.add(qlOffBase + 64))
                val q6Qh0Raw = vld1qU8X4(qhBase.add(qhOffBase))
                val q6Qh1Raw = vld1qU8X4(qhBase.add(qhOffBase + 64))
                val q6Qh0 =
                    if (sb > 1) {
                        listOf(vshrqNU8(q6Qh0Raw.first, 2), vshrqNU8(q6Qh0Raw.second, 2), vshrqNU8(q6Qh0Raw.third, 2), vshrqNU8(q6Qh0Raw.field3, 2))
                    } else {
                        listOf(q6Qh0Raw.first, q6Qh0Raw.second, q6Qh0Raw.third, q6Qh0Raw.field3)
                    }
                val q6Qh1 =
                    if (sb > 1) {
                        listOf(vshrqNU8(q6Qh1Raw.first, 2), vshrqNU8(q6Qh1Raw.second, 2), vshrqNU8(q6Qh1Raw.third, 2), vshrqNU8(q6Qh1Raw.field3, 2))
                    } else {
                        listOf(q6Qh1Raw.first, q6Qh1Raw.second, q6Qh1Raw.third, q6Qh1Raw.field3)
                    }
                val q6Ql = listOf(q6Ql0.first, q6Ql0.second, q6Ql0.third, q6Ql0.field3, q6Ql1.first, q6Ql1.second, q6Ql1.third, q6Ql1.field3)
                val q6Qh = listOf(q6Qh0.first, q6Qh0.second, q6Qh0.third, q6Qh0.field3, q6Qh1.first, q6Qh1.second, q6Qh1.third, q6Qh1.field3)
                for (cp in 0 until 4) {
                    val q6L0 = vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql[cp], m4b), vshlqNU8(vandqU8(q6Qh[cp], maskLo), 4)))
                    val q6L1 = vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql[cp + 4], m4b), vshlqNU8(vandqU8(q6Qh[cp + 4], maskLo), 4)))
                    val q6H0 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql[cp], 4), vandqU8(q6Qh[cp], maskHi)))
                    val q6H1 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql[cp + 4], 4), vandqU8(q6Qh[cp + 4], maskHi)))
                    var sbAccL = vdupqNS32(0)
                    sbAccL = sdotAcc(sbAccL, q6L0, q8L[0])
                    sbAccL = sdotAcc(sbAccL, q6L1, q8L[1])
                    var sbAccH = vdupqNS32(0)
                    sbAccH = sdotAcc(sbAccH, q6H0, q8H[0])
                    sbAccH = sdotAcc(sbAccH, q6H1, q8H[1])
                    val sumL = vpaddS32(vgetLowS32(sbAccL), vgetHighS32(sbAccL))
                    val sumH = vpaddS32(vgetLowS32(sbAccH), vgetHighS32(sbAccH))
                    val scaleIdxL = half * 8 + sb
                    val scaleIdxH = half * 8 + sb + 4
                    val scaleL = loadI32x2(q6Scales[scaleIdxL * 8 + cp * 2].toInt(), q6Scales[scaleIdxL * 8 + cp * 2 + 1].toInt())
                    val scaleH = loadI32x2(q6Scales[scaleIdxH * 8 + cp * 2].toInt(), q6Scales[scaleIdxH * 8 + cp * 2 + 1].toInt())
                    acc[cp] = vmlaS32(acc[cp], sumL, scaleL)
                    acc[cp] = vmlaS32(acc[cp], sumH, scaleH)
                }
            }
        }
        acc[0] = vsubS32(acc[0], loadI32x2(bias[0] shl 5, bias[1] shl 5))
        acc[1] = vsubS32(acc[1], loadI32x2(bias[2] shl 5, bias[3] shl 5))
        acc[2] = vsubS32(acc[2], loadI32x2(bias[4] shl 5, bias[5] shl 5))
        acc[3] = vsubS32(acc[3], loadI32x2(bias[6] shl 5, bias[7] shl 5))
        val w01 = vmulF32(vcvtF32S32(acc[0]), vgetLowF32(sbScale0))
        val w23 = vmulF32(vcvtF32S32(acc[1]), vgetHighF32(sbScale0))
        val w45 = vmulF32(vcvtF32S32(acc[2]), vgetLowF32(sbScale1))
        val w67 = vmulF32(vcvtF32S32(acc[3]), vgetHighF32(sbScale1))
        accF320 = vaddqF32(accF320, vcombineF32(w01, w23))
        accF321 = vaddqF32(accF321, vcombineF32(w45, w67))
    }
    vst1qF32(out.asMutPtr(), accF320)
    vst1qF32(out.asMutPtr().add(4), accF321)
    return out
}

public fun vecDot8Q6kx8Q8k8x4(n: Int, xs: List<BlockQ6Kx8>, ys: List<BlockQ8K>): FloatArray {
    check(n.isMultipleOf(QK_K))
    val out = List(8) { 0.0f }
    val m4b = vdupqNU8(0x0f)
    val maskLo = vdupqNU8(0x03)
    val maskHi = vdupqNU8(0x30)
    var accF320 = vdupqNF32(0.0)
    var accF321 = vdupqNF32(0.0)
    for ((q6, q8) in (xs.zip(ys))) {
        val sbScale0 = vmulqF32(loadF16x4(q6.d.asPtr()), vdupqNF32(q8.d))
        val sbScale1 = vmulqF32(loadF16x4(q6.d.asPtr().add(4)), vdupqNF32(q8.d))
        val acc = List(2) { vdupqNS32(0) }
        val localStorageL2460C8 = AddressableArrayLocal(ShortArray((16 * 8).toInt()) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I16, (16 * 8).toInt()))
        var q6Scales by localStorageL2460C8
        for (i in 0 until 16) {
            val scales = vmovlS8(vld1S8(q6.scales.asPtr().add(i * 8)))
            vst1qS16(MutPtr((localStorageL2460C8.elementMutPtr(0)).rawAddress + ((i * 8).toLong() * (PrimitiveLayouts.I16).strideBytes)), scales)
        }
        val bias = List(8) { 0 }
        for (i1 in 0 until 16) {
            val bsum = q8.bsums[i1].toInt()
            for (c in 0 until 8) {
                bias[c] += bsum * q6Scales[i1 * 8 + c].toInt()
            }
        }
        for (half in 0 until 2) {
            val qlBase = q6.ql.asPtr().add(half * 512)
            val qhBase = q6.qh.asPtr().add(half * 256)
            for (sb in 0 until QK_K / 64) {
                val q8BaseL = q8.qs.asPtr().add(half * 128 + sb * 16)
                val q8BaseH = q8BaseL.add(64)
                val q8L = listOf(vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseL))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseL.add(4)))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseL.add(8)))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseL.add(12)))))
                val q8H = listOf(vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseH))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseH.add(4)))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseH.add(8)))), vreinterpretqS8S32(vld1qDupS32(constPtrFromRustCast(q8BaseH.add(12)))))
                val qlOffBase = sb * QK_K / 2
                val qhOffBase = qlOffBase and 255
                val q6Ql0 = vld1qU8X4(qlBase.add(qlOffBase))
                val q6Ql1 = vld1qU8X4(qlBase.add(qlOffBase + 64))
                val q6Qh0Raw = vld1qU8X4(qhBase.add(qhOffBase))
                val q6Qh1Raw = vld1qU8X4(qhBase.add(qhOffBase + 64))
                val q6Qh0 =
                    if (sb > 1) {
                        listOf(vshrqNU8(q6Qh0Raw.first, 2), vshrqNU8(q6Qh0Raw.second, 2), vshrqNU8(q6Qh0Raw.third, 2), vshrqNU8(q6Qh0Raw.field3, 2))
                    } else {
                        listOf(q6Qh0Raw.first, q6Qh0Raw.second, q6Qh0Raw.third, q6Qh0Raw.field3)
                    }
                val q6Qh1 =
                    if (sb > 1) {
                        listOf(vshrqNU8(q6Qh1Raw.first, 2), vshrqNU8(q6Qh1Raw.second, 2), vshrqNU8(q6Qh1Raw.third, 2), vshrqNU8(q6Qh1Raw.field3, 2))
                    } else {
                        listOf(q6Qh1Raw.first, q6Qh1Raw.second, q6Qh1Raw.third, q6Qh1Raw.field3)
                    }
                val q6Ql = listOf(q6Ql0.first, q6Ql0.second, q6Ql0.third, q6Ql0.field3, q6Ql1.first, q6Ql1.second, q6Ql1.third, q6Ql1.field3)
                val q6Qh = listOf(q6Qh0.first, q6Qh0.second, q6Qh0.third, q6Qh0.field3, q6Qh1.first, q6Qh1.second, q6Qh1.third, q6Qh1.field3)
                for (g in 0 until 2) {
                    var sbAccL = vdupqNS32(0)
                    var sbAccH = vdupqNS32(0)
                    for (chunk in 0 until 4) {
                        val idx = chunk * 2 + g
                        val q6L = vreinterpretqS8U8(vorrqU8(vandqU8(q6Ql[idx], m4b), vshlqNU8(vandqU8(q6Qh[idx], maskLo), 4)))
                        val q6H = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6Ql[idx], 4), vandqU8(q6Qh[idx], maskHi)))
                        sbAccL = sdotAcc(sbAccL, q6L, q8L[chunk])
                        sbAccH = sdotAcc(sbAccH, q6H, q8H[chunk])
                    }
                    val scaleIdxL = half * 8 + sb
                    val scaleIdxH = half * 8 + sb + 4
                    val scaleL = loadI32x4(q6Scales[scaleIdxL * 8 + g * 4].toInt(), q6Scales[scaleIdxL * 8 + g * 4 + 1].toInt(), q6Scales[scaleIdxL * 8 + g * 4 + 2].toInt(), q6Scales[scaleIdxL * 8 + g * 4 + 3].toInt())
                    val scaleH = loadI32x4(q6Scales[scaleIdxH * 8 + g * 4].toInt(), q6Scales[scaleIdxH * 8 + g * 4 + 1].toInt(), q6Scales[scaleIdxH * 8 + g * 4 + 2].toInt(), q6Scales[scaleIdxH * 8 + g * 4 + 3].toInt())
                    acc[g] = vmlaqS32(acc[g], sbAccL, scaleL)
                    acc[g] = vmlaqS32(acc[g], sbAccH, scaleH)
                }
            }
        }
        acc[0] = vsubqS32(acc[0], loadI32x4(bias[0] shl 5, bias[1] shl 5, bias[2] shl 5, bias[3] shl 5))
        acc[1] = vsubqS32(acc[1], loadI32x4(bias[4] shl 5, bias[5] shl 5, bias[6] shl 5, bias[7] shl 5))
        accF320 = vfmaqF32(accF320, vcvtqF32S32(acc[0]), sbScale0)
        accF321 = vfmaqF32(accF321, vcvtqF32S32(acc[1]), sbScale1)
    }
    vst1qF32(out.asMutPtr(), accF320)
    vst1qF32(out.asMutPtr().add(4), accF321)
    return out
}

internal fun vecDot8Q6kx8Q8k(n: Int, xs: List<BlockQ6Kx8>, ys: List<BlockQ8K>): FloatArray =
    if (get().i8mm) {
        vecDot8Q6kx8Q8k8x8(n, xs, ys)
    } else {
        vecDot8Q6kx8Q8k8x4(n, xs, ys)
    }

public fun matmulQ6kX8Gemv(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ6Kx8>, dst: MutableList<Float>): Result<Unit> {
    check(m == 1)
    check(k.isMultipleOf(QK_K))
    check(n.isMultipleOf(8))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    threadLocal(LHS_Q6K_GEMV_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8K>(typeLayoutT = ObjectSlotLayout<BlockQ8K>())
    val requiredLen = (kInBlocks * elemSize).divCeil(8)
    return LHS_Q6K_GEMV_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsB: MutableList<BlockQ8K> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
        BlockQ8K.fromFloat(lhs, lhsB)
        val pool = barrierPool()
        val lhsPtr = lhsB.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        val x8BlockBytes = sizeOf<BlockQ6Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ6Kx8>())
        // equal-sized gemv units: a static split skips the shared cursor, which costs
        // several us per small decode matmul
        pool.executeStatic(nGroups, { range ->
            val lhsRow = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (g in range) {
                val xs = fromRawParts(ConstPtr((repackedPtr + g * kInBlocks * x8BlockBytes).toLong()), kInBlocks)
                val results = vecDot8Q6kx8Q8k(k, xs, lhsRow)
                run {
                    copyNonoverlapping(results.asPtr(), MutPtr((dstPtr1).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                }
            }
        })
        Result.Ok(Unit)
    })
}

internal fun matmulQ6kX8(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ6Kx8>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK_K))
    check(n.isMultipleOf(8))
    val features = get()
    if (features.dotprod && m == 1) {
        return matmulQ6kX8Gemv(Triple(m, k, n), lhs, repacked, dst)
    }
    val useI8mm = features.i8mm
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    val rowGroups = m / 4
    threadLocal(LHS_Q6K_X8_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8Kx4>(typeLayoutT = ObjectSlotLayout<BlockQ8Kx4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_Q6K_X8_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ8Kx4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            if (useI8mm) {
                quantizeQ8kX4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 8)
            } else {
                quantizeQ8kX4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 4)
            }
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 8
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ8Kx4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ6Kx8>()).strideBytes)), kInBlocks)
                if (useI8mm) {
                    storeQ6kx84x8I8mm(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
                } else {
                    storeQ6kx84x8(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
                }
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ80x44x4(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ80x4>, rhs: List<BlockQ80x4>, kInBlocks: Int) {
    var sum0 = vdupqNF32(0.0)
    var sum1 = vdupqNF32(0.0)
    var sum2 = vdupqNF32(0.0)
    var sum3 = vdupqNF32(0.0)
    for (i in 0 until kInBlocks) {
        val x = rhs[i]
        val y = lhs[i]
        val bd = listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32())
        val bd1 = vld1qF32(bd.asPtr())
        val ad = listOf(y.d[0].toF32(), y.d[1].toF32(), y.d[2].toF32(), y.d[3].toF32())
        var ret0 = vdupqNS32(0)
        var ret1 = vdupqNS32(0)
        var ret2 = vdupqNS32(0)
        var ret3 = vdupqNS32(0)
        val y0 = vld1qS8X4(y.qs.asPtr())
        val x0 = vld1qS8X4(x.qs.asPtr())
        run {
            ret0 = sdotLaneqS32(ret0, x0.first, y0.first, 0)
            ret1 = sdotLaneqS32(ret1, x0.first, y0.first, 1)
            ret2 = sdotLaneqS32(ret2, x0.first, y0.first, 2)
            ret3 = sdotLaneqS32(ret3, x0.first, y0.first, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x0.second, y0.second, 0)
            ret1 = sdotLaneqS32(ret1, x0.second, y0.second, 1)
            ret2 = sdotLaneqS32(ret2, x0.second, y0.second, 2)
            ret3 = sdotLaneqS32(ret3, x0.second, y0.second, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x0.third, y0.third, 0)
            ret1 = sdotLaneqS32(ret1, x0.third, y0.third, 1)
            ret2 = sdotLaneqS32(ret2, x0.third, y0.third, 2)
            ret3 = sdotLaneqS32(ret3, x0.third, y0.third, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x0.field3, y0.field3, 0)
            ret1 = sdotLaneqS32(ret1, x0.field3, y0.field3, 1)
            ret2 = sdotLaneqS32(ret2, x0.field3, y0.field3, 2)
            ret3 = sdotLaneqS32(ret3, x0.field3, y0.field3, 3)
        }
        val y1 = vld1qS8X4(y.qs.asPtr().add(64))
        val x1 = vld1qS8X4(x.qs.asPtr().add(64))
        run {
            ret0 = sdotLaneqS32(ret0, x1.first, y1.first, 0)
            ret1 = sdotLaneqS32(ret1, x1.first, y1.first, 1)
            ret2 = sdotLaneqS32(ret2, x1.first, y1.first, 2)
            ret3 = sdotLaneqS32(ret3, x1.first, y1.first, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x1.second, y1.second, 0)
            ret1 = sdotLaneqS32(ret1, x1.second, y1.second, 1)
            ret2 = sdotLaneqS32(ret2, x1.second, y1.second, 2)
            ret3 = sdotLaneqS32(ret3, x1.second, y1.second, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x1.third, y1.third, 0)
            ret1 = sdotLaneqS32(ret1, x1.third, y1.third, 1)
            ret2 = sdotLaneqS32(ret2, x1.third, y1.third, 2)
            ret3 = sdotLaneqS32(ret3, x1.third, y1.third, 3)
        }
        run {
            ret0 = sdotLaneqS32(ret0, x1.field3, y1.field3, 0)
            ret1 = sdotLaneqS32(ret1, x1.field3, y1.field3, 1)
            ret2 = sdotLaneqS32(ret2, x1.field3, y1.field3, 2)
            ret3 = sdotLaneqS32(ret3, x1.field3, y1.field3, 3)
        }
        sum0 = vmlaqF32(sum0, vcvtqF32S32(ret0), vmulqNF32(bd1, ad[0]))
        sum1 = vmlaqF32(sum1, vcvtqF32S32(ret1), vmulqNF32(bd1, ad[1]))
        sum2 = vmlaqF32(sum2, vcvtqF32S32(ret2), vmulqNF32(bd1, ad[2]))
        sum3 = vmlaqF32(sum3, vcvtqF32S32(ret3), vmulqNF32(bd1, ad[3]))
    }
    vst1qF32(MutPtr((dst).rawAddress + ((row * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum0)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum1)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum2)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 3) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum3)
}

public fun vecDot4Q80x4Q804x4(n: Int, xs: List<BlockQ80x4>, ys: List<BlockQ80>): FloatArray {
    check(n.isMultipleOf(QK8_0))
    val out = List(4) { 0.0f }
    var acc = vdupqNF32(0.0)
    for ((x, y) in (xs.zip(ys))) {
        val bd = loadF16x4(x.d.asPtr())
        val ad = vdupqNF32(y.d.toF32())
        val y0 = vld1qS8(y.qs.asPtr())
        val y1 = vld1qS8(y.qs.asPtr().add(16))
        val x0 = vld1qS8X4(x.qs.asPtr())
        val x1 = vld1qS8X4(x.qs.asPtr().add(64))
        var ret = vdupqNS32(0)
        ret = sdotLaneqS32(ret, x0.first, y0, 0)
        ret = sdotLaneqS32(ret, x0.second, y0, 1)
        ret = sdotLaneqS32(ret, x0.third, y0, 2)
        ret = sdotLaneqS32(ret, x0.field3, y0, 3)
        ret = sdotLaneqS32(ret, x1.first, y1, 0)
        ret = sdotLaneqS32(ret, x1.second, y1, 1)
        ret = sdotLaneqS32(ret, x1.third, y1, 2)
        ret = sdotLaneqS32(ret, x1.field3, y1, 3)
        acc = vmlaqF32(acc, vcvtqF32S32(ret), vmulqF32(ad, bd))
    }
    vst1qF32(out.asMutPtr(), acc)
    return out
}

public fun vecDot4Q80x4Q804x8(n: Int, xs: List<BlockQ80x4>, ys: List<BlockQ80>): FloatArray {
    check(n.isMultipleOf(QK8_0))
    val out = List(4) { 0.0f }
    var acc = vdupqNF32(0.0)
    for ((x, y) in (xs.zip(ys))) {
        val bd = loadF16x4(x.d.asPtr())
        val ad = vdupqNF32(y.d.toF32())
        val a0 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr())))
        val a1 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(8))))
        val a2 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(16))))
        val a3 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(y.qs.asPtr().add(24))))
        val x0 = vld1qS8X4(x.qs.asPtr())
        val x1 = vld1qS8X4(x.qs.asPtr().add(64))
        var ret0 = vdupqNS32(0)
        var ret1 = vdupqNS32(0)
        ret0 = sdotAcc(ret0, x0.first, a0)
        ret1 = sdotAcc(ret1, x0.second, a0)
        ret0 = sdotAcc(ret0, x0.third, a1)
        ret1 = sdotAcc(ret1, x0.field3, a1)
        ret0 = sdotAcc(ret0, x1.first, a2)
        ret1 = sdotAcc(ret1, x1.second, a2)
        ret0 = sdotAcc(ret0, x1.third, a3)
        ret1 = sdotAcc(ret1, x1.field3, a3)
        val ret = vpaddqS32(ret0, ret1)
        acc = vmlaqF32(acc, vcvtqF32S32(ret), vmulqF32(ad, bd))
    }
    vst1qF32(out.asMutPtr(), acc)
    return out
}

internal fun vecDot4Q80x4Q80(n: Int, xs: List<BlockQ80x4>, ys: List<BlockQ80>): FloatArray =
    if (get().i8mm) {
        vecDot4Q80x4Q804x8(n, xs, ys)
    } else {
        vecDot4Q80x4Q804x4(n, xs, ys)
    }

public fun matmulQ80X4Gemv(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ80x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    threadLocal(LHS_Q8_0_GEMV_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>())
    val requiredLen = (kInBlocks * elemSize).divCeil(8)
    return LHS_Q8_0_GEMV_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsB: MutableList<BlockQ80> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), kInBlocks)
        BlockQ80.fromFloat(lhs, lhsB)
        val pool = barrierPool()
        val lhsPtr = lhsB.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        val x4BlockBytes = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
        // equal-sized gemv units: a static split skips the shared cursor, which costs
        // several us per small decode matmul
        pool.executeStatic(nGroups, { range ->
            val lhsRow = fromRawParts(ConstPtr(lhsPtr.toLong()), kInBlocks)
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (g in range) {
                val xs = fromRawParts(ConstPtr((repackedPtr + g * kInBlocks * x4BlockBytes).toLong()), kInBlocks)
                val results = vecDot4Q80x4Q80(k, xs, lhsRow)
                run {
                    copyNonoverlapping(results.asPtr(), MutPtr((dstPtr1).rawAddress + ((g * 4).toLong() * (PrimitiveLayouts.F32).strideBytes)), 4)
                }
            }
        })
        Result.Ok(Unit)
    })
}

internal fun matmulQ80X4(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ80x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    if (m == 1) {
        return matmulQ80X4Gemv(Triple(m, k, n), lhs, repacked, dst)
    }
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    val rowGroups = m / 4
    threadLocal(LHS_X4_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_X4_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ80x4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            quantizeQ80x4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 4)
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 4
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                storeQ80x44x4(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
            }
        })
        Result.Ok(Unit)
    })
}

public fun storeQ80x44x4I8mm(dst: MutPtr, n: Int, row: Int, col: Int, lhs: List<BlockQ80x4>, rhs: List<BlockQ80x4>, kInBlocks: Int) {
    var sum0 = vdupqNF32(0.0)
    var sum1 = vdupqNF32(0.0)
    var sum2 = vdupqNF32(0.0)
    var sum3 = vdupqNF32(0.0)
    for (i in 0 until kInBlocks) {
        val x = rhs[i]
        val y = lhs[i]
        val bd = listOf(x.d[0].toF32(), x.d[1].toF32(), x.d[2].toF32(), x.d[3].toF32())
        val bd1 = vld1qF32(bd.asPtr())
        val ad = listOf(y.d[0].toF32(), y.d[1].toF32(), y.d[2].toF32(), y.d[3].toF32())
        var acc00 = vdupqNS32(0)
        var acc01 = vdupqNS32(0)
        var acc10 = vdupqNS32(0)
        var acc11 = vdupqNS32(0)
        for (chunk in 0 until 4) {
            val offset = chunk * 32
            val a01 = vld1qS8(y.qs.asPtr().add(offset))
            val a23 = vld1qS8(y.qs.asPtr().add(offset + 16))
            val b01 = vld1qS8(x.qs.asPtr().add(offset))
            val b23 = vld1qS8(x.qs.asPtr().add(offset + 16))
            acc00 = smmlaS32(acc00, a01, b01)
            acc01 = smmlaS32(acc01, a01, b23)
            acc10 = smmlaS32(acc10, a23, b01)
            acc11 = smmlaS32(acc11, a23, b23)
        }
        val row0 = vcombineS32(vgetLowS32(acc00), vgetLowS32(acc01))
        val row1 = vcombineS32(vgetHighS32(acc00), vgetHighS32(acc01))
        val row2 = vcombineS32(vgetLowS32(acc10), vgetLowS32(acc11))
        val row3 = vcombineS32(vgetHighS32(acc10), vgetHighS32(acc11))
        sum0 = vmlaqF32(sum0, vcvtqF32S32(row0), vmulqNF32(bd1, ad[0]))
        sum1 = vmlaqF32(sum1, vcvtqF32S32(row1), vmulqNF32(bd1, ad[1]))
        sum2 = vmlaqF32(sum2, vcvtqF32S32(row2), vmulqNF32(bd1, ad[2]))
        sum3 = vmlaqF32(sum3, vcvtqF32S32(row3), vmulqNF32(bd1, ad[3]))
    }
    vst1qF32(MutPtr((dst).rawAddress + ((row * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum0)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 1) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum1)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 2) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum2)
    vst1qF32(MutPtr((dst).rawAddress + (((row + 3) * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes)), sum3)
}

internal fun matmulQ80X4I8mm(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ80x4>, dst: MutableList<Float>): Result<Unit> {
    check(k.isMultipleOf(QK8_0))
    check(n.isMultipleOf(4))
    check(m.isMultipleOf(4))
    val kInBlocks = k / QK8_0
    val nGroups = n / 4
    val rowGroups = m / 4
    threadLocal(LHS_X4_I8MM_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ80x4>(typeLayoutT = ObjectSlotLayout<BlockQ80x4>())
    val requiredLen = (rowGroups * kInBlocks * elemSize).divCeil(8)
    return LHS_X4_I8MM_SCRATCH.with({ cell ->
        var scratch = cell.borrowMut()
        if (scratch.len() < requiredLen) {
            scratch.resize(requiredLen, 0)
        }
        val lhsX4: MutableList<BlockQ80x4> = fromRawPartsMut(mutPtrFromRustCast(scratch.asMutPtr()), rowGroups * kInBlocks)
        for (group in 0 until rowGroups) {
            quantizeQ80x4Interleaved(lhs, k, group * 4, ((lhsX4).slice(group * kInBlocks until (group + 1) * kInBlocks)), 8)
        }
        val tilesTotal = rowGroups * nGroups
        val pool = barrierPool()
        val lhsPtr = lhsX4.asPtr().toInt()
        val repackedPtr = repacked.asPtr().toInt()
        val dstPtr = dst.asMutPtr().toInt()
        pool.executeChunked(tilesTotal, { range ->
            val lhsPtr1 = ConstPtr(lhsPtr.toLong())
            val repackedPtr1 = ConstPtr(repackedPtr.toLong())
            val dstPtr1 = MutPtr(dstPtr.toLong())
            for (tile in range) {
                val rowGroup = tile / nGroups
                val colGroup = tile - rowGroup * nGroups
                val row = rowGroup * 4
                val col = colGroup * 4
                val lhsTile = fromRawParts(ConstPtr((lhsPtr1).rawAddress + ((rowGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                val rhsTile = fromRawParts(ConstPtr((repackedPtr1).rawAddress + ((colGroup * kInBlocks).toLong() * (ObjectSlotLayout<BlockQ80x4>()).strideBytes)), kInBlocks)
                storeQ80x44x4I8mm(dstPtr1, n, row, col, lhsTile, rhsTile, kInBlocks)
            }
        })
        Result.Ok(Unit)
    })
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q8k_q8k: $n is not divisible by $qkK" }
    var sumf = 0.0f
    for ((xs1, ys1) in (xs.zip(ys))) {
        run {
            var sumI = vdupqNS32(0)
            val scale = xs1.d * ys1.d
            val xs2 = xs1.qs.asPtr()
            val ys2 = ys1.qs.asPtr()
            for (i in (0 until QK_K).stepBy(16)) {
                val xs3 = vld1qS8(xs2.add(i))
                val ys3 = vld1qS8(ys2.add(i))
                val xy = vdotqS32(xs3, ys3)
                sumI = vaddqS32(sumI, xy)
            }
            sumf += vaddvqS32(sumI).toFloat() * scale
        }
    }
    return sumf
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q6k_q8k: $n is not divisible by $qkK" }
    var sum = 0.0f
    run {
        val m4b = vdupqNU8(0xF)
        val mone = vdupqNU8(3)
        for ((x, y) in (xs.zip(ys))) {
            val dAll = x.d.toF32()
            var q6 = x.ql.asPtr()
            var qh = x.qh.asPtr()
            var q8 = y.qs.asPtr()
            var scale = x.scales.asPtr()
            val q8sums = vld1qS16X2(y.bsums.asPtr())
            val scales = vld1qS8(scale)
            val q6scales = int16x8x2T(vmovlS8(vgetLowS8(scales)), vmovlS8(vgetHighS8(scales)))
            val prod = vaddqS32(vaddqS32(vmullS16(vgetLowS16(q8sums.first), vgetLowS16(q6scales.first)), vmullS16(vgetHighS16(q8sums.first), vgetHighS16(q6scales.first))), vaddqS32(vmullS16(vgetLowS16(q8sums.second), vgetLowS16(q6scales.second)), vmullS16(vgetHighS16(q8sums.second), vgetHighS16(q6scales.second))))
            val isumMins = vaddvqS32(prod)
            var isum = 0
            for (j in 0 until QK_K / 128) {
                val qhbits = vld1qU8X2(qh)
                qh = qh.add(32)
                val q6bits = vld1qU8X4(q6)
                q6 = q6.add(64)
                val q8bytes = vld1qS8X4(q8)
                q8 = q8.add(64)
                val q6h0 = vshlqNU8(vandqU8(mone, qhbits.first), 4)
                val q6h1 = vshlqNU8(vandqU8(mone, qhbits.second), 4)
                val shifted = vshrqNU8(qhbits.first, 2)
                val q6h2 = vshlqNU8(vandqU8(mone, shifted), 4)
                val shifted1 = vshrqNU8(qhbits.second, 2)
                val q6h3 = vshlqNU8(vandqU8(mone, shifted1), 4)
                val q6bytes0 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits.first, m4b), q6h0))
                val q6bytes1 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits.second, m4b), q6h1))
                val q6bytes2 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits.third, m4b), q6h2))
                val q6bytes3 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits.field3, m4b), q6h3))
                val p0 = vdotqS32(q6bytes0, q8bytes.first)
                val p1 = vdotqS32(q6bytes1, q8bytes.second)
                val (scale0, scale1) = Pair(scale.toInt(), scale.add(1).toInt())
                isum += vaddvqS32(p0) * scale0 + vaddvqS32(p1) * scale1
                scale = scale.add(2)
                val p2 = vdotqS32(q6bytes2, q8bytes.third)
                val p3 = vdotqS32(q6bytes3, q8bytes.field3)
                val (scale01, scale11) = Pair(scale.toInt(), scale.add(1).toInt())
                isum += vaddvqS32(p2) * scale01 + vaddvqS32(p3) * scale11
                scale = scale.add(2)
                val q8bytes1 = vld1qS8X4(q8)
                q8 = q8.add(64)
                val shifted2 = vshrqNU8(qhbits.first, 4)
                val q6h01 = vshlqNU8(vandqU8(mone, shifted2), 4)
                val shifted3 = vshrqNU8(qhbits.second, 4)
                val q6h11 = vshlqNU8(vandqU8(mone, shifted3), 4)
                val shifted4 = vshrqNU8(qhbits.first, 6)
                val q6h21 = vshlqNU8(vandqU8(mone, shifted4), 4)
                val shifted5 = vshrqNU8(qhbits.second, 6)
                val q6h31 = vshlqNU8(vandqU8(mone, shifted5), 4)
                val q6bytes01 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits.first, 4), q6h01))
                val q6bytes11 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits.second, 4), q6h11))
                val q6bytes21 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits.third, 4), q6h21))
                val q6bytes31 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits.field3, 4), q6h31))
                val p01 = vdotqS32(q6bytes01, q8bytes1.first)
                val p11 = vdotqS32(q6bytes11, q8bytes1.second)
                val (scale02, scale12) = Pair(scale.toInt(), scale.add(1).toInt())
                isum += vaddvqS32(p01) * scale02 + vaddvqS32(p11) * scale12
                scale = scale.add(2)
                val p21 = vdotqS32(q6bytes21, q8bytes1.third)
                val p31 = vdotqS32(q6bytes31, q8bytes1.field3)
                val (scale03, scale13) = Pair(scale.toInt(), scale.add(1).toInt())
                isum += vaddvqS32(p21) * scale03 + vaddvqS32(p31) * scale13
                scale = scale.add(2)
            }
            sum += dAll * y.d * ((isum - 32 * isumMins).toFloat())
        }
    }
    return sum
}

/**
 * Four Q6K dot products sharing one Q8K load.
 */
internal fun vecDot4Q6kQ8k(n: Int, xs0: List<BlockQ6K>, xs1: List<BlockQ6K>, xs2: List<BlockQ6K>, xs3: List<BlockQ6K>, ys: List<BlockQ8K>): List<Any?> {
    check(n.isMultipleOf(QK_K)) { "vec_dot_4_q6k_q8k: $n is not divisible by $qkK" }
    var sum0 = 0.0f
    var sum1 = 0.0f
    var sum2 = 0.0f
    var sum3 = 0.0f
    run {
        val m4b = vdupqNU8(0xF)
        val mone = vdupqNU8(3)
        for ((x0, x1, x2, x3, y) in ((((xs0.zip(xs1)).zip(xs2)).zip(xs3)).zip(ys))) {
            // Compute isumMins for each column: dot(q8sums, columnScales).
            val yd = y.d
            // Q8K bsums - loaded once for all four columns.
            val q8sums = vld1qS16X2(y.bsums.asPtr())
            val isumMins0 =
                run {
                    val scalesV = vld1qS8(x0.scales.asPtr())
                    val q6sc = int16x8x2T(vmovlS8(vgetLowS8(scalesV)), vmovlS8(vgetHighS8(scalesV)))
                    val prod = vaddqS32(vaddqS32(vmullS16(vgetLowS16(q8sums.first), vgetLowS16(q6sc.first)), vmullS16(vgetHighS16(q8sums.first), vgetHighS16(q6sc.first))), vaddqS32(vmullS16(vgetLowS16(q8sums.second), vgetLowS16(q6sc.second)), vmullS16(vgetHighS16(q8sums.second), vgetHighS16(q6sc.second))))
                    vaddvqS32(prod)
                }
            val isumMins1 =
                run {
                    val scalesV1 = vld1qS8(x1.scales.asPtr())
                    val q6sc1 = int16x8x2T(vmovlS8(vgetLowS8(scalesV1)), vmovlS8(vgetHighS8(scalesV1)))
                    val prod1 = vaddqS32(vaddqS32(vmullS16(vgetLowS16(q8sums.first), vgetLowS16(q6sc1.first)), vmullS16(vgetHighS16(q8sums.first), vgetHighS16(q6sc1.first))), vaddqS32(vmullS16(vgetLowS16(q8sums.second), vgetLowS16(q6sc1.second)), vmullS16(vgetHighS16(q8sums.second), vgetHighS16(q6sc1.second))))
                    vaddvqS32(prod1)
                }
            val isumMins2 =
                run {
                    val scalesV2 = vld1qS8(x2.scales.asPtr())
                    val q6sc2 = int16x8x2T(vmovlS8(vgetLowS8(scalesV2)), vmovlS8(vgetHighS8(scalesV2)))
                    val prod2 = vaddqS32(vaddqS32(vmullS16(vgetLowS16(q8sums.first), vgetLowS16(q6sc2.first)), vmullS16(vgetHighS16(q8sums.first), vgetHighS16(q6sc2.first))), vaddqS32(vmullS16(vgetLowS16(q8sums.second), vgetLowS16(q6sc2.second)), vmullS16(vgetHighS16(q8sums.second), vgetHighS16(q6sc2.second))))
                    vaddvqS32(prod2)
                }
            val isumMins3 =
                run {
                    val scalesV3 = vld1qS8(x3.scales.asPtr())
                    val q6sc3 = int16x8x2T(vmovlS8(vgetLowS8(scalesV3)), vmovlS8(vgetHighS8(scalesV3)))
                    val prod3 = vaddqS32(vaddqS32(vmullS16(vgetLowS16(q8sums.first), vgetLowS16(q6sc3.first)), vmullS16(vgetHighS16(q8sums.first), vgetHighS16(q6sc3.first))), vaddqS32(vmullS16(vgetLowS16(q8sums.second), vgetLowS16(q6sc3.second)), vmullS16(vgetHighS16(q8sums.second), vgetHighS16(q6sc3.second))))
                    vaddvqS32(prod3)
                }
            var q60 = x0.ql.asPtr()
            var qh0 = x0.qh.asPtr()
            var sc0 = x0.scales.asPtr()
            var q61 = x1.ql.asPtr()
            var qh1 = x1.qh.asPtr()
            var sc1 = x1.scales.asPtr()
            var q62 = x2.ql.asPtr()
            var qh2 = x2.qh.asPtr()
            var sc2 = x2.scales.asPtr()
            var q63 = x3.ql.asPtr()
            var qh3 = x3.qh.asPtr()
            var sc3 = x3.scales.asPtr()
            var q8 = y.qs.asPtr()
            var isum0 = 0
            var isum1 = 0
            var isum2 = 0
            var isum3 = 0
            for (j in 0 until QK_K / 128) {
                // Decode one column's Q6 bits and accumulate into its isum.
                // q8lo/q8hi are shared from the outer scope.
                // Load Q8K bytes once - shared across all four columns.
                val q8lo = vld1qS8X4(q8)
                q8 = q8.add(64)
                val q8hi = vld1qS8X4(q8)
                q8 = q8.add(64)
                val qhb = vld1qU8X2(qh0)
                qh0 = qh0.add(32)
                val q6b = vld1qU8X4(q60)
                q60 = q60.add(64)
                val qh00 = vshlqNU8(vandqU8(mone, qhb.first), 4)
                val qh01 = vshlqNU8(vandqU8(mone, qhb.second), 4)
                val qh10 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.first, 2)), 4)
                val qh11 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.second, 2)), 4)
                val q6b0 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b.first, m4b), qh00))
                val q6b1 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b.second, m4b), qh01))
                val q6b2 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b.third, m4b), qh10))
                val q6b3 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b.field3, m4b), qh11))
                val p0 = vdotqS32(q6b0, q8lo.first)
                val p1 = vdotqS32(q6b1, q8lo.second)
                isum0 += vaddvqS32(p0) * (sc0.toInt()) + vaddvqS32(p1) * (sc0.add(1).toInt())
                sc0 = sc0.add(2)
                val p2 = vdotqS32(q6b2, q8lo.third)
                val p3 = vdotqS32(q6b3, q8lo.field3)
                isum0 += vaddvqS32(p2) * (sc0.toInt()) + vaddvqS32(p3) * (sc0.add(1).toInt())
                sc0 = sc0.add(2)
                val qh20 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.first, 4)), 4)
                val qh21 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.second, 4)), 4)
                val qh30 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.first, 6)), 4)
                val qh31 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb.second, 6)), 4)
                val q6b01 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b.first, 4), qh20))
                val q6b11 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b.second, 4), qh21))
                val q6b21 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b.third, 4), qh30))
                val q6b31 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b.field3, 4), qh31))
                val p01 = vdotqS32(q6b01, q8hi.first)
                val p11 = vdotqS32(q6b11, q8hi.second)
                isum0 += vaddvqS32(p01) * (sc0.toInt()) + vaddvqS32(p11) * (sc0.add(1).toInt())
                sc0 = sc0.add(2)
                val p21 = vdotqS32(q6b21, q8hi.third)
                val p31 = vdotqS32(q6b31, q8hi.field3)
                isum0 += vaddvqS32(p21) * (sc0.toInt()) + vaddvqS32(p31) * (sc0.add(1).toInt())
                sc0 = sc0.add(2)
                val qhb1 = vld1qU8X2(qh1)
                qh1 = qh1.add(32)
                val q6b1 = vld1qU8X4(q61)
                q61 = q61.add(64)
                val qh001 = vshlqNU8(vandqU8(mone, qhb1.first), 4)
                val qh011 = vshlqNU8(vandqU8(mone, qhb1.second), 4)
                val qh101 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.first, 2)), 4)
                val qh111 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.second, 2)), 4)
                val q6b02 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b1.first, m4b), qh001))
                val q6b12 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b1.second, m4b), qh011))
                val q6b22 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b1.third, m4b), qh101))
                val q6b32 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b1.field3, m4b), qh111))
                val p02 = vdotqS32(q6b02, q8lo.first)
                val p12 = vdotqS32(q6b12, q8lo.second)
                isum1 += vaddvqS32(p02) * (sc1.toInt()) + vaddvqS32(p12) * (sc1.add(1).toInt())
                sc1 = sc1.add(2)
                val p22 = vdotqS32(q6b22, q8lo.third)
                val p32 = vdotqS32(q6b32, q8lo.field3)
                isum1 += vaddvqS32(p22) * (sc1.toInt()) + vaddvqS32(p32) * (sc1.add(1).toInt())
                sc1 = sc1.add(2)
                val qh201 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.first, 4)), 4)
                val qh211 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.second, 4)), 4)
                val qh301 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.first, 6)), 4)
                val qh311 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb1.second, 6)), 4)
                val q6b03 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b1.first, 4), qh201))
                val q6b13 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b1.second, 4), qh211))
                val q6b23 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b1.third, 4), qh301))
                val q6b33 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b1.field3, 4), qh311))
                val p03 = vdotqS32(q6b03, q8hi.first)
                val p13 = vdotqS32(q6b13, q8hi.second)
                isum1 += vaddvqS32(p03) * (sc1.toInt()) + vaddvqS32(p13) * (sc1.add(1).toInt())
                sc1 = sc1.add(2)
                val p23 = vdotqS32(q6b23, q8hi.third)
                val p33 = vdotqS32(q6b33, q8hi.field3)
                isum1 += vaddvqS32(p23) * (sc1.toInt()) + vaddvqS32(p33) * (sc1.add(1).toInt())
                sc1 = sc1.add(2)
                val qhb2 = vld1qU8X2(qh2)
                qh2 = qh2.add(32)
                val q6b2 = vld1qU8X4(q62)
                q62 = q62.add(64)
                val qh002 = vshlqNU8(vandqU8(mone, qhb2.first), 4)
                val qh012 = vshlqNU8(vandqU8(mone, qhb2.second), 4)
                val qh102 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.first, 2)), 4)
                val qh112 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.second, 2)), 4)
                val q6b04 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b2.first, m4b), qh002))
                val q6b14 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b2.second, m4b), qh012))
                val q6b24 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b2.third, m4b), qh102))
                val q6b34 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b2.field3, m4b), qh112))
                val p04 = vdotqS32(q6b04, q8lo.first)
                val p14 = vdotqS32(q6b14, q8lo.second)
                isum2 += vaddvqS32(p04) * (sc2.toInt()) + vaddvqS32(p14) * (sc2.add(1).toInt())
                sc2 = sc2.add(2)
                val p24 = vdotqS32(q6b24, q8lo.third)
                val p34 = vdotqS32(q6b34, q8lo.field3)
                isum2 += vaddvqS32(p24) * (sc2.toInt()) + vaddvqS32(p34) * (sc2.add(1).toInt())
                sc2 = sc2.add(2)
                val qh202 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.first, 4)), 4)
                val qh212 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.second, 4)), 4)
                val qh302 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.first, 6)), 4)
                val qh312 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb2.second, 6)), 4)
                val q6b05 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b2.first, 4), qh202))
                val q6b15 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b2.second, 4), qh212))
                val q6b25 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b2.third, 4), qh302))
                val q6b35 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b2.field3, 4), qh312))
                val p05 = vdotqS32(q6b05, q8hi.first)
                val p15 = vdotqS32(q6b15, q8hi.second)
                isum2 += vaddvqS32(p05) * (sc2.toInt()) + vaddvqS32(p15) * (sc2.add(1).toInt())
                sc2 = sc2.add(2)
                val p25 = vdotqS32(q6b25, q8hi.third)
                val p35 = vdotqS32(q6b35, q8hi.field3)
                isum2 += vaddvqS32(p25) * (sc2.toInt()) + vaddvqS32(p35) * (sc2.add(1).toInt())
                sc2 = sc2.add(2)
                val qhb3 = vld1qU8X2(qh3)
                qh3 = qh3.add(32)
                val q6b3 = vld1qU8X4(q63)
                q63 = q63.add(64)
                val qh003 = vshlqNU8(vandqU8(mone, qhb3.first), 4)
                val qh013 = vshlqNU8(vandqU8(mone, qhb3.second), 4)
                val qh103 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.first, 2)), 4)
                val qh113 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.second, 2)), 4)
                val q6b06 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b3.first, m4b), qh003))
                val q6b16 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b3.second, m4b), qh013))
                val q6b26 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b3.third, m4b), qh103))
                val q6b36 = vreinterpretqS8U8(vorrqU8(vandqU8(q6b3.field3, m4b), qh113))
                val p06 = vdotqS32(q6b06, q8lo.first)
                val p16 = vdotqS32(q6b16, q8lo.second)
                isum3 += vaddvqS32(p06) * (sc3.toInt()) + vaddvqS32(p16) * (sc3.add(1).toInt())
                sc3 = sc3.add(2)
                val p26 = vdotqS32(q6b26, q8lo.third)
                val p36 = vdotqS32(q6b36, q8lo.field3)
                isum3 += vaddvqS32(p26) * (sc3.toInt()) + vaddvqS32(p36) * (sc3.add(1).toInt())
                sc3 = sc3.add(2)
                val qh203 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.first, 4)), 4)
                val qh213 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.second, 4)), 4)
                val qh303 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.first, 6)), 4)
                val qh313 = vshlqNU8(vandqU8(mone, vshrqNU8(qhb3.second, 6)), 4)
                val q6b07 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b3.first, 4), qh203))
                val q6b17 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b3.second, 4), qh213))
                val q6b27 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b3.third, 4), qh303))
                val q6b37 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6b3.field3, 4), qh313))
                val p07 = vdotqS32(q6b07, q8hi.first)
                val p17 = vdotqS32(q6b17, q8hi.second)
                isum3 += vaddvqS32(p07) * (sc3.toInt()) + vaddvqS32(p17) * (sc3.add(1).toInt())
                sc3 = sc3.add(2)
                val p27 = vdotqS32(q6b27, q8hi.third)
                val p37 = vdotqS32(q6b37, q8hi.field3)
                isum3 += vaddvqS32(p27) * (sc3.toInt()) + vaddvqS32(p37) * (sc3.add(1).toInt())
                sc3 = sc3.add(2)
            }
            sum0 += x0.d.toF32() * yd * ((isum0 - 32 * isumMins0).toFloat())
            sum1 += x1.d.toF32() * yd * ((isum1 - 32 * isumMins1).toFloat())
            sum2 += x2.d.toF32() * yd * ((isum2 - 32 * isumMins2).toFloat())
            sum3 += x3.d.toF32() * yd * ((isum3 - 32 * isumMins3).toFloat())
        }
    }
    return listOf(sum0, sum1, sum2, sum3)
}

internal fun vecDotQ5kQ8k(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    check(n.isMultipleOf(QK_K)) { "vec_dot_q5k_q8k: $n is not divisible by $qkK" }
    var sumf = 0.0f
    val localStorageL3483C4 = AddressableArrayLocal(UIntArray(4) { 0u }, FixedArrayTypeLayout(PrimitiveLayouts.U32, (4).toInt()))
    var utmp by localStorageL3483C4
    run {
        val m4b = vdupqNU8(0xF)
        val mone = vdupqNU8(1)
        val mtwo = vdupqNU8(2)
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = y.d * x.dmin.toF32()
            val q8sums = vpaddqS16(vld1qS16(y.bsums.asPtr()), vld1qS16(y.bsums.asPtr().add(8)))
            LittleEndian.readU32Into(x.scales, ((utmp).slice(0 until 3)))
            utmp[3] = ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)
            val uaux = utmp[1] and KMASK1
            utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
            utmp[2] = uaux
            utmp[0] = utmp[0] and KMASK1
            val mins8 = vld1U8(ConstPtr(((ConstPtr(localStorageL3483C4.elementConstPtr(0).rawAddress, localStorageL3483C4.elementConstPtr(0).rawMeta))).rawAddress + ((8).toLong() * (PrimitiveLayouts.U8).strideBytes)))
            val mins = vreinterpretqS16U16(vmovlU8(mins8))
            val prod = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins)))
            val sumiMins = vaddvqS32(prod)
            var scales = ConstPtr(localStorageL3483C4.elementConstPtr(0).rawAddress, localStorageL3483C4.elementConstPtr(0).rawMeta)
            var q5 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            val qhbits = vld1qU8X2(x.qh.asPtr())
            var sumi = 0
            for (j in 0 until QK_K / 64) {
                val q5bits = vld1qU8X2(q5)
                q5 = q5.add(32)
                val q8bytes = vld1qS8X4(q8)
                q8 = q8.add(64)
                val q5h0 = vshlqNU8(vandqU8(mone, qhbits.first), 4)
                val q5h1 = vshlqNU8(vandqU8(mone, qhbits.second), 4)
                val q5h2 = vshlqNU8(vandqU8(mtwo, qhbits.first), 3)
                val q5h3 = vshlqNU8(vandqU8(mtwo, qhbits.second), 3)
                qhbits.first = vshrqNU8(qhbits.first, 2)
                qhbits.second = vshrqNU8(qhbits.second, 2)
                val q5bytes0 = vreinterpretqS8U8(vorrqU8(vandqU8(q5bits.first, m4b), q5h0))
                val q5bytes1 = vreinterpretqS8U8(vorrqU8(vandqU8(q5bits.second, m4b), q5h1))
                val q5bytes2 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5bits.first, 4), q5h2))
                val q5bytes3 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5bits.second, 4), q5h3))
                val p0 = vdotqS32(q5bytes0, q8bytes.first)
                val p1 = vdotqS32(q5bytes1, q8bytes.second)
                sumi += vaddvqS32(vaddqS32(p0, p1)) * ((PrimitiveLayouts.U8).readNative((scales).rawAddress)).toInt()
                scales = ConstPtr((scales).rawAddress + ((1).toLong() * (PrimitiveLayouts.U8).strideBytes))
                val p2 = vdotqS32(q5bytes2, q8bytes.third)
                val p3 = vdotqS32(q5bytes3, q8bytes.field3)
                sumi += vaddvqS32(vaddqS32(p2, p3)) * ((PrimitiveLayouts.U8).readNative((scales).rawAddress)).toInt()
                scales = ConstPtr((scales).rawAddress + ((1).toLong() * (PrimitiveLayouts.U8).strideBytes))
            }
            sumf += d * sumi.toFloat() - dmin * sumiMins.toFloat()
        }
    }
    return sumf
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    check(n.isMultipleOf(QK_K)) { "vec_dot_q4k_q8k: $n is not divisible by $qkK" }
    var sumf = 0.0f
    val utmp = List(4) { 0u }
    val scales = List(16) { 0u.toUByte() }
    run {
        val m4b = vdupqNU8(0xF)
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = y.d * x.dmin.toF32()
            val q8sums = vpaddqS16(vld1qS16(y.bsums.asPtr()), vld1qS16(y.bsums.asPtr().add(8)))
            LittleEndian.readU32Into(x.scales, ((utmp).slice(0 until 3)))
            val mins8 = vld1U32(listOf(utmp[1] and KMASK1, ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)).asPtr())
            utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
            utmp[0] = utmp[0] and KMASK1
            val mins = vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(mins8)))
            val prod = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins)))
            sumf -= dmin * vaddvqS32(prod).toFloat()
            LittleEndian.writeU32Into(utmp, scales)
            var q4 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            var sumi1 = 0
            var sumi2 = 0
            for (j in 0 until QK_K / 64) {
                val q4bits = vld1qU8X2(q4)
                q4 = q4.add(32)
                val q8bytes = vld1qS8X2(q8)
                q8 = q8.add(32)
                val q4bytes = int8x16x2T(vreinterpretqS8U8(vandqU8(q4bits.first, m4b)), vreinterpretqS8U8(vandqU8(q4bits.second, m4b)))
                val p0 = vdotqS32(q4bytes.first, q8bytes.first)
                val p1 = vdotqS32(q4bytes.second, q8bytes.second)
                sumi1 += vaddvqS32(vaddqS32(p0, p1)) * scales[2 * j].toInt()
                val q8bytes1 = vld1qS8X2(q8)
                q8 = q8.add(32)
                val q4bytes1 = int8x16x2T(vreinterpretqS8U8(vshrqNU8(q4bits.first, 4)), vreinterpretqS8U8(vshrqNU8(q4bits.second, 4)))
                val p2 = vdotqS32(q4bytes1.first, q8bytes1.first)
                val p3 = vdotqS32(q4bytes1.second, q8bytes1.second)
                sumi2 += vaddvqS32(vaddqS32(p2, p3)) * scales[2 * j + 1].toInt()
            }
            sumf += d * (sumi1 + sumi2).toFloat()
        }
    }
    return sumf
}

/**
 * Four Q4K dot products sharing one Q8K load.
 */
internal fun vecDot4Q4kQ8k(n: Int, xs0: List<BlockQ4K>, xs1: List<BlockQ4K>, xs2: List<BlockQ4K>, xs3: List<BlockQ4K>, ys: List<BlockQ8K>): List<Any?> {
    val KMASK1: UInt = 0x3f3f3f3fu
    val KMASK2: UInt = 0x0f0f0f0fu
    val KMASK3: UInt = 0x03030303u
    // Decode Q4K scale+min bytes into an 8-entry scales array and a float32x4x2
    // mins vector. Returns the mins vector so the caller can compute min correction.
    // Accumulate one column's nibble-group contribution into a scalar sumi.
    check(n.isMultipleOf(QK_K)) { "vec_dot_4_q4k_q8k: $n is not divisible by $qkK" }
    var sum0 = 0.0f
    var sum1 = 0.0f
    var sum2 = 0.0f
    var sum3 = 0.0f
    val utmp = List(4) { 0u }
    val sc0 = List(16) { 0u.toUByte() }
    val sc1 = List(16) { 0u.toUByte() }
    val sc2 = List(16) { 0u.toUByte() }
    val sc3 = List(16) { 0u.toUByte() }
    run {
        val m4b = vdupqNU8(0xF)
        for ((x0, x1, x2, x3, y) in ((((xs0.zip(xs1)).zip(xs2)).zip(xs3)).zip(ys))) {
            // min correction: sum -= dmin * dot(q8sums, mins)
            val yd = y.d
            // Q8K bsums - loaded once for all four columns.
            val q8sums = vpaddqS16(vld1qS16(y.bsums.asPtr()), vld1qS16(y.bsums.asPtr().add(8)))
            // Decode scales and apply min correction for each column.
            val mins0 =
                run {
                    LittleEndian.readU32Into(x0.scales, ((utmp).slice(0 until 3)))
                    val mins8 = vld1U32(listOf(utmp[1] and KMASK1, ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)).asPtr())
                    utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
                    utmp[0] = utmp[0] and KMASK1
                    LittleEndian.writeU32Into(utmp, sc0)
                    vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(mins8)))
                }
            val mins1 =
                run {
                    LittleEndian.readU32Into(x1.scales, ((utmp).slice(0 until 3)))
                    val mins81 = vld1U32(listOf(utmp[1] and KMASK1, ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)).asPtr())
                    utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
                    utmp[0] = utmp[0] and KMASK1
                    LittleEndian.writeU32Into(utmp, sc1)
                    vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(mins81)))
                }
            val mins2 =
                run {
                    LittleEndian.readU32Into(x2.scales, ((utmp).slice(0 until 3)))
                    val mins82 = vld1U32(listOf(utmp[1] and KMASK1, ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)).asPtr())
                    utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
                    utmp[0] = utmp[0] and KMASK1
                    LittleEndian.writeU32Into(utmp, sc2)
                    vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(mins82)))
                }
            val mins3 =
                run {
                    LittleEndian.readU32Into(x3.scales, ((utmp).slice(0 until 3)))
                    val mins83 = vld1U32(listOf(utmp[1] and KMASK1, ((utmp[2] shr 4) and KMASK2) or (((utmp[1] shr 6) and KMASK3) shl 4)).asPtr())
                    utmp[1] = (utmp[2] and KMASK2) or (((utmp[0] shr 6) and KMASK3) shl 4)
                    utmp[0] = utmp[0] and KMASK1
                    LittleEndian.writeU32Into(utmp, sc3)
                    vreinterpretqS16U16(vmovlU8(vreinterpretU8U32(mins83)))
                }
            val d0 = yd * x0.d.toF32()
            val d1 = yd * x1.d.toF32()
            val d2 = yd * x2.d.toF32()
            val d3 = yd * x3.d.toF32()
            val prod = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins0)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins0)))
            sum0 -= yd * x0.dmin.toF32() * vaddvqS32(prod).toFloat()
            val prod1 = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins1)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins1)))
            sum1 -= yd * x1.dmin.toF32() * vaddvqS32(prod1).toFloat()
            val prod2 = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins2)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins2)))
            sum2 -= yd * x2.dmin.toF32() * vaddvqS32(prod2).toFloat()
            val prod3 = vaddqS32(vmullS16(vgetLowS16(q8sums), vgetLowS16(mins3)), vmullS16(vgetHighS16(q8sums), vgetHighS16(mins3)))
            sum3 -= yd * x3.dmin.toF32() * vaddvqS32(prod3).toFloat()
            var q40 = x0.qs.asPtr()
            var q41 = x1.qs.asPtr()
            var q42 = x2.qs.asPtr()
            var q43 = x3.qs.asPtr()
            var q8 = y.qs.asPtr()
            var s0a = vdupqNS32(0)
            var s0b = vdupqNS32(0)
            var s1a = vdupqNS32(0)
            var s1b = vdupqNS32(0)
            var s2a = vdupqNS32(0)
            var s2b = vdupqNS32(0)
            var s3a = vdupqNS32(0)
            var s3b = vdupqNS32(0)
            for (j in 0 until QK_K / 64) {
                // Load Q8K once and reuse across all four columns.
                val q8lo = vld1qS8X2(q8)
                q8 = q8.add(32)
                val q8hi = vld1qS8X2(q8)
                q8 = q8.add(32)
                val bits = vld1qU8X2(q40)
                q40 = q40.add(32)
                val q4lo = int8x16x2T(vreinterpretqS8U8(vandqU8(bits.first, m4b)), vreinterpretqS8U8(vandqU8(bits.second, m4b)))
                s0a = vmlaqNS32(s0a, vdotqS32Pair(q4lo.first, q8lo.first, q4lo.second, q8lo.second), sc0[2 * j].toInt())
                val q4hi = int8x16x2T(vreinterpretqS8U8(vshrqNU8(bits.first, 4)), vreinterpretqS8U8(vshrqNU8(bits.second, 4)))
                s0b = vmlaqNS32(s0b, vdotqS32Pair(q4hi.first, q8hi.first, q4hi.second, q8hi.second), sc0[2 * j + 1].toInt())
                val bits1 = vld1qU8X2(q41)
                q41 = q41.add(32)
                val q4lo1 = int8x16x2T(vreinterpretqS8U8(vandqU8(bits1.first, m4b)), vreinterpretqS8U8(vandqU8(bits1.second, m4b)))
                s1a = vmlaqNS32(s1a, vdotqS32Pair(q4lo1.first, q8lo.first, q4lo1.second, q8lo.second), sc1[2 * j].toInt())
                val q4hi1 = int8x16x2T(vreinterpretqS8U8(vshrqNU8(bits1.first, 4)), vreinterpretqS8U8(vshrqNU8(bits1.second, 4)))
                s1b = vmlaqNS32(s1b, vdotqS32Pair(q4hi1.first, q8hi.first, q4hi1.second, q8hi.second), sc1[2 * j + 1].toInt())
                val bits2 = vld1qU8X2(q42)
                q42 = q42.add(32)
                val q4lo2 = int8x16x2T(vreinterpretqS8U8(vandqU8(bits2.first, m4b)), vreinterpretqS8U8(vandqU8(bits2.second, m4b)))
                s2a = vmlaqNS32(s2a, vdotqS32Pair(q4lo2.first, q8lo.first, q4lo2.second, q8lo.second), sc2[2 * j].toInt())
                val q4hi2 = int8x16x2T(vreinterpretqS8U8(vshrqNU8(bits2.first, 4)), vreinterpretqS8U8(vshrqNU8(bits2.second, 4)))
                s2b = vmlaqNS32(s2b, vdotqS32Pair(q4hi2.first, q8hi.first, q4hi2.second, q8hi.second), sc2[2 * j + 1].toInt())
                val bits3 = vld1qU8X2(q43)
                q43 = q43.add(32)
                val q4lo3 = int8x16x2T(vreinterpretqS8U8(vandqU8(bits3.first, m4b)), vreinterpretqS8U8(vandqU8(bits3.second, m4b)))
                s3a = vmlaqNS32(s3a, vdotqS32Pair(q4lo3.first, q8lo.first, q4lo3.second, q8lo.second), sc3[2 * j].toInt())
                val q4hi3 = int8x16x2T(vreinterpretqS8U8(vshrqNU8(bits3.first, 4)), vreinterpretqS8U8(vshrqNU8(bits3.second, 4)))
                s3b = vmlaqNS32(s3b, vdotqS32Pair(q4hi3.first, q8hi.first, q4hi3.second, q8hi.second), sc3[2 * j + 1].toInt())
            }
            sum0 += d0 * vaddvqS32(vaddqS32(s0a, s0b)).toFloat()
            sum1 += d1 * vaddvqS32(vaddqS32(s1a, s1b)).toFloat()
            sum2 += d2 * vaddvqS32(vaddqS32(s2a, s2b)).toFloat()
            sum3 += d3 * vaddvqS32(vaddqS32(s3a, s3b)).toFloat()
        }
    }
    return listOf(sum0, sum1, sum2, sum3)
}

internal fun vecDotQ3kQ8k(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
    val KMASK1: UInt = 0x03030303u
    val KMASK2: UInt = 0x0f0f0f0fu
    check(n.isMultipleOf(QK_K)) { "vec_dot_q3k_q8k: $n is not divisible by $qkK" }
    var sumf = 0.0f
    val localStorageL3806C4 = AddressableArrayLocal(UIntArray(4) { 0u }, FixedArrayTypeLayout(PrimitiveLayouts.U32, (4).toInt()))
    var utmp by localStorageL3806C4
    val aux = List(3) { 0u }
    run {
        val m3b = vdupqNU8(0x3)
        val m0 = vdupqNU8(1)
        val m1 = vshlqNU8(m0, 1)
        val m2 = vshlqNU8(m0, 2)
        val m3 = vshlqNU8(m0, 3)
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            var q3 = x.qs.asPtr()
            val qh = x.hmask.asPtr()
            var q8 = y.qs.asPtr()
            val qhbits = vld1qU8X2(qh)
            var isum = 0
            // Set up scales
            LittleEndian.readU32Into(x.scales, aux)
            utmp[3] = ((aux[1] shr 4) and KMASK2) or (((aux[2] shr 6) and KMASK1) shl 4)
            utmp[2] = ((aux[0] shr 4) and KMASK2) or (((aux[2] shr 4) and KMASK1) shl 4)
            utmp[1] = (aux[1] and KMASK2) or (((aux[2] shr 2) and KMASK1) shl 4)
            utmp[0] = (aux[0] and KMASK2) or ((aux[2] and KMASK1) shl 4)
            var scale = MutPtr(localStorageL3806C4.elementMutPtr(0).rawAddress, localStorageL3806C4.elementMutPtr(0).rawMeta)
            for (j in 0 until 16) {
                (MutPtr((scale).rawAddress + ((j).toLong() * (PrimitiveLayouts.I8).strideBytes))).let { targetPtr ->
                    val targetLayout = PrimitiveLayouts.I8
                    var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                    currentValue -= (32)
                    targetLayout.writeNative(targetPtr.rawAddress, currentValue)
                }
            }
            for (j1 in 0 until QK_K / 128) {
                val q3bits = vld1qU8X2(q3)
                q3 = q3.add(32)
                val q8bytes1 = vld1qS8X4(q8)
                q8 = q8.add(64)
                val q8bytes2 = vld1qS8X4(q8)
                q8 = q8.add(64)
                val q3h0 = vshlqNU8(vbicqU8(m0, qhbits.first), 2)
                val q3h1 = vshlqNU8(vbicqU8(m0, qhbits.second), 2)
                val q3h2 = vshlqNU8(vbicqU8(m1, qhbits.first), 1)
                val q3h3 = vshlqNU8(vbicqU8(m1, qhbits.second), 1)
                val q3bytes0 = vsubqS8(vreinterpretqS8U8(vandqU8(q3bits.first, m3b)), vreinterpretqS8U8(q3h0))
                val q3bytes1 = vsubqS8(vreinterpretqS8U8(vandqU8(q3bits.second, m3b)), vreinterpretqS8U8(q3h1))
                val q3bytes2 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.first, 2), m3b)), vreinterpretqS8U8(q3h2))
                val q3bytes3 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.second, 2), m3b)), vreinterpretqS8U8(q3h3))
                val p0 = vdotqS32(q3bytes0, q8bytes1.first)
                val p1 = vdotqS32(q3bytes1, q8bytes1.second)
                val p2 = vdotqS32(q3bytes2, q8bytes1.third)
                val p3 = vdotqS32(q3bytes3, q8bytes1.field3)
                isum += vaddvqS32(p0) * ((PrimitiveLayouts.I8).readNative((scale).rawAddress)).toInt() + vaddvqS32(p1) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((1).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt() + vaddvqS32(p2) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((2).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt() + vaddvqS32(p3) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((3).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt()
                scale = MutPtr((scale).rawAddress + ((4).toLong() * (PrimitiveLayouts.I8).strideBytes))
                val q3h01 = vbicqU8(m2, qhbits.first)
                val q3h11 = vbicqU8(m2, qhbits.second)
                val q3h21 = vshrqNU8(vbicqU8(m3, qhbits.first), 1)
                val q3h31 = vshrqNU8(vbicqU8(m3, qhbits.second), 1)
                val q3bytes01 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.first, 4), m3b)), vreinterpretqS8U8(q3h01))
                val q3bytes11 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.second, 4), m3b)), vreinterpretqS8U8(q3h11))
                val q3bytes21 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.first, 6), m3b)), vreinterpretqS8U8(q3h21))
                val q3bytes31 = vsubqS8(vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits.second, 6), m3b)), vreinterpretqS8U8(q3h31))
                val p01 = vdotqS32(q3bytes01, q8bytes2.first)
                val p11 = vdotqS32(q3bytes11, q8bytes2.second)
                val p21 = vdotqS32(q3bytes21, q8bytes2.third)
                val p31 = vdotqS32(q3bytes31, q8bytes2.field3)
                isum += vaddvqS32(p01) * ((PrimitiveLayouts.I8).readNative((scale).rawAddress)).toInt() + vaddvqS32(p11) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((1).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt() + vaddvqS32(p21) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((2).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt() + vaddvqS32(p31) * (PrimitiveLayouts.I8).readNative((MutPtr((scale).rawAddress + ((3).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt()
                scale = MutPtr((scale).rawAddress + ((4).toLong() * (PrimitiveLayouts.I8).strideBytes))
                if (j1 == 0) {
                    qhbits.first = vshrqNU8(qhbits.first, 4)
                    qhbits.second = vshrqNU8(qhbits.second, 4)
                }
            }
            sumf += d * isum.toFloat()
        }
    }
    return sumf
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    check(n.isMultipleOf(QK_K)) { "vec_dot_q2k_q8k: $n is not divisible by $qkK" }
    var sumf = 0.0f
    val localStorageL3930C4 = AddressableArrayLocal(UByteArray(16) { 0u.toUByte() }, FixedArrayTypeLayout(PrimitiveLayouts.U8, (16).toInt()))
    var aux by localStorageL3930C4
    run {
        val m3 = vdupqNU8(0x3)
        val m4 = vdupqNU8(0xF)
        for ((x, y) in (xs.zip(ys))) {
            val d = y.d * x.d.toF32()
            val dmin = -y.d * x.dmin.toF32()
            var q2 = x.qs.asPtr()
            var q8 = y.qs.asPtr()
            val sc = x.scales.asPtr()
            val minsAndScales = vld1qU8(sc)
            val scales = vandqU8(minsAndScales, m4)
            vst1qU8(localStorageL3930C4.elementMutPtr(0), scales)
            val mins = vshrqNU8(minsAndScales, 4)
            val q8sums = vld1qS16X2(y.bsums.asPtr())
            val mins16 = int16x8x2T(vreinterpretqS16U16(vmovlU8(vgetLowU8(mins))), vreinterpretqS16U16(vmovlU8(vgetHighU8(mins))))
            val s0 = vaddqS32(vmullS16(vgetLowS16(mins16.first), vgetLowS16(q8sums.first)), vmullS16(vgetHighS16(mins16.first), vgetHighS16(q8sums.first)))
            val s1 = vaddqS32(vmullS16(vgetLowS16(mins16.second), vgetLowS16(q8sums.second)), vmullS16(vgetHighS16(mins16.second), vgetHighS16(q8sums.second)))
            sumf += dmin * vaddvqS32(vaddqS32(s0, s1)).toFloat()
            var isum = 0
            var isMatch = 0
            // Note: dotprod
            for (j in 0 until QK_K / 128) {
                val q2bits = vld1qU8X2(q2)
                q2 = q2.add(32)
                val q8bytes = vld1qS8X2(q8)
                q8 = q8.add(32)
                val q2bytes = int8x16x2T(vreinterpretqS8U8(vandqU8(q2bits.first, m3)), vreinterpretqS8U8(vandqU8(q2bits.second, m3)))
                isum += multiplyAccumWithScale(aux, isMatch, 0, q2bytes, q8bytes)
                val q8bytes1 = vld1qS8X2(q8)
                q8 = q8.add(32)
                q2bytes.first = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.first, 2), m3))
                q2bytes.second = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.second, 2), m3))
                isum += multiplyAccumWithScale(aux, isMatch, 2, q2bytes, q8bytes1)
                val q8bytes2 = vld1qS8X2(q8)
                q8 = q8.add(32)
                q2bytes.first = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.first, 4), m3))
                q2bytes.second = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.second, 4), m3))
                isum += multiplyAccumWithScale(aux, isMatch, 4, q2bytes, q8bytes2)
                val q8bytes3 = vld1qS8X2(q8)
                q8 = q8.add(32)
                q2bytes.first = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.first, 6), m3))
                q2bytes.second = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits.second, 6), m3))
                isum += multiplyAccumWithScale(aux, isMatch, 6, q2bytes, q8bytes3)
                isMatch += 8
            }
            sumf += d * isum.toFloat()
        }
    }
    return sumf
}

/**
 * Quantize a row of f32 activations into Q8K format
 */
internal fun quantizeRowQ8k(xs: List<Float>, ys: MutableList<BlockQ8K>) {
    check(xs.size.isMultipleOf(QK_K)) { "quantize_row_q8k: ${xs.size} is not a multiple of $qkK" }
    run {
        val (xChunks, _) = xs.asChunks<QkK>()
        for ((chunk, y) in (xChunks.iter().zip(ys))) {
            // Find the element with the maximum absolute value, preserving its sign.
            var (vabsMax0, vsmax0) = Pair(vdupqNF32(0.0), vdupqNF32(0.0))
            var (vabsMax1, vsmax1) = Pair(vdupqNF32(0.0), vdupqNF32(0.0))
            var (vabsMax2, vsmax2) = Pair(vdupqNF32(0.0), vdupqNF32(0.0))
            var (vabsMax3, vsmax3) = Pair(vdupqNF32(0.0), vdupqNF32(0.0))
            var p = chunk.asPtr()
            for (unused in 0 until QK_K / 16) {
                val (v0, v1) = Pair(vld1qF32(p), vld1qF32(p.add(4)))
                val (v2, v3) = Pair(vld1qF32(p.add(8)), vld1qF32(p.add(12)))
                p = p.add(16)
                Pair(vabsMax0, vsmax0) = mergeSignedMax(vabsMax0, vsmax0, vabsqF32(v0), v0)
                Pair(vabsMax1, vsmax1) = mergeSignedMax(vabsMax1, vsmax1, vabsqF32(v1), v1)
                Pair(vabsMax2, vsmax2) = mergeSignedMax(vabsMax2, vsmax2, vabsqF32(v2), v2)
                Pair(vabsMax3, vsmax3) = mergeSignedMax(vabsMax3, vsmax3, vabsqF32(v3), v3)
            }
            // Tree-reduce 4 accumulators to 1.
            val (abs01, smax01) = mergeSignedMax(vabsMax0, vsmax0, vabsMax1, vsmax1)
            val (abs23, smax23) = mergeSignedMax(vabsMax2, vsmax2, vabsMax3, vsmax3)
            val (absV, smaxV) = mergeSignedMax(abs01, smax01, abs23, smax23)
            // Cross lane reduce to scalar
            val maskLohi = vcgtF32(vgetHighF32(absV), vgetLowF32(absV))
            val absPair = vmaxF32(vgetLowF32(absV), vgetHighF32(absV))
            val smaxPair = vbslF32(maskLohi, vgetHighF32(smaxV), vgetLowF32(smaxV))
            val maxSigned =
                if (vgetLaneF32(absPair, 1) > vgetLaneF32(absPair, 0)) {
                    vgetLaneF32(smaxPair, 1)
                } else {
                    vgetLaneF32(smaxPair, 0)
                }
            if (maxSigned == 0.0f) {
                y.d = 0.0f
                y.qs.fill(0)
                y.bsums.fill(0)
                continue
            }
            val iscale = -127.0f / maxSigned
            val vscale = vdupqNF32(iscale)
            // Quantize f32 -> i8. Multiply, round-to-nearest, saturating narrow.
            var out = y.qs.asMutPtr()
            var p1 = chunk.asPtr()
            for (j in 0 until QK_K / 16) {
                val f0 = vmulqF32(vld1qF32(p1), vscale)
                val f1 = vmulqF32(vld1qF32(p1.add(4)), vscale)
                val f2 = vmulqF32(vld1qF32(p1.add(8)), vscale)
                val f3 = vmulqF32(vld1qF32(p1.add(12)), vscale)
                p1 = p1.add(16)
                val s01 = vcombineS16(vqmovnS32(vcvtaqS32F32(f0)), vqmovnS32(vcvtaqS32F32(f1)))
                val s23 = vcombineS16(vqmovnS32(vcvtaqS32F32(f2)), vqmovnS32(vcvtaqS32F32(f3)))
                val q = vcombineS8(vqmovnS16(s01), vqmovnS16(s23))
                vst1qS8(out, q)
                out = out.add(16)
                y.bsums[j] = vaddvqS32(vpaddlqS16(vpaddlqS8(q))).toShort()
            }
            y.d = 1.0f / iscale
        }
    }
}

public fun multiplyAccumWithScale(aux: UByteArray, isMatch: Int, index: Int, q2bytes: Int8x16x2T, q8bytes: Int8x16x2T): Int {
    val p1 = vdotqS32(q2bytes.first, q8bytes.first)
    val p2 = vdotqS32(q2bytes.second, q8bytes.second)
    return vaddvqS32(p1) * aux[isMatch + index].toInt() + vaddvqS32(p2) * aux[isMatch + 1 + index].toInt()
}

internal fun vecDot8Q4kQ8k(n: Int, xs: List<BlockQ4Kx8>, ys: List<BlockQ8K>): FloatArray {
    fun loadF16x4(ptr: ConstPtr): Float32x4T {
        val raw = vld1U64(ConstPtr(ptr.rawAddress, ptr.rawMeta))
        var result: Float32x4T
        asm("fcvtl {out:v}.4s, {inp:v}.4h", inp, vreg, raw, out, out, vreg, result, options(nostack, nomem))
        return result
    }
    check(n.isMultipleOf(QK_K))
    val out = List(8) { 0.0f }
    run {
        var vacc0 = vdupqNF32(0.0f)
        var vacc1 = vdupqNF32(0.0f)
        val m4b = vdupqNU8(0x0f)
        for ((q4, q8) in (xs.zip(ys))) {
            val q8d = q8.d
            val q8dV = vdupqNF32(q8d)
            val sbScale0 = vmulqF32(loadF16x4(q4.d.asPtr()), q8dV)
            val sbScale1 = vmulqF32(loadF16x4(q4.d.asPtr().add(4)), q8dV)
            val sbMin0 = vmulqF32(loadF16x4(q4.dmin.asPtr()), q8dV)
            val sbMin1 = vmulqF32(loadF16x4(q4.dmin.asPtr().add(4)), q8dV)
            val bsums = vpaddqS16(vld1qS16(q8.bsums.asPtr()), vld1qS16(q8.bsums.asPtr().add(8)))
            var bias0 = vdupqNS32(0)
            var bias1 = vdupqNS32(0)
            run {
                val (mins0, sc0) = decodeQ4kx8Scales(q4.scales.asPtr().add(0 * 24))
                val (mins1, sc1) = decodeQ4kx8Scales(q4.scales.asPtr().add(0 * 24 + 12))
                val q8p = q8.qs.asPtr().add(0 * 64)
                val q80 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p)))
                val q81 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(8))))
                val q82 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(16))))
                val q83 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(24))))
                val q84 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(32))))
                val q85 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(40))))
                val q86 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(48))))
                val q87 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p.add(56))))
                val q4p = q4.qs.asPtr().add(0 * QK_K)
                val s0 = vld1qU8X2(q4p)
                val s0h = vld1qU8X2(q4p.add(32))
                val s1 = vld1qU8X2(q4p.add(64))
                val s1h = vld1qU8X2(q4p.add(96))
                val s2 = vld1qU8X2(q4p.add(128))
                val s2h = vld1qU8X2(q4p.add(160))
                val s3 = vld1qU8X2(q4p.add(192))
                val s3h = vld1qU8X2(q4p.add(224))
                val (b00, b10) = Pair(s0.first, s0.second)
                val (b20, b30) = Pair(s0h.first, s0h.second)
                val (b01, b11) = Pair(s1.first, s1.second)
                val (b21, b31) = Pair(s1h.first, s1h.second)
                val (b02, b12) = Pair(s2.first, s2.second)
                val (b22, b32) = Pair(s2h.first, s2h.second)
                val (b03, b13) = Pair(s3.first, s3.second)
                val (b23, b33) = Pair(s3h.first, s3h.second)
                var a0 = vdupqNS32(0)
                a0 = sdotAcc(a0, vreinterpretqS8U8(vandqU8(b00, m4b)), q80)
                a0 = sdotAcc(a0, vreinterpretqS8U8(vandqU8(b01, m4b)), q81)
                a0 = sdotAcc(a0, vreinterpretqS8U8(vandqU8(b02, m4b)), q82)
                a0 = sdotAcc(a0, vreinterpretqS8U8(vandqU8(b03, m4b)), q83)
                var h0 = vdupqNS32(0)
                h0 = sdotAcc(h0, vreinterpretqS8U8(vshrqNU8(b00, 4)), q84)
                h0 = sdotAcc(h0, vreinterpretqS8U8(vshrqNU8(b01, 4)), q85)
                h0 = sdotAcc(h0, vreinterpretqS8U8(vshrqNU8(b02, 4)), q86)
                h0 = sdotAcc(h0, vreinterpretqS8U8(vshrqNU8(b03, 4)), q87)
                var a1 = vdupqNS32(0)
                a1 = sdotAcc(a1, vreinterpretqS8U8(vandqU8(b10, m4b)), q80)
                a1 = sdotAcc(a1, vreinterpretqS8U8(vandqU8(b11, m4b)), q81)
                a1 = sdotAcc(a1, vreinterpretqS8U8(vandqU8(b12, m4b)), q82)
                a1 = sdotAcc(a1, vreinterpretqS8U8(vandqU8(b13, m4b)), q83)
                var h1 = vdupqNS32(0)
                h1 = sdotAcc(h1, vreinterpretqS8U8(vshrqNU8(b10, 4)), q84)
                h1 = sdotAcc(h1, vreinterpretqS8U8(vshrqNU8(b11, 4)), q85)
                h1 = sdotAcc(h1, vreinterpretqS8U8(vshrqNU8(b12, 4)), q86)
                h1 = sdotAcc(h1, vreinterpretqS8U8(vshrqNU8(b13, 4)), q87)
                val sumfLo03 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc0)), vpaddqS32(a0, a1)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfLo03)
                val sumfHi03 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc1)), vpaddqS32(h0, h1)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfHi03)
                var a2 = vdupqNS32(0)
                a2 = sdotAcc(a2, vreinterpretqS8U8(vandqU8(b20, m4b)), q80)
                a2 = sdotAcc(a2, vreinterpretqS8U8(vandqU8(b21, m4b)), q81)
                a2 = sdotAcc(a2, vreinterpretqS8U8(vandqU8(b22, m4b)), q82)
                a2 = sdotAcc(a2, vreinterpretqS8U8(vandqU8(b23, m4b)), q83)
                var h2 = vdupqNS32(0)
                h2 = sdotAcc(h2, vreinterpretqS8U8(vshrqNU8(b20, 4)), q84)
                h2 = sdotAcc(h2, vreinterpretqS8U8(vshrqNU8(b21, 4)), q85)
                h2 = sdotAcc(h2, vreinterpretqS8U8(vshrqNU8(b22, 4)), q86)
                h2 = sdotAcc(h2, vreinterpretqS8U8(vshrqNU8(b23, 4)), q87)
                var a3 = vdupqNS32(0)
                a3 = sdotAcc(a3, vreinterpretqS8U8(vandqU8(b30, m4b)), q80)
                a3 = sdotAcc(a3, vreinterpretqS8U8(vandqU8(b31, m4b)), q81)
                a3 = sdotAcc(a3, vreinterpretqS8U8(vandqU8(b32, m4b)), q82)
                a3 = sdotAcc(a3, vreinterpretqS8U8(vandqU8(b33, m4b)), q83)
                var h3 = vdupqNS32(0)
                h3 = sdotAcc(h3, vreinterpretqS8U8(vshrqNU8(b30, 4)), q84)
                h3 = sdotAcc(h3, vreinterpretqS8U8(vshrqNU8(b31, 4)), q85)
                h3 = sdotAcc(h3, vreinterpretqS8U8(vshrqNU8(b32, 4)), q86)
                h3 = sdotAcc(h3, vreinterpretqS8U8(vshrqNU8(b33, 4)), q87)
                val sumfLo47 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc0)), vpaddqS32(a2, a3)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfLo47)
                val sumfHi47 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc1)), vpaddqS32(h2, h3)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfHi47)
                val bl = vdupNS16(vgetqLaneS16(bsums, 0 * 2))
                val bh = vdupNS16(vgetqLaneS16(bsums, 0 * 2 + 1))
                bias0 = vmlalS16(bias0, bl, vgetLowS16(mins0))
                bias0 = vmlalS16(bias0, bh, vgetLowS16(mins1))
                bias1 = vmlalS16(bias1, bl, vgetHighS16(mins0))
                bias1 = vmlalS16(bias1, bh, vgetHighS16(mins1))
            }
            run {
                val (mins01, sc01) = decodeQ4kx8Scales(q4.scales.asPtr().add(1 * 24))
                val (mins11, sc11) = decodeQ4kx8Scales(q4.scales.asPtr().add(1 * 24 + 12))
                val q8p1 = q8.qs.asPtr().add(1 * 64)
                val q801 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1)))
                val q811 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(8))))
                val q821 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(16))))
                val q831 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(24))))
                val q841 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(32))))
                val q851 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(40))))
                val q861 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(48))))
                val q871 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p1.add(56))))
                val q4p1 = q4.qs.asPtr().add(1 * QK_K)
                val s01 = vld1qU8X2(q4p1)
                val s0h1 = vld1qU8X2(q4p1.add(32))
                val s11 = vld1qU8X2(q4p1.add(64))
                val s1h1 = vld1qU8X2(q4p1.add(96))
                val s21 = vld1qU8X2(q4p1.add(128))
                val s2h1 = vld1qU8X2(q4p1.add(160))
                val s31 = vld1qU8X2(q4p1.add(192))
                val s3h1 = vld1qU8X2(q4p1.add(224))
                val (b001, b101) = Pair(s01.first, s01.second)
                val (b201, b301) = Pair(s0h1.first, s0h1.second)
                val (b011, b111) = Pair(s11.first, s11.second)
                val (b211, b311) = Pair(s1h1.first, s1h1.second)
                val (b021, b121) = Pair(s21.first, s21.second)
                val (b221, b321) = Pair(s2h1.first, s2h1.second)
                val (b031, b131) = Pair(s31.first, s31.second)
                val (b231, b331) = Pair(s3h1.first, s3h1.second)
                var a01 = vdupqNS32(0)
                a01 = sdotAcc(a01, vreinterpretqS8U8(vandqU8(b001, m4b)), q801)
                a01 = sdotAcc(a01, vreinterpretqS8U8(vandqU8(b011, m4b)), q811)
                a01 = sdotAcc(a01, vreinterpretqS8U8(vandqU8(b021, m4b)), q821)
                a01 = sdotAcc(a01, vreinterpretqS8U8(vandqU8(b031, m4b)), q831)
                var h01 = vdupqNS32(0)
                h01 = sdotAcc(h01, vreinterpretqS8U8(vshrqNU8(b001, 4)), q841)
                h01 = sdotAcc(h01, vreinterpretqS8U8(vshrqNU8(b011, 4)), q851)
                h01 = sdotAcc(h01, vreinterpretqS8U8(vshrqNU8(b021, 4)), q861)
                h01 = sdotAcc(h01, vreinterpretqS8U8(vshrqNU8(b031, 4)), q871)
                var a11 = vdupqNS32(0)
                a11 = sdotAcc(a11, vreinterpretqS8U8(vandqU8(b101, m4b)), q801)
                a11 = sdotAcc(a11, vreinterpretqS8U8(vandqU8(b111, m4b)), q811)
                a11 = sdotAcc(a11, vreinterpretqS8U8(vandqU8(b121, m4b)), q821)
                a11 = sdotAcc(a11, vreinterpretqS8U8(vandqU8(b131, m4b)), q831)
                var h11 = vdupqNS32(0)
                h11 = sdotAcc(h11, vreinterpretqS8U8(vshrqNU8(b101, 4)), q841)
                h11 = sdotAcc(h11, vreinterpretqS8U8(vshrqNU8(b111, 4)), q851)
                h11 = sdotAcc(h11, vreinterpretqS8U8(vshrqNU8(b121, 4)), q861)
                h11 = sdotAcc(h11, vreinterpretqS8U8(vshrqNU8(b131, 4)), q871)
                val sumfLo031 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc01)), vpaddqS32(a01, a11)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfLo031)
                val sumfHi031 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc11)), vpaddqS32(h01, h11)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfHi031)
                var a21 = vdupqNS32(0)
                a21 = sdotAcc(a21, vreinterpretqS8U8(vandqU8(b201, m4b)), q801)
                a21 = sdotAcc(a21, vreinterpretqS8U8(vandqU8(b211, m4b)), q811)
                a21 = sdotAcc(a21, vreinterpretqS8U8(vandqU8(b221, m4b)), q821)
                a21 = sdotAcc(a21, vreinterpretqS8U8(vandqU8(b231, m4b)), q831)
                var h21 = vdupqNS32(0)
                h21 = sdotAcc(h21, vreinterpretqS8U8(vshrqNU8(b201, 4)), q841)
                h21 = sdotAcc(h21, vreinterpretqS8U8(vshrqNU8(b211, 4)), q851)
                h21 = sdotAcc(h21, vreinterpretqS8U8(vshrqNU8(b221, 4)), q861)
                h21 = sdotAcc(h21, vreinterpretqS8U8(vshrqNU8(b231, 4)), q871)
                var a31 = vdupqNS32(0)
                a31 = sdotAcc(a31, vreinterpretqS8U8(vandqU8(b301, m4b)), q801)
                a31 = sdotAcc(a31, vreinterpretqS8U8(vandqU8(b311, m4b)), q811)
                a31 = sdotAcc(a31, vreinterpretqS8U8(vandqU8(b321, m4b)), q821)
                a31 = sdotAcc(a31, vreinterpretqS8U8(vandqU8(b331, m4b)), q831)
                var h31 = vdupqNS32(0)
                h31 = sdotAcc(h31, vreinterpretqS8U8(vshrqNU8(b301, 4)), q841)
                h31 = sdotAcc(h31, vreinterpretqS8U8(vshrqNU8(b311, 4)), q851)
                h31 = sdotAcc(h31, vreinterpretqS8U8(vshrqNU8(b321, 4)), q861)
                h31 = sdotAcc(h31, vreinterpretqS8U8(vshrqNU8(b331, 4)), q871)
                val sumfLo471 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc01)), vpaddqS32(a21, a31)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfLo471)
                val sumfHi471 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc11)), vpaddqS32(h21, h31)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfHi471)
                val bl1 = vdupNS16(vgetqLaneS16(bsums, 1 * 2))
                val bh1 = vdupNS16(vgetqLaneS16(bsums, 1 * 2 + 1))
                bias0 = vmlalS16(bias0, bl1, vgetLowS16(mins01))
                bias0 = vmlalS16(bias0, bh1, vgetLowS16(mins11))
                bias1 = vmlalS16(bias1, bl1, vgetHighS16(mins01))
                bias1 = vmlalS16(bias1, bh1, vgetHighS16(mins11))
            }
            run {
                val (mins02, sc02) = decodeQ4kx8Scales(q4.scales.asPtr().add(2 * 24))
                val (mins12, sc12) = decodeQ4kx8Scales(q4.scales.asPtr().add(2 * 24 + 12))
                val q8p2 = q8.qs.asPtr().add(2 * 64)
                val q802 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2)))
                val q812 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(8))))
                val q822 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(16))))
                val q832 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(24))))
                val q842 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(32))))
                val q852 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(40))))
                val q862 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(48))))
                val q872 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p2.add(56))))
                val q4p2 = q4.qs.asPtr().add(2 * QK_K)
                val s02 = vld1qU8X2(q4p2)
                val s0h2 = vld1qU8X2(q4p2.add(32))
                val s12 = vld1qU8X2(q4p2.add(64))
                val s1h2 = vld1qU8X2(q4p2.add(96))
                val s22 = vld1qU8X2(q4p2.add(128))
                val s2h2 = vld1qU8X2(q4p2.add(160))
                val s32 = vld1qU8X2(q4p2.add(192))
                val s3h2 = vld1qU8X2(q4p2.add(224))
                val (b002, b102) = Pair(s02.first, s02.second)
                val (b202, b302) = Pair(s0h2.first, s0h2.second)
                val (b012, b112) = Pair(s12.first, s12.second)
                val (b212, b312) = Pair(s1h2.first, s1h2.second)
                val (b022, b122) = Pair(s22.first, s22.second)
                val (b222, b322) = Pair(s2h2.first, s2h2.second)
                val (b032, b132) = Pair(s32.first, s32.second)
                val (b232, b332) = Pair(s3h2.first, s3h2.second)
                var a02 = vdupqNS32(0)
                a02 = sdotAcc(a02, vreinterpretqS8U8(vandqU8(b002, m4b)), q802)
                a02 = sdotAcc(a02, vreinterpretqS8U8(vandqU8(b012, m4b)), q812)
                a02 = sdotAcc(a02, vreinterpretqS8U8(vandqU8(b022, m4b)), q822)
                a02 = sdotAcc(a02, vreinterpretqS8U8(vandqU8(b032, m4b)), q832)
                var h02 = vdupqNS32(0)
                h02 = sdotAcc(h02, vreinterpretqS8U8(vshrqNU8(b002, 4)), q842)
                h02 = sdotAcc(h02, vreinterpretqS8U8(vshrqNU8(b012, 4)), q852)
                h02 = sdotAcc(h02, vreinterpretqS8U8(vshrqNU8(b022, 4)), q862)
                h02 = sdotAcc(h02, vreinterpretqS8U8(vshrqNU8(b032, 4)), q872)
                var a12 = vdupqNS32(0)
                a12 = sdotAcc(a12, vreinterpretqS8U8(vandqU8(b102, m4b)), q802)
                a12 = sdotAcc(a12, vreinterpretqS8U8(vandqU8(b112, m4b)), q812)
                a12 = sdotAcc(a12, vreinterpretqS8U8(vandqU8(b122, m4b)), q822)
                a12 = sdotAcc(a12, vreinterpretqS8U8(vandqU8(b132, m4b)), q832)
                var h12 = vdupqNS32(0)
                h12 = sdotAcc(h12, vreinterpretqS8U8(vshrqNU8(b102, 4)), q842)
                h12 = sdotAcc(h12, vreinterpretqS8U8(vshrqNU8(b112, 4)), q852)
                h12 = sdotAcc(h12, vreinterpretqS8U8(vshrqNU8(b122, 4)), q862)
                h12 = sdotAcc(h12, vreinterpretqS8U8(vshrqNU8(b132, 4)), q872)
                val sumfLo032 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc02)), vpaddqS32(a02, a12)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfLo032)
                val sumfHi032 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc12)), vpaddqS32(h02, h12)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfHi032)
                var a22 = vdupqNS32(0)
                a22 = sdotAcc(a22, vreinterpretqS8U8(vandqU8(b202, m4b)), q802)
                a22 = sdotAcc(a22, vreinterpretqS8U8(vandqU8(b212, m4b)), q812)
                a22 = sdotAcc(a22, vreinterpretqS8U8(vandqU8(b222, m4b)), q822)
                a22 = sdotAcc(a22, vreinterpretqS8U8(vandqU8(b232, m4b)), q832)
                var h22 = vdupqNS32(0)
                h22 = sdotAcc(h22, vreinterpretqS8U8(vshrqNU8(b202, 4)), q842)
                h22 = sdotAcc(h22, vreinterpretqS8U8(vshrqNU8(b212, 4)), q852)
                h22 = sdotAcc(h22, vreinterpretqS8U8(vshrqNU8(b222, 4)), q862)
                h22 = sdotAcc(h22, vreinterpretqS8U8(vshrqNU8(b232, 4)), q872)
                var a32 = vdupqNS32(0)
                a32 = sdotAcc(a32, vreinterpretqS8U8(vandqU8(b302, m4b)), q802)
                a32 = sdotAcc(a32, vreinterpretqS8U8(vandqU8(b312, m4b)), q812)
                a32 = sdotAcc(a32, vreinterpretqS8U8(vandqU8(b322, m4b)), q822)
                a32 = sdotAcc(a32, vreinterpretqS8U8(vandqU8(b332, m4b)), q832)
                var h32 = vdupqNS32(0)
                h32 = sdotAcc(h32, vreinterpretqS8U8(vshrqNU8(b302, 4)), q842)
                h32 = sdotAcc(h32, vreinterpretqS8U8(vshrqNU8(b312, 4)), q852)
                h32 = sdotAcc(h32, vreinterpretqS8U8(vshrqNU8(b322, 4)), q862)
                h32 = sdotAcc(h32, vreinterpretqS8U8(vshrqNU8(b332, 4)), q872)
                val sumfLo472 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc02)), vpaddqS32(a22, a32)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfLo472)
                val sumfHi472 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc12)), vpaddqS32(h22, h32)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfHi472)
                val bl2 = vdupNS16(vgetqLaneS16(bsums, 2 * 2))
                val bh2 = vdupNS16(vgetqLaneS16(bsums, 2 * 2 + 1))
                bias0 = vmlalS16(bias0, bl2, vgetLowS16(mins02))
                bias0 = vmlalS16(bias0, bh2, vgetLowS16(mins12))
                bias1 = vmlalS16(bias1, bl2, vgetHighS16(mins02))
                bias1 = vmlalS16(bias1, bh2, vgetHighS16(mins12))
            }
            run {
                val (mins03, sc03) = decodeQ4kx8Scales(q4.scales.asPtr().add(3 * 24))
                val (mins13, sc13) = decodeQ4kx8Scales(q4.scales.asPtr().add(3 * 24 + 12))
                val q8p3 = q8.qs.asPtr().add(3 * 64)
                val q803 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3)))
                val q813 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(8))))
                val q823 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(16))))
                val q833 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(24))))
                val q843 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(32))))
                val q853 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(40))))
                val q863 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(48))))
                val q873 = vreinterpretqS8S64(vld1qDupS64(constPtrFromRustCast(q8p3.add(56))))
                val q4p3 = q4.qs.asPtr().add(3 * QK_K)
                val s03 = vld1qU8X2(q4p3)
                val s0h3 = vld1qU8X2(q4p3.add(32))
                val s13 = vld1qU8X2(q4p3.add(64))
                val s1h3 = vld1qU8X2(q4p3.add(96))
                val s23 = vld1qU8X2(q4p3.add(128))
                val s2h3 = vld1qU8X2(q4p3.add(160))
                val s33 = vld1qU8X2(q4p3.add(192))
                val s3h3 = vld1qU8X2(q4p3.add(224))
                val (b003, b103) = Pair(s03.first, s03.second)
                val (b203, b303) = Pair(s0h3.first, s0h3.second)
                val (b013, b113) = Pair(s13.first, s13.second)
                val (b213, b313) = Pair(s1h3.first, s1h3.second)
                val (b023, b123) = Pair(s23.first, s23.second)
                val (b223, b323) = Pair(s2h3.first, s2h3.second)
                val (b033, b133) = Pair(s33.first, s33.second)
                val (b233, b333) = Pair(s3h3.first, s3h3.second)
                var a03 = vdupqNS32(0)
                a03 = sdotAcc(a03, vreinterpretqS8U8(vandqU8(b003, m4b)), q803)
                a03 = sdotAcc(a03, vreinterpretqS8U8(vandqU8(b013, m4b)), q813)
                a03 = sdotAcc(a03, vreinterpretqS8U8(vandqU8(b023, m4b)), q823)
                a03 = sdotAcc(a03, vreinterpretqS8U8(vandqU8(b033, m4b)), q833)
                var h03 = vdupqNS32(0)
                h03 = sdotAcc(h03, vreinterpretqS8U8(vshrqNU8(b003, 4)), q843)
                h03 = sdotAcc(h03, vreinterpretqS8U8(vshrqNU8(b013, 4)), q853)
                h03 = sdotAcc(h03, vreinterpretqS8U8(vshrqNU8(b023, 4)), q863)
                h03 = sdotAcc(h03, vreinterpretqS8U8(vshrqNU8(b033, 4)), q873)
                var a13 = vdupqNS32(0)
                a13 = sdotAcc(a13, vreinterpretqS8U8(vandqU8(b103, m4b)), q803)
                a13 = sdotAcc(a13, vreinterpretqS8U8(vandqU8(b113, m4b)), q813)
                a13 = sdotAcc(a13, vreinterpretqS8U8(vandqU8(b123, m4b)), q823)
                a13 = sdotAcc(a13, vreinterpretqS8U8(vandqU8(b133, m4b)), q833)
                var h13 = vdupqNS32(0)
                h13 = sdotAcc(h13, vreinterpretqS8U8(vshrqNU8(b103, 4)), q843)
                h13 = sdotAcc(h13, vreinterpretqS8U8(vshrqNU8(b113, 4)), q853)
                h13 = sdotAcc(h13, vreinterpretqS8U8(vshrqNU8(b123, 4)), q863)
                h13 = sdotAcc(h13, vreinterpretqS8U8(vshrqNU8(b133, 4)), q873)
                val sumfLo033 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc03)), vpaddqS32(a03, a13)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfLo033)
                val sumfHi033 = vcvtqF32S32(vmulqS32(vmovlS16(vgetLowS16(sc13)), vpaddqS32(h03, h13)))
                vacc0 = vfmaqF32(vacc0, sbScale0, sumfHi033)
                var a23 = vdupqNS32(0)
                a23 = sdotAcc(a23, vreinterpretqS8U8(vandqU8(b203, m4b)), q803)
                a23 = sdotAcc(a23, vreinterpretqS8U8(vandqU8(b213, m4b)), q813)
                a23 = sdotAcc(a23, vreinterpretqS8U8(vandqU8(b223, m4b)), q823)
                a23 = sdotAcc(a23, vreinterpretqS8U8(vandqU8(b233, m4b)), q833)
                var h23 = vdupqNS32(0)
                h23 = sdotAcc(h23, vreinterpretqS8U8(vshrqNU8(b203, 4)), q843)
                h23 = sdotAcc(h23, vreinterpretqS8U8(vshrqNU8(b213, 4)), q853)
                h23 = sdotAcc(h23, vreinterpretqS8U8(vshrqNU8(b223, 4)), q863)
                h23 = sdotAcc(h23, vreinterpretqS8U8(vshrqNU8(b233, 4)), q873)
                var a33 = vdupqNS32(0)
                a33 = sdotAcc(a33, vreinterpretqS8U8(vandqU8(b303, m4b)), q803)
                a33 = sdotAcc(a33, vreinterpretqS8U8(vandqU8(b313, m4b)), q813)
                a33 = sdotAcc(a33, vreinterpretqS8U8(vandqU8(b323, m4b)), q823)
                a33 = sdotAcc(a33, vreinterpretqS8U8(vandqU8(b333, m4b)), q833)
                var h33 = vdupqNS32(0)
                h33 = sdotAcc(h33, vreinterpretqS8U8(vshrqNU8(b303, 4)), q843)
                h33 = sdotAcc(h33, vreinterpretqS8U8(vshrqNU8(b313, 4)), q853)
                h33 = sdotAcc(h33, vreinterpretqS8U8(vshrqNU8(b323, 4)), q863)
                h33 = sdotAcc(h33, vreinterpretqS8U8(vshrqNU8(b333, 4)), q873)
                val sumfLo473 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc03)), vpaddqS32(a23, a33)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfLo473)
                val sumfHi473 = vcvtqF32S32(vmulqS32(vmovlS16(vgetHighS16(sc13)), vpaddqS32(h23, h33)))
                vacc1 = vfmaqF32(vacc1, sbScale1, sumfHi473)
                val bl3 = vdupNS16(vgetqLaneS16(bsums, 3 * 2))
                val bh3 = vdupNS16(vgetqLaneS16(bsums, 3 * 2 + 1))
                bias0 = vmlalS16(bias0, bl3, vgetLowS16(mins03))
                bias0 = vmlalS16(bias0, bh3, vgetLowS16(mins13))
                bias1 = vmlalS16(bias1, bl3, vgetHighS16(mins03))
                bias1 = vmlalS16(bias1, bh3, vgetHighS16(mins13))
            }
            vacc0 = vmlsqF32(vacc0, vcvtqF32S32(bias0), sbMin0)
            vacc1 = vmlsqF32(vacc1, vcvtqF32S32(bias1), sbMin1)
        }
        vst1qF32(out.asMutPtr(), vacc0)
        vst1qF32(out.asMutPtr().add(4), vacc1)
    }
    return out
}
