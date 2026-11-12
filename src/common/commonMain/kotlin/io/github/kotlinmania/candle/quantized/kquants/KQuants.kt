// port-lint: source quantized/k_quants.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.kquants

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.cpu.features.get
import io.github.kotlinmania.candle.cpu.vecDotBf16
import io.github.kotlinmania.candle.cpu.vecDotF16
import io.github.kotlinmania.candle.cpu.vecDotF32
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.avx.vecDotQ2kQ8k
import io.github.kotlinmania.candle.quantized.avx.vecDotQ3kQ8k
import io.github.kotlinmania.candle.quantized.avx.vecDotQ40Q80
import io.github.kotlinmania.candle.quantized.avx.vecDotQ4kQ8k
import io.github.kotlinmania.candle.quantized.avx.vecDotQ5kQ8k
import io.github.kotlinmania.candle.quantized.avx.vecDotQ6kQ8k
import io.github.kotlinmania.candle.quantized.avx.vecDotQ80Q80
import io.github.kotlinmania.candle.quantized.avx.vecDotQ8kQ8k
import io.github.kotlinmania.candle.quantized.neon.quantizeQ80
import io.github.kotlinmania.candle.quantized.neon.quantizeRowQ8k
import io.github.kotlinmania.candle.quantized.neon.vecDot4Q4kQ8k
import io.github.kotlinmania.candle.quantized.neon.vecDot4Q6kQ8k
import io.github.kotlinmania.candle.quantized.neon.vecDot4Q80Q80
import io.github.kotlinmania.candle.quantized.neon.vecDot8Q4kQ8k
import io.github.kotlinmania.candle.quantized.repack.BlockQ4Kx8
import io.github.kotlinmania.candle.quantized.utils.getScaleMinK4
import io.github.kotlinmania.candle.quantized.utils.groupForDequantization
import io.github.kotlinmania.candle.quantized.utils.groupForQuantization
import io.github.kotlinmania.candle.quantized.utils.makeQ3Quants
import io.github.kotlinmania.candle.quantized.utils.makeQkx1Quants
import io.github.kotlinmania.candle.quantized.utils.makeQkx3Quants
import io.github.kotlinmania.candle.quantized.utils.makeQpQuants
import io.github.kotlinmania.candle.quantized.utils.makeQxQuants
import io.github.kotlinmania.candle.quantized.utils.nearestInt
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.AddressableArrayLocal
import io.github.kotlinmania.rustlang.core.mem.FixedArrayTypeLayout
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public fun hasAvx2Fma(): Boolean {
    val features = get()
    return features.avx2 && features.fma
}

// Default to QK_K 256 rather than 64.
public val QK_K: Int = 256

public val K_SCALE_SIZE: Int = 12

public val QK4_0: Int = 32

public val QK4_1: Int = 32

public val QK5_0: Int = 32

public val QK5_1: Int = 32

public val QK8_0: Int = 32

public val QK8_1: Int = 32

public interface GgmlType<T, VecDotType> {
    public val DTYPE: GgmlDType

    public val BLCK_SIZE: Int

    public val DIRECT_COPY: Boolean
        get() = false

    public fun zeros(): T = null.assumeInit()

    public fun toFloat(xs: List<T>, ys: List<Float>)

    public fun fromFloat(xs: List<Float>, ys: List<T>)

    public fun fromFloatImatrix(xs: List<Float>, ys: List<T>, imatrixWeights: List<Float>, nPerRow: Int) {
        error("`from_float_imatrix` is unimplemented for ${T.DTYPE}")
    }

    public fun directCopy(xs: List<Float>, ys: List<T>) {}

    /**
     * Dot product used as a building block for quantized matmul.
     * n is the number of elements to be considered.
     */
    public fun vecDot(n: Int, xs: List<T>, ys: List<VecDotType>): Float

    /**
     * Two dot products sharing one LHS load: returns (dot(xs0,ys), dot(xs1,ys)).
     * Can be overriden with optimized kernel
     */
    public fun vecDot2(n: Int, xs0: List<T>, xs1: List<T>, ys: List<VecDotType>): Pair<Float, Float> = Pair(n.vecDot(xs0, ys), n.vecDot(xs1, ys))

    /**
     * Four dot products sharing one LHS load: returns (dot(xs0,ys),..dot(xs3,ys)).
     * Can be overriden with optimized kernel
     */
    public fun vecDot4(n: Int, xs0: List<T>, xs1: List<T>, xs2: List<T>, xs3: List<T>, ys: List<VecDotType>): List<Any?> {
        val (d0, d1) = n.vecDot2(xs0, xs1, ys)
        val (d2, d3) = n.vecDot2(xs2, xs3, ys)
        return listOf(d0, d1, d2, d3)
    }

    /**
     * Generic implementation of the dot product without simd optimizations.
     */
    public fun vecDotUnopt(n: Int, xs: List<T>, ys: List<VecDotType>): Float
}

public class BlockQ40(
    internal var d: F16 = F16(),
    internal var qs: UByteArray = UByteArray(QK4_0 / 2),
) : GgmlType<BlockQ40> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q4_0

        public val BLCK_SIZE: Int = QK4_0

        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
        public fun toFloat(xs: List<BlockQ40>, ys: List<Float>) {
            val k = ys.size
            val qk = BlockQ40.BLCK_SIZE
            check(k.isMultipleOf(qk)) { "dequantize_row_q4_0: $k is not divisible by $qk" }
            val nb = k / qk
            for (i in 0 until nb) {
                val d = xs[i].d.toF32()
                for (j in 0 until (qk / 2)) {
                    val x0 = (xs[i].qs[j] and 0x0Fu.toUByte()).toShort() - 8
                    val x1 = ((xs[i].qs[j].toUInt() shr 4).toUByte()).toShort() - 8
                    ys[i * qk + j] = (x0.toFloat()) * d
                    ys[i * qk + j + qk / 2] = (x1.toFloat()) * d
                }
            }
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ40>) {
            // quantizeRowQ40
            val qk = BlockQ40.BLCK_SIZE
            val k = xs.size
            check(k.isMultipleOf(qk)) { "$k is not divisible by $qk" }
            check(ys.size == k / qk)
            for ((i, ys1) in ys.enumerate()) {
                var amax = 0.0f
                var max = 0.0f
                val xs1 = ((xs).slice(i * qk until (i + 1) * qk))
                for (x in xs1.iter()) {
                    if (amax < x.abs()) {
                        amax = x.abs()
                        max = x
                    }
                }
                val d = max / -8.0
                val id =
                    if (d != 0.0f) {
                        1.0 / d
                    } else {
                        0.0
                    }
                ys1.d = fromF32(d)
                for ((j, q) in ys1.qs.iterMut().enumerate()) {
                    val x0 = xs1[j] * id
                    val x1 = xs1[qk / 2 + j] * id
                    val xi0 = UByte.min(15, (x0 + 8.5).toUByte())
                    val xi1 = UByte.min(15, (x1 + 8.5).toUByte())
                    q = xi0 or (xi1 shl 4)
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
        public fun vecDot(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ40Q80(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ40Q80(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ40Q80(n, xs, ys)
            return BlockQ40.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
            check(n.isMultipleOf(QK8_0)) { "vec_dot_q4_0_q8_0: $n is not divisible by $qk80" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                var sumI = 0
                for (j in 0 until QK8_0 / 2) {
                    val v0 = (xs1.qs[j] and 0x0F).toInt() - 8
                    val v1 = (xs1.qs[j] shr 4).toInt() - 8
                    sumI += v0 * ys1.qs[j].toInt() + v1 * ys1.qs[j + QK8_0 / 2].toInt()
                }
                sumf += sumI.toFloat() * toF32(xs1.d) * toF32(ys1.d)
            }
            return sumf
        }
    }
}

public class BlockQ41(
    internal var d: F16 = F16(),
    internal var m: F16 = F16(),
    internal var qs: UByteArray = UByteArray(QK4_1 / 2),
) : GgmlType<BlockQ41> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q4_1

        public val BLCK_SIZE: Int = QK4_1

        public fun vecDot(n: ULong, xs: List<BlockQ41>, ys: List<BlockQ81>): Float = BlockQ41.vecDotUnopt(n, xs, ys)

        public fun vecDotUnopt(n: Int, xs: List<BlockQ41>, ys: List<BlockQ81>): Float {
            // ggmlVecDotQ41Q81
            val qk = QK8_1
            check(n.isMultipleOf(qk)) { "vec_dot_q4_1_q8_1: $n is not divisible by $qk" }
            check((n / qk).isMultipleOf(2)) { "vec_dot_q4_1_q8_1: $n, nb is not divisible by 2" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                var sumi = 0
                for (j in 0 until qk / 2) {
                    val v0 = xs1.qs[j].toInt() and 0x0F
                    val v1 = xs1.qs[j].toInt() shr 4
                    sumi += (v0 * ys1.qs[j].toInt()) + (v1 * ys1.qs[j + qk / 2].toInt())
                }
                sumf += sumi.toFloat() * toF32(xs1.d) * toF32(ys1.d) + toF32(xs1.m) * toF32(ys1.s)
            }
            return sumf
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ41>) {
            // quantizeRowQ41
            val qk = BlockQ41.BLCK_SIZE
            check(ys.size * qk == xs.size)
            for ((i, ys1) in ys.enumerate()) {
                val xs1 = ((xs).slice(i * qk until (i + 1) * qk))
                var min = Float.INFINITY
                var max = Float.NEG_INFINITY
                for (x in xs1.iter()) {
                    min = Float.min(x, min)
                    max = Float.max(x, max)
                }
                val d = (max - min) / ((1 shl 4) - 1).toFloat()
                val id =
                    if (d != 0.0f) {
                        1.0f / d
                    } else {
                        0.0
                    }
                ys1.d = fromF32(d)
                ys1.m = fromF32(min)
                for ((j, q) in ys1.qs
                    .iterMut()
                    .take(qk / 2)
                    .enumerate()) {
                    val x0 = (xs1[j] - min) * id
                    val x1 = (xs1[qk / 2 + j] - min) * id
                    val xi0 = UByte.min(15, (x0 + 0.5).toUByte())
                    val xi1 = UByte.min(15, (x1 + 0.5).toUByte())
                    q = xi0 or (xi1 shl 4)
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
        public fun toFloat(xs: List<BlockQ41>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK4_1)) { "dequantize_row_q4_1: $k is not divisible by $qk41" }
            val nb = k / QK4_1
            for (i in 0 until nb) {
                val d = xs[i].d.toF32()
                val m = xs[i].m.toF32()
                for (j in 0 until (QK4_1 / 2)) {
                    val x0 = xs[i].qs[j] and 0x0Fu.toUByte()
                    val x1 = (xs[i].qs[j].toUInt() shr 4).toUByte()
                    ys[i * QK4_1 + j] = (x0.toFloat()) * d + m
                    ys[i * QK4_1 + j + QK4_1 / 2] = (x1.toFloat()) * d + m
                }
            }
        }
    }
}

public class BlockQ50(
    internal var d: F16 = F16(),
    internal var qh: UByteArray = UByteArray(4),
    internal var qs: UByteArray = UByteArray(QK5_0 / 2),
) : GgmlType<BlockQ50> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q5_0

        public val BLCK_SIZE: Int = QK5_0

        public fun vecDot(n: Int, xs: List<BlockQ50>, ys: List<BlockQ80>): Float {
            val qk = BlockQ50.BLCK_SIZE
            check(n.isMultipleOf(qk)) { "vec_dot_q5_0_q8_0: $n is not divisible by $qk" }
            check((n / qk).isMultipleOf(2)) { "vec_dot_q5_0_q8_0: $n, nb is not divisible by 2" }
            return BlockQ50.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ50>, ys: List<BlockQ80>): Float {
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                val qh = LittleEndian.readU32(xs1.qh)
                var sumi = 0
                for (j in 0 until BlockQ50.BLCK_SIZE / 2) {
                    val xh0 = (((qh and (1u shl j.toInt())) shr j.toInt()) shl 4).toUByte()
                    val xh1 = ((qh and (1u shl (j + 16).toInt())) shr (j + 12).toInt()).toUByte()
                    val x0 = ((xs1.qs[j] and 0x0F).toInt() or xh0.toInt()) - 16
                    val x1 = ((xs1.qs[j] shr 4).toInt() or xh1.toInt()) - 16
                    sumi += (x0 * ys1.qs[j].toInt()) + (x1 * ys1.qs[j + BlockQ50.BLCK_SIZE / 2].toInt())
                }
                sumf += sumi.toFloat() * toF32(xs1.d) * toF32(ys1.d)
            }
            return sumf
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ50>) {
            // quantizeRowQ50
            check(ys.size * BlockQ50.BLCK_SIZE == xs.size)
            for ((i, ys1) in ys.enumerate()) {
                val xs1 = ((xs).slice(i * BlockQ50.BLCK_SIZE until (i + 1) * BlockQ50.BLCK_SIZE))
                var amax = 0.0f
                var max = 0.0f
                for (x in xs1.iter()) {
                    if (amax < x.abs()) {
                        amax = x.abs()
                        max = x
                    }
                }
                val d = max / -16.0
                val id =
                    if (d != 0.0f) {
                        1.0 / d
                    } else {
                        0.0
                    }
                ys1.d = fromF32(d)
                var qh = 0u
                for (j in 0 until BlockQ50.BLCK_SIZE / 2) {
                    val x0 = xs1[j] * id
                    val x1 = xs1[j + BlockQ50.BLCK_SIZE / 2] * id
                    val xi0 = ((x0 + 16.5).toByte()).min(31).toUByte()
                    val xi1 = ((x1 + 16.5).toByte()).min(31).toUByte()
                    ys1.qs[j] = (xi0 and 0x0Fu.toUByte()) or (((xi1 and 0x0Fu.toUByte()).toUInt() shl 4).toUByte())
                    qh = qh or (((xi0.toUInt() and 0x10u) shr 4) shl j.toInt())
                    qh = qh or (((xi1.toUInt() and 0x10u) shr 4) shl (j + BlockQ50.BLCK_SIZE / 2).toInt())
                }
                LittleEndian.writeU32(ys1.qh, qh)
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
        public fun toFloat(xs: List<BlockQ50>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK5_0)) { "dequantize_row_q5_0: $k is not divisible by $qk50" }
            val nb = k / QK5_0
            for (i in 0 until nb) {
                val d = xs[i].d.toF32()
                val qh: UInt = LittleEndian.readU32(xs[i].qh)
                for (j in 0 until (QK5_0 / 2)) {
                    val xh0 = (((qh shr j.toInt()) shl 4) and 0x10u).toUByte()
                    val xh1 = ((qh shr (j + 12).toInt()) and 0x10u).toUByte()
                    val x0 = ((xs[i].qs[j] and 0x0Fu.toUByte()) or xh0).toInt() - 16
                    val x1 = (((xs[i].qs[j].toUInt() shr 4).toUByte()) or xh1).toInt() - 16
                    ys[i * QK5_0 + j] = (x0.toFloat()) * d
                    ys[i * QK5_0 + j + QK5_0 / 2] = (x1.toFloat()) * d
                }
            }
        }
    }
}

public class BlockQ51(
    internal var d: F16 = F16(),
    internal var m: F16 = F16(),
    internal var qh: UByteArray = UByteArray(4),
    internal var qs: UByteArray = UByteArray(QK5_1 / 2),
) : GgmlType<BlockQ51> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q5_1

        public val BLCK_SIZE: Int = QK5_1

        public fun vecDot(n: ULong, xs: List<BlockQ51>, ys: List<BlockQ81>): Float = BlockQ51.vecDotUnopt(n, xs, ys)

        public fun vecDotUnopt(n: Int, xs: List<BlockQ51>, ys: List<BlockQ81>): Float {
            val qk = BlockQ51.BLCK_SIZE
            check(n.isMultipleOf(qk)) { "vec_dot_q5_1_q8_1: $n is not divisible by $qk" }
            check((n / qk).isMultipleOf(2)) { "vec_dot_q5_1_q8_1: $n, nb is not divisible by 2" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                val qh = LittleEndian.readU32(xs1.qh)
                var sumi = 0
                for (j in 0 until BlockQ51.BLCK_SIZE / 2) {
                    val xh0 = ((qh shr j.toInt()) shl 4) and 0x10
                    val xh1 = (qh shr (j + 12).toInt()) and 0x10
                    val x0 = (xs1.qs[j].toInt() and 0xF) or xh0.toInt()
                    val x1 = (xs1.qs[j].toInt() shr 4) or xh1.toInt()
                    sumi += (x0 * ys1.qs[j].toInt()) + (x1 * ys1.qs[j + BlockQ51.BLCK_SIZE / 2].toInt())
                }
                sumf += sumi.toFloat() * toF32(xs1.d) * toF32(ys1.d) + toF32(xs1.m) * toF32(ys1.s)
            }
            return sumf
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ51>) {
            // quantizeRowQ51
            val qk = BlockQ51.BLCK_SIZE
            check(ys.size * qk == xs.size)
            for ((i, ys1) in ys.enumerate()) {
                val xs1 = ((xs).slice(i * qk until (i + 1) * qk))
                var min = Float.INFINITY
                var max = Float.NEG_INFINITY
                for (x in xs1.iter()) {
                    min = Float.min(x, min)
                    max = Float.max(x, max)
                }
                val d = (max - min) / ((1 shl 5) - 1).toFloat()
                val id =
                    if (d != 0.0f) {
                        1.0f / d
                    } else {
                        0.0
                    }
                ys1.d = fromF32(d)
                ys1.m = fromF32(min)
                var qh = 0u
                for ((j, q) in ys1.qs
                    .iterMut()
                    .take(qk / 2)
                    .enumerate()) {
                    val x0 = (xs1[j] - min) * id
                    val x1 = (xs1[qk / 2 + j] - min) * id
                    val xi0 = (x0 + 0.5).toUByte()
                    val xi1 = (x1 + 0.5).toUByte()
                    q = (xi0 and 0x0Fu.toUByte()) or (((xi1 and 0x0Fu.toUByte()).toUInt() shl 4).toUByte())
                    // get the 5-th bit and store it in qh at the right position
                    qh = qh or (((xi0.toUInt() and 0x10u) shr 4) shl j.toInt())
                    qh = qh or (((xi1.toUInt() and 0x10u) shr 4) shl (j + qk / 2).toInt())
                }
                LittleEndian.writeU32(ys1.qh, qh)
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
        public fun toFloat(xs: List<BlockQ51>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK5_1)) { "dequantize_row_q5_1: $k is not divisible by $qk51" }
            val nb = k / QK5_1
            for (i in 0 until nb) {
                val d = xs[i].d.toF32()
                val m = xs[i].m.toF32()
                val qh: UInt = LittleEndian.readU32(xs[i].qh)
                for (j in 0 until (QK5_1 / 2)) {
                    val xh0 = (((qh shr j.toInt()) shl 4) and 0x10u).toUByte()
                    val xh1 = ((qh shr (j + 12).toInt()) and 0x10u).toUByte()
                    val x0 = (xs[i].qs[j] and 0x0Fu.toUByte()) or xh0
                    val x1 = ((xs[i].qs[j].toUInt() shr 4).toUByte()) or xh1
                    ys[i * QK5_1 + j] = (x0.toFloat()) * d + m
                    ys[i * QK5_1 + j + QK5_1 / 2] = (x1.toFloat()) * d + m
                }
            }
        }
    }
}

public class BlockQ80(
    internal var d: F16 = F16(),
    internal var qs: ByteArray = ByteArray(QK8_0),
) : GgmlType<BlockQ80> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q8_0

        public val BLCK_SIZE: Int = QK8_0

        // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
        public fun toFloat(xs: List<BlockQ80>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK8_0)) { "dequantize_row_q8_0: $k is not divisible by $qk80" }
            val nb = k / QK8_0
            for (i in 0 until nb) {
                val d = xs[i].d.toF32()
                for (j in 0 until QK8_0) {
                    ys[i * QK8_0 + j] = xs[i].qs[j].toFloat() * d
                }
            }
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ80>) {
            run {
                quantizeQ80(xs, ys)
            }
        }

        public fun vecDot(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ80Q80(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ80Q80(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ80Q80(n, xs, ys)
            return BlockQ80.vecDotUnopt(n, xs, ys)
        }

        public fun vecDot4(n: Int, xs0: List<BlockQ80>, xs1: List<BlockQ80>, xs2: List<BlockQ80>, xs3: List<BlockQ80>, ys: List<BlockQ80>): List<Any?> {
            return vecDot4Q80Q80(n, xs0, xs1, xs2, xs3, ys)
            val (d0, d1) = BlockQ80.vecDot2(n, xs0, xs1, ys)
            val (d2, d3) = BlockQ80.vecDot2(n, xs2, xs3, ys)
            return listOf(d0, d1, d2, d3)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
            check(n.isMultipleOf(QK8_0)) { "vec_dot_q8_0_q8_0: $n is not divisible by $qk80" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                val sumI = (xs1.qs.iter().zip(ys1.qs.iter())).map({ (x, y) -> x.toInt() * y.toInt() }).sum<Int>()
                sumf += sumI.toFloat() * toF32(xs1.d) * toF32(ys1.d)
            }
            return sumf
        }
    }
}

public class BlockQ81(
    internal var d: F16 = F16(),
    internal var s: F16 = F16(),
    internal var qs: ByteArray = ByteArray(QK8_1),
) : GgmlType<BlockQ81> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q8_1

        public val BLCK_SIZE: Int = QK8_1

        public fun vecDot(n: ULong, xs: List<BlockQ81>, ys: List<BlockQ81>): Float = BlockQ81.vecDotUnopt(n, xs, ys)

        public fun vecDotUnopt(n: Int, xs: List<BlockQ81>, ys: List<BlockQ81>): Float {
            check(n.isMultipleOf(QK8_1)) { "vec_dot_q8_1_q8_1: $n is not divisible by $qk81" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                val sumI = (xs1.qs.iter().zip(ys1.qs.iter())).map({ (x, y) -> x.toInt() * y.toInt() }).sum<Int>()
                sumf += sumI.toFloat() * toF32(xs1.d) * toF32(ys1.d)
            }
            return sumf
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ81>) {
            // quantizeRowQ81
            check(ys.size * BlockQ81.BLCK_SIZE == xs.size)
            for ((i, ys1) in ys.enumerate()) {
                var amax = 0.0f
                val xs1 = ((xs).slice(i * BlockQ81.BLCK_SIZE until (i + 1) * BlockQ81.BLCK_SIZE))
                for (x in xs1.iter()) {
                    amax = amax.max(x.abs())
                }
                val d = amax / ((1 shl 7) - 1).toFloat()
                val id =
                    if (d != 0.0f) {
                        1.0f / d
                    } else {
                        0.0
                    }
                ys1.d = fromF32(d)
                var sum = 0
                for (j in 0 until BlockQ81.BLCK_SIZE / 2) {
                    val v0 = xs1[j] * id
                    val v1 = xs1[j + BlockQ81.BLCK_SIZE / 2] * id
                    ys1.qs[j] = Float.round(v0).toByte()
                    ys1.qs[j + BlockQ81.BLCK_SIZE / 2] = Float.round(v1).toByte()
                    sum += ys1.qs[j].toInt() + ys1.qs[j + BlockQ81.BLCK_SIZE / 2].toInt()
                }
                ys1.s = fromF32(sum.toFloat()) * ys1.d
            }
        }

        public fun toFloat(xs: List<BlockQ81>, ys: List<Float>) {
            check(xs.size * BlockQ81.BLCK_SIZE == ys.size)
            val (yChunks, _) = ys.asChunksMut()
            for ((block, ys1) in (xs.zip(yChunks))) {
                val d = block.d.toF32()
                for ((dst, src) in (ys1.iterMut().zip(block.qs.iter()))) {
                    dst = src.toFloat() * d
                }
            }
        }
    }
}

public class BlockQ2K(
    internal var scales: UByteArray = UByteArray(QK_K / 16),
    internal var qs: UByteArray = UByteArray(QK_K / 4),
    internal var d: F16 = F16(),
    internal var dmin: F16 = F16(),
) : GgmlType<BlockQ2K> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q2K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ2kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ2kQ8k(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ2kQ8k(n, xs, ys)
            return BlockQ2K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
            check(n.isMultipleOf(QK_K)) { "vec_dot_q2k_q8k: $n is not divisible by $qkK" }
            var sumf = 0.0
            for ((x, y) in (xs.zip(ys))) {
                var q2: List<Any?> = x.qs
                var q8: List<Any?> = y.qs
                val sc = x.scales
                var summs = 0
                for ((bsum, scale) in (y.bsums.iter().zip(sc))) {
                    summs += bsum.toInt() * ((scale shr 4).toInt())
                }
                val dall = y.d * x.d.toF32()
                val dmin = y.d * x.dmin.toF32()
                var isum = 0
                var isMatch = 0
                for (unused in 0 until (QK_K / 128)) {
                    var shift = 0
                    for (unused1 in 0 until 4) {
                        val d = (sc[isMatch] and 0xF).toInt()
                        isMatch += 1
                        var isuml = 0
                        for (l in 0 until 16) {
                            isuml += q8[l].toInt() * (((q2[l] shr shift.toInt()) and 3).toInt())
                        }
                        isum += d * isuml
                        val d1 = (sc[isMatch] and 0xF).toInt()
                        isMatch += 1
                        isuml = 0
                        for (l1 in 16 until 32) {
                            isuml += q8[l1].toInt() * (((q2[l1] shr shift.toInt()) and 3).toInt())
                        }
                        isum += d1 * isuml
                        shift += 2
                        // adjust the indexing
                        q8 = ((q8).slice(32 until q8.size))
                    }
                    // adjust the indexing
                    q2 = ((q2).slice(32 until q2.size))
                }
                sumf += dall * isum.toFloat() - dmin * summs.toFloat()
            }
            return sumf
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L279
        public fun fromFloat(xs: List<Float>, ys: List<BlockQ2K>) {
            val Q4SCALE: Float = 15.0f
            for ((block, x) in groupForQuantization(xs, ys)) {
                // calculate scales and mins
                val mins: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                for ((j, xScaleSlice) in x.chunks(16).enumerate()) {
                    Pair(scales[j], mins[j]) = makeQkx1Quants(3, 5, xScaleSlice)
                }
                // get max scale and max min and ensure they are >= 0.0
                val maxScale = scales.fold(0.0, { max, value -> value.max(max) })
                val maxMin = mins.fold(0.0, { max, value -> value.max(max) })
                if (maxScale > 0.0) {
                    val iscale = Q4SCALE / maxScale
                    for ((j1, scale) in scales.enumerate().take(QK_K / 16)) {
                        block.scales[j1] = nearestInt(iscale * scale).toUByte()
                    }
                    block.d = fromF32(maxScale / Q4SCALE)
                } else {
                    for (j2 in 0 until QK_K / 16) {
                        block.scales[j2] = 0
                    }
                    block.d = fromF32(0.0)
                }
                if (maxMin > 0.0) {
                    val iscale1 = Q4SCALE / maxMin
                    for ((j3, scale1) in block.scales.iterMut().enumerate()) {
                        val l = nearestInt(iscale1 * mins[j3]).toUByte()
                        scale1 = scale1 or ((l.toUInt() shl 4).toUByte())
                    }
                    block.dmin = fromF32(maxMin / Q4SCALE)
                } else {
                    block.dmin = fromF32(0.0)
                }
                val bigL: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j4 in 0 until QK_K / 16) {
                    val d = block.d.toF32() * (block.scales[j4] and 0xF).toFloat()
                    if (d == 0.0f) {
                        continue
                    }
                    val dm = block.dmin.toF32() * (block.scales[j4] shr 4).toFloat()
                    for (ii in 0 until 16) {
                        val ll = nearestInt((x[16 * j4 + ii] + dm) / d).clamp(0, 3)
                        bigL[16 * j4 + ii] = ll.toUByte()
                    }
                }
                for (j5 in (0 until QK_K).stepBy(128)) {
                    for (ll1 in 0 until 32) {
                        block.qs[j5 / 4 + ll1] = bigL[j5 + ll1] or ((bigL[j5 + ll1 + 32].toUInt() shl 2).toUByte()) or ((bigL[j5 + ll1 + 64].toUInt() shl 4).toUByte()) or ((bigL[j5 + ll1 + 96].toUInt() shl 6).toUByte())
                    }
                }
            }
        }

        public fun fromFloatImatrix(xs: List<Float>, ys: List<BlockQ2K>, imatrixWeights: List<Float>, nPerRow: Int) {
            for ((sblkIdx, block, x) in groupForQuantization(xs, ys).intoIter().enumerate()) {
                val mins: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                val weights: FloatArray = FloatArray(16) { 0.0f }
                val sw: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                val ls: UByteArray = UByteArray(QK_K / 16) { 0u.toUByte() }
                val lm: UByteArray = UByteArray(QK_K / 16) { 0u.toUByte() }
                val sumX2 = x.iter().map({ x1 -> x1 * x1 }).sum<Float>()
                val sigma2 = sumX2 / QK_K.toFloat()
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    for ((l, wElem, xElem) in (weights.zip(xScaleSlice)).enumerate()) {
                        val imatrixRow = sblkIdx % (nPerRow / QK_K)
                        val imatrixW = imatrixWeights[imatrixRow * QK_K + 16 * j + l]
                        wElem = imatrixW * (sigma2 + xElem * xElem).sqrt()
                    }
                    val sumw = weights.sum<Float>()
                    sw[j] = sumw
                    Pair(scales[j], mins[j]) = makeQkx3Quants(3, xScaleSlice, weights, -0.9f, 0.05f, 36, false)
                }
                val dBlock = makeQpQuants(QK_K / 16, 15u, scales, ls, sw)
                val mBlock = makeQpQuants(QK_K / 16, 15u, mins, lm, sw)
                block.d = fromF32(dBlock)
                block.dmin = fromF32(mBlock)
                for (j1 in 0 until QK_K / 16) {
                    block.scales[j1] = ls[j1] or ((lm[j1].toUInt() shl 4).toUByte())
                }
                val bigL: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j2 in 0 until QK_K / 16) {
                    val d = block.d.toF32() * (block.scales[j2] and 0xF).toFloat()
                    if (d == 0.0f) {
                        continue
                    }
                    val dm = block.dmin.toF32() * (block.scales[j2] shr 4).toFloat()
                    for (ii in 0 until 16) {
                        val ll = nearestInt((x[16 * j2 + ii] + dm) / d).clamp(0, 3)
                        bigL[16 * j2 + ii] = ll.toUByte()
                    }
                }
                for (j3 in (0 until QK_K).stepBy(128)) {
                    for (ll1 in 0 until 32) {
                        block.qs[j3 / 4 + ll1] = bigL[j3 + ll1] or ((bigL[j3 + ll1 + 32].toUInt() shl 2).toUByte()) or ((bigL[j3 + ll1 + 64].toUInt() shl 4).toUByte()) or ((bigL[j3 + ll1 + 96].toUInt() shl 6).toUByte())
                    }
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L354
        public fun toFloat(xs: List<BlockQ2K>, ys: List<Float>) {
            for ((block, y) in groupForDequantization(xs, ys)) {
                val d = block.d.toF32()
                val min = block.dmin.toF32()
                var isMatch = 0
                val (yChunks, _) = y.asChunksMut()
                val (qsChunks, _) = block.qs.asChunks()
                for ((yBlock, qs) in (yChunks.iterMut().zip(qsChunks))) {
                    // Step by 32 over q.
                    var shift = 0
                    var yBlockIndex = 0
                    for (j in 0 until 4) {
                        val sc = block.scales[isMatch]
                        isMatch += 1
                        val dl = d * (sc and 0xF).toFloat()
                        val ml = min * (sc shr 4).toFloat()
                        for (q in ((qs).slice(0 until 16))) {
                            val y1 = dl * ((q shr shift.toInt()) and 3).toFloat() - ml
                            yBlock[yBlockIndex] = y1
                            yBlockIndex += 1
                        }
                        val sc1 = block.scales[isMatch]
                        isMatch += 1
                        val dl1 = d * (sc1 and 0xF).toFloat()
                        val ml1 = min * (sc1 shr 4).toFloat()
                        for (q1 in ((qs).slice(16 until qs.size))) {
                            val y2 = dl1 * ((q1 shr shift.toInt()) and 3).toFloat() - ml1
                            yBlock[yBlockIndex] = y2
                            yBlockIndex += 1
                        }
                        shift += 2
                    }
                }
            }
        }
    }
}

public class BlockQ3K(
    internal var hmask: UByteArray = UByteArray(QK_K / 8),
    internal var qs: UByteArray = UByteArray(QK_K / 4),
    internal var scales: UByteArray = UByteArray(12),
    internal var d: F16 = F16(),
) : GgmlType<BlockQ3K> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q3K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ3kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ3kQ8k(n, xs, ys)
            return BlockQ3K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
            val KMASK1: UInt = 0x03030303u
            val KMASK2: UInt = 0x0f0f0f0fu
            check(n.isMultipleOf(QK_K)) { "vec_dot_q3k_q8k: $n is not divisible by $qkK" }
            val aux8: ByteArray = ByteArray(QK_K) { 0.toByte() }
            val aux16: ShortArray = ShortArray(8) { 0.toShort() }
            val sums: FloatArray = FloatArray(8) { 0.0f }
            val aux32: IntArray = IntArray(8) { 0 }
            val auxs: UIntArray = UIntArray(4) { 0u }
            for ((x, y) in (xs.zip(ys))) {
                var q3: List<UByte> = x.qs
                val hmask: List<UByte> = x.hmask
                var q8: List<Byte> = y.qs
                aux32.fill(0)
                var a = aux8
                var m = 1
                // Like the GGML original this is written this way to enable the compiler to vectorize it.
                for (unused in 0 until QK_K / 128) {
                    (a.iterMut().take(32).zip(q3)).forEach({ (aVal, q3Val) -> aVal = (q3Val and 3).toByte() })
                    (a.iterMut().take(32).zip(hmask)).forEach({ (aVal, hmaskVal) ->
                        aVal -=
                            if (hmaskVal and m != 0) {
                                0
                            } else {
                                4
                            }
                    })
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    (a.iterMut().take(32).zip(q3)).forEach({ (aVal, q3Val) -> aVal = ((q3Val shr 2) and 3).toByte() })
                    (a.iterMut().take(32).zip(hmask)).forEach({ (aVal, hmaskVal) ->
                        aVal -=
                            if (hmaskVal and m != 0) {
                                0
                            } else {
                                4
                            }
                    })
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    (a.iterMut().take(32).zip(q3)).forEach({ (aVal, q3Val) -> aVal = ((q3Val shr 4) and 3).toByte() })
                    (a.iterMut().take(32).zip(hmask)).forEach({ (aVal, hmaskVal) ->
                        aVal -=
                            if (hmaskVal and m != 0) {
                                0
                            } else {
                                4
                            }
                    })
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    (a.iterMut().take(32).zip(q3)).forEach({ (aVal, q3Val) -> aVal = ((q3Val shr 6) and 3).toByte() })
                    (a.iterMut().take(32).zip(hmask)).forEach({ (aVal, hmaskVal) ->
                        aVal -=
                            if (hmaskVal and m != 0) {
                                0
                            } else {
                                4
                            }
                    })
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    q3 = ((q3).slice(32 until q3.size))
                }
                a = aux8
                LittleEndian.readU32Into(x.scales, ((auxs).slice(0 until 3)))
                val tmp = auxs[2]
                auxs[2] = ((auxs[0] shr 4) and KMASK2) or (((tmp shr 4) and KMASK1) shl 4)
                auxs[3] = ((auxs[1] shr 4) and KMASK2) or (((tmp shr 6) and KMASK1) shl 4)
                auxs[0] = (auxs[0] and KMASK2) or (((tmp) and KMASK1) shl 4)
                auxs[1] = (auxs[1] and KMASK2) or (((tmp shr 2) and KMASK1) shl 4)
                for (aux in auxs) {
                    for (scale in aux.toLeBytes()) {
                        val scale1 = Byte.fromBeBytes(listOf(scale))
                        for (l in 0 until 8) {
                            aux16[l] = q8[l].toShort() * a[l].toShort()
                        }
                        for (l1 in 0 until 8) {
                            aux32[l1] += (scale1.toInt() - 32) * aux16[l1].toInt()
                        }
                        q8 = ((q8).slice(8 until q8.size))
                        a = ((a).slice(8 until a.size))
                        for (l2 in 0 until 8) {
                            aux16[l2] = q8[l2].toShort() * a[l2].toShort()
                        }
                        for (l3 in 0 until 8) {
                            aux32[l3] += (scale1.toInt() - 32) * aux16[l3].toInt()
                        }
                        q8 = ((q8).slice(8 until q8.size))
                        a = ((a).slice(8 until a.size))
                    }
                }
                val d = x.d.toF32() * y.d
                for (l4 in 0 until 8) {
                    sums[l4] += d * aux32[l4].toFloat()
                }
            }
            return sums.sum()
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ3K>) {
            for ((block, x) in groupForQuantization(xs, ys)) {
                val scales: FloatArray = FloatArray(QK_K / 16) { 0.0f }
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    scales[j] = makeQ3Quants(xScaleSlice, 4, true)
                }
                // Get max scale by absolute value.
                var maxScale: Float = 0.0f
                for (scale in scales) {
                    if (scale.abs() > maxScale.abs()) {
                        maxScale = scale
                    }
                }
                block.scales.fill(0)
                if (maxScale != 0.0f) {
                    val iscale = -32.0f / maxScale
                    for ((j1, scale1) in scales.enumerate()) {
                        val lVal = nearestInt(iscale * scale1)
                        val lVal1 = lVal.clamp(-32, 31) + 32
                        if (j1 < 8) {
                            block.scales[j1] = (lVal1 and 0xF).toUByte()
                        } else {
                            block.scales[j1 - 8] = block.scales[j1 - 8] or (((lVal1 and 0xF) shl 4).toUByte())
                        }
                        val lVal2 = lVal1 shr 4
                        block.scales[j1 % 4 + 8] = block.scales[j1 % 4 + 8] or ((lVal2 shl (2 * (j1 / 4)).toInt()).toUByte())
                    }
                    block.d = fromF32(1.0f / iscale)
                } else {
                    block.d = fromF32(0.0)
                }
                val l: ByteArray = ByteArray(QK_K) { 0.toByte() }
                for (j2 in 0 until QK_K / 16) {
                    val sc =
                        if (j2 < 8) {
                            block.scales[j2] and 0xF
                        } else {
                            block.scales[j2 - 8] shr 4
                        }
                    val sc1 = (sc or (((block.scales[8 + j2 % 4] shr (2 * (j2 / 4)).toInt()) and 3) shl 4)).toByte() - 32
                    val d = block.d.toF32() * sc1.toFloat()
                    if (d != 0.0f) {
                        for (ii in 0 until 16) {
                            val lVal3 = nearestInt(x[16 * j2 + ii] / d)
                            l[16 * j2 + ii] = (lVal3.clamp(-4, 3) + 4).toByte()
                        }
                    }
                }
                block.hmask.fill(0)
                var m = 0
                var hm = 1
                for (ll in l) {
                    if (ll > 3) {
                        block.hmask[m] = block.hmask[m] or hm
                        ll -= 4
                    }
                    m += 1
                    if (m == QK_K / 8) {
                        m = 0
                        hm = hm shl 1
                    }
                }
                for (j3 in (0 until QK_K).stepBy(128)) {
                    for (lVal4 in 0 until 32) {
                        block.qs[j3 / 4 + lVal4] = ((((((l[j3 + lVal4].toInt() or (l[j3 + lVal4 + 32] shl 2).toInt()).toByte()).toInt() or (l[j3 + lVal4 + 64] shl 4).toInt()).toByte()).toInt() or (l[j3 + lVal4 + 96] shl 6).toInt()).toByte()).toUByte()
                    }
                }
            }
        }

        public fun fromFloatImatrix(xs: List<Float>, ys: List<BlockQ3K>, imatrixWeights: List<Float>, nPerRow: Int) {
            for ((sblkIdx, block, x) in groupForQuantization(xs, ys).intoIter().enumerate()) {
                val localStorageL1281C12 = AddressableArrayLocal(FloatArray(QK_K / 16) { 0.0f }, FixedArrayTypeLayout(PrimitiveLayouts.F32, (QK_K / 16).toInt()))
                var scales by localStorageL1281C12
                val localStorageL1282C12 = AddressableArrayLocal(FloatArray(16) { 0.0f }, FixedArrayTypeLayout(PrimitiveLayouts.F32, (16).toInt()))
                var weights by localStorageL1282C12
                val localStorageL1283C12 = AddressableArrayLocal(FloatArray(QK_K / 16) { 0.0f }, FixedArrayTypeLayout(PrimitiveLayouts.F32, (QK_K / 16).toInt()))
                var sw by localStorageL1283C12
                val localStorageL1284C12 = AddressableArrayLocal(ByteArray(QK_K / 16) { 0.toByte() }, FixedArrayTypeLayout(PrimitiveLayouts.I8, (QK_K / 16).toInt()))
                var ls by localStorageL1284C12
                val localStorageL1285C12 = AddressableArrayLocal(ByteArray(QK_K) { 0.toByte() }, FixedArrayTypeLayout(PrimitiveLayouts.I8, (QK_K).toInt()))
                var l by localStorageL1285C12
                val sumX2 = x.iter().map({ x1 -> x1 * x1 }).sum<Float>()
                val sigma2 = 2.0f * sumX2 / QK_K.toFloat()
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    for ((lIdx, wElem, xElem) in (weights.zip(xScaleSlice)).enumerate()) {
                        val imatrixRow = sblkIdx % (nPerRow / QK_K)
                        val imatrixW = imatrixWeights[imatrixRow * QK_K + 16 * j + lIdx]
                        wElem = imatrixW * (sigma2 + xElem * xElem).sqrt()
                    }
                    val sumw = weights.sum<Float>()
                    sw[j] = sumw
                    scales[j] = makeQxQuants(16, 4, xScaleSlice.asPtr(), MutPtr((localStorageL1285C12.elementMutPtr(0)).rawAddress + ((16 * j).toLong() * (PrimitiveLayouts.I8).strideBytes)), 1, localStorageL1282C12.elementConstPtr(0))
                }
                block.scales.fill(0)
                val dBlock = makeQxQuants(QK_K / 16, 32, localStorageL1281C12.elementConstPtr(0), localStorageL1284C12.elementMutPtr(0), 1, localStorageL1283C12.elementConstPtr(0))
                block.d = fromF32(dBlock)
                for ((j1, lVal) in ls.enumerate().take(QK_K / 16)) {
                    if (j1 < 8) {
                        block.scales[j1] = (lVal and 0xF).toUByte()
                    } else {
                        block.scales[j1 - 8] = block.scales[j1 - 8] or (((lVal and 0xF) shl 4).toUByte())
                    }
                    val lVal1 = lVal shr 4
                    block.scales[j1 % 4 + 8] = block.scales[j1 % 4 + 8] or ((lVal1 shl (2 * (j1 / 4)).toInt()).toUByte())
                }
                for (j2 in 0 until QK_K / 16) {
                    val sc =
                        if (j2 < 8) {
                            block.scales[j2] and 0xF
                        } else {
                            block.scales[j2 - 8] shr 4
                        }
                    val sc1 = (sc or (((block.scales[8 + j2 % 4] shr (2 * (j2 / 4)).toInt()) and 3) shl 4)).toByte() - 32
                    val d = block.d.toF32() * sc1.toFloat()
                    if (d != 0.0f) {
                        for (ii in 0 until 16) {
                            val lVal2 = nearestInt(x[16 * j2 + ii] / d)
                            l[16 * j2 + ii] = (lVal2.clamp(-4, 3) + 4).toByte()
                        }
                    }
                }
                block.hmask.fill(0)
                var m = 0
                var hm = 1
                for (ll in l) {
                    if (ll > 3) {
                        block.hmask[m] = block.hmask[m] or hm
                        ll -= 4
                    }
                    m += 1
                    if (m == QK_K / 8) {
                        m = 0
                        hm = hm shl 1
                    }
                }
                for (j3 in (0 until QK_K).stepBy(128)) {
                    for (lVal3 in 0 until 32) {
                        block.qs[j3 / 4 + lVal3] = ((((((l[j3 + lVal3].toInt() or (l[j3 + lVal3 + 32] shl 2).toInt()).toByte()).toInt() or (l[j3 + lVal3 + 64] shl 4).toInt()).toByte()).toInt() or (l[j3 + lVal3 + 96] shl 6).toInt()).toByte()).toUByte()
                    }
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L533
        public fun toFloat(xs: List<BlockQ3K>, ys: List<Float>) {
            val KMASK1: UInt = 0x03030303u
            val KMASK2: UInt = 0x0f0f0f0fu
            for ((block, y) in groupForDequantization(xs, ys)) {
                // Reconstruct the scales
                val localStorageL1384C12 = AddressableArrayLocal(IntArray(4) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I32, (4).toInt()))
                var aux by localStorageL1384C12
                LittleEndian.readU32Into(block.scales, ((aux).slice(0 until 3)))
                val tmp = aux[2]
                aux[2] = ((aux[0] shr 4) and KMASK2) or (((tmp shr 4) and KMASK1) shl 4)
                aux[3] = ((aux[1] shr 4) and KMASK2) or (((tmp shr 6) and KMASK1) shl 4)
                aux[0] = (aux[0] and KMASK2) or (((tmp) and KMASK1) shl 4)
                aux[1] = (aux[1] and KMASK2) or (((tmp shr 2) and KMASK1) shl 4)
                // Transfer the scales into an i8 array
                val scales: List<Byte> = fromRawPartsMut(MutPtr<Byte>(localStorageL1384C12.elementMutPtr(0).rawAddress, localStorageL1384C12.elementMutPtr(0).rawMeta), 16)
                val dAll = block.d.toF32()
                var m = 1
                var isMatch = 0
                // Dequantize both 128 long blocks
                // 32 qs values per 128 long block
                // Each 16 elements get a scale
                val (yChunks, _) = y.asChunksMut()
                val (qsChunks, _) = block.qs.asChunks()
                for ((y1, qs) in (yChunks.iterMut().zip(qsChunks))) {
                    var shift = 0
                    val (scopedYChunks, _) = y1.asChunksMut()
                    for (shiftScopedY in scopedYChunks) {
                        val (shiftScopedChunks, _) = shiftScopedY.asChunksMut()
                        for ((scaleIndex, scaleScopedY) in shiftScopedChunks.iterMut().enumerate()) {
                            val dl = dAll * (scales[isMatch].toFloat() - 32.0f)
                            for ((i, innerY) in scaleScopedY.iterMut().enumerate()) {
                                val newY =
                                    dl *
                                        (
                                            ((qs[i + 16 * scaleIndex] shr shift.toInt()) and 3).toByte() -
                                                if ((block.hmask[i + 16 * scaleIndex] and m) == 0) {
                                                    4
                                                } else {
                                                    0
                                                }
                                        ).toFloat()
                                innerY = newY
                            }
                            // 16 block finished => advance scale index
                            isMatch += 1
                        }
                        // 32 block finished => increase shift and m
                        shift += 2
                        m = m shl 1
                    }
                }
            }
        }
    }
}

// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/kQuants.h#L82
public class BlockQ4K(
    internal var d: F16 = F16(),
    internal var dmin: F16 = F16(),
    internal var scales: UByteArray = UByteArray(K_SCALE_SIZE),
    internal var qs: UByteArray = UByteArray(QK_K / 2),
) : GgmlType<BlockQ4K> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q4K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ4kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ4kQ8k(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ4kQ8k(n, xs, ys)
            return BlockQ4K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDot4(n: Int, xs0: List<BlockQ4K>, xs1: List<BlockQ4K>, xs2: List<BlockQ4K>, xs3: List<BlockQ4K>, ys: List<BlockQ8K>): List<Any?> {
            return vecDot4Q4kQ8k(n, xs0, xs1, xs2, xs3, ys)
            val (d0, d1) = BlockQ4K.vecDot2(n, xs0, xs1, ys)
            val (d2, d3) = BlockQ4K.vecDot2(n, xs2, xs3, ys)
            return listOf(d0, d1, d2, d3)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
            val KMASK1: UInt = 0x3f3f3f3fu
            val KMASK2: UInt = 0x0f0f0f0fu
            val KMASK3: UInt = 0x03030303u
            check(n.isMultipleOf(QK_K)) { "vec_dot_q4k_q8k: $n is not divisible by $qkK" }
            val utmp: UIntArray = UIntArray(4) { 0u }
            val scales: UByteArray = UByteArray(8) { 0u.toUByte() }
            val mins: UByteArray = UByteArray(8) { 0u.toUByte() }
            val aux8: ByteArray = ByteArray(QK_K) { 0.toByte() }
            val aux16: ShortArray = ShortArray(8) { 0.toShort() }
            val sums: FloatArray = FloatArray(8) { 0.0f }
            val aux32: IntArray = IntArray(8) { 0 }
            var sumf = 0.0
            for ((y, x) in (ys.zip(xs))) {
                val q4 = x.qs
                val q8 = y.qs
                aux32.fill(0)
                var a = aux8
                var q41 = q4
                for (unused in 0 until QK_K / 64) {
                    for (l in 0 until 32) {
                        a[l] = (q41[l] and 0xF).toByte()
                    }
                    a = ((a).slice(32 until a.size))
                    for (l1 in 0 until 32) {
                        a[l1] = (q41[l1] shr 4).toByte()
                    }
                    a = ((a).slice(32 until a.size))
                    q41 = ((q41).slice(32 until q41.size))
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
                var sumi = 0
                for (j in 0 until QK_K / 16) {
                    sumi += y.bsums[j].toInt() * mins[j / 2].toInt()
                }
                var a1 = aux8
                var q81 = q8
                for (scale in scales) {
                    val scale1 = scale.toInt()
                    for (unused1 in 0 until 4) {
                        for (l2 in 0 until 8) {
                            aux16[l2] = q81[l2].toShort() * a1[l2].toShort()
                        }
                        for (l3 in 0 until 8) {
                            aux32[l3] += scale1 * aux16[l3].toInt()
                        }
                        q81 = ((q81).slice(8 until q81.size))
                        a1 = ((a1).slice(8 until a1.size))
                    }
                }
                val d = x.d.toF32() * y.d
                for (l4 in 0 until 8) {
                    sums[l4] += d * aux32[l4].toFloat()
                }
                val dmin = x.dmin.toF32() * y.d
                sumf -= dmin * sumi.toFloat()
            }
            return sumf + sums.sum<Float>()
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ4K>) {
            for ((block, x) in groupForQuantization(xs, ys)) {
                val mins: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    Pair(scales[j], mins[j]) = makeQkx1Quants(15, 5, xScaleSlice)
                }
                // get max scale and max min and ensure they are >= 0.0
                val maxScale = scales.fold(0.0, { max, value -> value.max(max) })
                val maxMin = mins.fold(0.0, { max, value -> value.max(max) })
                val invScale =
                    if (maxScale > 0.0) {
                        63.0 / maxScale
                    } else {
                        0.0
                    }
                val invMin =
                    if (maxMin > 0.0) {
                        63.0 / maxMin
                    } else {
                        0.0
                    }
                for (j1 in 0 until QK_K / 32) {
                    val ls = nearestInt(invScale * scales[j1]).min(63).toUByte()
                    val lm = nearestInt(invMin * mins[j1]).min(63).toUByte()
                    if (j1 < 4) {
                        block.scales[j1] = ls
                        block.scales[j1 + 4] = lm
                    } else {
                        block.scales[j1 + 4] = (ls and 0xFu.toUByte()) or (((lm and 0xFu.toUByte()).toUInt() shl 4).toUByte())
                        block.scales[j1 - 4] = block.scales[j1 - 4] or ((((ls.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                        block.scales[j1] = block.scales[j1] or ((((lm.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                    }
                }
                block.d = fromF32(maxScale / 63.0)
                block.dmin = fromF32(maxMin / 63.0)
                val l: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j2 in 0 until QK_K / 32) {
                    val (sc, m) = getScaleMinK4(j2, block.scales)
                    val d = block.d.toF32() * sc.toFloat()
                    if (d != 0.0f) {
                        val dm = block.dmin.toF32() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j2 + ii] + dm) / d)
                            l[32 * j2 + ii] = lVal.clamp(0, 15).toUByte()
                        }
                    }
                }
                val q = block.qs
                for (j3 in (0 until QK_K).stepBy(64)) {
                    for (lVal1 in 0 until 32) {
                        val offsetIndex = (j3 / 64) * 32 + lVal1
                        q[offsetIndex] = l[j3 + lVal1] or ((l[j3 + lVal1 + 32].toUInt() shl 4).toUByte())
                    }
                }
            }
        }

        public fun fromFloatImatrix(xs: List<Float>, ys: List<BlockQ4K>, imatrixWeights: List<Float>, nPerRow: Int) {
            for ((sblkIdx, block, x) in groupForQuantization(xs, ys).intoIter().enumerate()) {
                val mins: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val weights: FloatArray = FloatArray(32) { 0.0f }
                val sw: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val ls: UByteArray = UByteArray(QK_K / 32) { 0u.toUByte() }
                val lm: UByteArray = UByteArray(QK_K / 32) { 0u.toUByte() }
                val sumX2 = x.iter().map({ x1 -> x1 * x1 }).sum<Float>()
                val sigma2 = 2.0f * sumX2 / QK_K.toFloat()
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    for ((l, wElem, xElem) in (weights.zip(xScaleSlice)).enumerate()) {
                        val imatrixRow = sblkIdx % (nPerRow / QK_K)
                        val imatrixW = imatrixWeights[imatrixRow * QK_K + 32 * j + l]
                        wElem = imatrixW * (sigma2 + xElem * xElem).sqrt()
                    }
                    val sumw = weights.sum<Float>()
                    sw[j] = sumw
                    Pair(scales[j], mins[j]) = makeQkx3Quants(15, xScaleSlice, weights, -0.9f, 0.05f, 36, false)
                }
                val dBlock = makeQpQuants(QK_K / 32, 63u, scales, ls, sw)
                val mBlock = makeQpQuants(QK_K / 32, 63u, mins, lm, sw)
                for (j1 in 0 until QK_K / 32) {
                    val lsVal = ls[j1]
                    val lmVal = lm[j1]
                    if (j1 < 4) {
                        block.scales[j1] = lsVal
                        block.scales[j1 + 4] = lmVal
                    } else {
                        block.scales[j1 + 4] = (lsVal and 0xFu.toUByte()) or (((lmVal and 0xFu.toUByte()).toUInt() shl 4).toUByte())
                        block.scales[j1 - 4] = block.scales[j1 - 4] or ((((lsVal.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                        block.scales[j1] = block.scales[j1] or ((((lmVal.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                    }
                }
                block.d = fromF32(dBlock)
                block.dmin = fromF32(mBlock)
                val l1: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j2 in 0 until QK_K / 32) {
                    val (sc, m) = getScaleMinK4(j2, block.scales)
                    val d = block.d.toF32() * sc.toFloat()
                    if (d != 0.0f) {
                        val dm = block.dmin.toF32() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j2 + ii] + dm) / d)
                            l1[32 * j2 + ii] = lVal.clamp(0, 15).toUByte()
                        }
                    }
                }
                val q = block.qs
                for (j3 in (0 until QK_K).stepBy(64)) {
                    for (lVal1 in 0 until 32) {
                        val offsetIndex = (j3 / 64) * 32 + lVal1
                        q[offsetIndex] = l1[j3 + lVal1] or ((l1[j3 + lVal1 + 32].toUInt() shl 4).toUByte())
                    }
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L735
        public fun toFloat(xs: List<BlockQ4K>, ys: List<Float>) {
            for ((block, y) in groupForDequantization(xs, ys)) {
                val d = block.d.toF32()
                val min = block.dmin.toF32()
                val q = block.qs
                var isMatch = 0
                var ysIndex = 0
                for (j in (0 until QK_K).stepBy(64)) {
                    val q1 = ((q).slice(j / 2 until j / 2 + 32))
                    val (sc, m) = getScaleMinK4(isMatch, block.scales)
                    val d1 = d * sc.toFloat()
                    val m1 = min * m.toFloat()
                    val (sc1, m1) = getScaleMinK4(isMatch + 1, block.scales)
                    val d2 = d * sc1.toFloat()
                    val m2 = min * m1
                    for (q2 in q1) {
                        y[ysIndex] = d1 * (q2 and 0xF).toFloat() - m1
                        ysIndex += 1
                    }
                    for (q3 in q1) {
                        y[ysIndex] = d2 * (q3 shr 4).toFloat() - m2
                        ysIndex += 1
                    }
                    isMatch += 2
                }
            }
        }
    }
}

public class BlockQ5K(
    internal var d: F16 = F16(),
    internal var dmin: F16 = F16(),
    internal var scales: UByteArray = UByteArray(K_SCALE_SIZE),
    internal var qh: UByteArray = UByteArray(QK_K / 8),
    internal var qs: UByteArray = UByteArray(QK_K / 2),
) : GgmlType<BlockQ5K> {
    public companion object {
        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L928
        public val DTYPE: GgmlDType = GgmlDType.Q5K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ5kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ5kQ8k(n, xs, ys)
            return BlockQ5K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
            val KMASK1: UInt = 0x3f3f3f3fu
            val KMASK2: UInt = 0x0f0f0f0fu
            val KMASK3: UInt = 0x03030303u
            check(n.isMultipleOf(QK_K)) { "vec_dot_q5k_q8k: $n is not divisible by $qkK" }
            val utmp: UIntArray = UIntArray(4) { 0u }
            val scales: UByteArray = UByteArray(8) { 0u.toUByte() }
            val mins: UByteArray = UByteArray(8) { 0u.toUByte() }
            val aux8: ByteArray = ByteArray(QK_K) { 0.toByte() }
            val aux16: ShortArray = ShortArray(8) { 0.toShort() }
            val sums: FloatArray = FloatArray(8) { 0.0f }
            val aux32: IntArray = IntArray(8) { 0 }
            var sumf = 0.0
            for ((y, x) in (ys.zip(xs))) {
                val q5 = x.qs
                val hm = x.qh
                val q8 = y.qs
                aux32.fill(0)
                var a = aux8
                var q51 = q5
                var m = 1u.toUByte()
                for (unused in 0 until QK_K / 64) {
                    for (l in 0 until 32) {
                        a[l] = (q51[l] and 0xF).toByte()
                        a[l] +=
                            if (hm[l] and m != 0u.toUByte()) {
                                16
                            } else {
                                0
                            }
                    }
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    for (l1 in 0 until 32) {
                        a[l1] = (q51[l1] shr 4).toByte()
                        a[l1] +=
                            if (hm[l1] and m != 0u.toUByte()) {
                                16
                            } else {
                                0
                            }
                    }
                    a = ((a).slice(32 until a.size))
                    m = m shl 1
                    q51 = ((q51).slice(32 until q51.size))
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
                var sumi = 0
                for (j in 0 until QK_K / 16) {
                    sumi += y.bsums[j].toInt() * mins[j / 2].toInt()
                }
                var a1 = aux8
                var q81 = q8
                for (scale in scales) {
                    val scale1 = scale.toInt()
                    for (unused1 in 0 until 4) {
                        for (l2 in 0 until 8) {
                            aux16[l2] = q81[l2].toShort() * a1[l2].toShort()
                        }
                        for (l3 in 0 until 8) {
                            aux32[l3] += scale1 * aux16[l3].toInt()
                        }
                        q81 = ((q81).slice(8 until q81.size))
                        a1 = ((a1).slice(8 until a1.size))
                    }
                }
                val d = x.d.toF32() * y.d
                for (l4 in 0 until 8) {
                    sums[l4] += d * aux32[l4].toFloat()
                }
                val dmin = x.dmin.toF32() * y.d
                sumf -= dmin * sumi.toFloat()
            }
            return sumf + sums.sum<Float>()
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L793
        public fun fromFloat(xs: List<Float>, ys: List<BlockQ5K>) {
            for ((block, x) in groupForQuantization(xs, ys)) {
                val mins: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    Pair(scales[j], mins[j]) = makeQkx1Quants(31, 5, xScaleSlice)
                }
                // get max scale and max min and ensure they are >= 0.0
                val maxScale = scales.fold(0.0, { max, value -> value.max(max) })
                val maxMin = mins.fold(0.0, { max, value -> value.max(max) })
                val invScale =
                    if (maxScale > 0.0) {
                        63.0 / maxScale
                    } else {
                        0.0
                    }
                val invMin =
                    if (maxMin > 0.0) {
                        63.0 / maxMin
                    } else {
                        0.0
                    }
                for (j1 in 0 until QK_K / 32) {
                    val ls = nearestInt(invScale * scales[j1]).min(63).toUByte()
                    val lm = nearestInt(invMin * mins[j1]).min(63).toUByte()
                    if (j1 < 4) {
                        block.scales[j1] = ls
                        block.scales[j1 + 4] = lm
                    } else {
                        block.scales[j1 + 4] = (ls and 0xFu.toUByte()) or (((lm and 0xFu.toUByte()).toUInt() shl 4).toUByte())
                        block.scales[j1 - 4] = block.scales[j1 - 4] or ((((ls.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                        block.scales[j1] = block.scales[j1] or ((((lm.toUInt() shr 4).toUByte()).toUInt() shl 6).toUByte())
                    }
                }
                block.d = fromF32(maxScale / 63.0)
                block.dmin = fromF32(maxMin / 63.0)
                val l: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j2 in 0 until QK_K / 32) {
                    val (sc, m) = getScaleMinK4(j2, block.scales)
                    val d = block.d.toF32() * sc.toFloat()
                    if (d == 0.0f) {
                        continue
                    }
                    val dm = block.dmin.toF32() * m.toFloat()
                    for (ii in 0 until 32) {
                        val ll = nearestInt((x[32 * j2 + ii] + dm) / d)
                        l[32 * j2 + ii] = ll.clamp(0, 31).toUByte()
                    }
                }
                val qh = block.qh
                val ql = block.qs
                qh.fill(0)
                var m1 = 1
                var m2 = 2
                for (n in (0 until QK_K).stepBy(64)) {
                    val offset = (n / 64) * 32
                    for (j3 in 0 until 32) {
                        var l1 = l[n + j3]
                        if (l1 > 15u.toUByte()) {
                            l1 -= 16u
                            qh[j3] = qh[j3] or m1
                        }
                        var l2 = l[n + j3 + 32]
                        if (l2 > 15u.toUByte()) {
                            l2 -= 16u
                            qh[j3] = qh[j3] or m2
                        }
                        ql[offset + j3] = l1 or ((l2.toUInt() shl 4).toUByte())
                    }
                    m1 = m1 shl 2
                    m2 = m2 shl 2
                }
            }
        }

        public fun fromFloatImatrix(xs: List<Float>, ys: List<BlockQ5K>, imatrixWeights: List<Float>, nPerRow: Int) {
            for ((sblkIdx, block, x) in groupForQuantization(xs, ys).intoIter().enumerate()) {
                val mins: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val scales: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val weights: FloatArray = FloatArray(32) { 0.0f }
                val sw: FloatArray = FloatArray(QK_K / 32) { 0.0f }
                val ls: UByteArray = UByteArray(QK_K / 32) { 0u.toUByte() }
                val lm: UByteArray = UByteArray(QK_K / 32) { 0u.toUByte() }
                val sumX2 = x.iter().map({ x1 -> x1 * x1 }).sum<Float>()
                val sigma2 = 2.0f * sumX2 / QK_K.toFloat()
                val (xChunks, _) = x.asChunks()
                for ((j, xScaleSlice) in xChunks.iter().enumerate()) {
                    for ((l, wElem, xElem) in (weights.zip(xScaleSlice)).enumerate()) {
                        val imatrixRow = sblkIdx % (nPerRow / QK_K)
                        val imatrixW = imatrixWeights[imatrixRow * QK_K + 32 * j + l]
                        wElem = imatrixW * (sigma2 + xElem * xElem).sqrt()
                    }
                    val sumw = weights.sum<Float>()
                    sw[j] = sumw
                    Pair(scales[j], mins[j]) = makeQkx3Quants(31, xScaleSlice, weights, -0.9f, 0.05f, 36, false)
                }
                val dBlock = makeQpQuants(QK_K / 32, 63u, scales, ls, sw)
                val mBlock = makeQpQuants(QK_K / 32, 63u, mins, lm, sw)
                for (j1 in 0 until QK_K / 32) {
                    val lsVal = ls[j1].min(63)
                    val lmVal = lm[j1].min(63)
                    if (j1 < 4) {
                        block.scales[j1] = lsVal
                        block.scales[j1 + 4] = lmVal
                    } else {
                        block.scales[j1 + 4] = (lsVal and 0xF) or ((lmVal and 0xF) shl 4)
                        block.scales[j1 - 4] = block.scales[j1 - 4] or ((lsVal shr 4) shl 6)
                        block.scales[j1] = block.scales[j1] or ((lmVal shr 4) shl 6)
                    }
                }
                block.d = fromF32(dBlock)
                block.dmin = fromF32(mBlock)
                var l1: UByteArray = UByteArray(QK_K) { 0u.toUByte() }
                for (j2 in 0 until QK_K / 32) {
                    val (sc, m) = getScaleMinK4(j2, block.scales)
                    val d = block.d.toF32() * sc.toFloat()
                    if (d != 0.0f) {
                        val dm = block.dmin.toF32() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j2 + ii] + dm) / d)
                            l1[32 * j2 + ii] = lVal.clamp(0, 31).toUByte()
                        }
                    }
                }
                val qh = block.qh
                val ql = block.qs
                qh.fill(0)
                var m1 = 1
                var m2 = 2
                for (n in (0 until QK_K).stepBy(64)) {
                    val offset = (n / 64) * 32
                    for (j3 in 0 until 32) {
                        var l1 = l1[n + j3]
                        if (l1 > 15u.toUByte()) {
                            l1 -= 16u
                            qh[j3] = qh[j3] or m1
                        }
                        var l2 = l1[n + j3 + 32]
                        if (l2 > 15) {
                            l2 -= 16
                            qh[j3] = qh[j3] or m2
                        }
                        ql[offset + j3] = l1 or ((l2.toUInt() shl 4).toUByte())
                    }
                    m1 = m1 shl 2
                    m2 = m2 shl 2
                }
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L928
        public fun toFloat(xs: List<BlockQ5K>, ys: List<Float>) {
            for ((block, y) in groupForDequantization(xs, ys)) {
                val d = block.d.toF32()
                val min = block.dmin.toF32()
                val ql = block.qs
                val qh = block.qh
                var isMatch = 0
                var u1 = 1
                var u2 = 2
                var ysIndex = 0
                for (j in (0 until QK_K).stepBy(64)) {
                    val ql1 = ((ql).slice(j / 2 until j / 2 + 32))
                    val (sc, m) = getScaleMinK4(isMatch, block.scales)
                    val d1 = d * sc.toFloat()
                    val m1 = min * m.toFloat()
                    val (sc1, m1) = getScaleMinK4(isMatch + 1, block.scales)
                    val d2 = d * sc1.toFloat()
                    val m2 = min * m1
                    for ((ql2, qh1) in (ql1.iter().zip(qh))) {
                        val toAdd =
                            if (qh1 and u1 != 0) {
                                16.0f
                            } else {
                                0.0f
                            }
                        y[ysIndex] = d1 * ((ql2 and 0xF).toFloat() + toAdd) - m1
                        ysIndex += 1
                    }
                    for ((ql3, qh2) in (ql1.iter().zip(qh))) {
                        val toAdd1 =
                            if (qh2 and u2 != 0) {
                                16.0f
                            } else {
                                0.0f
                            }
                        y[ysIndex] = d2 * ((ql3 shr 4).toFloat() + toAdd1) - m2
                        ysIndex += 1
                    }
                    isMatch += 2
                    u1 = u1 shl 2
                    u2 = u2 shl 2
                }
            }
        }
    }
}

public class BlockQ6K(
    internal var ql: UByteArray = UByteArray(QK_K / 2),
    internal var qh: UByteArray = UByteArray(QK_K / 4),
    internal var scales: ByteArray = ByteArray(QK_K / 16),
    internal var d: F16 = F16(),
) : GgmlType<BlockQ6K> {
    public companion object {
        public val DTYPE: GgmlDType = GgmlDType.Q6K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ6kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ6kQ8k(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ6kQ8k(n, xs, ys)
            return BlockQ6K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDot4(n: Int, xs0: List<BlockQ6K>, xs1: List<BlockQ6K>, xs2: List<BlockQ6K>, xs3: List<BlockQ6K>, ys: List<BlockQ8K>): List<Any?> {
            if (hasAvx2Fma()) {
                // Check the runtime feature set once for the whole output quad.
                return listOf(vecDotQ6kQ8k(n, xs0, ys), vecDotQ6kQ8k(n, xs1, ys), vecDotQ6kQ8k(n, xs2, ys), vecDotQ6kQ8k(n, xs3, ys))
            }
            return vecDot4Q6kQ8k(n, xs0, xs1, xs2, xs3, ys)
            val (d0, d1) = BlockQ6K.vecDot2(n, xs0, xs1, ys)
            val (d2, d3) = BlockQ6K.vecDot2(n, xs2, xs3, ys)
            return listOf(d0, d1, d2, d3)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
            check(n.isMultipleOf(QK_K)) { "vec_dot_q6k_q8k: $n is not divisible by $qkK" }
            val aux8 = List(QK_K) { 0 }
            val aux16 = List(8) { 0 }
            val sums = List(8) { 0.0f }
            val aux32 = List(8) { 0.0f }
            for ((x, y) in (xs.zip(ys))) {
                val q4 = x.ql
                val qh = x.qh
                val q8 = y.qs
                aux32.fill(0.0f)
                for (j in (0 until QK_K).stepBy(128)) {
                    val aux81 = ((aux8).slice(j until aux8.size))
                    val q41 = ((q4).slice(j / 2 until q4.size))
                    val qh1 = ((qh).slice(j / 4 until qh.size))
                    for (l in 0 until 32) {
                        aux81[l] = (((q41[l] and 0xF) or ((qh1[l] and 3) shl 4)).toInt() - 32).toByte()
                        aux81[l + 32] = (((q41[l + 32] and 0xF) or (((qh1[l] shr 2) and 3) shl 4)).toInt() - 32).toByte()
                        aux81[l + 64] = (((q41[l] shr 4) or (((qh1[l] shr 4) and 3) shl 4)).toInt() - 32).toByte()
                        aux81[l + 96] = (((q41[l + 32] shr 4) or (((qh1[l] shr 6) and 3) shl 4)).toInt() - 32).toByte()
                    }
                }
                for ((j1, scale) in x.scales.iter().enumerate()) {
                    val scale1 = scale.toFloat()
                    val q81 = ((q8).slice(16 * j1 until q8.size))
                    val aux82 = ((aux8).slice(16 * j1 until aux8.size))
                    for (l1 in 0 until 8) {
                        aux16[l1] = q81[l1].toShort() * aux82[l1].toShort()
                    }
                    for (l2 in 0 until 8) {
                        aux32[l2] += scale1 * aux16[l2].toFloat()
                    }
                    val q82 = ((q81).slice(8 until q81.size))
                    val aux83 = ((aux82).slice(8 until aux82.size))
                    for (l3 in 0 until 8) {
                        aux16[l3] = q82[l3].toShort() * aux83[l3].toShort()
                    }
                    for (l4 in 0 until 8) {
                        aux32[l4] += scale1 * aux16[l4].toFloat()
                    }
                }
                val d = x.d.toF32() * y.d
                for ((sum, a) in (sums.iterMut().zip(aux32.iter()))) {
                    sum += a * d
                }
            }
            return sums.iter().sum()
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ6K>) {
            check(xs.size == ys.size * BlockQ6K.BLCK_SIZE)
            val localStorageL2140C8 = AddressableArrayLocal(ByteArray(QK_K) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I8, (QK_K).toInt()))
            var l by localStorageL2140C8
            val scales = List(QK_K / 16) { 0.0f }
            var x = xs.asPtr()
            val l1 = localStorageL2140C8.elementMutPtr(0)
            for (y in ys.iterMut()) {
                var maxScale = 0.0f
                var maxAbsScale = 0.0f
                for ((ib, scale) in scales.iterMut().enumerate()) {
                    val scale = makeQxQuants(16, 32, x.add(16 * ib), l1.add(16 * ib), 1, ConstPtr(0L))
                    scale = scale
                    val absScale = scale.abs()
                    if (absScale > maxAbsScale) {
                        maxAbsScale = absScale
                        maxScale = scale
                    }
                }
                val iscale = -128.0f / maxScale
                y.d = fromF32(1.0 / iscale)
                for ((yScale, scale1) in (y.scales.iterMut().zip(scales.iter()))) {
                    yScale = nearestInt(iscale * scale1).min(127).toByte()
                }
                for ((j, yScale1) in y.scales.iter().enumerate()) {
                    val d = y.d.toF32() * yScale1.toFloat()
                    if (d == 0.0f) {
                        continue
                    }
                    for (ii in 0 until 16) {
                        val ll = nearestInt(x.add(16 * j + ii) / d).clamp(-32, 31)
                        l1.add(16 * j + ii) = (ll + 32).toByte()
                    }
                }
                var ql = y.ql.asMutPtr()
                var qh = y.qh.asMutPtr()
                for (j1 in (0 until QK_K).stepBy(128)) {
                    for (lIdx in 0 until 32) {
                        val q1 = l1.add(j1 + lIdx) and 0xF
                        val q2 = l1.add(j1 + lIdx + 32) and 0xF
                        val q3 = l1.add(j1 + lIdx + 64) and 0xF
                        val q4 = l1.add(j1 + lIdx + 96) and 0xF
                        ql.add(lIdx) = (q1 or (q3 shl 4)).toUByte()
                        ql.add(lIdx + 32) = (q2 or (q4 shl 4)).toUByte()
                        qh.add(lIdx) = ((l1.add(j1 + lIdx) shr 4) or ((l1.add(j1 + lIdx + 32) shr 4) shl 2) or ((l1.add(j1 + lIdx + 64) shr 4) shl 4) or ((l1.add(j1 + lIdx + 96) shr 4) shl 6)).toUByte()
                    }
                    ql = ql.add(64)
                    qh = qh.add(32)
                }
                x = x.add(QK_K)
            }
        }

        public fun fromFloatImatrix(xs: List<Float>, ys: List<BlockQ6K>, imatrixWeights: List<Float>, nPerRow: Int) {
            check(xs.size == ys.size * BlockQ6K.BLCK_SIZE)
            val localStorageL2212C8 = AddressableArrayLocal(ByteArray(QK_K) { 0 }, FixedArrayTypeLayout(PrimitiveLayouts.I8, (QK_K).toInt()))
            var l by localStorageL2212C8
            val scales = List(QK_K / 16) { 0.0f }
            var x = xs.asPtr()
            val imatrixWeights1 = imatrixWeights.asPtr()
            val l1 = localStorageL2212C8.elementMutPtr(0)
            for ((sblkIdx, y) in ys.enumerate()) {
                var maxScale = 0.0f
                var maxAbsScale = 0.0f
                for ((ib, scale) in scales.iterMut().enumerate()) {
                    val imatrixRow = sblkIdx % (nPerRow / QK_K)
                    val scale = makeQxQuants(16, 32, x.add(16 * ib), l1.add(16 * ib), 1, imatrixWeights1.add(QK_K * imatrixRow + 16 * ib))
                    scale = scale
                    val absScale = scale.abs()
                    if (absScale > maxAbsScale) {
                        maxAbsScale = absScale
                        maxScale = scale
                    }
                }
                val iscale = -128.0f / maxScale
                y.d = fromF32(1.0 / iscale)
                for ((yScale, scale1) in (y.scales.iterMut().zip(scales.iter()))) {
                    yScale = nearestInt(iscale * scale1).min(127).toByte()
                }
                for ((j, yScale1) in y.scales.iter().enumerate()) {
                    val d = y.d.toF32() * yScale1.toFloat()
                    if (d == 0.0f) {
                        continue
                    }
                    for (ii in 0 until 16) {
                        val ll = nearestInt(x.add(16 * j + ii) / d).clamp(-32, 31)
                        l1.add(16 * j + ii) = (ll + 32).toByte()
                    }
                }
                var ql = y.ql.asMutPtr()
                var qh = y.qh.asMutPtr()
                for (j1 in (0 until QK_K).stepBy(128)) {
                    for (lIdx in 0 until 32) {
                        val q1 = l1.add(j1 + lIdx) and 0xF
                        val q2 = l1.add(j1 + lIdx + 32) and 0xF
                        val q3 = l1.add(j1 + lIdx + 64) and 0xF
                        val q4 = l1.add(j1 + lIdx + 96) and 0xF
                        ql.add(lIdx) = (q1 or (q3 shl 4)).toUByte()
                        ql.add(lIdx + 32) = (q2 or (q4 shl 4)).toUByte()
                        qh.add(lIdx) = ((l1.add(j1 + lIdx) shr 4) or ((l1.add(j1 + lIdx + 32) shr 4) shl 2) or ((l1.add(j1 + lIdx + 64) shr 4) shl 4) or ((l1.add(j1 + lIdx + 96) shr 4) shl 6)).toUByte()
                    }
                    ql = ql.add(64)
                    qh = qh.add(32)
                }
                x = x.add(QK_K)
            }
        }

        // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L1067
        public fun toFloat(xs: List<BlockQ6K>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK_K)) { "dequantize_row_q6k: $k is not divisible by $qkK" }
            for ((idxX, x) in xs.enumerate()) {
                val d = x.d.toF32()
                val ql = x.ql
                val qh = x.qh
                val sc = x.scales
                for (n in (0 until QK_K).stepBy(128)) {
                    val idx = n / 128
                    val ys1 = ((ys).slice(idxX * QK_K + n until ys.size))
                    val sc1 = ((sc).slice(8 * idx until sc.size))
                    val ql1 = ((ql).slice(64 * idx until ql.size))
                    val qh1 = ((qh).slice(32 * idx until qh.size))
                    for (l in 0 until 32) {
                        val isMatch = l / 16
                        val q1 = ((ql1[l] and 0xF) or ((qh1[l] and 3) shl 4)).toByte() - 32
                        val q2 = ((ql1[l + 32] and 0xF) or (((qh1[l] shr 2) and 3) shl 4)).toByte() - 32
                        val q3 = ((ql1[l] shr 4) or (((qh1[l] shr 4) and 3) shl 4)).toByte() - 32
                        val q4 = ((ql1[l + 32] shr 4) or (((qh1[l] shr 6) and 3) shl 4)).toByte() - 32
                        ys1[l] = d * sc1[isMatch].toFloat() * q1.toFloat()
                        ys1[l + 32] = d * sc1[isMatch + 2].toFloat() * q2.toFloat()
                        ys1[l + 64] = d * sc1[isMatch + 4].toFloat() * q3.toFloat()
                        ys1[l + 96] = d * sc1[isMatch + 6].toFloat() * q4.toFloat()
                    }
                }
            }
        }
    }
}

public class BlockQ8K(
    internal var d: Float = 0.0f,
    internal var qs: ByteArray = ByteArray(QK_K),
    internal var bsums: ShortArray = ShortArray(QK_K / 16),
) : GgmlType<BlockQ8K> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<BlockQ8K> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<BlockQ8K> {
                override val alignBytes: Long = 4L
                private val off2: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((4L + ((QK_K).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8).strideBytes)), 2L)
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off2 + ((QK_K / 16).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16).strideBytes)), 4L)
                override fun readNative(addr: Long): BlockQ8K =
                    BlockQ8K(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32).readNative(addr),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8, (QK_K).toInt())
                        ).readNative(addr + 4L).toByteArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16, (QK_K / 16).toInt())
                        ).readNative(addr + off2).toShortArray(),
                    )
                override fun writeNative(addr: Long, value: BlockQ8K) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F32).writeNative(addr, value.d)
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I8, (QK_K).toInt())
                    ).writeNative(addr + 4L, value.qs.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I16, (QK_K / 16).toInt())
                    ).writeNative(addr + off2, value.bsums.toList())
                }
            }

        public val DTYPE: GgmlDType = GgmlDType.Q8K

        public val BLCK_SIZE: Int = QK_K

        public fun vecDot(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
            if (hasAvx2Fma()) {
                return vecDotQ8kQ8k(n, xs, ys)
            }
            return io.github.kotlinmania.candle.quantized.neon
                .vecDotQ8kQ8k(n, xs, ys)
            return io.github.kotlinmania.candle.quantized.simd128
                .vecDotQ8kQ8k(n, xs, ys)
            return BlockQ8K.vecDotUnopt(n, xs, ys)
        }

        public fun vecDotUnopt(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
            check(n.isMultipleOf(QK_K)) { "vec_dot_q8k_q8k: $n is not divisible by $qkK" }
            // Generic implementation.
            var sumf = 0.0f
            for ((xs1, ys1) in (xs.zip(ys))) {
                val sumI = (xs1.qs.iter().zip(ys1.qs.iter())).map({ (x, y) -> x.toInt() * y.toInt() }).sum<Int>()
                sumf += sumI.toFloat() * xs1.d * ys1.d
            }
            return sumf
        }

        public fun fromFloat(xs: List<Float>, ys: List<BlockQ8K>) {
            return quantizeRowQ8k(xs, ys)
            val k = xs.size
            check(k.isMultipleOf(QK_K)) { "quantize_row_q8k: $k is not divisible by $qkK" }
            for ((i, y) in ys.enumerate()) {
                var max = 0.0f
                var amax = 0.0f
                val xs1 = ((xs).slice(i * QK_K until (i + 1) * QK_K))
                for (x in xs1.iter()) {
                    if (amax < x.abs()) {
                        amax = x.abs()
                        max = x
                    }
                }
                if (amax == 0.0f) {
                    y.d = 0.0f
                    y.qs.fill(0)
                    y.bsums.fill(0)
                } else {
                    val iscale = -127.0f / max
                    for ((j, q) in y.qs.iterMut().enumerate()) {
                        val v = (iscale * xs1[j]).round()
                        q = v.clamp(-128.0, 127.0).toByte()
                    }
                    for (j1 in 0 until QK_K / 16) {
                        var sum = 0
                        for (ii in 0 until 16) {
                            sum += y.qs[j1 * 16 + ii].toInt()
                        }
                        y.bsums[j1] = sum.toShort()
                    }
                    y.d = 1.0 / iscale
                }
            }
        }

        public fun toFloat(xs: List<BlockQ8K>, ys: List<Float>) {
            val k = ys.size
            check(k.isMultipleOf(QK_K)) { "dequantize_row_q8k: $k is not divisible by $qkK" }
            for ((i, x) in xs.enumerate()) {
                for ((j, q) in x.qs.iter().enumerate()) {
                    ys[i * QK_K + j] = x.d * q.toFloat()
                }
            }
        }
    }
}

/**
 * Number of lhs rows processed against each weight column quad before moving on to the next
 * quad in the generic (non-repacked) CPU matmul. With `m> 1` the quantized lhs tile stays in
 * cache while the rhs quad is streamed through once per tile instead of once per lhs row. This
 * matches the 16-row src1 blocking of `ggmlComputeForwardMulMat`.
 */
public val MATMUL_ROW_TILE: Int = 16

// https://github.com/ggml-org/llama.cpp/blob/aa3ee0eb0b80efca126cedf9bcb4fb5864b46ce3/ggml/src/ggml-cpu/ggml-cpu.c#L1205
public fun <VecDotType, T : GgmlType<T>> matmul(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, rhsT: List<T>, dst: List<Float>, vecDot4TFn: (Int, List<T>, List<T>, List<T>, List<T>, List<VecDotType>) -> List<Any?>, vecDot2TFn: (Int, List<T>, List<T>, List<VecDotType>) -> Pair<Float, Float>, vecDotTFn: (Int, List<T>, List<VecDotType>) -> Float): Result<Unit> {
    check(T.BLCK_SIZE == T.VecDotType.BLCK_SIZE)
    // Real asserts rather than debug ones: the workers below write `dst` through a raw pointer,
    // so an undersized `dst` would otherwise be written out of bounds instead of panicking.
    check(m * k == lhs.size)
    check(dst.size >= m * n) { "dst too small: ${dst.size} < ${m * n}" }
    val kInBlocks = k.divCeil(T.BLCK_SIZE)
    // Thread-local scratch buffer reused across calls to avoid per-matmul
    // heap allocation of the quantized LHS.
    // Using u64 ensures sufficient alignment regardless of `T.VecDotType`.
    threadLocal(LHS_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<VecDotType>(typeLayoutT = ObjectSlotLayout<VecDotType>())
    // Required scratch buffer length in u64
    val requiredScratchLen = (m * kInBlocks * elemSize).divCeil(8)
    return LHS_SCRATCH.with(
        closure@{ cell ->
            var scratch = cell.borrowMut()
            if (scratch.len() < requiredScratchLen) {
                scratch.resize(requiredScratchLen, 0)
            }
            // SAFETY: u64 ensures sufficient alignment. Resize ensures sufficient size.
            // All elements written before reading.
            val lhsB: List<VecDotType> = fromRawPartsMut(mutPtrFromRustCast<VecDotType>(scratch.asMutPtr()), m * kInBlocks)
            // f32, f16, and bf16 support direct copy
            if (T.DIRECT_COPY) {
                lhs.directCopy(lhsB)
            } else {
                for (rowIdx in 0 until m) {
                    val lhsBMut = ((lhsB).slice(rowIdx * kInBlocks until (rowIdx + 1) * kInBlocks))
                    val lhs1 = ((lhs).slice(rowIdx * k until (rowIdx + 1) * k))
                    lhs1.fromFloat(lhsBMut)
                }
            }
            val nQuad = n and 3.inv()
            val quadsTotal = nQuad / 4
            val nTail = n - nQuad
            // 0..=3
            val pool = barrierPool()
            val lhsB1: List<VecDotType> = lhsB
            val dstPtr = dst.asMutPtr().toInt()
            val lhsRow = { rowIdx1: Int -> ((lhsB1).slice(rowIdx1 * kInBlocks until (rowIdx1 + 1) * kInBlocks)) }
            // Four dot products of one lhs row against the column quad starting at `col`.
            // SAFETY (for the raw dst writes below): every (row, column) of dst is written by exactly
            // one work item, so concurrent workers never touch the same element.
            val dotQuad =
                { rowIdx1: Int, col: Int ->
                    val (d0, d1, d2, d3) = vecDot4TFn(k, ((rhsT).slice(col * kInBlocks until (col + 1) * kInBlocks)), ((rhsT).slice((col + 1) * kInBlocks until (col + 2) * kInBlocks)), ((rhsT).slice((col + 2) * kInBlocks until (col + 3) * kInBlocks)), ((rhsT).slice((col + 3) * kInBlocks until (col + 4) * kInBlocks)), lhsRow(rowIdx1))
                    run {
                        val base = MutPtr(((MutPtr<Float>(dstPtr.toLong()))).rawAddress + ((rowIdx1 * n + col).toLong() * (PrimitiveLayouts.F32).strideBytes))
                        (PrimitiveLayouts.F32).writeNative((base).rawAddress, d0)
                        (PrimitiveLayouts.F32).writeNative((MutPtr((base).rawAddress + ((1).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress, d1)
                        (PrimitiveLayouts.F32).writeNative((MutPtr((base).rawAddress + ((2).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress, d2)
                        (PrimitiveLayouts.F32).writeNative((MutPtr((base).rawAddress + ((3).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress, d3)
                    }
                }
            // The 0..=3 columns left over after the quads.
            val dotTail =
                { rowIdx1: Int ->
                    val dstRow = fromRawPartsMut(MutPtr(((MutPtr<Float>(dstPtr.toLong()))).rawAddress + ((rowIdx1 * n + nQuad).toLong() * (PrimitiveLayouts.F32).strideBytes)), nTail)
                    if (nTail >= 2) {
                        val col = nQuad
                        val (d0, d1) = vecDot2TFn(k, ((rhsT).slice(col * kInBlocks until (col + 1) * kInBlocks)), ((rhsT).slice((col + 1) * kInBlocks until (col + 2) * kInBlocks)), lhsRow(rowIdx1))
                        dstRow[0] = d0
                        dstRow[1] = d1
                    }
                    if (nTail and 1 == 1) {
                        val col1 = n - 1
                        dstRow[nTail - 1] = vecDotTFn(k, ((rhsT).slice(col1 * kInBlocks until (col1 + 1) * kInBlocks)), lhsRow(rowIdx1))
                    }
                }
            if (m == 1) {
                // Decode: a single row, the column quads are split across the pool.
                val dotRange =
                    { range: Range<Int> ->
                        for (quadIdx in range) {
                            dotQuad(0, quadIdx * 4)
                        }
                    }
                if (false && T.DTYPE == GgmlDType.Q6K) {
                    // Compact Q6K GEMV has equal-sized dot products; static ranges
                    // avoid the shared cursor overhead on this short decode path.
                    pool.executeStatic(quadsTotal, dotRange)
                } else {
                    pool.executeChunked(quadsTotal, dotRange)
                }
                dotTail(0)
                return@closure Ok(Unit)
            }
            // Prefill: the work items are (row tile, column quad) pairs, row tile major, so that a
            // worker walking a range of items keeps one tile of quantized lhs rows hot in L1 and
            // reads each weight quad once per tile instead of once per row.
            val rowTiles = m.divCeil(MATMUL_ROW_TILE)
            pool.executeChunked(rowTiles * quadsTotal, { range ->
                for (item in range) {
                    val rowStart = (item / quadsTotal) * MATMUL_ROW_TILE
                    val rowEnd = m.min(rowStart + MATMUL_ROW_TILE)
                    val col = (item % quadsTotal) * 4
                    for (rowIdx1 in rowStart until rowEnd) {
                        dotQuad(rowIdx1, col)
                    }
                }
            })
            if (nTail > 0) {
                pool.executeChunked(m, { rows ->
                    for (rowIdx1 in rows) {
                        dotTail(rowIdx1)
                    }
                })
            }
            Ok(Unit)
        },
    )
}

internal fun matmulQ4kX8(tupleArg0: Triple<Int, Int, Int>, lhs: List<Float>, repacked: List<BlockQ4Kx8>, dst: List<Float>): Result<Unit> {
    check(n.isMultipleOf(8))
    val kInBlocks = k / QK_K
    val nGroups = n / 8
    threadLocal(LHS_SCRATCH, std, cell, RefCell, Vec, std, cell, RefCell, new, Vec, new)
    val elemSize = sizeOf<BlockQ8K>(typeLayoutT = BlockQ8K.typeLayout())
    val requiredLen = (m * kInBlocks * elemSize).divCeil(8)
    return LHS_SCRATCH.with(
        closure1@{ cell ->
            var scratch = cell.borrowMut()
            if (scratch.len() < requiredLen) {
                scratch.resize(requiredLen, 0)
            }
            val lhsB: List<BlockQ8K> = fromRawPartsMut(mutPtrFromRustCast<BlockQ8K>(scratch.asMutPtr()), m * kInBlocks)
            for (rowIdx in 0 until m) {
                val lhsRow = ((lhs).slice(rowIdx * k until (rowIdx + 1) * k))
                val lhsBMut = ((lhsB).slice(rowIdx * kInBlocks until (rowIdx + 1) * kInBlocks))
                BlockQ8K.fromFloat(lhsRow, lhsBMut)
            }
            val pool = barrierPool()
            val lhsB1: List<BlockQ8K> = lhsB
            val repackedPtr = repacked.asPtr().toInt()
            val x8BlockBytes = sizeOf<BlockQ4Kx8>(typeLayoutT = ObjectSlotLayout<BlockQ4Kx8>())
            // equal-sized gemv units: a static split skips the shared cursor, which costs
            // several us per small decode matmul
            if (m == 1) {
                val lhsRowPtr = lhsB1.asPtr().toInt()
                val dstRowPtr = dst.asMutPtr().toInt()
                pool.executeStatic(nGroups, { range ->
                    val lhsRow1: List<BlockQ8K> = fromRawParts(ConstPtr<BlockQ8K>(lhsRowPtr.toLong()), kInBlocks)
                    val dstPtr = MutPtr<Float>(dstRowPtr.toLong())
                    for (g in range) {
                        val xs = fromRawParts(ConstPtr<BlockQ4Kx8>((repackedPtr + g * kInBlocks * x8BlockBytes).toLong()), kInBlocks)
                        val results = vecDot8Q4kQ8k(k, xs, lhsRow1)
                        run {
                            copyNonoverlapping(results.asPtr(), MutPtr((dstPtr).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                        }
                    }
                })
                return@closure1 Ok(Unit)
            }
            for (rowIdx1 in 0 until m) {
                val lhsRow1 = ((lhsB1).slice(rowIdx1 * kInBlocks until (rowIdx1 + 1) * kInBlocks))
                val lhsRowPtr1 = lhsRow1.asPtr().toInt()
                val dstRowPtr1 = ((dst).slice(rowIdx1 * n until (rowIdx1 + 1) * n)).asMutPtr().toInt()
                pool.executeStatic(nGroups, { range ->
                    val lhsRow2: List<BlockQ8K> = fromRawParts(ConstPtr<BlockQ8K>(lhsRowPtr1.toLong()), kInBlocks)
                    val dstPtr = MutPtr<Float>(dstRowPtr1.toLong())
                    for (g in range) {
                        val xs = fromRawParts(ConstPtr<BlockQ4Kx8>((repackedPtr + g * kInBlocks * x8BlockBytes).toLong()), kInBlocks)
                        val results = vecDot8Q4kQ8k(k, xs, lhsRow2)
                        run {
                            copyNonoverlapping(results.asPtr(), MutPtr((dstPtr).rawAddress + ((g * 8).toLong() * (PrimitiveLayouts.F32).strideBytes)), 8)
                        }
                    }
                })
            }
            Ok(Unit)
        },
    )
}

public fun <VecDotType, T : GgmlType<T>> matmulF16(mkn: Triple<Int, Int, Int>, lhs: List<F16>, rhsT: List<T>, dst: List<F16>, vecDotTFn: (Int, List<T>, List<VecDotType>) -> Float): Result<Unit> {
    val dst1 = dst
    val (m, k, n) = mkn
    if (m * k != lhs.size) {
        bail("unexpected lhs length {} {mkn:?}", lhs.size)
    }
    val kInLhsBlocks = k.divCeil(T.BLCK_SIZE)
    val kInRhsBlocks = k.divCeil(T.VecDotType.BLCK_SIZE)
    val lhsB = mutableListOf(T.VecDotType.zeros(), m * kInLhsBlocks)
    for (rowIdx in 0 until m) {
        val lhsB1 = ((lhsB).slice(rowIdx * kInLhsBlocks until (rowIdx + 1) * kInLhsBlocks))
        val lhs1 = ((lhs).slice(rowIdx * k until (rowIdx + 1) * k))
        val lhsF32: Vec = lhs1.iter().map({ x -> x.toF32() }).collect()
        lhsF32.fromFloat(lhsB1)
    }
    val lhsB2 = lhsB.asSlice()
    for (rowIdx1 in 0 until m) {
        val lhsRow = ((lhsB2).slice(rowIdx1 * kInLhsBlocks until (rowIdx1 + 1) * kInLhsBlocks))
        val dstRow = ((dst1).slice(rowIdx1 * n until (rowIdx1 + 1) * n))
        for ((colIdx, dst2) in dstRow.enumerate()) {
            val rhsCol = ((rhsT).slice(colIdx * kInRhsBlocks until (colIdx + 1) * kInRhsBlocks))
            val value = vecDotTFn(k, rhsCol, lhsRow)
            dst2 = fromF32(value)
        }
    }
    return Ok(Unit)
}

public val DTYPE: GgmlDType = GgmlDType.F32

public val BLCK_SIZE: Int = 1

public val DIRECT_COPY: Boolean = true

public fun ggmlTypeFloat(n: ULong, xs: List<Float>, ys: List<Float>): Float = Float.vecDotUnopt(n, xs, ys)

public fun ggmlTypeFloat(n: Int, xs: List<Float>, ys: List<Float>): Float {
    check(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
    check(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
    val res = 0.0f
    vecDotF32(xs.asPtr(), ys.asPtr(), res, n)
    return res
}

public fun ggmlTypeFloat(xs: List<Float>, ys: List<Float>) {
    check(xs.size == ys.size)
    ys.copyFromSlice(xs)
}

public val DTYPE1: GgmlDType = GgmlDType.F16

public val BLCK_SIZE1: Int = 1

public val DIRECT_COPY1: Boolean = true

public fun ggmlTypeF16(n: ULong, xs: List<F16>, ys: List<F16>): Float = F16.vecDotUnopt(n, xs, ys)

public fun ggmlTypeF16(n: Int, xs: List<F16>, ys: List<F16>): Float {
    check(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
    check(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
    val res = 0.0f
    vecDotF16(xs.asPtr(), ys.asPtr(), res, n)
    return res
}

public fun ggmlTypeF16(xs: List<Float>, ys: List<F16>) {
    check(xs.size == ys.size)
    ys.convertFromF32Slice(xs)
}

public val DTYPE2: GgmlDType = GgmlDType.BF16

public val BLCK_SIZE2: Int = 1

public val DIRECT_COPY2: Boolean = true

public fun ggmlTypeBf16(n: ULong, xs: List<Bf16>, ys: List<Bf16>): Float = Bf16.vecDotUnopt(n, xs, ys)

public fun ggmlTypeBf16(n: Int, xs: List<Bf16>, ys: List<Bf16>): Float {
    check(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
    check(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
    val res = 0.0f
    vecDotBf16(xs.asPtr(), ys.asPtr(), res, n)
    return res
}

public fun ggmlTypeBf16(xs: List<Float>, ys: List<Bf16>) {
    check(xs.size == ys.size)
    ys.convertFromF32Slice(xs)
}
