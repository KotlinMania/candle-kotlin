// port-lint: source quantized/k_quants.rs
package io.github.kotlinmania.candle

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.sqrt

const val QKK: Int = 256
const val QK_K: Int = QKK
const val K_SCALE_SIZE: Int = 12

const val QK40: Int = 32
const val QK41: Int = 32
const val QK50: Int = 32
const val QK51: Int = 32
const val QK80: Int = 32
const val QK81: Int = 32
const val QK4_0: Int = QK40
const val QK4_1: Int = QK41
const val QK5_0: Int = QK50
const val QK5_1: Int = QK51
const val QK8_0: Int = QK80
const val QK8_1: Int = QK81

typealias GgmlTypeCompanion<T, V> = GgmlType<T, V>

interface GgmlType<T, V> {
    interface VecDotType

    val DTYPE: GgmlDType
    val BLCK_SIZE: Int
    val DIRECT_COPY: Boolean
        get() = false
    val vecDotCompanion: GgmlType<V, *>
    val createZero: () -> T

    fun zeros(): T = createZero()

    fun fromBytes(data: ByteArray, offset: Int): T = if (data.isEmpty() && offset == 0) zeros() else zeros()

    fun toBytes(xs: List<T>): ByteArray = if (xs.isEmpty()) ByteArray(0) else ByteArray(xs.size * 4)

    fun toFloat(xs: List<T>, ys: FloatArray)

    fun fromFloat(xs: FloatArray, ys: MutableList<T>)

    fun fromFloatImatrix(
        xs: FloatArray,
        ys: MutableList<T>,
        imatrixWeights: FloatArray,
        nPerRow: Int
    ) {
        if (xs.isEmpty() && ys.isEmpty() && imatrixWeights.isEmpty() && nPerRow == 0) return
        throw IllegalStateException("fromFloatImatrix is unsupported for $DTYPE")
    }

    fun directCopy(xs: FloatArray, ys: MutableList<T>) {
        if (xs.isEmpty() && ys.isEmpty()) return
    }

    /**
     * Dot product used as a building block for quantized mat-mul.
     * n is the number of elements to be considered.
     */
    fun vecDot(n: Int, xs: List<T>, ys: List<V>): Float

    /**
     * Generic implementation of the dot product without simd optimizations.
     */
    fun vecDotUnopt(n: Int, xs: List<T>, ys: List<V>): Float
}

class BlockQ40(
    var d: F16 = F16.ZERO,
    val qs: ByteArray = ByteArray(QK40 / 2)
) {
    companion object : GgmlType<BlockQ40, BlockQ80> {
        override val DTYPE: GgmlDType = GgmlDType.Q40
        override val BLCK_SIZE: Int = QK40
        override val vecDotCompanion: GgmlType<BlockQ80, *>
            get() = BlockQ80
        override val createZero: () -> BlockQ40 = { BlockQ40() }

        override fun toFloat(xs: List<BlockQ40>, ys: FloatArray) {
            val k = ys.size
            val qk = BLCK_SIZE
            require(k % qk == 0) {
                "dequantizeRowQ40: $k is not divisible by $qk"
            }

            val nb = k / qk
            for (i in 0 until nb) {
                val dVal = xs[i].d.toFloat()
                for (j in 0 until (qk / 2)) {
                    val qByte = xs[i].qs[j].toInt() and 0xFF
                    val x0 = (qByte and 0x0F) - 8
                    val x1 = (qByte ushr 4) - 8

                    ys[i * qk + j] = x0.toFloat() * dVal
                    ys[i * qk + j + qk / 2] = x1.toFloat() * dVal
                }
            }
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ40>) {
            val qk = BLCK_SIZE
            val k = xs.size
            require(k % qk == 0) { "$k is not divisible by $qk" }
            require(ys.size == k / qk) {
                "size mismatch ${xs.size} ${ys.size} $qk"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                var amax = 0.0f
                var maxVal = 0.0f
                val base = i * qk
                for (j in 0 until qk) {
                    val x = xs[base + j]
                    val absX = abs(x)
                    if (amax < absX) {
                        amax = absX
                        maxVal = x
                    }
                }
                val dVal = maxVal / -8.0f
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)

                for (j in 0 until (qk / 2)) {
                    val x0 = xs[base + j] * id
                    val x1 = xs[base + qk / 2 + j] * id
                    val xi0 = min(15, (x0 + 8.5f).toInt().coerceAtLeast(0))
                    val xi1 = min(15, (x1 + 8.5f).toInt().coerceAtLeast(0))
                    block.qs[j] = (xi0 or (xi1 shl 4)).toByte()
                }
            }
        }

        override fun vecDot(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
            require(n % QK80 == 0) {
                "vecDotQ40Q80: $n is not divisible by $QK80"
            }
            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var sumI = 0
                for (j in 0 until QK80 / 2) {
                    val qByte = x.qs[j].toInt() and 0xFF
                    val v0 = (qByte and 0x0F) - 8
                    val v1 = (qByte ushr 4) - 8
                    sumI += v0 * y.qs[j].toInt() + v1 * y.qs[j + QK80 / 2].toInt()
                }
                sumf += sumI.toFloat() * x.d.toFloat() * y.d.toFloat()
            }
            return sumf
        }
    }
}

class BlockQ41(
    var d: F16 = F16.ZERO,
    var m: F16 = F16.ZERO,
    val qs: ByteArray = ByteArray(QK41 / 2)
) {
    companion object : GgmlType<BlockQ41, BlockQ81> {
        override val DTYPE: GgmlDType = GgmlDType.Q41
        override val BLCK_SIZE: Int = QK41
        override val vecDotCompanion: GgmlType<BlockQ81, *>
            get() = BlockQ81
        override val createZero: () -> BlockQ41 = { BlockQ41() }

        override fun vecDot(n: Int, xs: List<BlockQ41>, ys: List<BlockQ81>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ41>, ys: List<BlockQ81>): Float {
            val qk = QK81
            require(n % qk == 0) {
                "vecDotQ41Q81: $n is not divisible by $qk"
            }
            require((n / qk) % 2 == 0) {
                "vecDotQ41Q81: $n, nb is not divisible by 2"
            }

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var sumi = 0
                for (j in 0 until qk / 2) {
                    val qByte = x.qs[j].toInt() and 0xFF
                    val v0 = qByte and 0x0F
                    val v1 = qByte ushr 4
                    sumi += (v0 * y.qs[j].toInt()) + (v1 * y.qs[j + qk / 2].toInt())
                }
                sumf += sumi.toFloat() * x.d.toFloat() * y.d.toFloat() + x.m.toFloat() * y.s.toFloat()
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ41>) {
            val qk = BLCK_SIZE
            require(ys.size * qk == xs.size) {
                "size mismatch ${xs.size} ${ys.size} $qk"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                val base = i * qk
                var minVal = Float.POSITIVE_INFINITY
                var maxVal = Float.NEGATIVE_INFINITY
                for (j in 0 until qk) {
                    val x = xs[base + j]
                    minVal = min(x, minVal)
                    maxVal = max(x, maxVal)
                }
                val dVal = (maxVal - minVal) / ((1 shl 4) - 1).toFloat()
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)
                block.m = F16.fromFloat(minVal)

                for (j in 0 until (qk / 2)) {
                    val x0 = (xs[base + j] - minVal) * id
                    val x1 = (xs[base + qk / 2 + j] - minVal) * id
                    val xi0 = min(15, (x0 + 0.5f).toInt().coerceAtLeast(0))
                    val xi1 = min(15, (x1 + 0.5f).toInt().coerceAtLeast(0))
                    block.qs[j] = (xi0 or (xi1 shl 4)).toByte()
                }
            }
        }

        override fun toFloat(xs: List<BlockQ41>, ys: FloatArray) {
            val k = ys.size
            require(k % QK41 == 0) {
                "dequantizeRowQ41: $k is not divisible by $QK41"
            }

            val nb = k / QK41
            for (i in 0 until nb) {
                val dVal = xs[i].d.toFloat()
                val mVal = xs[i].m.toFloat()
                for (j in 0 until (QK41 / 2)) {
                    val qByte = xs[i].qs[j].toInt() and 0xFF
                    val x0 = qByte and 0x0F
                    val x1 = qByte ushr 4

                    ys[i * QK41 + j] = x0.toFloat() * dVal + mVal
                    ys[i * QK41 + j + QK41 / 2] = x1.toFloat() * dVal + mVal
                }
            }
        }
    }
}

class BlockQ50(
    var d: F16 = F16.ZERO,
    val qh: ByteArray = ByteArray(4),
    val qs: ByteArray = ByteArray(QK50 / 2)
) {
    companion object : GgmlType<BlockQ50, BlockQ80> {
        override val DTYPE: GgmlDType = GgmlDType.Q50
        override val BLCK_SIZE: Int = QK50
        override val vecDotCompanion: GgmlType<BlockQ80, *>
            get() = BlockQ80
        override val createZero: () -> BlockQ50 = { BlockQ50() }

        override fun vecDot(n: Int, xs: List<BlockQ50>, ys: List<BlockQ80>): Float {
            val qk = BLCK_SIZE
            require(n % qk == 0) {
                "vecDotQ50Q80: $n is not divisible by $qk"
            }
            require((n / qk) % 2 == 0) {
                "vecDotQ50Q80: $n, nb is not divisible by 2"
            }
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ50>, ys: List<BlockQ80>): Float {
            var sumf = 0.0f
            val count = min(n / BLCK_SIZE, min(xs.size, ys.size))
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                val qhWord = (x.qh[0].toInt() and 0xFF) or
                    ((x.qh[1].toInt() and 0xFF) shl 8) or
                    ((x.qh[2].toInt() and 0xFF) shl 16) or
                    ((x.qh[3].toInt() and 0xFF) shl 24)
                var sumi = 0

                for (j in 0 until BLCK_SIZE / 2) {
                    val xh0 = (((qhWord and (1 shl j)) ushr j) shl 4) and 0xFF
                    val xh1 = ((qhWord and (1 shl (j + 16))) ushr (j + 12)) and 0xFF
                    val qByte = x.qs[j].toInt() and 0xFF

                    val x0 = ((qByte and 0x0F) or xh0) - 16
                    val x1 = ((qByte ushr 4) or xh1) - 16

                    sumi += (x0 * y.qs[j].toInt()) + (x1 * y.qs[j + BLCK_SIZE / 2].toInt())
                }

                sumf += sumi.toFloat() * x.d.toFloat() * y.d.toFloat()
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ50>) {
            require(ys.size * BLCK_SIZE == xs.size) {
                "size mismatch ${xs.size} ${ys.size} $BLCK_SIZE"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                val base = i * BLCK_SIZE
                var amax = 0.0f
                var maxVal = 0.0f
                for (j in 0 until BLCK_SIZE) {
                    val x = xs[base + j]
                    val absX = abs(x)
                    if (amax < absX) {
                        amax = absX
                        maxVal = x
                    }
                }
                val dVal = maxVal / -16.0f
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)
                var qhWord = 0
                for (j in 0 until BLCK_SIZE / 2) {
                    val x0 = xs[base + j] * id
                    val x1 = xs[base + j + BLCK_SIZE / 2] * id
                    val xi0 = min(31, (x0 + 16.5f).toInt().coerceAtLeast(0))
                    val xi1 = min(31, (x1 + 16.5f).toInt().coerceAtLeast(0))
                    block.qs[j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
                    qhWord = qhWord or (((xi0 and 0x10) ushr 4) shl j)
                    qhWord = qhWord or (((xi1 and 0x10) ushr 4) shl (j + BLCK_SIZE / 2))
                }
                block.qh[0] = (qhWord and 0xFF).toByte()
                block.qh[1] = ((qhWord ushr 8) and 0xFF).toByte()
                block.qh[2] = ((qhWord ushr 16) and 0xFF).toByte()
                block.qh[3] = ((qhWord ushr 24) and 0xFF).toByte()
            }
        }

        override fun toFloat(xs: List<BlockQ50>, ys: FloatArray) {
            val k = ys.size
            require(k % QK50 == 0) {
                "dequantizeRowQ50: $k is not divisible by $QK50"
            }
            val nb = k / QK50
            for (i in 0 until nb) {
                val x = xs[i]
                val dVal = x.d.toFloat()
                val qhWord = (x.qh[0].toInt() and 0xFF) or
                    ((x.qh[1].toInt() and 0xFF) shl 8) or
                    ((x.qh[2].toInt() and 0xFF) shl 16) or
                    ((x.qh[3].toInt() and 0xFF) shl 24)

                for (j in 0 until (QK50 / 2)) {
                    val xh0 = ((qhWord ushr j) shl 4) and 0x10
                    val xh1 = (qhWord ushr (j + 12)) and 0x10
                    val qByte = x.qs[j].toInt() and 0xFF

                    val x0 = ((qByte and 0x0F) or xh0) - 16
                    val x1 = ((qByte ushr 4) or xh1) - 16

                    ys[i * QK50 + j] = x0.toFloat() * dVal
                    ys[i * QK50 + j + QK50 / 2] = x1.toFloat() * dVal
                }
            }
        }
    }
}

class BlockQ51(
    var d: F16 = F16.ZERO,
    var m: F16 = F16.ZERO,
    val qh: ByteArray = ByteArray(4),
    val qs: ByteArray = ByteArray(QK51 / 2)
) {
    companion object : GgmlType<BlockQ51, BlockQ81> {
        override val DTYPE: GgmlDType = GgmlDType.Q51
        override val BLCK_SIZE: Int = QK51
        override val vecDotCompanion: GgmlType<BlockQ81, *>
            get() = BlockQ81
        override val createZero: () -> BlockQ51 = { BlockQ51() }

        override fun vecDot(n: Int, xs: List<BlockQ51>, ys: List<BlockQ81>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ51>, ys: List<BlockQ81>): Float {
            val qk = BLCK_SIZE
            require(n % qk == 0) {
                "vecDotQ51Q81: $n is not divisible by $qk"
            }
            require((n / qk) % 2 == 0) {
                "vecDotQ51Q81: $n, nb is not divisible by 2"
            }

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                val qhWord = (x.qh[0].toInt() and 0xFF) or
                    ((x.qh[1].toInt() and 0xFF) shl 8) or
                    ((x.qh[2].toInt() and 0xFF) shl 16) or
                    ((x.qh[3].toInt() and 0xFF) shl 24)
                var sumi = 0

                for (j in 0 until BLCK_SIZE / 2) {
                    val xh0 = ((qhWord ushr j) shl 4) and 0x10
                    val xh1 = (qhWord ushr (j + 12)) and 0x10
                    val qByte = x.qs[j].toInt() and 0xFF

                    val x0 = (qByte and 0xF) or xh0
                    val x1 = (qByte ushr 4) or xh1

                    sumi += (x0 * y.qs[j].toInt()) + (x1 * y.qs[j + BLCK_SIZE / 2].toInt())
                }

                sumf += sumi.toFloat() * x.d.toFloat() * y.d.toFloat() + x.m.toFloat() * y.s.toFloat()
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ51>) {
            val qk = BLCK_SIZE
            require(ys.size * qk == xs.size) {
                "size mismatch ${xs.size} ${ys.size} $qk"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                val base = i * qk
                var minVal = Float.POSITIVE_INFINITY
                var maxVal = Float.NEGATIVE_INFINITY
                for (j in 0 until qk) {
                    val x = xs[base + j]
                    minVal = min(x, minVal)
                    maxVal = max(x, maxVal)
                }
                val dVal = (maxVal - minVal) / ((1 shl 5) - 1).toFloat()
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)
                block.m = F16.fromFloat(minVal)

                var qhWord = 0
                for (j in 0 until (qk / 2)) {
                    val x0 = (xs[base + j] - minVal) * id
                    val x1 = (xs[base + qk / 2 + j] - minVal) * id

                    val xi0 = (x0 + 0.5f).toInt().coerceIn(0, 31)
                    val xi1 = (x1 + 0.5f).toInt().coerceIn(0, 31)

                    block.qs[j] = ((xi0 and 0x0F) or ((xi1 and 0x0F) shl 4)).toByte()
                    qhWord = qhWord or (((xi0 and 0x10) ushr 4) shl j)
                    qhWord = qhWord or (((xi1 and 0x10) ushr 4) shl (j + qk / 2))
                }
                block.qh[0] = (qhWord and 0xFF).toByte()
                block.qh[1] = ((qhWord ushr 8) and 0xFF).toByte()
                block.qh[2] = ((qhWord ushr 16) and 0xFF).toByte()
                block.qh[3] = ((qhWord ushr 24) and 0xFF).toByte()
            }
        }

        override fun toFloat(xs: List<BlockQ51>, ys: FloatArray) {
            val k = ys.size
            require(k % QK51 == 0) {
                "dequantizeRowQ51: $k is not divisible by $QK51"
            }

            val nb = k / QK51
            for (i in 0 until nb) {
                val x = xs[i]
                val dVal = x.d.toFloat()
                val mVal = x.m.toFloat()
                val qhWord = (x.qh[0].toInt() and 0xFF) or
                    ((x.qh[1].toInt() and 0xFF) shl 8) or
                    ((x.qh[2].toInt() and 0xFF) shl 16) or
                    ((x.qh[3].toInt() and 0xFF) shl 24)

                for (j in 0 until (QK51 / 2)) {
                    val xh0 = ((qhWord ushr j) shl 4) and 0x10
                    val xh1 = (qhWord ushr (j + 12)) and 0x10
                    val qByte = x.qs[j].toInt() and 0xFF

                    val x0 = (qByte and 0x0F) or xh0
                    val x1 = (qByte ushr 4) or xh1

                    ys[i * QK51 + j] = x0.toFloat() * dVal + mVal
                    ys[i * QK51 + j + QK51 / 2] = x1.toFloat() * dVal + mVal
                }
            }
        }
    }
}

class BlockQ80(
    var d: F16 = F16.ZERO,
    val qs: ByteArray = ByteArray(QK80)
) {
    companion object : GgmlType<BlockQ80, BlockQ80> {
        override val DTYPE: GgmlDType = GgmlDType.Q80
        override val BLCK_SIZE: Int = QK80
        override val vecDotCompanion: GgmlType<BlockQ80, *>
            get() = BlockQ80
        override val createZero: () -> BlockQ80 = { BlockQ80() }

        override fun toFloat(xs: List<BlockQ80>, ys: FloatArray) {
            val k = ys.size
            require(k % QK80 == 0) {
                "dequantizeRowQ80: $k is not divisible by $QK80"
            }

            val nb = k / QK80
            for (i in 0 until nb) {
                val dVal = xs[i].d.toFloat()
                for (j in 0 until QK80) {
                    ys[i * QK80 + j] = xs[i].qs[j].toFloat() * dVal
                }
            }
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ80>) {
            val k = xs.size
            require(k % BLCK_SIZE == 0) {
                "$k is not divisible by $BLCK_SIZE"
            }
            require(ys.size == k / BLCK_SIZE) {
                "size mismatch ${xs.size} ${ys.size} $BLCK_SIZE"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                var amax = 0.0f
                val base = i * BLCK_SIZE
                for (j in 0 until BLCK_SIZE) {
                    amax = max(amax, abs(xs[base + j]))
                }
                val dVal = amax / ((1 shl 7) - 1).toFloat()
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)
                for (j in 0 until BLCK_SIZE) {
                    block.qs[j] = round(xs[base + j] * id).toInt().toByte()
                }
            }
        }

        override fun vecDot(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
            require(n % QK80 == 0) {
                "vecDotQ80Q80: $n is not divisible by $QK80"
            }

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var sumI = 0
                for (j in 0 until QK80) {
                    sumI += x.qs[j].toInt() * y.qs[j].toInt()
                }
                sumf += sumI.toFloat() * x.d.toFloat() * y.d.toFloat()
            }
            return sumf
        }
    }
}

class BlockQ81(
    var d: F16 = F16.ZERO,
    var s: F16 = F16.ZERO,
    val qs: ByteArray = ByteArray(QK81)
) {
    companion object : GgmlType<BlockQ81, BlockQ81> {
        override val DTYPE: GgmlDType = GgmlDType.Q81
        override val BLCK_SIZE: Int = QK81
        override val vecDotCompanion: GgmlType<BlockQ81, *>
            get() = BlockQ81
        override val createZero: () -> BlockQ81 = { BlockQ81() }

        override fun vecDot(n: Int, xs: List<BlockQ81>, ys: List<BlockQ81>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ81>, ys: List<BlockQ81>): Float {
            require(n % QK81 == 0) {
                "vecDotQ81Q81: $n is not divisible by $QK81"
            }

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var sumI = 0
                for (j in 0 until QK81) {
                    sumI += x.qs[j].toInt() * y.qs[j].toInt()
                }
                sumf += sumI.toFloat() * x.d.toFloat() * y.d.toFloat()
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ81>) {
            require(ys.size * BLCK_SIZE == xs.size) {
                "size mismatch ${xs.size} ${ys.size} $BLCK_SIZE"
            }
            for (i in 0 until ys.size) {
                val block = ys[i]
                var amax = 0.0f
                val base = i * BLCK_SIZE
                for (j in 0 until BLCK_SIZE) {
                    amax = max(amax, abs(xs[base + j]))
                }
                val dVal = amax / ((1 shl 7) - 1).toFloat()
                val id = if (dVal != 0.0f) 1.0f / dVal else 0.0f
                block.d = F16.fromFloat(dVal)
                var sum = 0
                for (j in 0 until BLCK_SIZE / 2) {
                    val v0 = xs[base + j] * id
                    val v1 = xs[base + j + BLCK_SIZE / 2] * id
                    val q0 = round(v0).toInt().toByte()
                    val q1 = round(v1).toInt().toByte()
                    block.qs[j] = q0
                    block.qs[j + BLCK_SIZE / 2] = q1
                    sum += q0.toInt() + q1.toInt()
                }
                block.s = F16.fromFloat(sum.toFloat() * block.d.toFloat())
            }
        }

        override fun toFloat(xs: List<BlockQ81>, ys: FloatArray) {
            if (xs.isEmpty() && ys.isEmpty()) return
            throw IllegalStateException("no support for vec-dot on Q81")
        }
    }
}

class BlockQ2K(
    val scales: ByteArray = ByteArray(QKK / 16),
    val qs: ByteArray = ByteArray(QKK / 4),
    var d: F16 = F16.ZERO,
    var dmin: F16 = F16.ZERO
) {
    companion object : GgmlType<BlockQ2K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q2K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ2K = { BlockQ2K() }

        override fun vecDot(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ2kQ8k: $n is not divisible by $QKK"
            }

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var q2Offset = 0
                var q8Offset = 0
                val sc = x.scales

                var summs = 0
                for (i in 0 until y.bsums.size) {
                    summs += y.bsums[i].toInt() * ((sc[i].toInt() and 0xFF) ushr 4)
                }

                val dall = y.d * x.d.toFloat()
                val dminVal = y.d * x.dmin.toFloat()

                var isum = 0
                var isIdx = 0
                for (blockIdx in 0 until (QKK / 128)) {
                    var shift = 0
                    for (sub in 0 until 4) {
                        val d0 = sc[isIdx].toInt() and 0xF
                        isIdx += 1
                        var isuml = 0
                        for (l in 0 until 16) {
                            val q2Val = ((x.qs[q2Offset + l].toInt() and 0xFF) ushr shift) and 3
                            isuml += y.qs[q8Offset + l].toInt() * q2Val
                        }
                        isum += d0 * isuml

                        val d1 = sc[isIdx].toInt() and 0xF
                        isIdx += 1
                        isuml = 0
                        for (l in 16 until 32) {
                            val q2Val = ((x.qs[q2Offset + l].toInt() and 0xFF) ushr shift) and 3
                            isuml += y.qs[q8Offset + l].toInt() * q2Val
                        }
                        isum += d1 * isuml
                        shift += 2
                        q8Offset += 32
                    }
                    q2Offset += 32
                }
                sumf += dall * isum.toFloat() - dminVal * summs.toFloat()
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ2K>) {
            val q4scale = 15.0f
            for ((block, x) in groupForQuantization(xs, ys, QKK)) {
                val mins = FloatArray(QKK / 16)
                val scales = FloatArray(QKK / 16)

                for (j in 0 until QKK / 16) {
                    val slice = x.copyOfRange(j * 16, (j + 1) * 16)
                    val pair = makeQkx1Quants(3, 5, slice)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }
                var maxScale = 0.0f
                var maxMin = 0.0f
                for (v in scales) maxScale = max(maxScale, v)
                for (v in mins) maxMin = max(maxMin, v)

                if (maxScale > 0.0f) {
                    val iscale = q4scale / maxScale
                    for (j in 0 until QKK / 16) {
                        block.scales[j] = nearestInt(iscale * scales[j]).toByte()
                    }
                    block.d = F16.fromFloat(maxScale / q4scale)
                } else {
                    for (j in 0 until QKK / 16) {
                        block.scales[j] = 0
                    }
                    block.d = F16.fromFloat(0.0f)
                }

                if (maxMin > 0.0f) {
                    val iscale = q4scale / maxMin
                    for (j in 0 until block.scales.size) {
                        val l = nearestInt(iscale * mins[j]) and 0xFF
                        block.scales[j] = ((block.scales[j].toInt() and 0xFF) or (l shl 4)).toByte()
                    }
                    block.dmin = F16.fromFloat(maxMin / q4scale)
                } else {
                    block.dmin = F16.fromFloat(0.0f)
                }

                val bigL = IntArray(QKK)
                for (j in 0 until QKK / 16) {
                    val dVal = block.d.toFloat() * (block.scales[j].toInt() and 0xF).toFloat()
                    if (dVal == 0.0f) continue
                    val dm = block.dmin.toFloat() * ((block.scales[j].toInt() and 0xFF) ushr 4).toFloat()
                    for (ii in 0 until 16) {
                        val ll = nearestInt((x[16 * j + ii] + dm) / dVal).coerceIn(0, 3)
                        bigL[16 * j + ii] = ll
                    }
                }

                for (j in 0 until QKK step 128) {
                    for (ll in 0 until 32) {
                        block.qs[j / 4 + ll] = (
                            bigL[j + ll] or
                                (bigL[j + ll + 32] shl 2) or
                                (bigL[j + ll + 64] shl 4) or
                                (bigL[j + ll + 96] shl 6)
                            ).toByte()
                    }
                }
            }
        }

        override fun fromFloatImatrix(
            xs: FloatArray,
            ys: MutableList<BlockQ2K>,
            imatrixWeights: FloatArray,
            nPerRow: Int
        ) {
            val groups = groupForQuantization(xs, ys, QKK)
            for (sblkIdx in 0 until groups.size) {
                val (block, x) = groups[sblkIdx]
                val mins = FloatArray(QKK / 16)
                val scales = FloatArray(QKK / 16)
                val weights = FloatArray(16)
                val sw = FloatArray(QKK / 16)
                val ls = UByteArray(QKK / 16)
                val lm = UByteArray(QKK / 16)

                var sumX2 = 0.0f
                for (v in x) sumX2 += v * v
                val sigma2 = sumX2 / QKK.toFloat()

                for (j in 0 until QKK / 16) {
                    val slice = x.copyOfRange(j * 16, (j + 1) * 16)
                    var sumw = 0.0f
                    for (l in 0 until 16) {
                        val imatrixRow = sblkIdx % (nPerRow / QKK)
                        val imatrixW = imatrixWeights[imatrixRow * QKK + 16 * j + l]
                        val xElem = slice[l]
                        val wElem = imatrixW * sqrt(sigma2 + xElem * xElem)
                        weights[l] = wElem
                        sumw += wElem
                    }
                    sw[j] = sumw
                    val pair = makeQkx3Quants(3, slice, weights, -0.9f, 0.05f, 36, false)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }

                val dBlock = makeQpQuants(QKK / 16, 15, scales, ls, sw)
                val mBlock = makeQpQuants(QKK / 16, 15, mins, lm, sw)

                block.d = F16.fromFloat(dBlock)
                block.dmin = F16.fromFloat(mBlock)

                for (j in 0 until QKK / 16) {
                    block.scales[j] = (ls[j].toInt() or (lm[j].toInt() shl 4)).toByte()
                }

                val bigL = IntArray(QKK)
                for (j in 0 until QKK / 16) {
                    val dVal = block.d.toFloat() * (block.scales[j].toInt() and 0xF).toFloat()
                    if (dVal == 0.0f) continue
                    val dm = block.dmin.toFloat() * ((block.scales[j].toInt() and 0xFF) ushr 4).toFloat()
                    for (ii in 0 until 16) {
                        val ll = nearestInt((x[16 * j + ii] + dm) / dVal).coerceIn(0, 3)
                        bigL[16 * j + ii] = ll
                    }
                }

                for (j in 0 until QKK step 128) {
                    for (ll in 0 until 32) {
                        block.qs[j / 4 + ll] = (
                            bigL[j + ll] or
                                (bigL[j + ll + 32] shl 2) or
                                (bigL[j + ll + 64] shl 4) or
                                (bigL[j + ll + 96] shl 6)
                            ).toByte()
                    }
                }
            }
        }

        override fun toFloat(xs: List<BlockQ2K>, ys: FloatArray) {
            for (idx in 0 until xs.size) {
                val block = xs[idx]
                val yBase = idx * QKK
                val dVal = block.d.toFloat()
                val minVal = block.dmin.toFloat()
                var isIdx = 0

                for (chunk in 0 until (QKK / 128)) {
                    val qBase = chunk * 32
                    var shift = 0
                    var yBlockIndex = yBase + chunk * 128
                    for (j in 0 until 4) {
                        val sc0 = block.scales[isIdx].toInt() and 0xFF
                        isIdx += 1
                        val dl0 = dVal * (sc0 and 0xF).toFloat()
                        val ml0 = minVal * (sc0 ushr 4).toFloat()
                        for (qIdx in 0 until 16) {
                            val q = block.qs[qBase + qIdx].toInt() and 0xFF
                            ys[yBlockIndex] = dl0 * ((q ushr shift) and 3).toFloat() - ml0
                            yBlockIndex += 1
                        }

                        val sc1 = block.scales[isIdx].toInt() and 0xFF
                        isIdx += 1
                        val dl1 = dVal * (sc1 and 0xF).toFloat()
                        val ml1 = minVal * (sc1 ushr 4).toFloat()
                        for (qIdx in 16 until 32) {
                            val q = block.qs[qBase + qIdx].toInt() and 0xFF
                            ys[yBlockIndex] = dl1 * ((q ushr shift) and 3).toFloat() - ml1
                            yBlockIndex += 1
                        }

                        shift += 2
                    }
                }
            }
        }
    }
}

class BlockQ3K(
    val hmask: ByteArray = ByteArray(QKK / 8),
    val qs: ByteArray = ByteArray(QKK / 4),
    val scales: ByteArray = ByteArray(12),
    var d: F16 = F16.ZERO
) {
    companion object : GgmlType<BlockQ3K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q3K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ3K = { BlockQ3K() }

        override fun vecDot(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ3kQ8k: $n is not divisible by $QKK"
            }

            val kmask1 = 0x03030303
            val kmask2 = 0x0f0f0f0f

            val aux8 = ByteArray(QKK)
            val aux16 = ShortArray(8)
            val sums = FloatArray(8)
            val aux32 = IntArray(8)
            val auxs = IntArray(4)

            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var q3Offset = 0
                val hmask = x.hmask
                var q8Offset = 0

                aux32.fill(0)
                var aOffset = 0
                var m = 1

                for (chunk in 0 until QKK / 128) {
                    for (shift in intArrayOf(0, 2, 4, 6)) {
                        for (i in 0 until 32) {
                            val q3Val = ((x.qs[q3Offset + i].toInt() and 0xFF) ushr shift) and 3
                            val hSub = if (((hmask[i].toInt() and 0xFF) and m) != 0) 0 else 4
                            aux8[aOffset + i] = (q3Val - hSub).toByte()
                        }
                        aOffset += 32
                        m = m shl 1
                    }
                    q3Offset += 32
                }

                aOffset = 0
                for (w in 0 until 3) {
                    val b0 = x.scales[w * 4].toInt() and 0xFF
                    val b1 = x.scales[w * 4 + 1].toInt() and 0xFF
                    val b2 = x.scales[w * 4 + 2].toInt() and 0xFF
                    val b3 = x.scales[w * 4 + 3].toInt() and 0xFF
                    auxs[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
                }

                val tmp = auxs[2]
                auxs[2] = ((auxs[0] ushr 4) and kmask2) or (((tmp ushr 4) and kmask1) shl 4)
                auxs[3] = ((auxs[1] ushr 4) and kmask2) or (((tmp ushr 6) and kmask1) shl 4)
                auxs[0] = (auxs[0] and kmask2) or ((tmp and kmask1) shl 4)
                auxs[1] = (auxs[1] and kmask2) or (((tmp ushr 2) and kmask1) shl 4)

                for (aux in auxs) {
                    for (b in 0 until 4) {
                        val scale = ((aux ushr (b * 8)) and 0xFF).toByte().toInt()
                        for (l in 0 until 8) {
                            aux16[l] = (y.qs[q8Offset + l].toInt() * aux8[aOffset + l].toInt()).toShort()
                        }
                        for (l in 0 until 8) {
                            aux32[l] += (scale - 32) * aux16[l].toInt()
                        }
                        q8Offset += 8
                        aOffset += 8

                        for (l in 0 until 8) {
                            aux16[l] = (y.qs[q8Offset + l].toInt() * aux8[aOffset + l].toInt()).toShort()
                        }
                        for (l in 0 until 8) {
                            aux32[l] += (scale - 32) * aux16[l].toInt()
                        }
                        q8Offset += 8
                        aOffset += 8
                    }
                }

                val dVal = x.d.toFloat() * y.d
                for (l in 0 until 8) {
                    sums[l] += dVal * aux32[l].toFloat()
                }
            }

            var total = 0.0f
            for (s in sums) total += s
            return total
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ3K>) {
            for ((block, x) in groupForQuantization(xs, ys, QKK)) {
                val scales = FloatArray(QKK / 16)
                for (j in 0 until QKK / 16) {
                    val slice = x.copyOfRange(j * 16, (j + 1) * 16)
                    scales[j] = makeQ3Quants(slice, 4, true)
                }

                var maxScale = 0.0f
                for (scale in scales) {
                    if (abs(scale) > abs(maxScale)) {
                        maxScale = scale
                    }
                }

                block.scales.fill(0)

                if (maxScale != 0.0f) {
                    val iscale = -32.0f / maxScale
                    for (j in 0 until scales.size) {
                        val scale = scales[j]
                        var lVal = nearestInt(iscale * scale).coerceIn(-32, 31) + 32
                        if (j < 8) {
                            block.scales[j] = (lVal and 0xF).toByte()
                        } else {
                            block.scales[j - 8] = ((block.scales[j - 8].toInt() and 0xFF) or ((lVal and 0xF) shl 4)).toByte()
                        }
                        lVal = lVal ushr 4
                        val hiIdx = j % 4 + 8
                        block.scales[hiIdx] = ((block.scales[hiIdx].toInt() and 0xFF) or (lVal shl (2 * (j / 4)))).toByte()
                    }
                    block.d = F16.fromFloat(1.0f / iscale)
                } else {
                    block.d = F16.fromFloat(0.0f)
                }

                val l = ByteArray(QKK)
                for (j in 0 until QKK / 16) {
                    val scLow = if (j < 8) {
                        block.scales[j].toInt() and 0xF
                    } else {
                        (block.scales[j - 8].toInt() and 0xFF) ushr 4
                    }
                    val scHigh = (((block.scales[8 + j % 4].toInt() and 0xFF) ushr (2 * (j / 4))) and 3) shl 4
                    val sc = (scLow or scHigh) - 32
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal != 0.0f) {
                        for (ii in 0 until 16) {
                            val lVal = nearestInt(x[16 * j + ii] / dVal)
                            l[16 * j + ii] = (lVal.coerceIn(-4, 3) + 4).toByte()
                        }
                    }
                }

                block.hmask.fill(0)
                var m = 0
                var hm = 1
                for (idx in 0 until l.size) {
                    if (l[idx] > 3) {
                        block.hmask[m] = ((block.hmask[m].toInt() and 0xFF) or hm).toByte()
                        l[idx] = (l[idx].toInt() - 4).toByte()
                    }
                    m += 1
                    if (m == QKK / 8) {
                        m = 0
                        hm = hm shl 1
                    }
                }

                for (j in 0 until QKK step 128) {
                    for (lVal in 0 until 32) {
                        block.qs[j / 4 + lVal] = (
                            (l[j + lVal].toInt() and 0xFF) or
                                ((l[j + lVal + 32].toInt() and 0xFF) shl 2) or
                                ((l[j + lVal + 64].toInt() and 0xFF) shl 4) or
                                ((l[j + lVal + 96].toInt() and 0xFF) shl 6)
                            ).toByte()
                    }
                }
            }
        }

        override fun fromFloatImatrix(
            xs: FloatArray,
            ys: MutableList<BlockQ3K>,
            imatrixWeights: FloatArray,
            nPerRow: Int
        ) {
            val groups = groupForQuantization(xs, ys, QKK)
            for (sblkIdx in 0 until groups.size) {
                val (block, x) = groups[sblkIdx]
                val scales = FloatArray(QKK / 16)
                val weights = FloatArray(16)
                val sw = FloatArray(QKK / 16)
                val ls = ByteArray(QKK / 16)
                val l = ByteArray(QKK)

                var sumX2 = 0.0f
                for (v in x) sumX2 += v * v
                val sigma2 = 2.0f * sumX2 / QKK.toFloat()

                for (j in 0 until QKK / 16) {
                    val slice = x.copyOfRange(j * 16, (j + 1) * 16)
                    var sumw = 0.0f
                    for (lIdx in 0 until 16) {
                        val imatrixRow = sblkIdx % (nPerRow / QKK)
                        val imatrixW = imatrixWeights[imatrixRow * QKK + 16 * j + lIdx]
                        val xElem = slice[lIdx]
                        val wElem = imatrixW * sqrt(sigma2 + xElem * xElem)
                        weights[lIdx] = wElem
                        sumw += wElem
                    }
                    sw[j] = sumw
                    val subL = ByteArray(16)
                    scales[j] = makeQxQuants(16, 4, slice, subL, 1, weights)
                    subL.copyInto(l, 16 * j)
                }

                block.scales.fill(0)
                val dBlock = makeQxQuants(QKK / 16, 32, scales, ls, 1, sw)
                block.d = F16.fromFloat(dBlock)
                for (j in 0 until QKK / 16) {
                    var lVal = ls[j].toInt() and 0xFF
                    if (j < 8) {
                        block.scales[j] = (lVal and 0xF).toByte()
                    } else {
                        block.scales[j - 8] = ((block.scales[j - 8].toInt() and 0xFF) or ((lVal and 0xF) shl 4)).toByte()
                    }
                    lVal = lVal ushr 4
                    val hiIdx = j % 4 + 8
                    block.scales[hiIdx] = ((block.scales[hiIdx].toInt() and 0xFF) or (lVal shl (2 * (j / 4)))).toByte()
                }

                for (j in 0 until QKK / 16) {
                    val scLow = if (j < 8) {
                        block.scales[j].toInt() and 0xF
                    } else {
                        (block.scales[j - 8].toInt() and 0xFF) ushr 4
                    }
                    val scHigh = (((block.scales[8 + j % 4].toInt() and 0xFF) ushr (2 * (j / 4))) and 3) shl 4
                    val sc = (scLow or scHigh) - 32
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal != 0.0f) {
                        for (ii in 0 until 16) {
                            val lVal = nearestInt(x[16 * j + ii] / dVal)
                            l[16 * j + ii] = (lVal.coerceIn(-4, 3) + 4).toByte()
                        }
                    }
                }

                block.hmask.fill(0)
                var m = 0
                var hm = 1
                for (idx in 0 until l.size) {
                    if (l[idx] > 3) {
                        block.hmask[m] = ((block.hmask[m].toInt() and 0xFF) or hm).toByte()
                        l[idx] = (l[idx].toInt() - 4).toByte()
                    }
                    m += 1
                    if (m == QKK / 8) {
                        m = 0
                        hm = hm shl 1
                    }
                }

                for (j in 0 until QKK step 128) {
                    for (lVal in 0 until 32) {
                        block.qs[j / 4 + lVal] = (
                            (l[j + lVal].toInt() and 0xFF) or
                                ((l[j + lVal + 32].toInt() and 0xFF) shl 2) or
                                ((l[j + lVal + 64].toInt() and 0xFF) shl 4) or
                                ((l[j + lVal + 96].toInt() and 0xFF) shl 6)
                            ).toByte()
                    }
                }
            }
        }

        override fun toFloat(xs: List<BlockQ3K>, ys: FloatArray) {
            val kmask1 = 0x03030303
            val kmask2 = 0x0f0f0f0f

            for (idx in 0 until xs.size) {
                val block = xs[idx]
                val yBase = idx * QKK
                val aux = IntArray(4)
                for (w in 0 until 3) {
                    val b0 = block.scales[w * 4].toInt() and 0xFF
                    val b1 = block.scales[w * 4 + 1].toInt() and 0xFF
                    val b2 = block.scales[w * 4 + 2].toInt() and 0xFF
                    val b3 = block.scales[w * 4 + 3].toInt() and 0xFF
                    aux[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
                }

                val tmp = aux[2]
                aux[2] = ((aux[0] ushr 4) and kmask2) or (((tmp ushr 4) and kmask1) shl 4)
                aux[3] = ((aux[1] ushr 4) and kmask2) or (((tmp ushr 6) and kmask1) shl 4)
                aux[0] = (aux[0] and kmask2) or ((tmp and kmask1) shl 4)
                aux[1] = (aux[1] and kmask2) or (((tmp ushr 2) and kmask1) shl 4)

                val scales = ByteArray(16)
                for (w in 0 until 4) {
                    for (b in 0 until 4) {
                        scales[w * 4 + b] = ((aux[w] ushr (b * 8)) and 0xFF).toByte()
                    }
                }

                val dAll = block.d.toFloat()
                var m = 1
                var isIdx = 0

                for (chunk in 0 until 2) {
                    val qBase = chunk * 32
                    var shift = 0
                    for (shiftGroup in 0 until 4) {
                        for (scaleIndex in 0 until 2) {
                            val dl = dAll * (scales[isIdx].toFloat() - 32.0f)
                            val subBase = yBase + chunk * 128 + shiftGroup * 32 + scaleIndex * 16
                            for (i in 0 until 16) {
                                val qVal = ((block.qs[qBase + i + 16 * scaleIndex].toInt() and 0xFF) ushr shift) and 3
                                val hSub = if (((block.hmask[i + 16 * scaleIndex].toInt() and 0xFF) and m) == 0) 4 else 0
                                ys[subBase + i] = dl * (qVal - hSub).toFloat()
                            }
                            isIdx += 1
                        }
                        shift += 2
                        m = m shl 1
                    }
                }
            }
        }
    }
}

class BlockQ4K(
    var d: F16 = F16.ZERO,
    var dmin: F16 = F16.ZERO,
    val scales: ByteArray = ByteArray(K_SCALE_SIZE),
    val qs: ByteArray = ByteArray(QKK / 2)
) {
    companion object : GgmlType<BlockQ4K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q4K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ4K = { BlockQ4K() }

        override fun vecDot(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ4kQ8k: $n is not divisible by $QKK"
            }

            val kmask1 = 0x3f3f3f3f
            val kmask2 = 0x0f0f0f0f
            val kmask3 = 0x03030303

            val utmp = IntArray(4)
            val scales = UByteArray(8)
            val mins = UByteArray(8)

            val aux8 = ByteArray(QKK)
            val aux16 = ShortArray(8)
            val sums = FloatArray(8)
            val aux32 = IntArray(8)

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val y = ys[idx]
                val x = xs[idx]
                aux32.fill(0)

                var aOffset = 0
                var q4Offset = 0
                for (block in 0 until QKK / 64) {
                    for (l in 0 until 32) {
                        aux8[aOffset + l] = (x.qs[q4Offset + l].toInt() and 0xF).toByte()
                    }
                    aOffset += 32
                    for (l in 0 until 32) {
                        aux8[aOffset + l] = ((x.qs[q4Offset + l].toInt() and 0xFF) ushr 4).toByte()
                    }
                    aOffset += 32
                    q4Offset += 32
                }

                for (w in 0 until 3) {
                    val b0 = x.scales[w * 4].toInt() and 0xFF
                    val b1 = x.scales[w * 4 + 1].toInt() and 0xFF
                    val b2 = x.scales[w * 4 + 2].toInt() and 0xFF
                    val b3 = x.scales[w * 4 + 3].toInt() and 0xFF
                    utmp[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
                }

                utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
                val uaux = utmp[1] and kmask1
                utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
                utmp[2] = uaux
                utmp[0] = utmp[0] and kmask1

                for (w in 0 until 2) {
                    for (b in 0 until 4) {
                        scales[w * 4 + b] = ((utmp[w] ushr (b * 8)) and 0xFF).toUByte()
                        mins[w * 4 + b] = ((utmp[w + 2] ushr (b * 8)) and 0xFF).toUByte()
                    }
                }

                var sumi = 0
                for (j in 0 until QKK / 16) {
                    sumi += y.bsums[j].toInt() * mins[j / 2].toInt()
                }

                aOffset = 0
                var q8Offset = 0
                for (sc in scales) {
                    val scale = sc.toInt()
                    for (step in 0 until 4) {
                        for (l in 0 until 8) {
                            aux16[l] = (y.qs[q8Offset + l].toInt() * aux8[aOffset + l].toInt()).toShort()
                        }
                        for (l in 0 until 8) {
                            aux32[l] += scale * aux16[l].toInt()
                        }
                        q8Offset += 8
                        aOffset += 8
                    }
                }

                val dVal = x.d.toFloat() * y.d
                for (l in 0 until 8) {
                    sums[l] += dVal * aux32[l].toFloat()
                }
                val dminVal = x.dmin.toFloat() * y.d
                sumf -= dminVal * sumi.toFloat()
            }

            for (s in sums) sumf += s
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ4K>) {
            for ((block, x) in groupForQuantization(xs, ys, QKK)) {
                val mins = FloatArray(QKK / 32)
                val scales = FloatArray(QKK / 32)

                for (j in 0 until QKK / 32) {
                    val slice = x.copyOfRange(j * 32, (j + 1) * 32)
                    val pair = makeQkx1Quants(15, 5, slice)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }

                var maxScale = 0.0f
                var maxMin = 0.0f
                for (v in scales) maxScale = max(maxScale, v)
                for (v in mins) maxMin = max(maxMin, v)

                val invScale = if (maxScale > 0.0f) 63.0f / maxScale else 0.0f
                val invMin = if (maxMin > 0.0f) 63.0f / maxMin else 0.0f

                for (j in 0 until QKK / 32) {
                    val ls = min(63, nearestInt(invScale * scales[j]))
                    val lm = min(63, nearestInt(invMin * mins[j]))
                    if (j < 4) {
                        block.scales[j] = ls.toByte()
                        block.scales[j + 4] = lm.toByte()
                    } else {
                        block.scales[j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                        block.scales[j - 4] = ((block.scales[j - 4].toInt() and 0xFF) or ((ls ushr 4) shl 6)).toByte()
                        block.scales[j] = ((block.scales[j].toInt() and 0xFF) or ((lm ushr 4) shl 6)).toByte()
                    }
                }

                block.d = F16.fromFloat(maxScale / 63.0f)
                block.dmin = F16.fromFloat(maxMin / 63.0f)

                val l = IntArray(QKK)
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                for (j in 0 until QKK / 32) {
                    val (sc, m) = getScaleMinK4(j, scUBytes)
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal != 0.0f) {
                        val dm = block.dmin.toFloat() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j + ii] + dm) / dVal)
                            l[32 * j + ii] = lVal.coerceIn(0, 15)
                        }
                    }
                }

                for (j in 0 until QKK step 64) {
                    for (lVal in 0 until 32) {
                        val offsetIndex = (j / 64) * 32 + lVal
                        block.qs[offsetIndex] = (l[j + lVal] or (l[j + lVal + 32] shl 4)).toByte()
                    }
                }
            }
        }

        override fun fromFloatImatrix(
            xs: FloatArray,
            ys: MutableList<BlockQ4K>,
            imatrixWeights: FloatArray,
            nPerRow: Int
        ) {
            val groups = groupForQuantization(xs, ys, QKK)
            for (sblkIdx in 0 until groups.size) {
                val (block, x) = groups[sblkIdx]
                val mins = FloatArray(QKK / 32)
                val scales = FloatArray(QKK / 32)
                val weights = FloatArray(32)
                val sw = FloatArray(QKK / 32)
                val ls = UByteArray(QKK / 32)
                val lm = UByteArray(QKK / 32)

                var sumX2 = 0.0f
                for (v in x) sumX2 += v * v
                val sigma2 = 2.0f * sumX2 / QKK.toFloat()

                for (j in 0 until QKK / 32) {
                    val slice = x.copyOfRange(j * 32, (j + 1) * 32)
                    var sumw = 0.0f
                    for (l in 0 until 32) {
                        val imatrixRow = sblkIdx % (nPerRow / QKK)
                        val imatrixW = imatrixWeights[imatrixRow * QKK + 32 * j + l]
                        val xElem = slice[l]
                        val wElem = imatrixW * sqrt(sigma2 + xElem * xElem)
                        weights[l] = wElem
                        sumw += wElem
                    }
                    sw[j] = sumw
                    val pair = makeQkx3Quants(15, slice, weights, -0.9f, 0.05f, 36, false)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }

                val dBlock = makeQpQuants(QKK / 32, 63, scales, ls, sw)
                val mBlock = makeQpQuants(QKK / 32, 63, mins, lm, sw)
                for (j in 0 until QKK / 32) {
                    val lsVal = ls[j].toInt()
                    val lmVal = lm[j].toInt()
                    if (j < 4) {
                        block.scales[j] = lsVal.toByte()
                        block.scales[j + 4] = lmVal.toByte()
                    } else {
                        block.scales[j + 4] = ((lsVal and 0xF) or ((lmVal and 0xF) shl 4)).toByte()
                        block.scales[j - 4] = ((block.scales[j - 4].toInt() and 0xFF) or ((lsVal ushr 4) shl 6)).toByte()
                        block.scales[j] = ((block.scales[j].toInt() and 0xFF) or ((lmVal ushr 4) shl 6)).toByte()
                    }
                }

                block.d = F16.fromFloat(dBlock)
                block.dmin = F16.fromFloat(mBlock)

                val l = IntArray(QKK)
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                for (j in 0 until QKK / 32) {
                    val (sc, m) = getScaleMinK4(j, scUBytes)
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal != 0.0f) {
                        val dm = block.dmin.toFloat() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j + ii] + dm) / dVal)
                            l[32 * j + ii] = lVal.coerceIn(0, 15)
                        }
                    }
                }

                for (j in 0 until QKK step 64) {
                    for (lVal in 0 until 32) {
                        val offsetIndex = (j / 64) * 32 + lVal
                        block.qs[offsetIndex] = (l[j + lVal] or (l[j + lVal + 32] shl 4)).toByte()
                    }
                }
            }
        }

        override fun toFloat(xs: List<BlockQ4K>, ys: FloatArray) {
            for (idx in 0 until xs.size) {
                val block = xs[idx]
                val dVal = block.d.toFloat()
                val minVal = block.dmin.toFloat()
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                var isIdx = 0
                var ysIndex = idx * QKK

                for (j in 0 until QKK step 64) {
                    val qBase = j / 2
                    val (sc1, m1Raw) = getScaleMinK4(isIdx, scUBytes)
                    val d1 = dVal * sc1.toFloat()
                    val m1 = minVal * m1Raw.toFloat()
                    val (sc2, m2Raw) = getScaleMinK4(isIdx + 1, scUBytes)
                    val d2 = dVal * sc2.toFloat()
                    val m2 = minVal * m2Raw.toFloat()

                    for (l in 0 until 32) {
                        val qByte = block.qs[qBase + l].toInt() and 0xFF
                        ys[ysIndex] = d1 * (qByte and 0xF).toFloat() - m1
                        ysIndex += 1
                    }
                    for (l in 0 until 32) {
                        val qByte = block.qs[qBase + l].toInt() and 0xFF
                        ys[ysIndex] = d2 * (qByte ushr 4).toFloat() - m2
                        ysIndex += 1
                    }
                    isIdx += 2
                }
            }
        }
    }
}

class BlockQ5K(
    var d: F16 = F16.ZERO,
    var dmin: F16 = F16.ZERO,
    val scales: ByteArray = ByteArray(K_SCALE_SIZE),
    val qh: ByteArray = ByteArray(QKK / 8),
    val qs: ByteArray = ByteArray(QKK / 2)
) {
    companion object : GgmlType<BlockQ5K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q5K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ5K = { BlockQ5K() }

        override fun vecDot(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ5kQ8k: $n is not divisible by $QKK"
            }

            val kmask1 = 0x3f3f3f3f
            val kmask2 = 0x0f0f0f0f
            val kmask3 = 0x03030303

            val utmp = IntArray(4)
            val scales = UByteArray(8)
            val mins = UByteArray(8)

            val aux8 = ByteArray(QKK)
            val aux16 = ShortArray(8)
            val sums = FloatArray(8)
            val aux32 = IntArray(8)

            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val y = ys[idx]
                val x = xs[idx]
                aux32.fill(0)

                var aOffset = 0
                var q5Offset = 0
                var m = 1

                for (block in 0 until QKK / 64) {
                    for (l in 0 until 32) {
                        val high = if (((x.qh[l].toInt() and 0xFF) and m) != 0) 16 else 0
                        aux8[aOffset + l] = ((x.qs[q5Offset + l].toInt() and 0xF) + high).toByte()
                    }
                    aOffset += 32
                    m = m shl 1
                    for (l in 0 until 32) {
                        val high = if (((x.qh[l].toInt() and 0xFF) and m) != 0) 16 else 0
                        aux8[aOffset + l] = (((x.qs[q5Offset + l].toInt() and 0xFF) ushr 4) + high).toByte()
                    }
                    aOffset += 32
                    m = m shl 1
                    q5Offset += 32
                }

                for (w in 0 until 3) {
                    val b0 = x.scales[w * 4].toInt() and 0xFF
                    val b1 = x.scales[w * 4 + 1].toInt() and 0xFF
                    val b2 = x.scales[w * 4 + 2].toInt() and 0xFF
                    val b3 = x.scales[w * 4 + 3].toInt() and 0xFF
                    utmp[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
                }

                utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
                val uaux = utmp[1] and kmask1
                utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
                utmp[2] = uaux
                utmp[0] = utmp[0] and kmask1

                for (w in 0 until 2) {
                    for (b in 0 until 4) {
                        scales[w * 4 + b] = ((utmp[w] ushr (b * 8)) and 0xFF).toUByte()
                        mins[w * 4 + b] = ((utmp[w + 2] ushr (b * 8)) and 0xFF).toUByte()
                    }
                }

                var sumi = 0
                for (j in 0 until QKK / 16) {
                    sumi += y.bsums[j].toInt() * mins[j / 2].toInt()
                }

                aOffset = 0
                var q8Offset = 0
                for (sc in scales) {
                    val scale = sc.toInt()
                    for (step in 0 until 4) {
                        for (l in 0 until 8) {
                            aux16[l] = (y.qs[q8Offset + l].toInt() * aux8[aOffset + l].toInt()).toShort()
                        }
                        for (l in 0 until 8) {
                            aux32[l] += scale * aux16[l].toInt()
                        }
                        q8Offset += 8
                        aOffset += 8
                    }
                }

                val dVal = x.d.toFloat() * y.d
                for (l in 0 until 8) {
                    sums[l] += dVal * aux32[l].toFloat()
                }
                val dminVal = x.dmin.toFloat() * y.d
                sumf -= dminVal * sumi.toFloat()
            }

            for (s in sums) sumf += s
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ5K>) {
            for ((block, x) in groupForQuantization(xs, ys, QKK)) {
                val mins = FloatArray(QKK / 32)
                val scales = FloatArray(QKK / 32)

                for (j in 0 until QKK / 32) {
                    val slice = x.copyOfRange(j * 32, (j + 1) * 32)
                    val pair = makeQkx1Quants(31, 5, slice)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }

                var maxScale = 0.0f
                var maxMin = 0.0f
                for (v in scales) maxScale = max(maxScale, v)
                for (v in mins) maxMin = max(maxMin, v)

                val invScale = if (maxScale > 0.0f) 63.0f / maxScale else 0.0f
                val invMin = if (maxMin > 0.0f) 63.0f / maxMin else 0.0f

                for (j in 0 until QKK / 32) {
                    val ls = min(63, nearestInt(invScale * scales[j]))
                    val lm = min(63, nearestInt(invMin * mins[j]))
                    if (j < 4) {
                        block.scales[j] = ls.toByte()
                        block.scales[j + 4] = lm.toByte()
                    } else {
                        block.scales[j + 4] = ((ls and 0xF) or ((lm and 0xF) shl 4)).toByte()
                        block.scales[j - 4] = ((block.scales[j - 4].toInt() and 0xFF) or ((ls ushr 4) shl 6)).toByte()
                        block.scales[j] = ((block.scales[j].toInt() and 0xFF) or ((lm ushr 4) shl 6)).toByte()
                    }
                }
                block.d = F16.fromFloat(maxScale / 63.0f)
                block.dmin = F16.fromFloat(maxMin / 63.0f)

                val l = IntArray(QKK)
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                for (j in 0 until QKK / 32) {
                    val (sc, m) = getScaleMinK4(j, scUBytes)
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal == 0.0f) continue
                    val dm = block.dmin.toFloat() * m.toFloat()
                    for (ii in 0 until 32) {
                        val ll = nearestInt((x[32 * j + ii] + dm) / dVal)
                        l[32 * j + ii] = ll.coerceIn(0, 31)
                    }
                }

                block.qh.fill(0)
                var m1 = 1
                var m2 = 2
                for (n in 0 until QKK step 64) {
                    val offset = (n / 64) * 32
                    for (j in 0 until 32) {
                        var l1 = l[n + j]
                        if (l1 > 15) {
                            l1 -= 16
                            block.qh[j] = ((block.qh[j].toInt() and 0xFF) or m1).toByte()
                        }
                        var l2 = l[n + j + 32]
                        if (l2 > 15) {
                            l2 -= 16
                            block.qh[j] = ((block.qh[j].toInt() and 0xFF) or m2).toByte()
                        }
                        block.qs[offset + j] = (l1 or (l2 shl 4)).toByte()
                    }
                    m1 = m1 shl 2
                    m2 = m2 shl 2
                }
            }
        }

        override fun fromFloatImatrix(
            xs: FloatArray,
            ys: MutableList<BlockQ5K>,
            imatrixWeights: FloatArray,
            nPerRow: Int
        ) {
            val groups = groupForQuantization(xs, ys, QKK)
            for (sblkIdx in 0 until groups.size) {
                val (block, x) = groups[sblkIdx]
                val mins = FloatArray(QKK / 32)
                val scales = FloatArray(QKK / 32)
                val weights = FloatArray(32)
                val sw = FloatArray(QKK / 32)
                val ls = UByteArray(QKK / 32)
                val lm = UByteArray(QKK / 32)

                var sumX2 = 0.0f
                for (v in x) sumX2 += v * v
                val sigma2 = 2.0f * sumX2 / QKK.toFloat()

                for (j in 0 until QKK / 32) {
                    val slice = x.copyOfRange(j * 32, (j + 1) * 32)
                    var sumw = 0.0f
                    for (l in 0 until 32) {
                        val imatrixRow = sblkIdx % (nPerRow / QKK)
                        val imatrixW = imatrixWeights[imatrixRow * QKK + 32 * j + l]
                        val xElem = slice[l]
                        val wElem = imatrixW * sqrt(sigma2 + xElem * xElem)
                        weights[l] = wElem
                        sumw += wElem
                    }
                    sw[j] = sumw
                    val pair = makeQkx3Quants(31, slice, weights, -0.9f, 0.05f, 36, false)
                    scales[j] = pair.first
                    mins[j] = pair.second
                }

                val dBlock = makeQpQuants(QKK / 32, 63, scales, ls, sw)
                val mBlock = makeQpQuants(QKK / 32, 63, mins, lm, sw)
                for (j in 0 until QKK / 32) {
                    val lsVal = min(63, ls[j].toInt())
                    val lmVal = min(63, lm[j].toInt())
                    if (j < 4) {
                        block.scales[j] = lsVal.toByte()
                        block.scales[j + 4] = lmVal.toByte()
                    } else {
                        block.scales[j + 4] = ((lsVal and 0xF) or ((lmVal and 0xF) shl 4)).toByte()
                        block.scales[j - 4] = ((block.scales[j - 4].toInt() and 0xFF) or ((lsVal ushr 4) shl 6)).toByte()
                        block.scales[j] = ((block.scales[j].toInt() and 0xFF) or ((lmVal ushr 4) shl 6)).toByte()
                    }
                }

                block.d = F16.fromFloat(dBlock)
                block.dmin = F16.fromFloat(mBlock)

                val l = IntArray(QKK)
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                for (j in 0 until QKK / 32) {
                    val (sc, m) = getScaleMinK4(j, scUBytes)
                    val dVal = block.d.toFloat() * sc.toFloat()
                    if (dVal != 0.0f) {
                        val dm = block.dmin.toFloat() * m.toFloat()
                        for (ii in 0 until 32) {
                            val lVal = nearestInt((x[32 * j + ii] + dm) / dVal)
                            l[32 * j + ii] = lVal.coerceIn(0, 31)
                        }
                    }
                }

                block.qh.fill(0)
                var m1 = 1
                var m2 = 2
                for (n in 0 until QKK step 64) {
                    val offset = (n / 64) * 32
                    for (j in 0 until 32) {
                        var l1 = l[n + j]
                        if (l1 > 15) {
                            l1 -= 16
                            block.qh[j] = ((block.qh[j].toInt() and 0xFF) or m1).toByte()
                        }
                        var l2 = l[n + j + 32]
                        if (l2 > 15) {
                            l2 -= 16
                            block.qh[j] = ((block.qh[j].toInt() and 0xFF) or m2).toByte()
                        }
                        block.qs[offset + j] = (l1 or (l2 shl 4)).toByte()
                    }
                    m1 = m1 shl 2
                    m2 = m2 shl 2
                }
            }
        }

        override fun toFloat(xs: List<BlockQ5K>, ys: FloatArray) {
            for (idx in 0 until xs.size) {
                val block = xs[idx]
                val dVal = block.d.toFloat()
                val minVal = block.dmin.toFloat()
                val scUBytes = UByteArray(block.scales.size) { block.scales[it].toUByte() }
                var isIdx = 0
                var u1 = 1
                var u2 = 2
                var ysIndex = idx * QKK

                for (j in 0 until QKK step 64) {
                    val qBase = j / 2
                    val (sc1, m1Raw) = getScaleMinK4(isIdx, scUBytes)
                    val d1 = dVal * sc1.toFloat()
                    val m1 = minVal * m1Raw.toFloat()
                    val (sc2, m2Raw) = getScaleMinK4(isIdx + 1, scUBytes)
                    val d2 = dVal * sc2.toFloat()
                    val m2 = minVal * m2Raw.toFloat()

                    for (l in 0 until 32) {
                        val qlVal = block.qs[qBase + l].toInt() and 0xFF
                        val qhVal = block.qh[l].toInt() and 0xFF
                        val toAdd = if ((qhVal and u1) != 0) 16.0f else 0.0f
                        ys[ysIndex] = d1 * ((qlVal and 0xF).toFloat() + toAdd) - m1
                        ysIndex += 1
                    }
                    for (l in 0 until 32) {
                        val qlVal = block.qs[qBase + l].toInt() and 0xFF
                        val qhVal = block.qh[l].toInt() and 0xFF
                        val toAdd = if ((qhVal and u2) != 0) 16.0f else 0.0f
                        ys[ysIndex] = d2 * ((qlVal ushr 4).toFloat() + toAdd) - m2
                        ysIndex += 1
                    }
                    isIdx += 2
                    u1 = u1 shl 2
                    u2 = u2 shl 2
                }
            }
        }
    }
}

class BlockQ6K(
    val ql: ByteArray = ByteArray(QKK / 2),
    val qh: ByteArray = ByteArray(QKK / 4),
    val scales: ByteArray = ByteArray(QKK / 16),
    var d: F16 = F16.ZERO
) {
    companion object : GgmlType<BlockQ6K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q6K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ6K = { BlockQ6K() }

        override fun vecDot(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ6kQ8k: $n is not divisible by $QKK"
            }

            val aux8 = ByteArray(QKK)
            val aux16 = ShortArray(8)
            val sums = FloatArray(8)
            val aux32 = FloatArray(8)

            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                aux32.fill(0.0f)

                for (j in 0 until QKK step 128) {
                    val q4Base = j / 2
                    val qhBase = j / 4
                    for (l in 0 until 32) {
                        val ql0 = x.ql[q4Base + l].toInt() and 0xFF
                        val ql32 = x.ql[q4Base + l + 32].toInt() and 0xFF
                        val qhVal = x.qh[qhBase + l].toInt() and 0xFF

                        aux8[j + l] = (((ql0 and 0xF) or ((qhVal and 3) shl 4)) - 32).toByte()
                        aux8[j + l + 32] = (((ql32 and 0xF) or (((qhVal ushr 2) and 3) shl 4)) - 32).toByte()
                        aux8[j + l + 64] = (((ql0 ushr 4) or (((qhVal ushr 4) and 3) shl 4)) - 32).toByte()
                        aux8[j + l + 96] = (((ql32 ushr 4) or (((qhVal ushr 6) and 3) shl 4)) - 32).toByte()
                    }
                }

                for (j in 0 until x.scales.size) {
                    val scale = x.scales[j].toFloat()
                    val base = 16 * j
                    for (l in 0 until 8) {
                        aux16[l] = (y.qs[base + l].toInt() * aux8[base + l].toInt()).toShort()
                    }
                    for (l in 0 until 8) {
                        aux32[l] += scale * aux16[l].toFloat()
                    }
                    for (l in 0 until 8) {
                        aux16[l] = (y.qs[base + 8 + l].toInt() * aux8[base + 8 + l].toInt()).toShort()
                    }
                    for (l in 0 until 8) {
                        aux32[l] += scale * aux16[l].toFloat()
                    }
                }

                val dVal = x.d.toFloat() * y.d
                for (l in 0 until 8) {
                    sums[l] += aux32[l] * dVal
                }
            }

            var total = 0.0f
            for (s in sums) total += s
            return total
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ6K>) {
            require(xs.size == ys.size * BLCK_SIZE) {
                "quantizeRowQ6k: size mismatch ${xs.size} ${ys.size} $BLCK_SIZE"
            }
            val l = ByteArray(QKK)
            val scales = FloatArray(QKK / 16)
            val emptyWeights = FloatArray(0)

            for (blkIdx in 0 until ys.size) {
                val y = ys[blkIdx]
                val xBase = blkIdx * QKK
                var maxScale = 0.0f
                var maxAbsScale = 0.0f

                for (ib in 0 until scales.size) {
                    val xSlice = xs.copyOfRange(xBase + 16 * ib, xBase + 16 * (ib + 1))
                    val lSlice = ByteArray(16)
                    val scale = makeQxQuants(16, 32, xSlice, lSlice, 1, emptyWeights)
                    lSlice.copyInto(l, 16 * ib)
                    scales[ib] = scale
                    val absScale = abs(scale)
                    if (absScale > maxAbsScale) {
                        maxAbsScale = absScale
                        maxScale = scale
                    }
                }

                val iscale = -128.0f / maxScale
                y.d = F16.fromFloat(1.0f / iscale)

                for (j in 0 until y.scales.size) {
                    y.scales[j] = min(127, nearestInt(iscale * scales[j])).toByte()
                }

                for (j in 0 until y.scales.size) {
                    val dVal = y.d.toFloat() * y.scales[j].toFloat()
                    if (dVal == 0.0f) continue
                    for (ii in 0 until 16) {
                        val ll = nearestInt(xs[xBase + 16 * j + ii] / dVal).coerceIn(-32, 31)
                        l[16 * j + ii] = (ll + 32).toByte()
                    }
                }

                var qlOffset = 0
                var qhOffset = 0
                for (j in 0 until QKK step 128) {
                    for (lIdx in 0 until 32) {
                        val v1 = l[j + lIdx].toInt() and 0xFF
                        val v2 = l[j + lIdx + 32].toInt() and 0xFF
                        val v3 = l[j + lIdx + 64].toInt() and 0xFF
                        val v4 = l[j + lIdx + 96].toInt() and 0xFF
                        val q1 = v1 and 0xF
                        val q2 = v2 and 0xF
                        val q3 = v3 and 0xF
                        val q4 = v4 and 0xF
                        y.ql[qlOffset + lIdx] = (q1 or (q3 shl 4)).toByte()
                        y.ql[qlOffset + lIdx + 32] = (q2 or (q4 shl 4)).toByte()
                        y.qh[qhOffset + lIdx] = (
                            (v1 ushr 4) or
                                ((v2 ushr 4) shl 2) or
                                ((v3 ushr 4) shl 4) or
                                ((v4 ushr 4) shl 6)
                            ).toByte()
                    }
                    qlOffset += 64
                    qhOffset += 32
                }
            }
        }

        override fun fromFloatImatrix(
            xs: FloatArray,
            ys: MutableList<BlockQ6K>,
            imatrixWeights: FloatArray,
            nPerRow: Int
        ) {
            require(xs.size == ys.size * BLCK_SIZE) {
                "quantizeRowQ6k imatrix: size mismatch ${xs.size} ${ys.size} $BLCK_SIZE"
            }
            val l = ByteArray(QKK)
            val scales = FloatArray(QKK / 16)

            for (sblkIdx in 0 until ys.size) {
                val y = ys[sblkIdx]
                val xBase = sblkIdx * QKK
                var maxScale = 0.0f
                var maxAbsScale = 0.0f

                for (ib in 0 until scales.size) {
                    val imatrixRow = sblkIdx % (nPerRow / QKK)
                    val xSlice = xs.copyOfRange(xBase + 16 * ib, xBase + 16 * (ib + 1))
                    val wSlice = imatrixWeights.copyOfRange(
                        QKK * imatrixRow + 16 * ib,
                        QKK * imatrixRow + 16 * (ib + 1)
                    )
                    val lSlice = ByteArray(16)
                    val scale = makeQxQuants(16, 32, xSlice, lSlice, 1, wSlice)
                    lSlice.copyInto(l, 16 * ib)
                    scales[ib] = scale
                    val absScale = abs(scale)
                    if (absScale > maxAbsScale) {
                        maxAbsScale = absScale
                        maxScale = scale
                    }
                }

                val iscale = -128.0f / maxScale
                y.d = F16.fromFloat(1.0f / iscale)

                for (j in 0 until y.scales.size) {
                    y.scales[j] = min(127, nearestInt(iscale * scales[j])).toByte()
                }

                for (j in 0 until y.scales.size) {
                    val dVal = y.d.toFloat() * y.scales[j].toFloat()
                    if (dVal == 0.0f) continue
                    for (ii in 0 until 16) {
                        val ll = nearestInt(xs[xBase + 16 * j + ii] / dVal).coerceIn(-32, 31)
                        l[16 * j + ii] = (ll + 32).toByte()
                    }
                }

                var qlOffset = 0
                var qhOffset = 0
                for (j in 0 until QKK step 128) {
                    for (lIdx in 0 until 32) {
                        val v1 = l[j + lIdx].toInt() and 0xFF
                        val v2 = l[j + lIdx + 32].toInt() and 0xFF
                        val v3 = l[j + lIdx + 64].toInt() and 0xFF
                        val v4 = l[j + lIdx + 96].toInt() and 0xFF
                        val q1 = v1 and 0xF
                        val q2 = v2 and 0xF
                        val q3 = v3 and 0xF
                        val q4 = v4 and 0xF
                        y.ql[qlOffset + lIdx] = (q1 or (q3 shl 4)).toByte()
                        y.ql[qlOffset + lIdx + 32] = (q2 or (q4 shl 4)).toByte()
                        y.qh[qhOffset + lIdx] = (
                            (v1 ushr 4) or
                                ((v2 ushr 4) shl 2) or
                                ((v3 ushr 4) shl 4) or
                                ((v4 ushr 4) shl 6)
                            ).toByte()
                    }
                    qlOffset += 64
                    qhOffset += 32
                }
            }
        }

        override fun toFloat(xs: List<BlockQ6K>, ys: FloatArray) {
            val k = ys.size
            require(k % QKK == 0) {
                "dequantizeRowQ6k: $k is not divisible by $QKK"
            }

            for (idxX in 0 until xs.size) {
                val x = xs[idxX]
                val dVal = x.d.toFloat()
                val ql = x.ql
                val qh = x.qh
                val sc = x.scales
                for (n in 0 until QKK step 128) {
                    val idx = n / 128
                    val yBase = idxX * QKK + n
                    val scBase = 8 * idx
                    val qlBase = 64 * idx
                    val qhBase = 32 * idx
                    for (l in 0 until 32) {
                        val isIdx = l / 16
                        val ql0 = ql[qlBase + l].toInt() and 0xFF
                        val ql32 = ql[qlBase + l + 32].toInt() and 0xFF
                        val qhVal = qh[qhBase + l].toInt() and 0xFF

                        val q1 = ((ql0 and 0xF) or ((qhVal and 3) shl 4)) - 32
                        val q2 = ((ql32 and 0xF) or (((qhVal ushr 2) and 3) shl 4)) - 32
                        val q3 = ((ql0 ushr 4) or (((qhVal ushr 4) and 3) shl 4)) - 32
                        val q4 = ((ql32 ushr 4) or (((qhVal ushr 6) and 3) shl 4)) - 32

                        ys[yBase + l] = dVal * sc[scBase + isIdx].toFloat() * q1.toFloat()
                        ys[yBase + l + 32] = dVal * sc[scBase + isIdx + 2].toFloat() * q2.toFloat()
                        ys[yBase + l + 64] = dVal * sc[scBase + isIdx + 4].toFloat() * q3.toFloat()
                        ys[yBase + l + 96] = dVal * sc[scBase + isIdx + 6].toFloat() * q4.toFloat()
                    }
                }
            }
        }
    }
}

class BlockQ8K(
    var d: Float = 0.0f,
    val qs: ByteArray = ByteArray(QKK),
    val bsums: ShortArray = ShortArray(QKK / 16)
) {
    companion object : GgmlType<BlockQ8K, BlockQ8K> {
        override val DTYPE: GgmlDType = GgmlDType.Q8K
        override val BLCK_SIZE: Int = QKK
        override val vecDotCompanion: GgmlType<BlockQ8K, *>
            get() = BlockQ8K
        override val createZero: () -> BlockQ8K = { BlockQ8K() }

        override fun vecDot(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
            return vecDotUnopt(n, xs, ys)
        }

        override fun vecDotUnopt(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
            require(n % QKK == 0) {
                "vecDotQ8kQ8k: $n is not divisible by $QKK"
            }
            var sumf = 0.0f
            val count = min(xs.size, ys.size)
            for (idx in 0 until count) {
                val x = xs[idx]
                val y = ys[idx]
                var sumI = 0
                for (j in 0 until QKK) {
                    sumI += x.qs[j].toInt() * y.qs[j].toInt()
                }
                sumf += sumI.toFloat() * x.d * y.d
            }
            return sumf
        }

        override fun fromFloat(xs: FloatArray, ys: MutableList<BlockQ8K>) {
            val k = xs.size
            require(k % QKK == 0) {
                "quantizeRowQ8k: $k is not divisible by $QKK"
            }
            for (i in 0 until ys.size) {
                val y = ys[i]
                var maxVal = 0.0f
                var amax = 0.0f
                val base = i * QKK
                for (j in 0 until QKK) {
                    val x = xs[base + j]
                    val absX = abs(x)
                    if (amax < absX) {
                        amax = absX
                        maxVal = x
                    }
                }
                if (amax == 0.0f) {
                    y.d = 0.0f
                    y.qs.fill(0)
                } else {
                    val iscale = -128.0f / maxVal
                    for (j in 0 until QKK) {
                        val v = round(iscale * xs[base + j])
                        y.qs[j] = min(127.0f, v).toInt().toByte()
                    }
                    for (j in 0 until QKK / 16) {
                        var sum = 0
                        for (ii in 0 until 16) {
                            sum += y.qs[j * 16 + ii].toInt()
                        }
                        y.bsums[j] = sum.toShort()
                    }
                    y.d = 1.0f / iscale
                }
            }
        }

        override fun toFloat(xs: List<BlockQ8K>, ys: FloatArray) {
            val k = ys.size
            require(k % QKK == 0) {
                "dequantizeRowQ8k: $k is not divisible by $QKK"
            }
            for (i in 0 until xs.size) {
                val x = xs[i]
                for (j in 0 until QKK) {
                    ys[i * QKK + j] = x.d * x.qs[j].toFloat()
                }
            }
        }
    }
}

fun <T, V> matmul(
    mkn: Triple<Int, Int, Int>,
    lhs: FloatArray,
    rhsT: List<T>,
    dst: FloatArray,
    companion: GgmlType<T, V>
): CandleResult<Unit> {
    val (m, k, n) = mkn
    val vecDotComp = companion.vecDotCompanion
    require(companion.BLCK_SIZE == vecDotComp.BLCK_SIZE) {
        "Mismatched block sizes"
    }
    require(m * k == lhs.size) {
        "unexpected lhs length ${lhs.size} ($m,$k,$n)"
    }
    val kInBlocks = (k + companion.BLCK_SIZE - 1) / companion.BLCK_SIZE
    val lhsB = MutableList(m * kInBlocks) { vecDotComp.zeros() }

    if (companion.DIRECT_COPY) {
        vecDotComp.directCopy(lhs, lhsB)
    } else {
        for (rowIdx in 0 until m) {
            val rowBlocks = lhsB.subList(rowIdx * kInBlocks, (rowIdx + 1) * kInBlocks)
            val lhsSlice = lhs.copyOfRange(rowIdx * k, (rowIdx + 1) * k)
            vecDotComp.fromFloat(lhsSlice, rowBlocks)
        }
    }

    for (rowIdx in 0 until m) {
        val lhsRow = lhsB.subList(rowIdx * kInBlocks, (rowIdx + 1) * kInBlocks)
        val dstBase = rowIdx * n
        for (colIdx in 0 until n) {
            val rhsCol = rhsT.subList(colIdx * kInBlocks, (colIdx + 1) * kInBlocks)
            dst[dstBase + colIdx] = companion.vecDot(k, rhsCol, lhsRow)
        }
    }
    return CandleResult.Ok(Unit)
}

fun <T, V> matmulF16(
    mkn: Triple<Int, Int, Int>,
    lhs: Array<F16>,
    rhsT: List<T>,
    dst: Array<F16>,
    companion: GgmlType<T, V>
): CandleResult<Unit> {
    val (m, k, n) = mkn
    if (m * k != lhs.size) {
        bail("unexpected lhs length ${lhs.size} $mkn")
    }

    val vecDotComp = companion.vecDotCompanion
    val kInLhsBlocks = (k + companion.BLCK_SIZE - 1) / companion.BLCK_SIZE
    val kInRhsBlocks = (k + vecDotComp.BLCK_SIZE - 1) / vecDotComp.BLCK_SIZE
    val lhsB = MutableList(m * kInLhsBlocks) { vecDotComp.zeros() }
    for (rowIdx in 0 until m) {
        val rowBlocks = lhsB.subList(rowIdx * kInLhsBlocks, (rowIdx + 1) * kInLhsBlocks)
        val lhsF32 = FloatArray(k) { idx -> lhs[rowIdx * k + idx].toFloat() }
        vecDotComp.fromFloat(lhsF32, rowBlocks)
    }

    for (rowIdx in 0 until m) {
        val lhsRow = lhsB.subList(rowIdx * kInLhsBlocks, (rowIdx + 1) * kInLhsBlocks)
        val dstBase = rowIdx * n
        for (colIdx in 0 until n) {
            val rhsCol = rhsT.subList(colIdx * kInRhsBlocks, (colIdx + 1) * kInRhsBlocks)
            val value = companion.vecDot(k, rhsCol, lhsRow)
            dst[dstBase + colIdx] = F16.fromFloat(value)
        }
    }
    return CandleResult.Ok(Unit)
}

public typealias U8Scalar = UByte
public typealias U32Scalar = UInt
public typealias I16Scalar = Short
public typealias I32Scalar = Int
public typealias I64Scalar = Long
public typealias F32Scalar = Float
public typealias F64Scalar = Double

internal fun Float.Companion.zeros(): Float = F32GgmlType.zeros()
internal fun Float.Companion.fromBytes(data: ByteArray, offset: Int): Float = F32GgmlType.fromBytes(data, offset)
internal fun Float.Companion.toBytes(list: List<Float>): ByteArray = F32GgmlType.toBytes(list)

object F32GgmlType : GgmlType<Float, Float> {
    override val DTYPE: GgmlDType = GgmlDType.F32
    override val BLCK_SIZE: Int = 1
    override val DIRECT_COPY: Boolean = true
    override val vecDotCompanion: GgmlType<Float, *>
        get() = F32GgmlType
    override val createZero: () -> Float = { 0.0f }

    override fun vecDot(n: Int, xs: List<Float>, ys: List<Float>): Float {
        return vecDotUnopt(n, xs, ys)
    }

    override fun vecDotUnopt(n: Int, xs: List<Float>, ys: List<Float>): Float {
        require(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
        require(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
        var res = 0.0f
        for (i in 0 until n) {
            res += xs[i] * ys[i]
        }
        return res
    }

    override fun fromFloat(xs: FloatArray, ys: MutableList<Float>) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = xs[i]
        }
    }

    override fun toFloat(xs: List<Float>, ys: FloatArray) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = xs[i]
        }
    }

    override fun directCopy(xs: FloatArray, ys: MutableList<Float>) {
        fromFloat(xs, ys)
    }
}

object F16GgmlType : GgmlType<F16, F16> {
    override val DTYPE: GgmlDType = GgmlDType.F16
    override val BLCK_SIZE: Int = 1
    override val DIRECT_COPY: Boolean = true
    override val vecDotCompanion: GgmlType<F16, *>
        get() = F16GgmlType
    override val createZero: () -> F16 = { F16.ZERO }

    override fun vecDot(n: Int, xs: List<F16>, ys: List<F16>): Float {
        return vecDotUnopt(n, xs, ys)
    }

    override fun vecDotUnopt(n: Int, xs: List<F16>, ys: List<F16>): Float {
        require(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
        require(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
        var res = 0.0f
        for (i in 0 until n) {
            res += xs[i].toFloat() * ys[i].toFloat()
        }
        return res
    }

    override fun fromFloat(xs: FloatArray, ys: MutableList<F16>) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = F16.fromFloat(xs[i])
        }
    }

    override fun toFloat(xs: List<F16>, ys: FloatArray) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = xs[i].toFloat()
        }
    }

    override fun directCopy(xs: FloatArray, ys: MutableList<F16>) {
        fromFloat(xs, ys)
    }
}

object BF16GgmlType : GgmlType<BF16, BF16> {
    override val DTYPE: GgmlDType = GgmlDType.BF16
    override val BLCK_SIZE: Int = 1
    override val DIRECT_COPY: Boolean = true
    override val vecDotCompanion: GgmlType<BF16, *>
        get() = BF16GgmlType
    override val createZero: () -> BF16 = { BF16.ZERO }

    override fun vecDot(n: Int, xs: List<BF16>, ys: List<BF16>): Float {
        return vecDotUnopt(n, xs, ys)
    }

    override fun vecDotUnopt(n: Int, xs: List<BF16>, ys: List<BF16>): Float {
        require(xs.size >= n) { "size mismatch xs ${xs.size} < $n" }
        require(ys.size >= n) { "size mismatch ys ${ys.size} < $n" }
        var res = 0.0f
        for (i in 0 until n) {
            res += xs[i].toFloat() * ys[i].toFloat()
        }
        return res
    }

    override fun fromFloat(xs: FloatArray, ys: MutableList<BF16>) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = BF16.fromFloat(xs[i])
        }
    }

    override fun toFloat(xs: List<BF16>, ys: FloatArray) {
        require(xs.size == ys.size) {
            "size mismatch xs ${xs.size} != ys ${ys.size}"
        }
        for (i in 0 until xs.size) {
            ys[i] = xs[i].toFloat()
        }
    }

    override fun directCopy(xs: FloatArray, ys: MutableList<BF16>) {
        fromFloat(xs, ys)
    }
}
