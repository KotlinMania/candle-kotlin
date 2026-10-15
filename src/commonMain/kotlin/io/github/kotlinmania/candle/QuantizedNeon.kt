// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ8K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.QK80
import io.github.kotlinmania.candle.QKK

private fun vdotqS32(a: ByteArray, aOffset: Int, b: ByteArray, bOffset: Int): IntArray {
    val result = IntArray(4)
    for (lane in 0 until 4) {
        val low0 = a[aOffset + lane * 2].toInt() * b[bOffset + lane * 2].toInt()
        val low1 = a[aOffset + lane * 2 + 1].toInt() * b[bOffset + lane * 2 + 1].toInt()
        val high0 = a[aOffset + 8 + lane * 2].toInt() * b[bOffset + 8 + lane * 2].toInt()
        val high1 = a[aOffset + 8 + lane * 2 + 1].toInt() * b[bOffset + 8 + lane * 2 + 1].toInt()
        result[lane] = (low0 + low1) + (high0 + high1)
    }
    return result
}

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    val nb = n / QK80
    val sumv0 = FloatArray(4)
    for (i in 0 until nb) {
        val x0 = xs[i]
        val y0 = ys[i]

        val v0Low = ByteArray(16)
        val v0High = ByteArray(16)
        for (k in 0 until 16) {
            val raw = x0.qs[k].toInt() and 0xFF
            v0Low[k] = ((raw and 0x0F) - 8).toByte()
            v0High[k] = ((raw ushr 4) - 8).toByte()
        }

        val pl0 = vdotqS32(v0Low, 0, y0.qs, 0)
        val ph0 = vdotqS32(v0High, 0, y0.qs, 16)
        val scale = x0.d.toFloat() * y0.d.toFloat()
        for (lane in 0 until 4) {
            sumv0[lane] += (pl0[lane] + ph0[lane]).toFloat() * scale
        }
    }
    return sumv0[0] + sumv0[1] + sumv0[2] + sumv0[3]
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    val nb = n / QK80
    val sumv0 = FloatArray(4)
    for (i in 0 until nb) {
        val x0 = xs[i]
        val y0 = ys[i]

        val p0 = vdotqS32(x0.qs, 0, y0.qs, 0)
        val p1 = vdotqS32(x0.qs, 16, y0.qs, 16)
        val scale = x0.d.toFloat() * y0.d.toFloat()
        for (lane in 0 until 4) {
            sumv0[lane] += (p0[lane] + p1[lane]).toFloat() * scale
        }
    }
    return sumv0[0] + sumv0[1] + sumv0[2] + sumv0[3]
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val sumI = IntArray(4)
        val scale = x.d * y.d
        for (i in 0 until QKK step 16) {
            val xy = vdotqS32(x.qs, i, y.qs, i)
            for (lane in 0 until 4) {
                sumI[lane] += xy[lane]
            }
        }
        val total = sumI[0] + sumI[1] + sumI[2] + sumI[3]
        sumf += total.toFloat() * scale
    }
    return sumf
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    var sum = 0.0f
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val dAll = x.d.toFloat()

        var isumMins = 0
        for (k in 0 until 16) {
            isumMins += y.bsums[k].toInt() * x.scales[k].toInt()
        }

        var isum = 0
        var q6Offset = 0
        var qhOffset = 0
        var q8Offset = 0
        var scaleOffset = 0

        for (j in 0 until QKK / 128) {
            val q6Bytes0 = ByteArray(16)
            val q6Bytes1 = ByteArray(16)
            val q6Bytes2 = ByteArray(16)
            val q6Bytes3 = ByteArray(16)

            for (k in 0 until 16) {
                val qh0 = x.qh[qhOffset + k].toInt() and 0xFF
                val qh1 = x.qh[qhOffset + 16 + k].toInt() and 0xFF
                val ql0 = x.ql[q6Offset + k].toInt() and 0xFF
                val ql1 = x.ql[q6Offset + 16 + k].toInt() and 0xFF
                val ql2 = x.ql[q6Offset + 32 + k].toInt() and 0xFF
                val ql3 = x.ql[q6Offset + 48 + k].toInt() and 0xFF

                q6Bytes0[k] = ((ql0 and 0xF) or ((qh0 and 3) shl 4)).toByte()
                q6Bytes1[k] = ((ql1 and 0xF) or ((qh1 and 3) shl 4)).toByte()
                q6Bytes2[k] = ((ql2 and 0xF) or (((qh0 ushr 2) and 3) shl 4)).toByte()
                q6Bytes3[k] = ((ql3 and 0xF) or (((qh1 ushr 2) and 3) shl 4)).toByte()
            }

            val p0 = vdotqS32(q6Bytes0, 0, y.qs, q8Offset)
            val p1 = vdotqS32(q6Bytes1, 0, y.qs, q8Offset + 16)
            isum += (p0[0] + p0[1] + p0[2] + p0[3]) * x.scales[scaleOffset].toInt() +
                (p1[0] + p1[1] + p1[2] + p1[3]) * x.scales[scaleOffset + 1].toInt()
            scaleOffset += 2

            val p2 = vdotqS32(q6Bytes2, 0, y.qs, q8Offset + 32)
            val p3 = vdotqS32(q6Bytes3, 0, y.qs, q8Offset + 48)
            isum += (p2[0] + p2[1] + p2[2] + p2[3]) * x.scales[scaleOffset].toInt() +
                (p3[0] + p3[1] + p3[2] + p3[3]) * x.scales[scaleOffset + 1].toInt()
            scaleOffset += 2
            q8Offset += 64

            for (k in 0 until 16) {
                val qh0 = x.qh[qhOffset + k].toInt() and 0xFF
                val qh1 = x.qh[qhOffset + 16 + k].toInt() and 0xFF
                val ql0 = x.ql[q6Offset + k].toInt() and 0xFF
                val ql1 = x.ql[q6Offset + 16 + k].toInt() and 0xFF
                val ql2 = x.ql[q6Offset + 32 + k].toInt() and 0xFF
                val ql3 = x.ql[q6Offset + 48 + k].toInt() and 0xFF

                q6Bytes0[k] = ((ql0 ushr 4) or (((qh0 ushr 4) and 3) shl 4)).toByte()
                q6Bytes1[k] = ((ql1 ushr 4) or (((qh1 ushr 4) and 3) shl 4)).toByte()
                q6Bytes2[k] = ((ql2 ushr 4) or (((qh0 ushr 6) and 3) shl 4)).toByte()
                q6Bytes3[k] = ((ql3 ushr 4) or (((qh1 ushr 6) and 3) shl 4)).toByte()
            }

            val p4 = vdotqS32(q6Bytes0, 0, y.qs, q8Offset)
            val p5 = vdotqS32(q6Bytes1, 0, y.qs, q8Offset + 16)
            isum += (p4[0] + p4[1] + p4[2] + p4[3]) * x.scales[scaleOffset].toInt() +
                (p5[0] + p5[1] + p5[2] + p5[3]) * x.scales[scaleOffset + 1].toInt()
            scaleOffset += 2

            val p6 = vdotqS32(q6Bytes2, 0, y.qs, q8Offset + 32)
            val p7 = vdotqS32(q6Bytes3, 0, y.qs, q8Offset + 48)
            isum += (p6[0] + p6[1] + p6[2] + p6[3]) * x.scales[scaleOffset].toInt() +
                (p7[0] + p7[1] + p7[2] + p7[3]) * x.scales[scaleOffset + 1].toInt()
            scaleOffset += 2

            qhOffset += 32
            q6Offset += 64
            q8Offset += 64
        }
        sum += dAll * y.d * (isum - 32 * isumMins).toFloat()
    }
    return sum
}

internal fun vecDotQ5kQ8k(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ5kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

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

        var sumiMins = 0
        val scales = IntArray(8)
        for (w in 0 until 2) {
            for (b in 0 until 4) {
                val scIdx = w * 4 + b
                scales[scIdx] = (utmp[w] ushr (b * 8)) and 0xFF
                val mVal = (utmp[w + 2] ushr (b * 8)) and 0xFF
                val q8s = y.bsums[scIdx * 2].toInt() + y.bsums[scIdx * 2 + 1].toInt()
                sumiMins += q8s * mVal
            }
        }

        var q5Offset = 0
        var q8Offset = 0
        var scaleIdx = 0
        var sumi = 0

        for (j in 0 until QKK / 64) {
            val shift = 2 * j
            val q5Bytes0 = ByteArray(16)
            val q5Bytes1 = ByteArray(16)
            val q5Bytes2 = ByteArray(16)
            val q5Bytes3 = ByteArray(16)

            for (k in 0 until 16) {
                val q50 = x.qs[q5Offset + k].toInt() and 0xFF
                val q51 = x.qs[q5Offset + 16 + k].toInt() and 0xFF
                val qh0 = (x.qh[k].toInt() and 0xFF) ushr shift
                val qh1 = (x.qh[16 + k].toInt() and 0xFF) ushr shift

                q5Bytes0[k] = ((q50 and 0xF) or ((qh0 and 1) shl 4)).toByte()
                q5Bytes1[k] = ((q51 and 0xF) or ((qh1 and 1) shl 4)).toByte()
                q5Bytes2[k] = ((q50 ushr 4) or ((qh0 and 2) shl 3)).toByte()
                q5Bytes3[k] = ((q51 ushr 4) or ((qh1 and 2) shl 3)).toByte()
            }

            val p0 = vdotqS32(q5Bytes0, 0, y.qs, q8Offset)
            val p1 = vdotqS32(q5Bytes1, 0, y.qs, q8Offset + 16)
            val sum01 = (p0[0] + p1[0]) + (p0[1] + p1[1]) + (p0[2] + p1[2]) + (p0[3] + p1[3])
            sumi += sum01 * scales[scaleIdx]
            scaleIdx += 1

            val p2 = vdotqS32(q5Bytes2, 0, y.qs, q8Offset + 32)
            val p3 = vdotqS32(q5Bytes3, 0, y.qs, q8Offset + 48)
            val sum23 = (p2[0] + p3[0]) + (p2[1] + p3[1]) + (p2[2] + p3[2]) + (p2[3] + p3[3])
            sumi += sum23 * scales[scaleIdx]
            scaleIdx += 1

            q5Offset += 32
            q8Offset += 64
        }
        sumf += d * sumi.toFloat() - dmin * sumiMins.toFloat()
    }
    return sumf
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ4kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val scales = IntArray(16)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

        for (w in 0 until 3) {
            val b0 = x.scales[w * 4].toInt() and 0xFF
            val b1 = x.scales[w * 4 + 1].toInt() and 0xFF
            val b2 = x.scales[w * 4 + 2].toInt() and 0xFF
            val b3 = x.scales[w * 4 + 3].toInt() and 0xFF
            utmp[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
        }

        val minWords = intArrayOf(
            utmp[1] and kmask1,
            ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
        )
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[0] = utmp[0] and kmask1

        var prodSum = 0
        for (w in 0 until 2) {
            for (b in 0 until 4) {
                val scIdx = w * 4 + b
                val mVal = (minWords[w] ushr (b * 8)) and 0xFF
                val q8s = y.bsums[scIdx * 2].toInt() + y.bsums[scIdx * 2 + 1].toInt()
                prodSum += q8s * mVal
                scales[scIdx] = (utmp[w] ushr (b * 8)) and 0xFF
            }
        }
        sumf -= dmin * prodSum.toFloat()

        var q4Offset = 0
        var q8Offset = 0
        var sumi1 = 0
        var sumi2 = 0

        for (j in 0 until QKK / 64) {
            val q4Low0 = ByteArray(16)
            val q4Low1 = ByteArray(16)
            val q4High0 = ByteArray(16)
            val q4High1 = ByteArray(16)

            for (k in 0 until 16) {
                val b0 = x.qs[q4Offset + k].toInt() and 0xFF
                val b1 = x.qs[q4Offset + 16 + k].toInt() and 0xFF
                q4Low0[k] = (b0 and 0xF).toByte()
                q4Low1[k] = (b1 and 0xF).toByte()
                q4High0[k] = (b0 ushr 4).toByte()
                q4High1[k] = (b1 ushr 4).toByte()
            }

            val p0 = vdotqS32(q4Low0, 0, y.qs, q8Offset)
            val p1 = vdotqS32(q4Low1, 0, y.qs, q8Offset + 16)
            val sum01 = (p0[0] + p1[0]) + (p0[1] + p1[1]) + (p0[2] + p1[2]) + (p0[3] + p1[3])
            sumi1 += sum01 * scales[2 * j]

            val p2 = vdotqS32(q4High0, 0, y.qs, q8Offset + 32)
            val p3 = vdotqS32(q4High1, 0, y.qs, q8Offset + 48)
            val sum23 = (p2[0] + p3[0]) + (p2[1] + p3[1]) + (p2[2] + p3[2]) + (p2[3] + p3[3])
            sumi2 += sum23 * scales[2 * j + 1]

            q4Offset += 32
            q8Offset += 64
        }
        sumf += d * (sumi1 + sumi2).toFloat()
    }
    return sumf
}

internal fun vecDotQ3kQ8k(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ3kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val aux = IntArray(3)
    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()

        for (w in 0 until 3) {
            val b0 = x.scales[w * 4].toInt() and 0xFF
            val b1 = x.scales[w * 4 + 1].toInt() and 0xFF
            val b2 = x.scales[w * 4 + 2].toInt() and 0xFF
            val b3 = x.scales[w * 4 + 3].toInt() and 0xFF
            aux[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
        }

        utmp[3] = ((aux[1] ushr 4) and kmask2) or (((aux[2] ushr 6) and kmask1) shl 4)
        utmp[2] = ((aux[0] ushr 4) and kmask2) or (((aux[2] ushr 4) and kmask1) shl 4)
        utmp[1] = (aux[1] and kmask2) or (((aux[2] ushr 2) and kmask1) shl 4)
        utmp[0] = (aux[0] and kmask2) or ((aux[2] and kmask1) shl 4)

        val scale = IntArray(16)
        for (w in 0 until 4) {
            for (b in 0 until 4) {
                scale[w * 4 + b] = ((utmp[w] ushr (b * 8)) and 0xFF) - 32
            }
        }

        var q3Offset = 0
        var q8Offset = 0
        var scaleOffset = 0
        var isum = 0

        for (j in 0 until QKK / 128) {
            val hShiftBase = j * 4
            val q3Bytes0 = ByteArray(16)
            val q3Bytes1 = ByteArray(16)
            val q3Bytes2 = ByteArray(16)
            val q3Bytes3 = ByteArray(16)

            for (k in 0 until 16) {
                val q30 = x.qs[q3Offset + k].toInt() and 0xFF
                val q31 = x.qs[q3Offset + 16 + k].toInt() and 0xFF
                val qh0 = (x.hmask[k].toInt() and 0xFF) ushr hShiftBase
                val qh1 = (x.hmask[16 + k].toInt() and 0xFF) ushr hShiftBase

                val h0 = if ((qh0 and 1) == 0) 4 else 0
                val h1 = if ((qh1 and 1) == 0) 4 else 0
                val h2 = if ((qh0 and 2) == 0) 4 else 0
                val h3 = if ((qh1 and 2) == 0) 4 else 0

                q3Bytes0[k] = ((q30 and 3) - h0).toByte()
                q3Bytes1[k] = ((q31 and 3) - h1).toByte()
                q3Bytes2[k] = (((q30 ushr 2) and 3) - h2).toByte()
                q3Bytes3[k] = (((q31 ushr 2) and 3) - h3).toByte()
            }

            val p0 = vdotqS32(q3Bytes0, 0, y.qs, q8Offset)
            val p1 = vdotqS32(q3Bytes1, 0, y.qs, q8Offset + 16)
            val p2 = vdotqS32(q3Bytes2, 0, y.qs, q8Offset + 32)
            val p3 = vdotqS32(q3Bytes3, 0, y.qs, q8Offset + 48)
            isum += (p0[0] + p0[1] + p0[2] + p0[3]) * scale[scaleOffset] +
                (p1[0] + p1[1] + p1[2] + p1[3]) * scale[scaleOffset + 1] +
                (p2[0] + p2[1] + p2[2] + p2[3]) * scale[scaleOffset + 2] +
                (p3[0] + p3[1] + p3[2] + p3[3]) * scale[scaleOffset + 3]
            scaleOffset += 4
            q8Offset += 64

            for (k in 0 until 16) {
                val q30 = x.qs[q3Offset + k].toInt() and 0xFF
                val q31 = x.qs[q3Offset + 16 + k].toInt() and 0xFF
                val qh0 = (x.hmask[k].toInt() and 0xFF) ushr hShiftBase
                val qh1 = (x.hmask[16 + k].toInt() and 0xFF) ushr hShiftBase

                val h0 = if ((qh0 and 4) == 0) 4 else 0
                val h1 = if ((qh1 and 4) == 0) 4 else 0
                val h2 = if ((qh0 and 8) == 0) 4 else 0
                val h3 = if ((qh1 and 8) == 0) 4 else 0

                q3Bytes0[k] = (((q30 ushr 4) and 3) - h0).toByte()
                q3Bytes1[k] = (((q31 ushr 4) and 3) - h1).toByte()
                q3Bytes2[k] = (((q30 ushr 6) and 3) - h2).toByte()
                q3Bytes3[k] = (((q31 ushr 6) and 3) - h3).toByte()
            }

            val p4 = vdotqS32(q3Bytes0, 0, y.qs, q8Offset)
            val p5 = vdotqS32(q3Bytes1, 0, y.qs, q8Offset + 16)
            val p6 = vdotqS32(q3Bytes2, 0, y.qs, q8Offset + 32)
            val p7 = vdotqS32(q3Bytes3, 0, y.qs, q8Offset + 48)
            isum += (p4[0] + p4[1] + p4[2] + p4[3]) * scale[scaleOffset] +
                (p5[0] + p5[1] + p5[2] + p5[3]) * scale[scaleOffset + 1] +
                (p6[0] + p6[1] + p6[2] + p6[3]) * scale[scaleOffset + 2] +
                (p7[0] + p7[1] + p7[2] + p7[3]) * scale[scaleOffset + 3]
            scaleOffset += 4
            q8Offset += 64
            q3Offset += 32
        }
        sumf += d * isum.toFloat()
    }
    return sumf
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val aux = UByteArray(16)
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        var minProdSum = 0
        for (k in 0 until 16) {
            val scByte = x.scales[k].toInt() and 0xFF
            aux[k] = (scByte and 0xF).toUByte()
            val minVal = scByte ushr 4
            minProdSum += minVal * y.bsums[k].toInt()
        }
        sumf += dmin * minProdSum.toFloat()

        var isum = 0
        var isIdx = 0
        var q2Offset = 0
        var q8Offset = 0

        for (j in 0 until QKK / 128) {
            val q2Bytes = ByteArray(32)
            for (step in 0 until 4) {
                val shift = step * 2
                for (k in 0 until 32) {
                    q2Bytes[k] = (((x.qs[q2Offset + k].toInt() and 0xFF) ushr shift) and 3).toByte()
                }
                isum += multiplyAccumWithScale(aux, isIdx, step * 2, q2Bytes, y.qs, q8Offset)
                q8Offset += 32
            }
            q2Offset += 32
            isIdx += 8
        }
        sumf += d * isum.toFloat()
    }
    return sumf
}

private fun multiplyAccumWithScale(
    aux: UByteArray,
    isIdx: Int,
    index: Int,
    q2bytes: ByteArray,
    q8bytes: ByteArray,
    q8Offset: Int
): Int {
    val p1 = vdotqS32(q2bytes, 0, q8bytes, q8Offset)
    val p2 = vdotqS32(q2bytes, 16, q8bytes, q8Offset + 16)
    val sum1 = p1[0] + p1[1] + p1[2] + p1[3]
    val sum2 = p2[0] + p2[1] + p2[2] + p2[3]
    return sum1 * aux[isIdx + index].toInt() + sum2 * aux[isIdx + 1 + index].toInt()
}
