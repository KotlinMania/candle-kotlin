// port-lint: source quantized/avx.rs
package io.github.kotlinmania.candle.quantized.avx

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

internal fun sumI16PairsFloat(x: ShortArray): FloatArray {
    val result = FloatArray(8)
    for (i in 0 until 8) {
        result[i] = (x[i * 2].toInt() + x[i * 2 + 1].toInt()).toFloat()
    }
    return result
}

internal fun mulSumUs8PairsFloat(ax: ByteArray, sy: ByteArray): FloatArray {
    val dot = ShortArray(16)
    for (i in 0 until 16) {
        val u0 = ax[i * 2].toInt() and 0xFF
        val u1 = ax[i * 2 + 1].toInt() and 0xFF
        val s0 = sy[i * 2].toInt()
        val s1 = sy[i * 2 + 1].toInt()
        val prod = (u0 * s0 + u1 * s1).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
        dot[i] = prod.toShort()
    }
    return sumI16PairsFloat(dot)
}

internal fun hsumFloat8(x: FloatArray): Float {
    val low = FloatArray(4) { i -> x[i] + x[i + 4] }
    val half = FloatArray(2) { i -> low[i] + low[i + 2] }
    return half[0] + half[1]
}

internal fun bytesFromNibbles32(rsi: ByteArray): ByteArray {
    val result = ByteArray(32)
    for (i in 0 until 16) {
        val b = rsi[i].toInt() and 0xFF
        result[i] = (b and 0x0F).toByte()
        result[i + 16] = ((b ushr 4) and 0x0F).toByte()
    }
    return result
}

internal fun mulSumI8PairsFloat(x: ByteArray, y: ByteArray): FloatArray {
    val ax = ByteArray(32)
    val sy = ByteArray(32)
    for (i in 0 until 32) {
        val xv = x[i].toInt()
        val yv = y[i].toInt()
        if (xv < 0) {
            ax[i] = (-xv).toByte()
            sy[i] = (-yv).toByte()
        } else if (xv == 0) {
            ax[i] = 0
            sy[i] = 0
        } else {
            ax[i] = xv.toByte()
            sy[i] = yv.toByte()
        }
    }
    return mulSumUs8PairsFloat(ax, sy)
}

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    val acc = FloatArray(8)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = x.d.toFloat() * y.d.toFloat()
        val bx = bytesFromNibbles32(x.qs)
        for (i in 0 until 32) {
            bx[i] = (bx[i].toInt() - 8).toByte()
        }
        val q = mulSumI8PairsFloat(bx, y.qs)
        for (lane in 0 until 8) {
            acc[lane] += d * q[lane]
        }
    }
    return hsumFloat8(acc)
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    val acc = FloatArray(8)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = x.d.toFloat() * y.d.toFloat()
        val q = mulSumI8PairsFloat(x.qs, y.qs)
        for (lane in 0 until 8) {
            acc[lane] += d * q[lane]
        }
    }
    return hsumFloat8(acc)
}

private fun getScaleShuffle(i: Int): ByteArray {
    val result = ByteArray(16)
    val base = i * 2
    for (k in 0 until 8) {
        result[k] = base.toByte()
        result[k + 8] = (base + 1).toByte()
    }
    return result
}

private fun getScaleShuffleK4(i: Int): ByteArray {
    val result = ByteArray(32)
    val base = i * 2
    for (k in 0 until 32 step 2) {
        result[k] = base.toByte()
        result[k + 1] = (base + 1).toByte()
    }
    return result
}

private fun getScaleShuffleQ3k(i: Int): ByteArray {
    val result = ByteArray(32)
    val base0 = i * 4
    val base1 = i * 4 + 2
    for (k in 0 until 16 step 2) {
        result[k] = base0.toByte()
        result[k + 1] = (base0 + 1).toByte()
        result[16 + k] = base1.toByte()
        result[16 + k + 1] = (base1 + 1).toByte()
    }
    return result
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    val acc = FloatArray(8)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        var q4Offset = 0
        var qhOffset = 0
        var q8Offset = 0
        val sumi = IntArray(8)

        for (j in 0 until QKK / 128) {
            val isIdx = j * 4
            val shuf0 = getScaleShuffle(isIdx)
            val shuf1 = getScaleShuffle(isIdx + 1)
            val shuf2 = getScaleShuffle(isIdx + 2)
            val shuf3 = getScaleShuffle(isIdx + 3)

            val scale0 = ShortArray(16) { k -> x.scales[shuf0[k].toInt() and 0xF].toShort() }
            val scale1 = ShortArray(16) { k -> x.scales[shuf1[k].toInt() and 0xF].toShort() }
            val scale2 = ShortArray(16) { k -> x.scales[shuf2[k].toInt() and 0xF].toShort() }
            val scale3 = ShortArray(16) { k -> x.scales[shuf3[k].toInt() and 0xF].toShort() }

            for (lane in 0 until 8) {
                var laneSum = 0
                for (pair in 0 until 2) {
                    val elem = lane * 4 + pair * 2
                    val ql1A = x.ql[q4Offset + elem].toInt() and 0xFF
                    val ql1B = x.ql[q4Offset + elem + 1].toInt() and 0xFF
                    val ql2A = x.ql[q4Offset + 32 + elem].toInt() and 0xFF
                    val ql2B = x.ql[q4Offset + 32 + elem + 1].toInt() and 0xFF
                    val qhA = x.qh[qhOffset + elem].toInt() and 0xFF
                    val qhB = x.qh[qhOffset + elem + 1].toInt() and 0xFF

                    val q40A = ((ql1A and 0xF) or ((qhA and 3) shl 4)) - 32
                    val q40B = ((ql1B and 0xF) or ((qhB and 3) shl 4)) - 32
                    val q41A = ((ql2A and 0xF) or (((qhA ushr 2) and 3) shl 4)) - 32
                    val q41B = ((ql2B and 0xF) or (((qhB ushr 2) and 3) shl 4)) - 32
                    val q42A = ((ql1A ushr 4) or (((qhA ushr 4) and 3) shl 4)) - 32
                    val q42B = ((ql1B ushr 4) or (((qhB ushr 4) and 3) shl 4)) - 32
                    val q43A = ((ql2A ushr 4) or (((qhA ushr 6) and 3) shl 4)) - 32
                    val q43B = ((ql2B ushr 4) or (((qhB ushr 6) and 3) shl 4)) - 32

                    val p0 = q40A * y.qs[q8Offset + elem].toInt() + q40B * y.qs[q8Offset + elem + 1].toInt()
                    val p1 = q41A * y.qs[q8Offset + 32 + elem].toInt() + q41B * y.qs[q8Offset + 32 + elem + 1].toInt()
                    val p2 = q42A * y.qs[q8Offset + 64 + elem].toInt() + q42B * y.qs[q8Offset + 64 + elem + 1].toInt()
                    val p3 = q43A * y.qs[q8Offset + 96 + elem].toInt() + q43B * y.qs[q8Offset + 96 + elem + 1].toInt()

                    val scIdx = lane * 2 + pair
                    laneSum += scale0[scIdx].toInt() * p0 +
                        scale1[scIdx].toInt() * p1 +
                        scale2[scIdx].toInt() * p2 +
                        scale3[scIdx].toInt() * p3
                }
                sumi[lane] += laneSum
            }
            q4Offset += 64
            qhOffset += 32
            q8Offset += 128
        }
        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }
    return hsumFloat8(acc)
}

private fun mm256SetM128i(a: ByteArray, b: ByteArray): ByteArray {
    val result = ByteArray(32)
    for (i in 0 until 16) {
        result[i] = b[i]
        result[i + 16] = a[i]
    }
    return result
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    val acc = FloatArray(8)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        for (lane in 0 until 8) {
            val m0 = (x.scales[lane * 2].toInt() and 0xFF) ushr 4
            val m1 = (x.scales[lane * 2 + 1].toInt() and 0xFF) ushr 4
            val prod = m0 * y.bsums[lane * 2].toInt() + m1 * y.bsums[lane * 2 + 1].toInt()
            acc[lane] += dmin * prod.toFloat()
        }

        val lowScales = ByteArray(16)
        val highScales = ByteArray(16)
        for (k in 0 until 8) {
            lowScales[k * 2] = (x.scales[k].toInt() and 0xF).toByte()
            lowScales[k * 2 + 1] = 0
            highScales[k * 2] = (x.scales[k + 8].toInt() and 0xF).toByte()
            highScales[k * 2 + 1] = 0
        }
        val scales = arrayOf(
            mm256SetM128i(lowScales, lowScales),
            mm256SetM128i(highScales, highScales)
        )

        var q2Offset = 0
        var q8Offset = 0
        val sumi = IntArray(8)
        for (blockIdx in 0 until scales.size) {
            val scaleVec = scales[blockIdx]
            for (sStep in 0 until 4) {
                val shuf = getScaleShuffleQ3k(sStep)
                val shift = sStep * 2
                for (lane in 0 until 8) {
                    var laneAcc = 0
                    for (pair in 0 until 2) {
                        val elem = lane * 4 + pair * 2
                        val q2A = ((x.qs[q2Offset + elem].toInt() and 0xFF) ushr shift) and 3
                        val q2B = ((x.qs[q2Offset + elem + 1].toInt() and 0xFF) ushr shift) and 3
                        val q8A = y.qs[q8Offset + sStep * 32 + elem].toInt()
                        val q8B = y.qs[q8Offset + sStep * 32 + elem + 1].toInt()
                        val p = q2A * q8A + q2B * q8B
                        val scByteOffset = if (lane < 4) shuf[lane * 4 + pair * 2].toInt() else 16 + shuf[lane * 4 + pair * 2].toInt()
                        val sc = scaleVec[scByteOffset].toInt()
                        laneAcc += sc * p
                    }
                    sumi[lane] += laneAcc
                }
            }
            q2Offset += 32
            q8Offset += 128
        }

        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }
    return hsumFloat8(acc)
}

internal fun vecDotQ3kQ8k(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ3kQ8k: $n is not divisible by $QKK"
    }
    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f
    val aux = IntArray(3)
    val acc = FloatArray(8)
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

        val sWords = intArrayOf(
            (aux[0] and kmask2) or (((aux[2] and kmask1) shl 4)),
            (aux[1] and kmask2) or ((((aux[2] ushr 2) and kmask1) shl 4)),
            ((aux[0] ushr 4) and kmask2) or ((((aux[2] ushr 4) and kmask1) shl 4)),
            ((aux[1] ushr 4) and kmask2) or ((((aux[2] ushr 6) and kmask1) shl 4))
        )

        val scBytes = ByteArray(16)
        for (w in 0 until 4) {
            for (b in 0 until 4) {
                scBytes[w * 4 + b] = (((sWords[w] ushr (b * 8)) and 0xFF) - 32).toByte()
            }
        }

        val lowScales = ByteArray(16)
        val highScales = ByteArray(16)
        for (k in 0 until 8) {
            lowScales[k * 2] = scBytes[k]
            lowScales[k * 2 + 1] = if (scBytes[k] < 0) -1 else 0
            highScales[k * 2] = scBytes[k + 8]
            highScales[k * 2 + 1] = if (scBytes[k + 8] < 0) -1 else 0
        }
        val scales = arrayOf(
            mm256SetM128i(lowScales, lowScales),
            mm256SetM128i(highScales, highScales)
        )

        var q3Offset = 0
        var q8Offset = 0
        val sumi = IntArray(8)

        for (j in 0 until scales.size) {
            val scaleVec = scales[j]
            for (sStep in 0 until 4) {
                val shuf = getScaleShuffleQ3k(sStep)
                val shift = sStep * 2
                val hShift = if (j == 0) sStep else sStep + 4
                for (lane in 0 until 8) {
                    var laneAcc = 0
                    for (pair in 0 until 2) {
                        val elem = lane * 4 + pair * 2
                        val q3lA = ((x.qs[q3Offset + elem].toInt() and 0xFF) ushr shift) and 3
                        val q3lB = ((x.qs[q3Offset + elem + 1].toInt() and 0xFF) ushr shift) and 3
                        val q3hA = if (((x.hmask[elem].toInt() and 0xFF) and (1 shl hShift)) == 0) 4 else 0
                        val q3hB = if (((x.hmask[elem + 1].toInt() and 0xFF) and (1 shl hShift)) == 0) 4 else 0

                        val q8A = y.qs[q8Offset + sStep * 32 + elem].toInt()
                        val q8B = y.qs[q8Offset + sStep * 32 + elem + 1].toInt()
                        val p = (q3lA - q3hA) * q8A + (q3lB - q3hB) * q8B

                        val scByteOffset = if (lane < 4) shuf[lane * 4 + pair * 2].toInt() else 16 + shuf[lane * 4 + pair * 2].toInt()
                        val sc = scaleVec[scByteOffset].toInt()
                        laneAcc += sc * p
                    }
                    sumi[lane] += laneAcc
                }
            }
            q3Offset += 32
            q8Offset += 128
        }

        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }
    return hsumFloat8(acc)
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ4kQ8k: $n is not divisible by $QKK"
    }
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val acc = FloatArray(8)
    val accM = FloatArray(4)
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

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

        val scBytes = ByteArray(16)
        val minBytes = ByteArray(8)
        for (w in 0 until 2) {
            for (b in 0 until 4) {
                scBytes[(w * 4 + b) * 2] = ((utmp[w] ushr (b * 8)) and 0xFF).toByte()
                minBytes[w * 4 + b] = ((utmp[w + 2] ushr (b * 8)) and 0xFF).toByte()
            }
        }

        for (lane in 0 until 4) {
            val q8s0 = y.bsums[lane * 4].toInt() + y.bsums[lane * 4 + 1].toInt()
            val q8s1 = y.bsums[lane * 4 + 2].toInt() + y.bsums[lane * 4 + 3].toInt()
            val m0 = minBytes[lane * 2].toInt() and 0xFF
            val m1 = minBytes[lane * 2 + 1].toInt() and 0xFF
            accM[lane] += dmin * (m0 * q8s0 + m1 * q8s1).toFloat()
        }

        val scales = mm256SetM128i(scBytes, scBytes)
        var q4Offset = 0
        var q8Offset = 0
        val sumi = IntArray(8)

        for (j in 0 until QKK / 64) {
            val shufL = getScaleShuffleK4(2 * j)
            val shufH = getScaleShuffleK4(2 * j + 1)

            for (lane in 0 until 8) {
                var prodL = 0
                var prodH = 0
                for (pair in 0 until 2) {
                    val elem = lane * 4 + pair * 2
                    val b0 = x.qs[q4Offset + elem].toInt() and 0xFF
                    val b1 = x.qs[q4Offset + elem + 1].toInt() and 0xFF
                    val q8l0 = y.qs[q8Offset + elem].toInt()
                    val q8l1 = y.qs[q8Offset + elem + 1].toInt()
                    val q8h0 = y.qs[q8Offset + 32 + elem].toInt()
                    val q8h1 = y.qs[q8Offset + 32 + elem + 1].toInt()

                    val pL = (b0 and 0xF) * q8l0 + (b1 and 0xF) * q8l1
                    val pH = (b0 ushr 4) * q8h0 + (b1 ushr 4) * q8h1

                    val scL = scales[shufL[elem].toInt() and 0xFF].toInt() and 0xFF
                    val scH = scales[shufH[elem].toInt() and 0xFF].toInt() and 0xFF

                    prodL += scL * pL
                    prodH += scH * pH
                }
                sumi[lane] += prodL + prodH
            }
            q4Offset += 32
            q8Offset += 64
        }

        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }

    val sumM = accM[0] + accM[1] + accM[2] + accM[3]
    return hsumFloat8(acc) + sumM
}

internal fun vecDotQ5kQ8k(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ5kQ8k: $n is not divisible by $QKK"
    }
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val acc = FloatArray(8)
    var summs = 0.0f
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

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

        val scBytes = ByteArray(16)
        var minProdSum = 0
        for (w in 0 until 2) {
            for (b in 0 until 4) {
                val scIdx = w * 4 + b
                scBytes[scIdx * 2] = ((utmp[w] ushr (b * 8)) and 0xFF).toByte()
                val mVal = (utmp[w + 2] ushr (b * 8)) and 0xFF
                val q8s = y.bsums[scIdx * 2].toInt() + y.bsums[scIdx * 2 + 1].toInt()
                minProdSum += mVal * q8s
            }
        }
        summs += dmin * minProdSum.toFloat()

        val scales = mm256SetM128i(scBytes, scBytes)
        var q5Offset = 0
        var q8Offset = 0
        val sumi = IntArray(8)

        for (j in 0 until QKK / 64) {
            val shuf0 = getScaleShuffleK4(2 * j)
            val shuf1 = getScaleShuffleK4(2 * j + 1)
            val bit0 = 2 * j
            val bit1 = 2 * j + 1

            for (lane in 0 until 8) {
                var prod0 = 0
                var prod1 = 0
                for (pair in 0 until 2) {
                    val elem = lane * 4 + pair * 2
                    val b0 = x.qs[q5Offset + elem].toInt() and 0xFF
                    val b1 = x.qs[q5Offset + elem + 1].toInt() and 0xFF
                    val qh0 = x.qh[elem].toInt() and 0xFF
                    val qh1 = x.qh[elem + 1].toInt() and 0xFF

                    val q50A = (b0 and 0xF) + (((qh0 ushr bit0) and 1) shl 4)
                    val q50B = (b1 and 0xF) + (((qh1 ushr bit0) and 1) shl 4)
                    val q51A = (b0 ushr 4) + (((qh0 ushr bit1) and 1) shl 4)
                    val q51B = (b1 ushr 4) + (((qh1 ushr bit1) and 1) shl 4)

                    val q80A = y.qs[q8Offset + elem].toInt()
                    val q80B = y.qs[q8Offset + elem + 1].toInt()
                    val q81A = y.qs[q8Offset + 32 + elem].toInt()
                    val q81B = y.qs[q8Offset + 32 + elem + 1].toInt()

                    val p0 = q50A * q80A + q50B * q80B
                    val p1 = q51A * q81A + q51B * q81B

                    val sc0 = scales[shuf0[elem].toInt() and 0xFF].toInt() and 0xFF
                    val sc1 = scales[shuf1[elem].toInt() and 0xFF].toInt() and 0xFF

                    prod0 += sc0 * p0
                    prod1 += sc1 * p1
                }
                sumi[lane] += prod0 + prod1
            }
            q5Offset += 32
            q8Offset += 64
        }

        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }
    return hsumFloat8(acc) + summs
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    val acc = FloatArray(8)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val sumi = IntArray(8)
        for (j in 0 until QKK step 32) {
            for (lane in 0 until 8) {
                val x0 = x.qs[j + lane * 2].toInt()
                val x1 = x.qs[j + lane * 2 + 1].toInt()
                val y0 = y.qs[j + lane * 2].toInt()
                val y1 = y.qs[j + lane * 2 + 1].toInt()
                sumi[lane] += x0 * y0 + x1 * y1

                val x2 = x.qs[j + 16 + lane * 2].toInt()
                val x3 = x.qs[j + 16 + lane * 2 + 1].toInt()
                val y2 = y.qs[j + 16 + lane * 2].toInt()
                val y3 = y.qs[j + 16 + lane * 2 + 1].toInt()
                sumi[lane] += x2 * y2 + x3 * y3
            }
        }
        val d = x.d * y.d
        for (lane in 0 until 8) {
            acc[lane] += d * sumi[lane].toFloat()
        }
    }
    return hsumFloat8(acc)
}
