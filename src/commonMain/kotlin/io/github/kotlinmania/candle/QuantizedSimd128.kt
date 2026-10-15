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

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    val acc = FloatArray(4)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val sumXy = IntArray(4)

        for (lane in 0 until 4) {
            val low0 = (x.qs[lane * 2].toInt() and 0x0F) - 8
            val low1 = (x.qs[lane * 2 + 1].toInt() and 0x0F) - 8
            val yLow0 = y.qs[lane * 2].toInt()
            val yLow1 = y.qs[lane * 2 + 1].toInt()
            sumXy[lane] += low0 * yLow0 + low1 * yLow1

            val mid0 = (x.qs[8 + lane * 2].toInt() and 0x0F) - 8
            val mid1 = (x.qs[8 + lane * 2 + 1].toInt() and 0x0F) - 8
            val yMid0 = y.qs[8 + lane * 2].toInt()
            val yMid1 = y.qs[8 + lane * 2 + 1].toInt()
            sumXy[lane] += mid0 * yMid0 + mid1 * yMid1

            val high0 = ((x.qs[lane * 2].toInt() and 0xFF) ushr 4) - 8
            val high1 = ((x.qs[lane * 2 + 1].toInt() and 0xFF) ushr 4) - 8
            val yHigh0 = y.qs[16 + lane * 2].toInt()
            val yHigh1 = y.qs[16 + lane * 2 + 1].toInt()
            sumXy[lane] += high0 * yHigh0 + high1 * yHigh1

            val top0 = ((x.qs[8 + lane * 2].toInt() and 0xFF) ushr 4) - 8
            val top1 = ((x.qs[8 + lane * 2 + 1].toInt() and 0xFF) ushr 4) - 8
            val yTop0 = y.qs[24 + lane * 2].toInt()
            val yTop1 = y.qs[24 + lane * 2 + 1].toInt()
            sumXy[lane] += top0 * yTop0 + top1 * yTop1
        }

        val d = x.d.toFloat() * y.d.toFloat()
        for (lane in 0 until 4) {
            acc[lane] += sumXy[lane].toFloat() * d
        }
    }
    return acc[0] + acc[1] + acc[2] + acc[3]
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    val acc = FloatArray(4)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val sumXy = IntArray(4)

        for (step in 0 until 4) {
            val base = step * 8
            for (lane in 0 until 4) {
                val x0 = x.qs[base + lane * 2].toInt()
                val x1 = x.qs[base + lane * 2 + 1].toInt()
                val y0 = y.qs[base + lane * 2].toInt()
                val y1 = y.qs[base + lane * 2 + 1].toInt()
                sumXy[lane] += x0 * y0 + x1 * y1
            }
        }

        val d = x.d.toFloat() * y.d.toFloat()
        for (lane in 0 until 4) {
            acc[lane] += sumXy[lane].toFloat() * d
        }
    }
    return acc[0] + acc[1] + acc[2] + acc[3]
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    val sumf = FloatArray(4)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        var q2Offset = 0
        var q8Offset = 0
        val sc = x.scales

        val summs = IntArray(4)
        for (i in 0 until (QKK / 16) step 4) {
            for (lane in 0 until 4) {
                val bsum = y.bsums[i + lane].toInt()
                val scale = (sc[i + lane].toInt() and 0xFF) ushr 4
                summs[lane] += bsum * scale
            }
        }

        val dall = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

        val isum = IntArray(4)
        var isIdx = 0
        for (block in 0 until (QKK / 128)) {
            var shift = 0
            for (sub in 0 until 4) {
                val d0 = sc[isIdx].toInt() and 0xF
                isIdx += 1
                val isuml0 = ShortArray(8)
                for (l in 0 until 16 step 8) {
                    for (k in 0 until 8) {
                        val q8Val = y.qs[q8Offset + l + k].toInt()
                        val q2Val = ((x.qs[q2Offset + l + k].toInt() and 0xFF) ushr shift) and 3
                        isuml0[k] = (isuml0[k].toInt() + q2Val * q8Val).toShort()
                    }
                }
                for (lane in 0 until 4) {
                    isum[lane] += isuml0[lane].toInt() * d0
                    isum[lane] += isuml0[lane + 4].toInt() * d0
                }

                val d1 = sc[isIdx].toInt() and 0xF
                isIdx += 1
                val isuml1 = ShortArray(8)
                for (l in 16 until 32 step 8) {
                    for (k in 0 until 8) {
                        val q8Val = y.qs[q8Offset + l + k].toInt()
                        val q2Val = ((x.qs[q2Offset + l + k].toInt() and 0xFF) ushr shift) and 3
                        isuml1[k] = (isuml1[k].toInt() + q2Val * q8Val).toShort()
                    }
                }
                for (lane in 0 until 4) {
                    isum[lane] += isuml1[lane].toInt() * d1
                    isum[lane] += isuml1[lane + 4].toInt() * d1
                }

                shift += 2
                q8Offset += 32
            }
            q2Offset += 32
        }
        for (lane in 0 until 4) {
            sumf[lane] += isum[lane].toFloat() * dall - summs[lane].toFloat() * dmin
        }
    }
    return sumf[0] + sumf[1] + sumf[2] + sumf[3]
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
    val sums = FloatArray(4)

    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val y = ys[idx]
        val x = xs[idx]
        val q4 = x.qs
        val q8 = y.qs

        for (j in 0 until QKK / 64) {
            for (k in 0 until 16) {
                val q4Val1 = q4[32 * j + k].toInt() and 0xFF
                val q4Val2 = q4[32 * j + 16 + k].toInt() and 0xFF
                aux8[64 * j + k] = (q4Val1 and 0x0F).toUByte()
                aux8[64 * j + 16 + k] = (q4Val2 and 0x0F).toUByte()
                aux8[64 * j + 32 + k] = (q4Val1 ushr 4).toUByte()
                aux8[64 * j + 48 + k] = (q4Val2 ushr 4).toUByte()
            }
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
            val sWord = utmp[w]
            scales[w * 4] = (sWord and 0xFF).toUByte()
            scales[w * 4 + 1] = ((sWord ushr 8) and 0xFF).toUByte()
            scales[w * 4 + 2] = ((sWord ushr 16) and 0xFF).toUByte()
            scales[w * 4 + 3] = ((sWord ushr 24) and 0xFF).toUByte()

            val mWord = utmp[w + 2]
            mins[w * 4] = (mWord and 0xFF).toUByte()
            mins[w * 4 + 1] = ((mWord ushr 8) and 0xFF).toUByte()
            mins[w * 4 + 2] = ((mWord ushr 16) and 0xFF).toUByte()
            mins[w * 4 + 3] = ((mWord ushr 24) and 0xFF).toUByte()
        }

        val sumi = IntArray(4)
        for (j in 0 until QKK / 16 step 4) {
            val m1 = mins[j / 2].toInt()
            val m2 = mins[j / 2 + 1].toInt()
            sumi[0] += y.bsums[j].toInt() * m1
            sumi[1] += y.bsums[j + 1].toInt() * m1
            sumi[2] += y.bsums[j + 2].toInt() * m2
            sumi[3] += y.bsums[j + 3].toInt() * m2
        }

        val aux32 = IntArray(4)
        for (scaleI in 0 until scales.size) {
            val scale = scales[scaleI].toInt()
            for (j in 0 until 4) {
                val i = 32 * scaleI + 8 * j
                for (lane in 0 until 4) {
                    val lowProd = q8[i + lane].toInt() * aux8[i + lane].toInt()
                    val highProd = q8[i + 4 + lane].toInt() * aux8[i + 4 + lane].toInt()
                    aux32[lane] += scale * lowProd + scale * highProd
                }
            }
        }

        val d = x.d.toFloat() * y.d
        val dmin = x.dmin.toFloat() * y.d
        for (lane in 0 until 4) {
            sums[lane] += aux32[lane].toFloat() * d - sumi[lane].toFloat() * dmin
        }
    }
    return sums[0] + sums[1] + sums[2] + sums[3]
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    val aux8 = ByteArray(QKK)
    val sums = FloatArray(4)
    val count = minOf(xs.size, ys.size)

    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val q4 = x.ql
        val qh = x.qh
        val q8 = y.qs
        val aux32 = FloatArray(4)

        for (j in 0 until QKK step 128) {
            val q4Base = j / 2
            val qhBase = j / 4
            for (l in 0 until 32 step 16) {
                for (k in 0 until 16) {
                    val ql0 = q4[q4Base + l + k].toInt() and 0xFF
                    val ql32 = q4[q4Base + l + 32 + k].toInt() and 0xFF
                    val qhVal = qh[qhBase + l + k].toInt() and 0xFF

                    val a0 = ((ql0 and 0xF) or ((qhVal and 3) shl 4)) - 32
                    val a1 = ((ql32 and 0xF) or (((qhVal ushr 2) and 3) shl 4)) - 32
                    val a2 = ((ql0 ushr 4) or (((qhVal ushr 4) and 3) shl 4)) - 32
                    val a3 = ((ql32 ushr 4) or (((qhVal ushr 6) and 3) shl 4)) - 32

                    aux8[j + l + k] = a0.toByte()
                    aux8[j + l + 32 + k] = a1.toByte()
                    aux8[j + l + 64 + k] = a2.toByte()
                    aux8[j + l + 96 + k] = a3.toByte()
                }
            }
        }

        for (j in 0 until x.scales.size) {
            val scale = x.scales[j].toFloat()
            for (offset in intArrayOf(0, 8)) {
                val base = 16 * j + offset
                for (lane in 0 until 4) {
                    val lowProd = q8[base + lane].toInt() * aux8[base + lane].toInt()
                    val highProd = q8[base + 4 + lane].toInt() * aux8[base + 4 + lane].toInt()
                    aux32[lane] += lowProd.toFloat() * scale + highProd.toFloat() * scale
                }
            }
        }

        val d = x.d.toFloat() * y.d
        for (lane in 0 until 4) {
            sums[lane] += aux32[lane] * d
        }
    }
    return sums[0] + sums[1] + sums[2] + sums[3]
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    val acc = FloatArray(4)
    val count = minOf(xs.size, ys.size)
    for (idx in 0 until count) {
        val x = xs[idx]
        val y = ys[idx]
        val sumi = IntArray(4)
        for (j in 0 until QKK step 8) {
            for (lane in 0 until 4) {
                val x0 = x.qs[j + lane * 2].toInt()
                val x1 = x.qs[j + lane * 2 + 1].toInt()
                val y0 = y.qs[j + lane * 2].toInt()
                val y1 = y.qs[j + lane * 2 + 1].toInt()
                sumi[lane] += x0 * y0 + x1 * y1
            }
        }
        val d = x.d * y.d
        for (lane in 0 until 4) {
            acc[lane] += sumi[lane].toFloat() * d
        }
    }
    return acc[0] + acc[1] + acc[2] + acc[3]
}
