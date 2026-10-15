// port-lint: source quantized/utils.rs
package io.github.kotlinmania.candle

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

internal fun nearestInt(v: Float): Int = v.roundToInt()

/**
 * Validates that the input and output are the right size and returns a list which maps each
 * input region `xs` to its corresponding output block index in `ys`. Each output region is guaranteed
 * to be `blockSize` long.
 */
internal fun <T> groupForQuantization(
    xs: FloatArray,
    ys: MutableList<T>,
    blockSize: Int,
    dtype: GgmlDType,
): List<Pair<Int, FloatArray>> {
    val expectedBlocks = xs.size / blockSize
    val actualBlocks = ys.size

    // Validate that the input is the right size
    check(expectedBlocks == actualBlocks) {
        "quantize $dtype: expected $expectedBlocks blocks but only $actualBlocks were provided!"
    }

    val out = ArrayList<Pair<Int, FloatArray>>(actualBlocks)
    for (i in 0 until actualBlocks) {
        val start = i * blockSize
        out.add(i to xs.copyOfRange(start, start + blockSize))
    }
    return out
}

/**
 * Validates that the input and output are the right size and returns a list which maps each
 * input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
 * to be `blockSize` long.
 */
internal fun <T> groupForDequantization(
    xs: List<T>,
    ys: FloatArray,
    blockSize: Int,
    dtype: GgmlDType,
): List<Pair<T, Int>> {
    val actualOutputLen = ys.size
    val expectedOutputLen = xs.size * blockSize
    // Validate that the output is the right size
    check(expectedOutputLen == actualOutputLen) {
        "dequantize $dtype: ys (len = $actualOutputLen) does not match the expected length of $expectedOutputLen!"
    }

    // Zip the blocks and outputs together
    val out = ArrayList<Pair<T, Int>>(xs.size)
    for (i in xs.indices) {
        out.add(xs[i] to (i * blockSize))
    }
    return out
}

internal fun getScaleMinK4(j: Int, q: UByteArray): Pair<UByte, UByte> {
    return if (j < 4) {
        val d = (q[j].toInt() and 63).toUByte()
        val m = (q[j + 4].toInt() and 63).toUByte()
        d to m
    } else {
        val d = ((q[j + 4].toInt() and 0xF) or (((q[j - 4].toInt() ushr 6) shl 4))).toUByte()
        val m = ((q[j + 4].toInt() ushr 4) or (((q[j].toInt() ushr 6) shl 4))).toUByte()
        d to m
    }
}

internal fun makeQxQuants(
    n: Int,
    nmax: Int,
    x: FloatArray,
    xOffset: Int,
    ls: ByteArray,
    lsOffset: Int,
    rmseType: Int,
    qw: FloatArray?,
    qwOffset: Int = 0,
): Float {
    var maxVal = 0f
    var amax = 0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val ax = abs(xi)
        if (ax > amax) {
            amax = ax
            maxVal = xi
        }
    }
    if (amax == 0f) {
        // all zero
        for (i in 0 until n) {
            ls[lsOffset + i] = 0
        }
        return 0f
    }
    var iscale = -nmax.toFloat() / maxVal
    if (rmseType == 0) {
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(iscale * xi)
            ls[lsOffset + i] = (nmax + l.coerceIn(-nmax, nmax - 1)).toByte()
        }
        return 1.0f / iscale
    }
    val weightType = rmseType % 2
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val l = nearestInt(iscale * xi).coerceIn(-nmax, nmax - 1)
        ls[lsOffset + i] = (l + nmax).toByte()
        val w = if (qw != null) {
            qw[qwOffset + i]
        } else if (weightType == 1) {
            xi * xi
        } else {
            1.0f
        }
        val lf = l.toFloat()
        sumlx += w * xi * lf
        suml2 += w * lf * lf
    }
    var scale = sumlx / suml2
    var best = scale * sumlx
    for (itry in 0 until 3) {
        val curIscale = 1.0f / scale
        var slx = 0f
        var sl2 = 0f
        var changed = false
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(curIscale * xi).coerceIn(-nmax, nmax - 1)
            if (l + nmax != ls[lsOffset + i].toInt()) {
                changed = true
            }
            val w = if (qw != null) {
                qw[qwOffset + i]
            } else if (weightType == 1) {
                xi * xi
            } else {
                1.0f
            }
            val lf = l.toFloat()
            slx += w * xi * lf
            sl2 += w * lf * lf
        }
        if (!changed || sl2 == 0.0f || slx * slx <= best * sl2) {
            break
        }
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(curIscale * xi)
            ls[lsOffset + i] = (nmax + l.coerceIn(-nmax, nmax - 1)).toByte()
        }
        sumlx = slx
        suml2 = sl2
        scale = sumlx / suml2
        best = scale * sumlx
    }
    for (itry in 0 until 5) {
        var nChanged = 0
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val w = if (qw != null) {
                qw[qwOffset + i]
            } else if (weightType == 1) {
                xi * xi
            } else {
                1.0f
            }
            val l = ls[lsOffset + i].toInt() - nmax
            var slx = sumlx - w * xi * l.toFloat()
            if (slx > 0f) {
                var sl2 = suml2 - w * l.toFloat() * l.toFloat()
                val newL = nearestInt(xi * sl2 / slx).coerceIn(-nmax, nmax - 1)
                if (newL != l) {
                    slx += w * xi * newL.toFloat()
                    sl2 += w * newL.toFloat() * newL.toFloat()
                    if (sl2 > 0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        ls[lsOffset + i] = (nmax + newL).toByte()
                        sumlx = slx
                        suml2 = sl2
                        scale = sumlx / suml2
                        best = scale * sumlx
                        nChanged += 1
                    }
                }
            }
        }
        if (nChanged == 0) {
            break
        }
    }
    if (rmseType < 3) {
        return scale
    }
    for (isStep in -4 until 4) {
        if (isStep == 0) {
            continue
        }
        iscale = -(nmax.toFloat() + 0.1f * isStep.toFloat()) / maxVal
        var curSumlx = 0f
        var curSuml2 = 0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(iscale * xi).coerceIn(-nmax, nmax - 1)
            val w = if (qw != null) {
                qw[qwOffset + i]
            } else if (weightType == 1) {
                xi * xi
            } else {
                1.0f
            }
            val lf = l.toFloat()
            curSumlx += w * xi * lf
            curSuml2 += w * lf * lf
        }
        if (curSuml2 > 0f && curSumlx * curSumlx > best * curSuml2) {
            for (i in 0 until n) {
                val xi = x[xOffset + i]
                val l = nearestInt(iscale * xi)
                ls[lsOffset + i] = (nmax + l.coerceIn(-nmax, nmax - 1)).toByte()
            }
            scale = curSumlx / curSuml2
            best = scale * curSumlx
        }
    }
    return scale
}

internal fun makeQkx1Quants(nmax: Int, ntry: Int, x: FloatArray, xOffset: Int = 0, n: Int = x.size - xOffset): Pair<Float, Float> {
    val l = UByteArray(n)
    // Get min/max
    var minVal = x[xOffset]
    var maxVal = x[xOffset]
    for (i in 1 until n) {
        val v = x[xOffset + i]
        if (v < minVal) minVal = v
        if (v > maxVal) maxVal = v
    }

    // If min == max, all values are the same => nothing to do here
    if (maxVal == minVal) {
        return 0.0f to 0.0f
    }

    // Ensure min <= 0.0
    minVal = min(minVal, 0.0f)

    // Compute scale and inverse scale
    var iscale = nmax.toFloat() / (maxVal - minVal)
    var scale = 1.0f / iscale

    for (tryIdx in 0 until ntry) {
        var sumlx = 0.0f
        var suml2 = 0
        var didChange = false

        for (i in 0 until n) {
            val value = x[xOffset + i]
            val li = nearestInt(iscale * (value - minVal)).coerceIn(0, nmax)
            val clampedLi = li.toUByte()
            if (clampedLi != l[i]) {
                l[i] = clampedLi
                didChange = true
            }
            sumlx += (value - minVal) * li.toFloat()
            suml2 += li * li
        }
        scale = sumlx / suml2.toFloat()

        var sum = 0.0f
        for (i in 0 until n) {
            sum += x[xOffset + i] - scale * l[i].toInt().toFloat()
        }

        minVal = sum / n.toFloat()
        if (minVal > 0.0f) {
            minVal = 0.0f
        }
        iscale = 1.0f / scale
        if (!didChange) {
            break
        }
    }
    return scale to -minVal
}

internal fun makeQ3Quants(x: FloatArray, xOffset: Int, n: Int, nmax: Int, doRmse: Boolean): Float {
    val l = ByteArray(n)

    var maxVal = 0.0f
    var amax = 0.0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val ax = abs(xi)
        if (ax > amax) {
            amax = ax
            maxVal = xi
        }
    }

    if (amax == 0.0f) {
        return 0.0f
    }

    val iscale = -nmax.toFloat() / maxVal
    if (doRmse) {
        var sumlx = 0.0f
        var suml2 = 0.0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val li = (iscale * xi).roundToInt().coerceIn(-nmax, nmax - 1)
            l[i] = li.toByte()
            val w = xi * xi
            sumlx += w * xi * li.toFloat()
            suml2 += w * (li * li).toFloat()
        }
        for (itry in 0 until 5) {
            var nChanged = 0
            for (i in 0 until n) {
                val xi = x[xOffset + i]
                val w = xi * xi
                val curL = l[i].toInt()
                var slx = sumlx - w * xi * curL.toFloat()
                if (slx > 0.0f) {
                    var sl2 = suml2 - w * (curL * curL).toFloat()
                    val newL = (xi * sl2 / slx).roundToInt().coerceIn(-nmax, nmax - 1)
                    if (newL != curL) {
                        slx += w * xi * newL.toFloat()
                        sl2 += w * (newL * newL).toFloat()
                        if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            l[i] = newL.toByte()
                            sumlx = slx
                            suml2 = sl2
                            nChanged += 1
                        }
                    }
                }
            }
            if (nChanged == 0) {
                break
            }
        }
        for (i in 0 until n) {
            l[i] = (l[i].toInt() + nmax).toByte()
        }
        return sumlx / suml2
    }
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val li = (iscale * xi).roundToInt()
        l[i] = (li.coerceIn(-nmax, nmax - 1) + nmax).toByte()
    }
    return 1.0f / iscale
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L744
/**
 * (scale, min)
 */
internal fun makeQkx3Quants(
    nmax: Int,
    x: FloatArray,
    xOffset: Int,
    n: Int,
    weights: FloatArray?,
    weightsOffset: Int,
    rmin: Float,
    rdelta: Float,
    nstep: Int,
    useMad: Boolean,
): Pair<Float, Float> {
    val l = UByteArray(32)
    val lAux = UByteArray(32)

    var minVal = x[xOffset]
    var maxVal = x[xOffset]
    var sumW = if (weights != null) weights[weightsOffset] else x[xOffset] * x[xOffset]
    var sumX = sumW * x[xOffset]

    for (i in 1 until n) {
        val xi = x[xOffset + i]
        if (xi < minVal) {
            minVal = xi
        }
        if (xi > maxVal) {
            maxVal = xi
        }
        val w = if (weights != null) weights[weightsOffset + i] else xi * xi
        sumW += w
        sumX += w * xi
    }

    if (minVal > 0.0f) {
        minVal = 0.0f
    }

    if (maxVal <= minVal) {
        return 0.0f to -minVal
    }

    var iscale = nmax.toFloat() / (maxVal - minVal)
    var scale = 1.0f / iscale
    var bestMad = 0.0f

    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val lVal = nearestInt(iscale * (xi - minVal)).coerceIn(0, nmax).toUByte()
        l[i] = lVal
        val rawDiff = scale * lVal.toInt().toFloat() + minVal - xi
        val diff = if (useMad) abs(rawDiff) else rawDiff * rawDiff
        val w = if (weights != null) weights[weightsOffset + i] else xi * xi
        bestMad += w * diff
    }

    if (nstep < 1) {
        return scale to -minVal
    }

    for (isStep in 0..nstep) {
        iscale = (rmin + rdelta * isStep.toFloat() + nmax.toFloat()) / (maxVal - minVal)
        var sumL = 0.0f
        var sumL2 = 0.0f
        var sumXl = 0.0f

        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val lVal = nearestInt(iscale * (xi - minVal)).coerceIn(0, nmax).toUByte()
            lAux[i] = lVal
            val w = if (weights != null) weights[weightsOffset + i] else xi * xi
            val lf = lVal.toInt().toFloat()
            sumL += w * lf
            sumL2 += w * lf.pow(2)
            sumXl += w * lf * xi
        }

        val d = sumW * sumL2 - sumL * sumL
        if (d > 0.0f) {
            var thisScale = (sumW * sumXl - sumX * sumL) / d
            var thisMin = (sumL2 * sumX - sumL * sumXl) / d

            if (thisMin > 0.0f) {
                thisMin = 0.0f
                thisScale = sumXl / sumL2
            }

            var mad = 0.0f
            for (i in 0 until n) {
                val xi = x[xOffset + i]
                val rawDiff = thisScale * lAux[i].toInt().toFloat() + thisMin - xi
                val diff = if (useMad) abs(rawDiff) else rawDiff * rawDiff
                val w = if (weights != null) weights[weightsOffset + i] else xi * xi
                mad += w * diff
            }

            if (mad < bestMad) {
                lAux.copyInto(l)
                bestMad = mad
                scale = thisScale
                minVal = thisMin
            }
        }
    }

    return scale to -minVal
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L827
internal fun makeQpQuants(
    n: Int,
    nmax: UByte,
    x: FloatArray,
    xOffset: Int,
    l: UByteArray,
    lOffset: Int,
    quantWeights: FloatArray,
    qwOffset: Int,
): Float {
    val nmaxInt = nmax.toInt()
    var maxVal = 0.0f
    for (i in 0 until n) {
        maxVal = max(maxVal, x[xOffset + i])
    }
    if (maxVal == 0.0f) {
        for (i in 0 until n) {
            l[lOffset + i] = 0u
        }
        return 0.0f
    }

    var iscale = nmaxInt.toFloat() / maxVal
    for (i in 0 until n) {
        l[lOffset + i] = nearestInt(iscale * x[xOffset + i]).toUByte()
    }

    val scale = 1.0f / iscale
    var bestMse = 0.0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val li = l[lOffset + i].toInt().toFloat()
        val w = quantWeights[qwOffset + i]
        val diff = xi - scale * li
        bestMse += w * diff * diff
    }

    for (isStep in -4..4) {
        if (isStep == 0) {
            continue
        }
        val iscaleIs = (0.1f * isStep.toFloat() + nmaxInt.toFloat()) / maxVal
        val scaleIs = 1.0f / iscaleIs

        var mse = 0.0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val w = quantWeights[qwOffset + i]
            val li = min(nearestInt(iscaleIs * xi), nmaxInt)
            val diff = xi - scaleIs * li.toFloat()
            mse += w * diff * diff
        }

        if (mse < bestMse) {
            bestMse = mse
            iscale = iscaleIs
        }
    }

    var sumlx = 0.0f
    var suml2 = 0.0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val w = quantWeights[qwOffset + i]
        val liNew = min((iscale * xi).roundToInt(), nmaxInt)
        l[lOffset + i] = liNew.toUByte()
        val lf = liNew.toFloat()
        sumlx += w * xi * lf
        suml2 += w * lf.pow(2)
    }

    for (itry in 0 until 5) {
        var nChanged = 0
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val w = quantWeights[qwOffset + i]
            val curLi = l[lOffset + i].toInt()
            var slx = sumlx - w * xi * curLi.toFloat()
            var sl2 = suml2 - w * curLi.toFloat().pow(2)
            if (slx > 0.0f && sl2 > 0.0f) {
                val newLi = min(nearestInt(xi * sl2 / slx), nmaxInt)
                if (newLi != curLi) {
                    val newLf = newLi.toFloat()
                    slx += w * xi * newLf
                    sl2 += w * newLf.pow(2)
                    if (slx.pow(2) * suml2 > sumlx.pow(2) * sl2) {
                        l[lOffset + i] = newLi.toUByte()
                        sumlx = slx
                        suml2 = sl2
                        nChanged += 1
                    }
                }
            }
        }
        if (nChanged == 0) {
            break
        }
    }

    return sumlx / suml2
}
