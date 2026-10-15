// port-lint: source quantized/utils.rs
package io.github.kotlinmania.candle

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.roundToInt

private val Float.round: () -> Float get() = { round(this) }
private val Float.abs: () -> Float get() = { abs(this) }
private val Float.min: (Float) -> Float get() = { other -> min(this, other) }
private val Float.powi: (Int) -> Float get() = { n -> this.pow(n) }
private val Int.clamp: (Int, Int) -> Int get() = { lo, hi -> coerceIn(lo, hi) }
private val UByte.min: (UByte) -> UByte get() = { other -> if (this <= other) this else other }
private val FloatArray.len: () -> Int get() = { size }
private val UByteArray.len: () -> Int get() = { size }
private val <T> List<T>.len: () -> Int get() = { size }
private val <T> List<T>.iter: () -> List<T> get() = { this }
private val <T> MutableList<T>.iterMut: () -> List<Int> get() = { indices.toList() }
private val FloatArray.chunksExact: (Int) -> List<FloatArray>
    get() = { blockSize ->
        val count = size / blockSize
        val out = ArrayList<FloatArray>(count)
        for (i in 0 until count) {
            val start = i * blockSize
            out.add(copyOfRange(start, start + blockSize))
        }
        out
    }
private val FloatArray.chunksExactMut: (Int) -> List<Int>
    get() = { blockSize ->
        val count = size / blockSize
        val out = ArrayList<Int>(count)
        for (i in 0 until count) {
            out.add(i * blockSize)
        }
        out
    }
private val <T> List<T>.collect: () -> List<T> get() = { this }
private val UByteArray.copyFromSlice: (UByteArray) -> Unit get() = { src -> src.copyInto(this) }
private val debugAssertEq: (Any?, Any?, String) -> Unit
    get() = { a, b, msg ->
        check(a == b) { msg }
    }
private val assertEq: (Any?, Any?) -> Unit
    get() = { a, b ->
        check(a == b)
    }

internal fun nearestInt(v: Float): Int {
    return v.round().toInt()
}

/**
 * Validates that the input and output are the right size and returns an iterator which maps each
 * input region `xs` to its corresponding output block in `ys`. Each output region is guaranteed
 * to be `blockSize` long.
 */
internal fun <T> groupForQuantization(
    xs: FloatArray,
    ys: MutableList<T>,
    blockSize: Int,
    dtype: GgmlDType,
): List<Pair<Int, FloatArray>> {
    val expectedBlocks = xs.len() / blockSize
    val actualBlocks = ys.len()

    // Validate that the input is the right size
    debugAssertEq(
        expectedBlocks,
        actualBlocks,
        "quantize $dtype: expected $expectedBlocks blocks but only $actualBlocks were provided!",
    )

    return ys.iterMut().zip(xs.chunksExact(blockSize)).collect()
}

/**
 * Validates that the input and output are the right size and returns an iterator which maps each
 * input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
 * to be `blockSize` long.
 */
internal fun <T> groupForDequantization(
    xs: List<T>,
    ys: FloatArray,
    blockSize: Int,
    dtype: GgmlDType,
): List<Pair<T, Int>> {
    val actualOutputLen = ys.len()
    val expectedOutputLen = xs.len() * blockSize
    // Validate that the output is the right size
    debugAssertEq(
        expectedOutputLen,
        actualOutputLen,
        "dequantize $dtype: ys (len = $actualOutputLen) does not match the expected length of $expectedOutputLen!",
    )
    // Zip the blocks and outputs together
    return xs.iter().zip(ys.chunksExactMut(blockSize)).collect()
}

internal fun getScaleMinK4(j: Int, q: UByteArray): Pair<UByte, UByte> {
    return if (j < 4) {
        val d = (q[j].toInt() and 63).toUByte()
        val m = (q[j + 4].toInt() and 63).toUByte()
        Pair(d, m)
    } else {
        val d = ((q[j + 4].toInt() and 0xF) or (((q[j - 4].toInt() ushr 6) shl 4))).toUByte()
        val m = ((q[j + 4].toInt() ushr 4) or (((q[j].toInt() ushr 6) shl 4))).toUByte()
        Pair(d, m)
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
    var max = 0f
    var amax = 0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val ax = xi.abs()
        if (ax > amax) {
            amax = ax
            max = xi
        }
    }
    if (amax == 0f) {
        // all zero
        for (i in 0 until n) {
            ls[lsOffset + i] = 0
        }
        return 0f
    }
    var iscale = -nmax.toFloat() / max
    if (rmseType == 0) {
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(iscale * xi)
            ls[lsOffset + i] = (nmax + l.clamp(-nmax, nmax - 1)).toByte()
        }
        return 1.0f / iscale
    }
    val weightType = rmseType % 2
    var sumlx = 0f
    var suml2 = 0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val l = nearestInt(iscale * xi).clamp(-nmax, nmax - 1)
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
            val l = nearestInt(curIscale * xi).clamp(-nmax, nmax - 1)
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
            ls[lsOffset + i] = (nmax + l.clamp(-nmax, nmax - 1)).toByte()
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
                val newL = nearestInt(xi * sl2 / slx)
                val clampedL = newL.clamp(-nmax, nmax - 1)
                if (clampedL != l) {
                    slx += w * xi * clampedL.toFloat()
                    sl2 += w * clampedL.toFloat() * clampedL.toFloat()
                    if (sl2 > 0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        ls[lsOffset + i] = (nmax + clampedL).toByte()
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
        iscale = -(nmax.toFloat() + 0.1f * isStep.toFloat()) / max
        var curSumlx = 0f
        var curSuml2 = 0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val l = nearestInt(iscale * xi).clamp(-nmax, nmax - 1)
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
                ls[lsOffset + i] = (nmax + l.clamp(-nmax, nmax - 1)).toByte()
            }
            scale = curSumlx / curSuml2
            best = scale * curSumlx
        }
    }
    return scale
}

internal fun makeQkx1Quants(nmax: Int, ntry: Int, x: FloatArray, xOffset: Int = 0, n: Int = x.len() - xOffset): Pair<Float, Float> {
    val l = UByteArray(n)
    // Get min/max
    var min = x[xOffset]
    var max = x[xOffset]
    for (i in 1 until n) {
        val v = x[xOffset + i]
        if (v < min) min = v
        if (v > max) max = v
    }

    // If min == max, all values are the same => nothing to do here
    if (max == min) {
        return Pair(0.0f, 0.0f)
    }

    // Ensure min <= 0.0
    min = min.min(0.0f)

    // Compute scale and inverse scale
    var iscale = nmax.toFloat() / (max - min)
    var scale = 1.0f / iscale

    for (tryIdx in 0 until ntry) {
        var sumlx = 0.0f
        var suml2 = 0
        var didChange = false

        for (i in 0 until n) {
            val value = x[xOffset + i]
            val li = nearestInt(iscale * (value - min)).clamp(0, nmax)
            val clampedLi = li.toUByte()
            if (clampedLi != l[i]) {
                l[i] = clampedLi
                didChange = true
            }
            sumlx += (value - min) * li.toFloat()
            suml2 += li * li
        }
        scale = sumlx / suml2.toFloat()

        var sum = 0.0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            sum += xi - scale * l[i].toInt().toFloat()
        }

        min = sum / n.toFloat()
        if (min > 0.0f) {
            min = 0.0f
        }
        iscale = 1.0f / scale
        if (!didChange) {
            break
        }
    }
    return Pair(scale, -min)
}

internal fun makeQ3Quants(x: FloatArray, xOffset: Int, n: Int, nmax: Int, doRmse: Boolean): Float {
    val l = ByteArray(n)
    var max = 0.0f
    var amax = 0.0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val ax = xi.abs()
        if (ax > amax) {
            amax = ax
            max = xi
        }
    }

    if (amax == 0.0f) {
        return 0.0f
    }

    val iscale = -nmax.toFloat() / max
    if (doRmse) {
        var sumlx = 0.0f
        var suml2 = 0.0f
        for (i in 0 until n) {
            val li = (iscale * x[xOffset + i]).round().toInt()
            val clamped = li.clamp(-nmax, nmax - 1)
            l[i] = clamped.toByte()
            val w = x[xOffset + i] * x[xOffset + i]
            sumlx += w * x[xOffset + i] * clamped.toFloat()
            suml2 += w * (clamped * clamped).toFloat()
        }
        for (itry in 0 until 5) {
            var nChanged = 0
            for (i in 0 until n) {
                val w = x[xOffset + i] * x[xOffset + i]
                var slx = sumlx - w * x[xOffset + i] * l[i].toFloat()
                if (slx > 0.0f) {
                    var sl2 = suml2 - w * (l[i].toInt() * l[i].toInt()).toFloat()
                    var newL = (x[xOffset + i] * sl2 / slx).round().toInt()
                    newL = newL.clamp(-nmax, nmax - 1)
                    if (newL != l[i].toInt()) {
                        slx += w * x[xOffset + i] * newL.toFloat()
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
        val li = (iscale * x[xOffset + i]).round().toInt()
        l[i] = (li.clamp(-nmax, nmax - 1) + nmax).toByte()
    }
    return 1.0f / iscale
}

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
        if (x[xOffset + i] < minVal) minVal = x[xOffset + i]
        if (x[xOffset + i] > maxVal) maxVal = x[xOffset + i]
        val w = if (weights != null) {
            weights[weightsOffset + i]
        } else {
            x[xOffset + i] * x[xOffset + i]
        }
        sumW += w
        sumX += w * x[xOffset + i]
    }

    if (minVal > 0.0f) {
        minVal = 0.0f
    }

    if (maxVal <= minVal) {
        return Pair(0.0f, -minVal)
    }

    var iscale = nmax.toFloat() / (maxVal - minVal)
    var scale = 1.0f / iscale
    var bestMad = 0.0f

    for (i in 0 until n) {
        val lVal = nearestInt(iscale * (x[xOffset + i] - minVal)).clamp(0, nmax).toUByte()
        l[i] = lVal
        val rawDiff = scale * lVal.toFloat() + minVal - x[xOffset + i]
        val diff = if (useMad) rawDiff.abs() else rawDiff * rawDiff
        val w = if (weights != null) {
            weights[weightsOffset + i]
        } else {
            x[xOffset + i] * x[xOffset + i]
        }
        bestMad += w * diff
    }

    if (nstep < 1) {
        return Pair(scale, -minVal)
    }

    for (isStep in 0..nstep) {
        iscale = (rmin + rdelta * isStep.toFloat() + nmax.toFloat()) / (maxVal - minVal)
        var sumL = 0.0f
        var sumL2 = 0.0f
        var sumXl = 0.0f

        for (i in 0 until n) {
            val lVal = nearestInt(iscale * (x[xOffset + i] - minVal)).clamp(0, nmax).toUByte()
            lAux[i] = lVal
            val w = if (weights != null) weights[weightsOffset + i] else x[xOffset + i] * x[xOffset + i]
            sumL += w * lVal.toFloat()
            sumL2 += w * lVal.toFloat().powi(2)
            sumXl += w * lVal.toFloat() * x[xOffset + i]
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
                val rawDiff = thisScale * lAux[i].toFloat() + thisMin - x[xOffset + i]
                val diff = if (useMad) rawDiff.abs() else rawDiff * rawDiff
                val w = if (weights != null) {
                    weights[weightsOffset + i]
                } else {
                    x[xOffset + i] * x[xOffset + i]
                }
                mad += w * diff
            }

            if (mad < bestMad) {
                l.copyFromSlice(lAux)
                bestMad = mad
                scale = thisScale
                minVal = thisMin
            }
        }
    }
    return Pair(scale, -minVal)
}

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
    assertEq(x.len() - xOffset >= n, true)
    assertEq(l.len() - lOffset >= n, true)
    assertEq(quantWeights.len() - qwOffset >= n, true)
    var max = 0.0f
    for (i in 0 until n) max = max(max, x[xOffset + i])
    if (max == 0.0f) {
        for (i in 0 until n) l[lOffset + i] = 0u
        return 0.0f
    }
    var iscale = nmax.toFloat() / max
    for (i in 0 until n) l[lOffset + i] = nearestInt(iscale * x[xOffset + i]).toUByte()
    val scale = 1.0f / iscale
    var bestMse = 0.0f
    for (i in 0 until n) {
        val xi = x[xOffset + i]
        val li = l[lOffset + i]
        val w = quantWeights[qwOffset + i]
        val diff = xi - scale * li.toFloat()
        bestMse += w * diff * diff
    }

    for (isStep in -4..4) {
        if (isStep == 0) continue
        val iscaleIs = (0.1f * isStep.toFloat() + nmax.toFloat()) / max
        val scaleIs = 1.0f / iscaleIs

        var mse = 0.0f
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val w = quantWeights[qwOffset + i]
            var li = nearestInt(iscaleIs * xi).toUByte()
            li = li.min(nmax)
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
        var liNew = (iscale * xi).round().toInt().toUByte()
        liNew = liNew.min(nmax)
        l[lOffset + i] = liNew
        sumlx += w * xi * liNew.toFloat()
        suml2 += w * liNew.toFloat().powi(2)
    }

    for (itry in 0 until 5) {
        var nChanged = 0
        for (i in 0 until n) {
            val xi = x[xOffset + i]
            val w = quantWeights[qwOffset + i]
            val li = l[lOffset + i]
            var slx = sumlx - w * xi * li.toFloat()
            var sl2 = suml2 - w * li.toFloat().powi(2)
            if (slx > 0.0f && sl2 > 0.0f) {
                val newLi = nearestInt(xi * sl2 / slx).toUByte().min(nmax)
                if (newLi != li) {
                    slx += w * xi * newLi.toFloat()
                    sl2 += w * newLi.toFloat().powi(2)
                    if (slx.powi(2) * suml2 > sumlx.powi(2) * sl2) {
                        l[lOffset + i] = newLi
                        sumlx = slx
                        suml2 = sl2
                        nChanged += 1
                    }
                }
            }
        }
        if (nChanged == 0) break
    }
    return sumlx / suml2
}
