// port-lint: source quantized/utils.rs
@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.quantized.utils

import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.unwrapOr

internal fun nearestInt(v: Float): Int = v.round().toInt()

/**
 * Validates that the input and output are the right size and returns an iterator which maps each
 * input region `xs` to its corresponding output block in `ys`. Each output region is guaranteed
 * to be `T.BLCK_SIZE` long.
 */
internal fun <VecDotType, T : GgmlType> groupForQuantization(xs: List<Float>, ys: MutableList<T>): List<Pair<T, List<Float>>> {
    val blockSize = T.BLCK_SIZE
    val expectedBlocks = xs.size / blockSize
    val actualBlocks = ys.size
    // Validate that the input is the right size
    check(expectedBlocks == actualBlocks)
    return (ys.zip(xs.chunksExact(blockSize))).toMutableList()
}

/**
 * Validates that the input and output are the right size and returns an iterator which maps each
 * input block `xs` to its corresponding output region in `ys`. Each output region is guaranteed
 * to be `T.BLCK_SIZE` long.
 */
internal fun <VecDotType, T : GgmlType> groupForDequantization(xs: List<T>, ys: MutableList<Float>): List<Pair<T, MutableList<Float>>> {
    val blockSize = T.BLCK_SIZE
    val actualOutputLen = ys.size
    val expectedOutputLen = xs.size * blockSize
    // Validate that the output is the right size
    check(expectedOutputLen == actualOutputLen)
    // Zip the blocks and outputs together
    return (xs.zip(ys.chunksExactMut(blockSize))).toMutableList()
}

internal fun getScaleMinK4(j: Int, q: List<UByte>): Pair<UByte, UByte> =
    if (j < 4) {
        val d = q[j] and 63u.toUByte()
        val m = q[j + 4] and 63u.toUByte()
        Pair(d, m)
    } else {
        val d1 = (q[j + 4] and 0xFu.toUByte()) or ((((q[j - 4].toUInt() shr 6).toUByte()).toUInt() shl 4).toUByte())
        val m1 = ((q[j + 4].toUInt() shr 4).toUByte()) or ((((q[j].toUInt() shr 6).toUByte()).toUInt() shl 4).toUByte())
        Pair(d1, m1)
    }

internal fun makeQxQuants(n: Int, nmax: Int, x: ConstPtr, ls: MutPtr, rmseType: Int, qw: ConstPtr): Float {
    var max = 0.0f
    var amax = 0.0f
    for (i in 0 until n) {
        val x1 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
        val ax = x1.abs()
        if (ax > amax) {
            amax = ax
            max = x1
        }
    }
    if (amax == 0.0) {
        // all zero
        for (i1 in 0 until n) {
            (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i1).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, 0)
        }
        return 0.0f
    }
    var iscale = -(nmax.toFloat()) / max
    if (rmseType == 0) {
        for (i2 in 0 until n) {
            val x2 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i2).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            val l = nearestInt(iscale * x2)
            (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i2).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, (nmax + l.clamp(-nmax, nmax - 1)).toByte())
        }
        return 1.0f / iscale
    }
    val weightType = rmseType % 2
    var sumlx = 0.0f
    var suml2 = 0.0f
    for (i3 in 0 until n) {
        val x3 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i3).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
        val l1 = nearestInt(iscale * x3)
        val l2 = l1.clamp(-nmax, nmax - 1)
        (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i3).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, (l2 + nmax).toByte())
        val w =
            if (!((qw).rawAddress == 0L)) {
                (PrimitiveLayouts.F32).readNative((ConstPtr((qw).rawAddress + ((i3).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            } else if (weightType == 1) {
                x3 * x3
            } else {
                1.0
            }
        val l3 = l2.toFloat()
        sumlx += w * x3 * l3
        suml2 += w * l3 * l3
    }
    var scale = sumlx / suml2
    var best = scale * sumlx
    for (itry in 0 until 3) {
        val iscale1 = 1.0 / scale
        var slx = 0.0f
        var sl2 = 0.0f
        var changed = false
        for (i4 in 0 until n) {
            val x4 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i4).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            val l4 = nearestInt(iscale1 * x4)
            val l5 = l4.clamp(-nmax, nmax - 1)
            if (l5 + nmax != (PrimitiveLayouts.I8).readNative((MutPtr((ls).rawAddress + ((i4).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt()) {
                changed = true
            }
            val w1 =
                if (!((qw).rawAddress == 0L)) {
                    (PrimitiveLayouts.F32).readNative((ConstPtr((qw).rawAddress + ((i4).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
                } else if (weightType == 1) {
                    x4 * x4
                } else {
                    1.0
                }
            val l6 = l5.toFloat()
            slx += w1 * x4 * l6
            sl2 += w1 * l6 * l6
        }
        if (!changed || sl2 == 0.0 || slx * slx <= best * sl2) {
            break
        }
        for (i5 in 0 until n) {
            val x5 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i5).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            val l7 = nearestInt(iscale1 * x5)
            (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i5).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, (nmax + l7.clamp(-nmax, nmax - 1)).toByte())
        }
        sumlx = slx
        suml2 = sl2
        scale = sumlx / suml2
        best = scale * sumlx
    }
    for (itry1 in 0 until 5) {
        var nChanged = 0
        for (i6 in 0 until n) {
            val x6 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i6).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            val w2 =
                if (!((qw).rawAddress == 0L)) {
                    (PrimitiveLayouts.F32).readNative((ConstPtr((qw).rawAddress + ((i6).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
                } else if (weightType == 1) {
                    x6 * x6
                } else {
                    1.0
                }
            val l8 = (PrimitiveLayouts.I8).readNative((MutPtr((ls).rawAddress + ((i6).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress).toInt() - nmax
            var slx1 = sumlx - w2 * x6 * l8.toFloat()
            if (slx1 > 0.0f) {
                var sl21 = suml2 - w2 * l8.toFloat() * l8.toFloat()
                val newL = nearestInt(x6 * sl21 / slx1)
                val newL1 = newL.clamp(-nmax, nmax - 1)
                if (newL1 != l8) {
                    slx1 += w2 * x6 * newL1.toFloat()
                    sl21 += w2 * newL1.toFloat() * newL1.toFloat()
                    if (sl21 > 0.0f && slx1 * slx1 * suml2 > sumlx * sumlx * sl21) {
                        (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i6).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, (nmax + newL1).toByte())
                        sumlx = slx1
                        suml2 = sl21
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
    for (isMatch in -4 until 4) {
        if (isMatch == 0) {
            continue
        }
        iscale = -(nmax.toFloat() + 0.1f * isMatch.toFloat()) / max
        var sumlx1 = 0.0
        var suml21 = 0.0
        for (i7 in 0 until n) {
            val x7 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i7).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
            val l9 = nearestInt(iscale * x7)
            val l10 = l9.clamp(-nmax, nmax - 1)
            val w3 =
                if (!((qw).rawAddress == 0L)) {
                    (PrimitiveLayouts.F32).readNative((ConstPtr((qw).rawAddress + ((i7).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
                } else if (weightType == 1) {
                    x7 * x7
                } else {
                    1.0
                }
            val l11 = l10.toFloat()
            sumlx1 += w3 * x7 * l11
            suml21 += w3 * l11 * l11
        }
        if (suml21 > 0.0 && sumlx1 * sumlx1 > best * suml21) {
            for (i8 in 0 until n) {
                val x8 = (PrimitiveLayouts.F32).readNative((ConstPtr((x).rawAddress + ((i8).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress)
                val l12 = nearestInt(iscale * x8)
                (PrimitiveLayouts.I8).writeNative((MutPtr((ls).rawAddress + ((i8).toLong() * (PrimitiveLayouts.I8).strideBytes))).rawAddress, (nmax + l12.clamp(-nmax, nmax - 1)).toByte())
            }
            scale = sumlx1 / suml21
            best = scale * sumlx1
        }
    }
    return scale
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L224
internal fun makeQkx1Quants(nmax: Int, ntry: Int, x: List<Float>): Pair<Float, Float> {
    val n = x.size
    val l = mutableListOf(0, n)
    // Get min/max
    val min =
        x
            .iter()
            .take(n)
            .minBy({ a, b -> a.totalCmp(b) })
            .unwrapOr(x[0])
    val max = x.iter().maxBy({ a, b -> a.totalCmp(b) }).unwrapOr(x[0])
    // If min == max, all values are the same => nothing to do here
    if (max == min) {
        return Pair(0.0, 0.0)
    }
    // Ensure min <= 0.0
    var min1 = min.min(0.0)
    // Compute scale and inverse scale
    var iscale = nmax.toFloat() / (max - min1)
    var scale = 1.0f / iscale
    for (unused in 0 until ntry) {
        var sumlx = 0.0
        var suml2 = 0
        var didChange = false
        for ((i, value) in x.mapIndexed { index, item -> Pair(index, item) }.take(n)) {
            val li = nearestInt(iscale * (value - min1)).clamp(0, nmax)
            val clampedLi = li.toUByte()
            if (clampedLi != l[i]) {
                l[i] = clampedLi
                didChange = true
            }
            sumlx += (value - min1) * li.toFloat()
            suml2 += li * li
        }
        scale = sumlx / suml2.toFloat()
        val sum: Float = (x.iter().take(n).zip(ArrayDeque(l).take(n))).map({ (xi, li1) -> xi - scale * li1.toFloat() }).sum()
        min1 = sum / n.toFloat()
        if (min1 > 0.0) {
            min1 = 0.0
        }
        iscale = 1.0f / scale
        if (!didChange) {
            break
        }
    }
    return Pair(scale, -min1)
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/kQuants.c#L165
internal fun makeQ3Quants(x: List<Float>, nmax: Int, doRmse: Boolean): Float {
    val n = x.size
    val l = mutableListOf(0, n)
    var max = 0.0
    var amax = 0.0
    for (xi in x.iter().take(n)) {
        val ax = xi.abs()
        if (ax > amax) {
            amax = ax
            max = xi
        }
    }
    if (amax == 0.0) {
        return 0.0f
    }
    val iscale = -(nmax.toFloat()) / max
    if (doRmse) {
        var sumlx = 0.0
        var suml2 = 0.0
        for (i in 0 until n) {
            val li = (iscale * x[i]).round().toInt()
            val li1 = li.clamp(-nmax, nmax - 1)
            l[i] = li1.toByte()
            val w = x[i] * x[i]
            sumlx += w * x[i] * li1.toFloat()
            suml2 += w * (li1 * li1).toFloat()
        }
        for (unused in 0 until 5) {
            var nChanged = 0
            for (i1 in 0 until n) {
                val w1 = x[i1] * x[i1]
                var slx = sumlx - w1 * x[i1] * l[i1].toFloat()
                if (slx > 0.0f) {
                    var sl2 = suml2 - w1 * (l[i1].toInt() * l[i1].toInt()).toFloat()
                    var newL = (x[i1] * sl2 / slx).round().toInt()
                    newL = newL.clamp(-nmax, nmax - 1)
                    if (newL != l[i1].toInt()) {
                        slx += w1 * x[i1] * newL.toFloat()
                        sl2 += w1 * (newL * newL).toFloat()
                        if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                            l[i1] = newL.toByte()
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
        for (li2 in ArrayDeque(l)) {
            li2 += nmax.toByte()
        }
        return sumlx / suml2
    }
    for (i2 in 0 until n) {
        val li3 = (iscale * x[i2]).round().toInt()
        l[i2] = (li3.clamp(-nmax, nmax - 1) + nmax).toByte()
    }
    return 1.0f / iscale
}

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L744

/**
 * (scale, min)
 */
internal fun makeQkx3Quants(nmax: Int, x: List<Float>, weights: List<Float>?, rmin: Float, rdelta: Float, nstep: Int, useMad: Boolean): Pair<Float, Float> {
    val n = x.size
    val l: UByteArray = UByteArray(32) { 0u.toUByte() }
    val lAux: UByteArray = UByteArray(32) { 0u.toUByte() }
    var minVal = x[0]
    var maxVal = x[0]
    var sumW =
        when (weights) {
            null -> x[0] * x[0]
            else -> {
                val w = weights
                w[0]
            }
        }
    var sumX = sumW * x[0]
    for (i in 1 until n) {
        if (x[i] < minVal) {
            minVal = x[i]
        }
        if (x[i] > maxVal) {
            maxVal = x[i]
        }
        val w2 =
            when (weights) {
                null -> x[i] * x[i]
                else -> {
                    val w1 = weights
                    w1[i]
                }
            }
        sumW += w2
        sumX += w2 * x[i]
    }
    if (minVal > 0.0f) {
        minVal = 0.0f
    }
    if (maxVal <= minVal) {
        return Pair(0.0, -minVal)
    }
    var iscale = nmax.toFloat() / (maxVal - minVal)
    var scale = 1.0f / iscale
    var bestMad = 0.0
    for (i1 in 0 until n) {
        val lVal = nearestInt(iscale * (x[i1] - minVal)).clamp(0, nmax).toUByte()
        l[i1] = lVal
        val diff = scale * (lVal.toFloat()) + minVal - x[i1]
        val diff1 =
            if (useMad) {
                diff.abs()
            } else {
                diff * diff
            }
        val w4 =
            when (weights) {
                null -> x[i1] * x[i1]
                else -> {
                    val w3 = weights
                    w3[i1]
                }
            }
        bestMad += w4 * diff1
    }
    if (nstep < 1) {
        return Pair(scale, -minVal)
    }
    for (isMatch in 0..nstep) {
        iscale = (rmin + rdelta * isMatch.toFloat() + nmax.toFloat()) / (maxVal - minVal)
        var (sumL, sumL2, sumXl) = Triple(0.0, 0.0, 0.0)
        for (i2 in 0 until n) {
            val lVal1 = nearestInt(iscale * (x[i2] - minVal)).clamp(0, nmax).toUByte()
            lAux[i2] = lVal1
            val w6 =
                when (weights) {
                    null -> x[i2] * x[i2]
                    else -> {
                        val w5 = weights
                        w5[i2]
                    }
                }
            sumL += w6 * lVal1.toFloat()
            sumL2 += w6 * (lVal1.toFloat()).powi(2)
            sumXl += w6 * lVal1.toFloat() * x[i2]
        }
        val d = sumW * sumL2 - sumL * sumL
        if (d > 0.0) {
            var thisScale = (sumW * sumXl - sumX * sumL) / d
            var thisMin = (sumL2 * sumX - sumL * sumXl) / d
            if (thisMin > 0.0f) {
                thisMin = 0.0f
                thisScale = sumXl / sumL2
            }
            var mad = 0.0
            for (i3 in 0 until n) {
                val diff2 = thisScale * (lAux[i3].toFloat()) + thisMin - x[i3]
                val diff3 =
                    if (useMad) {
                        diff2.abs()
                    } else {
                        diff2 * diff2
                    }
                val w8 =
                    when (weights) {
                        null -> x[i3] * x[i3]
                        else -> {
                            val w7 = weights
                            w7[i3]
                        }
                    }
                mad += w8 * diff3
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

// https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/ggml/src/ggml-quants.c#L827
internal fun makeQpQuants(n: Int, nmax: UByte, x: List<Float>, l: MutableList<UByte>, quantWeights: List<Float>): Float {
    check(x.size == n)
    check(l.size == n)
    check(quantWeights.size == n)
    val max = x.iter().copied().fold(0.0, Float::max)
    if (max == 0.0) {
        l.forEach({ li -> li = 0 })
        return 0.0f
    }
    var iscale = nmax.toFloat() / max
    for ((xi, li) in (x.zip(l))) {
        li = nearestInt(iscale * xi).toUByte()
    }
    val scale = 1.0f / iscale
    var bestMse =
        ((x.zip(l)).zip(quantWeights))
            .map({ (tuple0, w) ->
                val (xi1, li1) = tuple0
                val diff = xi1 - scale * li1.toFloat()
                w * diff * diff
            })
            .sum<Float>()
    for (isMatch in -4..4) {
        if (isMatch == 0) {
            continue
        }
        val iscaleIs = (0.1f * isMatch.toFloat() + nmax.toFloat()) / max
        val scaleIs = 1.0f / iscaleIs
        val mse =
            (x.zip(quantWeights))
                .map({ (xi1, w) ->
                    var li1 = nearestInt(iscaleIs * xi1).toUByte()
                    li1 = li1.min(nmax)
                    val diff = xi1 - scaleIs * li1.toFloat()
                    w * diff * diff
                })
                .sum<Float>()
        if (mse < bestMse) {
            bestMse = mse
            iscale = iscaleIs
        }
    }
    var sumlx = 0.0
    var suml2 = 0.0
    for ((xi1, li1, w) in ((x.zip(l)).zip(quantWeights))) {
        var liNew = (iscale * xi1).round().toUByte()
        liNew = liNew.min(nmax)
        li1 = liNew
        sumlx += w * xi1 * liNew.toFloat()
        suml2 += w * (liNew.toFloat()).powi(2)
    }
    for (unused in 0 until 5) {
        var nChanged = 0
        for ((xi2, li2, w1) in ((x.zip(l)).zip(quantWeights))) {
            var slx = sumlx - w1 * xi2 * li2.toFloat()
            var sl2 = suml2 - w1 * (li2.toFloat()).powi(2)
            if (slx > 0.0f && sl2 > 0.0) {
                val newLi = (nearestInt(xi2 * sl2 / slx).toUByte()).min(nmax)
                if (newLi != li2) {
                    slx += w1 * xi2 * newLi.toFloat()
                    sl2 += w1 * (newLi.toFloat()).powi(2)
                    if (slx.powi(2) * suml2 > sumlx.powi(2) * sl2) {
                        li2 = newLi
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
