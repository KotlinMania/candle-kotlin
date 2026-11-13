package io.github.kotlinmania.candle.cpubackend

import io.github.kotlinmania.candle.dtype.IntDType
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.nditer.NdIter
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.numtraits.Float
import io.github.kotlinmania.rand.prelude.*
import io.github.kotlinmania.rayon.prelude.*

public fun <I, T> whereImpl(ndIter: NdIter, ysToSet: MutableList<T>, pred: List<I>, t: List<T>, f: List<T>, predFn: (Int, Int, Int) -> Int, tFn: (Int, Int, Int) -> Int, fFn: (Int, Int, Int) -> Int) where I : IntDType {
    val innerSize = ndIter.innerSize
    val (innerPs, innerTs, innerFs) = ndIter.innerStrides
    var dstOff = 0
    for ((pOff, tOff, fOff) in ndIter) {
        for (i in 0 until innerSize) {
            val ep = pred[predFn(pOff, i, innerPs)]
            val et = t[tFn(tOff, i, innerTs)]
            val ef = f[fFn(fOff, i, innerFs)]
            ysToSet[dstOff + i] =
                if (ep.isTrue()) {
                    et
                } else {
                    ef
                }
        }
        dstOff += innerSize
    }
}

public fun <T> copy2d(src: List<T>, dst: MutableList<T>, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
    // Both are contiguous - one memcpy covers everything.
    if (srcStride1 == d2 && dstStride1 == d2) {
        val srcStart = srcOffset
        val dstStart = dstOffset
        ((dst).slice(dstStart until dstStart + d1 * d2)).copyFromSlice(((src).slice(srcStart until srcStart + d1 * d2)))
        return
    }
    var srcIdx = srcOffset
    var dstIdx = dstOffset
    for (unused in 0 until d1) {
        ((dst).slice(dstIdx until dstIdx + d2)).copyFromSlice(((src).slice(srcIdx until srcIdx + d2)))
        srcIdx += srcStride1
        dstIdx += dstStride1
    }
}

public fun <T> copyStridedSrc(src: List<T>, dst: MutableList<T>, dstOffset: Int, srcL: Layout) {
    run {
        val matchVal = srcL.stridedBlocks()
        when {
            matchVal is StridedBlocks.SingleBlock -> {
                val startOffset = matchVal.startOffset
                val len = matchVal.len
                ((dst).slice(dstOffset until dstOffset + len)).copyFromSlice(((src).slice(startOffset until startOffset + len)))
            }
            matchVal is StridedBlocks.UniformBlocks -> {
                val startOffset1 = matchVal.startOffset
                val blockLen = matchVal.blockLen
                val count = matchVal.count
                val srcStride = matchVal.srcStride
                copy2d(src, dst, count, blockLen, srcStride, blockLen, startOffset1, dstOffset)
            }
            matchVal is StridedBlocks.MultipleBlocks && matchVal.blockLen == 1 -> {
                val blockStartIndex = matchVal.blockStartIndex
                val n = blockStartIndex.len()
                val dst1 = ((dst).slice(dstOffset until dstOffset + n))
                for ((dstElem, srcIndex) in (dst1.iterMut().zip(blockStartIndex))) {
                    dstElem = src[srcIndex]
                }
            }
            else -> {
                val blockStartIndex1 = matchVal.blockStartIndex
                val blockLen1 = matchVal.blockLen
                val nBlocks = blockStartIndex1.len()
                val dst2 = ((dst).slice(dstOffset until dstOffset + nBlocks * blockLen1))
                var dstIndex = 0
                for (srcIndex1 in blockStartIndex1) {
                    ((dst2).slice(dstIndex until dstIndex + blockLen1)).copyFromSlice(((src).slice(srcIndex1 until srcIndex1 + blockLen1)))
                    dstIndex += blockLen1
                }
            }
        }
    }
}

public fun <T : io.github.kotlinmania.numtraits.Float> elu(v: T, alpha: T, oneTFn: (Any?) -> Any?): T =
    if (v.isSignPositive()) {
        v
    } else {
        (v.exp() - oneTFn()) * alpha
    }

private const val MAP_DTYPE_TEMPLATE: String = "when \$storage {\$(CpuStorage. \$dtypes(e)=> CpuStorage. \$dtypes(\$fn(e)), )* s=> Err(Error.UnsupportedDTypeForOp(s.dtype(), \$name).bt())?, }"

public fun mapDtype(name: String = "", storage: String = "", fn: String = "", dtypes: String = ""): String =
    arrayOf("\$name" to name, "\$storage" to storage, "\$fn" to fn, "\$dtypes" to dtypes).fold(MAP_DTYPE_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
