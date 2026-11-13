// port-lint: source cpu_backend/conv2d.rs
package io.github.kotlinmania.candle.cpubackend.conv2d

import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.cpubackend.Im2Col
import io.github.kotlinmania.candle.cpubackend.MatMul
import io.github.kotlinmania.candle.cpubackend.copyStridedSrc
import io.github.kotlinmania.candle.cpubackend.utils.Map2
import io.github.kotlinmania.candle.dims4
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.numtraits.Num
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast

internal class Conv2D(
    internal var value: ParamsConv2D = ParamsConv2D(),
) : Map2 {
    public var field0: ParamsConv2D
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(inp: List<T>, inpL: Layout, k: List<T>, kL: Layout): Result<List<T>> where T : WithDType, T : Num {
        val p = this.value
        // Specialization: pick the best algorithm based on parameters.
        // 1x1 convolutions with stride=1, padding=0, dilation=1
        if (p.kH == 1 && p.kW == 1 && p.stride == 1 && p.padding == 0 && p.dilation == 1) {
            return conv2d1x1(p, inp, inpL, k, kL)
        } else if (p.kH == 1 && p.kW == 1) {
            // Other 1x1 convolutions for now are assumed faster with full im2col,
            // although with large enough input size, tiled will start beating it.
            return conv2dIm2colGemm(p, inp, inpL, k, kL)
        }
        // Note other cases
        // No fast path, fallback to default general impl.
        return when (DEFAULT_CONV2D_IMPL) {
            Conv2dImpl.TiledIm2Col -> conv2dTiled(p, inp, inpL, k, kL)
            Conv2dImpl.Direct -> conv2dDirect(p, inp, inpL, k, kL)
            Conv2dImpl.FullIm2Col -> conv2dIm2colGemm(p, inp, inpL, k, kL)
        }
    }

    public companion object {
        public val OP: String = "conv2d"
    }
}

public enum class Conv2dImpl(
    public val value: UInt,
) {
    TiledIm2Col(0u),
    FullIm2Col(1u),
    Direct(2u),
}

public val DEFAULT_CONV2D_IMPL: Conv2dImpl = Conv2dImpl.TiledIm2Col

/**
 * Fast kernel for 1x1 convolutions with stride=1, padding=0, dilation=1
 * These are just matrix multiplications: [cOut, cIn] @ [cIn, b*h*w] -> [cOut, b*h*w].
 */
public fun <T> conv2d1x1(p: ParamsConv2D, inp: List<T>, inpL: Layout, k: List<T>, kL: Layout): Result<List<T>> where T : WithDType, T : Num {
    val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
    val inpStride = inpL.stride()
    val (inpS0, inpS1, inpS2, inpS3) = listOf(inpStride[0], inpStride[1], inpStride[2], inpStride[3])
    val k1 = ((k).slice(kL.startOffset() until k.size))
    val kStride = kL.stride()
    val (kS0, kS1) = Pair(kStride[0], kStride[1])
    val (outH, outW) = Pair(p.outH(), p.outW())
    val spatialSize = outH * outW
    val dst = mutableListOf(T.zero(), p.bSize * p.cOut * spatialSize)
    val kReshaped1: List<T> =
        if (kS0 == p.cIn && kS1 == 1) {
            // Already contiguous, use slice directly
            ((k1).slice(0 until p.cOut * p.cIn))
        } else {
            // Reshape kernel to [cOut, cIn]
            val kReshaped = ArrayList<T>(p.cOut * p.cIn)
            (0 until p.cOut).forEach({ cOutIdx ->
                (0 until p.cIn).forEach({ cInIdx ->
                    val kIdx = cOutIdx * kS0 + cInIdx * kS1
                    kReshaped.add(k1[kIdx])
                })
            })
            kReshaped
        }
    val kLayout = Layout.contiguous(Pair(p.cOut, p.cIn))
    // Process each batch
    when (
        val tryResult =
            (0 until p.bSize).intoParIter().tryForEach(
                closure@{ bIdx ->
                    // Reshape input to [cIn, h*w] for this batch
                    var inpReshaped = ArrayList(p.cIn * spatialSize)
                    for (cInIdx in 0 until p.cIn) {
                        for (hIdx in 0 until p.iH) {
                            for (wIdx in 0 until p.iW) {
                                val inpIdx = bIdx * inpS0 + cInIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                                inpReshaped.add(inp1[inpIdx])
                            }
                        }
                    }
                    val inpLayout = Layout.contiguous(Pair(p.cIn, spatialSize))
                    // Perform matmul: [cOut, cIn] @ [cIn, spatialSize] -> [cOut, spatialSize]
                    val matmul = MatMul(listOf(1, p.cOut, spatialSize, p.cIn))
                    val result =
                        when (val tryResult = matmul.f(kReshaped1, kLayout, inpReshaped, inpLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return@closure Result.Err(tryResult.error)
                        }
                    // Copy result to output
                    val outOffset = bIdx * p.cOut * spatialSize
                    for ((i, r) in result.iter().mapIndexed { index, item -> Pair(index, item) }) {
                        run {
                            val ptr = mutPtrFromRustCast(dst.asPtr().add(outOffset + i))
                            (ObjectSlotLayout<T>()).writeNative((ptr).rawAddress, r)
                        }
                    }
                    Result.Ok(Unit)
                },
            )
    ) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(dst)
}

/**
 * General tiled convolution implementation using gemm.
 *
 * Similar to full im2col, but instead of materializing the full matrix, we process input/output in tiles, in parallel.
 */
public fun <T> conv2dTiled(p: ParamsConv2D, inp: List<T>, inpL: Layout, k: List<T>, kL: Layout): Result<List<T>> where T : WithDType, T : Num {
    // TILE_SIZE is number of output pixels (outH * outW) per tile.
    // Higher tile size can be faster due to better usage of gemm,
    // but lower tile sizes enable bigger parallelism across tiles.
    // This parameter is impactful and may be dynamic or even runtime tunable in the future.
    val TILE_SIZE: Int = 512
    val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
    val (inpS0, inpS1, inpS2, inpS3) =
        when (val tryResult = dims4(inpL.stride())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val k1 = ((k).slice(kL.startOffset() until k.size))
    val (kS0, kS1, kS2, kS3) =
        when (val tryResult = dims4(kL.stride())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (outH, outW) = Pair(p.outH(), p.outW())
    // Output shape: [bSize, cOut, outH, outW].
    val dst = mutableListOf(T.zero(), p.bSize * p.cOut * outH * outW)
    // Convert NCHW input to NHWC layout for tiled im2col.
    val contS0 = p.iH * p.iW * p.cIn
    val contS1 = p.iW * p.cIn
    val contS2 = p.cIn
    val inpCont = mutableListOf(T.zero(), p.bSize * p.cIn * p.iH * p.iW)
    for (bIdx in 0 until p.bSize) {
        for (hIdx in 0 until p.iH) {
            for (wIdx in 0 until p.iW) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                    val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                    inpCont[dstIdx] = inp1[srcIdx]
                }
            }
        }
    }
    // shape of k: [cOut, cIn, kH, kW]
    // strides of k: [kS0, kS1, kS2, kS3]
    // For matmul, we need flattened k in shape [cOut, kH * kW * cIn]
    // with stride [kH * kW * cIn, 1]
    val kSize = p.cIn * p.kH * p.kW
    val kFlat = ArrayList<T>(p.cOut * kSize)
    for (dstCIdx in 0 until p.cOut) {
        for (kh in 0 until p.kH) {
            for (kw in 0 until p.kW) {
                for (cInIdx in 0 until p.cIn) {
                    val kIdx = dstCIdx * kS0 + cInIdx * kS1 + kh * kS2 + kw * kS3
                    kFlat.add(k1[kIdx])
                }
            }
        }
    }
    // kLayout: [cOut, kSize] with stride [kSize, 1]
    val kLayout = Layout.contiguous(Pair(p.cOut, kSize))
    val totalOutPixels = outH * outW
    // Process batches and tiles in parallel using rayon.
    when (
        val tryResult =
            (0 until p.bSize).intoParIter().tryForEach({ bIdx1 ->
                val inpOffset = bIdx1 * contS0
                val outBatchOffset = bIdx1 * (p.cOut * outH * outW)
                val numTiles = totalOutPixels.divCeil(TILE_SIZE)
                (0 until numTiles).intoParIter().tryForEach(
                    closure1@{ tileIdx ->
                        // Determine actual tile size (may be smaller at the end) {
                        val tileStart = tileIdx * TILE_SIZE
                        val tileEnd = (tileStart + TILE_SIZE).min(totalOutPixels)
                        val tileSize = tileEnd - tileStart
                        // Precompute output coordinates.
                        // Used in both im2col extraction and writing output.
                        val outCoords: Vec = (tileStart until tileEnd).map({ idx -> Pair(idx / outW, idx % outW) }).collect()
                        // Build im2col tile: [kSize, tileSize]
                        // This represents the input patches needed for this tile of outputs
                        var colTile = mutableListOf(T.zero(), kSize * tileSize)
                        for ((tileIdx1, outY, outX) in outCoords.iter().mapIndexed { index, item -> Pair(index, item) }) {
                            // Extract the im2col patch for this output position
                            for (cIn in 0 until p.cIn) {
                                var patchOffset = cIn
                                for (kh1 in 0 until p.kH) {
                                    val inY = (outY * p.stride + kh1 * p.dilation) - p.padding
                                    if (inY < 0 || inY >= p.iH) {
                                        // Padding: already zero
                                        patchOffset += p.cIn * p.kW
                                        continue
                                    }
                                    for (kw1 in 0 until p.kW) {
                                        val inX = (outX * p.stride + kw1 * p.dilation) - p.padding
                                        if (inX >= 0 && inX < p.iW) {
                                            val inY1 = inY
                                            val inX1 = inX
                                            val inpIdx = inpOffset + inY1 * contS1 + inX1 * contS2 + cIn
                                            val colIdx = patchOffset * tileSize + tileIdx1
                                            colTile[colIdx] = inpCont[inpIdx]
                                        }
                                        // Move to next position (skip cIn channels)
                                        patchOffset += p.cIn
                                    }
                                }
                            }
                        }
                        // Now perform matmul: kCache [cOut, kSize] @ colTile [kSize, tileSize]
                        val matmul = MatMul(listOf(1, p.cOut, tileSize, kSize))
                        // Layouts for matmul
                        // kFlat layout: [cOut, kSize] with stride [kSize, 1]
                        // colTile layout: [kSize, tileSize] with stride [tileSize, 1]
                        val colLayout = Layout.contiguous(Pair(kSize, tileSize))
                        // Perform matmul
                        val result =
                            when (val tryResult = matmul.f(kFlat, kLayout, colTile, colLayout)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return@closure1 Result.Err(tryResult.error)
                            }
                        // Copy results to output: result is [cOut, tileSize]
                        for ((tileIdx2, outY1, outX1) in outCoords.iter().mapIndexed { index, item -> Pair(index, item) }) {
                            val dstBase = outBatchOffset + outY1 * outW + outX1
                            for (cOutIdx in 0 until p.cOut) {
                                val dstIdx1 = dstBase + cOutIdx * (outH * outW)
                                val resultIdx = cOutIdx * tileSize + tileIdx2
                                // SAFETY: Each batch processes a distinct region of the output buffer.
                                // Within each batch, tiles process non-overlapping output positions.
                                // Therefore, no two threads will write to the same dstIdx.
                                run {
                                    val ptr = mutPtrFromRustCast(dst.asPtr().add(dstIdx1))
                                    (ObjectSlotLayout<T>()).writeNative((ptr).rawAddress, result[resultIdx])
                                }
                            }
                        }
                        Result.Ok(Unit)
                    },
                )
            })
    ) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(dst)
}

/**
 * General direct convolution impl. Decently fast for small inputs and kernels, but loses to full/tiled gemm.
 */
public fun <T> conv2dDirect(p: ParamsConv2D, inp: List<T>, inpL: Layout, k: List<T>, kL: Layout, zeroTFn: (Any?) -> Any?, vecDotTFn: (Any?) -> Any?): Result<List<T>> where T : WithDType, T : Num {
    val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
    val (inpS0, inpS1, inpS2, inpS3) =
        when (val tryResult = dims4(inpL.stride())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val k1 = ((k).slice(kL.startOffset() until k.size))
    val (kS0, kS1, kS2, kS3) =
        when (val tryResult = dims4(kL.stride())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (outH, outW) = Pair(p.outH(), p.outW())
    // Output shape: [bSize, cOut, outH, outW].
    val dst = mutableListOf(zeroTFn(), p.bSize * p.cOut * outH * outW)
    // Convert NCHW input to NHWC layout for direct convolution.
    val contS0 = p.iH * p.iW * p.cIn
    val contS1 = p.iW * p.cIn
    val contS2 = p.cIn
    val inpCont = mutableListOf(zeroTFn(), p.bSize * p.cIn * p.iH * p.iW)
    for (bIdx in 0 until p.bSize) {
        for (hIdx in 0 until p.iH) {
            for (wIdx in 0 until p.iW) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                    val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                    inpCont[dstIdx] = inp1[srcIdx]
                }
            }
        }
    }
    val inpContLen = inpCont.size
    val kCache: List<List<T>> =
        (0 until p.cOut)
            .map({ dstCIdx ->
                (0 until p.kH * p.kW)
                    .flatMap({ kwKh ->
                        val offsetH = kwKh / p.kW
                        val offsetW = kwKh % p.kW
                        (0 until p.cIn).map({ cInIdx ->
                            k1[dstCIdx * kS0 + cInIdx * kS1 + offsetH * kS2 + offsetW * kS3]
                        })
                    })
                    .collect()
            })
            .toMutableList()
    for (bIdx1 in 0 until p.bSize) {
        for (offsetH in 0 until p.kH) {
            for (offsetW in 0 until p.kW) {
                val kOffset = offsetH * p.kW + offsetW
                (0 until p.cOut).intoParIter().forEach({ dstCIdx ->
                    val kCont = ((kCache[dstCIdx]).slice(kOffset * p.cIn until (kOffset + 1) * p.cIn))
                    val baseDstIdx = dstCIdx * outW * outH
                    val batchDstIdx = baseDstIdx + bIdx1 * p.cOut * outH * outW
                    val batchSrcIdx = bIdx1 * contS0
                    for (dstH in 0 until outH) {
                        val srcH = p.stride * dstH + offsetH * p.dilation
                        if (srcH < p.padding || srcH >= p.iH + p.padding) {
                            continue
                        }
                        val srcH1 = srcH - p.padding
                        val hDstIdx = batchDstIdx + dstH * outW
                        val hSrcIdx = batchSrcIdx + srcH1 * contS1
                        for (dstW in 0 until outW) {
                            val srcW = p.stride * dstW + offsetW * p.dilation
                            if (srcW < p.padding || srcW >= p.iW + p.padding) {
                                continue
                            }
                            val srcW1 = srcW - p.padding
                            val dstIdx1 = hDstIdx + dstW
                            val inpIdx1 = hSrcIdx + srcW1 * contS2
                            val inpIdx2 = (inpIdx1 + p.cIn).min(inpContLen)
                            val inpCont1 = ((inpCont).slice(inpIdx1 until inpIdx2))
                            val d = zeroTFn()
                            run {
                                vecDotTFn(inpCont1.asPtr(), kCont.asPtr(), d, p.cIn)
                                val ptr = mutPtrFromRustCast(dst.asPtr().add(dstIdx1))
                                (ptr).let { targetPtr ->
                                    val targetLayout = ObjectSlotLayout<T>()
                                    var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                                    currentValue += (d)
                                    targetLayout.writeNative(targetPtr.rawAddress, currentValue)
                                }
                            }
                        }
                    }
                })
            }
        }
    }
    return Result.Ok(dst)
}

public fun <T : WithDType> allocUninitVec(size: Int): List<T> {
    val v = ArrayList<T>(size)
    v.setLen(size)
    return v
}

/**
 * Full im2col + gemm convolution implementation.
 *
 * For large inputs im2col and copyStridedSrc for output gets expensive.
 */
public fun <T> conv2dIm2colGemm(p: ParamsConv2D, inp: List<T>, inpL: Layout, kernel: List<T>, kernelL: Layout): Result<List<T>> where T : WithDType, T : Num {
    val op = Im2Col(hK = p.kH, wK = p.kW, padding = p.padding, stride = p.stride, dilation = p.dilation)
    val col =
        when (val tryResult = op.f(inp, inpL)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val b = p.bSize
    val n = p.cOut
    val (hOut, wOut) = Pair(p.outH(), p.outW())
    val k = op.hK * op.wK * p.cIn
    val m = hOut * wOut
    val colL = Layout.contiguous(Triple(b, m, k))
    val res: List<T> =
        if (kernelL.isContiguous()) {
            val kernelL1 =
                when (
                    val tryResult =
                        when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.broadcastAs(Triple(b, k, n))
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = MatMul(listOf(b, m, n, k)).f(col, colL, kernel, kernelL1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            // copy_strided_src_ materializes the kernel into kernelC starting at index 0, so the
            // matmul layout must use offset 0, not the original (strided) kernel's start offset.
            val kernelC = allocUninitVec(kernelL.shape().elemCount())
            copyStridedSrc(kernel, kernelC, 0, kernelL)
            val kernelL2 =
                when (
                    val tryResult =
                        when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), 0).transpose(1, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.broadcastAs(Triple(b, k, n))
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = MatMul(listOf(b, m, n, k)).f(col, colL, kernelC, kernelL2)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
    val resL =
        when (
            val tryResult =
                when (val tryResult = Layout.contiguous(listOf(b, hOut, wOut, p.cOut)).transpose(1, 2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(1, 3)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val resT = allocUninitVec(resL.shape().elemCount())
    copyStridedSrc(res, resT, 0, resL)
    return Result.Ok(resT)
}
