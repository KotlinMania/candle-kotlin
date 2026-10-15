// port-lint: source cpu_backend/conv2d.rs
package io.github.kotlinmania.candle

public class Conv2D(
    public val params: ParamsConv2D,
) : Map2 {
    override val op: String = "conv2d"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val p = params

        // Specialization: pick the best algorithm based on parameters.
        // 1x1 convolutions with stride=1, padding=0, dilation=1
        if (p.kH == 1 && p.kW == 1 && p.stride == 1 && p.padding == 0 && p.dilation == 1) {
            return conv2d1x1(p, v1, l1, v2, l2)
        } else if (p.kH == 1 && p.kW == 1) {
            // Other 1x1 convolutions for now are assumed faster with full im2col,
            // although with large enough input size, tiled will start beating it.
            return conv2dIm2colGemm(p, v1, l1, v2, l2)
        }

        // No fast path, fallback to default general impl.
        return when (defaultConv2dImpl) {
            Conv2dImpl.TiledIm2Col -> conv2dTiled(p, v1, l1, v2, l2)
            Conv2dImpl.Direct -> conv2dDirect(p, v1, l1, v2, l2)
            Conv2dImpl.FullIm2Col -> conv2dIm2colGemm(p, v1, l1, v2, l2)
        }
    }
}

public enum class Conv2dImpl {
    TiledIm2Col,
    FullIm2Col,
    Direct,
}

public val defaultConv2dImpl: Conv2dImpl = Conv2dImpl.TiledIm2Col

/**
 * Fast kernel for 1x1 convolutions with stride=1, padding=0, dilation=1
 * These are just matrix multiplications: [cOut, cIn] @ [cIn, b*h*w] -> [cOut, b*h*w].
 */
public fun conv2d1x1(
    p: ParamsConv2D,
    inp: List<Double>,
    inpL: Layout,
    k: List<Double>,
    kL: Layout,
): List<Double> {
    val inpStart = inpL.startOffset()
    val inpStride = inpL.stride()
    val inpS0 = inpStride[0]
    val inpS1 = inpStride[1]
    val inpS2 = inpStride[2]
    val inpS3 = inpStride[3]
    val kStart = kL.startOffset()
    val kStride = kL.stride()
    val kS0 = kStride[0]
    val kS1 = kStride[1]
    val outH = p.outH()
    val outW = p.outW()

    val spatialSize = outH * outW
    val dst = MutableList(p.bSize * p.cOut * spatialSize) { 0.0 }
    val kReshaped: List<Double> = if (kS0 == p.cIn && kS1 == 1) {
        // Already contiguous, use slice directly
        k.subList(kStart, kStart + p.cOut * p.cIn)
    } else {
        // Reshape kernel to [cOut, cIn]
        val reshaped = ArrayList<Double>(p.cOut * p.cIn)
        for (cOutIdx in 0 until p.cOut) {
            for (cInIdx in 0 until p.cIn) {
                val kIdx = kStart + cOutIdx * kS0 + cInIdx * kS1
                reshaped.add(k[kIdx])
            }
        }
        reshaped
    }
    val kLayout = Layout.contiguous(Shape(p.cOut, p.cIn))

    // Process each batch
    for (bIdx in 0 until p.bSize) {
        // Reshape input to [cIn, h*w] for this batch
        val inpReshaped = ArrayList<Double>(p.cIn * spatialSize)
        for (cInIdx in 0 until p.cIn) {
            for (hIdx in 0 until p.iH) {
                for (wIdx in 0 until p.iW) {
                    val inpIdx = inpStart + bIdx * inpS0 + cInIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                    inpReshaped.add(inp[inpIdx])
                }
            }
        }
        val inpLayout = Layout.contiguous(Shape(p.cIn, spatialSize))

        // Perform matmul: [cOut, cIn] @ [cIn, spatialSize] -> [cOut, spatialSize]
        val matmul = MatMul(Dims4(1, p.cOut, spatialSize, p.cIn))
        val result = matmul.f(kReshaped, kLayout, inpReshaped, inpLayout)

        // Copy result to output
        val outOffset = bIdx * p.cOut * spatialSize
        for (i in result.indices) {
            dst[outOffset + i] = result[i]
        }
    }

    return dst
}

/**
 * General tiled convolution implementation using gemm.
 *
 * Similar to full im2col, but instead of materializing the full matrix, we process input/output in tiles, in parallel.
 */
public fun conv2dTiled(
    p: ParamsConv2D,
    inp: List<Double>,
    inpL: Layout,
    k: List<Double>,
    kL: Layout,
): List<Double> {
    val inpStart = inpL.startOffset()
    val (inpS0, inpS1, inpS2, inpS3) = dims4(inpL.stride())
    val kStart = kL.startOffset()
    val (kS0, kS1, kS2, kS3) = dims4(kL.stride())
    val outH = p.outH()
    val outW = p.outW()

    // Output shape: [bSize, cOut, outH, outW].
    val dst = MutableList(p.bSize * p.cOut * outH * outW) { 0.0 }

    // Convert NCHW input to NHWC layout for tiled im2col.
    val contS0 = p.iH * p.iW * p.cIn
    val contS1 = p.iW * p.cIn
    val contS2 = p.cIn
    val inpCont = DoubleArray(p.bSize * p.cIn * p.iH * p.iW)
    for (bIdx in 0 until p.bSize) {
        for (hIdx in 0 until p.iH) {
            for (wIdx in 0 until p.iW) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = inpStart + bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                    val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                    inpCont[dstIdx] = inp[srcIdx]
                }
            }
        }
    }

    // shape of k: [cOut, cIn, kH, kW]
    // strides of k: [kS0, kS1, kS2, kS3]
    // For matmul, we need flattened k in shape [cOut, kH * kW * cIn]
    // with stride [kH * kW * cIn, 1]
    val kSize = p.cIn * p.kH * p.kW
    val kFlat = ArrayList<Double>(p.cOut * kSize)
    for (dstCIdx in 0 until p.cOut) {
        for (kh in 0 until p.kH) {
            for (kw in 0 until p.kW) {
                for (cInIdx in 0 until p.cIn) {
                    val kIdx = kStart + dstCIdx * kS0 + cInIdx * kS1 + kh * kS2 + kw * kS3
                    kFlat.add(k[kIdx])
                }
            }
        }
    }
    // kLayout: [cOut, kSize] with stride [kSize, 1]
    val kLayout = Layout.contiguous(Shape(p.cOut, kSize))

    // tileCapacity is number of output pixels (outH * outW) per tile.
    // Higher tile size can be faster due to better usage of gemm,
    // but lower tile sizes enable bigger parallelism across tiles.
    // This parameter is impactful and may be dynamic or even runtime tunable in the future.
    val tileCapacity = 512

    val totalOutPixels = outH * outW

    // Process batches and tiles.
    for (bIdx in 0 until p.bSize) {
        val inpOffset = bIdx * contS0
        val outBatchOffset = bIdx * (p.cOut * outH * outW)

        val numTiles = if (totalOutPixels == 0) 0 else (totalOutPixels + tileCapacity - 1) / tileCapacity
        for (tIdx in 0 until numTiles) {
            // Determine actual tile size (may be smaller at the end)
            val tileStart = tIdx * tileCapacity
            val tileEnd = minOf(tileStart + tileCapacity, totalOutPixels)
            val tileSize = tileEnd - tileStart

            // Precompute output coordinates.
            // Used in both im2col extraction and writing output.
            val outCoords = ArrayList<Pair<Int, Int>>(tileSize)
            for (idx in tileStart until tileEnd) {
                outCoords.add(Pair(idx / outW, idx % outW))
            }

            // Build im2col tile: [kSize, tileSize]
            // This represents the input patches needed for this tile of outputs
            val colTile = MutableList(kSize * tileSize) { 0.0 }

            for ((tileIdx, coord) in outCoords.withIndex()) {
                val outY = coord.first
                val outX = coord.second
                // Extract the im2col patch for this output position
                for (cIn in 0 until p.cIn) {
                    var patchOffset = cIn
                    for (kh in 0 until p.kH) {
                        val inY = outY * p.stride + kh * p.dilation - p.padding
                        if (inY < 0 || inY >= p.iH) {
                            // Padding: already zero
                            patchOffset += p.cIn * p.kW
                            continue
                        }
                        for (kw in 0 until p.kW) {
                            val inX = outX * p.stride + kw * p.dilation - p.padding

                            if (inX >= 0 && inX < p.iW) {
                                val inpIdx = inpOffset + inY * contS1 + inX * contS2 + cIn
                                val colIdx = patchOffset * tileSize + tileIdx
                                colTile[colIdx] = inpCont[inpIdx]
                            }
                            // Move to next position (skip cIn channels)
                            patchOffset += p.cIn
                        }
                    }
                }
            }

            // Now perform matmul: kCache [cOut, kSize] @ colTile [kSize, tileSize]
            val matmul = MatMul(Dims4(1, p.cOut, tileSize, kSize))

            // Layouts for matmul
            // kFlat layout: [cOut, kSize] with stride [kSize, 1]
            // colTile layout: [kSize, tileSize] with stride [tileSize, 1]
            val colLayout = Layout.contiguous(Shape(kSize, tileSize))

            // Perform matmul
            val result = matmul.f(kFlat, kLayout, colTile, colLayout)

            // Copy results to output: result is [cOut, tileSize]
            for ((tileIdx, coord) in outCoords.withIndex()) {
                val outY = coord.first
                val outX = coord.second
                val dstBase = outBatchOffset + outY * outW + outX

                for (cOutIdx in 0 until p.cOut) {
                    val dstIdx = dstBase + cOutIdx * (outH * outW)
                    val resultIdx = cOutIdx * tileSize + tileIdx
                    // Each batch processes a distinct region of the output buffer.
                    // Within each batch, tiles process non-overlapping output positions.
                    // Therefore, no two threads will write to the same dstIdx.
                    dst[dstIdx] = result[resultIdx]
                }
            }
        }
    }

    return dst
}

/**
 * General direct convolution impl. Decently fast for small inputs and kernels, but loses to full/tiled gemm.
 */
public fun conv2dDirect(
    p: ParamsConv2D,
    inp: List<Double>,
    inpL: Layout,
    k: List<Double>,
    kL: Layout,
): List<Double> {
    val inpStart = inpL.startOffset()
    val (inpS0, inpS1, inpS2, inpS3) = dims4(inpL.stride())
    val kStart = kL.startOffset()
    val (kS0, kS1, kS2, kS3) = dims4(kL.stride())
    val outH = p.outH()
    val outW = p.outW()

    // Output shape: [bSize, cOut, outH, outW].
    val dst = MutableList(p.bSize * p.cOut * outH * outW) { 0.0 }

    // Convert NCHW input to NHWC layout for direct convolution.
    val contS0 = p.iH * p.iW * p.cIn
    val contS1 = p.iW * p.cIn
    val contS2 = p.cIn
    val inpCont = DoubleArray(p.bSize * p.cIn * p.iH * p.iW)
    for (bIdx in 0 until p.bSize) {
        for (hIdx in 0 until p.iH) {
            for (wIdx in 0 until p.iW) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = inpStart + bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                    val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                    inpCont[dstIdx] = inp[srcIdx]
                }
            }
        }
    }
    val inpContLen = inpCont.size

    val kCache = ArrayList<DoubleArray>(p.cOut)
    for (dstCIdx in 0 until p.cOut) {
        val row = DoubleArray(p.kH * p.kW * p.cIn)
        var rowIdx = 0
        for (kwKh in 0 until (p.kH * p.kW)) {
            val offsetH = kwKh / p.kW
            val offsetW = kwKh % p.kW
            for (cInIdx in 0 until p.cIn) {
                row[rowIdx++] = k[kStart + dstCIdx * kS0 + cInIdx * kS1 + offsetH * kS2 + offsetW * kS3]
            }
        }
        kCache.add(row)
    }

    for (bIdx in 0 until p.bSize) {
        for (offsetH in 0 until p.kH) {
            for (offsetW in 0 until p.kW) {
                val kOffset = offsetH * p.kW + offsetW

                for (dstCIdx in 0 until p.cOut) {
                    val kRow = kCache[dstCIdx]
                    val kRowStart = kOffset * p.cIn
                    val baseDstIdx = dstCIdx * outW * outH
                    val batchDstIdx = baseDstIdx + bIdx * p.cOut * outH * outW
                    val batchSrcIdx = bIdx * contS0

                    for (dstH in 0 until outH) {
                        val srcH = p.stride * dstH + offsetH * p.dilation
                        if (srcH < p.padding || srcH >= p.iH + p.padding) {
                            continue
                        }
                        val adjSrcH = srcH - p.padding
                        val hDstIdx = batchDstIdx + dstH * outW
                        val hSrcIdx = batchSrcIdx + adjSrcH * contS1

                        for (dstW in 0 until outW) {
                            val srcW = p.stride * dstW + offsetW * p.dilation
                            if (srcW < p.padding || srcW >= p.iW + p.padding) {
                                continue
                            }
                            val adjSrcW = srcW - p.padding
                            val dstIdx = hDstIdx + dstW
                            val inpIdx1 = hSrcIdx + adjSrcW * contS2
                            val inpIdx2 = minOf(inpIdx1 + p.cIn, inpContLen)
                            val count = inpIdx2 - inpIdx1
                            var d = 0.0
                            for (c in 0 until count) {
                                d += inpCont[inpIdx1 + c] * kRow[kRowStart + c]
                            }
                            dst[dstIdx] += d
                        }
                    }
                }
            }
        }
    }

    return dst
}

public fun allocUninitVec(size: Int): MutableList<Double> = MutableList(size) { 0.0 }

/**
 * Full im2col + gemm convolution implementation.
 *
 * For large inputs im2col and copyStridedSrc for output gets expensive.
 */
public fun conv2dIm2colGemm(
    p: ParamsConv2D,
    inp: List<Double>,
    inpL: Layout,
    kernel: List<Double>,
    kernelL: Layout,
): List<Double> {
    val op = Im2Col(
        hK = p.kH,
        wK = p.kW,
        padding = p.padding,
        stride = p.stride,
        dilation = p.dilation,
    )
    val col = op.f(inp, inpL)
    val b = p.bSize
    val n = p.cOut
    val hOut = p.outH()
    val wOut = p.outW()
    val k = op.hK * op.wK * p.cIn
    val m = hOut * wOut
    val colL = Layout.contiguous(Shape(b, m, k))
    val res: List<Double> = if (kernelL.isContiguous()) {
        val kLayout = Layout.contiguousWithOffset(Shape(1, n, k), kernelL.startOffset())
            .transpose(1, 2)
            .broadcastAs(Shape(b, k, n))
        MatMul(Dims4(b, m, n, k)).f(col, colL, kernel, kLayout)
    } else {
        // Make the kernel contiguous if not already the case.
        val kernelC = allocUninitVec(kernelL.shape().elemCount())
        copyStridedSrc(kernel, kernelC, 0, kernelL)
        val kLayout = Layout.contiguousWithOffset(Shape(1, n, k), kernelL.startOffset())
            .transpose(1, 2)
            .broadcastAs(Shape(b, k, n))
        MatMul(Dims4(b, m, n, k)).f(col, colL, kernelC, kLayout)
    }
    val resL = Layout.contiguous(Shape(b, hOut, wOut, p.cOut))
        .transpose(1, 2)
        .transpose(1, 3)
    val resT = allocUninitVec(resL.shape().elemCount())
    copyStridedSrc(res, resT, 0, resL)
    return resT
}
