// port-lint: source candle-core/src/layout.rs
package io.github.kotlinmania.candle

public data class ContiguousOffsets(
    val start: Int,
    val end: Int,
)

public data class Layout(
    val shape: Shape,
    val stride: List<Int>,
    val startOffset: Int = 0,
) {
    public val dims: List<Int> get() = shape.dims

    public fun dim(dimIndex: Int): Int = shape.dim(dimIndex)

    public fun outerStrideForDim(dim: Int): Int? {
        val dims = dims
        val strides = stride

        var expected = 1
        for (i in (dims.size - 1) downTo dim) {
            if (strides[i] != expected) {
                return null
            }
            expected *= dims[i]
        }

        if (dim == 0) {
            return expected
        }

        val outerStride = strides[dim - 1]
        var expectedOuter = outerStride
        for (k in (dim - 2) downTo 0) {
            expectedOuter *= dims[k + 1]
            if (strides[k] != expectedOuter) {
                return null
            }
        }

        return outerStride
    }

    public fun contiguousOffsets(): ContiguousOffsets? {
        if (isContiguous()) {
            return ContiguousOffsets(startOffset, startOffset + shape.elemCount())
        }
        return null
    }

    public fun isContiguous(): Boolean = shape.isContiguous(stride)

    public fun isFortranContiguous(): Boolean = shape.isFortranContiguous(stride)

    public fun isScalar(): Boolean {
        val dims = dims
        return dims.isEmpty() || dims.all { it == 1 }
    }

    public fun isScalarBroadcast(): Boolean = stride.all { it == 0 }

    public fun isScalarLike(): Boolean = isScalar() || isScalarBroadcast()

    public fun narrow(dim: Int, start: Int, len: Int): Layout {
        val dims = shape.dims
        if (dim < 0 || dim >= dims.size) {
            throw CandleException.DimOutOfRange(shape, dim, "narrow")
        }
        if (start + len > dims[dim]) {
            throw CandleException.NarrowInvalidArgs(shape, dim, start, len, "start + len > dim_len")
        }
        val newDims = dims.toMutableList()
        newDims[dim] = len
        return Layout(
            shape = Shape(newDims),
            stride = stride,
            startOffset = startOffset + stride[dim] * start,
        )
    }

    public fun transpose(dim1: Int, dim2: Int): Layout {
        val rank = shape.rank
        if (rank <= dim1 || rank <= dim2) {
            throw CandleException.UnexpectedNumberOfDims(maxOf(dim1, dim2), rank, shape)
        }
        val newStride = stride.toMutableList()
        val newDims = shape.dims.toMutableList()
        val tmpD = newDims[dim1]
        newDims[dim1] = newDims[dim2]
        newDims[dim2] = tmpD

        val tmpS = newStride[dim1]
        newStride[dim1] = newStride[dim2]
        newStride[dim2] = tmpS

        return Layout(
            shape = Shape(newDims),
            stride = newStride,
            startOffset = startOffset,
        )
    }

    public fun permute(idxs: List<Int>): Layout {
        val isPermutation = idxs.size == shape.rank && (0 until idxs.size).all { idxs.contains(it) }
        if (!isPermutation) {
            bail("dimension mismatch in permute, tensor $dims, dims: $idxs")
        }
        val permStride = ArrayList<Int>(idxs.size)
        val permDims = ArrayList<Int>(idxs.size)
        for (idx in idxs) {
            permStride.add(stride[idx])
            permDims.add(dims[idx])
        }
        return Layout(
            shape = Shape(permDims),
            stride = permStride,
            startOffset = startOffset,
        )
    }

    public fun broadcastAs(targetShape: Shape): Layout {
        if (targetShape.rank < shape.rank) {
            throw CandleException.BroadcastIncompatibleShapes(shape, targetShape)
        }
        val addedDims = targetShape.rank - shape.rank
        val newStride = ArrayList<Int>(targetShape.rank)
        for (i in 0 until addedDims) {
            newStride.add(0)
        }
        val srcDims = dims
        for (i in srcDims.indices) {
            val dstDim = targetShape.dims[addedDims + i]
            val srcDim = srcDims[i]
            val srcStride = stride[i]
            val s =
                if (dstDim == srcDim) {
                    srcStride
                } else if (srcDim != 1) {
                    throw CandleException.BroadcastIncompatibleShapes(shape, targetShape)
                } else {
                    0
                }
            newStride.add(s)
        }
        return Layout(
            shape = targetShape,
            stride = newStride,
            startOffset = startOffset,
        )
    }

    public fun stridedIndex(): StridedIndex = StridedIndex.fromLayout(this)

    public fun stridedBlocks(): StridedBlocks {
        var blockLen = 1
        var contiguousDims = 0
        val dims = dims
        for (i in dims.indices.reversed()) {
            val dim = dims[i]
            val s = stride[i]
            if (dim == 1) {
                contiguousDims++
                continue
            }
            if (s != blockLen) {
                break
            }
            blockLen *= dim
            contiguousDims++
        }
        val indexDims = dims.size - contiguousDims
        return when (indexDims) {
            0 -> StridedBlocks.SingleBlock(startOffset, blockLen)
            1 -> StridedBlocks.UniformBlocks(startOffset, blockLen, dims[0], stride[0])
            else -> {
                val blockStartIndex =
                    StridedIndex(
                        dims.subList(0, indexDims),
                        stride.subList(0, indexDims),
                        startOffset,
                    )
                StridedBlocks.MultipleBlocks(blockStartIndex, blockLen)
            }
        }
    }

    public companion object {
        public fun contiguous(shape: Shape): Layout = contiguousWithOffset(shape, 0)

        public fun contiguousWithOffset(shape: Shape, startOffset: Int = 0): Layout =
            Layout(shape, shape.strideContiguous(), startOffset)
    }
}
