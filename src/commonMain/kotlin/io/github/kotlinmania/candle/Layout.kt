// port-lint: source layout.rs
package io.github.kotlinmania.candle

import kotlin.math.max

/** Tensor Layouts including contiguous or sparse strides */
public data class Layout(
    val shape: Shape,
    // The strides are given in number of elements and not in bytes.
    val stride: List<Int>,
    val startOffset: Int = 0,
) {
    private val self: Layout get() = this
    public val dims: List<Int> get() = shape.dims

    public companion object {
        public val Self: Companion get() = this
        public fun new(shape: Shape, stride: List<Int>, startOffset: Int): Layout {
            return Layout(
                shape,
                stride,
                startOffset,
            )
        }

        public fun contiguousWithOffset(shape: Shape, startOffset: Int = 0): Layout {
            val shape = shape.into()
            val stride = shape.strideContiguous()
            return Layout(
                shape,
                stride,
                startOffset,
            )
        }

        public fun contiguous(shape: Shape): Layout {
            return Self.contiguousWithOffset(shape, 0)
        }
    }

    public fun dims(): List<Int> {
        return self.shape.dims()
    }

    /** The dimension size for a specified dimension index. */
    public fun dim(dim: Int): Int {
        val dim = dim.toIndex(self.shape, "dim")
        return self.dims()[dim]
    }

    public fun shape(): Shape {
        return self.shape
    }

    public fun stride(): List<Int> {
        return self.stride
    }

    public fun startOffset(): Int {
        return self.startOffset
    }

    /**
     * Returns outer stride along `dim` if valid.
     *
     * Two conditions must hold:
     *  1. Inner dims `[dim..]` has standard contiguous strides.
     *  2. Outer dims `[..dim]` are contiguous among themselves, i.e.
     *     `stride[k] == dims[k+1] * stride[k+1]` for `k` in `0..dim-1`.
     *
     * When the tensor is fully contiguous this returns `Some(dims[dim..].product())`.
     */
    public fun outerStrideForDim(dim: Int): Int? {
        val dims = self.dims()
        val strides = self.stride()

        // 1. Inner `dims[dim..]` must have contiguous strides.
        var expected = 1
        for (i in (dim until dims.len()).rev()) {
            if (strides[i] != expected) {
                return null
            }
            expected *= dims[i]
        }

        if (dim == 0) {
            // No outer dims.
            // `expected = dims[dim..].product()`
            return expected
        }

        // 2. Outer `dims[0..dim]` must be internally contiguous.
        val outerStride = strides[dim - 1]
        var expectedOuter = outerStride
        for (k in (0 until dim - 1).rev()) {
            expectedOuter *= dims[k + 1]
            if (strides[k] != expectedOuter) {
                return null
            }
        }

        return outerStride
    }

    /**
     * Returns the appropriate start and stop offset if the data is stored in a C
     * contiguous (aka row major) way.
     */
    public fun contiguousOffsets(): ContiguousOffsets? {
        return if (self.isContiguous()) {
            val startO = self.startOffset
            ContiguousOffsets(startO, startO + self.shape.elemCount())
        } else {
            null
        }
    }

    /**
     * Returns true if the data is stored in a C contiguous (aka row major) way.
     * Note that this does not implies that the start offset is 0 or that there are no extra
     * elements at the end of the storage.
     */
    public fun isContiguous(): Boolean {
        return self.shape.isContiguous(self.stride)
    }

    /** Returns true if the data is stored in a Fortran contiguous (aka column major) way. */
    public fun isFortranContiguous(): Boolean {
        return self.shape.isFortranContiguous(self.stride)
    }

    public fun isScalar(): Boolean {
        val dims = self.dims()
        return dims.isEmpty() || dims.iter().all { d -> d == 1 }
    }

    /** Returns true if the data is actually a scalar during broadcast */
    public fun isScalarBroadcast(): Boolean {
        return self.stride().iter().all { s -> s == 0 }
    }

    public fun isScalarLike(): Boolean {
        return self.isScalar() || self.isScalarBroadcast()
    }

    public fun narrow(dim: Int, start: Int, len: Int): Layout {
        val dims = self.shape().dims()
        if (dim >= dims.len()) {
            throw CandleException.DimOutOfRange(
                shape = self.shape().clone(),
                dim = dim.toInt(),
                op = "narrow",
            )
                .bt()
        }
        if (start + len > dims[dim]) {
            throw CandleException.NarrowInvalidArgs(
                shape = self.shape.clone(),
                dim = dim,
                start = start,
                len = len,
                msg = "start + len > dim_len",
            )
                .bt()
        }
        val dimsMut = dims.toVec()
        dimsMut[dim] = len
        return Layout(
            shape = Shape.from(dimsMut),
            stride = self.stride.clone(),
            startOffset = self.startOffset + self.stride[dim] * start,
        )
    }

    public fun transpose(dim1: Int, dim2: Int): Layout {
        val rank = self.shape.rank()
        if (rank <= dim1 || rank <= dim2) {
            throw CandleException.UnexpectedNumberOfDims(
                expected = max(dim1, dim2),
                got = rank,
                shape = self.shape().clone(),
            )
                .bt()
        }
        val stride = self.stride().toVec()
        val dims = self.shape().dims().toVec()
        dims.swap(dim1, dim2)
        stride.swap(dim1, dim2)
        return Layout(
            shape = Shape.from(dims),
            stride = stride,
            startOffset = self.startOffset,
        )
    }

    public fun permute(idxs: List<Int>): Layout {
        val isPermutation =
            idxs.len() == self.shape.rank() && (0 until idxs.len()).all { i -> idxs.contains(i) }
        if (!isPermutation) {
            bail(
                "dimension mismatch in permute, tensor" +
                    " ${self.dims()}," +
                    " dims: $idxs",
            )
        }
        val stride = self.stride()
        val dims = self.shape().dims()
        val permStride = stride.toVec()
        val permDims = dims.toVec()
        for ((i, idx) in idxs.iter().enumerate()) {
            permStride[i] = stride[idx]
            permDims[i] = dims[idx]
        }
        return Layout(
            shape = Shape.from(permDims),
            stride = permStride,
            startOffset = self.startOffset,
        )
    }

    public fun broadcastAs(shape: Shape): Layout {
        val shape = shape.into()
        if (shape.rank() < self.shape().rank()) {
            throw CandleException.BroadcastIncompatibleShapes(
                srcShape = self.shape().clone(),
                dstShape = shape,
            )
                .bt()
        }
        val addedDims = shape.rank() - self.shape().rank()
        val stride = MutableList(addedDims) { 0 }
        for ((dstDim, srcPair) in shape.dims().subList(addedDims, shape.dims().len())
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        ) {
            val (srcDim, srcStride) = srcPair
            val s = if (dstDim == srcDim) {
                srcStride
            } else if (srcDim != 1) {
                throw CandleException.BroadcastIncompatibleShapes(
                    srcShape = self.shape().clone(),
                    dstShape = shape,
                ).bt()
            } else {
                0
            }
            stride.push(s)
        }
        return Layout(
            shape,
            stride,
            self.startOffset,
        )
    }

    public fun stridedIndex(): StridedIndex {
        return StridedIndex.fromLayout(self)
    }

    public fun stridedBlocks(): StridedBlocks {
        var blockLen = 1
        var contiguousDims = 0 // Counted from the right.
        for ((stride, dim) in self.stride().iter().zip(self.dims().iter()).rev()) {
            // Size-1 dimensions are trivially contiguous regardless of their stride.
            if (dim == 1) {
                contiguousDims += 1
                continue
            }
            if (stride != blockLen) {
                break
            }
            blockLen *= dim
            contiguousDims += 1
        }
        val indexDims = self.dims().len() - contiguousDims
        return when (indexDims) {
            0 -> StridedBlocks.SingleBlock(
                startOffset = self.startOffset,
                len = blockLen,
            )
            1 -> StridedBlocks.UniformBlocks(
                startOffset = self.startOffset,
                blockLen = blockLen,
                count = self.dims()[0],
                srcStride = self.stride[0],
            )
            else -> {
                val blockStartIndex = StridedIndex.new(
                    self.dims().subList(0, indexDims),
                    self.stride.subList(0, indexDims),
                    self.startOffset,
                )
                StridedBlocks.MultipleBlocks(
                    blockStartIndex,
                    blockLen,
                )
            }
        }
    }
}

public data class ContiguousOffsets(
    val start: Int,
    val end: Int,
) {
    public val first: Int get() = start
    public val second: Int get() = end
}

