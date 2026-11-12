// port-lint: source layout.rs
package io.github.kotlinmania.candle.layout

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.Dim
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.candle.stridedindex.StridedIndex
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.fromResidual

// Tensor Layouts including contiguous or sparse strides

public data class Layout(
    public var shape: Shape = Shape(),
    // The strides are given in number of elements and not in bytes.
    public var stride: List<Int> = emptyList(),
    public var startOffset: Int = 0,
) {
    public fun dims(): List<Int> = this.shape.dims()

    /**
     * The dimension size for a specified dimension index.
     */
    public fun <D : Dim> dim(dim: D): Result<Int> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape, "dim")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(this.dims()[dim1])
    }

    /**
     * Returns outer stride along `dim` if valid.
     *
     * Two conditions must hold:
     *  1. Inner dims `[dim ..]` has standard contiguous strides.
     *  2. Outer dims `[.. dim]` are contiguous among themselves, i.e.
     *     `stride[k] == dims[k + 1] * stride[k + 1]` for `k` in `0 .. dim - 1`.
     *
     * When the tensor is fully contiguous this returns `Some(dims[dim ..].product())`.
     */
    internal fun outerStrideForDim(dim: Int): Int? {
        val dims = this.dims()
        val strides = this.stride()
        // 1. Inner `dims[dim ..]` must have contiguous strides.
        var expected = 1
        for (i in (dim until dims.size).rev()) {
            if (strides[i] != expected) {
                return null
            }
            expected *= dims[i]
        }
        if (dim == 0) {
            // No outer dims.
            // `expected = dims[dim ..].product()`
            return expected
        }
        // 2. Outer `dims[0 .. dim]` must be internally contiguous.
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
     * Checks if more than one logical index lands on the same cell (or when we can't prove it does not)
     */
    public fun hasInternalOverlap(): Boolean = !this.range().isSomeAnd({ f -> f.injective })

    /**
     * Returns range of cells this layout can reach, and wether it reaches all of them.
     * Returns `None` on arithmetic overflow.
     */
    public fun range(): LayoutRange? {
        // Filter out dims <= 1 as their strides are irrelevant.
        val axes: List<Pair<Int, Int>> = (this.dims().zip(this.stride())).filter({ (d, _) -> d > 1 }).map({ (d, s) -> Pair(d, s) }).collect()
        axes.sortUnstableByKey({ (_, s) -> s })
        var span = 0
        var injective = true
        var dense = true
        for ((d, s) in axes) {
            if (s <= span) {
                // This dim can land on a cell another dim already reaches.
                injective = false
                dense = false
            } else if (s - span != 1) {
                // Has a gap
                dense = false
            }
            span = (((span).toLong() + (((((d - 1)).toLong() * (s).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt() ?: return null)).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt() ?: return null)
        }
        val lo = this.startOffset()
        return LayoutRange(lo = lo, hi = (((lo).toLong() + (span).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt() ?: return null), injective = injective, dense = dense)
    }

    /**
     * Relation between this layout and another
     */
    public fun relation(other: Layout): LayoutRelation {
        if (this == other) {
            return LayoutRelation.Identical
        }
        if (this.shape().elemCount() == 0 || other.shape().elemCount() == 0) {
            return LayoutRelation.Disjoint
        }
        // Extract [LayoutRange] from layout.
        val (a1, b1) =
            run {
                val matchVal = Pair(this.range(), other.range())
                when {
                    matchVal.first != null && matchVal.second != null -> {
                        val a = matchVal.first
                        val b = matchVal.second
                        Pair(a, b)
                    }
                    else -> {
                        return LayoutRelation.Unknown
                    }
                }
            }
        if (a1.separatedFrom(b1)) {
            return LayoutRelation.Disjoint
        }
        if (a1.denselyContains(b1) || b1.denselyContains(a1)) {
            return LayoutRelation.Overlapping
        }
        // We end up here when layouts ranges overlap and neither is dense, such as disjoint
        // column slices. Figuring out these cases is a bounded integer feasibility problem
        // over the strides. This is NP-hard in general. Cheap at common tensor ranks.
        // If we want to move more cases out of unknown into disjoint/overlapping it can be done using the
        // same approach as numpy: https://github.com/numpy/numpy/blob/main/numpy/_core/src/common/memOverlap.c
        return LayoutRelation.Unknown
    }

    /**
     * Returns the appropriate start and stop offset if the data is stored in a C
     * contiguous (aka row major) way.
     */
    public fun contiguousOffsets(): Pair<Int, Int>? =
        if (this.isContiguous()) {
            val startO = this.startOffset
            Pair(startO, startO + this.shape.elemCount())
        } else {
            null
        }

    /**
     * Returns true if the data is stored in a C contiguous (aka row major) way.
     * Note that this does not implies that the start offset is 0 or that there are no extra
     * elements at the end of the storage.
     */
    public fun isContiguous(): Boolean = this.shape.isContiguous(this.stride)

    /**
     * Returns true if the data is stored in a Fortran contiguous (aka column major) way.
     */
    public fun isFortranContiguous(): Boolean = this.shape.isFortranContiguous(this.stride)

    public fun isScalar(): Boolean {
        val dims = this.dims()
        return dims.isEmpty() || dims.all({ d -> d == 1 })
    }

    /**
     * Returns true if the data is actually a scalar during broadcast
     */
    public fun isScalarBroadcast(): Boolean = this.stride().all({ s -> s == 0 })

    public fun isScalarLike(): Boolean = this.isScalar() || this.isScalarBroadcast()

    public fun narrow(dim: Int, start: Int, len: Int): Result<Layout> {
        val dims = this.shape().dims()
        if (dim >= dims.len()) {
            when (val tryResult = Err(Error.DimOutOfRange(shape = this.shape().clone(), dim = dim, op = "narrow").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (start + len > dims[dim]) {
            when (val tryResult = Err(Error.NarrowInvalidArgs(shape = this.shape.clone(), dim = dim, start = start, len = len, msg = "start + len > dim_len").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val dims1 = dims.toVec()
        dims1[dim] = len
        return Ok(Layout(shape = Shape.from(dims1), stride = this.stride.toMutableList(), startOffset = this.startOffset + this.stride[dim] * start))
    }

    public fun transpose(dim1: Int, dim2: Int): Result<Layout> {
        val rank = this.shape.rank()
        if (rank <= dim1 || rank <= dim2) {
            when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = Int.max(dim1, dim2), got = rank, shape = this.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        var stride = this.stride().toVec()
        val dims = this.shape().dims().toVec()
        dims.swap(dim1, dim2)
        stride.swap(dim1, dim2)
        return Ok(Layout(shape = Shape.from(dims), stride = stride, startOffset = this.startOffset))
    }

    public fun permute(idxs: List<Int>): Result<Layout> {
        val isPermutation = idxs.size == this.shape.rank() && (0 until idxs.size).all({ i -> idxs.contains(i) })
        if (!isPermutation) {
            bail("dimension mismatch in permute, tensor {:?}, dims: {:?}", this.dims(), idxs)
        }
        val stride = this.stride()
        val dims = this.shape().dims()
        val permStride = stride.toVec()
        val permDims = dims.toVec()
        for ((i, idx) in idxs.enumerate()) {
            permStride[i] = stride[idx]
            permDims[i] = dims[idx]
        }
        return Ok(Layout(shape = Shape.from(permDims), stride = permStride, startOffset = this.startOffset))
    }

    public fun broadcastAs(shape: Shape): Result<Layout> {
        val shape1 = shape
        if (shape1.rank() < this.shape().rank()) {
            return Err(Error.BroadcastIncompatibleShapes(srcShape = this.shape().clone(), dstShape = shape1).bt())
        }
        val addedDims = shape1.rank() - this.shape().rank()
        var stride = mutableListOf(0, addedDims)
        for ((dstDim, srcDim, srcStride) in (((shape1.dims()).slice(addedDims until shape1.dims().size)).zip((this.dims().zip(this.stride()))))) {
            val s =
                if (dstDim == srcDim) {
                    srcStride
                } else if (srcDim != 1) {
                    return Err(Error.BroadcastIncompatibleShapes(srcShape = this.shape().clone(), dstShape = shape1).bt())
                } else {
                    0
                }
            stride.add(s)
        }
        return Ok(Layout(shape = shape1, stride = stride, startOffset = this.startOffset))
    }

    internal fun stridedIndex(): StridedIndex = StridedIndex.fromLayout(this)

    internal fun stridedBlocks(): StridedBlocks {
        var blockLen = 1
        var contiguousDims = 0
        // Counted from the right.
        for ((stride, dim) in (this.stride().zip(this.dims())).rev()) {
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
        val indexDims = this.dims().size - contiguousDims
        return when (indexDims) {
            0 -> StridedBlocks.SingleBlock(startOffset = this.startOffset, len = blockLen)
            1 -> StridedBlocks.UniformBlocks(startOffset = this.startOffset, blockLen = blockLen, count = this.dims()[0], srcStride = this.stride[0])
            else -> {
                val blockStartIndex = StridedIndex.new(((this.dims()).slice(0 until indexDims)), ((this.stride).slice(0 until indexDims)), this.startOffset)
                StridedBlocks.MultipleBlocks(blockStartIndex = blockStartIndex, blockLen = blockLen)
            }
        }
    }

    public companion object {
        public fun new(shape: Shape, stride: List<Int>, startOffset: Int): Layout = Layout(shape = shape, stride = stride, startOffset = startOffset)

        public fun contiguousWithOffset(shape: Shape, startOffset: Int): Layout {
            val shape1 = shape
            val stride = shape1.strideContiguous()
            return Layout(shape = shape1, stride = stride, startOffset = startOffset)
        }

        public fun contiguous(shape: Shape): Layout = Layout.contiguousWithOffset(shape, 0)
    }
}

/**
 * Describes the range of cells a [Layout] can reach within its allocation,
 * and whether it reaches all of them.
 */
public data class LayoutRange(
    /**
     * Lowest reachable cell. Equal to layout `startOffset`.
     */
    public var lo: Int = 0,
    /**
     * Highest reachable cell (inclusive).
     */
    public var hi: Int = 0,
    /**
     * Proven to map distinct logical indices to distinct cells.
     */
    public var injective: Boolean = false,
    /**
     * Indicates that layout occupies every cell in `lo ..= hi`.
     * Does not necessarily mean that layout is contiguous.
     */
    public var dense: Boolean = false,
) {
    /**
     * Bounding intervals cannot meet, which means these layouts are seperate.
     */
    public fun separatedFrom(other: LayoutRange): Boolean = this.hi < other.lo || other.hi < this.lo

    /**
     * If a dense layout contains another's `lo` there is overlap.
     */
    public fun denselyContains(other: LayoutRange): Boolean = this.dense && other.lo >= this.lo && other.lo <= this.hi

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<LayoutRange> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<LayoutRange> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 24L
                override fun readNative(addr: Long): LayoutRange =
                    LayoutRange(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 17L),
                    )
                override fun writeNative(addr: Long, value: LayoutRange) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.lo)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.hi)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 16L, value.injective)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 17L, value.dense)
                }
            }
    }
}

/**
 * How two layouts over the same allocation relate.
 */
public enum class LayoutRelation(
    public val value: UInt,
) {
    /**
     * Simple assignment. x[i] += x[i]
     */
    Identical(0u),

    /**
     * Completely distinct layouts.
     */
    Disjoint(1u),

    /**
     * Any kind of overlap.
     */
    Overlapping(2u),

    /**
     * Could not prove relation.
     */
    Unknown(3u),
}
