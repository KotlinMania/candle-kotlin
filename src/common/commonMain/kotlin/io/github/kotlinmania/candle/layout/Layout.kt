// port-lint: source layout.rs
package io.github.kotlinmania.candle.layout

import io.github.kotlinmania.candle.Dim
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.candle.stridedindex.StridedIndex

// Tensor Layouts including contiguous or sparse strides

public open class Layout(
    public var shape: Shape,
    // The strides are given in number of elements and not in bytes.
    public var stride: List<Int>,
    public var startOffset: Int,
) {
    public open fun dims(): List<Int> = this.shape.dims()

    /**
     * The dimension size for a specified dimension index.
     */
    public open fun <D : Dim> dim(dim: D): Result<Int> {
        val dim1 = dim.toIndex(this.shape, "dim").getOrElse { return Result.failure(it) }
        return Result.success(this.dims()[dim1])
    }

    public open fun shape(): Shape = this.shape

    public open fun stride(): List<Int> = this.stride

    public open fun startOffset(): Int = this.startOffset

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
    internal open fun outerStrideForDim(dim: Int): Int? {
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
    public open fun hasInternalOverlap(): Boolean = !this.range().isSomeAnd({ f -> f.injective })

    /**
     * Returns range of cells this layout can reach, and wether it reaches all of them.
     * Returns `None` on arithmetic overflow.
     */
    public open fun range(): LayoutRange? {
        // Filter out dims <= 1 as their strides are irrelevant.
        val axes: List<Pair<Int, Int>>  = (this.dims().zip(this.stride())).filter({ (d, _) -> d > 1 }).map({ (d, s) -> Pair(d, s) }).collect()
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
    public open fun relation(other: Layout): LayoutRelation {
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
                    else -> { return LayoutRelation.Unknown }
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
    public open fun contiguousOffsets(): Pair<Int, Int>? =
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
    public open fun isContiguous(): Boolean = this.shape.isContiguous(this.stride)

    /**
     * Returns true if the data is stored in a Fortran contiguous (aka column major) way.
     */
    public open fun isFortranContiguous(): Boolean = this.shape.isFortranContiguous(this.stride)

    public open fun isScalar(): Boolean {
        val dims = this.dims()
        return dims.isEmpty() || dims.all({ d -> d == 1 })
    }

    /**
     * Returns true if the data is actually a scalar during broadcast
     */
    public open fun isScalarBroadcast(): Boolean = this.stride().all({ s -> s == 0 })

    public open fun isScalarLike(): Boolean = this.isScalar() || this.isScalarBroadcast()

    public open fun narrow(dim: Int, start: Int, len: Int): Result<Layout> {
        val dims = this.shape().dims()
        if (dim >= dims.len()) {
            Result.failure(Error.DimOutOfRange(shape = this.shape().clone(), dim = dim, op = "narrow").bt()).getOrElse { return Result.failure(it) }
        }
        if (start + len > dims[dim]) {
            Result.failure(Error.NarrowInvalidArgs(shape = this.shape.clone(), dim = dim, start = start, len = len, msg = "start + len > dim_len").bt()).getOrElse { return Result.failure(it) }
        }
        val dims1 = dims.toVec()
        dims1[dim] = len
        return Result.success(Layout(shape = Shape.from(dims1), stride = this.stride.toMutableList(), startOffset = this.startOffset + this.stride[dim] * start))
    }

    public open fun transpose(dim1: Int, dim2: Int): Result<Layout> {
        val rank = this.shape.rank()
        if (rank <= dim1 || rank <= dim2) {
            Result.failure(Error.UnexpectedNumberOfDims(expected = Int.max(dim1, dim2), got = rank, shape = this.shape().clone()).bt()).getOrElse { return Result.failure(it) }
        }
        var stride = this.stride().toVec()
        val dims = this.shape().dims().toVec()
        dims.swap(dim1, dim2)
        stride.swap(dim1, dim2)
        return Result.success(Layout(shape = Shape.from(dims), stride = stride, startOffset = this.startOffset))
    }

    public open fun permute(idxs: List<Int>): Result<Layout> {
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
        return Result.success(Layout(shape = Shape.from(permDims), stride = permStride, startOffset = this.startOffset))
    }

    public open fun broadcastAs(shape: Shape): Result<Layout> {
        val shape1 = shape
        if (shape1.rank() < this.shape().rank()) {
            return Result.failure(Error.BroadcastIncompatibleShapes(srcShape = this.shape().clone(), dstShape = shape1).bt())
        }
        val addedDims = shape1.rank() - this.shape().rank()
        var stride = mutableListOf(0, addedDims)
        for ((dstDim, srcDim, srcStride) in (shape1.dims()[addedDims until shape1.dims().size].zip((this.dims().zip(this.stride()))))) {
            val s =
                if (dstDim == srcDim) {
                    srcStride
                } else if (srcDim != 1) {
                    return Result.failure(Error.BroadcastIncompatibleShapes(srcShape = this.shape().clone(), dstShape = shape1).bt())
                } else {
                    0
                }
            stride.add(s)
        }
        return Result.success(Layout(shape = shape1, stride = stride, startOffset = this.startOffset))
    }

    internal open fun stridedIndex(): StridedIndex = StridedIndex.fromLayout(this)

    internal open fun stridedBlocks(): StridedBlocks {
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
                val blockStartIndex = StridedIndex.new(this.dims()[0 until indexDims], this.stride[0 until indexDims], this.startOffset)
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
public open class LayoutRange(
    /**
     * Lowest reachable cell. Equal to layout `startOffset`.
     */
    public var lo: Int,
    /**
     * Highest reachable cell (inclusive).
     */
    public var hi: Int,
    /**
     * Proven to map distinct logical indices to distinct cells.
     */
    public var injective: Boolean,
    /**
     * Indicates that layout occupies every cell in `lo ..= hi`.
     * Does not necessarily mean that layout is contiguous.
     */
    public var dense: Boolean,
) {
    /**
     * Bounding intervals cannot meet, which means these layouts are seperate.
     */
    public open fun separatedFrom(other: LayoutRange): Boolean = this.hi < other.lo || other.hi < this.lo

    /**
     * If a dense layout contains another's `lo` there is overlap.
     */
    public open fun denselyContains(other: LayoutRange): Boolean = this.dense && other.lo >= this.lo && other.lo <= this.hi
}

/**
 * How two layouts over the same allocation relate.
 */
public sealed class LayoutRelation {
    /**
     * Simple assignment. x[i] += x[i]
     */
    public data object Identical : LayoutRelation()
    /**
     * Completely distinct layouts.
     */
    public data object Disjoint : LayoutRelation()
    /**
     * Any kind of overlap.
     */
    public data object Overlapping : LayoutRelation()
    /**
     * Could not prove relation.
     */
    public data object Unknown : LayoutRelation()

    public fun __rustEq(other: LayoutRelation): Boolean = this == other
}
