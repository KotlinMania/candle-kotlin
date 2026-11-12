// port-lint: source tensor_cat.rs
package io.github.kotlinmania.candle

/**
 * Concatenates two or more tensors along a particular dimension.
 *
 * All tensors must of the same rank, and the output will have
 * the same rank
 *
 * ```kotlin
 * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 * val b = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 *
 * val c = Tensor.cat(listOf(a, b), 0)
 *
 * val c1 = Tensor.cat(listOf(a, b), 1)
 * ```
 */
public fun Tensor.Companion.cat(
    args: List<Tensor>,
    dim: Int,
): Tensor {
    if (args.isEmpty()) {
        throw CandleException.OpRequiresAtLeastOneTensor("cat")
    }
    val arg0 = args[0]
    if (args.size == 1) {
        return arg0
    }
    val dimIdx = Shape.toIndex(dim, arg0.shape(), "cat")
    for (arg in args) {
        arg.checkDim(dimIdx, "cat")
    }
    for ((argIdx, arg) in args.withIndex()) {
        if (arg0.rank() != arg.rank()) {
            throw CandleException.UnexpectedNumberOfDims(
                expected = arg0.rank(),
                got = arg.rank(),
                shape = arg.shape(),
            )
        }
        for (dimI in 0 until arg0.rank()) {
            val v1 = arg0.dim(dimI)
            val v2 = arg.dim(dimI)
            if (dimI != dimIdx && v1 != v2) {
                throw CandleException.ShapeMismatchCat(
                    dim = dimI,
                    firstShape = arg0.shape(),
                    n = argIdx + 1,
                    nthShape = arg.shape(),
                )
            }
        }
    }
    val allCopy2dCompat =
        args.all { v ->
            v.isContiguous() || v.layout().outerStrideForDim(dimIdx) != null
        }
    return if (allCopy2dCompat) {
        catContiguous(args, dimIdx)
    } else if (dimIdx == 0) {
        cat0(args)
    } else {
        val transposedArgs = args.map { it.transpose(0, dimIdx) }
        val catResult = cat0(transposedArgs)
        catResult.transpose(0, dimIdx)
    }
}

/**
 * Concatenates two or more tensors along a particular dimension.
 *
 * All tensors must be of the same rank, and the output will have
 * the same rank.
 *
 * ```kotlin
 * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 * val b = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 *
 * val c = Tensor.cat(listOf(a, b), 0)
 * check(c.shape().dims() == listOf(4, 3))
 *
 * val c1 = Tensor.cat(listOf(a, b), 1)
 * check(c1.shape().dims() == listOf(2, 6))
 * ```
 */
public fun Tensor.Companion.cat(
    args: List<Tensor>,
    dim: Dim,
): Tensor {
    if (args.isEmpty()) {
        throw CandleException.OpRequiresAtLeastOneTensor("cat")
    }
    val dimIdx = dim.toIndex(args[0].shape(), "cat")
    return cat(args, dimIdx)
}

internal fun Tensor.Companion.cat0(args: List<Tensor>): Tensor {
    if (args.isEmpty()) {
        throw CandleException.OpRequiresAtLeastOneTensor("cat")
    }
    val arg0 = args[0]
    if (args.size == 1) {
        return arg0
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims
    val catDims = firstDims.toMutableList()
    catDims[0] = 0
    val offsets = mutableListOf(0)
    for ((argIdx, arg) in args.withIndex()) {
        if (arg.dtype() != dtype) {
            throw CandleException.DTypeMismatchBinaryOp(
                lhs = dtype,
                rhs = arg.dtype(),
                op = "cat",
            )
        }
        if (arg.device().location() != device.location()) {
            throw CandleException.DeviceMismatchBinaryOp(
                lhs = device.location(),
                rhs = arg.device().location(),
                op = "cat",
            )
        }
        if (rank != arg.rank()) {
            throw CandleException.UnexpectedNumberOfDims(
                expected = rank,
                got = arg.rank(),
                shape = arg.shape(),
            )
        }
        for (dimIdx in 0 until arg0.rank()) {
            val v1 = firstDims[dimIdx]
            val v2 = arg.dim(dimIdx)
            if (dimIdx == 0) {
                catDims[0] += v2
            }
            if (dimIdx != 0 && v1 != v2) {
                throw CandleException.ShapeMismatchCat(
                    dim = dimIdx,
                    firstShape = arg0.shape(),
                    n = argIdx + 1,
                    nthShape = arg.shape(),
                )
            }
        }
        val nextOffset = offsets.last() + arg.elemCount()
        offsets.add(nextOffset)
    }
    val shape = Shape(catDims)
    val op = BackpropOp.new(args) { a -> Op.Cat(a, 0) }
    val storage = device.allocUninit(shape, dtype)
    for ((i, arg) in args.withIndex()) {
        val offset = offsets[i]
        arg.storage().copyStridedSrc(storage, offset, arg.layout())
    }
    return fromStorage(storage, shape, op, false)
}

internal fun Tensor.Companion.catContiguous(
    args: List<Tensor>,
    dim: Int,
): Tensor {
    if (args.isEmpty()) {
        throw CandleException.OpRequiresAtLeastOneTensor("cat")
    }
    val arg0 = args[0]
    if (args.size == 1) {
        return arg0
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims
    val catDims = firstDims.toMutableList()
    catDims[dim] = 0
    for ((argIdx, arg) in args.withIndex()) {
        if (arg.dtype() != dtype) {
            throw CandleException.DTypeMismatchBinaryOp(
                lhs = dtype,
                rhs = arg.dtype(),
                op = "cat",
            )
        }
        if (arg.device().location() != device.location()) {
            throw CandleException.DeviceMismatchBinaryOp(
                lhs = device.location(),
                rhs = arg.device().location(),
                op = "cat",
            )
        }
        if (rank != arg.rank()) {
            throw CandleException.UnexpectedNumberOfDims(
                expected = rank,
                got = arg.rank(),
                shape = arg.shape(),
            )
        }
        for (dimIdx in 0 until arg0.rank()) {
            val v1 = firstDims[dimIdx]
            val v2 = arg.dim(dimIdx)
            if (dimIdx == dim) {
                catDims[dim] += v2
            }
            if (dimIdx != dim && v1 != v2) {
                throw CandleException.ShapeMismatchCat(
                    dim = dimIdx,
                    firstShape = arg0.shape(),
                    n = argIdx + 1,
                    nthShape = arg.shape(),
                )
            }
        }
    }
    val catTargetDimLen = catDims[dim]
    var blockSize = 1
    for (i in (dim + 1) until catDims.size) {
        blockSize *= catDims[i]
    }
    val shape = Shape(catDims)
    val op = BackpropOp.new(args) { a -> Op.Cat(a, dim) }
    val storage = device.allocUninit(shape, dtype)
    var dstO = 0
    for (arg in args) {
        val argDims = arg.shape().dims
        var d1 = 1
        for (i in 0 until dim) {
            d1 *= argDims[i]
        }
        val d2 = blockSize * argDims[dim]
        val dstS = blockSize * catTargetDimLen
        val srcO = arg.layout().startOffset
        val srcS =
            if (arg.isContiguous()) {
                d2
            } else {
                arg.layout().outerStrideForDim(dim) ?: d2
            }
        arg.storage().copy2d(storage, d1, d2, srcS, dstS, srcO, dstO)
        dstO += d2
    }
    return fromStorage(storage, shape, op, false)
}

/**
 * Set the values on `this` using values from `src`. The copy starts at the specified
 * `offset` for the target dimension `dim` on `this`.
 * `this` and `src` must have the same shape except on dimension `dim` where the `this` size
 * has to be greater than or equal to `offset` plus the `src` size.
 *
 * Note that this modifies `this` in place and as such is not compatible with
 * back-propagation.
 */
public fun Tensor.sliceSet(
    src: Tensor,
    dim: Int,
    offset: Int,
) {
    val dimIdx = Shape.toIndex(dim, shape(), "slice-set")
    if (!isContiguous() || !src.isContiguous()) {
        throw CandleException.RequiresContiguous("slice-set")
    }
    if (sameStorage(src)) {
        bail("cannot use slice_set when self and src share their storage")
    }
    if (dtype() != src.dtype()) {
        throw CandleException.DTypeMismatchBinaryOp(
            lhs = dtype(),
            rhs = src.dtype(),
            op = "slice-set",
        )
    }
    if (device().location() != src.device().location()) {
        throw CandleException.DeviceMismatchBinaryOp(
            lhs = device().location(),
            rhs = src.device().location(),
            op = "slice-set",
        )
    }
    if (rank() != src.rank()) {
        throw CandleException.UnexpectedNumberOfDims(
            expected = rank(),
            got = src.rank(),
            shape = shape(),
        )
    }
    for (dimI in 0 until rank()) {
        val v1 = dim(dimI)
        val v2 = src.dim(dimI)
        if (dimI == dimIdx && v2 + offset > v1) {
            bail("shape mismatch on target dim, dst: $v1, src: $v2 + $offset")
        }
        if (dimI != dimIdx && v1 != v2) {
            bail("shape mismatch on dim $dimI, $v1 <> $v2")
        }
    }
    var blockSize = 1
    for (i in (dimIdx + 1) until src.dims().size) {
        blockSize *= src.dim(i)
    }
    var d1 = 1
    for (i in 0 until dimIdx) {
        d1 *= src.dim(i)
    }
    val d2 = blockSize * src.dim(dimIdx)
    val dstO = layout().startOffset + offset * blockSize
    val srcO = src.layout().startOffset
    src.storage().copy2d(
        storage(),
        d1,
        d2,
        d2,
        blockSize * dim(dimIdx),
        srcO,
        dstO,
    )
}

/**
 * Set the values on `this` using values from `src`. The copy starts at the specified
 * `offset` for the target dimension `dim` on `this`.
 * `this` and `src` must have the same shape except on dimension `dim` where the `this` size
 * has to be greater than or equal to `offset` plus the `src` size.
 *
 * Note that this modifies `this` in place and as such is not compatible with
 * back-propagation.
 */
public fun Tensor.sliceSet(
    src: Tensor,
    dim: Dim,
    offset: Int,
) {
    val dimIdx = dim.toIndex(shape(), "slice-set")
    sliceSet(src, dimIdx, offset)
}
