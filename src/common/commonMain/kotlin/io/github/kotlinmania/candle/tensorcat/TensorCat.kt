// port-lint: source tensor_cat.rs
package io.github.kotlinmania.candle.tensorcat

import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.shape.Dim
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage
import io.github.kotlinmania.rustlang.core.result.unwrapOr

/**
 * Concatenates two or more tensors along a particular dimension.
 *
 * All tensors must of the same rank, and the output will have
 * the same rank
 *
 * ```kotlin
 * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
 * val b = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
 * val c = Tensor.cat(listOf(a, b), 0).getOrElse { return Result.failure(it) }
 * check(c.shape().dims() == listOf(4, 3))
 * val c1 = Tensor.cat(listOf(a, b), 1).getOrElse { return Result.failure(it) }
 * check(c1.shape().dims() == listOf(2, 6))
 * ```
 */
public fun <D : Dim> Tensor.Companion.cat(args: List<Tensor>, dim: D): Result<Tensor> {
    if (args.isEmpty()) {
        when (val tryResult = Result.Err(Error.OpRequiresAtLeastOneTensor(op = "cat").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.Ok(arg0.clone())
    }
    val dim1 =
        when (val tryResult = dim.toIndex(arg0.shape(), "cat")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (arg in args) {
        when (val tryResult = arg.asRef().checkDim(dim1, "cat")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    for ((argIdx, arg1) in args.mapIndexed { index, item -> Pair(index, item) }) {
        val arg2 = arg1.asRef()
        if (arg0.rank() != arg2.rank()) {
            when (val tryResult = Result.Err(Error.UnexpectedNumberOfDims(expected = arg0.rank(), got = arg2.rank(), shape = arg2.shape().clone()).bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        for ((dimIdx, v1, v2) in (
            arg0
                .shape()
                .dims()
                .iter()
                .zip(arg2.shape().dims().iter())
        ).mapIndexed { index, item -> Pair(index, item) }) {
            if (dimIdx != dim1 && v1 != v2) {
                when (val tryResult = Result.Err(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg2.shape().clone()).bt())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        }
    }
    val allCopy2dCompat =
        args.all({ v ->
            val t = v.asRef()
            t.isContiguous() || t.layout().outerStrideForDim(dim1) != null
        })
    return if (allCopy2dCompat) {
        Tensor.catContiguous(args, dim1)
    } else if (dim1 == 0) {
        Tensor.cat0(args)
    } else {
        val args1: List<Tensor> =
            when (val tryResult = args.map({ a -> a.asRef().transpose(0, dim1) }).collect<Result<Vec>>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val cat =
            when (val tryResult = Tensor.cat0(args1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        cat.transpose(0, dim1)
    }
}

public fun Tensor.Companion.cat0(args: List<Tensor>): Result<Tensor> {
    if (args.isEmpty()) {
        when (val tryResult = Result.Err(Error.OpRequiresAtLeastOneTensor(op = "cat").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.Ok(arg0.clone())
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims()
    val catDims = firstDims.toVec()
    catDims[0] = 0
    val offsets = mutableListOf(0)
    for ((argIdx, arg) in args.mapIndexed { index, item -> Pair(index, item) }) {
        val arg1 = arg.asRef()
        if (arg1.dtype() != dtype) {
            when (val tryResult = Result.Err(Error.DTypeMismatchBinaryOp(lhs = dtype, rhs = arg1.dtype(), op = "cat").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        if (arg1.device().location() != device.location()) {
            when (val tryResult = Result.Err(Error.DeviceMismatchBinaryOp(lhs = device.location(), rhs = arg1.device().location(), op = "cat").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        if (rank != arg1.rank()) {
            when (val tryResult = Result.Err(Error.UnexpectedNumberOfDims(expected = rank, got = arg1.rank(), shape = arg1.shape().clone()).bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        for ((dimIdx, v1, v2) in (
            arg0
                .shape()
                .dims()
                .iter()
                .zip(arg1.shape().dims().iter())
        ).mapIndexed { index, item -> Pair(index, item) }) {
            if (dimIdx == 0) {
                catDims[0] += v2
            }
            if (dimIdx != 0 && v1 != v2) {
                when (val tryResult = Result.Err(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg1.shape().clone()).bt())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        }
        val nextOffset =
            when (val tryResult = offsets.last().context("empty offsets")) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } + arg1.elemCount()
        offsets.add(nextOffset)
    }
    val shape = Shape.from(catDims)
    val op = BackpropOp.new(args, { args1 -> Op.Cat(args1, 0) })
    val storage =
        when (val tryResult = device.allocUninit(shape, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for ((arg2, offset) in (args.zip(ArrayDeque(offsets)))) {
        val arg3 = arg2.asRef()
        when (val tryResult = arg3.storage().copyStridedSrc(storage, offset, arg3.layout())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(fromStorage(storage, shape, op, false))
}

public fun Tensor.Companion.catContiguous(args: List<Tensor>, dim: Int): Result<Tensor> {
    if (args.isEmpty()) {
        when (val tryResult = Result.Err(Error.OpRequiresAtLeastOneTensor(op = "cat").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.Ok(arg0.clone())
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims()
    val catDims = firstDims.toVec()
    catDims[dim] = 0
    for ((argIdx, arg) in args.mapIndexed { index, item -> Pair(index, item) }) {
        val arg1 = arg.asRef()
        if (arg1.dtype() != dtype) {
            when (val tryResult = Result.Err(Error.DTypeMismatchBinaryOp(lhs = dtype, rhs = arg1.dtype(), op = "cat").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        if (arg1.device().location() != device.location()) {
            when (val tryResult = Result.Err(Error.DeviceMismatchBinaryOp(lhs = device.location(), rhs = arg1.device().location(), op = "cat").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        if (rank != arg1.rank()) {
            when (val tryResult = Result.Err(Error.UnexpectedNumberOfDims(expected = rank, got = arg1.rank(), shape = arg1.shape().clone()).bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        for ((dimIdx, v1, v2) in (
            arg0
                .shape()
                .dims()
                .iter()
                .zip(arg1.shape().dims().iter())
        ).mapIndexed { index, item -> Pair(index, item) }) {
            if (dimIdx == dim) {
                catDims[dim] += v2
            }
            if (dimIdx != dim && v1 != v2) {
                when (val tryResult = Result.Err(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg1.shape().clone()).bt())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        }
    }
    val catTargetDimLen = catDims[dim]
    val blockSize: Int = catDims.iter().skip(1 + dim).product()
    val shape = Shape.from(catDims)
    val op = BackpropOp.new(args, { args1 -> Op.Cat(args1, dim) })
    val storage =
        when (val tryResult = device.allocUninit(shape, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    var dstO = 0
    for (arg2 in args.iter()) {
        val arg3 = arg2.asRef()
        val argDims = arg3.shape().dims()
        val d1: Int = argDims.iter().take(dim).product()
        val d2 = blockSize * argDims[dim]
        val dstS = blockSize * catTargetDimLen
        val srcO = arg3.layout().startOffset()
        val srcS =
            if (arg3.isContiguous()) {
                d2
            } else {
                arg3.layout().outerStrideForDim(dim).unwrapOr(d2)
            }
        when (val tryResult = arg3.storage().copy2d(storage, d1, d2, srcS, dstS, srcO, dstO)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        dstO += d2
    }
    return Result.Ok(fromStorage(storage, shape, op, false))
}

/**
 * Set the values on `self` using values from `src`. The copy starts at the specified
 * `offset` for the target dimension `dim` on `self`.
 * `self` and `src` must have the same shape except on dimension `dim` where the `self` size
 * has to be greater than or equal to `offset` plus the `src` size.
 *
 * Note that this modifies `self` in place and as such is not compatible with
 * back-propagation.
 */
public fun <D : Dim> Tensor.sliceSet(src: Tensor, dim: D, offset: Int): Result<Unit> {
    val dim1 =
        when (val tryResult = dim.toIndex(this.shape(), "slice-set")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (!this.isContiguous() || !src.isContiguous()) {
        when (val tryResult = Result.Err(Error.RequiresContiguous(op = "slice-set").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    if (this.sameStorage(src)) {
        bail("cannot use slice_set when self and src share their storage")
    }
    if (this.dtype() != src.dtype()) {
        when (val tryResult = Result.Err(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = src.dtype(), op = "slice-set").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    if (this.device().location() != src.device().location()) {
        when (val tryResult = Result.Err(Error.DeviceMismatchBinaryOp(lhs = this.device().location(), rhs = src.device().location(), op = "slice-set").bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    if (this.rank() != src.rank()) {
        when (val tryResult = Result.Err(Error.UnexpectedNumberOfDims(expected = this.rank(), got = src.rank(), shape = this.shape().clone()).bt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    for ((dimIdx, v1, v2) in (this.dims().iter().zip(src.dims().iter())).mapIndexed { index, item -> Pair(index, item) }) {
        if (dimIdx == dim1 && v2 + offset > v1) {
            bail("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
        }
        if (dimIdx != dim1 && v1 != v2) {
            bail("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
        }
    }
    val blockSize: Int =
        src
            .dims()
            .iter()
            .skip(1 + dim1)
            .product()
    val d1: Int =
        src
            .dims()
            .iter()
            .take(dim1)
            .product()
    val d2 = blockSize * src.dims()[dim1]
    val dstO = this.layout().startOffset() + offset * blockSize
    val srcO = src.layout().startOffset()
    when (val tryResult = src.storage().copy2d(this.storageMut(), d1, d2, d2, blockSize * this.dims()[dim1], srcO, dstO)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}
