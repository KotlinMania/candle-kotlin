// port-lint: source tensor_cat.rs
package io.github.kotlinmania.candle.tensorcat

import io.github.kotlinmania.candle.Cat
import io.github.kotlinmania.candle.Dim
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.fromStorage
import io.github.kotlinmania.candle.new
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor

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
public fun <A : AsRef<Tensor>, D : Dim> Tensor.Companion.cat(args: List<A>, dim: D): Result<Tensor> {
    if (args.isEmpty()) {
        Result.failure(Error.OpRequiresAtLeastOneTensor(op = "cat").bt()).getOrElse { return Result.failure(it) }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.success(arg0.clone())
    }
    val dim1 = dim.toIndex(arg0.shape(), "cat").getOrElse { return Result.failure(it) }
    for (arg in args) {
        arg.asRef().checkDim(dim1, "cat").getOrElse { return Result.failure(it) }
    }
    for ((argIdx, arg1) in args.enumerate()) {
        val arg2 = arg1.asRef()
        if (arg0.rank() != arg2.rank()) {
            Result.failure(Error.UnexpectedNumberOfDims(expected = arg0.rank(), got = arg2.rank(), shape = arg2.shape().clone()).bt()).getOrElse { return Result.failure(it) }
        }
        for ((dimIdx, v1, v2) in (arg0.shape().dims().iter().zip(arg2.shape().dims().iter())).enumerate()) {
            if (dimIdx != dim1 && v1 != v2) {
                Result.failure(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg2.shape().clone()).bt()).getOrElse { return Result.failure(it) }
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
        val args1: List<Tensor> = args.map({ a -> a.asRef().transpose(0, dim1) }).collect<Result<Vec>>().getOrElse { return Result.failure(it) }
        val cat = Tensor.cat0(args1).getOrElse { return Result.failure(it) }
        cat.transpose(0, dim1)
    }
}

public fun <A : AsRef<Tensor>> Tensor.Companion.cat0(args: List<A>): Result<Tensor> {
    if (args.isEmpty()) {
        Result.failure(Error.OpRequiresAtLeastOneTensor(op = "cat").bt()).getOrElse { return Result.failure(it) }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.success(arg0.clone())
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims()
    val catDims = firstDims.toVec()
    catDims[0] = 0
    val offsets = mutableListOf(0)
    for ((argIdx, arg) in args.enumerate()) {
        val arg1 = arg.asRef()
        if (arg1.dtype() != dtype) {
            Result.failure(Error.DTypeMismatchBinaryOp(lhs = dtype, rhs = arg1.dtype(), op = "cat").bt()).getOrElse { return Result.failure(it) }
        }
        if (arg1.device().location() != device.location()) {
            Result.failure(Error.DeviceMismatchBinaryOp(lhs = device.location(), rhs = arg1.device().location(), op = "cat").bt()).getOrElse { return Result.failure(it) }
        }
        if (rank != arg1.rank()) {
            Result.failure(Error.UnexpectedNumberOfDims(expected = rank, got = arg1.rank(), shape = arg1.shape().clone()).bt()).getOrElse { return Result.failure(it) }
        }
        for ((dimIdx, v1, v2) in (arg0.shape().dims().iter().zip(arg1.shape().dims().iter())).enumerate()) {
            if (dimIdx == 0) {
                catDims[0] += v2
            }
            if (dimIdx != 0 && v1 != v2) {
                Result.failure(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg1.shape().clone()).bt()).getOrElse { return Result.failure(it) }
            }
        }
        val nextOffset = offsets.last().context("empty offsets").getOrElse { return Result.failure(it) } + arg1.elemCount()
        offsets.add(nextOffset)
    }
    val shape = Shape.from(catDims)
    val op = new(args, { args1 -> Cat(args1, 0) })
    val storage = device.allocUninit(shape, dtype).getOrElse { return Result.failure(it) }
    for ((arg2, offset) in (args.zip(ArrayDeque(offsets)))) {
        val arg3 = arg2.asRef()
        arg3.storage().copyStridedSrc(storage, offset, arg3.layout()).getOrElse { return Result.failure(it) }
    }
    return Result.success(fromStorage(storage, shape, op, false))
}

public fun <A : AsRef<Tensor>> Tensor.Companion.catContiguous(args: List<A>, dim: Int): Result<Tensor> {
    if (args.isEmpty()) {
        Result.failure(Error.OpRequiresAtLeastOneTensor(op = "cat").bt()).getOrElse { return Result.failure(it) }
    }
    val arg0 = args[0].asRef()
    if (args.size == 1) {
        return Result.success(arg0.clone())
    }
    val rank = arg0.rank()
    val device = arg0.device()
    val dtype = arg0.dtype()
    val firstDims = arg0.shape().dims()
    val catDims = firstDims.toVec()
    catDims[dim] = 0
    for ((argIdx, arg) in args.enumerate()) {
        val arg1 = arg.asRef()
        if (arg1.dtype() != dtype) {
            Result.failure(Error.DTypeMismatchBinaryOp(lhs = dtype, rhs = arg1.dtype(), op = "cat").bt()).getOrElse { return Result.failure(it) }
        }
        if (arg1.device().location() != device.location()) {
            Result.failure(Error.DeviceMismatchBinaryOp(lhs = device.location(), rhs = arg1.device().location(), op = "cat").bt()).getOrElse { return Result.failure(it) }
        }
        if (rank != arg1.rank()) {
            Result.failure(Error.UnexpectedNumberOfDims(expected = rank, got = arg1.rank(), shape = arg1.shape().clone()).bt()).getOrElse { return Result.failure(it) }
        }
        for ((dimIdx, v1, v2) in (arg0.shape().dims().iter().zip(arg1.shape().dims().iter())).enumerate()) {
            if (dimIdx == dim) {
                catDims[dim] += v2
            }
            if (dimIdx != dim && v1 != v2) {
                Result.failure(Error.ShapeMismatchCat(dim = dimIdx, firstShape = arg0.shape().clone(), n = argIdx + 1, nthShape = arg1.shape().clone()).bt()).getOrElse { return Result.failure(it) }
            }
        }
    }
    val catTargetDimLen = catDims[dim]
    val blockSize: Int = catDims.iter().skip(1 + dim).product()
    val shape = Shape.from(catDims)
    val op = new(args, { args1 -> Cat(args1, dim) })
    val storage = device.allocUninit(shape, dtype).getOrElse { return Result.failure(it) }
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
        arg3.storage().copy2d(storage, d1, d2, srcS, dstS, srcO, dstO).getOrElse { return Result.failure(it) }
        dstO += d2
    }
    return Result.success(fromStorage(storage, shape, op, false))
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
    val dim1 = dim.toIndex(this.shape(), "slice-set").getOrElse { return Result.failure(it) }
    if (!this.isContiguous() || !src.isContiguous()) {
        Result.failure(Error.RequiresContiguous(op = "slice-set").bt()).getOrElse { return Result.failure(it) }
    }
    if (this.sameStorage(src)) {
        bail("cannot use slice_set when self and src share their storage")
    }
    if (this.dtype() != src.dtype()) {
        Result.failure(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = src.dtype(), op = "slice-set").bt()).getOrElse { return Result.failure(it) }
    }
    if (this.device().location() != src.device().location()) {
        Result.failure(Error.DeviceMismatchBinaryOp(lhs = this.device().location(), rhs = src.device().location(), op = "slice-set").bt()).getOrElse { return Result.failure(it) }
    }
    if (this.rank() != src.rank()) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = this.rank(), got = src.rank(), shape = this.shape().clone()).bt()).getOrElse { return Result.failure(it) }
    }
    for ((dimIdx, v1, v2) in (this.dims().iter().zip(src.dims().iter())).enumerate()) {
        if (dimIdx == dim1 && v2 + offset > v1) {
            bail("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
        }
        if (dimIdx != dim1 && v1 != v2) {
            bail("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
        }
    }
    val blockSize: Int = src.dims().iter().skip(1 + dim1).product()
    val d1: Int = src.dims().iter().take(dim1).product()
    val d2 = blockSize * src.dims()[dim1]
    val dstO = this.layout().startOffset() + offset * blockSize
    val srcO = src.layout().startOffset()
    src.storage().copy2d(this.storageMut(), d1, d2, d2, blockSize * this.dims()[dim1], srcO, dstO).getOrElse { return Result.failure(it) }
    return Result.success(Unit)
}
