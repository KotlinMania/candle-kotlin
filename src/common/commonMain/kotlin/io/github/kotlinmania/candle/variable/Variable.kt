// port-lint: source variable.rs
package io.github.kotlinmania.candle.variable

import io.github.kotlinmania.candle.NdArray
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.FloatDType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor

public typealias Target = Tensor

/**
 * A variable is a wrapper around a tensor, however variables can have their content modified
 * whereas tensors are immutable.
 */
public open class Var(
    public var value: Tensor,
) {
    public var field0: Tensor
        get() = value
        set(v) {
            value = v
        }

    public open fun fmt(f: StringBuilder) {
        f.append((this.value).toString())
    }

    public open fun deref(): Tensor = this.value.asRef()

    public open fun asDetachedTensor(): Tensor = this.value.detach()

    public open fun asTensor(): Tensor = this.value

    /**
     * Consumes this `Var` and return the underlying tensor.
     */
    public open fun intoInner(): Tensor = this.value

    /**
     * Sets the content of the inner tensor, this does not require a mutable reference as inner
     * mutability is used.
     */
    public open fun set(src: Tensor): Result<Unit> {
        if (this.sameStorage(src)) {
            val msg = "cannot set a variable to a tensor that is derived from its value"
            Result.failure(Error.CannotSetVar(msg = msg).bt()).getOrElse { return Result.failure(it) }
        }
        var (dst, layout) = this.storageMutAndLayout()
        if (!layout.isContiguous()) {
            val msg1 = "cannot set a non-contiguous variable"
            Result.failure(Error.CannotSetVar(msg = msg1).bt()).getOrElse { return Result.failure(it) }
        }
        val (src1, srcL) = src.storageAndLayout()
        if (layout.shape() != srcL.shape()) {
            Result.failure(Error.ShapeMismatchBinaryOp(lhs = layout.shape().clone(), rhs = srcL.shape().clone(), op = "set").bt()).getOrElse { return Result.failure(it) }
        }
        src1.copyStridedSrc(dst, layout.startOffset(), srcL).getOrElse { return Result.failure(it) }
        return Result.success(Unit)
    }

    public companion object {
        public fun zeros(shape: Shape, dtype: DType, device: Device): Result<Var> {
            val inner = Tensor.zerosImpl(shape, dtype, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun ones(shape: Shape, dtype: DType, device: Device): Result<Var> {
            val inner = Tensor.onesImpl(shape, dtype, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
        public fun fromTensor(t: Tensor): Result<Var> {
            return if (t.isVariable()) {
                Result.success(Var(t.clone()))
            } else {
                val inner = t.makeVar().getOrElse { return Result.failure(it) }
                Result.success(Var(inner))
            }
        }

        public fun randF64(lo: Double, up: Double, s: Shape, dtype: DType, device: Device): Result<Var> {
            val inner = Tensor.randF64Impl(lo, up, s, dtype, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun randnF64(mean: Double, std: Double, s: Shape, dtype: DType, device: Device): Result<Var> {
            val inner = Tensor.randnF64Impl(mean, std, s, dtype, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun <T : FloatDType> rand(lo: T, up: T, s: Shape, device: Device): Result<Var> {
            val inner = Tensor.randImpl(lo, up, s, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun <T : FloatDType> randn(mean: T, std: T, s: Shape, device: Device): Result<Var> {
            val inner = Tensor.randnImpl(mean, std, s, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         * This is similar to `new` but the resulting tensor is a variable.
         */
        public fun <A : NdArray> new(array: A, device: Device): Result<Var> {
            val shape = array.shape().getOrElse { return Result.failure(it) }
            val inner = Tensor.newImpl(array, shape, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun <D : WithDType> fromVec(data: List<D>, shape: Shape, device: Device): Result<Var> {
            val inner = Tensor.fromVecImpl(data, shape, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }

        public fun <D : WithDType> fromSlice(array: List<D>, shape: Shape, device: Device): Result<Var> {
            val inner = Tensor.newImpl(array, shape, device, true).getOrElse { return Result.failure(it) }
            return Result.success(Var(inner))
        }
    }
}
