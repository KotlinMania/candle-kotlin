// port-lint: source variable.rs
package io.github.kotlinmania.candle.variable

import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.device.NdArray
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.FloatDType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor

/**
 * A variable is a wrapper around a tensor, however variables can have their content modified
 * whereas tensors are immutable.
 */
public data class Var(
    public var value: Tensor = Tensor(),
) {
    public var field0: Tensor
        get() = value
        set(v) {
            value = v
        }

    public fun fmt(f: StringBuilder) {
        f.append((this.value).toString())
    }

    public fun deref(): Tensor = this.value.asRef()

    public fun asDetachedTensor(): Tensor = this.value.detach()

    public fun asTensor(): Tensor = this.value

    /**
     * Consumes this `Var` and return the underlying tensor.
     */
    public fun intoInner(): Tensor = this.value

    /**
     * Sets the content of the inner tensor, this does not require a mutable reference as inner
     * mutability is used.
     */
    public fun set(src: Tensor): Result<Unit> {
        if (this.sameStorage(src)) {
            val msg = "cannot set a variable to a tensor that is derived from its value"
            when (val tryResult = Result.Err(Error.CannotSetVar(msg = msg).bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        var (dst, layout) = this.storageMutAndLayout()
        if (!layout.isContiguous()) {
            val msg1 = "cannot set a non-contiguous variable"
            when (val tryResult = Result.Err(Error.CannotSetVar(msg = msg1).bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        val (src1, srcL) = src.storageAndLayout()
        if (layout.shape() != srcL.shape()) {
            when (val tryResult = Result.Err(Error.ShapeMismatchBinaryOp(lhs = layout.shape().clone(), rhs = srcL.shape().clone(), op = "set").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        when (val tryResult = src1.copyStridedSrc(dst, layout.startOffset(), srcL)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(Unit)
    }

    public companion object {
        public fun zeros(shape: Shape, dtype: DType, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.zerosImpl(shape, dtype, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun ones(shape: Shape, dtype: DType, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.onesImpl(shape, dtype, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
        public fun fromTensor(t: Tensor): Result<Var> {
            return if (t.isVariable()) {
                Result.Ok(Var(t.clone()))
            } else {
                val inner =
                    when (val tryResult = t.makeVar()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Var(inner))
            }
        }

        public fun randF64(lo: Double, up: Double, s: Shape, dtype: DType, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.randF64Impl(lo, up, s, dtype, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun randnF64(mean: Double, std: Double, s: Shape, dtype: DType, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.randnF64Impl(mean, std, s, dtype, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun <T : FloatDType> rand(lo: T, up: T, s: Shape, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.randImpl(lo, up, s, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun <T : FloatDType> randn(mean: T, std: T, s: Shape, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.randnImpl(mean, std, s, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         * This is similar to `new` but the resulting tensor is a variable.
         */
        public fun <A : NdArray> new(array: A, device: Device): Result<Var> {
            val shape =
                when (val tryResult = array.shape()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val inner =
                when (val tryResult = Tensor.newImpl(array, shape, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun <D : WithDType> fromVec(data: List<D>, shape: Shape, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.fromVecImpl(data, shape, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }

        public fun <D : WithDType> fromSlice(array: List<D>, shape: Shape, device: Device): Result<Var> {
            val inner =
                when (val tryResult = Tensor.newImpl(array, shape, device, true)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            return Result.Ok(Var(inner))
        }
    }
}
