// port-lint: source variable.rs
package io.github.kotlinmania.candle

// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// We do not expose a public way to create variables as this would break the invariant that the
// tensor within a variable is actually with `isVariable` set to `true`.

/**
 * A variable is a wrapper around a tensor, however variables can have their content modified
 * whereas tensors are immutable.
 */
public class Var(
    public val tensor: Tensor,
) {
    override fun toString(): String = tensor.toString()

    public fun fmt(): String = tensor.toString()

    public fun deref(): Tensor = tensor.asRef()

    public fun shape(): Shape = tensor.shape()

    public fun dtype(): DType = tensor.dtype()

    public fun device(): Device = tensor.device()

    public fun asDetachedTensor(): Tensor = tensor.detach()

    public fun asTensor(): Tensor = tensor

    /**
     * Consumes this `Var` and return the underlying tensor.
     */
    public fun intoInner(): Tensor = tensor

    /**
     * Sets the content of the inner tensor, this does not require a mutable reference as inner
     * mutability is used.
     */
    public fun set(src: Tensor) {
        if (tensor.sameStorage(src)) {
            val msg = "cannot set a variable to a tensor that is derived from its value"
            throw CandleException.CannotSetVar(msg)
        }
        val (dst, layout) = tensor.storageMutAndLayout()
        if (!layout.isContiguous()) {
            val msg = "cannot set a non-contiguous variable"
            throw CandleException.CannotSetVar(msg)
        }
        val (srcStorage, srcL) = src.storageAndLayout()
        if (layout.shape() != srcL.shape()) {
            throw CandleException.ShapeMismatchBinaryOp(
                lhs = layout.shape(),
                rhs = srcL.shape(),
                op = "set",
            )
        }
        srcStorage.copyStridedSrc(dst, layout.startOffset(), srcL)
    }

    public companion object {
        public fun zeros(shape: Shape, dtype: DType, device: Device): Var {
            val inner = Tensor.zerosImpl(shape, dtype, device, true)
            return Var(inner)
        }

        public fun ones(shape: Shape, dtype: DType, device: Device): Var {
            val inner = Tensor.onesImpl(shape, dtype, device, true)
            return Var(inner)
        }

        // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
        public fun fromTensor(t: Tensor): Var =
            if (t.isVariable()) {
                Var(t.clone())
            } else {
                val inner = t.makeVar()
                Var(inner)
            }

        public fun randF64(
            lo: Double,
            up: Double,
            s: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.randF64Impl(lo, up, s, dtype, device, true)
            return Var(inner)
        }

        public fun randnF64(
            mean: Double,
            std: Double,
            s: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.randnF64Impl(mean, std, s, dtype, device, true)
            return Var(inner)
        }

        public fun rand(
            lo: Float,
            up: Float,
            s: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.randImpl(lo.toDouble(), up.toDouble(), s, DType.F32, device, true)
            return Var(inner)
        }

        public fun randn(
            mean: Float,
            std: Float,
            s: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.randnImpl(mean.toDouble(), std.toDouble(), s, DType.F32, device, true)
            return Var(inner)
        }

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         * This is similar to `new` but the resulting tensor is a variable.
         */
        public fun new(array: NdArray, device: Device): Var {
            val shape = array.shape()
            val storage = device.storage(array)
            val inner = fromStorage(storage, shape, BackpropOp.none(), true)
            return Var(inner)
        }

        public fun <D> fromVec(
            data: List<D>,
            shape: Shape,
            device: Device,
        ): Var {
            val storage = device.storageOwned(data)
            val inner = fromStorage(storage, shape, BackpropOp.none(), true)
            return Var(inner)
        }

        public fun <D> fromSlice(
            array: List<D>,
            shape: Shape,
            device: Device,
        ): Var {
            val storage = device.storageFromSlice(array)
            val inner = fromStorage(storage, shape, BackpropOp.none(), true)
            return Var(inner)
        }
    }
}
