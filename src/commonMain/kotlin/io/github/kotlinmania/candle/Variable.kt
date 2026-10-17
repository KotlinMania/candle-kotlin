// port-lint: source variable.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error

// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// We do not expose a public way to create variables as this would break the invariant that the
// tensor within a variable is actually with `isVariable` set to `true`.

// A variable is a wrapper around a tensor, however variables can have their content modified
// whereas tensors are immutable.
public class Var(
    public val tensor: Tensor,
) {
    public fun fmt(f: StringBuilder = StringBuilder()): String = self[0].fmt(Formatter(f))

    public interface Target

    public fun deref(): Tensor = self[0].asRef()

    public fun asDetachedTensor(): Tensor = self[0].detach()

    public fun asTensor(): Tensor = self.self[0]

    // Consumes this `Var` and return the underlying tensor.
    public fun intoInner(): Tensor = self.self[0]

    // Sets the content of the inner tensor, this does not require a mutable reference as inner
    // mutability is used.
    public fun set(src: Tensor) {
        if (self.sameStorage(src)) {
            val msg = "cannot set a variable to a tensor that is derived from its value"
            Err<Unit>(Error.CannotSetVar(msg).bt())
        }
        val (dst, layout) = self.storageMutAndLayout()
        if (!layout.isContiguous()) {
            val msg = "cannot set a non-contiguous variable"
            Err<Unit>(Error.CannotSetVar(msg).bt())
        }
        val (srcStorage, srcL) = src.storageAndLayout()
        if (layout.shape() != srcL.shape()) {
            Err<Unit>(
                Error
                    .ShapeMismatchBinaryOp(
                        layout.shape().clone(),
                        srcL.shape().clone(),
                        "set",
                    ).bt(),
            )
        }
        srcStorage.copyStridedSrc(dst, layout.startOffset(), srcL)
        Ok(Unit)
    }

    public companion object {
        public fun zeros(
            shape: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.zerosImpl(shape, dtype, device, true)
            return Ok(Self(inner))
        }

        public fun ones(
            shape: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.onesImpl(shape, dtype, device, true)
            return Ok(Self(inner))
        }

        // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
        public fun fromTensor(t: Tensor): Var =
            if (t.isVariable()) {
                Ok(Self(t.clone()))
            } else {
                val inner = t.makeVar()
                Ok(Self(inner))
            }

        public fun randF64(
            lo: Double,
            up: Double,
            s: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.randF64Impl(lo, up, s, dtype, device, true)
            return Ok(Self(inner))
        }

        public fun randnF64(
            mean: Double,
            std: Double,
            s: Shape,
            dtype: DType,
            device: Device,
        ): Var {
            val inner = Tensor.randnF64Impl(mean, std, s, dtype, device, true)
            return Ok(Self(inner))
        }

        public fun rand(
            lo: Float,
            up: Float,
            s: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.randImpl(lo, up, s, device, true)
            return Ok(Self(inner))
        }

        public fun randn(
            mean: Float,
            std: Float,
            s: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.randnImpl(mean, std, s, device, true)
            return Ok(Self(inner))
        }

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         * This is similar to `new` but the resulting tensor is a variable.
         */
        public fun new(
            array: NdArray,
            device: Device,
        ): Var {
            val shape = array.shape()
            val inner = Tensor.newImpl(array, shape, device, true)
            return Ok(Self(inner))
        }

        public fun <D> fromVec(
            data: List<D>,
            shape: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.fromVecImpl(data, shape, device, true)
            return Ok(Self(inner))
        }

        public fun <D> fromSlice(
            array: List<D>,
            shape: Shape,
            device: Device,
        ): Var {
            val inner = Tensor.newImpl(array, shape.into(), device, true)
            return Ok(Self(inner))
        }
    }
}

private val Var.self: Var get() = this

private operator fun Var.get(i: Int): Tensor = this.tensor

private val Var.Self: (Tensor) -> Var get() = { Var(it) }
private val Var.Companion.Self: (Tensor) -> Var get() = { Var(it) }

private fun Var.sameStorage(rhs: Tensor): Boolean = this.tensor.sameStorage(rhs)

private fun Var.storageMutAndLayout(): Pair<Storage, Layout> = this.tensor.storageMutAndLayout()

private fun <T> Err(e: Error): Nothing = throw e
