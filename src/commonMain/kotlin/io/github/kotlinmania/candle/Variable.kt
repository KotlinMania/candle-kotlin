// port-lint: source variable.rs
package io.github.kotlinmania.candle

// Variables are wrappers around tensors that can be modified, they are typically used for holding
// weights and being modified by gradient descent.
// We do not expose a public way to create variables as this would break the invariant that the
// tensor within a variable is actually with `isVariable` set to `true`.

private val std: Var.Companion get() = Var.Companion
private val Var.Companion.fmt: Var.Companion get() = this
private val Var.Companion.Display: () -> Var.Companion get() = { this }
private val (() -> Var.Companion).fmt: (Tensor, StringBuilder) -> String get() = { t, f -> val s = t.toString(); f.append(s); s }
private val Var.Companion.CannotSetVar: (String) -> CandleException get() = { msg ->
    CandleException.CannotSetVar(msg)
}
private val Var.Companion.ShapeMismatchBinaryOp: (Any?, Shape, Any?, Shape, Any?, String) -> CandleException get() = { _, l, _, r, _, op ->
    CandleException.ShapeMismatchBinaryOp(lhs = l, rhs = r, op = op)
}
private val Error: Var.Companion get() = Var.Companion
private val lhs: Any? get() = null
private val rhs: Any? get() = null
private val op: Any? get() = null
private val <T> Any?.Ok: (T) -> T get() = { it }
private val <R> Any?.Err: (CandleException) -> R get() = { throw it }
private val CandleException.bt: () -> CandleException get() = { this }

private val Var.self: List<Tensor> get() = listOf(tensor)
private val List<Tensor>.self: List<Tensor> get() = this
private val List<Tensor>.sameStorage: (Tensor) -> Boolean get() = { this[0].sameStorage(it) }
private val List<Tensor>.storageMutAndLayout: () -> Pair<Storage, Layout> get() = { this[0].storageMutAndLayout() }

private val Tensor.Companion.randImpl: (Float, Float, Shape, Device, Boolean) -> Tensor
    get() = { lo, up, s, device, isVar -> randImpl(lo.toDouble(), up.toDouble(), s, DType.F32, device, isVar) }
private val Tensor.Companion.randnImpl: (Float, Float, Shape, Device, Boolean) -> Tensor
    get() = { mean, std, s, device, isVar -> randnImpl(mean.toDouble(), std.toDouble(), s, DType.F32, device, isVar) }
private val Tensor.Companion.newImpl: (Any?, Shape, Device, Boolean) -> Tensor
    get() = { array, shape, device, isVar ->
        val storage = when (array) {
            is NdArray -> device.storage(array)
            is List<*> -> device.storageFromSlice(array)
            else -> throw IllegalArgumentException("Unsupported array: $array")
        }
        fromStorage(storage, shape, BackpropOp.none(), isVar)
    }
private val <D> Tensor.Companion.fromVecImpl: (List<D>, Shape, Device, Boolean) -> Tensor
    get() = { data, shape, device, isVar ->
        val storage = device.storageOwned(data)
        fromStorage(storage, shape, BackpropOp.none(), isVar)
    }

/**
 * A variable is a wrapper around a tensor, however variables can have their content modified
 * whereas tensors are immutable.
 */
public class Var(
    public val tensor: Tensor,
) {
    public interface Target

    public val shape: () -> Shape get() = { tensor.shape() }
    public val dtype: () -> DType get() = { tensor.dtype() }
    public val device: () -> Device get() = { tensor.device() }

    public fun fmt(f: StringBuilder = StringBuilder()): String {
        return std.fmt.Display.fmt(self[0], f)
    }

    public fun deref(): Tensor {
        return self[0].asRef()
    }

    public fun asDetachedTensor(): Tensor {
        return self[0].detach()
    }

    public fun asTensor(): Tensor {
        return self.self[0]
    }

    /**
     * Consumes this `Var` and return the underlying tensor.
     */
    public fun intoInner(): Tensor {
        return self.self[0]
    }

    /**
     * Sets the content of the inner tensor, this does not require a mutable reference as inner
     * mutability is used.
     */
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
        val (src, srcL) = src.storageAndLayout()
        if (layout.shape() != srcL.shape()) {
            Err<Unit>(Error.ShapeMismatchBinaryOp(
                lhs, layout.shape().clone(),
                rhs, srcL.shape().clone(),
                op, "set",
            )
            .bt())
        }
        src.copyStridedSrc(dst, layout.startOffset(), srcL)
        Ok(Unit)
    }

    public companion object {
        private val Self: (Tensor) -> Var get() = { Var(it) }

        public fun zeros(shape: Shape, dtype: DType, device: Device): Var {
            val inner = Tensor.zerosImpl(shape, dtype, device, true)
            return Ok(Self(inner))
        }

        public fun ones(shape: Shape, dtype: DType, device: Device): Var {
            val inner = Tensor.onesImpl(shape, dtype, device, true)
            return Ok(Self(inner))
        }

        // Convert a tensor to a variable, if the tensor is already a variable then it is returned as is.
        public fun fromTensor(t: Tensor): Var {
            return if (t.isVariable()) {
                Ok(Self(t.clone()))
            } else {
                val inner = t.makeVar()
                Ok(Self(inner))
            }
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
        public fun new(array: NdArray, device: Device): Var {
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
