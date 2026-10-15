// port-lint: source custom_op.rs
package io.github.kotlinmania.candle

/**
 * Unary ops that can be defined in user-land.
 */
public interface CustomOp1 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(storage: CpuStorage, layout: Layout): Pair<CpuStorage, Shape>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(storage: CudaStorage, layout: Layout): Pair<CudaStorage, Shape> =
        throw CandleException.Cuda("no cuda implementation for ${name()}")

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(storage: MetalStorage, layout: Layout): Pair<MetalStorage, Shape> =
        throw CandleException.Metal("no metal implementation for ${name()}")

    /**
     * This function takes as argument the argument arg used in the forward pass, the result
     * produced by the forward operation res and the gradient of the result gradRes.
     * The function should return the gradient of the argument.
     */
    public fun bwd(arg: Tensor, res: Tensor, gradRes: Tensor): Tensor? =
        throw CandleException.BackwardNotSupported(name())
}

public interface CustomOp2 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
    ): Pair<CpuStorage, Shape>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(
        s1: CudaStorage,
        l1: Layout,
        s2: CudaStorage,
        l2: Layout,
    ): Pair<CudaStorage, Shape> =
        throw CandleException.Cuda("no cuda implementation for ${name()}")

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(
        s1: MetalStorage,
        l1: Layout,
        s2: MetalStorage,
        l2: Layout,
    ): Pair<MetalStorage, Shape> =
        throw CandleException.Metal("no metal implementation for ${name()}")

    public fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Pair<Tensor?, Tensor?> =
        throw CandleException.BackwardNotSupported(name())
}

public interface CustomOp3 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    ): Pair<CpuStorage, Shape>

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(
        s1: CudaStorage,
        l1: Layout,
        s2: CudaStorage,
        l2: Layout,
        s3: CudaStorage,
        l3: Layout,
    ): Pair<CudaStorage, Shape> =
        throw CandleException.Cuda("no cuda implementation for ${name()}")

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(
        s1: MetalStorage,
        l1: Layout,
        s2: MetalStorage,
        l2: Layout,
        s3: MetalStorage,
        l3: Layout,
    ): Pair<MetalStorage, Shape> =
        throw CandleException.Metal("no metal implementation for ${name()}")

    public fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        arg3: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Triple<Tensor?, Tensor?, Tensor?> =
        throw CandleException.BackwardNotSupported(name())
}

/**
 * Applies a unary custom op without backward support
 */
public fun Tensor.applyOp1NoBwd(c: CustomOp1): Tensor {
    val (storage, shape) = storage().applyOp1(layout(), c)
    return Tensor.fromStorage(storage, shape, BackpropOp.none(), false)
}

/**
 * Applies a binary custom op without backward support
 */
public fun Tensor.applyOp2NoBwd(rhs: Tensor, c: CustomOp2): Tensor {
    val (storage, shape) = storage().applyOp2(layout(), rhs.storage(), rhs.layout(), c)
    return Tensor.fromStorage(storage, shape, BackpropOp.none(), false)
}

/**
 * Applies a ternary custom op without backward support
 */
public fun Tensor.applyOp3NoBwd(t2: Tensor, t3: Tensor, c: CustomOp3): Tensor {
    val (storage, shape) = storage().applyOp3(layout(), t2.storage(), t2.layout(), t3.storage(), t3.layout(), c)
    return Tensor.fromStorage(storage, shape, BackpropOp.none(), false)
}

/**
 * Applies a unary custom op.
 */
public fun Tensor.applyOp1Arc(c: CustomOp1): Tensor {
    val (storage, shape) = storage().applyOp1(layout(), c)
    val op = BackpropOp.new1(this) { s -> Op.CustomOp1(s, c) }
    return Tensor.fromStorage(storage, shape, op, false)
}

public fun Tensor.applyOp1(c: CustomOp1): Tensor = applyOp1Arc(c)

/**
 * Applies a binary custom op.
 */
public fun Tensor.applyOp2Arc(rhs: Tensor, c: CustomOp2): Tensor {
    val (storage, shape) = storage().applyOp2(layout(), rhs.storage(), rhs.layout(), c)
    val op = BackpropOp.new2(this, rhs) { t1, t2 -> Op.CustomOp2(t1, t2, c) }
    return Tensor.fromStorage(storage, shape, op, false)
}

public fun Tensor.applyOp2(r: Tensor, c: CustomOp2): Tensor = applyOp2Arc(r, c)

/**
 * Applies a ternary custom op.
 */
public fun Tensor.applyOp3Arc(t2: Tensor, t3: Tensor, c: CustomOp3): Tensor {
    val (storage, shape) = storage().applyOp3(layout(), t2.storage(), t2.layout(), t3.storage(), t3.layout(), c)
    val op = BackpropOp.new3(this, t2, t3) { t1, t2, t3 -> Op.CustomOp3(t1, t2, t3, c) }
    return Tensor.fromStorage(storage, shape, op, false)
}

public fun Tensor.applyOp3(t2: Tensor, t3: Tensor, c: CustomOp3): Tensor = applyOp3Arc(t2, t3, c)

/**
 * Unary ops that can be defined in user-land.
 * These ops work in place and as such back-prop is unsupported.
 */
public interface InplaceOp1 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(storage: CpuStorage, layout: Layout)

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(storage: CudaStorage, layout: Layout) {
        throw CandleException.Cuda("no cuda implementation for ${name()}")
    }

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(storage: MetalStorage, layout: Layout) {
        throw CandleException.Metal("no metal implementation for ${name()}")
    }
}

public interface InplaceOp2 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(s1: CpuStorage, l1: Layout, s2: CpuStorage, l2: Layout)

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(s1: CudaStorage, l1: Layout, s2: CudaStorage, l2: Layout) {
        throw CandleException.Cuda("no cuda implementation for ${name()}")
    }

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(s1: MetalStorage, l1: Layout, s2: MetalStorage, l2: Layout) {
        throw CandleException.Metal("no metal implementation for ${name()}")
    }
}

public interface InplaceOp3 {
    public fun name(): String

    /**
     * The forward pass, as run on a cpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cpuFwd(
        s1: CpuStorage,
        l1: Layout,
        s2: CpuStorage,
        l2: Layout,
        s3: CpuStorage,
        l3: Layout,
    )

    /**
     * The forward pass, as run on a gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun cudaFwd(
        s1: CudaStorage,
        l1: Layout,
        s2: CudaStorage,
        l2: Layout,
        s3: CudaStorage,
        l3: Layout,
    ) {
        throw CandleException.Cuda("no cuda implementation for ${name()}")
    }

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(
        s1: MetalStorage,
        l1: Layout,
        s2: MetalStorage,
        l2: Layout,
        s3: MetalStorage,
        l3: Layout,
    ) {
        throw CandleException.Metal("no metal implementation for ${name()}")
    }
}

/**
 * Applies a unary custom op in place.
 */
public fun Tensor.inplaceOp1(c: InplaceOp1) {
    storageMut().inplaceOp1(layout(), c)
}

/**
 * Applies a unary custom op in place (for the first tensor).
 */
public fun Tensor.inplaceOp2(rhs: Tensor, c: InplaceOp2) {
    storageMut().inplaceOp2(layout(), rhs.storage(), rhs.layout(), c)
}

/**
 * Applies a ternary custom op in place (for the first tensor).
 */
public fun Tensor.inplaceOp3(t2: Tensor, t3: Tensor, c: InplaceOp3) {
    storageMut().inplaceOp3(
        layout(),
        t2.storage(),
        t2.layout(),
        t3.storage(),
        t3.layout(),
        c,
    )
}

public class UgIOp1(
    public val opName: String,
) : InplaceOp1 {
    override fun name(): String = opName

    override fun cpuFwd(storage: CpuStorage, layout: Layout) {
        bail("ug ops are only supported on metal/cuda at the moment")
    }

    override fun metalFwd(storage: MetalStorage, layout: Layout) {
        val elemCount = layout.shape().elemCount()
        if (storage.dtype() != DType.F32) {
            bail("input is not a f32 tensor")
        }
        val (g, b) = if (elemCount % 32 == 0) {
            Pair(elemCount / 32, 32)
        } else {
            Pair(elemCount, 1)
        }
        if (g < 0 || b < 0) {
            bail("invalid grid or block dims for $opName")
        }
    }

    override fun cudaFwd(storage: CudaStorage, layout: Layout) {
        val elemCount = layout.shape().elemCount()
        val offsets = layout.contiguousOffsets() ?: bail("input has to be contiguous")
        val (g, b) = if (elemCount % 32 == 0) {
            Pair(elemCount / 32, 32)
        } else {
            Pair(elemCount, 1)
        }
        if (offsets.first > offsets.second || g < 0 || b < 0) {
            bail("invalid slice or launch config for ${storage.dtype()} in $opName")
        }
    }

    public companion object {
        public fun new(name: String, device: Device = Device.Cpu): UgIOp1 {
            when (device) {
                is Device.Cuda -> device.asCudaDevice()
                is Device.Metal -> device.asMetalDevice()
                is Device.Cpu -> Unit
            }
            return UgIOp1(name)
        }
    }
}
