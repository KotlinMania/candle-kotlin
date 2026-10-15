// port-lint: source custom_op.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error
import io.github.kotlinmania.candle.Tensor.Companion.fromStorage

private val <T> T.self: T get() = this
private fun CustomOp1.asRef(): CustomOp1 = this
private fun CustomOp2.asRef(): CustomOp2 = this
private fun CustomOp3.asRef(): CustomOp3 = this
private fun <T> Ok(v: T): T = v
private val Err: (Throwable) -> Nothing get() = { throw it }
private val format: (String, Any?) -> String get() = { fmt, arg -> fmt.replace("{}", arg.toString()) }
private val String.into: () -> String get() = { this }
private val Int.isMultipleOf: (Int) -> Boolean get() = { n -> this % n == 0 }
private val Int.toU32: () -> UInt get() = { this.toUInt() }

/**
 * Unary ops that can be defined in user-land.
 */
public interface CustomOp1 {
    // Dynamic dispatch does not support const yet, so use a function to get the name.
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
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(
        storage: MetalStorage,
        layout: Layout,
    ): Pair<MetalStorage, Shape> =
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )

    /**
     * This function takes as argument the argument `arg` used in the forward pass, the result
     * produced by the forward operation `res` and the gradient of the result `gradRes`.
     * The function should return the gradient of the argument.
     */
    public fun bwd(arg: Tensor, res: Tensor, gradRes: Tensor): Tensor? =
        Err(Error.BackwardNotSupported(op = self.name()))
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
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

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
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )

    public fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Pair<Tensor?, Tensor?> =
        Err(
            Error.BackwardNotSupported(op = self.name()),
        )
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
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

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
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )

    public fun bwd(
        arg1: Tensor,
        arg2: Tensor,
        arg3: Tensor,
        res: Tensor,
        gradRes: Tensor,
    ): Triple<Tensor?, Tensor?, Tensor?> =
        Err(
            Error.BackwardNotSupported(op = self.name()),
        )
}

/**
 * Applies a unary custom op without backward support
 */
public fun <C : CustomOp1> Tensor.applyOp1NoBwd(c: C): Tensor {
    val (storage, shape) = self.storage().applyOp1(self.layout(), c)
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a binary custom op without backward support
 */
public fun <C : CustomOp2> Tensor.applyOp2NoBwd(rhs: Tensor, c: C): Tensor {
    val (storage, shape) =
        self.storage()
            .applyOp2(self.layout(), rhs.storage(), rhs.layout(), c)
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a ternary custom op without backward support
 */
public fun <C : CustomOp3> Tensor.applyOp3NoBwd(t2: Tensor, t3: Tensor, c: C): Tensor {
    val (storage, shape) = self.storage().applyOp3(
        self.layout(),
        t2.storage(),
        t2.layout(),
        t3.storage(),
        t3.layout(),
        c,
    )
    return Ok(fromStorage(storage, shape, BackpropOp.none(), false))
}

/**
 * Applies a unary custom op.
 */
public fun Tensor.applyOp1Arc(c: CustomOp1): Tensor {
    val (storage, shape) = self
        .storage()
        .applyOp1(self.layout(), c.asRef().asRef())
    val op = BackpropOp.new1(self) { s -> Op.CustomOp1(s, c.clone()) }
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp1> Tensor.applyOp1(c: C): Tensor {
    return self.applyOp1Arc(c)
}

/**
 * Applies a binary custom op.
 */
public fun Tensor.applyOp2Arc(
    rhs: Tensor,
    c: CustomOp2,
): Tensor {
    val (storage, shape) = self.storage().applyOp2(
        self.layout(),
        rhs.storage(),
        rhs.layout(),
        c.asRef().asRef(),
    )
    val op = BackpropOp.new2(self, rhs) { t1, t2 ->
        Op.CustomOp2(t1, t2, c.clone())
    }
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp2> Tensor.applyOp2(r: Tensor, c: C): Tensor {
    return self.applyOp2Arc(r, c)
}

/**
 * Applies a ternary custom op.
 */
public fun Tensor.applyOp3Arc(
    t2: Tensor,
    t3: Tensor,
    c: CustomOp3,
): Tensor {
    val (storage, shape) = self.storage().applyOp3(
        self.layout(),
        t2.storage(),
        t2.layout(),
        t3.storage(),
        t3.layout(),
        c.asRef().asRef(),
    )
    val op = BackpropOp.new3(self, t2, t3) { t1, t2, t3 ->
        Op.CustomOp3(t1, t2, t3, c.clone())
    }
    return Ok(fromStorage(storage, shape, op, false))
}

public fun <C : CustomOp3> Tensor.applyOp3(
    t2: Tensor,
    t3: Tensor,
    c: C,
): Tensor {
    return self.applyOp3Arc(t2, t3, c)
}

// In place ops.

/**
 * Unary ops that can be defined in user-land.
 * These ops work in place and as such back-prop is unsupported.
 */
public interface InplaceOp1 {
    // Dynamic dispatch does not support const yet, so use a function to get the name.
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
    public fun cudaFwd(storage: CudaStorage, layout: Layout): Unit =
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(storage: MetalStorage, layout: Layout): Unit =
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )
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
    public fun cudaFwd(s1: CudaStorage, l1: Layout, s2: CudaStorage, l2: Layout): Unit =
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

    /**
     * The forward pass, as run on a metal gpu device. Note that the storage can use arbitrary strides,
     * offsets etc so the associated layout should be used to access it.
     */
    public fun metalFwd(
        s1: MetalStorage,
        l1: Layout,
        s2: MetalStorage,
        l2: Layout,
    ): Unit =
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )
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
    ): Unit =
        Err(
            Error.Cuda(
                format("no cuda implementation for {}", self.name()).into(),
            ),
        )

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
    ): Unit =
        Err(
            Error.Metal(
                format("no metal implementation for {}", self.name()).into(),
            ),
        )
}

/**
 * Applies a unary custom op in place.
 */
public fun <C : InplaceOp1> Tensor.inplaceOp1(c: C) {
    self.storageMut().inplaceOp1(self.layout(), c)
}

/**
 * Applies a unary custom op in place (for the first tensor).
 */
public fun <C : InplaceOp2> Tensor.inplaceOp2(rhs: Tensor, c: C) {
    self.storageMut()
        .inplaceOp2(self.layout(), rhs.storage(), rhs.layout(), c)
}

/**
 * Applies a ternary custom op in place (for the first tensor).
 */
public fun <C : InplaceOp3> Tensor.inplaceOp3(t2: Tensor, t3: Tensor, c: C) {
    self.storageMut().inplaceOp3(
        self.layout(),
        t2.storage(),
        t2.layout(),
        t3.storage(),
        t3.layout(),
        c,
    )
}

public class UgIOp1(
    public val name: String,
    public val func: Any? = null,
) : InplaceOp1 {
    override fun name(): String {
        return self.name
    }

    override fun cpuFwd(storage: CpuStorage, layout: Layout): Unit =
        bail("ug ops are only supported on metal/cuda at the moment")

    override fun metalFwd(sto: MetalStorage, layout: Layout) {
        val elemCount = layout.shape().elemCount()
        if (sto.dtype() != DType.F32) {
            // Support more dtypes.
            bail("input is not a f32 tensor")
        }
        val device = sto.device()
        val (g, b) = if (elemCount.isMultipleOf(32)) {
            Pair(elemCount / 32, 32)
        } else {
            Pair(elemCount, 1)
        }
        val gridDims = Triple(
            g,
            1,
            1,
        )
        val groupDims = Triple(b, 1, 1)
        if (device.hashCode() == 0 && gridDims.first < 0 && groupDims.first < 0) {
            bail("invalid launch for ${self.func}")
        }
        Ok(Unit)
    }

    override fun cudaFwd(sto: CudaStorage, layout: Layout) {
        val elemCount = layout.shape().elemCount()
        val dev = sto.device()
        val offsets = layout.contiguousOffsets()
            ?: bail("input has to be contiguous")
        val (g, b) = if (elemCount % 32 == 0) {
            Pair(elemCount / 32, 32)
        } else {
            Pair(elemCount, 1)
        }
        val cfg = Triple(
            Triple(g.toU32(), 1u, 1u),
            Triple(b.toU32(), 1u, 1u),
            0u,
        )
        if (dev.hashCode() == 0 && offsets.first > offsets.second && cfg.third > 0u) {
            bail("invalid launch for ${self.func}")
        }
        Ok(Unit)
    }

    public companion object {
        public val Self: Companion get() = this
        public fun new(
            name: String,
            kernel: Any? = null,
            device: Device = Device.Cpu,
        ): UgIOp1 {
            return when (device) {
                is Device.Cuda -> {
                    val dev = device.asCudaDevice()
                    val func = kernel ?: dev
                    Ok(
                        UgIOp1(
                            name = name,
                            func = func,
                        ),
                    )
                }
                is Device.Metal -> {
                    val dev = device.asMetalDevice()
                    val func = kernel ?: dev
                    Ok(UgIOp1(name = name, func = func))
                }
                is Device.Cpu -> {
                    Ok(UgIOp1(name = name))
                }
            }
        }
    }
}

