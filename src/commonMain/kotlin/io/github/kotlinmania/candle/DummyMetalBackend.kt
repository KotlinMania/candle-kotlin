package io.github.kotlinmania.candle

import io.github.kotlinmania.thiserror.StdError
import io.github.kotlinmania.candle.CandleException as Error

/**
 * Implementation of the Metal backend when Metal support has not been compiled in.
 *
 * Provides stub device, storage, and error types that return NotCompiledWithMetalSupport
 * errors when Metal backend operations are invoked without Metal compilation enabled.
 */
// port-lint: source dummy_metal_backend.rs

private val Err: (Error) -> Nothing = { throw it }
private val fail: () -> Nothing = {
    throw Error.Msg("metal support has not been enabled, add `metal` feature to enable.")
}
private val Ok: (Unit) -> Unit = { }

public sealed class MetalError(
    message: String,
) : Exception(message),
    StdError {
    public data class Message(
        override val message: String,
    ) : MetalError(message)

    public companion object {
        public fun from(e: String): MetalError = MetalError.Message(e)
    }
}

public data class MetalDevice(
    val gpuId: Int = 0,
) : BackendDevice {
    public interface Device

    public interface Storage

    private val self: MetalDevice get() = this

    override fun location(): DeviceLocation {
        self
        fail()
    }

    override fun sameDevice(other: BackendDevice): Boolean = fail()

    override fun setSeed(seed: Long): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun getCurrentSeed(): Long {
        self
        Err(Error.NotCompiledWithMetalSupport)
    }

    override fun zerosImpl(
        shape: Shape,
        dtype: DType,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <T> storageFromSlice(s: List<T>): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun storageFromCpuStorage(storage: CpuStorage): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun storageFromCpuStorageOwned(storage: CpuStorage): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun randUniform(
        shape: Shape,
        dtype: DType,
        min: Double,
        max: Double,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        std: Double,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun synchronize() {
        self
        Ok(Unit)
    }

    public fun capture(path: String) {
        self
        path.hashCode()
        Err(Error.NotCompiledWithMetalSupport)
    }

    public companion object {
        public fun new(o: Int): MetalDevice = Err(Error.NotCompiledWithMetalSupport)
    }
}

public class MetalStorage : BackendStorage {
    private val self: MetalStorage get() = this

    public companion object {
        public fun new(
            buffer: Any,
            device: MetalDevice,
            count: Int,
            dtype: DType,
        ): MetalStorage {
            buffer.hashCode()
            device.hashCode()
            count.hashCode()
            dtype.hashCode()
            return MetalStorage()
        }
    }

    override fun tryClone(layout: Layout): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun dtype(): DType {
        self
        fail()
    }

    override fun device(): BackendDevice {
        self
        fail()
    }

    override fun constSet(
        v: Scalar,
        l: Layout,
    ): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun toCpuStorage(): CpuStorage {
        self
        Err(Error.NotCompiledWithMetalSupport)
    }

    override fun affine(
        layout: Layout,
        mul: Double,
        add: Double,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun powf(
        layout: Layout,
        e: Double,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun elu(
        layout: Layout,
        alpha: Double,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        dims: List<Int>,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun cmp(
        op: CmpOp,
        rhs: BackendStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun toDtype(
        layout: Layout,
        targetDType: DType,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : UnaryOpT> unaryImpl(l: Layout): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : BinaryOpT> binaryImpl(
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun whereCond(
        layout: Layout,
        onTrue: BackendStorage,
        trueL: Layout,
        onFalse: BackendStorage,
        falseL: Layout,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun indexSelect(
        rhs: BackendStorage,
        lhsL: Layout,
        rhsL: Layout,
        dim: Int,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun gather(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        dim: Int,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun scatterAddSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun indexAdd(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun copyStridedSrc(
        dst: BackendStorage,
        dstOffset: Int,
        srcLayout: Layout,
    ): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun copy2d(
        dst: BackendStorage,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport,
        )

    override fun avgPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun maxPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleNearest1d(
        l: Layout,
        targetSize: Int,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleNearest2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleBilinear2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?,
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport,
        )
}
