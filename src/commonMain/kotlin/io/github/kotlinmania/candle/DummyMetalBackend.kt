package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error
import io.github.kotlinmania.thiserror.StdError

/**
 * Implementation of the Metal backend when Metal support has not been compiled in.
 *
 * Provides stub device, storage, and error types that return NotCompiledWithMetalSupport
 * errors when Metal backend operations are invoked without Metal compilation enabled.
 */
// port-lint: source dummy_metal_backend.rs

private val Err: (CandleException) -> Nothing = { throw it }
private val fail: () -> Nothing = {
    throw CandleException.Msg("metal support has not been enabled, add `metal` feature to enable.")
}
private val Ok: (Unit) -> Unit = { }

public sealed class MetalError(message: String) : Exception(message), StdError {
    public data class Message(override val message: String) : MetalError(message)

    public companion object {
        public fun from(e: String): MetalError {
            return MetalError.Message(e)
        }
    }
}

public data class MetalDevice(val gpuId: Int = 0) : BackendDevice {
    public interface Device
    public interface Storage
    private val self: MetalDevice get() = this

    override fun location(): DeviceLocation {
        self; return fail()
    }

    override fun sameDevice(
        other: BackendDevice
    ): Boolean = fail()

    override fun setSeed(
        seed: Long
    ): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun getCurrentSeed(): Long {
        self; return Err(Error.NotCompiledWithMetalSupport)
    }

    override fun zerosImpl(
        shape: Shape, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun allocUninit(
        shape: Shape, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <T> storageFromSlice(
        s: List<T>
    ): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun storageFromCpuStorage(
        s: CpuStorage
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun storageFromCpuStorageOwned(
        s: CpuStorage
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun randUniform(
        s: Shape, d: DType, lo: Double, up: Double
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun randNormal(
        s: Shape, d: DType, mean: Double, std: Double
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun synchronize() {
        self; Ok(Unit)
    }

    public companion object {
        public fun new(
            o: Int
        ): MetalDevice = Err(Error.NotCompiledWithMetalSupport)
    }
}

public class MetalStorage : BackendStorage {
    private val self: MetalStorage get() = this

    public companion object {
        public fun new(buffer: Any, device: MetalDevice, count: Int, dtype: DType): MetalStorage {
            buffer.hashCode()
            device.hashCode()
            count.hashCode()
            dtype.hashCode()
            return MetalStorage()
        }
    }

    override fun tryClone(
        l: Layout
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun dtype(): DType {
        self; return fail()
    }

    override fun device(): BackendDevice {
        self; return fail()
    }

    override fun constSet(
        s: Scalar, l: Layout
    ): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun toCpuStorage(): CpuStorage {
        self; return Err(Error.NotCompiledWithMetalSupport)
    }

    override fun affine(
        l: Layout, mul: Double, add: Double
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun powf(
        l: Layout, e: Double
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun elu(
        l: Layout, alpha: Double
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun reduceOp(
        op: ReduceOp, l: Layout, dims: List<Int>
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun cmp(
        op: CmpOp, rhs: BackendStorage, lhsL: Layout, rhsL: Layout
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun toDtype(
        l: Layout, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : UnaryOpT> unaryImpl(
        l: Layout
    ): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : BinaryOpT> binaryImpl(
        rhs: MetalStorage, lhsL: Layout, rhsL: Layout
    ): MetalStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun whereCond(
        l: Layout, t: BackendStorage, tl: Layout, f: BackendStorage, fl: Layout
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun indexSelect(
        rhs: BackendStorage, lhsL: Layout, rhsL: Layout, d: Int
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun gather(
        l: Layout, indexes: BackendStorage, indexesL: Layout, d: Int
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        d: Int
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun scatterAddSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        d: Int
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun indexAdd(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        d: Int
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun copyStridedSrc(
        dst: BackendStorage, dstOffset: Int, srcL: Layout
    ): Unit = Err(Error.NotCompiledWithMetalSupport)

    override fun copy2d(
        dst: BackendStorage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int
    ): Unit =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    override fun avgPool2d(
        l: Layout, k: Pair<Int, Int>, s: Pair<Int, Int>
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun maxPool2d(
        l: Layout, k: Pair<Int, Int>, s: Pair<Int, Int>
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleNearest1d(
        l: Layout, sz: Int
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleNearest2d(
        l: Layout, h: Int, w: Int
    ): BackendStorage = Err(Error.NotCompiledWithMetalSupport)

    override fun upsampleBilinear2d(
        l: Layout,
        h: Int,
        w: Int,
        align: Boolean,
        scaleH: Double?,
        scaleW: Double?
    ): BackendStorage =
        Err(
            Error.NotCompiledWithMetalSupport
        )
}
