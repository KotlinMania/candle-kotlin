// port-lint: source dummy_cuda_backend.rs
package io.github.kotlinmania.candle

import kotlin.jvm.JvmInline

/**
 * Implementation of the Cuda backend when Cuda support has not been compiled in.
 *
 * Provides stub device and storage implementations that return NotCompiledWithCudaSupport
 * errors when CUDA backend operations are invoked without CUDA compilation enabled.
 */

private val Err: (CandleException) -> Nothing = { throw it }
private val Error = CandleException
private val fail: () -> Nothing = {
    throw CandleException.Msg("cuda support has not been enabled, add `cuda` feature to enable.")
}
private val Ok: (Unit) -> Unit = { }

@JvmInline
public value class DeviceId(public val id: Int)

public data class CudaDevice(val gpuId: Int = 0) : BackendDevice {
    public interface Device
    public interface Storage
    private val self: CudaDevice get() = this

    public fun id(): DeviceId {
        self; return DeviceId(0)
    }

    override fun location(): DeviceLocation {
        self; return fail()
    }

    override fun sameDevice(
        other: BackendDevice
    ): Boolean = fail()

    override fun setSeed(
        seed: Long
    ): Unit = Err(Error.NotCompiledWithCudaSupport)

    override fun getCurrentSeed(): Long {
        self; return Err(Error.NotCompiledWithCudaSupport)
    }

    override fun zerosImpl(
        shape: Shape, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun allocUninit(
        shape: Shape, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    public fun <T> storageFromSlice(
        s: List<T>
    ): CudaStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun storageFromCpuStorage(
        s: CpuStorage
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun storageFromCpuStorageOwned(
        s: CpuStorage
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun randUniform(
        s: Shape, d: DType, lo: Double, up: Double
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun randNormal(
        s: Shape, d: DType, mean: Double, std: Double
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun synchronize() {
        self; Ok(Unit)
    }

    public companion object {
        public fun new(
            o: Int
        ): CudaDevice = Err(Error.NotCompiledWithCudaSupport)

        public fun newWithStream(
            o: Int
        ): CudaDevice = Err(Error.NotCompiledWithCudaSupport)
    }
}

public class CudaStorage : BackendStorage {
    private val self: CudaStorage get() = this

    public fun transferToDevice(
        dst: CudaDevice
    ): CudaStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun tryClone(
        l: Layout
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun dtype(): DType {
        self; return fail()
    }

    override fun device(): BackendDevice {
        self; return fail()
    }

    override fun constSet(
        s: Scalar, l: Layout
    ): Unit = Err(Error.NotCompiledWithCudaSupport)

    override fun toCpuStorage(): CpuStorage {
        self; return Err(Error.NotCompiledWithCudaSupport)
    }

    override fun affine(
        l: Layout, mul: Double, add: Double
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun powf(
        l: Layout, e: Double
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun elu(
        l: Layout, alpha: Double
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun reduceOp(
        op: ReduceOp, l: Layout, dims: List<Int>
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun cmp(
        op: CmpOp, rhs: BackendStorage, lhsL: Layout, rhsL: Layout
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun toDtype(
        l: Layout, dtype: DType
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    public fun <B : UnaryOpT> unaryImpl(
        l: Layout
    ): CudaStorage = Err(Error.NotCompiledWithCudaSupport)

    public fun <B : BinaryOpT> binaryImpl(
        rhs: CudaStorage, lhsL: Layout, rhsL: Layout
    ): CudaStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun whereCond(
        l: Layout, t: BackendStorage, tl: Layout, f: BackendStorage, fl: Layout
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    override fun indexSelect(
        rhs: BackendStorage, lhsL: Layout, rhsL: Layout, d: Int
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun gather(
        l: Layout, indexes: BackendStorage, indexesL: Layout, d: Int
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        d: Int
    ): Unit =
        Err(
            Error.NotCompiledWithCudaSupport
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
            Error.NotCompiledWithCudaSupport
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
            Error.NotCompiledWithCudaSupport
        )

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    override fun copyStridedSrc(
        dst: BackendStorage, dstOffset: Int, srcL: Layout
    ): Unit = Err(Error.NotCompiledWithCudaSupport)

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
            Error.NotCompiledWithCudaSupport
        )

    override fun avgPool2d(
        l: Layout, k: Pair<Int, Int>, s: Pair<Int, Int>
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun maxPool2d(
        l: Layout, k: Pair<Int, Int>, s: Pair<Int, Int>
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun upsampleNearest1d(
        l: Layout, sz: Int
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun upsampleNearest2d(
        l: Layout, h: Int, w: Int
    ): BackendStorage = Err(Error.NotCompiledWithCudaSupport)

    override fun upsampleBilinear2d(
        l: Layout,
        h: Int,
        w: Int,
        align: Boolean,
        scaleH: Double?,
        scaleW: Double?
    ): BackendStorage =
        Err(
            Error.NotCompiledWithCudaSupport
        )
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun gemmReducedPrecisionF16(): Boolean {
    return true
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun setGemmReducedPrecisionF16(b: Boolean): Unit = Unit

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun gemmReducedPrecisionBf16(): Boolean {
    return true
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun setGemmReducedPrecisionBf16(b: Boolean): Unit = Unit

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun gemmReducedPrecisionF32(): Boolean {
    return true
}

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun setGemmReducedPrecisionF32(b: Boolean): Unit = Unit
