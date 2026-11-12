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
private val fail: () -> Nothing = {
    throw CandleException
        .Msg("cuda support has not been enabled, add `cuda` feature to enable.")
}
private val Ok: (Unit) -> Unit = { }

@JvmInline
public value class DeviceId(
    public val id: Int,
)

public data class CudaDevice(
    val gpuId: Int = 0,
) : BackendDevice {
    public interface Device

    public interface Storage

    private val self: CudaDevice get() = this
    public val ordinal: Int get() = gpuId

    public fun clone(): CudaDevice = this

    public fun id(): DeviceId {
        self
        return DeviceId(0)
    }

    override fun location(): DeviceLocation {
        self
        fail()
    }

    override fun sameDevice(other: BackendDevice): Boolean = fail()

    override fun setSeed(seed: Long): Unit = Err(CandleException.NotCompiledWithCudaSupport)

    override fun getCurrentSeed(): Long {
        self
        Err(CandleException.NotCompiledWithCudaSupport)
    }

    override fun zerosImpl(
        shape: Shape,
        dtype: DType,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    public fun <T> storageFromSlice(s: List<T>): CudaStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun storageFromCpuStorage(storage: CpuStorage): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun storageFromCpuStorageOwned(storage: CpuStorage): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun randUniform(
        shape: Shape,
        dtype: DType,
        min: Double,
        max: Double,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        std: Double,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun synchronize() {
        self
        Ok(Unit)
    }

    public companion object {
        public fun new(o: Int): CudaDevice = Err(CandleException.NotCompiledWithCudaSupport)

        public fun newWithStream(o: Int): CudaDevice = Err(CandleException.NotCompiledWithCudaSupport)
    }
}

public class CudaStorage() : BackendStorage {
    public constructor(device: CudaDevice, dtype: DType = DType.F32) : this()

    private val self: CudaStorage get() = this
    internal val dtype: DType get() = dtype()
    public val slice: io.github.kotlinmania.candle.cuda.CudaStorageSlice get() = fail()

    public fun <T> asCudaSlice(): List<T> = fail()

    public fun matmulWithLayout(
        rhs: CudaStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): CudaStorage = fail()

    public companion object {
        public fun wrapCudaSlice(
            slice: Any,
            dev: CudaDevice,
        ): CudaStorage = fail()
    }

    public fun transferToDevice(dst: CudaDevice): CudaStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun tryClone(layout: Layout): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

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
    ): Unit = Err(CandleException.NotCompiledWithCudaSupport)

    override fun toCpuStorage(): CpuStorage {
        self
        Err(CandleException.NotCompiledWithCudaSupport)
    }

    override fun affine(
        layout: Layout,
        mul: Double,
        add: Double,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun powf(
        layout: Layout,
        e: Double,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun elu(
        layout: Layout,
        alpha: Double,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        dims: List<Int>,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun cmp(
        op: CmpOp,
        rhs: BackendStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun toDtype(
        layout: Layout,
        targetDType: DType,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    public fun <B : UnaryOpT> unaryImpl(l: Layout): CudaStorage = Err(CandleException.NotCompiledWithCudaSupport)

    public fun <B : BinaryOpT> binaryImpl(
        rhs: CudaStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): CudaStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun whereCond(
        layout: Layout,
        onTrue: BackendStorage,
        trueL: Layout,
        onFalse: BackendStorage,
        falseL: Layout,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun indexSelect(
        rhs: BackendStorage,
        lhsL: Layout,
        rhsL: Layout,
        dim: Int,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun gather(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        dim: Int,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        Err(
            CandleException.NotCompiledWithCudaSupport,
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
            CandleException.NotCompiledWithCudaSupport,
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
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun copyStridedSrc(
        dst: BackendStorage,
        dstOffset: Int,
        srcLayout: Layout,
    ): Unit = Err(CandleException.NotCompiledWithCudaSupport)

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
            CandleException.NotCompiledWithCudaSupport,
        )

    override fun avgPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun maxPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun upsampleNearest1d(
        l: Layout,
        targetSize: Int,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun upsampleNearest2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
    ): BackendStorage = Err(CandleException.NotCompiledWithCudaSupport)

    override fun upsampleBilinear2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?,
    ): BackendStorage =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun gemmReducedPrecisionF16(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
public fun setGemmReducedPrecisionF16(b: Boolean): Unit = Unit

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun gemmReducedPrecisionBf16(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun setGemmReducedPrecisionBf16(b: Boolean): Unit = Unit

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun gemmReducedPrecisionF32(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun setGemmReducedPrecisionF32(b: Boolean): Unit = Unit
