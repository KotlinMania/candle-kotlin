// port-lint: source dummy_cuda_backend.rs
package io.github.kotlinmania.candle

import kotlin.jvm.JvmInline

/**
 * Implementation of the Cuda backend when Cuda support has not been compiled in.
 */

@JvmInline
public value class DeviceId(
    public val id: Int,
)

public data class CudaDevice(
    val gpuId: Int = 0,
) : BackendDevice {
    public interface Device
    public interface Storage

    public fun id(): DeviceId = DeviceId(gpuId)

    override fun location(): DeviceLocation =
        throw CandleException.NotCompiledWithCudaSupport

    override fun sameDevice(other: BackendDevice): Boolean =
        throw CandleException.NotCompiledWithCudaSupport

    override fun setSeed(seed: Long): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun getCurrentSeed(): Long =
        throw CandleException.NotCompiledWithCudaSupport

    override fun zerosImpl(shape: Shape, dtype: DType): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun allocUninit(shape: Shape, dtype: DType): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    public fun <T> storageFromSlice(slice: List<T>): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun storageFromCpuStorage(storage: CpuStorage): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun storageFromCpuStorageOwned(storage: CpuStorage): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun randUniform(shape: Shape, dtype: DType, min: Double, max: Double): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun synchronize() {}

    public companion object {
        public fun new(ordinal: Int): CudaDevice =
            throw CandleException.NotCompiledWithCudaSupport

        public fun newWithStream(ordinal: Int): CudaDevice =
            throw CandleException.NotCompiledWithCudaSupport
    }
}

public class CudaStorage : BackendStorage {
    public fun transferToDevice(dst: CudaDevice): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun tryClone(layout: Layout): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun dtype(): DType =
        throw CandleException.NotCompiledWithCudaSupport

    override fun device(): BackendDevice =
        throw CandleException.NotCompiledWithCudaSupport

    override fun constSet(v: Scalar, l: Layout): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun toCpuStorage(): CpuStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun affine(layout: Layout, mul: Double, add: Double): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun powf(layout: Layout, e: Double): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun elu(layout: Layout, alpha: Double): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun reduceOp(op: ReduceOp, layout: Layout, dims: List<Int>): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun cmp(op: CmpOp, rhs: BackendStorage, lhsL: Layout, rhsL: Layout): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun toDtype(layout: Layout, targetDType: DType): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    public fun <B : UnaryOpT> unaryImpl(layout: Layout): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport

    public fun <B : BinaryOpT> binaryImpl(
        rhs: CudaStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun whereCond(
        layout: Layout,
        onTrue: BackendStorage,
        trueL: Layout,
        onFalse: BackendStorage,
        falseL: Layout,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun indexSelect(rhs: BackendStorage, lhsL: Layout, rhsL: Layout, dim: Int): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun gather(l: Layout, indexes: BackendStorage, indexesL: Layout, dim: Int): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun scatterAddSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun indexAdd(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun copyStridedSrc(dst: BackendStorage, dstOffset: Int, srcLayout: Layout): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun copy2d(
        dst: BackendStorage,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ): Unit =
        throw CandleException.NotCompiledWithCudaSupport

    override fun avgPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun maxPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun upsampleNearest1d(l: Layout, targetSize: Int): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun upsampleNearest2d(l: Layout, targetH: Int, targetW: Int): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport

    override fun upsampleBilinear2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?,
    ): BackendStorage =
        throw CandleException.NotCompiledWithCudaSupport
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
public fun setGemmReducedPrecisionF16(b: Boolean) {
    b.hashCode()
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun gemmReducedPrecisionBf16(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun setGemmReducedPrecisionBf16(b: Boolean) {
    b.hashCode()
}

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun gemmReducedPrecisionF32(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun setGemmReducedPrecisionF32(b: Boolean) {
    b.hashCode()
}
