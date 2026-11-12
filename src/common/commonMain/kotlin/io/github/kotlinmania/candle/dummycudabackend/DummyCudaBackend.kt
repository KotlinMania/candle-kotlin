// port-lint: source dummy_cuda_backend.rs
package io.github.kotlinmania.candle.dummycudabackend

import io.github.kotlinmania.candle.BackendDevice
import io.github.kotlinmania.candle.BackendStorage
import io.github.kotlinmania.candle.BinaryOpT
import io.github.kotlinmania.candle.CmpOp
import io.github.kotlinmania.candle.ParamsConv1D
import io.github.kotlinmania.candle.ParamsConv2D
import io.github.kotlinmania.candle.ParamsConvTranspose1D
import io.github.kotlinmania.candle.ParamsConvTranspose2D
import io.github.kotlinmania.candle.ReduceOp
import io.github.kotlinmania.candle.Scalar
import io.github.kotlinmania.candle.UnaryOpT
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape

// Implementation of the Cuda backend when Cuda support has not been compiled in.

public typealias Device = CudaDevice

public typealias Storage = CudaStorage

public open class CudaDevice : BackendDevice {
    public open fun id(): DeviceId = DeviceId(0)

    public open fun setSeed(arg0: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun getCurrentSeed(): Result<ULong> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun location(): DeviceLocation {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public open fun sameDevice(arg0: CudaDevice): Boolean {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public open fun zerosImpl(shape: Shape, dtype: DType): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun allocUninit(shape: Shape, dtype: DType): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun <T : WithDType> storageFromSlice(arg0: List<T>): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun storageFromCpuStorage(arg0: CpuStorage): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun storageFromCpuStorageOwned(arg0: CpuStorage): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun randUniform(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun randNormal(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun synchronize(): Result<Unit> = Result.success(Unit)

    public companion object {
        public fun newWithStream(arg0: ULong): Result<CudaDevice> = Result.failure(Error.NotCompiledWithCudaSupport)

        public fun new(arg0: ULong): Result<CudaDevice> = Result.failure(Error.NotCompiledWithCudaSupport)
    }
}

public open class CudaStorage : BackendStorage {
    public open fun transferToDevice(dst: CudaDevice): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun tryClone(arg0: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun dtype(): DType {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public open fun device(): CudaDevice {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public open fun constSet(arg0: Scalar, arg1: Layout): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun toCpuStorage(): Result<CpuStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun affine(arg0: Layout, arg1: Double, arg2: Double): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun powf(arg0: Layout, arg1: Double): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun elu(arg0: Layout, arg1: Double): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun reduceOp(arg0: ReduceOp, arg1: Layout, arg2: List<ULong>): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun cmp(arg0: CmpOp, arg1: CudaStorage, arg2: Layout, arg3: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun toDtype(arg0: Layout, arg1: DType): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun <B : UnaryOpT> unaryImpl(arg0: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun <B : BinaryOpT> binaryImpl(arg0: CudaStorage, arg1: Layout, arg2: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun whereCond(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun conv1d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConv1D): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun convTranspose1d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConvTranspose1D): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun conv2d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConv2D): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun convTranspose2d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun indexSelect(arg0: CudaStorage, arg1: Layout, arg2: Layout, arg3: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun gather(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun scatterSet(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun scatterAddSet(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun indexAdd(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun matmul(arg0: CudaStorage, arg1: List<Any?>, arg2: Layout, arg3: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun copyStridedSrc(arg0: CudaStorage, arg1: ULong, arg2: Layout): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun copy2d(arg0: CudaStorage, arg1: ULong, arg2: ULong, arg3: ULong, arg4: ULong, arg5: ULong, arg6: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun avgPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun maxPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun upsampleNearest1d(arg0: Layout, arg1: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun upsampleNearest2d(arg0: Layout, arg1: ULong, arg2: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun upsampleBilinear2d(arg0: Layout, arg1: ULong, arg2: ULong, arg3: Boolean, arg4: Double?, arg5: Double?): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)
}

public open class DeviceId(
    public var value: Int,
) {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }
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
public fun setGemmReducedPrecisionF16(arg0: Boolean) {}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun gemmReducedPrecisionBf16(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
public fun setGemmReducedPrecisionBf16(arg0: Boolean) {}

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun gemmReducedPrecisionF32(): Boolean = true

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
public fun setGemmReducedPrecisionF32(b: Boolean) {}
