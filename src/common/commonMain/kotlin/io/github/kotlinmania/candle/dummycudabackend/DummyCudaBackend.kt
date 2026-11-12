// port-lint: source dummy_cuda_backend.rs
package io.github.kotlinmania.candle.dummycudabackend

import io.github.kotlinmania.candle.backend.BackendDevice
import io.github.kotlinmania.candle.backend.BackendStorage
import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

// Implementation of the Cuda backend when Cuda support has not been compiled in.

public typealias Device = CudaDevice

public typealias Storage = CudaStorage

public class CudaDevice : BackendDevice {
    public fun id(): DeviceId = DeviceId(0)

    public fun setSeed(arg0: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun getCurrentSeed(): Result<ULong> = Err(Error.NotCompiledWithCudaSupport)

    public fun location(): DeviceLocation {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public fun sameDevice(arg0: CudaDevice): Boolean {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public fun zerosImpl(shape: Shape, dtype: DType): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun allocUninit(shape: Shape, dtype: DType): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun <T : WithDType> storageFromSlice(arg0: List<T>): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun storageFromCpuStorage(arg0: CpuStorage): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun storageFromCpuStorageOwned(arg0: CpuStorage): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun randUniform(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun randNormal(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun synchronize(): Result<Unit> = Ok(Unit)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaDevice> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaDevice> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CudaDevice =
                    CudaDevice()
                override fun writeNative(addr: Long, value: CudaDevice) {
                }
            }

        public fun newWithStream(arg0: ULong): Result<CudaDevice> = Err(Error.NotCompiledWithCudaSupport)

        public fun new(arg0: ULong): Result<CudaDevice> = Err(Error.NotCompiledWithCudaSupport)
    }
}

public class CudaStorage : BackendStorage {
    public fun transferToDevice(dst: CudaDevice): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun tryClone(arg0: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun dtype(): DType {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public fun device(): CudaDevice {
        error("cuda support has not been enabled, add `cuda` feature to enable.")
    }

    public fun constSet(arg0: Scalar, arg1: Layout): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun toCpuStorage(): Result<CpuStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun affine(arg0: Layout, arg1: Double, arg2: Double): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun powf(arg0: Layout, arg1: Double): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun elu(arg0: Layout, arg1: Double): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun reduceOp(arg0: ReduceOp, arg1: Layout, arg2: List<ULong>): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun cmp(arg0: CmpOp, arg1: CudaStorage, arg2: Layout, arg3: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun toDtype(arg0: Layout, arg1: DType): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun <B : UnaryOpT> unaryImpl(arg0: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun <B : BinaryOpT> binaryImpl(arg0: CudaStorage, arg1: Layout, arg2: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun whereCond(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun conv1d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConv1D): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun convTranspose1d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConvTranspose1D): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun conv2d(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ParamsConv2D): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun convTranspose2d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun indexSelect(arg0: CudaStorage, arg1: Layout, arg2: Layout, arg3: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun gather(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun scatterSet(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun scatterAddSet(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun indexAdd(arg0: Layout, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout, arg5: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun matmul(arg0: CudaStorage, arg1: List<Any?>, arg2: Layout, arg3: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun copyStridedSrc(arg0: CudaStorage, arg1: ULong, arg2: Layout): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun copy2d(arg0: CudaStorage, arg1: ULong, arg2: ULong, arg3: ULong, arg4: ULong, arg5: ULong, arg6: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun avgPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun maxPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun upsampleNearest1d(arg0: Layout, arg1: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun upsampleNearest2d(arg0: Layout, arg1: ULong, arg2: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun upsampleBilinear2d(arg0: Layout, arg1: ULong, arg2: ULong, arg3: Boolean, arg4: Double?, arg5: Double?): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaStorage> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaStorage> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CudaStorage =
                    CudaStorage()
                override fun writeNative(addr: Long, value: CudaStorage) {
                }
            }
    }
}

public data class DeviceId(
    public var value: Int = 0,
) {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<DeviceId> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<DeviceId> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): DeviceId =
                    DeviceId(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: DeviceId) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.value)
                }
            }
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
