// port-lint: source dummy_metal_backend.rs
package io.github.kotlinmania.candle.dummymetalbackend

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
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

public typealias Device = MetalDevice

public typealias Storage = MetalStorage

public class MetalDevice : BackendDevice {
    public fun setSeed(arg0: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun getCurrentSeed(): Result<ULong> = Err(Error.NotCompiledWithMetalSupport)

    public fun location(): DeviceLocation {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public fun sameDevice(arg0: MetalDevice): Boolean {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public fun zerosImpl(shape: Shape, dtype: DType): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun allocUninit(shape: Shape, dtype: DType): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun <T : WithDType> storageFromSlice(arg0: List<T>): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun storageFromCpuStorage(arg0: CpuStorage): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun storageFromCpuStorageOwned(arg0: CpuStorage): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun randUniform(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun randNormal(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun synchronize(): Result<Unit> = Ok(Unit)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<MetalDevice> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<MetalDevice> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): MetalDevice =
                    MetalDevice()
                override fun writeNative(addr: Long, value: MetalDevice) {
                }
            }

        public fun new(arg0: ULong): Result<MetalDevice> = Err(Error.NotCompiledWithMetalSupport)
    }
}

public class MetalStorage : BackendStorage {
    public fun tryClone(arg0: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun dtype(): DType {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public fun device(): MetalDevice {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public fun constSet(arg0: Scalar, arg1: Layout): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun toCpuStorage(): Result<CpuStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun affine(arg0: Layout, arg1: Double, arg2: Double): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun powf(arg0: Layout, arg1: Double): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun elu(arg0: Layout, arg1: Double): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun reduceOp(arg0: ReduceOp, arg1: Layout, arg2: List<ULong>): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun cmp(arg0: CmpOp, arg1: MetalStorage, arg2: Layout, arg3: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun toDtype(arg0: Layout, arg1: DType): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : UnaryOpT> unaryImpl(arg0: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun <B : BinaryOpT> binaryImpl(arg0: MetalStorage, arg1: Layout, arg2: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun whereCond(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun conv1d(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ParamsConv1D): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun convTranspose1d(l: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConvTranspose1D): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun conv2d(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ParamsConv2D): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun convTranspose2d(l: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun indexSelect(arg0: MetalStorage, arg1: Layout, arg2: Layout, arg3: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun gather(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun scatterSet(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun scatterAddSet(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun indexAdd(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun matmul(arg0: MetalStorage, arg1: List<Any?>, arg2: Layout, arg3: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun copyStridedSrc(arg0: MetalStorage, arg1: ULong, arg2: Layout): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun copy2d(arg0: MetalStorage, arg1: ULong, arg2: ULong, arg3: ULong, arg4: ULong, arg5: ULong, arg6: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun avgPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun maxPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun upsampleNearest1d(arg0: Layout, arg1: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun upsampleNearest2d(arg0: Layout, arg1: ULong, arg2: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun upsampleBilinear2d(arg0: Layout, arg1: ULong, arg2: ULong, arg3: Boolean, arg4: Double?, arg5: Double?): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<MetalStorage> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<MetalStorage> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): MetalStorage =
                    MetalStorage()
                override fun writeNative(addr: Long, value: MetalStorage) {
                }
            }
    }
}

public sealed class MetalError : From<String> {
    public data class Message(
        public var value: String,
    ) : MetalError()

    public companion object {
        public fun from(e: String): MetalError = MetalError.Message(e)
    }
}
