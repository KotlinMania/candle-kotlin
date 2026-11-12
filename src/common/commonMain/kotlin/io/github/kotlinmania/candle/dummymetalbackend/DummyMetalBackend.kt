// port-lint: source dummy_metal_backend.rs
package io.github.kotlinmania.candle.dummymetalbackend

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

public typealias Device = MetalDevice

public typealias Storage = MetalStorage

public open class MetalDevice : BackendDevice {
    public open fun setSeed(arg0: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun getCurrentSeed(): Result<ULong> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun location(): DeviceLocation {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public open fun sameDevice(arg0: MetalDevice): Boolean {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public open fun zerosImpl(shape: Shape, dtype: DType): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun allocUninit(shape: Shape, dtype: DType): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun <T : WithDType> storageFromSlice(arg0: List<T>): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun storageFromCpuStorage(arg0: CpuStorage): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun storageFromCpuStorageOwned(arg0: CpuStorage): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun randUniform(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun randNormal(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun synchronize(): Result<Unit> = Result.success(Unit)

    public companion object {
        public fun new(arg0: ULong): Result<MetalDevice> = Result.failure(Error.NotCompiledWithMetalSupport)
    }
}

public open class MetalStorage : BackendStorage {
    public open fun tryClone(arg0: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun dtype(): DType {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public open fun device(): MetalDevice {
        error("metal support has not been enabled, add `metal` feature to enable.")
    }

    public open fun constSet(arg0: Scalar, arg1: Layout): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun toCpuStorage(): Result<CpuStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun affine(arg0: Layout, arg1: Double, arg2: Double): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun powf(arg0: Layout, arg1: Double): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun elu(arg0: Layout, arg1: Double): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun reduceOp(arg0: ReduceOp, arg1: Layout, arg2: List<ULong>): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun cmp(arg0: CmpOp, arg1: MetalStorage, arg2: Layout, arg3: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun toDtype(arg0: Layout, arg1: DType): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun <B : UnaryOpT> unaryImpl(arg0: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun <B : BinaryOpT> binaryImpl(arg0: MetalStorage, arg1: Layout, arg2: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun whereCond(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun conv1d(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ParamsConv1D): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun convTranspose1d(l: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConvTranspose1D): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun conv2d(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ParamsConv2D): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun convTranspose2d(l: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun indexSelect(arg0: MetalStorage, arg1: Layout, arg2: Layout, arg3: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun gather(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun scatterSet(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun scatterAddSet(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun indexAdd(arg0: Layout, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout, arg5: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun matmul(arg0: MetalStorage, arg1: List<Any?>, arg2: Layout, arg3: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun copyStridedSrc(arg0: MetalStorage, arg1: ULong, arg2: Layout): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun copy2d(arg0: MetalStorage, arg1: ULong, arg2: ULong, arg3: ULong, arg4: ULong, arg5: ULong, arg6: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun avgPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun maxPool2d(arg0: Layout, arg1: Pair<ULong, ULong>, arg2: Pair<ULong, ULong>): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun upsampleNearest1d(arg0: Layout, arg1: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun upsampleNearest2d(arg0: Layout, arg1: ULong, arg2: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun upsampleBilinear2d(arg0: Layout, arg1: ULong, arg2: ULong, arg3: Boolean, arg4: Double?, arg5: Double?): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)
}

public sealed class MetalError : From<String> {
    public data class Message(public var value: String) : MetalError()

    public companion object {
        public fun from(e: String): MetalError = MetalError.Message(e)
    }
}
