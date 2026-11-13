// port-lint: source backend.rs
package io.github.kotlinmania.candle.backend

import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.shape.Shape

// Traits to Define Backend Behavior

public interface BackendStorage<T, Device> {
    public fun tryClone(arg0: Layout): Result<T>

    public fun dtype(): DType

    public fun device(): Device

    public fun toCpuStorage(): Result<CpuStorage>

    public fun affine(arg0: Layout, arg1: Double, arg2: Double): Result<T>

    public fun powf(arg0: Layout, arg1: Double): Result<T>

    public fun elu(arg0: Layout, arg1: Double): Result<T>

    public fun reduceOp(arg0: ReduceOp, arg1: Layout, arg2: List<Int>): Result<T>

    public fun cmp(arg0: CmpOp, arg1: T, arg2: Layout, arg3: Layout): Result<T>

    public fun toDtype(arg0: Layout, arg1: DType): Result<T>

    public fun <B : UnaryOpT> unaryImpl(arg0: Layout): Result<T>

    public fun <B : BinaryOpT> binaryImpl(arg0: T, arg1: Layout, arg2: Layout): Result<T>

    public fun whereCond(arg0: Layout, arg1: T, arg2: Layout, arg3: T, arg4: Layout): Result<T>

    public fun conv1d(l: Layout, kernel: T, kernelL: Layout, params: ParamsConv1D): Result<T>

    public fun convTranspose1d(l: Layout, kernel: T, kernelL: Layout, params: ParamsConvTranspose1D): Result<T>

    public fun conv2d(l: Layout, kernel: T, kernelL: Layout, params: ParamsConv2D): Result<T>

    public fun convTranspose2d(l: Layout, kernel: T, kernelL: Layout, params: ParamsConvTranspose2D): Result<T>

    public fun avgPool2d(arg0: Layout, arg1: Pair<Int, Int>, arg2: Pair<Int, Int>): Result<T>

    public fun maxPool2d(arg0: Layout, arg1: Pair<Int, Int>, arg2: Pair<Int, Int>): Result<T>

    public fun upsampleNearest1d(arg0: Layout, arg1: Int): Result<T>

    public fun upsampleNearest2d(arg0: Layout, arg1: Int, arg2: Int): Result<T>

    public fun upsampleBilinear2d(arg0: Layout, arg1: Int, arg2: Int, arg3: Boolean, arg4: Double?, arg5: Double?): Result<T>

    public fun gather(arg0: Layout, arg1: T, arg2: Layout, arg3: Int): Result<T>

    public fun scatterSet(arg0: Layout, arg1: T, arg2: Layout, arg3: T, arg4: Layout, arg5: Int): Result<Unit>

    public fun scatterAddSet(arg0: Layout, arg1: T, arg2: Layout, arg3: T, arg4: Layout, arg5: Int): Result<Unit>

    public fun indexSelect(arg0: T, arg1: Layout, arg2: Layout, arg3: Int): Result<T>

    public fun indexAdd(arg0: Layout, arg1: T, arg2: Layout, arg3: T, arg4: Layout, arg5: Int): Result<T>

    public fun matmul(arg0: T, arg1: List<Any?>, arg2: Layout, arg3: Layout): Result<T>

    public fun copyStridedSrc(arg0: T, arg1: Int, arg2: Layout): Result<Unit>

    public fun copy2d(arg0: T, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int): Result<Unit>

    public fun constSet(arg0: Scalar, arg1: Layout): Result<Unit>
}

public interface BackendDevice<T, Storage> {
    public fun new(arg0: Int): Result<T>

    public fun location(): DeviceLocation

    public fun sameDevice(arg0: T): Boolean

    public fun zerosImpl(shape: Shape, dtype: DType): Result<Storage>

    /**
     * # Safety
     * This function is unchecked as it doesn't initialize the underlying data store.
     * The caller should ensure that the data is properly initialized as early as possible
     * after this call.
     */
    public fun allocUninit(shape: Shape, dtype: DType): Result<Storage>

    public fun <T : WithDType> storageFromSlice(arg0: List<T>): Result<Storage>

    public fun storageFromCpuStorage(arg0: CpuStorage): Result<Storage>

    public fun storageFromCpuStorageOwned(arg0: CpuStorage): Result<Storage>

    public fun randUniform(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<Storage>

    public fun randNormal(arg0: Shape, arg1: DType, arg2: Double, arg3: Double): Result<Storage>

    public fun setSeed(arg0: ULong): Result<Unit>

    public fun getCurrentSeed(): Result<ULong>

    /**
     * Synchronize should block until all the operations on the device are completed.
     */
    public fun synchronize(): Result<Unit>
}
