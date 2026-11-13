// port-lint: source device.rs
package io.github.kotlinmania.candle.device

import io.github.kotlinmania.candle.cpubackend.CpuDevice
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.FloatDType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.metalbackend.device.MetalDevice
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.utils.cudaIsAvailable
import io.github.kotlinmania.candle.utils.metalIsAvailable
import io.github.kotlinmania.candle.utils.withThreadpool
import io.github.kotlinmania.rustlang.core.F16

/**
 * A `DeviceLocation` represents a physical device whereas multiple `Device`
 * can live on the same location (typically for cuda devices).
 */
public sealed class DeviceLocation {
    public data object Cpu : DeviceLocation()
    public data class Cuda(
        public var gpuId: Int,
    ) : DeviceLocation()
    public data class Metal(
        public var gpuId: Int,
    ) : DeviceLocation()

    public fun rustEquals(other: DeviceLocation): Boolean = this == other
}

/**
 * Cpu, Cuda, or Metal
 */
public sealed class Device {
    public data object Cpu : Device()
    public data class Cuda(
        public var value: CudaDevice,
    ) : Device()
    public data class Metal(
        public var value: MetalDevice,
    ) : Device()

    public fun asCudaDevice(): Result<CudaDevice> =
        when (this) {
            is Cuda -> {
                val (d) = this
                Result.Ok(d)
            }
            Device.Cpu -> bail("expected a cuda device, got cpu")
            is Metal -> bail("expected a cuda device, got Metal")
        }

    public fun asMetalDevice(): Result<MetalDevice> =
        when (this) {
            is Cuda -> bail("expected a metal device, got cuda")
            Device.Cpu -> bail("expected a metal device, got cpu")
            is Metal -> {
                val (d) = this
                Result.Ok(d)
            }
        }

    /**
     * Run `f` with device specific context.
     *
     * On CPU this installs candle's private rayon thread pool for the
     * duration of `f`, keeping worker threads warm across the many short
     * parallel sections in a model forward pass. Currently noop for other backends.
     */
    public fun <R> withContext(f: () -> R): R =
        when (this) {
            Device.Cpu -> withThreadpool(f)
            else -> f()
        }

    public fun setSeed(seed: ULong): Result<Unit> =
        when (this) {
            Device.Cpu -> CpuDevice.setSeed(seed)
            is Cuda -> {
                val (c) = this
                c.setSeed(seed)
            }
            is Metal -> {
                val (m) = this
                m.setSeed(seed)
            }
        }

    public fun getCurrentSeed(): Result<ULong> =
        when (this) {
            Device.Cpu -> CpuDevice.getCurrentSeed()
            is Cuda -> {
                val (c) = this
                c.getCurrentSeed()
            }
            is Metal -> {
                val (m) = this
                m.getCurrentSeed()
            }
        }

    public fun sameDevice(rhs: Device): Boolean =
        run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first == Device.Cpu && matchVal.second == Device.Cpu -> true
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs) = matchVal.first as Cuda
                    val (rhs1) = matchVal.second as Cuda
                    lhs.sameDevice(rhs1)
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs1) = matchVal.first as Metal
                    val (rhs2) = matchVal.second as Metal
                    lhs1.sameDevice(rhs2)
                }
                else -> false
            }
        }

    public fun location(): DeviceLocation =
        when (this) {
            Device.Cpu -> DeviceLocation.Cpu
            is Cuda -> {
                val (device) = this
                device.location()
            }
            is Device.Metal -> {
                val (device1) = this
                device1.location()
            }
        }

    public fun isCpu(): Boolean =
        when (this) {
            Device.Cpu -> true
            else -> false
        }

    public fun isCuda(): Boolean =
        when (this) {
            is Cuda -> true
            else -> false
        }

    public fun isMetal(): Boolean =
        when (this) {
            is Metal -> true
            else -> false
        }

    public fun supportsBf16(): Boolean =
        when (this) {
            is Cuda -> true
            is Metal -> true
            Device.Cpu -> false
        }

    /**
     * Return `BF16` for devices that support it, otherwise default to `F32`.
     */
    public fun bf16DefaultToF32(): DType =
        if (this.supportsBf16()) {
            DType.BF16
        } else {
            DType.F32
        }

    internal fun randUniformF64(lo: Double, up: Double, shape: Shape, dtype: DType): Result<Storage> {
        return when (this) {
            Device.Cpu -> {
                val storage =
                    when (val tryResult = CpuDevice.randUniform(shape, dtype, lo, up)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage))
            }
            is Device.Cuda -> {
                val (device) = this
                // Note: Remove the special case if we start supporting generating f16/bf16 directly.
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage1 =
                        when (val tryResult = device.randUniform(shape, DType.F32, lo, up)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Storage.Cuda(storage1).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage2 =
                        when (val tryResult = device.randUniform(shape, dtype, lo, up)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage2))
                }
            }
            is Device.Metal -> {
                val (device1) = this
                val storage3 =
                    when (val tryResult = device1.randUniform(shape, dtype, lo, up)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage3))
            }
        }
    }

    internal fun <T : FloatDType> randUniform(lo: T, up: T, shape: Shape): Result<Storage> = this.randUniformF64(lo.toF64(), up.toF64(), shape, T.DTYPE)

    internal fun randNormalF64(mean: Double, std: Double, shape: Shape, dtype: DType): Result<Storage> {
        return when (this) {
            Device.Cpu -> {
                val storage =
                    when (val tryResult = CpuDevice.randNormal(shape, dtype, mean, std)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage))
            }
            is Device.Cuda -> {
                val (device) = this
                // Note: Remove the special case if we start supporting generating f16/bf16 directly.
                if (dtype == DType.F16 || dtype == DType.BF16) {
                    val storage1 =
                        when (val tryResult = device.randNormal(shape, DType.F32, mean, std)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Storage.Cuda(storage1).toDtype(Layout.contiguous(shape), dtype)
                } else {
                    val storage2 =
                        when (val tryResult = device.randNormal(shape, dtype, mean, std)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage2))
                }
            }
            is Device.Metal -> {
                val (device1) = this
                val storage3 =
                    when (val tryResult = device1.randNormal(shape, dtype, mean, std)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage3))
            }
        }
    }

    internal fun <T : FloatDType> randNormal(mean: T, std: T, shape: Shape): Result<Storage> = this.randNormalF64(mean.toF64(), std.toF64(), shape, T.DTYPE)

    internal fun zeros(shape: Shape, dtype: DType): Result<Storage> {
        return when (this) {
            Device.Cpu -> {
                val storage =
                    when (val tryResult = CpuDevice.zerosImpl(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage))
            }
            is Device.Cuda -> {
                val (device) = this
                val storage1 =
                    when (val tryResult = device.zerosImpl(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage1))
            }
            is Device.Metal -> {
                val (device1) = this
                val storage2 =
                    when (val tryResult = device1.zerosImpl(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage2))
            }
        }
    }

    internal fun allocUninit(shape: Shape, dtype: DType): Result<Storage> {
        return when (this) {
            Device.Cpu -> {
                val storage =
                    when (val tryResult = CpuDevice.allocUninit(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage))
            }
            is Device.Cuda -> {
                val (device) = this
                val storage1 =
                    when (val tryResult = device.allocUninit(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage1))
            }
            is Device.Metal -> {
                val (device1) = this
                val storage2 =
                    when (val tryResult = device1.allocUninit(shape, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage2))
            }
        }
    }

    internal fun <D : WithDType> storageFromSlice(data: List<D>): Result<Storage> {
        return when (this) {
            Device.Cpu -> Result.Ok(Storage.Cpu(data.toCpuStorage()))
            is Device.Cuda -> {
                val (device) = this
                val storage =
                    when (val tryResult = device.storageFromSlice(data)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage))
            }
            is Device.Metal -> {
                val (device1) = this
                val storage1 =
                    when (val tryResult = device1.storageFromSlice(data)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage1))
            }
        }
    }

    internal fun <A : NdArray> storage(array: A): Result<Storage> {
        return when (this) {
            Device.Cpu -> Result.Ok(Storage.Cpu(array.toCpuStorage()))
            is Device.Cuda -> {
                val (device) = this
                val storage = array.toCpuStorage()
                val storage1 =
                    when (val tryResult = device.storageFromCpuStorageOwned(storage)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage1))
            }
            is Device.Metal -> {
                val (device1) = this
                val storage2 = array.toCpuStorage()
                val storage3 =
                    when (val tryResult = device1.storageFromCpuStorageOwned(storage2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage3))
            }
        }
    }

    internal fun <S : WithDType> storageOwned(data: List<S>, toCpuStorageOwnedSFn: (Any?) -> Any?): Result<Storage> {
        return when (this) {
            Device.Cpu -> Result.Ok(Storage.Cpu(toCpuStorageOwnedSFn(data)))
            is Device.Cuda -> {
                val (device) = this
                val storage = toCpuStorageOwnedSFn(data)
                val storage1 =
                    when (val tryResult = device.storageFromCpuStorageOwned(storage)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage1))
            }
            is Device.Metal -> {
                val (device1) = this
                val storage2 = toCpuStorageOwnedSFn(data)
                val storage3 =
                    when (val tryResult = device1.storageFromCpuStorageOwned(storage2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage3))
            }
        }
    }

    public fun synchronize(): Result<Unit> =
        when (this) {
            Device.Cpu -> Result.Ok(Unit)
            is Cuda -> {
                val (d) = this
                d.synchronize()
            }
            is Metal -> {
                val (d1) = this
                d1.synchronize()
            }
        }

    public companion object {
        public fun newCuda(ordinal: ULong): Result<Device> {
            return Result.Ok(
                Device.Cuda(
                    when (val tryResult = CudaDevice.new(ordinal)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                ),
            )
        }

        public fun newCudaWithStream(ordinal: ULong): Result<Device> {
            return Result.Ok(
                Device.Cuda(
                    when (val tryResult = CudaDevice.newWithStream(ordinal.toInt())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                ),
            )
        }

        public fun newMetal(ordinal: ULong): Result<Device> {
            return Result.Ok(
                Device.Metal(
                    when (val tryResult = MetalDevice.new(ordinal)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                ),
            )
        }

        public fun cudaIfAvailable(ordinal: Int): Result<Device> =
            if (cudaIsAvailable()) {
                Device.newCuda(ordinal)
            } else {
                Result.Ok(Device.Cpu)
            }

        public fun metalIfAvailable(ordinal: Int): Result<Device> =
            if (metalIsAvailable()) {
                Device.newMetal(ordinal)
            } else {
                Result.Ok(Device.Cpu)
            }
    }
}

public interface NdArray {
    public fun shape(): Result<Shape>

    public fun toCpuStorage(): CpuStorage
}

public fun <S : WithDType> S.shape(): Result<Shape> = Result.Ok(Shape.from())

public fun <S : WithDType> S.toCpuStorage(toCpuStorageSFn: (Any?) -> Any?): CpuStorage = toCpuStorageSFn(listOf(this))

public fun <S : WithDType> List<S>.shape(n: Int): Result<Shape> = Result.Ok(Shape.from(this.size))

public fun <S : WithDType> List<S>.toCpuStorage(n: Int, toCpuStorageSFn: (Any?) -> Any?): CpuStorage = toCpuStorageSFn(this.asSlice())

public fun <S : WithDType> List<S>.shape(): Result<Shape> = Result.Ok(Shape.from(this.size))

public fun <S : WithDType> List<S>.toCpuStorage(toCpuStorageSFn: (Any?) -> Any?): CpuStorage = toCpuStorageSFn(this)

public fun <S : WithDType> List<List<S>>.shape(n: Int, m: Int): Result<Shape> = Result.Ok(Shape.from(Pair(m, n)))

public fun <S : WithDType> List<List<S>>.toCpuStorage(n: Int, m: Int, toCpuStorageOwnedSFn: (Any?) -> Any?): CpuStorage = toCpuStorageOwnedSFn(this.concat())

public fun <S : WithDType> List<List<List<S>>>.shape(n1: Int, n2: Int, n3: Int): Result<Shape> = Result.Ok(Shape.from(Triple(n1, n2, n3)))

public fun <S : WithDType> List<List<List<S>>>.toCpuStorage(n1: Int, n2: Int, n3: Int, toCpuStorageOwnedSFn: (Any?) -> Any?): CpuStorage {
    val vec = ArrayList(n1 * n2 * n3)
    for (i1 in 0 until n1) {
        for (i2 in 0 until n2) {
            vec.extend(this[i1][i2])
        }
    }
    return toCpuStorageOwnedSFn(vec)
}

public fun <S : WithDType> List<List<List<List<S>>>>.shape(n1: Int, n2: Int, n3: Int, n4: Int): Result<Shape> = Result.Ok(Shape.from(listOf(n1, n2, n3, n4)))

public fun <S : WithDType> List<List<List<List<S>>>>.toCpuStorage(n1: Int, n2: Int, n3: Int, n4: Int, toCpuStorageOwnedSFn: (Any?) -> Any?): CpuStorage {
    val vec = ArrayList(n1 * n2 * n3 * n4)
    for (i1 in 0 until n1) {
        for (i2 in 0 until n2) {
            for (i3 in 0 until n3) {
                vec.extend(this[i1][i2][i3])
            }
        }
    }
    return toCpuStorageOwnedSFn(vec)
}
