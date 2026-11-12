package io.github.kotlinmania.candle.metalbackend

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.CpuStorageRef
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.metalbackend.device.DeviceId
import io.github.kotlinmania.candle.metalbackend.device.MetalDevice
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.RESOURCE_OPTIONS
import io.github.kotlinmania.candlemetalkernels.callRandomNormal
import io.github.kotlinmania.candlemetalkernels.callRandomUniform
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.Commands
import io.github.kotlinmania.candlemetalkernels.metal.ResidencySet
import io.github.kotlinmania.objc2foundation.NSRange
import io.github.kotlinmania.objc2foundation.NSString
import io.github.kotlinmania.rustlang.core.ffi.CVoid
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.RwLock

public typealias Storage = MetalStorage

public fun bufferO(buffer: Buffer, l: Layout, dtype: DType): BufferOffset = BufferOffset(buffer = buffer, offsetInBytes = l.startOffset() * dtype.sizeInBytes())

public fun <T> readToVec(buffer: Buffer, n: Int): List<T> {
    val ptr = constPtrFromRustCast<T>(buffer.contents())
    check(!((ptr).rawAddress == 0L))
    val slice = fromRawParts(ptr, n)
    return slice.toVec()
}

public fun backendDeviceMetalDevice(ordinal: Int): Result<MetalDevice> {
    val device = Device.all().swapRemove(ordinal)
    val commandQueue =
        when (val tryResult = device.newCommandQueue().mapErr(MetalError::from)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    commandQueue.setLabel(NSString.fromStr("candle-metal"))
    val kernels = Kernels.new()
    val residencySet = ResidencySet.new(device)
    val seedBuf =
        when (val tryResult = device.newBufferWithData(constPtrFromRustCast<CVoid>(listOf(299792458uL).asPtr()), sizeOf<ULong>(typeLayoutT = PrimitiveLayouts.U64), RESOURCE_OPTIONS).mapErr(MetalError::from)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    seedBuf.setLabel("rng_seed")
    residencySet.insert(seedBuf)
    val seed = Mutex.new(seedBuf)
    val commands =
        when (val tryResult = Commands.new(commandQueue, residencySet).mapErr(MetalError::from)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(MetalDevice(id = DeviceId.new(), device = device, commands = commands, buffers = RwLock.new(HashMap.new()), privateBuffers = RwLock.new(HashMap.new()), kernels = kernels, seed = seed, seedValue = RwLock.new(299792458), residencySet = residencySet))
}

public fun MetalDevice.location(): DeviceLocation = DeviceLocation.Metal(gpuId = this.registryId().toInt())

public fun MetalDevice.sameDevice(rhs: MetalDevice): Boolean = this.id == rhs.id

public fun MetalDevice.allocUninit(shape: Shape, dtype: DType): Result<MetalStorage> {
    val buffer =
        when (
            val tryResult =
                this
                    .newBufferBuilder()
                    .withSizeFor(shape.elemCount(), dtype)
                    .withLabel("alloc-uninit")
                    .build()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(MetalStorage.new(buffer, this.clone(), shape.elemCount(), dtype))
}

public fun MetalDevice.zerosImpl(shape: Shape, dtype: DType): Result<MetalStorage> {
    val size = shape.elemCount() * dtype.sizeInBytes()
    val buffer =
        when (
            val tryResult =
                this
                    .newBufferBuilder()
                    .withZeros(size)
                    .withLabel("zeros")
                    .build()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return Ok(MetalStorage.new(buffer, this.clone(), shape.elemCount(), dtype))
}

public fun <T : WithDType> MetalDevice.storageFromSlice(s: List<T>, cpuStorageRefTFn: (Any?) -> Any?): Result<MetalStorage> {
    val label = "storage_from_slice"
    val (count, buffer) =
        when (val matchVal = cpuStorageRefTFn(s)) {
            is CpuStorageRef.U8 -> {
                val (storage) = matchVal
                Pair(
                    storage.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.U32 -> {
                val (storage1) = matchVal
                Pair(
                    storage1.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage1)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.I16 -> {
                val (storage2) = matchVal
                Pair(
                    storage2.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage2)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.I32 -> {
                val (storage3) = matchVal
                Pair(
                    storage3.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage3)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.I64 -> {
                val (storage4) = matchVal
                Pair(
                    storage4.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage4)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.Bf16 -> {
                val (storage5) = matchVal
                Pair(
                    storage5.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage5)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.F16 -> {
                val (storage6) = matchVal
                Pair(
                    storage6.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage6)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.F32 -> {
                val (storage7) = matchVal
                Pair(
                    storage7.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage7)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.F64 -> {
                val (storage8) = matchVal
                Pair(
                    storage8.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage8)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.F8e4m3 -> {
                val (storage9) = matchVal
                Pair(
                    storage9.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage9)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorageRef.F6e2m3 -> {
                return Err(Error.UnsupportedDTypeForOp(T.DTYPE, "to_dtype").bt())
            }
            is CpuStorageRef.F6e3m2 -> {
                return Err(Error.UnsupportedDTypeForOp(T.DTYPE, "to_dtype").bt())
            }
            is CpuStorageRef.F4 -> {
                return Err(Error.UnsupportedDTypeForOp(T.DTYPE, "to_dtype").bt())
            }
            is CpuStorageRef.F8e8m0 -> {
                return Err(Error.UnsupportedDTypeForOp(T.DTYPE, "to_dtype").bt())
            }
        }
    return Ok(
        MetalDevice.Storage.new(
            when (val tryResult = buffer) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
            this.clone(),
            count,
            T.DTYPE,
        ),
    )
}

public fun MetalDevice.storageFromCpuStorage(storage: CpuStorage): Result<MetalStorage> {
    val label = "storage_from_cpu_storage"
    val (count, buffer) =
        when (storage) {
            is CpuStorage.U8 -> {
                val (storage1) = storage
                Pair(
                    storage1.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage1)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.U32 -> {
                val (storage2) = storage
                Pair(
                    storage2.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage2)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.I16 -> {
                val (storage3) = storage
                Pair(
                    storage3.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage3)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.I32 -> {
                val (storage4) = storage
                Pair(
                    storage4.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage4)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.I64 -> {
                val (storage5) = storage
                Pair(
                    storage5.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage5)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.Bf16 -> {
                val (storage6) = storage
                Pair(
                    storage6.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage6)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.F16 -> {
                val (storage7) = storage
                Pair(
                    storage7.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage7)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.F32 -> {
                val (storage8) = storage
                Pair(
                    storage8.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage8)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.F64 -> {
                val (storage9) = storage
                Pair(
                    storage9.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage9)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.F8e4m3 -> {
                val (storage10) = storage
                Pair(
                    storage10.len(),
                    this
                        .newBufferBuilder()
                        .withData(storage10)
                        .withLabel(label)
                        .build(),
                )
            }
            is CpuStorage.F6e2m3 -> {
                return Err(Error.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
            }
            is CpuStorage.F6e3m2 -> {
                return Err(Error.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
            }
            is CpuStorage.F4 -> {
                return Err(Error.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
            }
            is CpuStorage.F8e8m0 -> {
                return Err(Error.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
            }
        }
    return Ok(
        MetalDevice.Storage.new(
            when (val tryResult = buffer) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
            this.clone(),
            count,
            storage.dtype(),
        ),
    )
}

public fun MetalDevice.storageFromCpuStorageOwned(storage: CpuStorage): Result<MetalStorage> = this.storageFromCpuStorage(storage)

public fun MetalDevice.randUniform(shape: Shape, dtype: DType, min: Double, max: Double): Result<MetalStorage> {
    val name =
        when (dtype) {
            DType.f32 -> "rand_uniform_f32"
            DType.f16 -> "rand_uniform_f16"
            DType.bf16 -> "rand_uniform_bf16"
            else -> {
                val dtype1 = dtype
                bail("rand_uniform not implemented for {dtype:?}")
            }
        }
    val buffer =
        when (
            val tryResult =
                this
                    .newBufferBuilder()
                    .withSizeFor(shape.elemCount(), dtype)
                    .withLabel("rand_uniform")
                    .build()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val encoder =
        when (val tryResult = this.commandEncoder()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = callRandomUniform(this.device, encoder, this.kernels, name, min.toFloat(), max.toFloat(), shape.elemCount(), this.seed.lock().unwrap(), buffer).mapErr(MetalError::from)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(MetalDevice.Storage.new(buffer, this.clone(), shape.elemCount(), dtype))
}

public fun MetalDevice.randNormal(shape: Shape, dtype: DType, mean: Double, stddev: Double): Result<MetalStorage> {
    val name =
        when (dtype) {
            DType.f32 -> "rand_normal_f32"
            DType.f16 -> "rand_normal_f16"
            DType.bf16 -> "rand_normal_bf16"
            else -> {
                val dtype1 = dtype
                bail("rand_uniform not implemented for {dtype:?}")
            }
        }
    val buffer =
        when (
            val tryResult =
                this
                    .newBufferBuilder()
                    .withSizeFor(shape.elemCount(), dtype)
                    .withLabel("rand_normal")
                    .build()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val encoder =
        when (val tryResult = this.commandEncoder()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    when (val tryResult = callRandomNormal(this.device, encoder, this.kernels, name, mean.toFloat(), stddev.toFloat(), shape.elemCount(), this.seed.lock().unwrap(), buffer).mapErr(MetalError::from)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(MetalDevice.Storage.new(buffer, this.clone(), shape.elemCount(), dtype))
}

public fun MetalDevice.setSeed(seed: ULong): Result<Unit> {
    this.seedValue.write().unwrap() = seed
    val seedBuffer =
        when (val tryResult = this.seed.tryLock().mapErr(MetalError::from)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val contents = seedBuffer.data()
    run {
        copyNonoverlapping(listOf(seed).asPtr(), mutPtrFromRustCast<ULong>(contents), 1)
    }
    seedBuffer.didModifyRange(NSRange.new(0, 8))
    return Ok(Unit)
}

public fun MetalDevice.getCurrentSeed(): Result<ULong> = Ok(this.seedValue.read().unwrap())

public fun MetalDevice.synchronize(): Result<Unit> = this.waitUntilCompleted()
