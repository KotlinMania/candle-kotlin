// port-lint: source cuda_backend/device.rs
package io.github.kotlinmania.candle.cudabackend.device

import io.github.kotlinmania.candle.BackendDevice
import io.github.kotlinmania.candle.RustAny
import io.github.kotlinmania.candle.RustTypeId
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.CpuStorageRef
import io.github.kotlinmania.candle.cudabackend.Affine
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorageSlice
import io.github.kotlinmania.candle.cudabackend.error.CudaError
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.cudarc.driver.CudaFunction
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

public typealias Target = CudaFunction

public typealias Storage = CudaStorage

/**
 * Unique identifier for cuda devices.
 */
public open class DeviceId(
    public var value: Int,
) {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }

    public companion object {
        public fun new(): DeviceId {
            val COUNTER: atomic.AtomicUsize  = atomic.AtomicUsize.new(1)
            return DeviceId(COUNTER.fetchAdd(1, atomic.Ordering.Relaxed))
        }
    }
}

public open class CudaRng(
    public var value: cudarc.CudaRng,
) {
    public var field0: cudarc.CudaRng
        get() = value
        set(v) {
            value = v
        }
}

public val CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: Int = 4096

public typealias CudaGraphHtodCacheKey = Triple<DeviceId, RustTypeId, List<UByte>>

public typealias CudaGraphHtodCache = Map<CudaGraphHtodCacheKey, RustAny>

public open class CudaGraphHtodCacheGuard {
    public open fun drop() {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth ->
            check(depth.get() > 0)
            depth.set(depth.get().saturatingSub(1))
        })
    }
}

public open class ModuleStore(
    public var mdls: List<cudarc.CudaModule?>,
)

public open class CudaDevice(
    public var id: DeviceId,
    public var context: cudarc.CudaContext,
    public var modules: RwLock<ModuleStore>,
    public var customModules: RwLock<Map<String, cudarc.CudaModule>>,
    public var stream: cudarc.CudaStream,
    internal var blas: cudarc.CudaBlas,
    public var curand: Mutex<CudaRng>,
    public var seedValue: RwLock<ULong>,
) : BackendDevice {
    public open fun fmt(f: StringBuilder) {
        f.append("CudaDevice(${this.id})")
    }

    public open fun <T : cudarc.DeviceRepr> alloc(len: Int): Result<cudarc.CudaSlice<T>> = this.stream.alloc<T>(len).w()

    public open fun <T> allocZeros(len: Int): Result<cudarc.CudaSlice<T>> where T : cudarc.DeviceRepr, T : cudarc.ValidAsZeroBits = this.stream.allocZeros<T>(len).w()

    public open fun enableCudaGraphHtodCache(): CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth -> depth.set(depth.get() + 1) })
        return CudaGraphHtodCacheGuard()
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>, Dst : cudarc.DevicePtrMut<T>> memcpyHtod(src: Src, dst: Dst): Result<Unit> {
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return this.memcpyHtodCaptureCached(src, dst)
        }
        return this.stream.memcpyHtod(src, dst).w()
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.DevicePtr<T>> cloneDtoh(src: Src): Result<List<T>> {
        this.checkCaptureCopy("clone_dtoh").getOrElse { return Result.failure(it) }
        return this.stream.cloneDtoh(src).w()
    }

    public open fun <T, Src : cudarc.DevicePtr<T>, Dst : cudarc.DevicePtrMut<T>> memcpyDtod(src: Src, dst: Dst): Result<Unit> = this.stream.memcpyDtod(src, dst).w()

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.DevicePtr<T>, Dst : cudarc.HostSlice<T>> memcpyDtoh(src: Src, dst: Dst): Result<Unit> {
        this.checkCaptureCopy("memcpy_dtoh").getOrElse { return Result.failure(it) }
        return this.stream.memcpyDtoh(src, dst).w()
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cloneHtod(src: Src): Result<cudarc.CudaSlice<T>> {
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return this.cloneHtodCaptureCached(src)
        }
        return this.stream.cloneHtod(src).w()
    }

    public open fun checkCaptureCopy(op: String): Result<Unit> {
        if (cudaGraphHtodCacheEnabled()) {
            val isCapturing = this.cudaGraphCaptureActive()
            if (isCapturing) {
                bail("{op} during CUDA graph capture: host/device copies are not permitted while capturing (use param cache for kernel parameters)")
            }
        }
        return Result.success(Unit)
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>, Dst : cudarc.DevicePtrMut<T>> memcpyHtodCaptureCached(src: Src, dst: Dst): Result<Unit> {
        check(dst.len() >= src.len())
        when (val matchVal = this.cachedHtodSlice(src).getOrElse { return Result.failure(it) }) {
            null -> {}
            else -> {
                val cached = matchVal
                return this.stream.memcpyDtod(cached, dst).w()
            }
        }
        return this.stream.memcpyHtod(src, dst).w()
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cloneHtodCaptureCached(src: Src): Result<cudarc.CudaSlice<T>> {
        when (val matchVal = this.cachedHtodSlice(src).getOrElse { return Result.failure(it) }) {
            null -> {}
            else -> {
                val cached = matchVal
                return Result.success(cached)
            }
        }
        return this.stream.cloneHtod(src).w()
    }

    public open fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cachedHtodSlice(src: Src): Result<cudarc.CudaSlice<T>?> {
        val (srcSlice, sync) = src.streamSyncedSlice(this.stream)
        val byteLen = sizeOfVal(srcSlice)
        if (byteLen > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES) {
            if (this.cudaGraphCaptureActive()) {
                bail("CUDA graph capture cannot upload uncached host data")
            }
            return Result.success(null)
        }
        val bytes = fromRawParts(srcSlice.asPtr(), byteLen).toVec()
        val key = Triple(this.id, RustTypeId.of<T>(), bytes)
        when (val matchVal = CUDA_GRAPH_HTOD_CACHE.with({ cache ->
            cache.borrow().get(key)?.let({ cached ->
                cached.downcastRef<cudarc.CudaSlice<T>>().cloned()
            })
        })) {
            null -> {}
            else -> {
                val cached = matchVal
                return Result.success(cached)
            }
        }
        if (this.cudaGraphCaptureActive()) {
            bail("CUDA graph capture missing cached host data")
        }
        val cached1 = this.stream.cloneHtod(src).w().getOrElse { return Result.failure(it) }
        CUDA_GRAPH_HTOD_CACHE.with({ cache ->
            cache.borrowMut().insert(key, cached1.clone())
        })
        return Result.success(cached1)
    }

    public open fun cudaGraphCaptureActive(): Boolean =
        run {
            val matchVal = this.stream.captureStatus()
            when {
                matchVal is Ok && (matchVal as Ok).value != cudarc.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE -> {
                    val (status) = matchVal
                    true
                }
                else -> false
            }
        }

    public open fun cudaStream(): cudarc.CudaStream = this.stream.clone()

    /**
     * When turned on, all cuda tensors **created after calling this function** will
     * not track uses via cuda events.
     *
     * # Safety
     *
     * It is up to the user to ensure proper synchronization between multiple streams:
     * - Ensure that no tensor is freed before a use on another stream is finished.
     * - Ensure that a tensor is not used on another stream before allocation on the
     *   allocating stream finishes.
     * - Ensure that a tensor is not written two concurrently by multiple streams.
     */
    public open fun disableEventTracking() {
        this.context.disableEventTracking()
    }

    public open fun isEventTracking(): Boolean = this.context.isEventTracking()

    public open fun id(): DeviceId = this.id

    public open fun getOrLoadCustomFunc(fnName: String, moduleName: String, ptx: String): Result<CudaFunc> {
        val ms = this.customModules.read().unwrap()
        when (val matchVal = ms.get(moduleName)) {
            null -> {}
            else -> {
                val mdl = matchVal
                val func = mdl.loadFunction(fnName).w().getOrElse { return Result.failure(it) }
                return Result.success(CudaFunc(func = func, stream = this.stream.clone()))
            }
        }
        drop(ms)
        val ms1 = this.customModules.write().unwrap()
        val cudaModule = this.context.loadModule(ptx.into()).w().getOrElse { return Result.failure(it) }
        ms1.insert(moduleName.toString(), cudaModule.clone())
        val func1 = cudaModule.loadFunction(fnName).w().getOrElse { return Result.failure(it) }
        return Result.success(CudaFunc(func = func1, stream = this.stream.clone()))
    }

    public open fun getOrLoadFunc(fnName: String, mdl: kernels.Module): Result<CudaFunc> {
        val ms = this.modules.read().unwrap()
        when (val matchVal = ms.mdls[mdl.index()].asRef()) {
            null -> {}
            else -> {
                val mdl1 = matchVal
                val func = mdl1.loadFunction(fnName).w().getOrElse { return Result.failure(it) }
                return Result.success(CudaFunc(func = func, stream = this.stream.clone()))
            }
        }
        drop(ms)
        val ms1 = this.modules.write().unwrap()
        val cudaModule = this.context.loadModule(mdl.ptx().into()).w().getOrElse { return Result.failure(it) }
        ms1.mdls[mdl.index()] = cudaModule.clone()
        val func1 = cudaModule.loadFunction(fnName).w().getOrElse { return Result.failure(it) }
        return Result.success(CudaFunc(func = func1, stream = this.stream.clone()))
    }

    public open fun cublasHandle(): cudarc.CudaBlas = this.blas.clone()

    public open fun setSeed(seed: ULong): Result<Unit> {
        // We do not call setSeed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        val curand = this.curand.lock().unwrap()
        curand.first = cudarc.CudaRng.new(seed, this.stream.clone()).w().getOrElse { return Result.failure(it) }
        this.seedValue.write().unwrap() = seed
        return Result.success(Unit)
    }

    public open fun getCurrentSeed(): Result<ULong> = Result.success(this.seedValue.read().unwrap())

    public open fun location(): DeviceLocation = DeviceLocation.Cuda(gpuId = this.context.ordinal())

    public open fun sameDevice(rhs: CudaDevice): Boolean = this.id == rhs.id

    public open fun zerosImpl(shape: Shape, dtype: DType): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.u8 -> {
                    val data = this.allocZeros<UByte>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u8(data)
                }
                DType.u32 -> {
                    val data1 = this.allocZeros<UInt>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u32(data1)
                }
                DType.i16 -> {
                    val data2 = this.allocZeros<Short>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i16(data2)
                }
                DType.i32 -> {
                    val data3 = this.allocZeros<Int>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i32(data3)
                }
                DType.i64 -> {
                    val data4 = this.allocZeros<Long>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i64(data4)
                }
                DType.bf16 -> {
                    val data5 = this.allocZeros<Bf16>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.bf16(data5)
                }
                DType.f16 -> {
                    val data6 = this.allocZeros<F16>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f16(data6)
                }
                DType.f32 -> {
                    val data7 = this.allocZeros<Float>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data7)
                }
                DType.f64 -> {
                    val data8 = this.allocZeros<Double>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data8)
                }
                DType.f8e4m3 -> {
                    val data9 = this.allocZeros<F8e4m3>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f8e4m3(data9)
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Result.failure(CudaError.InternalError("Dummy types not supported in CUDA backend").into())
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun randUniform(shape: Shape, dtype: DType, lo: Double, up: Double): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val curand = this.curand.lock().unwrap()
        val slice =
            when (dtype) {
                // Note: Add support for F16 and BF16 though this is likely to require some upstream
                // cudarc changes.
                DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f16, DType.bf16 -> Result.failure(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_uniform")).w().getOrElse { return Result.failure(it) }
                DType.f32 -> {
                    val data = this.alloc<Float>(elemCount).getOrElse { return Result.failure(it) }
                    curand.first.fillWithUniform(data).w().getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data)
                }
                DType.f64 -> {
                    val data1 = this.alloc<Double>(elemCount).getOrElse { return Result.failure(it) }
                    curand.first.fillWithUniform(data1).w().getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data1)
                }
                DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    Result.failure(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_uniform")).w().getOrElse { return Result.failure(it) }
                }
            }
        val slice1 =
            if (lo == 0.0 && up == 1.0) {
                slice
            } else {
                val layout = Layout.contiguous(shape)
                Affine(up - lo, lo).map(slice, this, layout).getOrElse { return Result.failure(it) }
            }
        return Result.success(CudaStorage(slice = slice1, device = this.clone()))
    }

    public open fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): Result<CudaStorage> {
        // Note: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        val elemCount = shape.elemCount()
        val curand = this.curand.lock().unwrap()
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        val elemCountRound =
            if (elemCount % 2 == 1) {
                elemCount + 1
            } else {
                elemCount
            }
        val slice =
            when (dtype) {
                DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f16, DType.bf16 -> Result.failure(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_normal")).w().getOrElse { return Result.failure(it) }
                DType.f32 -> {
                    val data = this.alloc<Float>(elemCountRound).getOrElse { return Result.failure(it) }
                    curand.first.fillWithNormal(data, mean.toFloat(), std.toFloat()).w().getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data)
                }
                DType.f64 -> {
                    val data1 = this.alloc<Double>(elemCountRound).getOrElse { return Result.failure(it) }
                    curand.first.fillWithNormal(data1, mean, std).w().getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data1)
                }
                DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    Result.failure(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_normal")).w().getOrElse { return Result.failure(it) }
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun allocUninit(shape: Shape, dtype: DType): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.u8 -> {
                    val data = this.alloc<UByte>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u8(data)
                }
                DType.u32 -> {
                    val data1 = this.alloc<UInt>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u32(data1)
                }
                DType.i16 -> {
                    val data2 = this.alloc<Short>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i16(data2)
                }
                DType.i32 -> {
                    val data3 = this.alloc<Int>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i32(data3)
                }
                DType.i64 -> {
                    val data4 = this.alloc<Long>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i64(data4)
                }
                DType.bf16 -> {
                    val data5 = this.alloc<Bf16>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.bf16(data5)
                }
                DType.f16 -> {
                    val data6 = this.alloc<F16>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f16(data6)
                }
                DType.f32 -> {
                    val data7 = this.alloc<Float>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data7)
                }
                DType.f64 -> {
                    val data8 = this.alloc<Double>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data8)
                }
                DType.f8e4m3 -> {
                    val data9 = this.alloc<F8e4m3>(elemCount).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f8e4m3(data9)
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Result.failure(CudaError.InternalError("Dummy types not supported in CUDA backend").into())
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun <T : WithDType> storageFromSlice(s: List<T>, cpuStorageRefTFn: (Any?) -> Any?): Result<CudaStorage> {
        val slice =
            when (val matchVal = cpuStorageRefTFn(s)) {
                is CpuStorageRef.U8 -> {
                    val (storage) = matchVal
                    val data = this.cloneHtod(storage).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorageRef.U32 -> {
                    val (storage1) = matchVal
                    val data1 = this.cloneHtod(storage1).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorageRef.I16 -> {
                    val (storage2) = matchVal
                    val data2 = this.cloneHtod(storage2).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorageRef.I32 -> {
                    val (storage3) = matchVal
                    val data3 = this.cloneHtod(storage3).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorageRef.I64 -> {
                    val (storage4) = matchVal
                    val data4 = this.cloneHtod(storage4).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorageRef.Bf16 -> {
                    val (storage5) = matchVal
                    val data5 = this.cloneHtod(storage5).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorageRef.F16 -> {
                    val (storage6) = matchVal
                    val data6 = this.cloneHtod(storage6).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorageRef.F32 -> {
                    val (storage7) = matchVal
                    val data7 = this.cloneHtod(storage7).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorageRef.F64 -> {
                    val (storage8) = matchVal
                    val data8 = this.cloneHtod(storage8).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorageRef.F8e4m3 -> {
                    val (storage9) = matchVal
                    val data9 = this.cloneHtod(storage9).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorageRef.F4 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F6e2m3 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F6e3m2 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F8e8m0 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun storageFromCpuStorage(storage: CpuStorage): Result<CudaStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val (storage1) = storage
                    val data = this.cloneHtod(storage1).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorage.U32 -> {
                    val (storage2) = storage
                    val data1 = this.cloneHtod(storage2).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorage.I16 -> {
                    val (storage3) = storage
                    val data2 = this.cloneHtod(storage3).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorage.I32 -> {
                    val (storage4) = storage
                    val data3 = this.cloneHtod(storage4).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorage.I64 -> {
                    val (storage5) = storage
                    val data4 = this.cloneHtod(storage5).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorage.Bf16 -> {
                    val (storage6) = storage
                    val data5 = this.cloneHtod(storage6).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorage.F16 -> {
                    val (storage7) = storage
                    val data6 = this.cloneHtod(storage7).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorage.F32 -> {
                    val (storage8) = storage
                    val data7 = this.cloneHtod(storage8).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorage.F64 -> {
                    val (storage9) = storage
                    val data8 = this.cloneHtod(storage9).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorage.F8e4m3 -> {
                    val (storage10) = storage
                    val data9 = this.cloneHtod(storage10).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorage.F4 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F6e2m3 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F6e3m2 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F8e8m0 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun storageFromCpuStorageOwned(storage: CpuStorage): Result<CudaStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val (storage1) = storage
                    val data = this.cloneHtod(storage1).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorage.U32 -> {
                    val (storage2) = storage
                    val data1 = this.cloneHtod(storage2).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorage.I16 -> {
                    val (storage3) = storage
                    val data2 = this.cloneHtod(storage3).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorage.I32 -> {
                    val (storage4) = storage
                    val data3 = this.cloneHtod(storage4).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorage.I64 -> {
                    val (storage5) = storage
                    val data4 = this.cloneHtod(storage5).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorage.Bf16 -> {
                    val (storage6) = storage
                    val data5 = this.cloneHtod(storage6).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorage.F16 -> {
                    val (storage7) = storage
                    val data6 = this.cloneHtod(storage7).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorage.F32 -> {
                    val (storage8) = storage
                    val data7 = this.cloneHtod(storage8).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorage.F64 -> {
                    val (storage9) = storage
                    val data8 = this.cloneHtod(storage9).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorage.F8e4m3 -> {
                    val (storage10) = storage
                    val data9 = this.cloneHtod(storage10).getOrElse { return Result.failure(it) }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorage.F4 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F6e2m3 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F6e3m2 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F8e8m0 -> {
                    return Result.failure(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
            }
        return Result.success(CudaStorage(slice = slice, device = this.clone()))
    }

    public open fun synchronize(): Result<Unit> {
        this.stream.synchronize().mapErr(Error::wrap).getOrElse { return Result.failure(it) }
        return Result.success(Unit)
    }

    public companion object {
        public fun newWithStream(ordinal: Int): Result<CudaDevice> {
            val context = cudarc.CudaContext.new(ordinal).w().getOrElse { return Result.failure(it) }
            val stream = context.newStream().w().getOrElse { return Result.failure(it) }
            return CudaDevice.fromContextAndStream(context, stream)
        }

        public fun fromContextAndStream(context: cudarc.CudaContext, stream: cudarc.CudaStream): Result<CudaDevice> {
            val blas = cudarc.CudaBlas.new(stream.clone()).w().getOrElse { return Result.failure(it) }
            val curand = cudarc.CudaRng.new(299792458, stream.clone()).w().getOrElse { return Result.failure(it) }
            val moduleStore = ModuleStore(mdls = List(kernels.ALL_IDS.len()) { null })
            return Result.success(CudaDevice(id = DeviceId.new(), context = context, stream = stream, blas = blas, curand = Mutex.new(CudaRng(curand)), modules = RwLock.new(moduleStore), customModules = RwLock.new(HashMap.new()), seedValue = RwLock.new(299792458)))
        }

        public fun new(ordinal: Int): Result<CudaDevice> {
            val context = cudarc.CudaContext.new(ordinal).w().getOrElse { return Result.failure(it) }
            val stream = context.perThreadStream()
            return CudaDevice.fromContextAndStream(context, stream)
        }
    }
}

internal fun cudaGraphHtodCacheEnabled(): Boolean = CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth -> depth.get() > 0 })

public open class CudaFunc(
    public var func: CudaFunction,
    public var stream: cudarc.CudaStream,
) {
    public open fun deref(): CudaFunction = this.func

    public open fun intoCudaFunction(): CudaFunction = this.func

    public open fun builder(): cudarc.LaunchArgs = this.stream.launchBuilder(this.func)
}

private const val BUILDER_ARG_TEMPLATE: String = "\$(val arg= \$arg; \$b.arg(& arg); )*"

public fun builderArg(b: String = "", arg: String = ""): String =
    arrayOf("\$b" to b, "\$arg" to arg).fold(BUILDER_ARG_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
