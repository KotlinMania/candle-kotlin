// port-lint: source cuda_backend/device.rs
package io.github.kotlinmania.candle.cudabackend.device

import io.github.kotlinmania.candle.RustAny
import io.github.kotlinmania.candle.RustTypeId
import io.github.kotlinmania.candle.backend.BackendDevice
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
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.RwLock
import io.github.kotlinmania.rustlang.std.sync.atomic

public typealias Target = CudaFunction

public typealias Storage = CudaStorage

/**
 * Unique identifier for cuda devices.
 */
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

        public fun new(): DeviceId {
            val COUNTER: atomic.AtomicUsize = atomic.AtomicUsize.new(1)
            return DeviceId(COUNTER.fetchAdd(1, atomic.Ordering.Relaxed))
        }
    }
}

public data class CudaRng(
    public var value: cudarc.CudaRng = cudarc.CudaRng(),
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

public class CudaGraphHtodCacheGuard {
    public fun drop() {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth ->
            check(depth.get() > 0)
            depth.set(depth.get().saturatingSub(1))
        })
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaGraphHtodCacheGuard> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CudaGraphHtodCacheGuard> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CudaGraphHtodCacheGuard =
                    CudaGraphHtodCacheGuard()
                override fun writeNative(addr: Long, value: CudaGraphHtodCacheGuard) {
                }
            }
    }
}

public data class ModuleStore(
    public var mdls: List<cudarc.CudaModule> = List(kernels.ALL_IDS.len()) { cudarc.CudaModule() },
)

public class CudaDevice internal constructor(
    public var id: DeviceId,
    public var context: cudarc.CudaContext,
    modulesCtorArg: RwLock<ModuleStore>,
    customModulesCtorArg: RwLock<Map<String, cudarc.CudaModule>>,
    public var stream: cudarc.CudaStream,
    internal var blas: cudarc.CudaBlas,
    curandCtorArg: Mutex<CudaRng>,
    seedValueCtorArg: RwLock<ULong>,
) : BackendDevice {
    public var modules: RwLock<ModuleStore> = modulesCtorArg
        internal set

    public var customModules: RwLock<Map<String, cudarc.CudaModule>> = customModulesCtorArg
        internal set

    public var curand: Mutex<CudaRng> = curandCtorArg
        internal set

    public var seedValue: RwLock<ULong> = seedValueCtorArg
        internal set

    public constructor() : this(
        id = DeviceId(),
        context = cudarc.CudaContext(),
        modulesCtorArg = RwLock<ModuleStore>(),
        customModulesCtorArg = RwLock<Map<String, cudarc.CudaModule>>(),
        stream = cudarc.CudaStream(),
        blas = cudarc.CudaBlas(),
        curandCtorArg = Mutex<CudaRng>(),
        seedValueCtorArg = RwLock<ULong>(),
    )

    public fun fmt(f: StringBuilder) {
        f.append("CudaDevice(${this.id})")
    }

    public fun <T : cudarc.DeviceRepr> alloc(len: Int): Result<cudarc.CudaSlice<T>> = this.stream.alloc<T>(len).w()

    public fun <T> allocZeros(len: Int): Result<cudarc.CudaSlice<T>> where T : cudarc.DeviceRepr, T : cudarc.ValidAsZeroBits = this.stream.allocZeros<T>(len).w()

    public fun enableCudaGraphHtodCache(): CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth -> depth.set(depth.get() + 1) })
        return CudaGraphHtodCacheGuard()
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>, Dst : cudarc.DevicePtrMut<T>> memcpyHtod(src: Src, dst: Dst): Result<Unit> {
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return this.memcpyHtodCaptureCached(src, dst)
        }
        return this.stream.memcpyHtod(src, dst).w()
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.DevicePtr<T>> cloneDtoh(src: Src): Result<List<T>> {
        when (val tryResult = this.checkCaptureCopy("clone_dtoh")) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return this.stream.cloneDtoh(src).w()
    }

    public fun <T, Src : cudarc.DevicePtr<T>, Dst : cudarc.DevicePtrMut<T>> memcpyDtod(src: Src, dst: Dst): Result<Unit> = this.stream.memcpyDtod(src, dst).w()

    public fun <T : cudarc.DeviceRepr, Src : cudarc.DevicePtr<T>, Dst : cudarc.HostSlice<T>> memcpyDtoh(src: Src, dst: Dst): Result<Unit> {
        when (val tryResult = this.checkCaptureCopy("memcpy_dtoh")) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return this.stream.memcpyDtoh(src, dst).w()
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cloneHtod(src: Src): Result<cudarc.CudaSlice<T>> {
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return this.cloneHtodCaptureCached(src)
        }
        return this.stream.cloneHtod(src).w()
    }

    public fun checkCaptureCopy(op: String): Result<Unit> {
        if (cudaGraphHtodCacheEnabled()) {
            val isCapturing = this.cudaGraphCaptureActive()
            if (isCapturing) {
                bail("{op} during CUDA graph capture: host/device copies are not permitted while capturing (use param cache for kernel parameters)")
            }
        }
        return Ok(Unit)
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>, Dst : cudarc.DevicePtrMut<T>> memcpyHtodCaptureCached(src: Src, dst: Dst): Result<Unit> {
        check(dst.len() >= src.len())
        when (
            val matchVal =
                when (val tryResult = this.cachedHtodSlice(src)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
        ) {
            null -> {}
            else -> {
                val cached = matchVal
                return this.stream.memcpyDtod(cached, dst).w()
            }
        }
        return this.stream.memcpyHtod(src, dst).w()
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cloneHtodCaptureCached(src: Src): Result<cudarc.CudaSlice<T>> {
        when (
            val matchVal =
                when (val tryResult = this.cachedHtodSlice(src)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
        ) {
            null -> {}
            else -> {
                val cached = matchVal
                return Ok(cached)
            }
        }
        return this.stream.cloneHtod(src).w()
    }

    public fun <T : cudarc.DeviceRepr, Src : cudarc.HostSlice<T>> cachedHtodSlice(src: Src): Result<cudarc.CudaSlice<T>?> {
        val (srcSlice, sync) = src.streamSyncedSlice(this.stream)
        val byteLen = sizeOfVal(srcSlice)
        if (byteLen > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES) {
            if (this.cudaGraphCaptureActive()) {
                bail("CUDA graph capture cannot upload uncached host data")
            }
            return Ok(null)
        }
        val bytes = fromRawParts(srcSlice.asPtr().cast<UByte>(), byteLen).toVec()
        val key = Triple(this.id, RustTypeId.of<T>(), bytes)
        when (
            val matchVal =
                CUDA_GRAPH_HTOD_CACHE.with({ cache ->
                    cache.borrow().get(key)?.let({ cached ->
                        cached.downcastRef<cudarc.CudaSlice<T>>().cloned()
                    })
                })
        ) {
            null -> {}
            else -> {
                val cached = matchVal
                return Ok(cached)
            }
        }
        if (this.cudaGraphCaptureActive()) {
            bail("CUDA graph capture missing cached host data")
        }
        val cached1 =
            when (val tryResult = this.stream.cloneHtod(src).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        CUDA_GRAPH_HTOD_CACHE.with({ cache ->
            cache.borrowMut().insert(key, cached1.clone())
        })
        return Ok(cached1)
    }

    public fun cudaGraphCaptureActive(): Boolean =
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

    public fun cudaStream(): cudarc.CudaStream = this.stream.clone()

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
    public fun disableEventTracking() {
        this.context.disableEventTracking()
    }

    public fun isEventTracking(): Boolean = this.context.isEventTracking()

    public fun getOrLoadCustomFunc(fnName: String, moduleName: String, ptx: String): Result<CudaFunc> {
        val ms = this.customModules.read().unwrap()
        when (val matchVal = ms.get(moduleName)) {
            null -> {}
            else -> {
                val mdl = matchVal
                val func =
                    when (val tryResult = mdl.loadFunction(fnName).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                return Ok(CudaFunc(func = func, stream = this.stream.clone()))
            }
        }
        drop(ms)
        val ms1 = this.customModules.write().unwrap()
        val cudaModule =
            when (val tryResult = this.context.loadModule(ptx.into()).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        ms1.insert(moduleName.toString(), cudaModule.clone())
        val func1 =
            when (val tryResult = cudaModule.loadFunction(fnName).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaFunc(func = func1, stream = this.stream.clone()))
    }

    public fun getOrLoadFunc(fnName: String, mdl: kernels.Module): Result<CudaFunc> {
        val ms = this.modules.read().unwrap()
        when (val matchVal = ms.mdls[mdl.index()].asRef()) {
            null -> {}
            else -> {
                val mdl1 = matchVal
                val func =
                    when (val tryResult = mdl1.loadFunction(fnName).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                return Ok(CudaFunc(func = func, stream = this.stream.clone()))
            }
        }
        drop(ms)
        val ms1 = this.modules.write().unwrap()
        val cudaModule =
            when (val tryResult = this.context.loadModule(mdl.ptx().into()).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        ms1.mdls[mdl.index()] = cudaModule.clone()
        val func1 =
            when (val tryResult = cudaModule.loadFunction(fnName).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaFunc(func = func1, stream = this.stream.clone()))
    }

    public fun cublasHandle(): cudarc.CudaBlas = this.blas.clone()

    public fun setSeed(seed: ULong): Result<Unit> {
        // We do not call setSeed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        val curand = this.curand.lock().unwrap()
        curand.first =
            when (val tryResult = cudarc.CudaRng.new(seed, this.stream.clone()).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        this.seedValue.write().unwrap() = seed
        return Ok(Unit)
    }

    public fun getCurrentSeed(): Result<ULong> = Ok(this.seedValue.read().unwrap())

    public fun location(): DeviceLocation = DeviceLocation.Cuda(gpuId = this.context.ordinal())

    public fun sameDevice(rhs: CudaDevice): Boolean = this.id == rhs.id

    public fun zerosImpl(shape: Shape, dtype: DType): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.u8 -> {
                    val data =
                        when (val tryResult = this.allocZeros<UByte>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(data)
                }
                DType.u32 -> {
                    val data1 =
                        when (val tryResult = this.allocZeros<UInt>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(data1)
                }
                DType.i16 -> {
                    val data2 =
                        when (val tryResult = this.allocZeros<Short>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(data2)
                }
                DType.i32 -> {
                    val data3 =
                        when (val tryResult = this.allocZeros<Int>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(data3)
                }
                DType.i64 -> {
                    val data4 =
                        when (val tryResult = this.allocZeros<Long>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(data4)
                }
                DType.bf16 -> {
                    val data5 =
                        when (val tryResult = this.allocZeros<Bf16>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(data5)
                }
                DType.f16 -> {
                    val data6 =
                        when (val tryResult = this.allocZeros<F16>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(data6)
                }
                DType.f32 -> {
                    val data7 =
                        when (val tryResult = this.allocZeros<Float>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(data7)
                }
                DType.f64 -> {
                    val data8 =
                        when (val tryResult = this.allocZeros<Double>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(data8)
                }
                DType.f8e4m3 -> {
                    val data9 =
                        when (val tryResult = this.allocZeros<F8e4m3>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(data9)
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Err(CudaError.InternalError("Dummy types not supported in CUDA backend").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun randUniform(shape: Shape, dtype: DType, lo: Double, up: Double): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val curand = this.curand.lock().unwrap()
        val slice =
            when (dtype) {
                // Note: Add support for F16 and BF16 though this is likely to require some upstream
                // cudarc changes.
                DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f16, DType.bf16 ->
                    when (val tryResult = Err(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_uniform")).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f32 -> {
                    val data =
                        when (val tryResult = this.alloc<Float>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    when (val tryResult = curand.first.fillWithUniform(data).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f32(data)
                }
                DType.f64 -> {
                    val data1 =
                        when (val tryResult = this.alloc<Double>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    when (val tryResult = curand.first.fillWithUniform(data1).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f64(data1)
                }
                DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    when (val tryResult = Err(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_uniform")).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        val slice1 =
            if (lo == 0.0 && up == 1.0) {
                slice
            } else {
                val layout = Layout.contiguous(shape)
                when (val tryResult = Affine(up - lo, lo).map(slice, this, layout)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        return Ok(CudaStorage(slice = slice1, device = this.clone()))
    }

    public fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): Result<CudaStorage> {
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
                DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f16, DType.bf16 ->
                    when (val tryResult = Err(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_normal")).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f32 -> {
                    val data =
                        when (val tryResult = this.alloc<Float>(elemCountRound)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    when (val tryResult = curand.first.fillWithNormal(data, mean.toFloat(), std.toFloat()).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f32(data)
                }
                DType.f64 -> {
                    val data1 =
                        when (val tryResult = this.alloc<Double>(elemCountRound)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    when (val tryResult = curand.first.fillWithNormal(data1, mean, std).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f64(data1)
                }
                DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    when (val tryResult = Err(CudaError.UnsupportedDtype(dtype = dtype, op = "rand_normal")).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun allocUninit(shape: Shape, dtype: DType): Result<CudaStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.u8 -> {
                    val data =
                        when (val tryResult = this.alloc<UByte>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(data)
                }
                DType.u32 -> {
                    val data1 =
                        when (val tryResult = this.alloc<UInt>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(data1)
                }
                DType.i16 -> {
                    val data2 =
                        when (val tryResult = this.alloc<Short>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(data2)
                }
                DType.i32 -> {
                    val data3 =
                        when (val tryResult = this.alloc<Int>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(data3)
                }
                DType.i64 -> {
                    val data4 =
                        when (val tryResult = this.alloc<Long>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(data4)
                }
                DType.bf16 -> {
                    val data5 =
                        when (val tryResult = this.alloc<Bf16>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(data5)
                }
                DType.f16 -> {
                    val data6 =
                        when (val tryResult = this.alloc<F16>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(data6)
                }
                DType.f32 -> {
                    val data7 =
                        when (val tryResult = this.alloc<Float>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(data7)
                }
                DType.f64 -> {
                    val data8 =
                        when (val tryResult = this.alloc<Double>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(data8)
                }
                DType.f8e4m3 -> {
                    val data9 =
                        when (val tryResult = this.alloc<F8e4m3>(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(data9)
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Err(CudaError.InternalError("Dummy types not supported in CUDA backend").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun <T : WithDType> storageFromSlice(s: List<T>, cpuStorageRefTFn: (Any?) -> Any?): Result<CudaStorage> {
        val slice =
            when (val matchVal = cpuStorageRefTFn(s)) {
                is CpuStorageRef.U8 -> {
                    val (storage) = matchVal
                    val data =
                        when (val tryResult = this.cloneHtod(storage)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorageRef.U32 -> {
                    val (storage1) = matchVal
                    val data1 =
                        when (val tryResult = this.cloneHtod(storage1)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorageRef.I16 -> {
                    val (storage2) = matchVal
                    val data2 =
                        when (val tryResult = this.cloneHtod(storage2)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorageRef.I32 -> {
                    val (storage3) = matchVal
                    val data3 =
                        when (val tryResult = this.cloneHtod(storage3)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorageRef.I64 -> {
                    val (storage4) = matchVal
                    val data4 =
                        when (val tryResult = this.cloneHtod(storage4)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorageRef.Bf16 -> {
                    val (storage5) = matchVal
                    val data5 =
                        when (val tryResult = this.cloneHtod(storage5)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorageRef.F16 -> {
                    val (storage6) = matchVal
                    val data6 =
                        when (val tryResult = this.cloneHtod(storage6)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorageRef.F32 -> {
                    val (storage7) = matchVal
                    val data7 =
                        when (val tryResult = this.cloneHtod(storage7)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorageRef.F64 -> {
                    val (storage8) = matchVal
                    val data8 =
                        when (val tryResult = this.cloneHtod(storage8)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorageRef.F8e4m3 -> {
                    val (storage9) = matchVal
                    val data9 =
                        when (val tryResult = this.cloneHtod(storage9)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorageRef.F4 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F6e2m3 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F6e3m2 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
                is CpuStorageRef.F8e8m0 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = T.DTYPE, op = "storage_from_slice").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun storageFromCpuStorage(storage: CpuStorage): Result<CudaStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val (storage1) = storage
                    val data =
                        when (val tryResult = this.cloneHtod(storage1)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorage.U32 -> {
                    val (storage2) = storage
                    val data1 =
                        when (val tryResult = this.cloneHtod(storage2)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorage.I16 -> {
                    val (storage3) = storage
                    val data2 =
                        when (val tryResult = this.cloneHtod(storage3)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorage.I32 -> {
                    val (storage4) = storage
                    val data3 =
                        when (val tryResult = this.cloneHtod(storage4)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorage.I64 -> {
                    val (storage5) = storage
                    val data4 =
                        when (val tryResult = this.cloneHtod(storage5)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorage.Bf16 -> {
                    val (storage6) = storage
                    val data5 =
                        when (val tryResult = this.cloneHtod(storage6)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorage.F16 -> {
                    val (storage7) = storage
                    val data6 =
                        when (val tryResult = this.cloneHtod(storage7)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorage.F32 -> {
                    val (storage8) = storage
                    val data7 =
                        when (val tryResult = this.cloneHtod(storage8)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorage.F64 -> {
                    val (storage9) = storage
                    val data8 =
                        when (val tryResult = this.cloneHtod(storage9)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorage.F8e4m3 -> {
                    val (storage10) = storage
                    val data9 =
                        when (val tryResult = this.cloneHtod(storage10)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorage.F4 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F6e2m3 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F6e3m2 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
                is CpuStorage.F8e8m0 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun storageFromCpuStorageOwned(storage: CpuStorage): Result<CudaStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val (storage1) = storage
                    val data =
                        when (val tryResult = this.cloneHtod(storage1)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(data)
                }
                is CpuStorage.U32 -> {
                    val (storage2) = storage
                    val data1 =
                        when (val tryResult = this.cloneHtod(storage2)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(data1)
                }
                is CpuStorage.I16 -> {
                    val (storage3) = storage
                    val data2 =
                        when (val tryResult = this.cloneHtod(storage3)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(data2)
                }
                is CpuStorage.I32 -> {
                    val (storage4) = storage
                    val data3 =
                        when (val tryResult = this.cloneHtod(storage4)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(data3)
                }
                is CpuStorage.I64 -> {
                    val (storage5) = storage
                    val data4 =
                        when (val tryResult = this.cloneHtod(storage5)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(data4)
                }
                is CpuStorage.Bf16 -> {
                    val (storage6) = storage
                    val data5 =
                        when (val tryResult = this.cloneHtod(storage6)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(data5)
                }
                is CpuStorage.F16 -> {
                    val (storage7) = storage
                    val data6 =
                        when (val tryResult = this.cloneHtod(storage7)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(data6)
                }
                is CpuStorage.F32 -> {
                    val (storage8) = storage
                    val data7 =
                        when (val tryResult = this.cloneHtod(storage8)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(data7)
                }
                is CpuStorage.F64 -> {
                    val (storage9) = storage
                    val data8 =
                        when (val tryResult = this.cloneHtod(storage9)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(data8)
                }
                is CpuStorage.F8e4m3 -> {
                    val (storage10) = storage
                    val data9 =
                        when (val tryResult = this.cloneHtod(storage10)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(data9)
                }
                is CpuStorage.F4 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F6e2m3 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F6e3m2 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
                is CpuStorage.F8e8m0 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = storage.dtype(), op = "storage_from_cpu_storage_owned").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = this.clone()))
    }

    public fun synchronize(): Result<Unit> {
        when (val tryResult = this.stream.synchronize().mapErr(Error::wrap)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }

    public companion object {
        public operator fun invoke(
            id: DeviceId = DeviceId(),
            context: cudarc.CudaContext = cudarc.CudaContext(),
            modules: RwLock<ModuleStore> = RwLock<ModuleStore>(),
            customModules: RwLock<Map<String, cudarc.CudaModule>> = RwLock<Map<String, cudarc.CudaModule>>(),
            stream: cudarc.CudaStream = cudarc.CudaStream(),
            blas: cudarc.CudaBlas = cudarc.CudaBlas(),
            curand: Mutex<CudaRng> = Mutex<CudaRng>(),
            seedValue: RwLock<ULong> = RwLock<ULong>(),
        ): CudaDevice =
            CudaDevice(
                id = id,
                context = context,
                modulesCtorArg = modules,
                customModulesCtorArg = customModules,
                stream = stream,
                blas = blas,
                curandCtorArg = curand,
                seedValueCtorArg = seedValue,
            )

        public fun newWithStream(ordinal: Int): Result<CudaDevice> {
            val context =
                when (val tryResult = cudarc.CudaContext.new(ordinal).w()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val stream =
                when (val tryResult = context.newStream().w()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return CudaDevice.fromContextAndStream(context, stream)
        }

        public fun fromContextAndStream(context: cudarc.CudaContext, stream: cudarc.CudaStream): Result<CudaDevice> {
            val blas =
                when (val tryResult = cudarc.CudaBlas.new(stream.clone()).w()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val curand =
                when (val tryResult = cudarc.CudaRng.new(299792458, stream.clone()).w()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val moduleStore = ModuleStore(mdls = List(kernels.ALL_IDS.len()) { null })
            return Ok(CudaDevice(id = DeviceId.new(), context = context, stream = stream, blas = blas, curand = Mutex.new(CudaRng(curand)), modules = RwLock.new(moduleStore), customModules = RwLock.new(HashMap.new()), seedValue = RwLock.new(299792458)))
        }

        public fun new(ordinal: Int): Result<CudaDevice> {
            val context =
                when (val tryResult = cudarc.CudaContext.new(ordinal).w()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val stream = context.perThreadStream()
            return CudaDevice.fromContextAndStream(context, stream)
        }
    }
}

internal fun cudaGraphHtodCacheEnabled(): Boolean = CUDA_GRAPH_HTOD_CACHE_DEPTH.with({ depth -> depth.get() > 0 })

public data class CudaFunc(
    public var func: CudaFunction = CudaFunction(),
    public var stream: cudarc.CudaStream = cudarc.CudaStream(),
) {
    public fun deref(): CudaFunction = this.func

    public fun intoCudaFunction(): CudaFunction = this.func

    public fun builder(): cudarc.LaunchArgs = this.stream.launchBuilder(this.func)
}

private const val BUILDER_ARG_TEMPLATE: String = "\$(val arg= \$arg; \$b.arg(& arg); )*"

public fun builderArg(b: String = "", arg: String = ""): String =
    arrayOf("\$b" to b, "\$arg" to arg).fold(BUILDER_ARG_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
