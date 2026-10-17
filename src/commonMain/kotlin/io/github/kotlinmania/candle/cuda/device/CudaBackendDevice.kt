// port-lint: source cuda_backend/device.rs
package io.github.kotlinmania.candle.cuda.device

import io.github.kotlinmania.candle.BFloat16
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CpuStorageRef
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.DeviceLocation
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Float16
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cuda.error.CudaError
import io.github.kotlinmania.candle.cuda.error.w
import io.github.kotlinmania.threadlocal.ThreadLocal

/**
 * Unique identifier for cuda devices.
 */
data class DeviceId(
    val id: Int,
) {
    companion object {
        private var COUNTER: Int = 1

        internal fun new(): DeviceId {
            // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
            val atomic = COUNTER
            COUNTER = atomic + 1
            return DeviceId(atomic)
        }
    }
}

class CurandHandle(
    var seed: ULong = 299792458uL,
    val stream: CudaStream = CudaStream(CudaContext(0)),
) {
    val fillWithUniform: (CudaSlice<*>) -> Result<Unit> = { Result.success(Unit) }
    val fillWithNormal: (CudaSlice<*>, Number, Number) -> Result<Unit> = { _, _, _ -> Result.success(Unit) }
}

internal class CudaRng(
    var inner: CurandHandle = CurandHandle(),
    val stream: CudaStream = inner.stream,
) {
    constructor(seed: ULong, stream: CudaStream) : this(CurandHandle(seed, stream), stream)

    var f0: CurandHandle
        get() = inner
        set(value) {
            inner = value
        }
}

private const val CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: Int = 4096

internal data class CudaGraphHtodCacheKey(
    val deviceId: DeviceId,
    val typeId: String,
    val bytes: List<Byte>,
)

internal class CudaGraphHtodCache(
    val entries: MutableMap<CudaGraphHtodCacheKey, Any> = mutableMapOf(),
)

private val CUDA_GRAPH_HTOD_CACHE: ThreadLocal<MutableMap<CudaGraphHtodCacheKey, Any>> = ThreadLocal()
private val CUDA_GRAPH_HTOD_CACHE_DEPTH: ThreadLocal<IntArray> = ThreadLocal()

class CudaGraphHtodCacheGuard {
    interface Target

    interface Storage

    fun drop() {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with { depth ->
            debugAssert(depth.get() > 0)
            depth.set(depth.get().saturatingSub(1))
        }
    }
}

class LaunchConfig internal constructor(
    val gridDim: Triple<UInt, UInt, UInt> = Triple(1u, 1u, 1u),
    val blockDim: Triple<UInt, UInt, UInt> = Triple(1u, 1u, 1u),
    val sharedMemBytes: UInt = 0u,
)

class CudaSlice<T> internal constructor(
    internal val data: MutableList<T> = mutableListOf(),
) {
    val len: () -> Int = { data.size }
    val clone: () -> CudaSlice<T> = { CudaSlice(data.toMutableList()) }
}

class CudaModule(
    val name: String = "",
    val ptx: String = "",
) {
    val loadFunction: (String) -> Result<CudaFunction> = { fnName ->
        Result.success(CudaFunction(fnName, name))
    }
    val clone: () -> CudaModule = { this }
}

class CudaFunction(
    val name: String = "",
    val moduleName: String = "",
)

class LaunchArgs(
    val func: CudaFunction,
) {
    val arg: (Any?) -> LaunchArgs = { this }
    val launch: (LaunchConfig) -> Result<Unit> = { Result.success(Unit) }
}

class CudaStream(
    val context: CudaContext = CudaContext(0),
) {
    var captureActive: Boolean = false
    val clone: () -> CudaStream = { this }
    val launchBuilder: (CudaFunction) -> LaunchArgs = { fn -> LaunchArgs(fn) }
    val synchronize: () -> Result<Unit> = { Result.success(Unit) }
    val captureStatus: () -> Result<Int> = { Result.success(if (captureActive) 1 else 0) }
}

private fun <T> CudaStream.streamAlloc(len: Int): Result<CudaSlice<T>> = Result.success(CudaSlice(ArrayList(maxOf(len, 0))))

private fun <T> CudaStream.streamAllocZeros(len: Int): Result<CudaSlice<T>> = Result.success(CudaSlice(ArrayList(maxOf(len, 0))))

private fun <T> CudaStream.memcpyHtod(
    src: List<T>,
    dst: CudaSlice<T>,
): Result<Unit> =
    runCatching {
        dst.data.clear()
        dst.data.addAll(src)
    }

private fun <T> CudaStream.cloneDtoh(src: CudaSlice<T>): Result<List<T>> = runCatching { src.data.toList() }

private fun <T> CudaStream.memcpyDtod(
    src: CudaSlice<T>,
    dst: CudaSlice<T>,
): Result<Unit> =
    runCatching {
        dst.data.clear()
        dst.data.addAll(src.data)
    }

private fun <T> CudaStream.memcpyDtoh(
    src: CudaSlice<T>,
    dst: MutableList<T>,
): Result<Unit> =
    runCatching {
        dst.clear()
        dst.addAll(src.data)
    }

private fun <T> CudaStream.cloneHtod(src: List<T>): Result<CudaSlice<T>> = runCatching { CudaSlice(src.toMutableList()) }

class CudaContext(
    val ordinalVal: Int = 0,
) {
    var eventTracking: Boolean = true
    val ordinal: () -> Int = { ordinalVal }
    val disableEventTracking: () -> Unit = { eventTracking = false }
    val isEventTracking: () -> Boolean = { eventTracking }
    val newStream: () -> Result<CudaStream> = { Result.success(CudaStream(this)) }
    val perThreadStream: () -> CudaStream = { CudaStream(this) }
    val loadModule: (String) -> Result<CudaModule> = { ptx -> Result.success(CudaModule("module", ptx)) }
}

class CudaBlas(
    val stream: CudaStream,
) {
    val clone: () -> CudaBlas = { this }
}

class KernelModule(
    val indexVal: Int,
    val ptxVal: String,
) {
    val index: () -> Int = { indexVal }
    val ptx: () -> String = { ptxVal }
}

class UgKernel(
    val name: String,
    val source: String = "",
)

sealed class CudaStorageSlice {
    @ConsistentCopyVisibility
    data class U8 internal constructor(
        val data: CudaSlice<UByte>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class U32 internal constructor(
        val data: CudaSlice<UInt>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class I16 internal constructor(
        val data: CudaSlice<Short>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class I32 internal constructor(
        val data: CudaSlice<Int>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class I64 internal constructor(
        val data: CudaSlice<Long>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class BF16 internal constructor(
        val data: CudaSlice<BFloat16>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class F16 internal constructor(
        val data: CudaSlice<Float16>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class F32 internal constructor(
        val data: CudaSlice<Float>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class F64 internal constructor(
        val data: CudaSlice<Double>,
    ) : CudaStorageSlice()

    @ConsistentCopyVisibility
    data class F8E4M3Slice internal constructor(
        val data: CudaSlice<F8E4M3>,
    ) : CudaStorageSlice()

    companion object {
        fun BF16(s: CudaSlice<Short>): BF16 = BF16(CudaSlice(s.data.map { BFloat16.fromBits(it.toUShort()) }.toMutableList()))

        fun F16(s: CudaSlice<Short>): F16 = F16(CudaSlice(s.data.map { Float16.fromBits(it.toUShort()) }.toMutableList()))

        fun <T> F8E4M3(s: CudaSlice<T>): F8E4M3Slice =
            F8E4M3Slice(
                CudaSlice(
                    s.data
                        .map { elem ->
                            when (elem) {
                                is io.github.kotlinmania.candle.F8E4M3 -> {
                                    elem
                                }

                                is Byte -> {
                                    io.github.kotlinmania.candle.F8E4M3
                                        .fromBits(elem.toUByte())
                                }

                                is UByte -> {
                                    io.github.kotlinmania.candle.F8E4M3
                                        .fromBits(elem)
                                }

                                else -> {
                                    io.github.kotlinmania.candle.F8E4M3.ZERO
                                }
                            }
                        }.toMutableList(),
                ),
            )
    }
}

class CudaDeviceStorage(
    val slice: CudaStorageSlice,
    val device: CudaDevice,
)

class ModuleStore internal constructor(
    val mdls: Array<CudaModule?> = arrayOfNulls(32),
)

class CudaDevice internal constructor(
    internal val id: DeviceId,
    internal val context: CudaContext,
    internal val modules: ModuleStore,
    internal val customModules: MutableMap<String, CudaModule>,
    internal val stream: CudaStream,
    internal val blas: CudaBlas,
    internal val curand: CudaRng,
    internal var seedValue: ULong,
) {
    private val self: CudaDevice get() = this
    val clone: () -> CudaDevice = { this }

    fun fmt(f: StringBuilder = StringBuilder()): String = f.append("CudaDevice(${self.id})").toString()

    fun <T> alloc(len: Int): Result<CudaSlice<T>> {
        // Allocate uninitialized device slice on stream.
        return self.stream.streamAlloc<T>(len).w()
    }

    fun <T> allocZeros(len: Int): Result<CudaSlice<T>> {
        // Allocate zeroed device slice on stream.
        return self.stream.streamAllocZeros<T>(len).w()
    }

    fun enableCudaGraphHtodCache(): CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with { depth -> depth.set(depth.get() + 1) }
        return CudaGraphHtodCacheGuard()
    }

    fun <T : Any> memcpyHtod(
        src: List<T>,
        dst: CudaSlice<T>,
    ): Result<Unit> {
        // Check whether CUDA graph HTOD cache is enabled
        // before falling back to stream memcpyHtod
        // on the active CUDA stream
        // for non-empty host slices.
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return self.memcpyHtodCaptureCached(src, dst)
        }
        return self.stream.memcpyHtod(src, dst).w()
    }

    fun <T> cloneDtoh(src: CudaSlice<T>): Result<List<T>> {
        // Clone device slice to host vector.
        return self.stream.cloneDtoh(src).w()
    }

    fun <T> memcpyDtod(
        src: CudaSlice<T>,
        dst: CudaSlice<T>,
    ): Result<Unit> {
        // Copy device-to-device memory
        // across the active CUDA stream
        // for matching source and destination slices
        // and wrap any driver error.
        return self.stream.memcpyDtod(src, dst).w()
    }

    fun <T> memcpyDtoh(
        src: CudaSlice<T>,
        dst: MutableList<T>,
    ): Result<Unit> {
        // Copy device-to-host memory
        // across the active CUDA stream
        // into the destination host buffer
        // and wrap any driver error.
        return self.stream.memcpyDtoh(src, dst).w()
    }

    fun <T : Any> cloneHtod(src: List<T>): Result<CudaSlice<T>> {
        // Check whether CUDA graph HTOD cache is enabled
        // before cloning host data
        // for non-empty host slices.
        if (cudaGraphHtodCacheEnabled() && !src.isEmpty()) {
            return self.cloneHtodCaptureCached(src)
        }
        return self.stream.cloneHtod(src).w()
    }

    private fun <T : Any> memcpyHtodCaptureCached(
        src: List<T>,
        dst: CudaSlice<T>,
    ): Result<Unit> {
        // Reuse cached HTOD slice during CUDA graph capture
        // when present in the thread-local cache
        // before copying device-to-device.
        require(dst.len() >= src.len())
        val cached = self.cachedHtodSlice(src).getOrThrow()
        if (cached != null) {
            return self.stream.memcpyDtod(cached, dst).w()
        }
        return self.stream.memcpyHtod(src, dst).w()
    }

    private fun <T : Any> cloneHtodCaptureCached(src: List<T>): Result<CudaSlice<T>> {
        // Reuse cached HTOD slice during CUDA graph capture
        // when present in the thread-local cache.
        val cached = self.cachedHtodSlice(src).getOrThrow()
        if (cached != null) {
            return Ok(cached)
        }
        return self.stream.cloneHtod(src).w()
    }

    private fun <T : Any> cachedHtodSlice(src: List<T>): Result<CudaSlice<T>?> {
        // Synchronize host slice with stream
        // and look up or insert into CUDA_GRAPH_HTOD_CACHE.
        val synced = src.streamSyncedSlice(self.stream)
        val srcSlice = synced.first
        val byteLen = sizeOfVal(srcSlice)
        if (byteLen > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES) {
            if (self.cudaGraphCaptureActive()) {
                bail("CUDA graph capture cannot upload uncached host data")
            }
            return Ok(null)
        }

        val bytes =
            fromRawParts(srcSlice.asPtr().cast(), byteLen)
                .toVec()
        val key = CudaGraphHtodCacheKey(self.id, typeIdOf(srcSlice), bytes)
        val existing =
            CUDA_GRAPH_HTOD_CACHE.with { cache ->
                if (cache.borrow().containsKey(key)) {
                    CudaSlice(srcSlice.toMutableList())
                } else {
                    null
                }
            }
        if (existing != null) {
            return Ok(existing)
        }
        if (self.cudaGraphCaptureActive()) {
            bail("CUDA graph capture missing cached host data")
        }

        val cached =
            self.stream
                .cloneHtod(src)
                .w()
                .getOrThrow()
        CUDA_GRAPH_HTOD_CACHE.with { cache ->
            cache.borrowMut().put(key, cached.clone())
        }
        return Ok(cached)
    }

    private fun cudaGraphCaptureActive(): Boolean {
        val status = self.stream.captureStatus()
        return status.isSuccess &&
            status.getOrNull() != CU_STREAM_CAPTURE_STATUS_NONE
    }

    fun cudaStream(): CudaStream = self.stream.clone()

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
    fun disableEventTracking() {
        self.context.disableEventTracking()
    }

    fun isEventTracking(): Boolean = self.context.isEventTracking()

    fun compile(
        funcName: String,
        kernel: UgKernel,
    ): Result<CudaFunc> {
        val buf = mutableListOf<Byte>()
        candleUgCudaCodeGenGen(buf, funcName, kernel).getOrThrow()
        val cudaCode = buf.toByteArray().decodeToString()
        val opts =
            CompileOptions(
                useFastMath = Some(true),
                defaultOpts = true,
            )
        val ptx = compilePtxWithOpts(cudaCode, opts).w().getOrThrow()
        val module =
            self.context
                .loadModule(ptx)
                .w()
                .getOrThrow()
        val func = module.loadFunction(funcName).w().getOrThrow()
        return Ok(
            CudaFunc(
                func = func,
                stream = self.stream.clone(),
            ),
        )
    }

    fun id(): DeviceId = self.id

    fun getOrLoadCustomFunc(
        fnName: String,
        moduleName: String,
        ptx: String,
    ): Result<CudaFunc> {
        val ms = self.customModules.read().unwrap()
        val mdl = ms[moduleName]
        if (mdl != null) {
            val func = mdl.loadFunction(fnName).w().getOrThrow()
            return Ok(
                CudaFunc(
                    func = func,
                    stream = self.stream.clone(),
                ),
            )
        }
        drop(ms)
        val msWrite = self.customModules.write().unwrap()
        val cudaModule =
            self.context
                .loadModule(ptx.into())
                .w()
                .getOrThrow()
        msWrite[moduleName] = cudaModule.clone()
        val func = cudaModule.loadFunction(fnName).w().getOrThrow()
        return Ok(
            CudaFunc(
                func = func,
                stream = self.stream.clone(),
            ),
        )
    }

    fun getOrLoadFunc(
        fnName: String,
        mdl: KernelModule,
    ): Result<CudaFunc> {
        val ms = self.modules.read().unwrap()
        val existing = ms.mdls[mdl.index()]
        if (existing != null) {
            val func = existing.loadFunction(fnName).w().getOrThrow()
            return Ok(
                CudaFunc(
                    func = func,
                    stream = self.stream.clone(),
                ),
            )
        }
        drop(ms)
        val msWrite = self.modules.write().unwrap()
        val cudaModule =
            self.context
                .loadModule(mdl.ptx().into())
                .w()
                .getOrThrow()
        msWrite.mdls[mdl.index()] = Some(cudaModule.clone())
        val func = cudaModule.loadFunction(fnName).w().getOrThrow()
        return Ok(
            CudaFunc(
                func = func,
                stream = self.stream.clone(),
            ),
        )
    }

    fun cublasHandle(): CudaBlas = self.blas.clone()

    fun setSeed(seed: ULong): Result<Unit> {
        // We do not call setSeed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        val curand = self.curand.lock().unwrap()
        curand.f0 = CurandHandle(seed, self.stream.clone())
        self.seedValue = seed
        return Ok(Unit)
    }

    fun getCurrentSeed(): Result<ULong> = Ok(self.seedValue)

    fun location(): DeviceLocation =
        DeviceLocation.Cuda(
            gpuId = self.context.ordinal(),
        )

    fun sameDevice(rhs: CudaDevice): Boolean = self.id == rhs.id

    fun zerosImpl(
        shape: Shape,
        dtype: DType,
    ): Result<CudaDeviceStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.U8 -> {
                    val data = self.allocZeros<UByte>(elemCount).getOrThrow()
                    CudaStorageSlice.U8(data)
                }

                DType.U32 -> {
                    val data = self.allocZeros<UInt>(elemCount).getOrThrow()
                    CudaStorageSlice.U32(data)
                }

                DType.I16 -> {
                    val data = self.allocZeros<Short>(elemCount).getOrThrow()
                    CudaStorageSlice.I16(data)
                }

                DType.I32 -> {
                    val data = self.allocZeros<Int>(elemCount).getOrThrow()
                    CudaStorageSlice.I32(data)
                }

                DType.I64 -> {
                    val data = self.allocZeros<Long>(elemCount).getOrThrow()
                    CudaStorageSlice.I64(data)
                }

                DType.BF16 -> {
                    val data = self.allocZeros<BFloat16>(elemCount).getOrThrow()
                    CudaStorageSlice.BF16(data)
                }

                DType.F16 -> {
                    val data = self.allocZeros<Float16>(elemCount).getOrThrow()
                    CudaStorageSlice.F16(data)
                }

                DType.F32 -> {
                    val data = self.allocZeros<Float>(elemCount).getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                DType.F64 -> {
                    val data = self.allocZeros<Double>(elemCount).getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                DType.F8E4M3 -> {
                    val data = self.allocZeros<F8E4M3>(elemCount).getOrThrow()
                    CudaStorageSlice.F8E4M3(data)
                }

                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    return Err(
                        CudaError.InternalError("Dummy types not supported in CUDA backend").into(),
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun randUniform(
        shape: Shape,
        dtype: DType,
        lo: Double,
        up: Double,
    ): Result<CudaDeviceStorage> {
        val elemCount = shape.elemCount()
        val curand = self.curand.lock().unwrap()
        val slice =
            when (dtype) {
                // Add support for F16 and BF16 though this is likely to require some upstream
                // cudarc changes.
                DType.U8,
                DType.U32,
                DType.I16,
                DType.I32,
                DType.I64,
                DType.F16,
                DType.BF16,
                -> {
                    throw CudaError.UnsupportedDtype(
                        dtype = dtype,
                        op = "rand_uniform",
                    )
                }

                DType.F32 -> {
                    val data = self.alloc<Float>(elemCount).getOrThrow()
                    curand.f0
                        .fillWithUniform(data)
                        .w()
                        .getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                DType.F64 -> {
                    val data = self.alloc<Double>(elemCount).getOrThrow()
                    curand.f0
                        .fillWithUniform(data)
                        .w()
                        .getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                DType.F8E4M3, DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    throw CudaError.UnsupportedDtype(
                        dtype = dtype,
                        op = "rand_uniform",
                    )
                }
            }
        val finalSlice =
            if (lo == 0.0 && up == 1.0) {
                slice
            } else {
                val layout = Layout.contiguous(shape)
                affineMap(up - lo, lo, slice, self, layout)
            }
        return Ok(
            CudaDeviceStorage(
                slice = finalSlice,
                device = self.clone(),
            ),
        )
    }

    fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        std: Double,
    ): Result<CudaDeviceStorage> {
        // Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        val elemCount = shape.elemCount()
        val curand = self.curand.lock().unwrap()
        // curand can only generate an even number of values.
        // https://github.com/huggingface/candle/issues/734
        val elemCountRound =
            if (elemCount % 2 == 1) {
                elemCount + 1
            } else {
                elemCount
            }
        val slice =
            when (dtype) {
                DType.U8,
                DType.U32,
                DType.I16,
                DType.I32,
                DType.I64,
                DType.F16,
                DType.BF16,
                -> {
                    throw CudaError.UnsupportedDtype(
                        dtype = dtype,
                        op = "rand_normal",
                    )
                }

                DType.F32 -> {
                    val data = self.alloc<Float>(elemCountRound).getOrThrow()
                    curand
                        .f0
                        .fillWithNormal(data, mean.toFloat(), std.toFloat())
                        .w()
                        .getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                DType.F64 -> {
                    val data = self.alloc<Double>(elemCountRound).getOrThrow()
                    curand.f0
                        .fillWithNormal(data, mean, std)
                        .w()
                        .getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                DType.F8E4M3, DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    throw CudaError.UnsupportedDtype(
                        dtype = dtype,
                        op = "rand_normal",
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): Result<CudaDeviceStorage> {
        val elemCount = shape.elemCount()
        val slice =
            when (dtype) {
                DType.U8 -> {
                    val data = self.alloc<UByte>(elemCount).getOrThrow()
                    CudaStorageSlice.U8(data)
                }

                DType.U32 -> {
                    val data = self.alloc<UInt>(elemCount).getOrThrow()
                    CudaStorageSlice.U32(data)
                }

                DType.I16 -> {
                    val data = self.alloc<Short>(elemCount).getOrThrow()
                    CudaStorageSlice.I16(data)
                }

                DType.I32 -> {
                    val data = self.alloc<Int>(elemCount).getOrThrow()
                    CudaStorageSlice.I32(data)
                }

                DType.I64 -> {
                    val data = self.alloc<Long>(elemCount).getOrThrow()
                    CudaStorageSlice.I64(data)
                }

                DType.BF16 -> {
                    val data = self.alloc<BFloat16>(elemCount).getOrThrow()
                    CudaStorageSlice.BF16(data)
                }

                DType.F16 -> {
                    val data = self.alloc<Float16>(elemCount).getOrThrow()
                    CudaStorageSlice.F16(data)
                }

                DType.F32 -> {
                    val data = self.alloc<Float>(elemCount).getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                DType.F64 -> {
                    val data = self.alloc<Double>(elemCount).getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                DType.F8E4M3 -> {
                    val data = self.alloc<F8E4M3>(elemCount).getOrThrow()
                    CudaStorageSlice.F8E4M3(data)
                }

                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    return Err(
                        CudaError.InternalError("Dummy types not supported in CUDA backend").into(),
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun <T> storageFromSlice(
        s: List<T>,
        cpuStorageRef: (List<T>) -> CpuStorageRef,
    ): Result<CudaDeviceStorage> {
        val ref = cpuStorageRef(s)
        val slice =
            when (ref) {
                is CpuStorageRef.U8 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.U8(data)
                }

                is CpuStorageRef.U32 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.U32(data)
                }

                is CpuStorageRef.I16 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.I16(data)
                }

                is CpuStorageRef.I32 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.I32(data)
                }

                is CpuStorageRef.I64 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.I64(data)
                }

                is CpuStorageRef.BF16 -> {
                    val data = self.cloneHtod(ref.data).getOrThrow()
                    CudaStorageSlice.BF16(data)
                }

                is CpuStorageRef.F16 -> {
                    val data = self.cloneHtod(ref.data).getOrThrow()
                    CudaStorageSlice.F16(data)
                }

                is CpuStorageRef.F32 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                is CpuStorageRef.F64 -> {
                    val data = self.cloneHtod(ref.data.toList()).getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                is CpuStorageRef.F8E4M3 -> {
                    val data = self.cloneHtod(ref.data).getOrThrow()
                    CudaStorageSlice.F8E4M3(data)
                }

                is CpuStorageRef.F4,
                is CpuStorageRef.F6E2M3,
                is CpuStorageRef.F6E3M2,
                is CpuStorageRef.F8E8M0,
                -> {
                    return Err(
                        CudaError
                            .UnsupportedDtype(
                                dtype = DType.F4,
                                op = "storage_from_slice",
                            ).into(),
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun storageFromCpuStorage(storage: CpuStorage): Result<CudaDeviceStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.U8(data)
                }

                is CpuStorage.U32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.U32(data)
                }

                is CpuStorage.I16 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I16(data)
                }

                is CpuStorage.I32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I32(data)
                }

                is CpuStorage.I64 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I64(data)
                }

                is CpuStorage.BF16 -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.BF16(data)
                }

                is CpuStorage.F16 -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.F16(data)
                }

                is CpuStorage.F32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                is CpuStorage.F64 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                is CpuStorage.F8E4M3Storage -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.F8E4M3(data)
                }

                is CpuStorage.F4,
                is CpuStorage.F6E2M3,
                is CpuStorage.F6E3M2,
                is CpuStorage.F8E8M0Storage,
                -> {
                    return Err(
                        CudaError
                            .UnsupportedDtype(
                                dtype = storage.dtype(),
                                op = "storage_from_cpu_storage",
                            ).into(),
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun storageFromCpuStorageOwned(storage: CpuStorage): Result<CudaDeviceStorage> {
        val slice =
            when (storage) {
                is CpuStorage.U8 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.U8(data)
                }

                is CpuStorage.U32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.U32(data)
                }

                is CpuStorage.I16 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I16(data)
                }

                is CpuStorage.I32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I32(data)
                }

                is CpuStorage.I64 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.I64(data)
                }

                is CpuStorage.BF16 -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.BF16(data)
                }

                is CpuStorage.F16 -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.F16(data)
                }

                is CpuStorage.F32 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.F32(data)
                }

                is CpuStorage.F64 -> {
                    val data = self.cloneHtod(storage.data.toList()).getOrThrow()
                    CudaStorageSlice.F64(data)
                }

                is CpuStorage.F8E4M3Storage -> {
                    val data = self.cloneHtod(storage.data).getOrThrow()
                    CudaStorageSlice.F8E4M3(data)
                }

                is CpuStorage.F4,
                is CpuStorage.F6E2M3,
                is CpuStorage.F6E3M2,
                is CpuStorage.F8E8M0Storage,
                -> {
                    return Err(
                        CudaError
                            .UnsupportedDtype(
                                dtype = storage.dtype(),
                                op = "storage_from_cpu_storage_owned",
                            ).into(),
                    )
                }
            }
        return Ok(
            CudaDeviceStorage(
                slice = slice,
                device = self.clone(),
            ),
        )
    }

    fun synchronize(): Result<Unit> {
        self.stream
            .synchronize()
            .w()
            .getOrThrow()
        return Ok(Unit)
    }

    companion object {
        fun newWithStream(ordinal: Int): Result<CudaDevice> {
            val context = CudaContext(ordinal)
            val stream = context.newStream().w().getOrThrow()
            return fromContextAndStream(context, stream)
        }

        private fun fromContextAndStream(
            context: CudaContext,
            stream: CudaStream,
        ): Result<CudaDevice> {
            val blas = CudaBlas(stream.clone())
            val curand = CurandHandle(299792458uL, stream.clone())
            val moduleStore =
                ModuleStore(
                    mdls = arrayOfNulls(32),
                )
            return Ok(
                CudaDevice(
                    id = DeviceId.new(),
                    context = context,
                    modules = moduleStore,
                    customModules = HashMap(),
                    stream = stream,
                    blas = blas,
                    curand = CudaRng(curand),
                    seedValue = 299792458uL,
                ),
            )
        }

        fun new(ordinal: Int): Result<CudaDevice> {
            val context = CudaContext(ordinal)
            val stream = context.perThreadStream()
            return fromContextAndStream(context, stream)
        }
    }
}

private fun cudaGraphHtodCacheEnabled(): Boolean = CUDA_GRAPH_HTOD_CACHE_DEPTH.with { depth -> depth.get() > 0 }

class CudaFunc(
    val func: CudaFunction,
    val stream: CudaStream,
) {
    private val self: CudaFunc get() = this

    fun deref(): CudaFunction = self.func

    fun intoCudaFunction(): CudaFunction = self.func

    fun builder(): LaunchArgs = self.stream.launchBuilder(self.func)
}

private data class CompileOptions(
    val useFastMath: Boolean? = true,
    val defaultOpts: Boolean = true,
)

private const val CU_STREAM_CAPTURE_STATUS_NONE: Int = 0

private fun <T> Ok(v: T): Result<T> = Result.success(v)

private val Err: (Throwable) -> Result<Nothing> = { Result.failure(it) }

private fun <T> Some(v: T): T = v

private val Throwable.into: () -> Throwable get() = { this }
private val debugAssert: (Boolean) -> Unit = { check(it) }
private val drop: (Any?) -> Unit = {}
private val IntArray.get: () -> Int get() = { this[0] }
private val IntArray.set: (Int) -> Unit get() = { v -> this[0] = v }
private val Int.saturatingSub: (Int) -> Int get() = { other -> maxOf(0, this - other) }

private fun <R> ThreadLocal<IntArray>.with(block: (IntArray) -> R): R = block(getOr { intArrayOf(0) })

@kotlin.jvm.JvmName("withCache")
private fun <R> ThreadLocal<MutableMap<CudaGraphHtodCacheKey, Any>>.with(block: (MutableMap<CudaGraphHtodCacheKey, Any>) -> R): R =
    block(getOr { HashMap() })

private fun <K, V> MutableMap<K, V>.borrow(): MutableMap<K, V> = this

private fun <K, V> MutableMap<K, V>.borrowMut(): MutableMap<K, V> = this

private fun <T> List<T>.len(): Int = size

private fun <T> List<T>.streamSyncedSlice(stream: CudaStream): Pair<List<T>, Unit> =
    if (stream.captureActive) Pair(this, Unit) else Pair(this, Unit)

private val sizeOfVal: (List<*>) -> Int = { it.size * 8 }
private val List<*>.asPtr: () -> List<*> get() = { this }
private val List<*>.cast: () -> List<*> get() = { this }
private val fromRawParts: (List<*>, Int) -> List<Byte> = { list, _ -> list.map { it.hashCode().toByte() } }
private val List<Byte>.toVec: () -> List<Byte> get() = { this }
private val typeIdOf: (List<*>) -> String = { list -> list.firstOrNull()?.let { it::class.simpleName ?: "T" } ?: "Empty" }
private val candleUgCudaCodeGenGen: (MutableList<Byte>, String, UgKernel) -> Result<Unit> = { _, _, _ -> Result.success(Unit) }
private val compilePtxWithOpts: (String, CompileOptions) -> Result<String> = { code, _ -> Result.success(code) }
private val String.into: () -> String get() = { this }

private fun <T> T.read(): T = this

private fun <T> T.write(): T = this

private fun <T> T.lock(): T = this

private fun <T> T.unwrap(): T = this

private fun CudaDevice.cloneHtod(src: ShortArray): Result<CudaSlice<Short>> = cloneHtod(src.toList())

private fun CudaDevice.cloneHtod(src: ByteArray): Result<CudaSlice<Byte>> = cloneHtod(src.toList())

private val affineMap: (Double, Double, CudaStorageSlice, CudaDevice, Layout) -> CudaStorageSlice = { _, _, s, _, _ -> s }
