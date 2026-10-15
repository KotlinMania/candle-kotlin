// port-lint: source cuda_backend/device.rs
package io.github.kotlinmania.candle.cuda.device

import io.github.kotlinmania.candle.BFloat16
import io.github.kotlinmania.candle.BackendDevice
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CpuStorageRef
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.DeviceLocation
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Float16
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.WithDType
import io.github.kotlinmania.candle.cuda.error.CudaError
import io.github.kotlinmania.candle.cuda.error.w
import io.github.kotlinmania.threadlocal.ThreadLocal

/**
 * Unique identifier for cuda devices.
 *
 * Used to distinguish CUDA device instances and key per-device caches.
 */
data class DeviceId(val id: Int) {
    companion object {
        private var counter: Int = 1

        /**
         * Allocates a fresh unique [DeviceId] using a monotonically increasing counter.
         */
        internal fun new(): DeviceId {
            val nextId = counter
            counter += 1
            return DeviceId(nextId)
        }
    }
}

internal class CudaRng(
    var seed: ULong,
    val stream: CudaStream
) {
    val fillWithUniformF32: (MutableList<Float>) -> Unit = { data ->
        var state = seed xor 0x9E3779B97F4A7C15uL
        for (i in data.indices) {
            state = state * 6364136223846793005uL + 1442695040888963407uL
            val bits = (state shr 40).toInt()
            data[i] = bits.toFloat() / (1 shl 24).toFloat()
        }
        seed = state
    }

    val fillWithUniformF64: (MutableList<Double>) -> Unit = { data ->
        var state = seed xor 0x9E3779B97F4A7C15uL
        for (i in data.indices) {
            state = state * 6364136223846793005uL + 1442695040888963407uL
            val bits = (state shr 11).toLong()
            data[i] = bits.toDouble() / (1L shl 53).toDouble()
        }
        seed = state
    }

    val fillWithNormalF32: (MutableList<Float>, Float, Float) -> Unit = { data, mean, std ->
        fillWithUniformF32(data)
        for (i in data.indices) {
            data[i] = mean + (data[i] - 0.5f) * std
        }
    }

    val fillWithNormalF64: (MutableList<Double>, Double, Double) -> Unit = { data, mean, std ->
        fillWithUniformF64(data)
        for (i in data.indices) {
            data[i] = mean + (data[i] - 0.5) * std
        }
    }
}

private const val CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: Int = 4096

internal data class CudaGraphHtodCacheKey(
    val deviceId: DeviceId,
    val typeName: String,
    val bytes: List<Byte>
)

internal class CudaGraphHtodCache(
    val entries: MutableMap<CudaGraphHtodCacheKey, Any> = mutableMapOf()
)

private val CUDA_GRAPH_HTOD_CACHE: ThreadLocal<MutableMap<CudaGraphHtodCacheKey, Any>> = ThreadLocal()
private val CUDA_GRAPH_HTOD_CACHE_DEPTH: ThreadLocal<IntArray> = ThreadLocal()

class CudaGraphHtodCacheGuard {
    interface Target
    interface Storage

    fun drop() {
        val depth = CUDA_GRAPH_HTOD_CACHE_DEPTH.getOr { intArrayOf(0) }
        check(depth[0] > 0)
        depth[0] = maxOf(0, depth[0] - 1)
    }
}

class CudaSlice<T>(val data: MutableList<T>)

class CudaModule(val name: String, val ptx: String) {
    val loadFunction: (String) -> Result<CudaFunction> = { fnName ->
        Result.success(CudaFunction(fnName, name))
    }
}

class CudaFunction(val name: String, val moduleName: String)

class LaunchArgs(val func: CudaFunction)

class CudaStream(val context: CudaContext) {
    var captureActive: Boolean = false
}

class CudaContext(val ordinal: Int) {
    var eventTracking: Boolean = true
}

class CudaBlas(val stream: CudaStream)

class KernelModule(val index: Int, val ptx: String)

class UgKernel(val name: String, val source: String = "")

sealed class CudaStorageSlice {
    data class U8(val data: CudaSlice<UByte>) : CudaStorageSlice()
    data class U32(val data: CudaSlice<UInt>) : CudaStorageSlice()
    data class I16(val data: CudaSlice<Short>) : CudaStorageSlice()
    data class I32(val data: CudaSlice<Int>) : CudaStorageSlice()
    data class I64(val data: CudaSlice<Long>) : CudaStorageSlice()
    data class BF16(val data: CudaSlice<BFloat16>) : CudaStorageSlice()
    data class F16(val data: CudaSlice<Float16>) : CudaStorageSlice()
    data class F32(val data: CudaSlice<Float>) : CudaStorageSlice()
    data class F64(val data: CudaSlice<Double>) : CudaStorageSlice()
    data class F8E4M3Slice(val data: CudaSlice<F8E4M3>) : CudaStorageSlice()
}

class CudaDeviceStorage(
    val slice: CudaStorageSlice,
    val device: CudaDevice
)

class ModuleStore(
    val mdls: Array<CudaModule?> = arrayOfNulls(32)
)

class CudaDevice(
    private val id: DeviceId,
    internal val context: CudaContext,
    private val modules: ModuleStore,
    private val customModules: MutableMap<String, CudaModule>,
    private val stream: CudaStream,
    internal val blas: CudaBlas,
    private val curand: CudaRng,
    private var seedValue: ULong
) {

    fun fmt(): String = "CudaDevice($id)"

    fun <T> alloc(len: Int): Result<CudaSlice<T>> = runCatching {
        val list = ArrayList<T>(maxOf(len, 0))
        CudaSlice(list)
    }.w()

    fun <T> allocZeros(len: Int): Result<CudaSlice<T>> = runCatching {
        val list = ArrayList<T>(maxOf(len, 0))
        CudaSlice(list)
    }.w()

    fun enableCudaGraphHtodCache(): CudaGraphHtodCacheGuard {
        val depth = CUDA_GRAPH_HTOD_CACHE_DEPTH.getOr { intArrayOf(0) }
        depth[0] += 1
        return CudaGraphHtodCacheGuard()
    }

    fun <T : Any> memcpyHtod(
        src: List<T>,
        dst: CudaSlice<T>
    ): Result<Unit> {
        if (cudaGraphHtodCacheEnabled() && src.isNotEmpty()) {
            return memcpyHtodCaptureCached(src, dst)
        }
        return runCatching {
            dst.data.clear()
            dst.data.addAll(src)
        }.w()
    }

    fun <T> cloneDtoh(
        src: CudaSlice<T>
    ): Result<List<T>> = runCatching {
        src.data.toList()
    }.w()

    fun <T> memcpyDtod(
        src: CudaSlice<T>,
        dst: CudaSlice<T>
    ): Result<Unit> = runCatching {
        dst.data.clear()
        dst.data.addAll(src.data)
    }.w()

    fun <T> memcpyDtoh(
        src: CudaSlice<T>,
        dst: MutableList<T>
    ): Result<Unit> = runCatching {
        dst.clear()
        dst.addAll(src.data)
    }.w()

    fun <T : Any> cloneHtod(
        src: List<T>
    ): Result<CudaSlice<T>> {
        if (cudaGraphHtodCacheEnabled() && src.isNotEmpty()) {
            return cloneHtodCaptureCached(src)
        }
        return runCatching {
            CudaSlice(src.toMutableList())
        }.w()
    }

    private fun <T : Any> memcpyHtodCaptureCached(
        src: List<T>,
        dst: CudaSlice<T>
    ): Result<Unit> = runCatching {
        require(dst.data.size >= src.size || dst.data.isEmpty())
        val cached = cachedHtodSlice(src).getOrThrow()
        if (cached != null) {
            return memcpyDtod(cached, dst)
        }
        dst.data.clear()
        dst.data.addAll(src)
    }.w()

    private fun <T : Any> cloneHtodCaptureCached(
        src: List<T>
    ): Result<CudaSlice<T>> = runCatching {
        val cached = cachedHtodSlice(src).getOrThrow()
        if (cached != null) {
            return@runCatching cached
        }
        CudaSlice(src.toMutableList())
    }.w()

    private fun <T : Any> cachedHtodSlice(
        src: List<T>
    ): Result<CudaSlice<T>?> = runCatching {
        val byteLen = src.size * 8
        if (byteLen > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES) {
            if (cudaGraphCaptureActive()) {
                throw IllegalStateException("CUDA graph capture cannot upload uncached host data")
            }
            return@runCatching null
        }

        val typeName = src.firstOrNull()?.let { it::class.simpleName ?: "Unknown" } ?: "Empty"
        val bytes = src.map { it.hashCode().toByte() }
        val key = CudaGraphHtodCacheKey(id, typeName, bytes)
        val cache = CUDA_GRAPH_HTOD_CACHE.getOr { HashMap() }
        val existing = cache[key]
        if (existing is CudaSlice<*>) {
            val typed = CudaSlice(existing.data.map { element -> element as T }.toMutableList())
            return@runCatching typed
        }
        if (cudaGraphCaptureActive()) {
            throw IllegalStateException("CUDA graph capture missing cached host data")
        }

        val cached = CudaSlice(src.toMutableList())
        cache[key] = cached
        cached
    }

    private fun cudaGraphCaptureActive(): Boolean {
        return stream.captureActive
    }

    fun cudaStream(): CudaStream {
        return stream
    }

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
        context.eventTracking = false
    }

    fun isEventTracking(): Boolean {
        return context.eventTracking
    }

    fun compile(
        funcName: String,
        kernel: UgKernel
    ): Result<CudaFunc> = runCatching {
        val cudaCode = kernel.source.ifEmpty { "// generated kernel $funcName" }
        val module = CudaModule(funcName, cudaCode)
        val func = module.loadFunction(funcName).w().getOrThrow()
        CudaFunc(
            func = func,
            stream = stream
        )
    }

    fun id(): DeviceId {
        return id
    }

    fun getOrLoadCustomFunc(
        fnName: String,
        moduleName: String,
        ptx: String
    ): Result<CudaFunc> = runCatching {
        val existing = customModules[moduleName]
        if (existing != null) {
            val func = existing.loadFunction(fnName).w().getOrThrow()
            return@runCatching CudaFunc(
                func = func,
                stream = stream
            )
        }
        val cudaModule = CudaModule(moduleName, ptx)
        customModules[moduleName] = cudaModule
        val func = cudaModule.loadFunction(fnName).w().getOrThrow()
        CudaFunc(
            func = func,
            stream = stream
        )
    }

    fun getOrLoadFunc(fnName: String, mdl: KernelModule): Result<CudaFunc> = runCatching {
        val existing = modules.mdls[mdl.index]
        if (existing != null) {
            val func = existing.loadFunction(fnName).w().getOrThrow()
            return@runCatching CudaFunc(
                func = func,
                stream = stream
            )
        }
        val cudaModule = CudaModule("module_${mdl.index}", mdl.ptx)
        modules.mdls[mdl.index] = cudaModule
        val func = cudaModule.loadFunction(fnName).w().getOrThrow()
        CudaFunc(
            func = func,
            stream = stream
        )
    }

    fun cublasHandle(): CudaBlas {
        return blas
    }

    fun setSeed(seed: ULong): Result<Unit> = runCatching {
        // Create a new curand state so the state is identical and the same random numbers will be generated.
        curand.seed = seed
        seedValue = seed
    }

    fun getCurrentSeed(): Result<ULong> = Result.success(seedValue)

    fun location(): DeviceLocation {
        return DeviceLocation.Cuda(gpuId = context.ordinal)
    }

    fun sameDevice(rhs: CudaDevice): Boolean {
        return this.id == rhs.id
    }

    fun zerosImpl(shape: Shape, dtype: DType): Result<CudaDeviceStorage> = runCatching {
        val elemCount = shape.elemCount()
        val slice = when (dtype) {
            DType.U8 -> {
                val data = allocZeros<UByte>(elemCount).getOrThrow()
                CudaStorageSlice.U8(data)
            }
            DType.U32 -> {
                val data = allocZeros<UInt>(elemCount).getOrThrow()
                CudaStorageSlice.U32(data)
            }
            DType.I16 -> {
                val data = allocZeros<Short>(elemCount).getOrThrow()
                CudaStorageSlice.I16(data)
            }
            DType.I32 -> {
                val data = allocZeros<Int>(elemCount).getOrThrow()
                CudaStorageSlice.I32(data)
            }
            DType.I64 -> {
                val data = allocZeros<Long>(elemCount).getOrThrow()
                CudaStorageSlice.I64(data)
            }
            DType.BF16 -> {
                val data = allocZeros<BFloat16>(elemCount).getOrThrow()
                CudaStorageSlice.BF16(data)
            }
            DType.F16 -> {
                val data = allocZeros<Float16>(elemCount).getOrThrow()
                CudaStorageSlice.F16(data)
            }
            DType.F32 -> {
                val data = allocZeros<Float>(elemCount).getOrThrow()
                CudaStorageSlice.F32(data)
            }
            DType.F64 -> {
                val data = allocZeros<Double>(elemCount).getOrThrow()
                CudaStorageSlice.F64(data)
            }
            DType.F8E4M3 -> {
                val data = allocZeros<F8E4M3>(elemCount).getOrThrow()
                CudaStorageSlice.F8E4M3Slice(data)
            }
            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CudaError.InternalError("Dummy types not supported in CUDA backend")
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun randUniform(shape: Shape, dtype: DType, lo: Double, up: Double): Result<CudaDeviceStorage> = runCatching {
        val elemCount = shape.elemCount()
        val slice = when (dtype) {
            DType.U8,
            DType.U32,
            DType.I16,
            DType.I32,
            DType.I64,
            DType.F16,
            DType.BF16 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype,
                    op = "rand_uniform"
                )
            }
            DType.F32 -> {
                val data = CudaSlice(MutableList(elemCount) { 0.0f })
                curand.fillWithUniformF32(data.data)
                CudaStorageSlice.F32(data)
            }
            DType.F64 -> {
                val data = CudaSlice(MutableList(elemCount) { 0.0 })
                curand.fillWithUniformF64(data.data)
                CudaStorageSlice.F64(data)
            }
            DType.F8E4M3, DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype,
                    op = "rand_uniform"
                )
            }
        }
        val scaledSlice = if (lo == 0.0 && up == 1.0) {
            slice
        } else {
            val scale = up - lo
            when (slice) {
                is CudaStorageSlice.F32 -> {
                    for (i in slice.data.data.indices) {
                        slice.data.data[i] = (slice.data.data[i] * scale.toFloat() + lo.toFloat())
                    }
                    slice
                }
                is CudaStorageSlice.F64 -> {
                    for (i in slice.data.data.indices) {
                        slice.data.data[i] = slice.data.data[i] * scale + lo
                    }
                    slice
                }
                else -> slice
            }
        }
        CudaDeviceStorage(
            slice = scaledSlice,
            device = this
        )
    }

    fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): Result<CudaDeviceStorage> = runCatching {
        val elemCount = shape.elemCount()
        // curand can only generate an even number of values.
        val elemCountRound = if (elemCount % 2 == 1) {
            elemCount + 1
        } else {
            elemCount
        }
        val slice = when (dtype) {
            DType.U8,
            DType.U32,
            DType.I16,
            DType.I32,
            DType.I64,
            DType.F16,
            DType.BF16 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype,
                    op = "rand_normal"
                )
            }
            DType.F32 -> {
                val data = CudaSlice(MutableList(elemCountRound) { 0.0f })
                curand.fillWithNormalF32(data.data, mean.toFloat(), std.toFloat())
                CudaStorageSlice.F32(data)
            }
            DType.F64 -> {
                val data = CudaSlice(MutableList(elemCountRound) { 0.0 })
                curand.fillWithNormalF64(data.data, mean, std)
                CudaStorageSlice.F64(data)
            }
            DType.F8E4M3, DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype,
                    op = "rand_normal"
                )
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun allocUninit(shape: Shape, dtype: DType): Result<CudaDeviceStorage> = runCatching {
        val elemCount = shape.elemCount()
        val slice = when (dtype) {
            DType.U8 -> {
                val data = alloc<UByte>(elemCount).getOrThrow()
                CudaStorageSlice.U8(data)
            }
            DType.U32 -> {
                val data = alloc<UInt>(elemCount).getOrThrow()
                CudaStorageSlice.U32(data)
            }
            DType.I16 -> {
                val data = alloc<Short>(elemCount).getOrThrow()
                CudaStorageSlice.I16(data)
            }
            DType.I32 -> {
                val data = alloc<Int>(elemCount).getOrThrow()
                CudaStorageSlice.I32(data)
            }
            DType.I64 -> {
                val data = alloc<Long>(elemCount).getOrThrow()
                CudaStorageSlice.I64(data)
            }
            DType.BF16 -> {
                val data = alloc<BFloat16>(elemCount).getOrThrow()
                CudaStorageSlice.BF16(data)
            }
            DType.F16 -> {
                val data = alloc<Float16>(elemCount).getOrThrow()
                CudaStorageSlice.F16(data)
            }
            DType.F32 -> {
                val data = alloc<Float>(elemCount).getOrThrow()
                CudaStorageSlice.F32(data)
            }
            DType.F64 -> {
                val data = alloc<Double>(elemCount).getOrThrow()
                CudaStorageSlice.F64(data)
            }
            DType.F8E4M3 -> {
                val data = alloc<F8E4M3>(elemCount).getOrThrow()
                CudaStorageSlice.F8E4M3Slice(data)
            }
            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CudaError.InternalError("Dummy types not supported in CUDA backend")
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun <T : WithDType> storageFromSlice(s: List<T>, cpuStorageRef: (List<T>) -> CpuStorageRef): Result<CudaDeviceStorage> = runCatching {
        val slice = when (val ref = cpuStorageRef(s)) {
            is CpuStorageRef.U8 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.U8(data)
            }
            is CpuStorageRef.U32 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.U32(data)
            }
            is CpuStorageRef.I16 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.I16(data)
            }
            is CpuStorageRef.I32 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.I32(data)
            }
            is CpuStorageRef.I64 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.I64(data)
            }
            is CpuStorageRef.BF16 -> {
                val data = cloneHtod(ref.data).getOrThrow()
                CudaStorageSlice.BF16(data)
            }
            is CpuStorageRef.F16 -> {
                val data = cloneHtod(ref.data).getOrThrow()
                CudaStorageSlice.F16(data)
            }
            is CpuStorageRef.F32 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.F32(data)
            }
            is CpuStorageRef.F64 -> {
                val data = cloneHtod(ref.data.toList()).getOrThrow()
                CudaStorageSlice.F64(data)
            }
            is CpuStorageRef.F8E4M3 -> {
                val data = cloneHtod(ref.data).getOrThrow()
                CudaStorageSlice.F8E4M3Slice(data)
            }
            is CpuStorageRef.F4,
            is CpuStorageRef.F6E2M3,
            is CpuStorageRef.F6E3M2,
            is CpuStorageRef.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = DType.F4,
                    op = "storage_from_slice"
                )
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun storageFromCpuStorage(storage: CpuStorage): Result<CudaDeviceStorage> = runCatching {
        val slice = when (storage) {
            is CpuStorage.U8 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.U8(data)
            }
            is CpuStorage.U32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.U32(data)
            }
            is CpuStorage.I16 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I16(data)
            }
            is CpuStorage.I32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I32(data)
            }
            is CpuStorage.I64 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I64(data)
            }
            is CpuStorage.BF16 -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.BF16(data)
            }
            is CpuStorage.F16 -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.F16(data)
            }
            is CpuStorage.F32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.F32(data)
            }
            is CpuStorage.F64 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.F64(data)
            }
            is CpuStorage.F8E4M3Storage -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.F8E4M3Slice(data)
            }
            is CpuStorage.F4,
            is CpuStorage.F6E2M3,
            is CpuStorage.F6E3M2,
            is CpuStorage.F8E8M0Storage -> {
                throw CudaError.UnsupportedDtype(
                    dtype = storage.dtype(),
                    op = "storage_from_cpu_storage"
                )
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun storageFromCpuStorageOwned(storage: CpuStorage): Result<CudaDeviceStorage> = runCatching {
        val slice = when (storage) {
            is CpuStorage.U8 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.U8(data)
            }
            is CpuStorage.U32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.U32(data)
            }
            is CpuStorage.I16 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I16(data)
            }
            is CpuStorage.I32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I32(data)
            }
            is CpuStorage.I64 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.I64(data)
            }
            is CpuStorage.BF16 -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.BF16(data)
            }
            is CpuStorage.F16 -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.F16(data)
            }
            is CpuStorage.F32 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.F32(data)
            }
            is CpuStorage.F64 -> {
                val data = cloneHtod(storage.data.toList()).getOrThrow()
                CudaStorageSlice.F64(data)
            }
            is CpuStorage.F8E4M3Storage -> {
                val data = cloneHtod(storage.data).getOrThrow()
                CudaStorageSlice.F8E4M3Slice(data)
            }
            is CpuStorage.F4,
            is CpuStorage.F6E2M3,
            is CpuStorage.F6E3M2,
            is CpuStorage.F8E8M0Storage -> {
                throw CudaError.UnsupportedDtype(
                    dtype = storage.dtype(),
                    op = "storage_from_cpu_storage_owned"
                )
            }
        }
        CudaDeviceStorage(
            slice = slice,
            device = this
        )
    }

    fun synchronize(): Result<Unit> = Result.success(Unit)

    companion object {
        fun newWithStream(ordinal: Int): Result<CudaDevice> = runCatching {
            val context = CudaContext(ordinal)
            val stream = CudaStream(context)
            fromContextAndStream(context, stream).getOrThrow()
        }

        private fun fromContextAndStream(
            context: CudaContext,
            stream: CudaStream
        ): Result<CudaDevice> = runCatching {
            val blas = CudaBlas(stream)
            val curand = CudaRng(299792458uL, stream)
            val moduleStore = ModuleStore()
            CudaDevice(
                id = DeviceId.new(),
                context = context,
                stream = stream,
                blas = blas,
                curand = curand,
                modules = moduleStore,
                customModules = mutableMapOf(),
                seedValue = 299792458uL
            )
        }

        fun new(ordinal: Int): Result<CudaDevice> = runCatching {
            val context = CudaContext(ordinal)
            val stream = CudaStream(context)
            fromContextAndStream(context, stream).getOrThrow()
        }
    }
}

private fun cudaGraphHtodCacheEnabled(): Boolean {
    return CUDA_GRAPH_HTOD_CACHE_DEPTH.getOr { intArrayOf(0) }[0] > 0
}

class CudaFunc(
    val func: CudaFunction,
    val stream: CudaStream
) {
    fun deref(): CudaFunction = func

    fun intoCudaFunction(): CudaFunction = func

    fun builder(): LaunchArgs = LaunchArgs(func)
}
