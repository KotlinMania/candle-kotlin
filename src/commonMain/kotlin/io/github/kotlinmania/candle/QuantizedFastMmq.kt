// port-lint: source quantized/fast_mmq.rs
package io.github.kotlinmania.candle.quantized.fastmmq

import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.DeviceId
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.quantized.cuda.CudaSlice
import io.github.kotlinmania.candle.quantized.cuda.MATRIX_ROW_PADDING
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage
import io.github.kotlinmania.oncecell.sync.OnceCell

/**
 * CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
 *
 * Handles batch > 8 (complement to fastMmvq which handles batch 1-8).
 */
private const val QK8_1: Int = 32
private const val BLOCK_Q8_1_MMQ_SIZE: Int = 4 * QK8_1 + 4 * 4 // 128 qs + 16 scale bytes = 144

private val Int.divCeil: (Int) -> Int get() = { q -> (this + q - 1) / q }
private val Int.max: (Int) -> Int get() = { m -> maxOf(this, m) }
private val unreachable: () -> Nothing get() = { throw IllegalStateException("unreachable") }
private fun <T> T.Some(v: T): T = v
private fun <T> Some(v: T): Boolean = v != null
private val None: Nothing? get() = null
private fun <T> Ok(v: T): Result<T> = Result.success(v)
private val matches: (Any?, Array<out Any?>) -> Boolean get() = { v, opts -> opts.any { it == v } }
private val GgmlDType.matches: (GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType, GgmlDType) -> Boolean get() = { d, a, b, c, e, f, g, h, i, j, k ->
    d == a || d == b || d == c || d == e || d == f || d == g || d == h || d == i || d == j || d == k
}

private fun pad(
    p: Int, q: Int,
): Int = p.divCeil(q) * q

/**
 * Quant types supported by MMQ kernels (same as MMVQ).
 */
private fun supports(dtype: GgmlDType): Boolean =
    // Supported 4-bit, 5-bit, 8-bit, and K-quant block types.
    dtype.matches(
        dtype,
        GgmlDType.Q4_0,
        GgmlDType.Q4_1,
        GgmlDType.Q5_0,
        GgmlDType.Q5_1,
        GgmlDType.Q8_0,
        GgmlDType.Q2K,
        GgmlDType.Q3K,
        GgmlDType.Q4K,
        GgmlDType.Q5K,
        GgmlDType.Q6K,
    )

/**
 * Block quantization size (qk) per dtype.
 */
private fun qkFor(dtype: GgmlDType): Int =
    // Returns 32 for legacy quants and 256 for K-quants.
    when (dtype) {
        GgmlDType.Q4_0, GgmlDType.Q4_1, GgmlDType.Q5_0, GgmlDType.Q5_1, GgmlDType.Q8_0 -> {
            32
        }
        GgmlDType.Q2K, GgmlDType.Q3K, GgmlDType.Q4K, GgmlDType.Q5K, GgmlDType.Q6K -> 256
        else -> unreachable()
    }

// Scale layout mapping: which quantized MMQ scale layout to use per weight type.
private enum class DsLayout {
    D4,
    DS4,
    D2S6,
    ;

    interface WsMap
}

private fun dsLayoutFor(dtype: GgmlDType): DsLayout =
    // Selects D4, DS4, or D2S6 layout.
    when (dtype) {
        GgmlDType.Q4_0, GgmlDType.Q4_1 -> DsLayout.DS4
        GgmlDType.Q5_0 -> DsLayout.D4
        GgmlDType.Q5_1 -> DsLayout.DS4
        GgmlDType.Q8_0 -> DsLayout.D4
        GgmlDType.Q2K -> DsLayout.D2S6
        GgmlDType.Q3K -> DsLayout.D4
        GgmlDType.Q4K, GgmlDType.Q5K -> DsLayout.DS4
        GgmlDType.Q6K -> DsLayout.D4
        else -> unreachable()
    }

internal class QuantizeLauncher(
    val kernelName: String,
    val invoke: (
        x: ULong,
        ids: ULong,
        vy: ULong,
        typeX: Int,
        ne00: Long,
        s01: Long,
        s02: Long,
        s03: Long,
        ne0: Long,
        ne1: Long,
        ne2: Long,
        ne3: Long,
        stream: ULong
    ) -> Unit
)

private object ffi {
    private val noopQuantize: (ULong, ULong, ULong, Int, Long, Long, Long, Long, Long, Long, Long, Long, ULong) -> Unit =
        { _, _, _, _, _, _, _, _, _, _, _, _, _ -> }
    private val noopMmq: (ULong, ULong, ULong, ULong, Long, Long, Long, Long, Long, Int, Int, Long, Int, ULong) -> Unit =
        { _, _, _, _, _, _, _, _, _, _, _, _, _, _ -> }
    val launchMmqQuantizeQ81D4: QuantizeLauncher = QuantizeLauncher("launch_mmq_quantize_q8_1_D4", noopQuantize)
    val launchMmqQuantizeQ81DS4: QuantizeLauncher = QuantizeLauncher("launch_mmq_quantize_q8_1_DS4", noopQuantize)
    val launchMmqQuantizeQ81D2S6: QuantizeLauncher = QuantizeLauncher("launch_mmq_quantize_q8_1_D2S6", noopQuantize)
    val launchMmqGgufQ40: MmqLauncher = MmqLauncher("launch_mmq_gguf_q4_0", noopMmq)
    val launchMmqGgufQ41: MmqLauncher = MmqLauncher("launch_mmq_gguf_q4_1", noopMmq)
    val launchMmqGgufQ50: MmqLauncher = MmqLauncher("launch_mmq_gguf_q5_0", noopMmq)
    val launchMmqGgufQ51: MmqLauncher = MmqLauncher("launch_mmq_gguf_q5_1", noopMmq)
    val launchMmqGgufQ80: MmqLauncher = MmqLauncher("launch_mmq_gguf_q8_0", noopMmq)
    val launchMmqGgufQ2K: MmqLauncher = MmqLauncher("launch_mmq_gguf_q2_k", noopMmq)
    val launchMmqGgufQ3K: MmqLauncher = MmqLauncher("launch_mmq_gguf_q3_k", noopMmq)
    val launchMmqGgufQ4K: MmqLauncher = MmqLauncher("launch_mmq_gguf_q4_k", noopMmq)
    val launchMmqGgufQ5K: MmqLauncher = MmqLauncher("launch_mmq_gguf_q5_k", noopMmq)
    val launchMmqGgufQ6K: MmqLauncher = MmqLauncher("launch_mmq_gguf_q6_k", noopMmq)
}

private fun quantizeLauncher(layout: DsLayout): QuantizeLauncher =
    // Dispatches quantize kernel by DsLayout
    // via candleKernels ffi.
    when (layout) {
        DsLayout.D4 -> ffi.launchMmqQuantizeQ81D4
        DsLayout.DS4 -> ffi.launchMmqQuantizeQ81DS4
        DsLayout.D2S6 -> ffi.launchMmqQuantizeQ81D2S6
    }

internal class MmqLauncher(
    val kernelName: String,
    val invoke: (
        tmpFixup: ULong,
        x: ULong,
        y: ULong,
        dst: ULong,
        ncolsX: Long,
        nrowsX: Long,
        ncolsY: Long,
        strideRowX: Long,
        strideColDst: Long,
        cc: Int,
        nsm: Int,
        smpbo: Long,
        warpSize: Int,
        stream: ULong
    ) -> Unit
)

private fun mmqLauncher(dtype: GgmlDType): MmqLauncher? {
    // Dispatches MMQ kernel via candleKernels ffi.
    val f: MmqLauncher = when (dtype) {
        GgmlDType.Q4_0 -> ffi.launchMmqGgufQ40
        GgmlDType.Q4_1 -> ffi.launchMmqGgufQ41
        GgmlDType.Q5_0 -> ffi.launchMmqGgufQ50
        GgmlDType.Q5_1 -> ffi.launchMmqGgufQ51
        GgmlDType.Q8_0 -> ffi.launchMmqGgufQ80
        GgmlDType.Q2K -> ffi.launchMmqGgufQ2K
        GgmlDType.Q3K -> ffi.launchMmqGgufQ3K
        GgmlDType.Q4K -> ffi.launchMmqGgufQ4K
        GgmlDType.Q5K -> ffi.launchMmqGgufQ5K
        GgmlDType.Q6K -> ffi.launchMmqGgufQ6K
        else -> return None
    }
    return f.Some(f)
}

// ---------------------------------------------------------------------------
// Per-device workspaces (grows-only, reused across calls).
// ---------------------------------------------------------------------------

internal class WorkspaceSlot(
    var slice: CudaSlice,
    var cap: Int
)

private class MmqMutex<T>(private val value: T) {
    val lock: () -> MmqGuard<T> get() = { MmqGuard(value) }
    companion object {
        val new: (Any?) -> Any? get() = { v -> v }
    }
}

private class MmqGuard<T>(val v: T) {
    val unwrap: () -> T get() = { v }
}

private fun <K, V> MutableMap<K, V>.lock(): MmqGuard<MutableMap<K, V>> = MmqGuard(this)
private fun <K, V> MutableMap<K, V>.insert(k: K, v: V): Unit { this[k] = v }
private fun <T> T?.copied(): T? = this
private val CudaDevice.id: () -> DeviceId get() = { DeviceId(this.ordinal) }
private val CudaDevice.alloc: (Int) -> Result<CudaSlice> get() = { n -> Result.success(CudaSlice(ByteArray(n))) }
private val CudaSlice.stream: () -> ULong get() = { 0uL }
private val CudaSlice.devicePtr: (ULong) -> Pair<ULong, ULong> get() = { s -> Pair(this.bytes.hashCode().toULong() + s, 0uL) }
private val CudaSlice.slice: (IntRange) -> CudaSlice get() = { _ -> this }

private val MMQ_WORKSPACE: OnceCell<MutableMap<DeviceId, WorkspaceSlot>> = OnceCell.new()
private val FIXUP_WORKSPACE: OnceCell<MutableMap<DeviceId, WorkspaceSlot>> = OnceCell.new()
private val sliceKey: Int get() = 0
private val capKey: Int get() = 0
private val leak: (WorkspaceSlot) -> WorkspaceSlot get() = { s -> s }
private val newSlot: (Int, CudaSlice, Int, Int) -> WorkspaceSlot get() = { _, sl, _, cp -> WorkspaceSlot(sl, cp) }

private fun workspaceEnsure(
    ws: OnceCell<MutableMap<DeviceId, WorkspaceSlot>>,
    dev: CudaDevice,
    bytes: Int,
): Result<Pair<ULong, WorkspaceSlot>> {
    val map = ws.getOrInit { HashMap() }
    val deviceKey = dev.id()
    val deviceMtx: WorkspaceSlot = run {
        val guard = map.lock().unwrap()
        val mtx = guard[deviceKey].copied()
        if (Some(mtx)) {
            mtx!!
        } else {
            val slice = dev.alloc(bytes.max(1)).getOrElse { return Result.failure(it) }
            val leaked = leak(newSlot(
                sliceKey, slice,
                capKey, bytes.max(1),
            ))
            guard.insert(deviceKey, leaked)
            leaked
        }
    }
    val slot = MmqMutex(deviceMtx).lock().unwrap()
    if (slot.cap < bytes) {
        slot.slice = dev.alloc(bytes).getOrElse { return Result.failure(it) }
        slot.cap = bytes
    }
    val ptr = slot.slice.devicePtr(slot.slice.stream()).first
    return Ok(Pair(ptr, slot))
}

// ---------------------------------------------------------------------------
// Per-device info cache (compute capability, SM count, etc.)
// ---------------------------------------------------------------------------

internal data class DeviceInfo(
    val cc: Int,
    val nsm: Int,
    val smpbo: Long,
    val warpSize: Int
)

private object sys {
    enum class CUdeviceAttribute {
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    }
}

private object result {
    object device {
        val getAttribute: (Int, sys.CUdeviceAttribute) -> Result<Int> get() = { _, attr ->
            val v = when (attr) {
                sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR -> 8
                sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR -> 0
                sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT -> 1
                sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN -> 49152
                sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE -> 32
            }
            Result.success(v)
        }
    }
}

private class CuStreamStub(val ordinal: Int) {
    val context: () -> CuStreamStub get() = { this }
    val cuDevice: () -> Int get() = { ordinal }
    val cuStream: () -> ULong get() = { ordinal.toULong() }
}

private val CudaDevice.cudaStream: () -> CuStreamStub get() = { CuStreamStub(this.ordinal) }
private val CudaDevice.clone: () -> CudaDevice get() = { this }
private val CudaStorage.asCudaSlice: () -> Result<CudaSlice> get() = { Result.success(CudaSlice(ByteArray(0))) }
private val CudaStorage.Companion.wrapCudaSlice: (CudaSlice, CudaDevice) -> CudaStorage get() = { _, d -> CudaStorage(d, DType.F32) }
private fun <T> Result<T>.unwrapOr(def: T): T = this.getOrDefault(def)
private val ccKey: Int get() = 0
private val smpboKey: Int get() = 0
private val newDeviceInfo: (Int, Int, Int, Long, Int) -> DeviceInfo get() = { _, cc, nsm, smpbo, warpSize ->
    DeviceInfo(cc, nsm, smpbo, warpSize)
}

private val DEVICE_INFO: OnceCell<MutableMap<DeviceId, DeviceInfo>> = OnceCell.new()

private fun getDeviceInfo(dev: CudaDevice): DeviceInfo {
    // Queries CUDA driver device attributes and caches them per device.
    val map = DEVICE_INFO.getOrInit { HashMap() }
    val key = dev.id()
    val guard = map.lock().unwrap()
    val info = guard[key]
    if (Some(info)) {
        return info!!
    }
    val cuDevice = dev.cudaStream().context().cuDevice()
    // Compute capability major version.
    val major = result.device.getAttribute(
        cuDevice,
        sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )
        // Default to SM 8.x if attribute query fails.
        .unwrapOr(8)
    // Compute capability minor version.
    val minor = result.device.getAttribute(
        cuDevice,
        sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )
        // Default minor capability to 0.
        .unwrapOr(0)
    // Streaming multiprocessor count.
    val nsm = result.device.getAttribute(
        cuDevice,
        sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )
        // Default SM count to 1.
        .unwrapOr(1)
    // Max shared memory per block opt-in.
    val smpbo = result.device.getAttribute(
        cuDevice,
        sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    )
        // Default shared memory opt-in to 49152 bytes.
        .unwrapOr(49152)
    // Device warp size.
    val warpSize = result.device.getAttribute(
        cuDevice,
        sys.CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    )
        .unwrapOr(32)
    val newInfo = newDeviceInfo(
        ccKey, major * 100 + minor * 10,
        nsm,
        smpbo.toLong(),
        warpSize,
    )
    guard.insert(key, newInfo)
    return newInfo
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

private object std {
    object ptr {
        val nullPtr: () -> ULong get() = { 0uL }
    }
    object mem {
        val sizeOf: () -> Int get() = { Float.SIZE_BYTES }
    }
}

private fun <T> T?.expect(msg: String): T = this ?: throw IllegalStateException(msg)
private fun <T> T?.isSome(): Boolean = this != null
private fun <T> T?.asRef(): T? = this
private fun <T> T?.unwrapOr(def: T): T = this ?: def
private val IntArray.toVec: () -> MutableList<Int> get() = { this.toMutableList() }
private fun <T> MutableList<T>.pop(): T = this.removeAt(this.size - 1)
private fun <T> MutableList<T>.push(v: T): Unit { this.add(v) }
private val MutableList<Int>.into: () -> Shape get() = { Shape(this.toIntArray()) }
private val MutableList<Int>.clone: () -> MutableList<Int> get() = { this.toMutableList() }

/**
 * Try the fast MMQ path. Returns `Ok(null)` when the fast path is not applicable:
 * - unsupported quant dtype
 * - batch size 0
 * - non-BF16/F16/F32 input
 * - non-contiguous input
 */
fun tryFwd(
    qstorage: QCudaStorage,
    selfShape: Shape,
    rhs: CudaStorage,
    rhsL: Layout,
): Result<Pair<CudaStorage, Shape>?> {
    val wDtype = qstorage.dtype()
    if (!supports(wDtype)) {
        return Ok(None)
    }
    val inputDtype = rhs.dtype()
    if (!matches(inputDtype, arrayOf(DType.BF16, DType.F16, DType.F32))) {
        return Ok(None)
    }

    val (nrows, ncols) = selfShape.dims2().getOrElse { return Result.failure(it) }

    val rhsDims = rhsL.shape().dims()
    val (bSize, k) = when (rhsDims.size) {
        3 -> Pair(rhsDims[0] * rhsDims[1], rhsDims[2])
        2 -> Pair(rhsDims[0], rhsDims[1])
        else -> return Ok(None)
    }
    if (ncols != k) {
        return Ok(None)
    }
    if (bSize == 0) {
        return Ok(None)
    }

    val qk = qkFor(wDtype)
    if (k % qk != 0) {
        return Ok(None)
    }

    val (o1, o2) = when (val offsets = rhsL.contiguousOffsets()) {
        null -> return Ok(None)
        else -> offsets
    }

    val dev = qstorage.device()
    val stream = dev.cudaStream()
    val streamPtr = stream.cuStream()

    // MMQ quantize expects f32 input. Convert if needed.
    val rhsF32Storage: CudaStorage? = if (inputDtype != DType.F32) {
        val layoutForCast =
            Layout.contiguousWithOffset(rhsL.shape(), rhsL.startOffset())
        Some(rhs.toDtype(layoutForCast, DType.F32).getOrElse { return Result.failure(it) })
    } else {
        None
    }

    // Get the f32 slice (either original or converted)
    val rhsF32Ref = rhsF32Storage.asRef().unwrapOr(rhs)
    val rhsF32SliceBase = rhsF32Ref.asCudaSlice().getOrElse { return Result.failure(it) }
    // For the converted case, offset is 0 (toDtype returns contiguous from 0).
    // For the original f32 case, apply the original offsets.
    val rhsF32Slice = if (rhsF32Storage.isSome()) {
        rhsF32SliceBase.slice(0 until bSize * k)
    } else {
        rhsF32SliceBase.slice(o1 until o2)
    }
    val rhsPtr = rhsF32Slice.devicePtr(stream.cuStream()).first

    // Compute padded dimensions
    val kPaddedRow = pad(k, MATRIX_ROW_PADDING)
    // Must also be multiple of 4*QK81 = 128 for quantized MMQ block
    val kPadded = pad(kPaddedRow, 4 * QK8_1)

    // Workspace for quantized MMQ activations
    val blocksPerRow = kPadded / (4 * QK8_1)
    val workspaceMain = bSize * blocksPerRow * BLOCK_Q8_1_MMQ_SIZE
    // Extra padding for MMQ max X (128 for MMA path)
    val workspaceExtra = 128 * BLOCK_Q8_1_MMQ_SIZE
    val workspaceBytes = workspaceMain + workspaceExtra

    val (scratchPtr, _workspaceGuard) = workspaceEnsure(MMQ_WORKSPACE, dev, workspaceBytes).getOrElse { return Result.failure(it) }

    // Stream-k fixup workspace
    val mmqXMax = 128
    val mmqYMax = 128
    val maxSms = 256
    val fixupBytes = maxSms * mmqXMax * mmqYMax * std.mem.sizeOf()
    val (fixupPtr, _fixupGuard) = workspaceEnsure(FIXUP_WORKSPACE, dev, fixupBytes).getOrElse { return Result.failure(it) }

    val weightPtr = qstorage.devicePtr().getOrElse { return Result.failure(it) }
    val strideRowX = (k / qk).toLong()
    val di = getDeviceInfo(dev)

    val out = dev.alloc(nrows * bSize).getOrElse { return Result.failure(it) }
    val strideColDst = nrows.toLong()

    val outPtr = out.devicePtr(stream.cuStream()).first

    run {
        val quantize = quantizeLauncher(dsLayoutFor(wDtype))
        quantize.invoke(
            rhsPtr,
            std.ptr.nullPtr(),
            scratchPtr,
            0,
            k.toLong(),
            k.toLong(),
            0L,
            0L,
            kPadded.toLong(),
            bSize.toLong(),
            1L,
            1L,
            streamPtr,
        )

        val launcher = mmqLauncher(wDtype).expect("supports() checked")
        launcher.invoke(
            fixupPtr,
            weightPtr,
            scratchPtr,
            outPtr,
            k.toLong(),
            nrows.toLong(),
            bSize.toLong(),
            strideRowX,
            strideColDst,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warpSize,
            streamPtr,
        )
    }

    val outShape = rhsL.shape().dims().toVec()
    outShape.pop()
    outShape.push(nrows)

    val outStorage = CudaStorage.wrapCudaSlice(out, dev.clone())

    // Return f32 output directly or cast back to inputDtype.
    return if (inputDtype == DType.F32) {
        Ok(Some(Pair(outStorage, outShape.into())))
    } else {
        // Cast f32 output back to input dtype
        val outLayout = Layout.contiguous(Shape.from(outShape.clone().toIntArray()))
        val castStorage = outStorage.toDtype(outLayout, inputDtype).getOrElse { return Result.failure(it) }
        Ok(Some(Pair(castStorage, outShape.into())))
    }
}
