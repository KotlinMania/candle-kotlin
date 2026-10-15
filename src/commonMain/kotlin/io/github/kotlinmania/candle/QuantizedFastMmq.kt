// port-lint: source quantized/fast_mmq.rs
package io.github.kotlinmania.candle.quantized.fastmmq

import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.cuda.DeviceId
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.cuda.CudaSlice
import io.github.kotlinmania.candle.quantized.cuda.MATRIX_ROW_PADDING
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage

private const val QK8_1: Int = 32
private const val BLOCK_Q8_1_MMQ_SIZE: Int = 4 * QK8_1 + 4 * 4 // 128 qs + 16 scale bytes = 144

private fun pad(p: Int, q: Int): Int {
    return ((p + q - 1) / q) * q
}

/** Quant types supported by MMQ kernels (same as MMVQ). */
private fun supports(dtype: GgmlDType): Boolean {
    return when (dtype) {
        GgmlDType.Q4_0,
        GgmlDType.Q4_1,
        GgmlDType.Q5_0,
        GgmlDType.Q5_1,
        GgmlDType.Q8_0,
        GgmlDType.Q2K,
        GgmlDType.Q3K,
        GgmlDType.Q4K,
        GgmlDType.Q5K,
        GgmlDType.Q6K -> true
        else -> false
    }
}

/** qk (block quantization size) per dtype. */
private fun qkFor(dtype: GgmlDType): Int {
    return when (dtype) {
        GgmlDType.Q4_0,
        GgmlDType.Q4_1,
        GgmlDType.Q5_0,
        GgmlDType.Q5_1,
        GgmlDType.Q8_0 -> 32
        GgmlDType.Q2K,
        GgmlDType.Q3K,
        GgmlDType.Q4K,
        GgmlDType.Q5K,
        GgmlDType.Q6K -> 256
        else -> throw IllegalStateException("Unsupported dtype for qk: $dtype")
    }
}

// Scale layout mapping: which quantized MMQ scale layout to use per weight type.
private enum class DsLayout {
    D4,
    DS4,
    D2S6,
}

private fun dsLayoutFor(dtype: GgmlDType): DsLayout {
    return when (dtype) {
        GgmlDType.Q4_0, GgmlDType.Q4_1 -> DsLayout.DS4
        GgmlDType.Q5_0 -> DsLayout.D4
        GgmlDType.Q5_1 -> DsLayout.DS4
        GgmlDType.Q8_0 -> DsLayout.D4
        GgmlDType.Q2K -> DsLayout.D2S6
        GgmlDType.Q3K -> DsLayout.D4
        GgmlDType.Q4K, GgmlDType.Q5K -> DsLayout.DS4
        GgmlDType.Q6K -> DsLayout.D4
        else -> throw IllegalStateException("Unsupported dtype for layout: $dtype")
    }
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

private fun quantizeLauncher(layout: DsLayout): QuantizeLauncher {
    val kernelName = when (layout) {
        DsLayout.D4 -> "launch_mmq_quantize_q8_1_D4"
        DsLayout.DS4 -> "launch_mmq_quantize_q8_1_DS4"
        DsLayout.D2S6 -> "launch_mmq_quantize_q8_1_D2S6"
    }
    return QuantizeLauncher(kernelName) { _, _, _, _, _, _, _, _, _, _, _, _, _ -> }
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
    val kernelName = when (dtype) {
        GgmlDType.Q4_0 -> "launch_mmq_gguf_q4_0"
        GgmlDType.Q4_1 -> "launch_mmq_gguf_q4_1"
        GgmlDType.Q5_0 -> "launch_mmq_gguf_q5_0"
        GgmlDType.Q5_1 -> "launch_mmq_gguf_q5_1"
        GgmlDType.Q8_0 -> "launch_mmq_gguf_q8_0"
        GgmlDType.Q2K -> "launch_mmq_gguf_q2_k"
        GgmlDType.Q3K -> "launch_mmq_gguf_q3_k"
        GgmlDType.Q4K -> "launch_mmq_gguf_q4_k"
        GgmlDType.Q5K -> "launch_mmq_gguf_q5_k"
        GgmlDType.Q6K -> "launch_mmq_gguf_q6_k"
        else -> return null
    }
    return MmqLauncher(kernelName) { _, _, _, _, _, _, _, _, _, _, _, _, _, _ -> }
}

// ---------------------------------------------------------------------------
// Per-device workspaces (grows-only, reused across calls).
// ---------------------------------------------------------------------------

internal class WorkspaceSlot(
    var slice: CudaSlice,
    var cap: Int
)

private val MMQ_WORKSPACE: MutableMap<DeviceId, WorkspaceSlot> = mutableMapOf()
private val FIXUP_WORKSPACE: MutableMap<DeviceId, WorkspaceSlot> = mutableMapOf()

private fun workspaceEnsure(
    ws: MutableMap<DeviceId, WorkspaceSlot>,
    dev: CudaDevice,
    bytes: Int
): Result<Pair<ULong, WorkspaceSlot>> = runCatching {
    val deviceKey = DeviceId(dev.ordinal)
    val initialBytes = maxOf(bytes, 1)
    val slot = ws.getOrPut(deviceKey) {
        WorkspaceSlot(CudaSlice(ByteArray(initialBytes)), initialBytes)
    }
    if (slot.cap < bytes) {
        slot.slice = CudaSlice(ByteArray(bytes))
        slot.cap = bytes
    }
    val ptr = slot.slice.bytes.hashCode().toULong()
    Pair(ptr, slot)
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

private val DEVICE_INFO: MutableMap<DeviceId, DeviceInfo> = mutableMapOf()

private fun getDeviceInfo(dev: CudaDevice): DeviceInfo {
    val key = DeviceId(dev.ordinal)
    val cached = DEVICE_INFO[key]
    if (cached != null) {
        return cached
    }
    val major = 8
    val minor = 0
    val nsm = 1
    val smpbo = 49152L
    val warpSize = 32
    val info = DeviceInfo(
        cc = major * 100 + minor * 10,
        nsm = nsm,
        smpbo = smpbo,
        warpSize = warpSize
    )
    DEVICE_INFO[key] = info
    return info
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

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
    rhsL: Layout
): Result<Pair<CudaStorage, Shape>?> = runCatching {
    val wDtype = qstorage.dtype()
    if (!supports(wDtype)) {
        return@runCatching null
    }
    val inputDtype = rhs.dtype
    if (inputDtype != DType.BF16 && inputDtype != DType.F16 && inputDtype != DType.F32) {
        return@runCatching null
    }

    val (nrows, ncols) = selfShape.dims2().getOrThrow()

    val rhsDims = rhsL.shape().dims
    val (bSize, k) = when (rhsDims.size) {
        3 -> Pair(rhsDims[0] * rhsDims[1], rhsDims[2])
        2 -> Pair(rhsDims[0], rhsDims[1])
        else -> return@runCatching null
    }
    if (ncols != k) {
        return@runCatching null
    }
    if (bSize == 0) {
        return@runCatching null
    }

    val qk = qkFor(wDtype)
    if (k % qk != 0) {
        return@runCatching null
    }

    val offsets = rhsL.contiguousOffsets() ?: return@runCatching null
    val (o1, o2) = offsets

    val dev = qstorage.device()
    val streamPtr = dev.ordinal.toULong()

    // MMQ quantize expects f32 input. Convert if needed.
    val rhsF32Storage: CudaStorage? = if (inputDtype != DType.F32) {
        val layoutForCast = Layout.contiguousWithOffset(rhsL.shape(), rhsL.startOffset())
        rhs.toDtype(layoutForCast, DType.F32).getOrThrow()
    } else {
        null
    }

    // Get the f32 slice (either original or converted)
    val rhsPtr = if (rhsF32Storage != null) {
        (bSize * k).toULong()
    } else {
        (o1 + o2).toULong()
    }

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

    val (scratchPtr, _) = workspaceEnsure(MMQ_WORKSPACE, dev, workspaceBytes).getOrThrow()

    // Stream-k fixup workspace
    val mmqXMax = 128
    val mmqYMax = 128
    val maxSms = 256
    val fixupBytes = maxSms * mmqXMax * mmqYMax * Float.SIZE_BYTES
    val (fixupPtr, _) = workspaceEnsure(FIXUP_WORKSPACE, dev, fixupBytes).getOrThrow()

    val weightPtr = qstorage.devicePtr().getOrThrow()
    val strideRowX = (k / qk).toLong()
    val di = getDeviceInfo(dev)

    val outStorage = CudaStorage(dev, DType.F32)
    val strideColDst = nrows.toLong()
    val outPtr = (nrows * bSize).toULong()

    val quantize = quantizeLauncher(dsLayoutFor(wDtype))
    quantize.invoke(
        rhsPtr,
        0uL,
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
        streamPtr
    )

    val launcher = mmqLauncher(wDtype) ?: throw IllegalStateException("supports() checked")
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
        streamPtr
    )

    val outDims = rhsDims.dropLast(1) + nrows
    val outShape = Shape(outDims)

    if (inputDtype == DType.F32) {
        Pair(outStorage, outShape)
    } else {
        // Cast f32 output back to input dtype
        val outLayout = Layout.contiguous(outShape)
        val castStorage = outStorage.toDtype(outLayout, inputDtype).getOrThrow()
        Pair(castStorage, outShape)
    }
}
