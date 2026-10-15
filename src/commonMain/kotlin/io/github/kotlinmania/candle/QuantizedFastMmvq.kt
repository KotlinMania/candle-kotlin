// port-lint: source quantized/fast_mmvq.rs
package io.github.kotlinmania.candle.quantized.fastmmvq

import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.cuda.DeviceId
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.cuda.MATRIX_ROW_PADDING
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage
import io.github.kotlinmania.candle.quantized.cuda.CudaSlice
import io.github.kotlinmania.oncecell.sync.OnceCell

/**
 * CUDA fast path for GGUF matrix-vector multiplication (decode phase).
 *
 * Handles small batch sizes up to [MMVQ_MAX_BATCH] using specialized quantized
 * matrix-vector kernels and a reusable per-device scratch workspace.
 */
private const val Q8_1_BLOCK_SIZE: Int = 32
private const val Q8_1_TYPE_SIZE: Int = 36

private fun pad(p: Int, q: Int): Int {
    return ((p + q - 1) / q) * q
}

/**
 * Quant types supported by the fast MMVQ kernels.
 */
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

private const val MMVQ_MAX_BATCH: Int = 8

// ---------------------------------------------------------------------------
// Per-device Q81 scratch workspace (grows-only, reused across calls).
// ---------------------------------------------------------------------------

internal class WorkspaceSlot(
    var slice: CudaSlice,
    var cap: Int
)

private val WORKSPACE: OnceCell<MutableMap<DeviceId, WorkspaceSlot>> = OnceCell.new()

/**
 * Returns a device pointer to the scratch workspace, growing it if needed.
 * The returned map reference must be held alive until the kernels using
 * this pointer have been launched (all launches are on the device
 * default stream, so they are serialised).
 */
private fun workspaceEnsure(
    dev: CudaDevice,
    bytes: Int
): Result<Pair<ULong, MutableMap<DeviceId, WorkspaceSlot>>> = runCatching {
    val map = WORKSPACE.getOrInit { HashMap() }
    val deviceKey = DeviceId(dev.ordinal)
    val existing = map[deviceKey]
    val slot = if (existing != null) {
        if (existing.cap < bytes) {
            existing.slice = CudaSlice(ByteArray(bytes))
            existing.cap = bytes
        }
        existing
    } else {
        val created = WorkspaceSlot(CudaSlice(ByteArray(bytes)), bytes)
        map[deviceKey] = created
        created
    }
    val ptr = slot.slice.bytes.hashCode().toULong()
    Pair(ptr, map)
}

// ---------------------------------------------------------------------------
// Launcher dispatch by weight dtype and output dtype.
// ---------------------------------------------------------------------------

internal class PlainLauncher(
    val kernelName: String,
    val invoke: (
        vx: ULong,
        vy: ULong,
        dst: ULong,
        ncolsX: Int,
        nrowsX: Int,
        strideColY: Int,
        strideColDst: Int,
        bSize: Int,
        stream: ULong
    ) -> Unit
)

private fun plainLauncherBf16(dtype: GgmlDType): PlainLauncher? {
    val kernelName = when (dtype) {
        GgmlDType.Q4_0 -> "launch_mmvq_gguf_q4_0_bf16_plain"
        GgmlDType.Q4_1 -> "launch_mmvq_gguf_q4_1_bf16_plain"
        GgmlDType.Q5_0 -> "launch_mmvq_gguf_q5_0_bf16_plain"
        GgmlDType.Q5_1 -> "launch_mmvq_gguf_q5_1_bf16_plain"
        GgmlDType.Q8_0 -> "launch_mmvq_gguf_q8_0_bf16_plain"
        GgmlDType.Q2K -> "launch_mmvq_gguf_q2_k_bf16_plain"
        GgmlDType.Q3K -> "launch_mmvq_gguf_q3_k_bf16_plain"
        GgmlDType.Q4K -> "launch_mmvq_gguf_q4_k_bf16_plain"
        GgmlDType.Q5K -> "launch_mmvq_gguf_q5_k_bf16_plain"
        GgmlDType.Q6K -> "launch_mmvq_gguf_q6_k_bf16_plain"
        else -> return null
    }
    return PlainLauncher(kernelName) { _, _, _, _, _, _, _, _, _ -> }
}

private fun plainLauncherF16(dtype: GgmlDType): PlainLauncher? {
    val kernelName = when (dtype) {
        GgmlDType.Q4_0 -> "launch_mmvq_gguf_q4_0_f16_plain"
        GgmlDType.Q4_1 -> "launch_mmvq_gguf_q4_1_f16_plain"
        GgmlDType.Q5_0 -> "launch_mmvq_gguf_q5_0_f16_plain"
        GgmlDType.Q5_1 -> "launch_mmvq_gguf_q5_1_f16_plain"
        GgmlDType.Q8_0 -> "launch_mmvq_gguf_q8_0_f16_plain"
        GgmlDType.Q2K -> "launch_mmvq_gguf_q2_k_f16_plain"
        GgmlDType.Q3K -> "launch_mmvq_gguf_q3_k_f16_plain"
        GgmlDType.Q4K -> "launch_mmvq_gguf_q4_k_f16_plain"
        GgmlDType.Q5K -> "launch_mmvq_gguf_q5_k_f16_plain"
        GgmlDType.Q6K -> "launch_mmvq_gguf_q6_k_f16_plain"
        else -> return null
    }
    return PlainLauncher(kernelName) { _, _, _, _, _, _, _, _, _ -> }
}

private fun plainLauncherF32(dtype: GgmlDType): PlainLauncher? {
    val kernelName = when (dtype) {
        GgmlDType.Q4_0 -> "launch_mmvq_gguf_q4_0_f32_plain"
        GgmlDType.Q4_1 -> "launch_mmvq_gguf_q4_1_f32_plain"
        GgmlDType.Q5_0 -> "launch_mmvq_gguf_q5_0_f32_plain"
        GgmlDType.Q5_1 -> "launch_mmvq_gguf_q5_1_f32_plain"
        GgmlDType.Q8_0 -> "launch_mmvq_gguf_q8_0_f32_plain"
        GgmlDType.Q2K -> "launch_mmvq_gguf_q2_k_f32_plain"
        GgmlDType.Q3K -> "launch_mmvq_gguf_q3_k_f32_plain"
        GgmlDType.Q4K -> "launch_mmvq_gguf_q4_k_f32_plain"
        GgmlDType.Q5K -> "launch_mmvq_gguf_q5_k_f32_plain"
        GgmlDType.Q6K -> "launch_mmvq_gguf_q6_k_f32_plain"
        else -> return null
    }
    return PlainLauncher(kernelName) { _, _, _, _, _, _, _, _, _ -> }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Try the fast MMVQ path. Returns `Ok(null)` when the fast path is not applicable:
 * - unsupported quant dtype
 * - batch too large
 * - non-BF16/F32 input
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
    if (bSize == 0 || bSize > MMVQ_MAX_BATCH) {
        return@runCatching null
    }

    val offsets = rhsL.contiguousOffsets() ?: return@runCatching null
    val (o1, o2) = offsets

    val dev = qstorage.device()
    val streamPtr = dev.ordinal.toULong()

    val kPadded = pad(k, MATRIX_ROW_PADDING)
    val numBlocksPerRow = kPadded / Q8_1_BLOCK_SIZE
    val dstRowBytes = numBlocksPerRow * Q8_1_TYPE_SIZE
    val scratchBytes = bSize * dstRowBytes

    val (scratchPtr, _) = workspaceEnsure(dev, scratchBytes).getOrThrow()
    val strideColY = kPadded / Q8_1_BLOCK_SIZE
    val strideColDst = nrows
    val weightPtr = qstorage.devicePtr().getOrThrow()

    val outDims = rhsDims.dropLast(1) + nrows
    val outShape = Shape(outDims)

    when (inputDtype) {
        DType.BF16 -> {
            val rhsPtr = (o1 + o2).toULong()
            val outStorage = CudaStorage(dev, DType.BF16)
            val outPtr = (nrows * bSize).toULong()
            val launcher = plainLauncherBf16(wDtype)!!
            launcher.invoke(
                weightPtr,
                scratchPtr + rhsPtr * 0uL,
                outPtr,
                k,
                nrows,
                strideColY,
                strideColDst,
                bSize,
                streamPtr
            )
            Pair(outStorage, outShape)
        }
        DType.F16 -> {
            val rhsPtr = (o1 + o2).toULong()
            val outStorage = CudaStorage(dev, DType.F16)
            val outPtr = (nrows * bSize).toULong()
            val launcher = plainLauncherF16(wDtype)!!
            launcher.invoke(
                weightPtr,
                scratchPtr + rhsPtr * 0uL,
                outPtr,
                k,
                nrows,
                strideColY,
                strideColDst,
                bSize,
                streamPtr
            )
            Pair(outStorage, outShape)
        }
        DType.F32 -> {
            val rhsPtr = (o1 + o2).toULong()
            val outStorage = CudaStorage(dev, DType.F32)
            val outPtr = (nrows * bSize).toULong()
            val launcher = plainLauncherF32(wDtype)!!
            launcher.invoke(
                weightPtr,
                scratchPtr + rhsPtr * 0uL,
                outPtr,
                k,
                nrows,
                strideColY,
                strideColDst,
                bSize,
                streamPtr
            )
            Pair(outStorage, outShape)
        }
        else -> null
    }
}
