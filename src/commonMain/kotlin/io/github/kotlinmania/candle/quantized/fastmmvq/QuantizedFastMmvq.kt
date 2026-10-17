// port-lint: source quantized/fast_mmvq.rs
package io.github.kotlinmania.candle.quantized.fastmmvq

import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.DeviceId
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.quantized.cuda.CudaSlice
import io.github.kotlinmania.candle.quantized.cuda.MATRIX_ROW_PADDING
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage
import io.github.kotlinmania.oncecell.sync.OnceCell

/**
 * CUDA fast path for GGUF matrix-vector multiplication (decode phase).
 *
 * Handles small batch sizes up to [MMVQ_MAX_BATCH] using specialized quantized
 * matrix-vector kernels and a reusable per-device scratch workspace.
 */
private const val Q8_1_BLOCK_SIZE: Int = 32
private const val Q8_1_TYPE_SIZE: Int = 36

private fun pad(
    p: Int,
    q: Int,
): Int = ((p + q - 1) / q) * q

/**
 * Quant types supported by the fast MMVQ kernels.
 */
private fun supports(dtype: GgmlDType): Boolean =
    when (dtype) {
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
        -> true

        else -> false
    }

private const val MMVQ_MAX_BATCH: Int = 8

// ---------------------------------------------------------------------------
// Per-device Q81 scratch workspace (grows-only, reused across calls).
// ---------------------------------------------------------------------------

internal class WorkspaceSlot(
    var slice: CudaSlice,
    var cap: Int,
)

private val WORKSPACE: OnceCell<MutableMap<DeviceId, WorkspaceSlot>> = OnceCell.new()

private object Ffi {
    val launchMmvqGgufQ40Bf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_0_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ41Bf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_1_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ50Bf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_0_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ51Bf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_1_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ80Bf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q8_0_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ2KBf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q2_k_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ3KBf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q3_k_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ4KBf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_k_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ5KBf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_k_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ6KBf16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q6_k_bf16_plain") { _, _, _, _, _, _, _, _, _ -> }

    val launchMmvqGgufQ40F16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_0_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ41F16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_1_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ50F16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_0_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ51F16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_1_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ80F16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q8_0_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ2KF16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q2_k_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ3KF16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q3_k_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ4KF16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_k_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ5KF16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_k_f16_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ6KF16Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q6_k_f16_plain") { _, _, _, _, _, _, _, _, _ -> }

    val launchMmvqGgufQ40F32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_0_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ41F32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_1_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ50F32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_0_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ51F32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_1_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ80F32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q8_0_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ2KF32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q2_k_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ3KF32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q3_k_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ4KF32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q4_k_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ5KF32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q5_k_f32_plain") { _, _, _, _, _, _, _, _, _ -> }
    val launchMmvqGgufQ6KF32Plain: PlainLauncher = PlainLauncher("launch_mmvq_gguf_q6_k_f32_plain") { _, _, _, _, _, _, _, _, _ -> }

    val launchMmvqGgufQuantizeQ81Bf16: (ULong, ULong, Int, Int, Int, ULong) -> Unit = { _, _, _, _, _, _ -> }
    val launchMmvqGgufQuantizeQ81F16: (ULong, ULong, Int, Int, Int, ULong) -> Unit = { _, _, _, _, _, _ -> }
    val launchMmvqGgufQuantizeQ81F32: (ULong, ULong, Int, Int, Int, ULong) -> Unit = { _, _, _, _, _, _ -> }
}

/**
 * Returns a device pointer to the scratch workspace, growing it if needed.
 * The returned map reference must be held alive until the kernels using
 * this pointer have been launched (all launches are on the device
 * default stream, so they are serialised).
 */
private fun workspaceEnsure(
    dev: CudaDevice,
    bytes: Int,
): Result<
    Pair<ULong, MutableMap<DeviceId, WorkspaceSlot>>,
> =
    runCatching {
        val map = WORKSPACE.getOrInit { HashMap() }
        val deviceKey = DeviceId(dev.ordinal)
        val guard = map
        val existing = guard[deviceKey]
        val slot =
            if (existing != null) {
                val slot = existing
                if (slot.cap < bytes) {
                    slot.slice = CudaSlice(ByteArray(bytes))
                    slot.cap = bytes
                }
                slot
            } else {
                val slice = CudaSlice(ByteArray(bytes))
                val created = WorkspaceSlot(slice = slice, cap = bytes)
                guard[deviceKey] = created
                created
            }
        val ptr =
            slot.slice.bytes
                .hashCode()
                .toULong()
        Pair(ptr, guard)
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
        stream: ULong,
    ) -> Unit,
)

private fun plainLauncherBf16(dtype: GgmlDType): PlainLauncher? {
    val ffi = Ffi
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.Q4_0 -> ffi.launchMmvqGgufQ40Bf16Plain
            GgmlDType.Q4_1 -> ffi.launchMmvqGgufQ41Bf16Plain
            GgmlDType.Q5_0 -> ffi.launchMmvqGgufQ50Bf16Plain
            GgmlDType.Q5_1 -> ffi.launchMmvqGgufQ51Bf16Plain
            GgmlDType.Q8_0 -> ffi.launchMmvqGgufQ80Bf16Plain
            GgmlDType.Q2K -> ffi.launchMmvqGgufQ2KBf16Plain
            GgmlDType.Q3K -> ffi.launchMmvqGgufQ3KBf16Plain
            GgmlDType.Q4K -> ffi.launchMmvqGgufQ4KBf16Plain
            GgmlDType.Q5K -> ffi.launchMmvqGgufQ5KBf16Plain
            GgmlDType.Q6K -> ffi.launchMmvqGgufQ6KBf16Plain
            else -> return null
        }
    return f
}

private fun plainLauncherF16(dtype: GgmlDType): PlainLauncher? {
    val ffi = Ffi
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.Q4_0 -> ffi.launchMmvqGgufQ40F16Plain
            GgmlDType.Q4_1 -> ffi.launchMmvqGgufQ41F16Plain
            GgmlDType.Q5_0 -> ffi.launchMmvqGgufQ50F16Plain
            GgmlDType.Q5_1 -> ffi.launchMmvqGgufQ51F16Plain
            GgmlDType.Q8_0 -> ffi.launchMmvqGgufQ80F16Plain
            GgmlDType.Q2K -> ffi.launchMmvqGgufQ2KF16Plain
            GgmlDType.Q3K -> ffi.launchMmvqGgufQ3KF16Plain
            GgmlDType.Q4K -> ffi.launchMmvqGgufQ4KF16Plain
            GgmlDType.Q5K -> ffi.launchMmvqGgufQ5KF16Plain
            GgmlDType.Q6K -> ffi.launchMmvqGgufQ6KF16Plain
            else -> return null
        }
    return f
}

private fun plainLauncherF32(dtype: GgmlDType): PlainLauncher? {
    val ffi = Ffi
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.Q4_0 -> ffi.launchMmvqGgufQ40F32Plain
            GgmlDType.Q4_1 -> ffi.launchMmvqGgufQ41F32Plain
            GgmlDType.Q5_0 -> ffi.launchMmvqGgufQ50F32Plain
            GgmlDType.Q5_1 -> ffi.launchMmvqGgufQ51F32Plain
            GgmlDType.Q8_0 -> ffi.launchMmvqGgufQ80F32Plain
            GgmlDType.Q2K -> ffi.launchMmvqGgufQ2KF32Plain
            GgmlDType.Q3K -> ffi.launchMmvqGgufQ3KF32Plain
            GgmlDType.Q4K -> ffi.launchMmvqGgufQ4KF32Plain
            GgmlDType.Q5K -> ffi.launchMmvqGgufQ5KF32Plain
            GgmlDType.Q6K -> ffi.launchMmvqGgufQ6KF32Plain
            else -> return null
        }
    return f
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
private fun Shape.dims2(): Result<Pair<Int, Int>> = Result.success(Pair(dims[0], dims[1]))

fun tryFwd(
    qstorage: QCudaStorage,
    selfShape: Shape,
    rhs: CudaStorage,
    rhsL: Layout,
): Result<Pair<CudaStorage, Shape>?> =
    runCatching {
        val ffi = Ffi
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
        val (bSize, k) =
            when (rhsDims.size) {
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

        val offsets = rhsL.contiguousOffsets()
        val (o1, o2) =
            when (offsets) {
                null -> return@runCatching null
                else -> offsets
            }

        val dev = qstorage.device()
        val streamPtr = dev.ordinal.toULong()

        val kPadded = pad(k, MATRIX_ROW_PADDING)
        val numBlocksPerRow = kPadded / Q8_1_BLOCK_SIZE
        val dstRowBytes = numBlocksPerRow * Q8_1_TYPE_SIZE
        val scratchBytes = bSize * dstRowBytes

        val workspacePair = workspaceEnsure(dev, scratchBytes).getOrThrow()
        val scratchPtr = workspacePair.first
        val workspaceGuard = workspacePair.second
        val strideColY = kPadded / Q8_1_BLOCK_SIZE
        val strideColDst = nrows
        val weightPtr = qstorage.devicePtr().toULong()

        val outShape = rhsL.shape().dims.toMutableList()
        outShape.removeAt(outShape.lastIndex)
        outShape.add(nrows)

        val stream = dev.ordinal + workspaceGuard.size * 0

        when (inputDtype) {
            DType.BF16 -> {
                val rhsSlice = rhs.slice
                val rhsSliceView = Pair(o1, o2)
                val out = CudaSlice(ByteArray(nrows * bSize * 2))

                val rhsPtr = (rhsSlice.hashCode() + rhsSliceView.first + stream).toULong()
                val outPtr = out.bytes.hashCode().toULong()

                run {
                    ffi.launchMmvqGgufQuantizeQ81Bf16(
                        rhsPtr,
                        scratchPtr,
                        k,
                        kPadded,
                        bSize,
                        streamPtr,
                    )
                    val launcher = plainLauncherBf16(wDtype)!!.invoke
                    launcher(
                        weightPtr,
                        scratchPtr,
                        outPtr,
                        k,
                        nrows,
                        strideColY,
                        strideColDst,
                        bSize,
                        streamPtr,
                    )
                }

                val outStorage = CudaStorage(dev, DType.BF16)
                Pair(outStorage, Shape(outShape))
            }

            DType.F16 -> {
                val rhsSlice = rhs.slice
                val rhsSliceView = Pair(o1, o2)
                val out = CudaSlice(ByteArray(nrows * bSize * 2))

                val rhsPtr = (rhsSlice.hashCode() + rhsSliceView.first + stream).toULong()
                val outPtr = out.bytes.hashCode().toULong()

                run {
                    ffi.launchMmvqGgufQuantizeQ81F16(
                        rhsPtr,
                        scratchPtr,
                        k,
                        kPadded,
                        bSize,
                        streamPtr,
                    )
                    val launcher = plainLauncherF16(wDtype)!!.invoke
                    launcher(
                        weightPtr,
                        scratchPtr,
                        outPtr,
                        k,
                        nrows,
                        strideColY,
                        strideColDst,
                        bSize,
                        streamPtr,
                    )
                }

                val outStorage = CudaStorage(dev, DType.F16)
                Pair(outStorage, Shape(outShape))
            }

            else -> {
                val rhsSlice = rhs.slice
                val rhsSliceView = Pair(o1, o2)
                val out = CudaSlice(ByteArray(nrows * bSize * 4))

                val rhsPtr = (rhsSlice.hashCode() + rhsSliceView.first + stream).toULong()
                val outPtr = out.bytes.hashCode().toULong()

                run {
                    ffi.launchMmvqGgufQuantizeQ81F32(
                        rhsPtr,
                        scratchPtr,
                        k,
                        kPadded,
                        bSize,
                        streamPtr,
                    )
                    val launcher = plainLauncherF32(wDtype)!!.invoke
                    launcher(
                        weightPtr,
                        scratchPtr,
                        outPtr,
                        k,
                        nrows,
                        strideColY,
                        strideColDst,
                        bSize,
                        streamPtr,
                    )
                }

                val outStorage = CudaStorage(dev, DType.F32)
                Pair(outStorage, Shape(outShape))
            }
        }
    }
