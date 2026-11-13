// port-lint: source quantized/fast_mmvq.rs
package io.github.kotlinmania.candle.quantized.fastmmvq

import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.dummycudabackend.DeviceId
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.cuda.MATRIX_ROW_PADDING
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candlekernels.ffi
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

// CUDA fast path for GGUF matmul with BF16/F32 activations.

public fun interface FnCallback {
    public operator fun invoke(vx: ConstPtr, vy: ConstPtr, dst: MutPtr, ncolsX: Int, nrowsX: Int, strideColY: Int, strideColDst: Int, bSize: Int, stream: MutPtr)
}

public val Q8_1_BLOCK_SIZE: Int = 32

public val Q8_1_TYPE_SIZE: Int = 36

// 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
public fun pad(p: Int, q: Int): Int = p.divCeil(q) * q

/**
 * Quant types supported by the fast MMVQ kernels.
 */
public fun supports(dtype: GgmlDType): Boolean =
    when (dtype) {
        GgmlDType.q40, GgmlDType.q41, GgmlDType.q50, GgmlDType.q51, GgmlDType.q80, GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k -> true
        else -> false
    }

public val MMVQ_MAX_BATCH: Int = 8

// ---------------------------------------------------------------------------
// Per-device Q8_1 scratch workspace (grows-only, reused across calls).
// ---------------------------------------------------------------------------
public class WorkspaceSlot internal constructor(
    sliceCtorArg: CudaSlice<UByte>,
    public var cap: Int,
) {
    public var slice: CudaSlice<UByte> = sliceCtorArg
        internal set

    public constructor() : this(
        sliceCtorArg = CudaSlice<UByte>(),
        cap = 0,
    )

    public operator fun component1(): CudaSlice<UByte> = this.slice

    public operator fun component2(): Int = this.cap

    public fun copy(slice: CudaSlice<UByte> = this.slice, cap: Int = this.cap): WorkspaceSlot = WorkspaceSlot(sliceCtorArg = slice, cap = cap)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is WorkspaceSlot) return false
        if (this.slice != other.slice) return false
        if (this.cap != other.cap) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.slice.hashCode()
        result = 31 * result + this.cap.hashCode()
        return result
    }

    public companion object {
        public operator fun invoke(
            slice: CudaSlice<UByte> = CudaSlice<UByte>(),
            cap: Int = 0,
        ): WorkspaceSlot =
            WorkspaceSlot(
                sliceCtorArg = slice,
                cap = cap,
            )
    }
}

public var WORKSPACE: OnceLock<Mutex<MutableMap<DeviceId, WorkspaceSlot>>> = OnceLock.new()

/**
 * Returns a device pointer to the scratch workspace, growing it if needed.
 * The returned `MutexGuard` must be held alive until the kernels using
 * this pointer have been launched (all launches are on the device's
 * default stream, so they are serialised).
 */
public fun workspaceEnsure(dev: CudaDevice, bytes: Int): Result<Pair<ULong, MutexGuard<MutableMap<DeviceId, WorkspaceSlot>>>> {
    val map = WORKSPACE.getOrInit({ Mutex.new(mutableMapOf()) })
    val deviceKey = dev.id()
    val guard = map.lock().unwrap()
    val slot1 =
        when (val matchVal = guard.entry(deviceKey)) {
            is Entry.Occupied -> {
                val (entry) = matchVal
                val slot = entry.intoMut()
                if (slot.cap < bytes) {
                    slot.slice =
                        when (val tryResult = dev.alloc<UByte>(bytes)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    slot.cap = bytes
                }
                slot
            }
            is Entry.Vacant -> {
                val (entry1) = matchVal
                val slice =
                    when (val tryResult = dev.alloc<UByte>(bytes)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                entry1.insert(WorkspaceSlot(slice = slice, cap = bytes))
            }
        }
    val ptr = slot1.slice.devicePtr(slot1.slice.stream()).first
    return Result.Ok(Pair(ptr, guard))
}

// ---------------------------------------------------------------------------
// Launcher dispatch by weight dtype and output dtype.
// ---------------------------------------------------------------------------
public fun interface PlainLauncher {
    public operator fun invoke(vx: ConstPtr, vy: ConstPtr, dst: MutPtr, ncolsX: Int, nrowsX: Int, strideColY: Int, strideColDst: Int, bSize: Int, stream: MutPtr)
}

public fun plainLauncherBf16(dtype: GgmlDType): PlainLauncher? {
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.q40 -> ffi.launchMmvqGgufQ40Bf16Plain
            GgmlDType.q41 -> ffi.launchMmvqGgufQ41Bf16Plain
            GgmlDType.q50 -> ffi.launchMmvqGgufQ50Bf16Plain
            GgmlDType.q51 -> ffi.launchMmvqGgufQ51Bf16Plain
            GgmlDType.q80 -> ffi.launchMmvqGgufQ80Bf16Plain
            GgmlDType.q2k -> ffi.launchMmvqGgufQ2KBf16Plain
            GgmlDType.q3k -> ffi.launchMmvqGgufQ3KBf16Plain
            GgmlDType.q4k -> ffi.launchMmvqGgufQ4KBf16Plain
            GgmlDType.q5k -> ffi.launchMmvqGgufQ5KBf16Plain
            GgmlDType.q6k -> ffi.launchMmvqGgufQ6KBf16Plain
            else -> {
                return null
            }
        }
    return f
}

public fun plainLauncherF16(dtype: GgmlDType): PlainLauncher? {
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.q40 -> ffi.launchMmvqGgufQ40F16Plain
            GgmlDType.q41 -> ffi.launchMmvqGgufQ41F16Plain
            GgmlDType.q50 -> ffi.launchMmvqGgufQ50F16Plain
            GgmlDType.q51 -> ffi.launchMmvqGgufQ51F16Plain
            GgmlDType.q80 -> ffi.launchMmvqGgufQ80F16Plain
            GgmlDType.q2k -> ffi.launchMmvqGgufQ2KF16Plain
            GgmlDType.q3k -> ffi.launchMmvqGgufQ3KF16Plain
            GgmlDType.q4k -> ffi.launchMmvqGgufQ4KF16Plain
            GgmlDType.q5k -> ffi.launchMmvqGgufQ5KF16Plain
            GgmlDType.q6k -> ffi.launchMmvqGgufQ6KF16Plain
            else -> {
                return null
            }
        }
    return f
}

public fun plainLauncherF32(dtype: GgmlDType): PlainLauncher? {
    val f: PlainLauncher =
        when (dtype) {
            GgmlDType.q40 -> ffi.launchMmvqGgufQ40F32Plain
            GgmlDType.q41 -> ffi.launchMmvqGgufQ41F32Plain
            GgmlDType.q50 -> ffi.launchMmvqGgufQ50F32Plain
            GgmlDType.q51 -> ffi.launchMmvqGgufQ51F32Plain
            GgmlDType.q80 -> ffi.launchMmvqGgufQ80F32Plain
            GgmlDType.q2k -> ffi.launchMmvqGgufQ2KF32Plain
            GgmlDType.q3k -> ffi.launchMmvqGgufQ3KF32Plain
            GgmlDType.q4k -> ffi.launchMmvqGgufQ4KF32Plain
            GgmlDType.q5k -> ffi.launchMmvqGgufQ5KF32Plain
            GgmlDType.q6k -> ffi.launchMmvqGgufQ6KF32Plain
            else -> {
                return null
            }
        }
    return f
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Try the fast MMVQ path. Returns `Ok(None)` when the fast path is not applicable:
 * - unsupported quant dtype
 * - batch too large
 * - non-BF16/F32 input
 */
public fun tryFwd(qstorage: QCudaStorage, selfShape: Shape, rhs: CudaStorage, rhsL: Layout): Result<Pair<CudaStorage, Shape>?> {
    // Gate checks.
    val wDtype = qstorage.dtype()
    if (!supports(wDtype)) {
        return Result.Ok(null)
    }
    val inputDtype = rhs.dtype()
    if (!when (inputDtype) {
            DType.bf16, DType.f16, DType.f32 -> true
            else -> false
        }
    ) {
        return Result.Ok(null)
    }
    val (nrows, ncols) =
        when (val tryResult = selfShape.dims2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (bSize, k2) =
        run {
            val matchVal = rhsL.shape().dims()
            when {
                matchVal.size == 3 -> {
                    val b = matchVal[0]
                    val m = matchVal[1]
                    val k = matchVal[2]
                    Pair(b * m, k)
                }
                matchVal.size == 2 -> {
                    val b1 = matchVal[0]
                    val k1 = matchVal[1]
                    Pair(b1, k1)
                }
                else -> {
                    return Result.Ok(null)
                }
            }
        }
    if (ncols != k2) {
        return Result.Ok(null)
    }
    if (bSize == 0 || bSize > MMVQ_MAX_BATCH) {
        return Result.Ok(null)
    }
    val (o1, o2) =
        when (val matchVal1 = rhsL.contiguousOffsets()) {
            null -> {
                return Result.Ok(null)
            }
            else -> {
                val offsets = matchVal1
                offsets
            }
        }
    val dev = qstorage.device()
    val streamPtr = mutPtrFromRustCast(dev.cudaStream().cuStream())
    val kPadded = pad(k2, MATRIX_ROW_PADDING)
    val numBlocksPerRow = kPadded / Q8_1_BLOCK_SIZE
    val dstRowBytes = numBlocksPerRow * Q8_1_TYPE_SIZE
    val scratchBytes = bSize * dstRowBytes
    val (scratchPtr, workspaceGuard) =
        when (val tryResult = workspaceEnsure(dev, scratchBytes)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val scratchPtr1 = mutPtrFromRustCast(scratchPtr)
    val strideColY = (kPadded / Q8_1_BLOCK_SIZE)
    val strideColDst = nrows.toInt()
    val weightPtr =
        constPtrFromRustCast(
            when (val tryResult = qstorage.devicePtr()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        )
    val outShape = rhsL.shape().dims().toVec()
    outShape.pop()
    outShape.push(nrows)
    val stream = dev.cudaStream()
    return when (inputDtype) {
        DType.bf16 -> {
            val rhsSlice =
                when (val tryResult = rhs.asCudaSlice<Bf16>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsSlice1 = rhsSlice.slice(o1 until o2)
            val out =
                when (val tryResult = dev.alloc<Bf16>(nrows * bSize)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsPtr = constPtrFromRustCast(rhsSlice1.devicePtr(stream).first)
            val outPtr = mutPtrFromRustCast(out.devicePtr(stream).first)
            run {
                ffi.launchMmvqGgufQuantizeQ81Bf16(rhsPtr, scratchPtr1, k2.toInt(), kPadded, bSize.toInt(), streamPtr)
                val launcher = plainLauncherBf16(wDtype)!!
                launcher(weightPtr, ConstPtr(scratchPtr1.rawAddress, scratchPtr1.rawMeta), outPtr, k2.toInt(), nrows.toInt(), strideColY, strideColDst, bSize.toInt(), streamPtr)
            }
            val outStorage = CudaStorage.wrapCudaSlice(out, dev.clone())
            Result.Ok(Pair(outStorage, outShape.into()))
        }
        DType.f16 -> {
            val rhsSlice2 =
                when (val tryResult = rhs.asCudaSlice<F16>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsSlice3 = rhsSlice2.slice(o1 until o2)
            val out1 =
                when (val tryResult = dev.alloc<F16>(nrows * bSize)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsPtr1 = constPtrFromRustCast(rhsSlice3.devicePtr(stream).first)
            val outPtr1 = mutPtrFromRustCast(out1.devicePtr(stream).first)
            run {
                ffi.launchMmvqGgufQuantizeQ81F16(rhsPtr1, scratchPtr1, k2.toInt(), kPadded, bSize.toInt(), streamPtr)
                val launcher1 = plainLauncherF16(wDtype)!!
                launcher1(weightPtr, ConstPtr(scratchPtr1.rawAddress, scratchPtr1.rawMeta), outPtr1, k2.toInt(), nrows.toInt(), strideColY, strideColDst, bSize.toInt(), streamPtr)
            }
            val outStorage1 = CudaStorage.wrapCudaSlice(out1, dev.clone())
            Result.Ok(Pair(outStorage1, outShape.into()))
        }
        DType.f32 -> {
            val rhsSlice4 =
                when (val tryResult = rhs.asCudaSlice<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsSlice5 = rhsSlice4.slice(o1 until o2)
            val out2 =
                when (val tryResult = dev.alloc<Float>(nrows * bSize)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhsPtr2 = constPtrFromRustCast(rhsSlice5.devicePtr(stream).first)
            val outPtr2 = mutPtrFromRustCast(out2.devicePtr(stream).first)
            run {
                ffi.launchMmvqGgufQuantizeQ81F32(rhsPtr2, scratchPtr1, k2.toInt(), kPadded, bSize.toInt(), streamPtr)
                val launcher2 = plainLauncherF32(wDtype)!!
                launcher2(weightPtr, ConstPtr(scratchPtr1.rawAddress, scratchPtr1.rawMeta), outPtr2, k2.toInt(), nrows.toInt(), strideColY, strideColDst, bSize.toInt(), streamPtr)
            }
            val outStorage2 = CudaStorage.wrapCudaSlice(out2, dev.clone())
            Result.Ok(Pair(outStorage2, outShape.into()))
        }
        else -> Result.Ok(null)
    }
}
