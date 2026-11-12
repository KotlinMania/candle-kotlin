// port-lint: source quantized/fast_mmq.rs
package io.github.kotlinmania.candle.quantized.fastmmq

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
import io.github.kotlinmania.cudarc.driver.sys
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.OnceLock

// CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
// Handles batch > 8 (complement to fastMmvq which handles batch 1-8).

public fun interface FnCallback {
    public operator fun invoke(x: ConstPtr, ids: ConstPtr, vy: MutPtr, typeX: Int, ne00: Long, s01: Long, s02: Long, s03: Long, ne0: Long, ne1: Long, ne2: Long, ne3: Long, stream: MutPtr)
}

public val QK8_1: Int = 32

public val BLOCK_Q8_1_MMQ_SIZE: Int = 4 * QK8_1 + 4 * 4

// 128 qs + 16 scale bytes = 144
public fun pad(p: Int, q: Int): Int = p.divCeil(q) * q

/**
 * Quant types supported by MMQ kernels (same as MMVQ).
 */
public fun supports(dtype: GgmlDType): Boolean =
    when (dtype) {
        GgmlDType.q40, GgmlDType.q41, GgmlDType.q50, GgmlDType.q51, GgmlDType.q80, GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k -> true
        else -> false
    }

/**
 * qk (block quantization size) per dtype.
 */
public fun qkFor(dtype: GgmlDType): Int =
    when (dtype) {
        GgmlDType.q40, GgmlDType.q41, GgmlDType.q50, GgmlDType.q51, GgmlDType.q80 -> {
            32
        }
        GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k -> 256
        else -> error("unreachable")
    }

// dsLayout mapping: which Q81Mmq scale layout to use per weight type.
public enum class DsLayout(
    public val value: UInt,
) {
    D4(0u),
    DS4(1u),
    D2S6(2u),
}

public fun dsLayoutFor(dtype: GgmlDType): DsLayout =
    when (dtype) {
        GgmlDType.q40, GgmlDType.q41 -> DsLayout.DS4
        GgmlDType.q50 -> DsLayout.D4
        GgmlDType.q51 -> DsLayout.DS4
        GgmlDType.q80 -> DsLayout.D4
        GgmlDType.q2k -> DsLayout.D2S6
        GgmlDType.q3k -> DsLayout.D4
        GgmlDType.q4k, GgmlDType.q5k -> DsLayout.DS4
        GgmlDType.q6k -> DsLayout.D4
        else -> error("unreachable")
    }

public fun interface QuantizeLauncher {
    public operator fun invoke(x: ConstPtr, ids: ConstPtr, vy: MutPtr, typeX: Int, ne00: Long, s01: Long, s02: Long, s03: Long, ne0: Long, ne1: Long, ne2: Long, ne3: Long, stream: MutPtr)
}

public fun quantizeLauncher(layout: DsLayout): QuantizeLauncher =
    when (layout) {
        DsLayout.d4 -> ffi.launchMmqQuantizeQ81D4
        DsLayout.ds4 -> ffi.launchMmqQuantizeQ81Ds4
        DsLayout.d2s6 -> ffi.launchMmqQuantizeQ81D2s6
    }

public fun interface MmqLauncher {
    public operator fun invoke(tmpFixup: MutPtr, x: ConstPtr, y: ConstPtr, dst: MutPtr, ncolsX: Long, nrowsX: Long, ncolsY: Long, strideRowX: Long, strideColDst: Long, cc: Int, nsm: Int, smpbo: Long, warpSize: Int, stream: MutPtr)
}

public fun mmqLauncher(dtype: GgmlDType): MmqLauncher? {
    val f: MmqLauncher =
        when (dtype) {
            GgmlDType.q40 -> ffi.launchMmqGgufQ40
            GgmlDType.q41 -> ffi.launchMmqGgufQ41
            GgmlDType.q50 -> ffi.launchMmqGgufQ50
            GgmlDType.q51 -> ffi.launchMmqGgufQ51
            GgmlDType.q80 -> ffi.launchMmqGgufQ80
            GgmlDType.q2k -> ffi.launchMmqGgufQ2K
            GgmlDType.q3k -> ffi.launchMmqGgufQ3K
            GgmlDType.q4k -> ffi.launchMmqGgufQ4K
            GgmlDType.q5k -> ffi.launchMmqGgufQ5K
            GgmlDType.q6k -> ffi.launchMmqGgufQ6K
            else -> {
                return null
            }
        }
    return f
}

// ---------------------------------------------------------------------------
// Per-device workspaces (grows-only, reused across calls).
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

public typealias WsMap = Mutex<Map<DeviceId, Mutex<WorkspaceSlot>>>

public var MMQ_WORKSPACE: OnceLock<WsMap> = OnceLock.new()

public var FIXUP_WORKSPACE: OnceLock<WsMap> = OnceLock.new()

public fun workspaceEnsure(ws: OnceLock<WsMap>, dev: CudaDevice, bytes: Int): Result<Pair<ULong, MutexGuard<WorkspaceSlot>>> {
    val map = ws.getOrInit({ Mutex.new(HashMap.new()) })
    val deviceKey = dev.id()
    val deviceMtx: Mutex<WorkspaceSlot> =
        run {
            val guard = map.lock().unwrap()
            when (val matchVal = guard.get(deviceKey).copied()) {
                null -> {
                    val slice =
                        when (val tryResult = dev.alloc<UByte>(bytes.max(1))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val leaked = Box.leak(Mutex.new(WorkspaceSlot(slice = slice, cap = bytes.max(1))))
                    guard.insert(deviceKey, leaked)
                    leaked
                }
                else -> {
                    val mtx = matchVal
                    mtx
                }
            }
        }
    val slot = deviceMtx.lock().unwrap()
    if (slot.cap < bytes) {
        slot.slice =
            when (val tryResult = dev.alloc<UByte>(bytes)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        slot.cap = bytes
    }
    val ptr = slot.slice.devicePtr(slot.slice.stream()).first
    return Ok(Pair(ptr, slot))
}

// ---------------------------------------------------------------------------
// Per-device info cache (compute capability, SM count, etc.)
// ---------------------------------------------------------------------------
public data class DeviceInfo(
    public var cc: Int = 0,
    public var nsm: Int = 0,
    public var smpbo: Long = 0L,
    public var warpSize: Int = 0,
) {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<DeviceInfo> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<DeviceInfo> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 24L
                override fun readNative(addr: Long): DeviceInfo =
                    DeviceInfo(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).readNative(addr + 4L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).readNative(addr + 16L),
                    )
                override fun writeNative(addr: Long, value: DeviceInfo) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).writeNative(addr, value.cc)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).writeNative(addr + 4L, value.nsm)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I64).writeNative(addr + 8L, value.smpbo)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.I32).writeNative(addr + 16L, value.warpSize)
                }
            }
    }
}

public var DEVICE_INFO: OnceLock<Mutex<Map<DeviceId, DeviceInfo>>> = OnceLock.new()

public fun getDeviceInfo(dev: CudaDevice): DeviceInfo {
    val map = DEVICE_INFO.getOrInit({ Mutex.new(HashMap.new()) })
    val key = dev.id()
    val guard = map.lock().unwrap()
    when (val matchVal = guard.get(key)) {
        null -> {}
        else -> {
            val info = matchVal
            return info
        }
    }
    val cuDevice = dev.cudaStream().context().cuDevice()
    val major = result.device.getAttribute(cuDevice, sys.CudeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).unwrapOr(8)
    val minor = result.device.getAttribute(cuDevice, sys.CudeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR).unwrapOr(0)
    val nsm = result.device.getAttribute(cuDevice, sys.CudeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrapOr(1)
    val smpbo = result.device.getAttribute(cuDevice, sys.CudeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN).unwrapOr(49152)
    val warpSize = result.device.getAttribute(cuDevice, sys.CudeviceAttribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrapOr(32)
    val info1 = DeviceInfo(cc = major * 100 + minor * 10, nsm = nsm, smpbo = smpbo.toLong(), warpSize = warpSize)
    guard.insert(key, info1)
    return info1
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Try the fast MMQ path. Returns `Ok(None)` when the fast path is not applicable:
 * - unsupported quant dtype
 * - batch size 0
 * - non-BF16/F16/F32 input
 * - non-contiguous input
 */
public fun tryFwd(qstorage: QCudaStorage, selfShape: Shape, rhs: CudaStorage, rhsL: Layout): Result<Pair<CudaStorage, Shape>?> {
    // Stream-k fixup workspace
    val MMQ_X_MAX: Int = 128
    val MMQ_Y_MAX: Int = 128
    val MAX_SMS: Int = 256
    val wDtype = qstorage.dtype()
    if (!supports(wDtype)) {
        return Ok(null)
    }
    val inputDtype = rhs.dtype()
    if (!when (inputDtype) {
            DType.bf16, DType.f16, DType.f32 -> true
            else -> false
        }
    ) {
        return Ok(null)
    }
    val (nrows, ncols) =
        when (val tryResult = selfShape.dims2()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
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
                    return Ok(null)
                }
            }
        }
    if (ncols != k2) {
        return Ok(null)
    }
    if (bSize == 0) {
        return Ok(null)
    }
    val qk = qkFor(wDtype)
    if (k2 % qk != 0) {
        return Ok(null)
    }
    val (o1, o2) =
        when (val matchVal1 = rhsL.contiguousOffsets()) {
            null -> {
                return Ok(null)
            }
            else -> {
                val offsets = matchVal1
                offsets
            }
        }
    val dev = qstorage.device()
    val stream = dev.cudaStream()
    val streamPtr = mutPtrFromRustCast(stream.cuStream())
    // MMQ quantize expects f32 input. Convert if needed.
    val rhsF32Storage: CudaStorage? =
        if (inputDtype != DType.F32) {
            val layoutForCast = Layout.contiguousWithOffset(rhsL.shape(), rhsL.startOffset())
            when (val tryResult = rhs.toDtype(layoutForCast, DType.F32)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        } else {
            null
        }
    // Get the f32 slice (either original or converted)
    val rhsF32Ref = (rhsF32Storage ?: rhs)
    val rhsF32Slice =
        when (val tryResult = rhsF32Ref.asCudaSlice<Float>()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    // For the converted case, offset is 0 (toDtype returns contiguous from 0).
    // For the original f32 case, apply the original offsets.
    val rhsF32Slice1 =
        if (rhsF32Storage != null) {
            rhsF32Slice.slice(0 until bSize * k2)
        } else {
            rhsF32Slice.slice(o1 until o2)
        }
    val rhsPtr = constPtrFromRustCast(rhsF32Slice1.devicePtr(stream).first)
    // Compute padded dimensions
    val kPadded = pad(k2, MATRIX_ROW_PADDING)
    // Must also be multiple of 4*QK8_1 = 128 for blockQ81Mmq
    val kPadded1 = pad(kPadded, 4 * QK8_1)
    // Workspace for blockQ81Mmq quantized activations
    val blocksPerRow = kPadded1 / (4 * QK8_1)
    val workspaceMain = bSize * blocksPerRow * BLOCK_Q8_1_MMQ_SIZE
    // Extra padding for mmqXMax (128 for MMA path)
    val workspaceExtra = 128 * BLOCK_Q8_1_MMQ_SIZE
    val workspaceBytes = workspaceMain + workspaceExtra
    val (scratchPtr, workspaceGuard) =
        when (val tryResult = workspaceEnsure(MMQ_WORKSPACE, dev, workspaceBytes)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val scratchPtr1 = mutPtrFromRustCast(scratchPtr)
    val fixupBytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * sizeOf<Float>(typeLayoutT = PrimitiveLayouts.F32)
    val (fixupPtr, fixupGuard) =
        when (val tryResult = workspaceEnsure(FIXUP_WORKSPACE, dev, fixupBytes)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val fixupPtr1 = mutPtrFromRustCast(fixupPtr)
    val weightPtr =
        constPtrFromRustCast(
            when (val tryResult = qstorage.devicePtr()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    val strideRowX = (k2 / qk).toLong()
    val di = getDeviceInfo(dev)
    val out =
        when (val tryResult = dev.alloc<Float>(nrows * bSize)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val strideColDst = nrows.toLong()
    val outPtr = mutPtrFromRustCast(out.devicePtr(stream).first)
    run {
        val quantize = quantizeLauncher(dsLayoutFor(wDtype))
        quantize(rhsPtr, ConstPtr(0L), scratchPtr1, 0, k2.toLong(), k2.toLong(), 0, 0, kPadded1.toLong(), bSize.toLong(), 1, 1, streamPtr)
        val launcher = mmqLauncher(wDtype).expect("supports() checked")
        launcher(fixupPtr1, weightPtr, ConstPtr(scratchPtr1.rawAddress, scratchPtr1.rawMeta), outPtr, k2.toLong(), nrows.toLong(), bSize.toLong(), strideRowX, strideColDst, di.cc, di.nsm, di.smpbo, di.warpSize, streamPtr)
    }
    val outShape = rhsL.shape().dims().toVec()
    outShape.pop()
    outShape.push(nrows)
    val outStorage = CudaStorage.wrapCudaSlice(out, dev.clone())
    return if (inputDtype == DType.F32) {
        Ok(Pair(outStorage, outShape.into()))
    } else {
        // Cast f32 output back to input dtype
        val outLayout = Layout.contiguous(Shape.from(outShape.clone()))
        val castStorage =
            when (val tryResult = outStorage.toDtype(outLayout, inputDtype)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        Ok(Pair(castStorage, outShape.into()))
    }
}
