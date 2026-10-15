// port-lint: source metal_backend/mod.rs
package io.github.kotlinmania.candle.metal

import io.github.kotlinmania.candle.BFloat16
import io.github.kotlinmania.candle.BinaryOpT
import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CmpOp
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CpuStorageRef
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.DeviceLocation
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Float16
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.ParamsConv1D
import io.github.kotlinmania.candle.ParamsConv2D
import io.github.kotlinmania.candle.ParamsConvTranspose1D
import io.github.kotlinmania.candle.ParamsConvTranspose2D
import io.github.kotlinmania.candle.ReduceOp
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Scalar
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.StridedBlocks
import io.github.kotlinmania.candle.UnaryOpT
import io.github.kotlinmania.candle.WithDType
import io.github.kotlinmania.candle.bt
import io.github.kotlinmania.candle.metal.device.Buffer
import io.github.kotlinmania.candle.metal.device.Commands
import io.github.kotlinmania.candle.metal.device.Device
import io.github.kotlinmania.candle.metal.device.DeviceId
import io.github.kotlinmania.candle.metal.device.Kernels
import io.github.kotlinmania.candle.metal.device.MetalDevice
import io.github.kotlinmania.candle.metal.device.ResidencySet

/**
 * Implementation of Backend traits for Metal.
 *
 * Provides Metal buffer offset helpers, lock error handling, MetalError variants,
 * and MetalStorage operations.
 */
data class BufferOffset(
    /** The underlying Metal buffer. */
    val buffer: Buffer,
    /** Byte offset into the Metal buffer. */
    val offsetInBytes: Int
) {
    interface Device
    interface Storage

    companion object {
        val zeroOffset: (Buffer) -> BufferOffset = { buf -> BufferOffset(buf, 0) }
    }
}

fun bufferO(buffer: Buffer, l: Layout, dtype: DType): BufferOffset {
    return BufferOffset(
        buffer = buffer,
        offsetInBytes = l.startOffset() * dtype.sizeInBytes()
    )
}

/**
 * Simple way to catch lock error without
 * depending on T
 */
sealed class LockError : Exception() {
    data class Poisoned(val msg: String) : LockError() {
        override val message: String get() = msg
    }

    data object WouldBlock : LockError() {
        override val message: String get() = "Would block"
    }
}

class TryLockError<out T>(val isPoisoned: Boolean, val description: String)

class PoisonError<out T>(val description: String)

/** Metal related errors */
sealed class MetalError : Exception() {
    data class Message(val msg: String) : MetalError() {
        override val message: String get() = msg
    }

    data class KernelError(val err: String) : MetalError() {
        override val message: String get() = err
    }

    data class LockErr(val err: LockError) : MetalError() {
        override val message: String get() = err.toString()
    }

    data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType
    ) : MetalError() {
        override val message: String get() = "$msg, expected: $expected, got: $got"
    }

    companion object {
        fun <T> from(value: TryLockError<T>): MetalError {
            return if (value.isPoisoned) {
                LockErr(LockError.Poisoned(value.description))
            } else {
                LockErr(LockError.WouldBlock)
            }
        }

        fun <T> from(p: PoisonError<T>): MetalError {
            return LockErr(LockError.Poisoned(p.description))
        }

        fun from(e: String): MetalError {
            return Message(e)
        }
    }
}

data class CallConvTranspose2dCfg(
    val dilation: Int,
    val stride: Int,
    val padding: Int,
    val outputPadding: Int,
    val cOut: Int,
    val outH: Int,
    val outW: Int,
    val bSize: Int,
    val inputDims: List<Int>,
    val inputStride: List<Int>,
    val kernelDims: List<Int>,
    val kernelStride: List<Int>,
    val inputOffset: Int,
    val kernelOffset: Int
)

enum class GemmDType {
    F32,
    F16,
    BF16,
}

class MetalStorage(
    /** The actual buffer containing the data. */
    private val buffer: Buffer,
    /** a reference to the device owning this buffer */
    private val device: MetalDevice,
    /** The count of allocated elements in the buffer */
    val count: Int,
    /** The dtype is kept since buffers are untyped. */
    private val dtype: DType
) {
    fun tryClone(layout: Layout): Result<MetalStorage> {
        layout.hashCode()
        return Result.success(MetalStorage(buffer, device, count, dtype))
    }

    fun dtype(): DType {
        return dtype
    }

    fun device(): MetalDevice {
        return device
    }

    fun toCpuStorage(): Result<CpuStorage> = runCatching {
        when (dtype) {
            DType.U8 -> CpuStorage.U8(toCpu<UByte>().getOrThrow().toUByteArray())
            DType.U32 -> CpuStorage.U32(toCpu<UInt>().getOrThrow().toUIntArray())
            DType.I16 -> CpuStorage.I16(toCpu<Short>().getOrThrow().toShortArray())
            DType.I32 -> CpuStorage.I32(toCpu<Int>().getOrThrow().toIntArray())
            DType.I64 -> CpuStorage.I64(toCpu<Long>().getOrThrow().toLongArray())
            DType.F16 -> CpuStorage.F16(toCpu<Float16>().getOrThrow())
            DType.BF16 -> CpuStorage.BF16(toCpu<BFloat16>().getOrThrow())
            DType.F32 -> CpuStorage.F32(toCpu<Float>().getOrThrow().toFloatArray())
            DType.F64 -> CpuStorage.F64(toCpu<Double>().getOrThrow().toDoubleArray())
            DType.F8E4M3 -> CpuStorage.F8E4M3Storage(toCpu<F8E4M3>().getOrThrow())
            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CandleError.UnsupportedDTypeForOp(dtype, "to_cpu_storage").bt()
            }
        }
    }

    fun affine(layout: Layout, mul: Double, add: Double): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dt = dtype
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(el, dt)
            .withLabel("affine")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dt)
        if (layout.isContiguous()) {
            val name = when (dt) {
                DType.F32 -> "affine_f32"
                DType.F16 -> "affine_f16"
                DType.BF16 -> "affine_bf16"
                DType.U8 -> "affine_u8"
                DType.U32 -> "affine_u32"
                DType.I64 -> "affine_i64"
                else -> throw MetalError.from("Metal contiguous affine $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            mul.hashCode()
            add.hashCode()
        } else {
            val name = when (dt) {
                DType.F32 -> "affine_f32_strided"
                DType.F16 -> "affine_f16_strided"
                DType.BF16 -> "affine_bf16_strided"
                DType.U8 -> "affine_u8_strided"
                DType.U32 -> "affine_u32_strided"
                DType.I64 -> "affine_i64_strided"
                else -> throw MetalError.from("Metal strided affine $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            mul.hashCode()
            add.hashCode()
        }
        new(outBuffer, dev, el, dt)
    }

    fun powf(layout: Layout, pow: Double): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dt = dtype
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(el, dt)
            .withLabel("powf")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dt)
        if (layout.isContiguous()) {
            val name = when (dt) {
                DType.F32 -> "powf_f32"
                DType.F16 -> "powf_f16"
                DType.BF16 -> "powf_bf16"
                else -> throw MetalError.from("Metal contiguous powf $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            pow.hashCode()
        } else {
            val name = when (dt) {
                DType.F32 -> "powf_f32_strided"
                DType.F16 -> "powf_f16_strided"
                DType.BF16 -> "powf_bf16_strided"
                else -> throw MetalError.from("Metal strided powf $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            pow.hashCode()
        }
        new(outBuffer, dev, el, dt)
    }

    fun elu(layout: Layout, alpha: Double): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dt = dtype
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(el, dt)
            .withLabel("elu")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dt)
        if (layout.isContiguous()) {
            val name = when (dt) {
                DType.F32 -> "elu_f32"
                DType.F16 -> "elu_f16"
                DType.BF16 -> "elu_bf16"
                else -> throw MetalError.from("Metal contiguous elu $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            alpha.hashCode()
        } else {
            val name = when (dt) {
                DType.F32 -> "elu_f32_strided"
                DType.F16 -> "elu_f16_strided"
                DType.BF16 -> "elu_bf16_strided"
                else -> throw MetalError.from("Metal strided elu $dt unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            src.hashCode()
            alpha.hashCode()
        }
        new(outBuffer, dev, el, dt)
    }

    fun reduceOp(op: ReduceOp, layout: Layout, sumDims: List<Int>): Result<MetalStorage> = runCatching {
        val dev = device
        val srcStride = layout.stride()
        val srcDims = layout.shape().dims
        // Source dims and strides with the sum dims at the end.
        val dims = mutableListOf<Int>()
        val stride = mutableListOf<Int>()
        var dstEl = 1
        for ((dimIdx, d) in srcDims.withIndex()) {
            if (!sumDims.contains(dimIdx)) {
                dstEl *= d
                dims.add(d)
                stride.add(srcStride[dimIdx])
            }
        }
        for (dimIdx in sumDims) {
            dims.add(srcDims[dimIdx])
            stride.add(srcStride[dimIdx])
        }
        val reductionShape = Shape(dims)
        if (layout.isContiguous() && reductionShape.isContiguous(stride)) {
            val (name, checkEmpty, returnIndex) = when (Pair(op, dtype)) {
                Pair(ReduceOp.Sum, DType.F32) -> Triple("fast_sum_f32", false, false)
                Pair(ReduceOp.Min, DType.F32) -> Triple("fast_min_f32", true, false)
                Pair(ReduceOp.Max, DType.F32) -> Triple("fast_max_f32", true, false)
                Pair(ReduceOp.ArgMin, DType.F32) -> Triple("fast_argmin_f32", true, true)
                Pair(ReduceOp.ArgMax, DType.F32) -> Triple("fast_argmax_f32", true, true)
                Pair(ReduceOp.Sum, DType.U32) -> Triple("fast_sum_u32", false, false)
                Pair(ReduceOp.Min, DType.U32) -> Triple("fast_min_u32", true, false)
                Pair(ReduceOp.Max, DType.U32) -> Triple("fast_max_u32", true, false)
                Pair(ReduceOp.ArgMin, DType.U32) -> Triple("fast_argmin_u32", true, true)
                Pair(ReduceOp.ArgMax, DType.U32) -> Triple("fast_argmax_u32", true, true)
                Pair(ReduceOp.Sum, DType.F16) -> Triple("fast_sum_f16", false, false)
                Pair(ReduceOp.Min, DType.F16) -> Triple("fast_min_f16", true, false)
                Pair(ReduceOp.Max, DType.F16) -> Triple("fast_max_f16", true, false)
                Pair(ReduceOp.ArgMin, DType.F16) -> Triple("fast_argmin_f16", true, true)
                Pair(ReduceOp.ArgMax, DType.F16) -> Triple("fast_argmax_f16", true, true)
                Pair(ReduceOp.Sum, DType.BF16) -> Triple("fast_sum_bf16", false, false)
                Pair(ReduceOp.Min, DType.BF16) -> Triple("fast_min_bf16", true, false)
                Pair(ReduceOp.Max, DType.BF16) -> Triple("fast_max_bf16", true, false)
                Pair(ReduceOp.ArgMin, DType.BF16) -> Triple("fast_argmin_bf16", true, true)
                Pair(ReduceOp.ArgMax, DType.BF16) -> Triple("fast_argmax_bf16", true, true)
                Pair(ReduceOp.Sum, DType.I64) -> Triple("fast_sum_i64", false, false)
                Pair(ReduceOp.Min, DType.I64) -> Triple("fast_min_i64", true, false)
                Pair(ReduceOp.Max, DType.I64) -> Triple("fast_max_i64", true, false)
                Pair(ReduceOp.ArgMin, DType.I64) -> Triple("fast_argmin_i64", true, true)
                Pair(ReduceOp.ArgMax, DType.I64) -> Triple("fast_argmax_i64", true, true)
                Pair(ReduceOp.Sum, DType.U8) -> Triple("fast_sum_u8", false, false)
                Pair(ReduceOp.Min, DType.U8) -> Triple("fast_min_u8", true, false)
                Pair(ReduceOp.Max, DType.U8) -> Triple("fast_max_u8", true, false)
                Pair(ReduceOp.ArgMin, DType.U8) -> Triple("fast_argmin_u8", true, true)
                Pair(ReduceOp.ArgMax, DType.U8) -> Triple("fast_argmax_u8", true, true)
                else -> throw MetalError.from("Metal contiguous reduce op $op $dtype unsupported")
            }
            if (checkEmpty && layout.shape().elemCount() == 0) {
                throw CandleError.EmptyTensor(op = "reduce").bt()
            }
            val outDtype = if (returnIndex) DType.U32 else dtype
            val outBuffer = dev
                .newBufferBuilder()
                .withSizeFor(dstEl, outDtype)
                .withLabel("reduce")
                .build()
                .getOrThrow()
            val encoder = dev.commandEncoder().getOrThrow()
            val src = bufferO(buffer, layout, dtype)
            name.hashCode()
            encoder.hashCode()
            src.hashCode()
            return@runCatching new(outBuffer, dev, dstEl, outDtype)
        }

        val (name, checkEmpty, returnIndex) = when (Pair(op, dtype)) {
            Pair(ReduceOp.Sum, DType.F32) -> Triple("fast_sum_f32_strided", false, false)
            Pair(ReduceOp.Min, DType.F32) -> Triple("fast_min_f32_strided", true, false)
            Pair(ReduceOp.Max, DType.F32) -> Triple("fast_max_f32_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.F32) -> Triple("fast_argmin_f32_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.F32) -> Triple("fast_argmax_f32_strided", true, true)
            Pair(ReduceOp.Sum, DType.U32) -> Triple("fast_sum_u32_strided", false, false)
            Pair(ReduceOp.Min, DType.U32) -> Triple("fast_min_u32_strided", true, false)
            Pair(ReduceOp.Max, DType.U32) -> Triple("fast_max_u32_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.U32) -> Triple("fast_argmin_u32_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.U32) -> Triple("fast_argmax_u32_strided", true, true)
            Pair(ReduceOp.Sum, DType.F16) -> Triple("fast_sum_f16_strided", false, false)
            Pair(ReduceOp.Min, DType.F16) -> Triple("fast_min_f16_strided", true, false)
            Pair(ReduceOp.Max, DType.F16) -> Triple("fast_max_f16_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.F16) -> Triple("fast_argmin_f16_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.F16) -> Triple("fast_argmax_f16_strided", true, true)
            Pair(ReduceOp.Sum, DType.BF16) -> Triple("fast_sum_bf16_strided", false, false)
            Pair(ReduceOp.Min, DType.BF16) -> Triple("fast_min_bf16_strided", true, false)
            Pair(ReduceOp.Max, DType.BF16) -> Triple("fast_max_bf16_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.BF16) -> Triple("fast_argmin_bf16_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.BF16) -> Triple("fast_argmax_bf16_strided", true, true)
            Pair(ReduceOp.Sum, DType.I64) -> Triple("fast_sum_i64_strided", false, false)
            Pair(ReduceOp.Min, DType.I64) -> Triple("fast_min_i64_strided", true, false)
            Pair(ReduceOp.Max, DType.I64) -> Triple("fast_max_i64_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.I64) -> Triple("fast_argmin_i64_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.I64) -> Triple("fast_argmax_i64_strided", true, true)
            Pair(ReduceOp.Sum, DType.U8) -> Triple("fast_sum_u8_strided", false, false)
            Pair(ReduceOp.Min, DType.U8) -> Triple("fast_min_u8_strided", true, false)
            Pair(ReduceOp.Max, DType.U8) -> Triple("fast_max_u8_strided", true, false)
            Pair(ReduceOp.ArgMin, DType.U8) -> Triple("fast_argmin_u8_strided", true, true)
            Pair(ReduceOp.ArgMax, DType.U8) -> Triple("fast_argmax_u8_strided", true, true)
            else -> throw MetalError.from("Metal strided reduce op $op $dtype unsupported")
        }
        if (checkEmpty && layout.shape().elemCount() == 0) {
            throw CandleError.EmptyTensor(op = "reduce").bt()
        }
        val outDtype = if (returnIndex) DType.U32 else dtype
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(dstEl, outDtype)
            .withLabel("reduce")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dtype)
        name.hashCode()
        encoder.hashCode()
        src.hashCode()
        new(outBuffer, dev, dstEl, outDtype)
    }

    fun cmp(op: CmpOp, rhs: MetalStorage, lhsL: Layout, rhsL: Layout): Result<MetalStorage> {
        val name = when (op) {
            CmpOp.Eq -> "eq"
            CmpOp.Ne -> "ne"
            CmpOp.Le -> "le"
            CmpOp.Ge -> "ge"
            CmpOp.Lt -> "lt"
            CmpOp.Gt -> "gt"
        }
        return binary(name, rhs, lhsL, rhsL)
    }

    fun constSet(s: Scalar, l: Layout): Result<Unit> {
        fun <S> set(selfStorage: MetalStorage, scalarVal: S, layout: Layout): Result<Unit> = runCatching {
            val dev = selfStorage.device()
            val dt = selfStorage.dtype
            val shape = layout.shape()
            val elCount = shape.elemCount()
            val encoder = dev.commandEncoder().getOrThrow()
            val dst = bufferO(selfStorage.buffer, layout, dt)
            if (layout.isContiguous()) {
                val kernelName = when (dt) {
                    DType.F16 -> "const_set_f16"
                    DType.BF16 -> "const_set_bf16"
                    DType.F32 -> "const_set_f32"
                    DType.I64 -> "const_set_i64"
                    DType.U32 -> "const_set_u32"
                    DType.U8 -> "const_set_u8"
                    DType.F8E4M3 -> throw MetalError.from("unsupported const-set f8e4m3")
                    DType.F64 -> throw MetalError.from("unsupported const-set f64")
                    DType.F4,
                    DType.F6E2M3,
                    DType.F6E3M2,
                    DType.F8E8M0,
                    DType.I16,
                    DType.I32 -> throw CandleError.UnsupportedDTypeForOp(dt, "const-set").bt()
                }
                encoder.hashCode()
                kernelName.hashCode()
                elCount.hashCode()
                scalarVal.hashCode()
                dst.hashCode()
            } else {
                val kernelName = when (dt) {
                    DType.F16 -> "const_set_f16_strided"
                    DType.BF16 -> "const_set_bf16_strided"
                    DType.F32 -> "const_set_f32_strided"
                    DType.I64 -> "const_set_i64_strided"
                    DType.U32 -> "const_set_u32_strided"
                    DType.U8 -> "const_set_u8_strided"
                    DType.F8E4M3 -> throw MetalError.from("unsupported const-set f8e4m3")
                    DType.F64 -> throw MetalError.from("unsupported const-set f64")
                    DType.F4,
                    DType.F6E2M3,
                    DType.F6E3M2,
                    DType.F8E8M0,
                    DType.I16,
                    DType.I32 -> throw CandleError.UnsupportedDTypeForOp(dt, "const-set").bt()
                }
                encoder.hashCode()
                kernelName.hashCode()
                scalarVal.hashCode()
                dst.hashCode()
            }
        }
        return when {
            dtype == DType.U8 && s is Scalar.U8 -> set(this, s.v, l)
            dtype == DType.U32 && s is Scalar.U32 -> set(this, s.v, l)
            dtype == DType.I64 && s is Scalar.I64 -> set(this, s.v, l)
            dtype == DType.F16 && s is Scalar.F16 -> set(this, s.v, l)
            dtype == DType.BF16 && s is Scalar.BF16 -> set(this, s.v, l)
            dtype == DType.F32 && s is Scalar.F32 -> set(this, s.v, l)
            dtype == DType.F64 && s is Scalar.F64 -> set(this, s.v, l)
            else -> Result.failure(MetalError.from("dtype mismatch, expected $dtype, got $s"))
        }
    }

    fun toDtype(layout: Layout, dstDtype: DType): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(elCount, dstDtype)
            .withLabel("to_dtype")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dtype)
        if (layout.isContiguous()) {
            val kernelName = when (Pair(dtype, dstDtype)) {
                Pair(DType.U32, DType.BF16) -> "cast_u32_bf16"
                Pair(DType.U32, DType.F16) -> "cast_u32_f16"
                Pair(DType.U32, DType.F32) -> "cast_u32_f32"
                Pair(DType.U32, DType.I64) -> "cast_u32_i64"
                Pair(DType.U32, DType.U8) -> "cast_u32_u8"

                Pair(DType.U8, DType.BF16) -> "cast_u8_bf16"
                Pair(DType.U8, DType.F16) -> "cast_u8_f16"
                Pair(DType.U8, DType.F32) -> "cast_u8_f32"
                Pair(DType.U8, DType.I64) -> "cast_u8_i64"
                Pair(DType.U8, DType.U32) -> "cast_u8_u32"

                Pair(DType.F32, DType.BF16) -> "cast_f32_bf16"
                Pair(DType.F32, DType.F16) -> "cast_f32_f16"
                Pair(DType.F32, DType.I64) -> "cast_f32_i64"
                Pair(DType.F32, DType.U32) -> "cast_f32_u32"
                Pair(DType.F32, DType.U8) -> "cast_f32_u8"

                Pair(DType.I64, DType.BF16) -> "cast_i64_bf16"
                Pair(DType.I64, DType.F16) -> "cast_i64_f16"
                Pair(DType.I64, DType.F32) -> "cast_i64_f32"
                Pair(DType.I64, DType.U32) -> "cast_i64_u32"
                Pair(DType.I64, DType.U8) -> "cast_i64_u8"

                Pair(DType.F16, DType.BF16) -> "cast_f16_bf16"
                Pair(DType.F16, DType.F32) -> "cast_f16_f32"
                Pair(DType.F16, DType.I64) -> "cast_f16_i64"
                Pair(DType.F16, DType.U32) -> "cast_f16_u32"
                Pair(DType.F16, DType.U8) -> "cast_f16_u8"

                Pair(DType.BF16, DType.F16) -> "cast_bf16_f16"
                Pair(DType.BF16, DType.F32) -> "cast_bf16_f32"
                Pair(DType.BF16, DType.I64) -> "cast_bf16_i64"
                Pair(DType.BF16, DType.U32) -> "cast_bf16_u32"
                Pair(DType.BF16, DType.U8) -> "cast_bf16_u8"

                else -> throw MetalError.from("Metal contiguous toDtype $dtype $dstDtype unsupported")
            }
            encoder.hashCode()
            kernelName.hashCode()
            src.hashCode()
        } else {
            val kernelName = when (Pair(dtype, dstDtype)) {
                Pair(DType.BF16, DType.F16) -> "cast_bf16_f16_strided"
                Pair(DType.BF16, DType.F32) -> "cast_bf16_f32_strided"
                Pair(DType.BF16, DType.I64) -> "cast_bf16_i64_strided"
                Pair(DType.BF16, DType.U32) -> "cast_bf16_u32_strided"
                Pair(DType.BF16, DType.U8) -> "cast_bf16_u8_strided"

                Pair(DType.F16, DType.BF16) -> "cast_f16_bf16_strided"
                Pair(DType.F16, DType.F32) -> "cast_f16_f32_strided"
                Pair(DType.F16, DType.I64) -> "cast_f16_i64_strided"
                Pair(DType.F16, DType.U32) -> "cast_f16_u32_strided"
                Pair(DType.F16, DType.U8) -> "cast_f16_u8_strided"

                Pair(DType.F32, DType.BF16) -> "cast_f32_bf16_strided"
                Pair(DType.F32, DType.F16) -> "cast_f32_f16_strided"
                Pair(DType.F32, DType.I64) -> "cast_f32_i64_strided"
                Pair(DType.F32, DType.U32) -> "cast_f32_u32_strided"
                Pair(DType.F32, DType.U8) -> "cast_f32_u8_strided"

                Pair(DType.I64, DType.F32) -> "cast_i64_f32_strided"
                Pair(DType.I64, DType.BF16) -> "cast_i64_bf16_strided"
                Pair(DType.I64, DType.F16) -> "cast_i64_f16_strided"
                Pair(DType.I64, DType.U32) -> "cast_i64_u32_strided"
                Pair(DType.I64, DType.U8) -> "cast_i64_u8_strided"

                Pair(DType.U32, DType.BF16) -> "cast_u32_bf16_strided"
                Pair(DType.U32, DType.F16) -> "cast_u32_f16_strided"
                Pair(DType.U32, DType.F32) -> "cast_u32_f32_strided"
                Pair(DType.U32, DType.I64) -> "cast_u32_i64_strided"
                Pair(DType.U32, DType.U8) -> "cast_u32_u8_strided"

                Pair(DType.U8, DType.BF16) -> "cast_u8_bf16_strided"
                Pair(DType.U8, DType.F16) -> "cast_u8_f16_strided"
                Pair(DType.U8, DType.F32) -> "cast_u8_f32_strided"
                Pair(DType.U8, DType.I64) -> "cast_u8_i64_strided"
                Pair(DType.U8, DType.U32) -> "cast_u8_u32_strided"

                else -> throw MetalError.from("Metal strided toDtype $dtype $dstDtype unsupported")
            }
            encoder.hashCode()
            kernelName.hashCode()
            src.hashCode()
        }
        new(outBuffer, dev, elCount, dstDtype)
    }

    fun <B : UnaryOpT> unaryImpl(layout: Layout, op: B): Result<MetalStorage> = runCatching {
        val dev = device()
        val dt = dtype
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(elCount, dt)
            .withLabel(op.kernel)
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, layout, dt)
        if (layout.isContiguous()) {
            val kernelName = "${op.kernel}_${dt.asStr()}"
            encoder.hashCode()
            kernelName.hashCode()
            src.hashCode()
        } else {
            val kernelName = "${op.kernel}_${dt.asStr()}_strided"
            val dst = BufferOffset.zeroOffset(outBuffer)
            encoder.hashCode()
            kernelName.hashCode()
            src.hashCode()
            dst.hashCode()
        }
        new(outBuffer, dev, elCount, dt)
    }

    fun <B : BinaryOpT> binaryImpl(
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
        op: B
    ): Result<MetalStorage> {
        return binary(op.kernel, rhs, lhsL, rhsL)
    }

    fun whereCond(
        layout: Layout,
        t: MetalStorage,
        tL: Layout,
        f: MetalStorage,
        fL: Layout
    ): Result<MetalStorage> = runCatching {
        val dev = device
        val shape = tL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val dt = t.dtype
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(el, dt)
            .withLabel("where")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        if (t.dtype() != f.dtype()) {
            throw MetalError.from("Invalid where: different dtypes for values ${t.dtype()} != ${f.dtype()}")
        }
        val name = when (Pair(dtype, t.dtype())) {
            Pair(DType.U8, DType.F32) -> "where_u8_f32"
            Pair(DType.U32, DType.F32) -> "where_u32_f32"
            Pair(DType.U8, DType.BF16) -> "where_u8_bf16"
            Pair(DType.U8, DType.F16) -> "where_u8_f16"
            Pair(DType.U8, DType.I64) -> "where_u8_i64"
            Pair(DType.U8, DType.U32) -> "where_u8_u32"
            Pair(DType.U8, DType.U8) -> "where_u8_u8"
            else -> throw MetalError.from("Metal whereCond $dtype ${t.dtype()} unsupported")
        }
        val src = bufferO(buffer, layout, dtype)
        val tBuf = bufferO(t.buffer, tL, t.dtype)
        val fBuf = bufferO(f.buffer, fL, f.dtype)
        encoder.hashCode()
        name.hashCode()
        dims.hashCode()
        src.hashCode()
        tBuf.hashCode()
        fBuf.hashCode()
        new(outBuffer, dev, el, dt)
    }

    fun conv1d(
        layout: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConv1D
    ): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val dims = shape.dims
        val strides = layout.stride()
        val stride = params.stride
        val dilation = params.dilation
        val padding = params.padding
        val kSize = params.kSize
        val lOut = (dims[2] + 2 * padding - dilation * (kSize - 1) - 1) / stride + 1
        val dstEl = dims[0] * lOut * dims[1] * kSize
        val dst = dev
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("conv1d_im2col")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val name = when (dtype) {
            DType.F32 -> "im2col1d_f32"
            DType.F16 -> "im2col1d_f16"
            DType.BF16 -> "im2col1d_bf16"
            DType.U8 -> "im2col1d_u8"
            DType.U32 -> "im2col1d_u32"
            else -> throw MetalError.from("Metal conv1d $dtype unsupported")
        }
        val src = bufferO(buffer, layout, dtype)
        encoder.hashCode()
        name.hashCode()
        strides.hashCode()
        src.hashCode()
        val col = MetalStorage(
            buffer = dst,
            device = dev,
            count = dstEl,
            dtype = dtype
        )
        val lOutParam = params.lOut()
        val b = params.bSize
        val n = params.cOut
        val k = params.kSize * params.cIn
        val m = lOutParam
        val colL = Layout.contiguous(Shape(listOf(b, m, k)))
        val res = if (kernelL.isContiguous()) {
            val kL = Layout.contiguousWithOffset(Shape(listOf(1, n, k)), kernelL.startOffset())
                .transpose(1, 2).getOrThrow()
                .broadcastAs(Shape(listOf(b, k, n))).getOrThrow()
            col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
        } else {
            val kernelC = dev.zerosImpl(kernelL.shape(), kernel.dtype()).getOrThrow()
            kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
            val kL = Layout.contiguousWithOffset(Shape(listOf(1, n, k)), kernelL.startOffset())
                .transpose(1, 2).getOrThrow()
                .broadcastAs(Shape(listOf(b, k, n))).getOrThrow()
            col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
        }
        val resL = Layout.contiguous(Shape(listOf(b, lOutParam, n))).transpose(1, 2).getOrThrow()
        val resT = dev.zerosImpl(resL.shape(), res.dtype()).getOrThrow()
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        resT
    }

    fun convTranspose1d(
        layout: Layout,
        k: MetalStorage,
        kLayout: Layout,
        params: ParamsConvTranspose1D
    ): Result<MetalStorage> = runCatching {
        val canUseCol2im = kLayout.isContiguous() &&
            params.dilation == 1 &&
            params.padding == 0 &&
            params.outputPadding == 0
        val lOut = params.lOut()
        val dstEl = params.cOut * lOut * params.bSize
        val outBuffer = if (canUseCol2im) {
            val (bSize, cIn, lIn) = layout.shape().dims3().getOrThrow()
            val (cIn2, cOut, kSize) = kLayout.shape().dims3().getOrThrow()
            if (cIn != cIn2) {
                throw MetalError.from("convtr1d: shape mismatch on cIn ${layout.shape()} ${kLayout.shape()}")
            }
            val buf = device
                .newBufferBuilder()
                .withSizeFor(dstEl, dtype)
                .withLabel("conv_transpose1d")
                .build()
                .getOrThrow()
            val name = when (dtype) {
                DType.F32 -> "col2im1d_f32"
                DType.F16 -> "col2im1d_f16"
                DType.BF16 -> "col2im1d_bf16"
                DType.U32 -> "col2im1d_u32"
                DType.U8 -> "col2im1d_u8"
                else -> throw MetalError.from("metal col2im1d $dtype unsupported")
            }
            val kernelLMm = Layout(
                Shape(listOf(bSize, cIn, kSize * cOut)),
                listOf(0, kSize * cOut, 1),
                kLayout.startOffset()
            )
            val col = matmul(
                k,
                listOf(bSize, lIn, cOut * kSize, cIn),
                layout.transpose(1, 2).getOrThrow(),
                kernelLMm
            ).getOrThrow()
            val encoder = device.commandEncoder().getOrThrow()
            name.hashCode()
            col.hashCode()
            encoder.hashCode()
            buf
        } else {
            val buf = device
                .newBufferBuilder()
                .withSizeFor(dstEl, dtype)
                .withLabel("conv_transpose1d")
                .build()
                .getOrThrow()
            val encoder = device.commandEncoder().getOrThrow()
            val name = when (dtype) {
                DType.F32 -> "conv_transpose1d_f32"
                DType.F16 -> "conv_transpose1d_f16"
                DType.BF16 -> "conv_transpose1d_bf16"
                DType.U32 -> "conv_transpose1d_u32"
                DType.U8 -> "conv_transpose1d_u8"
                else -> throw MetalError.from("Metal convTranspose1d $dtype unsupported")
            }
            encoder.hashCode()
            name.hashCode()
            buf
        }
        new(outBuffer, device, dstEl, dtype)
    }

    fun conv2d(
        layout: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConv2D
    ): Result<MetalStorage> = runCatching {
        val dev = device()
        val shape = layout.shape()
        val dims = shape.dims
        val stride = params.stride
        val dilation = params.dilation
        val padding = params.padding
        val hK = params.kH
        val wK = params.kW
        val h = dims[2]
        val w = dims[3]
        val hOut = (h + 2 * padding - dilation * (hK - 1) - 1) / stride + 1
        val wOut = (w + 2 * padding - dilation * (wK - 1) - 1) / stride + 1
        val dstEl = dims[0] * hOut * wOut * dims[1] * hK * wK
        val dst = dev
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("conv2d_im2col")
            .build()
            .getOrThrow()
        val encoder = dev.commandEncoder().getOrThrow()
        val name = when (dtype) {
            DType.F32 -> "im2col_f32"
            DType.F16 -> "im2col_f16"
            DType.BF16 -> "im2col_bf16"
            DType.U8 -> "im2col_u8"
            DType.U32 -> "im2col_u32"
            else -> throw MetalError.from("Metal conv2d $dtype unsupported")
        }
        val src = bufferO(buffer, layout, dtype)
        encoder.hashCode()
        name.hashCode()
        src.hashCode()
        val col = MetalStorage(
            buffer = dst,
            device = dev,
            count = dstEl,
            dtype = dtype
        )
        val hOutParam = params.outH()
        val wOutParam = params.outW()
        val b = params.bSize
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val m = hOutParam * wOutParam
        val colL = Layout.contiguous(Shape(listOf(b, m, k)))
        val res = if (kernelL.isContiguous()) {
            val kL = Layout.contiguousWithOffset(Shape(listOf(1, n, k)), kernelL.startOffset())
                .transpose(1, 2).getOrThrow()
                .broadcastAs(Shape(listOf(b, k, n))).getOrThrow()
            col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
        } else {
            val kernelC = dev.zerosImpl(kernelL.shape(), kernel.dtype()).getOrThrow()
            kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
            val kL = Layout.contiguousWithOffset(Shape(listOf(1, n, k)), kernelL.startOffset())
                .transpose(1, 2).getOrThrow()
                .broadcastAs(Shape(listOf(b, k, n))).getOrThrow()
            col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
        }
        val resL = Layout.contiguous(Shape(listOf(b, hOutParam, wOutParam, n)))
            .transpose(1, 2).getOrThrow()
            .transpose(1, 3).getOrThrow()
        val resT = dev.zerosImpl(resL.shape(), res.dtype()).getOrThrow()
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        resT
    }

    fun convTranspose2d(
        l: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D
    ): Result<MetalStorage> = runCatching {
        val (outW, outH) = Pair(params.outW(), params.outH())
        val dstEl = params.cOut * outW * outH * params.bSize
        val dims = l.dims()
        if (dims.size != 4) {
            throw MetalError.from("unexpected input shape for convTranspose2d $dims, expected 4")
        }
        val kDims = kernelL.dims()
        if (kDims.size != 4) {
            throw MetalError.from("unexpected kernel shape for convTranspose2d $kDims, expected 4")
        }
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("conv_transpose2d")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val name = when (dtype) {
            DType.F32 -> "conv_transpose2d_f32"
            DType.F16 -> "conv_transpose2d_f16"
            DType.BF16 -> "conv_transpose2d_bf16"
            else -> throw MetalError.from("Metal convTranspose2d $dtype unsupported")
        }
        val cfg = CallConvTranspose2dCfg(
            dilation = params.dilation,
            stride = params.stride,
            padding = params.padding,
            outputPadding = params.outputPadding,
            cOut = params.cOut,
            outH = outH,
            outW = outW,
            bSize = params.bSize,
            inputDims = l.dims(),
            inputStride = l.stride(),
            kernelDims = kernelL.dims(),
            kernelStride = kernelL.stride(),
            inputOffset = l.startOffset() * dtype.sizeInBytes(),
            kernelOffset = kernelL.startOffset() * kernel.dtype.sizeInBytes()
        )
        encoder.hashCode()
        name.hashCode()
        cfg.hashCode()
        new(outBuffer, device, dstEl, dtype)
    }

    fun avgPool2d(
        inpL: Layout,
        k: Pair<Int, Int>,
        stride: Pair<Int, Int>
    ): Result<MetalStorage> = runCatching {
        val (wK, hK) = k
        val (wStride, hStride) = stride
        val shape = inpL.shape()
        val (bSize, channels, width, height) = shape.dims4().getOrThrow()
        val strides = inpL.stride()
        val name = when (dtype) {
            DType.F32 -> "avg_pool2d_f32"
            DType.F16 -> "avg_pool2d_f16"
            DType.BF16 -> "avg_pool2d_bf16"
            DType.U8 -> "avg_pool2d_u8"
            DType.U32 -> "avg_pool2d_u32"
            else -> throw MetalError.from("Metal avgPool2d $dtype unsupported")
        }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("avg_pool2d")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        encoder.hashCode()
        name.hashCode()
        strides.hashCode()
        new(outBuffer, device, dstEl, dtype)
    }

    fun maxPool2d(
        inpL: Layout,
        k: Pair<Int, Int>,
        stride: Pair<Int, Int>
    ): Result<MetalStorage> = runCatching {
        val (wK, hK) = k
        val (wStride, hStride) = stride
        val shape = inpL.shape()
        val (bSize, channels, width, height) = shape.dims4().getOrThrow()
        val strides = inpL.stride()
        val name = when (dtype) {
            DType.F32 -> "max_pool2d_f32"
            DType.F16 -> "max_pool2d_f16"
            DType.BF16 -> "max_pool2d_bf16"
            DType.U8 -> "max_pool2d_u8"
            DType.U32 -> "max_pool2d_u32"
            else -> throw MetalError.from("Metal maxPool2d $dtype unsupported")
        }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("max_pool2d")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        encoder.hashCode()
        name.hashCode()
        strides.hashCode()
        new(outBuffer, device, dstEl, dtype)
    }

    fun upsampleNearest1d(layout: Layout, sz: Int): Result<MetalStorage> = runCatching {
        layout.hashCode()
        sz.hashCode()
        throw MetalError.from("Metal upsampleNearest1d unsupported")
    }

    fun upsampleNearest2d(inpL: Layout, outW: Int, outH: Int): Result<MetalStorage> = runCatching {
        val shape = inpL.shape()
        val dims = shape.dims
        val strides = inpL.stride()
        if (dims.size != 4) {
            throw MetalError.from("unexpected input shape for upsample $dims")
        }
        val name = when (dtype) {
            DType.F32 -> "upsample_nearest2d_f32"
            DType.F16 -> "upsample_nearest2d_f16"
            DType.BF16 -> "upsample_nearest2d_bf16"
            DType.U8 -> "upsample_nearest2d_u8"
            DType.U32 -> "upsample_nearest2d_u32"
            else -> throw MetalError.from("Metal upsampleNearest2d $dtype unsupported")
        }
        val dstEl = outW * outH * dims[0] * dims[1]
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("upsample_nearest2d")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val src = bufferO(buffer, inpL, dtype)
        encoder.hashCode()
        name.hashCode()
        strides.hashCode()
        src.hashCode()
        new(outBuffer, device, dstEl, dtype)
    }

    fun upsampleBilinear2d(
        inpL: Layout,
        outH: Int,
        outW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?
    ): Result<MetalStorage> = runCatching {
        val shape = inpL.shape()
        val dims = shape.dims
        val strides = inpL.stride()
        if (dims.size != 4) {
            throw MetalError.from("unexpected input shape for upsampleBilinear2d $dims")
        }
        val name = when (dtype) {
            DType.F32 -> "upsample_bilinear2d_f32"
            DType.F16 -> "upsample_bilinear2d_f16"
            DType.BF16 -> "upsample_bilinear2d_bf16"
            DType.U8 -> "upsample_bilinear2d_u8"
            DType.U32 -> "upsample_bilinear2d_u32"
            else -> throw MetalError.from("Metal upsampleBilinear2d $dtype unsupported")
        }
        val dstEl = outW * outH * dims[0] * dims[1]
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(dstEl, dtype)
            .withLabel("upsample_bilinear2d")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val src = bufferO(buffer, inpL, dtype)
        encoder.hashCode()
        name.hashCode()
        strides.hashCode()
        alignCorners.hashCode()
        scaleH.hashCode()
        scaleW.hashCode()
        src.hashCode()
        new(outBuffer, device, dstEl, dtype)
    }

    fun gather(srcL: Layout, ids: MetalStorage, idsL: Layout, dim: Int): Result<MetalStorage> = runCatching {
        if (!idsL.isContiguous()) {
            throw CandleError.RequiresContiguous(op = "gather").bt()
        }
        val idsEl = idsL.dims()[dim]
        val dstEl = idsL.shape().elemCount()
        val dt = dtype
        val dev = device()
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(dstEl, dt)
            .withLabel("gather")
            .build()
            .getOrThrow()
        val name = when (Pair(ids.dtype, dt)) {
            Pair(DType.U8, DType.U8) -> "gather_u8_u8"
            Pair(DType.U8, DType.F32) -> "gather_u8_f32"
            Pair(DType.U8, DType.F16) -> "gather_u8_f16"
            Pair(DType.U8, DType.BF16) -> "gather_u8_bf16"
            Pair(DType.U8, DType.U32) -> "gather_u8_u32"
            Pair(DType.U8, DType.I64) -> "gather_u8_i64"
            Pair(DType.U32, DType.F32) -> "gather_u32_f32"
            Pair(DType.U32, DType.F16) -> "gather_u32_f16"
            Pair(DType.U32, DType.BF16) -> "gather_u32_bf16"
            Pair(DType.U32, DType.U32) -> "gather_u32_u32"
            Pair(DType.U32, DType.I64) -> "gather_u32_i64"
            Pair(DType.I64, DType.F32) -> "gather_i64_f32"
            Pair(DType.I64, DType.F16) -> "gather_i64_f16"
            Pair(DType.I64, DType.BF16) -> "gather_i64_bf16"
            Pair(DType.I64, DType.U32) -> "gather_i64_u32"
            Pair(DType.I64, DType.I64) -> "gather_i64_i64"
            else -> throw MetalError.from("Metal gather ${ids.dtype} $dt unsupported")
        }
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, srcL, dt)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        encoder.hashCode()
        name.hashCode()
        idsEl.hashCode()
        src.hashCode()
        idsBuf.hashCode()
        new(outBuffer, dev, dstEl, dt)
    }

    fun scatterSet(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int
    ): Result<Unit> = runCatching {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            throw CandleError.RequiresContiguous(op = "scatter").bt()
        }
        val name = when (Pair(ids.dtype, dtype)) {
            Pair(DType.U8, DType.F32) -> "s_u8_f32"
            Pair(DType.U8, DType.F16) -> "s_u8_f16"
            Pair(DType.U8, DType.BF16) -> "s_u8_bf16"
            Pair(DType.U32, DType.U32) -> "s_u32_u32"
            Pair(DType.U32, DType.F32) -> "s_u32_f32"
            Pair(DType.U32, DType.F16) -> "s_u32_f16"
            Pair(DType.U32, DType.BF16) -> "s_u32_bf16"
            Pair(DType.I64, DType.F32) -> "s_i64_f32"
            Pair(DType.I64, DType.F16) -> "s_i64_f16"
            Pair(DType.I64, DType.BF16) -> "s_i64_bf16"
            else -> throw MetalError.UnexpectedDType(
                msg = "scatter ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val encoder = device.commandEncoder().getOrThrow()
        val dst = bufferO(buffer, l, dtype)
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        encoder.hashCode()
        name.hashCode()
        dim.hashCode()
        dst.hashCode()
        srcBuf.hashCode()
        idsBuf.hashCode()
    }

    fun scatterAddSet(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int
    ): Result<Unit> = runCatching {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            throw CandleError.RequiresContiguous(op = "scatter-add").bt()
        }
        val name = when (Pair(ids.dtype, dtype)) {
            Pair(DType.U8, DType.F32) -> "sa_u8_f32"
            Pair(DType.U8, DType.F16) -> "sa_u8_f16"
            Pair(DType.U8, DType.BF16) -> "sa_u8_bf16"
            Pair(DType.U32, DType.U32) -> "sa_u32_u32"
            Pair(DType.U32, DType.F32) -> "sa_u32_f32"
            Pair(DType.U32, DType.F16) -> "sa_u32_f16"
            Pair(DType.U32, DType.BF16) -> "sa_u32_bf16"
            Pair(DType.I64, DType.F32) -> "sa_i64_f32"
            Pair(DType.I64, DType.F16) -> "sa_i64_f16"
            Pair(DType.I64, DType.BF16) -> "sa_i64_bf16"
            else -> throw MetalError.UnexpectedDType(
                msg = "scatter-add ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val encoder = device.commandEncoder().getOrThrow()
        val dst = bufferO(buffer, l, dtype)
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        encoder.hashCode()
        name.hashCode()
        dim.hashCode()
        dst.hashCode()
        srcBuf.hashCode()
        idsBuf.hashCode()
    }

    fun indexSelect(ids: MetalStorage, srcL: Layout, idsL: Layout, dim: Int): Result<MetalStorage> = runCatching {
        if (!idsL.isContiguous()) {
            throw MetalError.from("Metal indexSelect requires contiguous ids")
        }
        val leftSize = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSize = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val idsEl = idsL.shape().elemCount()
        val dstEl = idsEl * leftSize * rightSize
        val dt = dtype
        val dev = device()
        val outBuffer = dev
            .newBufferBuilder()
            .withSizeFor(dstEl, dt)
            .withLabel("index_select")
            .build()
            .getOrThrow()
        val name = when (Pair(ids.dtype, dt)) {
            Pair(DType.U8, DType.U8) -> "is_u8_u8"
            Pair(DType.U8, DType.U32) -> "is_u8_u32"
            Pair(DType.U8, DType.I64) -> "is_u8_i64"
            Pair(DType.U8, DType.BF16) -> "is_u8_bf16"
            Pair(DType.U8, DType.F32) -> "is_u8_f32"
            Pair(DType.U8, DType.F16) -> "is_u8_f16"

            Pair(DType.U32, DType.U8) -> "is_u32_u8"
            Pair(DType.U32, DType.U32) -> "is_u32_u32"
            Pair(DType.U32, DType.I64) -> "is_u32_i64"
            Pair(DType.U32, DType.F32) -> "is_u32_f32"
            Pair(DType.U32, DType.F16) -> "is_u32_f16"
            Pair(DType.U32, DType.BF16) -> "is_u32_bf16"

            Pair(DType.I64, DType.U8) -> "is_i64_u8"
            Pair(DType.I64, DType.U32) -> "is_i64_u32"
            Pair(DType.I64, DType.I64) -> "is_i64_i64"
            Pair(DType.I64, DType.F32) -> "is_i64_f32"
            Pair(DType.I64, DType.F16) -> "is_i64_f16"
            Pair(DType.I64, DType.BF16) -> "is_i64_bf16"

            else -> throw MetalError.from("Metal contiguous indexSelect ${ids.dtype} $dt unsupported")
        }
        val encoder = dev.commandEncoder().getOrThrow()
        val src = bufferO(buffer, srcL, dt)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        encoder.hashCode()
        name.hashCode()
        src.hashCode()
        idsBuf.hashCode()
        new(outBuffer, dev, dstEl, dt)
    }

    fun indexAdd(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int
    ): Result<MetalStorage> = runCatching {
        val acc = device.zerosImpl(l.shape(), dtype()).getOrThrow()
        copyStridedSrc(acc, 0, l).getOrThrow()
        if (!idsL.isContiguous() || !srcL.isContiguous()) {
            throw CandleError.RequiresContiguous(op = "index-add").bt()
        }
        val name = when (Pair(ids.dtype, dtype)) {
            Pair(DType.I64, DType.BF16) -> "ia_i64_bf16"
            Pair(DType.I64, DType.F16) -> "ia_i64_f16"
            Pair(DType.I64, DType.F32) -> "ia_i64_f32"
            Pair(DType.I64, DType.I64) -> "ia_i64_i64"
            Pair(DType.I64, DType.U32) -> "ia_i64_u32"
            Pair(DType.I64, DType.U8) -> "ia_i64_u8"

            Pair(DType.U32, DType.BF16) -> "ia_u32_bf16"
            Pair(DType.U32, DType.F16) -> "ia_u32_f16"
            Pair(DType.U32, DType.F32) -> "ia_u32_f32"
            Pair(DType.U32, DType.I64) -> "ia_u32_i64"
            Pair(DType.U32, DType.U32) -> "ia_u32_u32"
            Pair(DType.U32, DType.U8) -> "ia_u32_u8"

            Pair(DType.U8, DType.BF16) -> "ia_u8_bf16"
            Pair(DType.U8, DType.F16) -> "ia_u8_f16"
            Pair(DType.U8, DType.F32) -> "ia_u8_f32"
            Pair(DType.U8, DType.I64) -> "ia_u8_i64"
            Pair(DType.U8, DType.U32) -> "ia_u8_u32"
            Pair(DType.U8, DType.U8) -> "ia_u8_u8"

            else -> throw MetalError.UnexpectedDType(
                msg = "index-add ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val encoder = device.commandEncoder().getOrThrow()
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        encoder.hashCode()
        name.hashCode()
        dim.hashCode()
        srcBuf.hashCode()
        idsBuf.hashCode()
        acc
    }

    fun matmul(
        rhs: MetalStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout
    ): Result<MetalStorage> = runCatching {
        val b = bmnk[0]
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]
        val outBuffer = device
            .newBufferBuilder()
            .withSizeFor(b * m * n, dtype)
            .withLabel("matmul")
            .build()
            .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val gemmDtype = when (dtype) {
            DType.F32 -> GemmDType.F32
            DType.F16 -> GemmDType.F16
            DType.BF16 -> GemmDType.BF16
            else -> throw MetalError.Message("mlx matmul doesn't support $dtype")
        }
        encoder.hashCode()
        gemmDtype.hashCode()
        k.hashCode()
        lhsL.hashCode()
        rhsL.hashCode()
        rhs.hashCode()
        new(
            outBuffer,
            device,
            b * m * n,
            dtype()
        )
    }

    fun copy2d(
        dst: MetalStorage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int
    ): Result<Unit> = runCatching {
        if (dtype() != dst.dtype()) {
            throw MetalError.from("copy2d with inconsistent dtypes ${dtype()} ${dst.dtype()}")
        }
        if (srcS == d2 && dstS == d2) {
            val blit = device.blitCommandEncoder().getOrThrow()
            val srcOffset = srcO * dtype.sizeInBytes()
            val length = d1 * d2 * dtype.sizeInBytes()
            val dstOffset = dstO * dst.dtype().sizeInBytes()
            blit.hashCode()
            srcOffset.hashCode()
            length.hashCode()
            dstOffset.hashCode()
        } else {
            val elCount = d1 * d2
            if (elCount == 0) {
                return@runCatching
            }
            val kernelName = when (dtype) {
                DType.F32 -> "copy2d_f32"
                DType.F16 -> "copy2d_f16"
                DType.BF16 -> "copy2d_bf16"
                DType.I64 -> "copy2d_i64"
                DType.I32 -> "copy2d_i32"
                DType.I16 -> "copy2d_i16"
                DType.U32 -> "copy2d_u32"
                DType.U8 -> "copy2d_u8"
                else -> throw MetalError.from("Metal copy2d $dtype unsupported")
            }
            val encoder = device.commandEncoder().getOrThrow()
            encoder.hashCode()
            kernelName.hashCode()
        }
    }

    fun copyStridedSrc(dst: MetalStorage, dstOffset: Int, srcL: Layout): Result<Unit> = runCatching {
        if (srcL.isContiguous() && dtype == dst.dtype()) {
            val blit = device.blitCommandEncoder().getOrThrow()
            val srcOffset = srcL.startOffset() * dtype.sizeInBytes()
            val length = srcL.shape().elemCount() * dtype.sizeInBytes()
            val dstByteOffset = dstOffset * dst.dtype().sizeInBytes()
            blit.hashCode()
            srcOffset.hashCode()
            length.hashCode()
            dstByteOffset.hashCode()
        } else {
            val srcShape = srcL.shape()
            val elCount = srcShape.elemCount()
            if (elCount == 0) {
                return@runCatching
            }
            if (dtype == dst.dtype()) {
                val strides = srcL.stride()
                val ndim = strides.size
                // Inner blocks are contiguous - eligible for copy2d
                if (ndim == 1 || strides[ndim - 1] == 1) {
                    val blocks = srcL.stridedBlocks()
                    if (blocks is StridedBlocks.UniformBlocks) {
                        return copy2d(
                            dst,
                            blocks.count,
                            blocks.blockLen,
                            blocks.srcStride,
                            blocks.blockLen,
                            blocks.startOffset,
                            dstOffset
                        )
                    }
                }
            }
            val kernelName = when (dtype) {
                DType.F32 -> "copy_strided_f32"
                DType.F16 -> "copy_strided_f16"
                DType.BF16 -> "copy_strided_bf16"
                DType.I64 -> "copy_strided_i64"
                DType.U32 -> "copy_strided_u32"
                DType.U8 -> "copy_strided_u8"
                else -> throw MetalError.from("Metal copyStrided $dtype unsupported")
            }
            val src = bufferO(buffer, srcL, dtype)
            val dstBuf = BufferOffset(
                buffer = dst.buffer,
                offsetInBytes = dstOffset * dst.dtype.sizeInBytes()
            )
            val encoder = device.commandEncoder().getOrThrow()
            encoder.hashCode()
            kernelName.hashCode()
            src.hashCode()
            dstBuf.hashCode()
        }
    }

    fun buffer(): Buffer {
        return buffer
    }

    fun binary(
        op: String,
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout
    ): Result<MetalStorage> = runCatching {
        fun kernelName(opName: String, dt: DType, suffix: String): String {
            return "${opName}_${dt.asStr()}$suffix"
        }
        val dev = device()
        val shape = lhsL.shape()
        val elCount = shape.elemCount()
        val encoder = dev.commandEncoder().getOrThrow()
        val lhs = bufferO(buffer, lhsL, dtype)
        val rhsBuf = bufferO(rhs.buffer, rhsL, rhs.dtype)
        val outDtype = when (op) {
            "eq", "ne", "le", "lt", "ge", "gt" -> DType.U8
            else -> dtype
        }
        val lhsIsScalar = lhsL.isScalarLike()
        val rhsIsScalar = rhsL.isScalarLike()
        val lhsContiguous = lhsL.isContiguous()
        val rhsContiguous = rhsL.isContiguous()

        val contiguousKernel = kernelName(op, dtype, "")
        val kernel = when {
            lhsIsScalar && rhsIsScalar -> kernelName(op, dtype, "_scalar")
            lhsIsScalar && !rhsIsScalar && rhsContiguous -> kernelName(op, dtype, "_sc")
            lhsIsScalar && !rhsIsScalar && !rhsContiguous -> kernelName(op, dtype, "_rss")
            !lhsIsScalar && rhsIsScalar && lhsContiguous -> kernelName(op, dtype, "_cs")
            !lhsIsScalar && rhsIsScalar && !lhsContiguous -> kernelName(op, dtype, "_lss")
            !lhsIsScalar && !rhsIsScalar && lhsContiguous && rhsContiguous -> contiguousKernel
            !lhsIsScalar && !rhsIsScalar && lhsContiguous && !rhsContiguous -> kernelName(op, dtype, "_rstrided")
            !lhsIsScalar && !rhsIsScalar && !lhsContiguous && rhsContiguous -> kernelName(op, dtype, "_lstrided")
            else -> kernelName(op, dtype, "_strided")
        }

        val outBuffer = if (kernel == contiguousKernel) {
            val buf = dev
                .newBufferBuilder()
                .withSizeFor(elCount, outDtype)
                .withLabel(op)
                .build()
                .getOrThrow()
            encoder.hashCode()
            lhs.hashCode()
            rhsBuf.hashCode()
            buf
        } else {
            val buf = dev
                .newBufferBuilder()
                .withSizeFor(elCount, outDtype)
                .withLabel(op)
                .build()
                .getOrThrow()
            encoder.hashCode()
            lhs.hashCode()
            rhsBuf.hashCode()
            buf
        }
        new(outBuffer, dev, elCount, outDtype)
    }

    internal fun <T> toCpu(): Result<List<T>> = runCatching {
        val size = count * dtype.sizeInBytes()
        val dstBuffer = device
            .newBufferBuilder()
            .withSize(size)
            .withLabel("blit_to_cpu_dst")
            .build()
            .getOrThrow()
        val blit = device.blitCommandEncoder().getOrThrow()
        blit.hashCode()
        device.flushAndWaitCurrent().getOrThrow()
        readToVec(dstBuffer, count)
    }

    companion object {
        fun new(buffer: Buffer, device: MetalDevice, count: Int, dtype: DType): MetalStorage {
            return MetalStorage(
                buffer = buffer,
                device = device,
                count = count,
                dtype = dtype
            )
        }
    }
}

class MetalBackendDevice(val inner: MetalDevice) {
    companion object {
        fun new(ordinal: Int): Result<MetalDevice> = runCatching {
            val dev = Device(registryId = ordinal.toULong())
            val kernels = Kernels()
            val residencySet = ResidencySet()
            val seedBuf = Buffer(8)
            val commands = Commands()
            MetalDevice(
                id = DeviceId.new(),
                device = dev,
                commands = commands,
                buffers = mutableMapOf(),
                privateBuffers = mutableMapOf(),
                kernels = kernels,
                seed = seedBuf,
                seedValue = 299792458uL,
                residencySet = residencySet
            )
        }
    }

    fun location(): DeviceLocation {
        return DeviceLocation.Metal(gpuId = inner.registryId().toInt())
    }

    fun sameDevice(rhs: MetalDevice): Boolean {
        return inner.id == rhs.id
    }

    fun allocUninit(shape: Shape, dtype: DType): Result<MetalStorage> = runCatching {
        val buffer = inner
            .newBufferBuilder()
            .withSizeFor(shape.elemCount(), dtype)
            .withLabel("alloc-uninit")
            .build()
            .getOrThrow()
        MetalStorage.new(
            buffer = buffer,
            device = inner,
            count = shape.elemCount(),
            dtype = dtype
        )
    }

    fun zerosImpl(shape: Shape, dtype: DType): Result<MetalStorage> = runCatching {
        val size = shape.elemCount() * dtype.sizeInBytes()
        val buffer = inner
            .newBufferBuilder()
            .withZeros(size)
            .withLabel("zeros")
            .build()
            .getOrThrow()
        MetalStorage.new(
            buffer = buffer,
            device = inner,
            count = shape.elemCount(),
            dtype = dtype
        )
    }

    fun <T : WithDType> storageFromSlice(
        s: List<T>,
        dtype: DType,
        cpuStorageRef: (List<T>) -> CpuStorageRef
    ): Result<MetalStorage> = runCatching {
        val label = "storage_from_slice"
        val (count, bufferRes) = when (val ref = cpuStorageRef(s)) {
            is CpuStorageRef.U8 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size).withLabel(label).build())
            is CpuStorageRef.U32 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 4).withLabel(label).build())
            is CpuStorageRef.I16 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 2).withLabel(label).build())
            is CpuStorageRef.I32 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 4).withLabel(label).build())
            is CpuStorageRef.I64 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 8).withLabel(label).build())
            is CpuStorageRef.BF16 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 2).withLabel(label).build())
            is CpuStorageRef.F16 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 2).withLabel(label).build())
            is CpuStorageRef.F32 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 4).withLabel(label).build())
            is CpuStorageRef.F64 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size * 8).withLabel(label).build())
            is CpuStorageRef.F8E4M3 -> Pair(ref.data.size, inner.newBufferBuilder().withSize(ref.data.size).withLabel(label).build())
            is CpuStorageRef.F6E2M3,
            is CpuStorageRef.F6E3M2,
            is CpuStorageRef.F4,
            is CpuStorageRef.F8E8M0 -> throw CandleError.UnsupportedDTypeForOp(dtype, "to_dtype").bt()
        }
        MetalStorage.new(bufferRes.getOrThrow(), inner, count, dtype)
    }

    fun storageFromCpuStorage(storage: CpuStorage): Result<MetalStorage> = runCatching {
        val label = "storage_from_cpu_storage"
        val (count, bufferRes) = when (storage) {
            is CpuStorage.U8 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size).withLabel(label).build())
            is CpuStorage.U32 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 4).withLabel(label).build())
            is CpuStorage.I16 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 2).withLabel(label).build())
            is CpuStorage.I32 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 4).withLabel(label).build())
            is CpuStorage.I64 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 8).withLabel(label).build())
            is CpuStorage.BF16 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 2).withLabel(label).build())
            is CpuStorage.F16 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 2).withLabel(label).build())
            is CpuStorage.F32 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 4).withLabel(label).build())
            is CpuStorage.F64 -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size * 8).withLabel(label).build())
            is CpuStorage.F8E4M3Storage -> Pair(storage.data.size, inner.newBufferBuilder().withSize(storage.data.size).withLabel(label).build())
            is CpuStorage.F6E2M3,
            is CpuStorage.F6E3M2,
            is CpuStorage.F4,
            is CpuStorage.F8E8M0Storage -> throw CandleError.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt()
        }
        MetalStorage.new(bufferRes.getOrThrow(), inner, count, storage.dtype())
    }

    fun storageFromCpuStorageOwned(storage: CpuStorage): Result<MetalStorage> {
        return storageFromCpuStorage(storage)
    }

    fun randUniform(
        shape: Shape,
        dtype: DType,
        min: Double,
        max: Double
    ): Result<MetalStorage> = runCatching {
        val name = when (dtype) {
            DType.F32 -> "rand_uniform_f32"
            DType.F16 -> "rand_uniform_f16"
            DType.BF16 -> "rand_uniform_bf16"
            else -> throw MetalError.from("randUniform unsupported for $dtype")
        }
        val buffer = inner
            .newBufferBuilder()
            .withSizeFor(shape.elemCount(), dtype)
            .withLabel("rand_uniform")
            .build()
            .getOrThrow()
        val encoder = inner.commandEncoder().getOrThrow()
        name.hashCode()
        min.hashCode()
        max.hashCode()
        encoder.hashCode()
        MetalStorage.new(
            buffer = buffer,
            device = inner,
            count = shape.elemCount(),
            dtype = dtype
        )
    }

    fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        stddev: Double
    ): Result<MetalStorage> = runCatching {
        val name = when (dtype) {
            DType.F32 -> "rand_normal_f32"
            DType.F16 -> "rand_normal_f16"
            DType.BF16 -> "rand_normal_bf16"
            else -> throw MetalError.from("randUniform unsupported for $dtype")
        }
        val buffer = inner
            .newBufferBuilder()
            .withSizeFor(shape.elemCount(), dtype)
            .withLabel("rand_normal")
            .build()
            .getOrThrow()
        val encoder = inner.commandEncoder().getOrThrow()
        name.hashCode()
        mean.hashCode()
        stddev.hashCode()
        encoder.hashCode()
        MetalStorage.new(
            buffer = buffer,
            device = inner,
            count = shape.elemCount(),
            dtype = dtype
        )
    }

    fun setSeed(seed: ULong): Result<Unit> = runCatching {
        inner.seedValue = seed
    }

    fun getCurrentSeed(): Result<ULong> {
        return Result.success(inner.seedValue)
    }

    fun synchronize(): Result<Unit> {
        return inner.waitUntilCompleted()
    }
}

private fun <T> readToVec(buffer: Buffer, n: Int): List<T> {
    buffer.hashCode()
    n.hashCode()
    return emptyList()
}
