package io.github.kotlinmania.candle.metal

import io.github.kotlinmania.candle.BinaryOpT
import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CmpOp
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CpuStorageRef
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.DeviceLocation
import io.github.kotlinmania.candle.F8E4M3
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
import io.github.kotlinmania.candle.metal.device.Arc
import io.github.kotlinmania.candle.metal.device.Buffer
import io.github.kotlinmania.candle.metal.device.BufferOffset
import io.github.kotlinmania.candle.metal.device.CallConvTranspose2dCfg
import io.github.kotlinmania.candle.metal.device.CandleMetalKernels
import io.github.kotlinmania.candle.metal.device.CandleMetalKernels.unary.contiguous
import io.github.kotlinmania.candle.metal.device.CandleMetalKernels.unary.strided
import io.github.kotlinmania.candle.metal.device.Commands
import io.github.kotlinmania.candle.metal.device.Device
import io.github.kotlinmania.candle.metal.device.DeviceId
import io.github.kotlinmania.candle.metal.device.Err
import io.github.kotlinmania.candle.metal.device.HashMap
import io.github.kotlinmania.candle.metal.device.Kernels
import io.github.kotlinmania.candle.metal.device.MetalDevice
import io.github.kotlinmania.candle.metal.device.Mutex
import io.github.kotlinmania.candle.metal.device.NSRange
import io.github.kotlinmania.candle.metal.device.NSString
import io.github.kotlinmania.candle.metal.device.Ok
import io.github.kotlinmania.candle.metal.device.PoisonError
import io.github.kotlinmania.candle.metal.device.RESOURCE_OPTIONS
import io.github.kotlinmania.candle.metal.device.ResidencySet
import io.github.kotlinmania.candle.metal.device.RwLock
import io.github.kotlinmania.candle.metal.device.Self
import io.github.kotlinmania.candle.metal.device.Some
import io.github.kotlinmania.candle.metal.device.Storage
import io.github.kotlinmania.candle.metal.device.TryLockError
import io.github.kotlinmania.candle.metal.device.asPtr
import io.github.kotlinmania.candle.metal.device.bail
import io.github.kotlinmania.candle.metal.device.broadcastAs
import io.github.kotlinmania.candle.metal.device.clone
import io.github.kotlinmania.candle.metal.device.contents
import io.github.kotlinmania.candle.metal.device.contiguous
import io.github.kotlinmania.candle.metal.device.contiguousWithOffset
import io.github.kotlinmania.candle.metal.device.copyNonoverlapping
import io.github.kotlinmania.candle.metal.device.dims3
import io.github.kotlinmania.candle.metal.device.dims4
import io.github.kotlinmania.candle.metal.device.drop
import io.github.kotlinmania.candle.metal.device.enumerate
import io.github.kotlinmania.candle.metal.device.format
import io.github.kotlinmania.candle.metal.device.from
import io.github.kotlinmania.candle.metal.device.fromRawParts
import io.github.kotlinmania.candle.metal.device.fromStr
import io.github.kotlinmania.candle.metal.device.into
import io.github.kotlinmania.candle.metal.device.isNull
import io.github.kotlinmania.candle.metal.device.iter
import io.github.kotlinmania.candle.metal.device.len
import io.github.kotlinmania.candle.metal.device.lock
import io.github.kotlinmania.candle.metal.device.mapErr
import io.github.kotlinmania.candle.metal.device.new
import io.github.kotlinmania.candle.metal.device.newMap
import io.github.kotlinmania.candle.metal.device.product
import io.github.kotlinmania.candle.metal.device.push
import io.github.kotlinmania.candle.metal.device.read
import io.github.kotlinmania.candle.metal.device.self
import io.github.kotlinmania.candle.metal.device.sizeOfU64
import io.github.kotlinmania.candle.metal.device.storage
import io.github.kotlinmania.candle.metal.device.swapRemove
import io.github.kotlinmania.candle.metal.device.toVec
import io.github.kotlinmania.candle.metal.device.tryLock
import io.github.kotlinmania.candle.metal.device.unwrap
import io.github.kotlinmania.candle.metal.device.write
import io.github.kotlinmania.candle.metal.device.zerosImpl
import io.github.kotlinmania.thiserror.StdError

fun bufferO(
    buffer: Buffer,
    l: Layout,
    dtype: DType,
): BufferOffset =
    BufferOffset(
        buffer = buffer,
        offsetInBytes = l.startOffset() * dtype.sizeInBytes(),
    )

/**
 * Simple way to catch lock error without
 * depending on T
 */
sealed class LockError :
    Exception(),
    StdError {
    data class Poisoned(
        val msg: String,
    ) : LockError() {
        override val message: String get() = msg
    }

    data object WouldBlock : LockError() {
        override val message: String get() = "Would block"
    }
}

/** Metal related errors */
sealed class MetalError :
    Exception(),
    StdError {
    data class Message(
        val msg: String,
    ) : MetalError() {
        override val message: String get() = msg
    }

    data class KernelError(
        val err: String,
    ) : MetalError() {
        override val message: String get() = err
    }

    data class LockErr(
        val err: LockError,
    ) : MetalError() {
        override val message: String get() = err.toString()
    }

    data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType,
    ) : MetalError() {
        override val message: String get() = "$msg, expected: $expected, got: $got"
    }

    companion object {
        fun <T> from(value: TryLockError<T>): MetalError =
            when (value) {
                is TryLockError.Poisoned -> MetalError.LockErr(LockError.Poisoned(value.p))
                is TryLockError.WouldBlock -> MetalError.LockErr(LockError.WouldBlock)
            }

        fun <T> from(p: PoisonError<T>): MetalError = MetalError.LockErr(LockError.Poisoned(p.toString()))

        fun from(e: String): MetalError = MetalError.Message(e)
    }
}

class MetalStorage(
    /** The actual buffer containing the data. */
    internal val buffer: Buffer,
    /** a reference to the device owning this buffer */
    internal val device: MetalDevice,
    /** The count of allocated elements in the buffer */
    val count: Int,
    /** The dtype is kept since buffers are untyped. */
    internal val dtype: DType,
) {
    interface Device

    fun tryClone(layout: Layout): Result<MetalStorage> {
        if (layout.startOffset() < 0) bail("invalid start offset")
        return Ok(self.clone())
    }

    fun dtype(): DType = self.dtype

    fun device(): MetalDevice = self.device

    fun toCpuStorage(): Result<CpuStorage> =
        when (self.dtype) {
            DType.U8 -> {
                Ok(CpuStorage.U8(self.toCpu<UByte>().getOrThrow().toUByteArray()))
            }

            DType.U32 -> {
                Ok(CpuStorage.U32(self.toCpu<UInt>().getOrThrow().toUIntArray()))
            }

            DType.I16 -> {
                Ok(CpuStorage.I16(self.toCpu<Short>().getOrThrow().toShortArray()))
            }

            DType.I32 -> {
                Ok(CpuStorage.I32(self.toCpu<Int>().getOrThrow().toIntArray()))
            }

            DType.I64 -> {
                Ok(CpuStorage.I64(self.toCpu<Long>().getOrThrow().toLongArray()))
            }

            DType.F16 -> {
                Ok(CpuStorage.F16(self.toCpu<Short>().getOrThrow().toShortArray()))
            }

            DType.BF16 -> {
                Ok(CpuStorage.BF16(self.toCpu<Short>().getOrThrow().toShortArray()))
            }

            DType.F32 -> {
                Ok(CpuStorage.F32(self.toCpu<Float>().getOrThrow().toFloatArray()))
            }

            DType.F64 -> {
                Ok(CpuStorage.F64(self.toCpu<Double>().getOrThrow().toDoubleArray()))
            }

            DType.F8E4M3 -> {
                Ok(CpuStorage.F8E4M3(self.toCpu<Byte>().getOrThrow().toByteArray()))
            }

            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                Err(CandleError.UnsupportedDTypeForOp(self.dtype, "to_cpu_storage").bt())
            }
        }

    fun affine(
        layout: Layout,
        mul: Double,
        add: Double,
    ): Result<MetalStorage> {
        val device = self.device().clone()

        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = self.dtype

        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(el, self.dtype)
                .withLabel("affine")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, dtype)
        if (layout.isContiguous()) {
            val name =
                when (self.dtype) {
                    DType.F32 -> "affine_f32"
                    DType.F16 -> "affine_f16"
                    DType.BF16 -> "affine_bf16"
                    DType.U8 -> "affine_u8"
                    DType.U32 -> "affine_u32"
                    DType.I64 -> "affine_i64"
                    else -> bail("Metal contiguous affine $dtype not implemented")
                }
            CandleMetalKernels
                .callAffine(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    self.dtype.sizeInBytes(),
                    el,
                    src,
                    buffer,
                    mul.toFloat(),
                    add.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        } else {
            val name =
                when (self.dtype) {
                    DType.F32 -> "affine_f32_strided"
                    DType.F16 -> "affine_f16_strided"
                    DType.BF16 -> "affine_bf16_strided"
                    DType.U8 -> "affine_u8_strided"
                    DType.U32 -> "affine_u32_strided"
                    DType.I64 -> "affine_i64_strided"
                    else -> bail("Metal strided affine $dtype not implemented")
                }
            CandleMetalKernels
                .callAffineStrided(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    layout.dims(),
                    src,
                    layout.stride(),
                    buffer,
                    mul.toFloat(),
                    add.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Self.new(buffer, device.clone(), el, dtype))
    }

    fun powf(
        layout: Layout,
        pow: Double,
    ): Result<MetalStorage> {
        val device = self.device().clone()

        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = self.dtype

        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(el, self.dtype)
                .withLabel("powf")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, dtype)
        if (layout.isContiguous()) {
            val name =
                when (self.dtype) {
                    DType.F32 -> "powf_f32"
                    DType.F16 -> "powf_f16"
                    DType.BF16 -> "powf_bf16"
                    else -> bail("Metal contiguous powf $dtype not implemented")
                }
            CandleMetalKernels
                .callPowf(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    self.dtype.sizeInBytes(),
                    el,
                    src,
                    buffer,
                    pow.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        } else {
            val name =
                when (self.dtype) {
                    DType.F32 -> "powf_f32_strided"
                    DType.F16 -> "powf_f16_strided"
                    DType.BF16 -> "powf_bf16_strided"
                    else -> bail("Metal strided powf $dtype not implemented")
                }
            CandleMetalKernels
                .callPowfStrided(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    layout.dims(),
                    src,
                    layout.stride(),
                    buffer,
                    pow.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Self.new(buffer, device.clone(), el, dtype))
    }

    fun elu(
        layout: Layout,
        alpha: Double,
    ): Result<MetalStorage> {
        val device = self.device().clone()

        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = self.dtype

        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(el, self.dtype)
                .withLabel("elu")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, self.dtype)
        if (layout.isContiguous()) {
            val name =
                when (self.dtype) {
                    DType.F32 -> "elu_f32"
                    DType.F16 -> "elu_f16"
                    DType.BF16 -> "elu_bf16"
                    else -> bail("Metal contiguous elu $dtype not implemented")
                }
            CandleMetalKernels
                .callElu(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    self.dtype.sizeInBytes(),
                    el,
                    src,
                    buffer,
                    alpha.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        } else {
            val name =
                when (self.dtype) {
                    DType.F32 -> "elu_f32_strided"
                    DType.F16 -> "elu_f16_strided"
                    DType.BF16 -> "elu_bf16_strided"
                    else -> bail("Metal strided elu $dtype not implemented")
                }
            CandleMetalKernels
                .callEluStrided(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    layout.dims(),
                    src,
                    layout.stride(),
                    buffer,
                    alpha.toFloat(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Self.new(buffer, device.clone(), el, dtype))
    }

    fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        sumDims: List<Int>,
    ): Result<MetalStorage> {
        val device = self.device.clone()

        val srcStride = layout.stride()
        val srcDims = layout.shape().dims()
        // Source dims and strides with the sum dims at the end.
        val dims = mutableListOf<Int>()
        val stride = mutableListOf<Int>()
        var dstEl = 1
        for ((dimIdx, d) in srcDims.iter().enumerate()) {
            if (!sumDims.contains(dimIdx)) {
                dstEl *= d
                dims.push(d)
                stride.push(srcStride[dimIdx])
            }
        }

        for (dimIdx in sumDims.iter()) {
            dims.push(srcDims[dimIdx])
            stride.push(srcStride[dimIdx])
        }

        val reductionShape = Shape.from(dims.clone())

        if (layout.isContiguous() && reductionShape.isContiguous(stride)) {
            val (name, checkEmpty, returnIndex) =
                when (Pair(op, self.dtype)) {
                    Pair(ReduceOp.Sum, DType.F32) -> {
                        Triple("fast_sum_f32", false, false)
                    }

                    Pair(ReduceOp.Min, DType.F32) -> {
                        Triple("fast_min_f32", true, false)
                    }

                    Pair(ReduceOp.Max, DType.F32) -> {
                        Triple("fast_max_f32", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.F32) -> {
                        Triple("fast_argmin_f32", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.F32) -> {
                        Triple("fast_argmax_f32", true, true)
                    }

                    Pair(ReduceOp.Sum, DType.U32) -> {
                        Triple("fast_sum_u32", false, false)
                    }

                    Pair(ReduceOp.Min, DType.U32) -> {
                        Triple("fast_min_u32", true, false)
                    }

                    Pair(ReduceOp.Max, DType.U32) -> {
                        Triple("fast_max_u32", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.U32) -> {
                        Triple("fast_argmin_u32", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.U32) -> {
                        Triple("fast_argmax_u32", true, true)
                    }

                    Pair(ReduceOp.Sum, DType.F16) -> {
                        Triple("fast_sum_f16", false, false)
                    }

                    Pair(ReduceOp.Min, DType.F16) -> {
                        Triple("fast_min_f16", true, false)
                    }

                    Pair(ReduceOp.Max, DType.F16) -> {
                        Triple("fast_max_f16", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.F16) -> {
                        Triple("fast_argmin_f16", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.F16) -> {
                        Triple("fast_argmax_f16", true, true)
                    }

                    Pair(ReduceOp.Sum, DType.BF16) -> {
                        Triple("fast_sum_bf16", false, false)
                    }

                    Pair(ReduceOp.Min, DType.BF16) -> {
                        Triple("fast_min_bf16", true, false)
                    }

                    Pair(ReduceOp.Max, DType.BF16) -> {
                        Triple("fast_max_bf16", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.BF16) -> {
                        Triple("fast_argmin_bf16", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.BF16) -> {
                        Triple("fast_argmax_bf16", true, true)
                    }

                    Pair(ReduceOp.Sum, DType.I64) -> {
                        Triple("fast_sum_i64", false, false)
                    }

                    Pair(ReduceOp.Min, DType.I64) -> {
                        Triple("fast_min_i64", true, false)
                    }

                    Pair(ReduceOp.Max, DType.I64) -> {
                        Triple("fast_max_i64", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.I64) -> {
                        Triple("fast_argmin_i64", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.I64) -> {
                        Triple("fast_argmax_i64", true, true)
                    }

                    Pair(ReduceOp.Sum, DType.U8) -> {
                        Triple("fast_sum_u8", false, false)
                    }

                    Pair(ReduceOp.Min, DType.U8) -> {
                        Triple("fast_min_u8", true, false)
                    }

                    Pair(ReduceOp.Max, DType.U8) -> {
                        Triple("fast_max_u8", true, false)
                    }

                    Pair(ReduceOp.ArgMin, DType.U8) -> {
                        Triple("fast_argmin_u8", true, true)
                    }

                    Pair(ReduceOp.ArgMax, DType.U8) -> {
                        Triple("fast_argmax_u8", true, true)
                    }

                    else -> {
                        bail("Metal contiguous reduce op $op ${self.dtype} not implemented")
                    }
                }
            if (checkEmpty && layout.shape().elemCount() == 0) {
                Err(CandleError.EmptyTensor(op = "reduce").bt()).getOrThrow()
            }
            val dtype = if (returnIndex) DType.U32 else self.dtype
            val buffer =
                device
                    .newBufferBuilder()
                    .withSizeFor(dstEl, dtype)
                    .withLabel("reduce")
                    .build()
                    .getOrThrow()
            val encoder = self.device.commandEncoder().getOrThrow()
            val src = bufferO(self.buffer, layout, self.dtype)
            CandleMetalKernels
                .callReduceContiguous(
                    device.device,
                    encoder,
                    device.kernels,
                    name,
                    srcDims,
                    dstEl,
                    src,
                    buffer,
                ).mapErr(MetalError.from)
                .getOrThrow()

            return Ok(Self.new(buffer, device, dstEl, dtype))
        }

        val (name, checkEmpty, returnIndex) =
            when (Pair(op, self.dtype)) {
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
                else -> bail("Metal strided reduce op $op ${self.dtype} not implemented")
            }
        if (checkEmpty && layout.shape().elemCount() == 0) {
            Err(CandleError.EmptyTensor(op = "reduce").bt()).getOrThrow()
        }
        val dtype = if (returnIndex) DType.U32 else self.dtype
        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(dstEl, dtype)
                .withLabel("reduce")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, self.dtype)
        CandleMetalKernels
            .callReduceStrided(
                device.device,
                encoder,
                device.kernels,
                name,
                dims,
                stride,
                dstEl,
                src,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()

        return Ok(Self.new(buffer, device, dstEl, dtype))
    }

    fun cmp(
        op: CmpOp,
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): Result<MetalStorage> {
        val name =
            when (op) {
                CmpOp.Eq -> "eq"
                CmpOp.Ne -> "ne"
                CmpOp.Le -> "le"
                CmpOp.Ge -> "ge"
                CmpOp.Lt -> "lt"
                CmpOp.Gt -> "gt"
            }
        return self.binary(name, rhs, lhsL, rhsL)
    }

    fun constSet(
        s: Scalar,
        l: Layout,
    ): Result<Unit> {
        fun <S> set(
            self_: MetalStorage,
            scalar: S,
            layout: Layout,
        ): Result<Unit> {
            val device = self_.device()
            val dtype = self_.dtype
            val shape = layout.shape()
            val elCount = shape.elemCount()
            val encoder = device.commandEncoder().getOrThrow()
            val dst = bufferO(self_.buffer, layout, self_.dtype)

            if (layout.isContiguous()) {
                val kernelName =
                    when (dtype) {
                        DType.F16 -> {
                            contiguous.constSet.HALF
                        }

                        DType.BF16 -> {
                            contiguous.constSet.BFLOAT
                        }

                        DType.F32 -> {
                            contiguous.constSet.FLOAT
                        }

                        DType.I64 -> {
                            contiguous.constSet.I64
                        }

                        DType.U32 -> {
                            contiguous.constSet.U32
                        }

                        DType.U8 -> {
                            contiguous.constSet.U8
                        }

                        DType.F8E4M3 -> {
                            bail("unsupported const-set f8e4m3")
                        }

                        DType.F64 -> {
                            bail("unsupported const-set f64")
                        }

                        DType.F4,
                        DType.F6E2M3,
                        DType.F6E3M2,
                        DType.F8E8M0,
                        DType.I16,
                        DType.I32,
                        -> {
                            return Err(CandleError.UnsupportedDTypeForOp(dtype, "const-set").bt())
                        }
                    }
                CandleMetalKernels
                    .callConstSetContiguous(
                        device.device,
                        encoder,
                        device.kernels,
                        kernelName,
                        dtype.sizeInBytes(),
                        elCount,
                        scalar,
                        dst,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
            } else {
                val kernelName =
                    when (dtype) {
                        DType.F16 -> {
                            strided.constSet.HALF
                        }

                        DType.BF16 -> {
                            strided.constSet.BFLOAT
                        }

                        DType.F32 -> {
                            strided.constSet.FLOAT
                        }

                        DType.I64 -> {
                            strided.constSet.I64
                        }

                        DType.U32 -> {
                            strided.constSet.U32
                        }

                        DType.U8 -> {
                            strided.constSet.U8
                        }

                        DType.F8E4M3 -> {
                            bail("unsupported const-set f8e4m3")
                        }

                        DType.F64 -> {
                            bail("unsupported const-set f64")
                        }

                        DType.F4,
                        DType.F6E2M3,
                        DType.F6E3M2,
                        DType.F8E8M0,
                        DType.I16,
                        DType.I32,
                        -> {
                            return Err(CandleError.UnsupportedDTypeForOp(dtype, "const-set").bt())
                        }
                    }
                CandleMetalKernels
                    .callConstSetStrided(
                        device.device,
                        encoder,
                        device.kernels,
                        kernelName,
                        layout.dims(),
                        scalar,
                        layout.stride(),
                        dst,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
            }
            return Ok(Unit)
        }
        return when {
            self.dtype == DType.U8 && s is Scalar.U8 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.U32 && s is Scalar.U32 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.I64 && s is Scalar.I64 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.F16 && s is Scalar.F16 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.BF16 && s is Scalar.BF16 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.F32 && s is Scalar.F32 -> {
                set(self, s.v, l)
            }

            self.dtype == DType.F64 && s is Scalar.F64 -> {
                set(self, s.v, l)
            }

            else -> {
                bail("dtype mismatch, expected ${self.dtype}, got $s")
            }
        }
    }

    fun toDtype(
        layout: Layout,
        dtype: DType,
    ): Result<MetalStorage> {
        val device = self.device()
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(elCount, dtype)
                .withLabel("to_dtype")
                .build()
                .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, self.dtype)
        if (layout.isContiguous()) {
            val kernelName =
                when (Pair(self.dtype, dtype)) {
                    Pair(DType.U32, DType.BF16) -> {
                        "cast_u32_bf16"
                    }

                    Pair(DType.U32, DType.F16) -> {
                        "cast_u32_f16"
                    }

                    Pair(DType.U32, DType.F32) -> {
                        "cast_u32_f32"
                    }

                    Pair(DType.U32, DType.I64) -> {
                        "cast_u32_i64"
                    }

                    Pair(DType.U32, DType.U8) -> {
                        "cast_u32_u8"
                    }

                    Pair(DType.U8, DType.BF16) -> {
                        "cast_u8_bf16"
                    }

                    Pair(DType.U8, DType.F16) -> {
                        "cast_u8_f16"
                    }

                    Pair(DType.U8, DType.F32) -> {
                        "cast_u8_f32"
                    }

                    Pair(DType.U8, DType.I64) -> {
                        "cast_u8_i64"
                    }

                    Pair(DType.U8, DType.U32) -> {
                        "cast_u8_u32"
                    }

                    Pair(DType.F32, DType.BF16) -> {
                        "cast_f32_bf16"
                    }

                    Pair(DType.F32, DType.F16) -> {
                        "cast_f32_f16"
                    }

                    Pair(DType.F32, DType.I64) -> {
                        "cast_f32_i64"
                    }

                    Pair(DType.F32, DType.U32) -> {
                        "cast_f32_u32"
                    }

                    Pair(DType.F32, DType.U8) -> {
                        "cast_f32_u8"
                    }

                    Pair(DType.I64, DType.BF16) -> {
                        "cast_i64_bf16"
                    }

                    Pair(DType.I64, DType.F16) -> {
                        "cast_i64_f16"
                    }

                    Pair(DType.I64, DType.F32) -> {
                        "cast_i64_f32"
                    }

                    Pair(DType.I64, DType.U32) -> {
                        "cast_i64_u32"
                    }

                    Pair(DType.I64, DType.U8) -> {
                        "cast_i64_u8"
                    }

                    Pair(DType.F16, DType.BF16) -> {
                        "cast_f16_bf16"
                    }

                    Pair(DType.F16, DType.F32) -> {
                        "cast_f16_f32"
                    }

                    Pair(DType.F16, DType.I64) -> {
                        "cast_f16_i64"
                    }

                    Pair(DType.F16, DType.U32) -> {
                        "cast_f16_u32"
                    }

                    Pair(DType.F16, DType.U8) -> {
                        "cast_f16_u8"
                    }

                    Pair(DType.BF16, DType.F16) -> {
                        "cast_bf16_f16"
                    }

                    Pair(DType.BF16, DType.F32) -> {
                        "cast_bf16_f32"
                    }

                    Pair(DType.BF16, DType.I64) -> {
                        "cast_bf16_i64"
                    }

                    Pair(DType.BF16, DType.U32) -> {
                        "cast_bf16_u32"
                    }

                    Pair(DType.BF16, DType.U8) -> {
                        "cast_bf16_u8"
                    }

                    else -> {
                        bail("Metal contiguous to_dtype ${self.dtype} $dtype not implemented")
                    }
                }
            CandleMetalKernels
                .callCastContiguous(
                    device.device,
                    encoder,
                    device.kernels,
                    kernelName,
                    self.dtype.sizeInBytes(),
                    elCount,
                    src,
                    buffer,
                ).mapErr(MetalError.from)
                .getOrThrow()
        } else {
            val kernelName =
                when (Pair(self.dtype, dtype)) {
                    Pair(DType.BF16, DType.F16) -> {
                        "cast_bf16_f16_strided"
                    }

                    Pair(DType.BF16, DType.F32) -> {
                        "cast_bf16_f32_strided"
                    }

                    Pair(DType.BF16, DType.I64) -> {
                        "cast_bf16_i64_strided"
                    }

                    Pair(DType.BF16, DType.U32) -> {
                        "cast_bf16_u32_strided"
                    }

                    Pair(DType.BF16, DType.U8) -> {
                        "cast_bf16_u8_strided"
                    }

                    Pair(DType.F16, DType.BF16) -> {
                        "cast_f16_bf16_strided"
                    }

                    Pair(DType.F16, DType.F32) -> {
                        "cast_f16_f32_strided"
                    }

                    Pair(DType.F16, DType.I64) -> {
                        "cast_f16_i64_strided"
                    }

                    Pair(DType.F16, DType.U32) -> {
                        "cast_f16_u32_strided"
                    }

                    Pair(DType.F16, DType.U8) -> {
                        "cast_f16_u8_strided"
                    }

                    Pair(DType.F32, DType.BF16) -> {
                        "cast_f32_bf16_strided"
                    }

                    Pair(DType.F32, DType.F16) -> {
                        "cast_f32_f16_strided"
                    }

                    Pair(DType.F32, DType.I64) -> {
                        "cast_f32_i64_strided"
                    }

                    Pair(DType.F32, DType.U32) -> {
                        "cast_f32_u32_strided"
                    }

                    Pair(DType.F32, DType.U8) -> {
                        "cast_f32_u8_strided"
                    }

                    Pair(DType.I64, DType.F32) -> {
                        "cast_i64_f32_strided"
                    }

                    Pair(DType.I64, DType.BF16) -> {
                        "cast_i64_bf16_strided"
                    }

                    Pair(DType.I64, DType.F16) -> {
                        "cast_i64_f16_strided"
                    }

                    Pair(DType.I64, DType.U32) -> {
                        "cast_i64_u32_strided"
                    }

                    Pair(DType.I64, DType.U8) -> {
                        "cast_i64_u8_strided"
                    }

                    Pair(DType.U32, DType.BF16) -> {
                        "cast_u32_bf16_strided"
                    }

                    Pair(DType.U32, DType.F16) -> {
                        "cast_u32_f16_strided"
                    }

                    Pair(DType.U32, DType.F32) -> {
                        "cast_u32_f32_strided"
                    }

                    Pair(DType.U32, DType.I64) -> {
                        "cast_u32_i64_strided"
                    }

                    Pair(DType.U32, DType.U8) -> {
                        "cast_u32_u8_strided"
                    }

                    Pair(DType.U8, DType.BF16) -> {
                        "cast_u8_bf16_strided"
                    }

                    Pair(DType.U8, DType.F16) -> {
                        "cast_u8_f16_strided"
                    }

                    Pair(DType.U8, DType.F32) -> {
                        "cast_u8_f32_strided"
                    }

                    Pair(DType.U8, DType.I64) -> {
                        "cast_u8_i64_strided"
                    }

                    Pair(DType.U8, DType.U32) -> {
                        "cast_u8_u32_strided"
                    }

                    else -> {
                        bail("Metal strided to_dtype ${self.dtype} $dtype not implemented")
                    }
                }
            CandleMetalKernels
                .callCastStrided(
                    device.device,
                    encoder,
                    device.kernels,
                    kernelName,
                    layout.dims(),
                    src,
                    layout.stride(),
                    buffer,
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Self.new(buffer, device.clone(), elCount, dtype))
    }

    fun <B : UnaryOpT> unaryImpl(
        layout: Layout,
        op: B,
    ): Result<MetalStorage> {
        val device = self.device()
        val dtype = self.dtype
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(elCount, dtype)
                .withLabel(op.kernel)
                .build()
                .getOrThrow()
        val encoder = device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, layout, self.dtype)

        if (layout.isContiguous()) {
            val kernelName =
                when (Pair(op.kernel, dtype)) {
                    Pair("uabs", DType.F16) -> {
                        contiguous.abs.HALF
                    }

                    Pair("uabs", DType.F32) -> {
                        contiguous.abs.FLOAT
                    }

                    Pair("uabs", DType.BF16) -> {
                        contiguous.abs.BFLOAT
                    }

                    Pair("uceil", DType.F16) -> {
                        contiguous.ceil.HALF
                    }

                    Pair("uceil", DType.F32) -> {
                        contiguous.ceil.FLOAT
                    }

                    Pair("uceil", DType.BF16) -> {
                        contiguous.ceil.BFLOAT
                    }

                    Pair("ucos", DType.F16) -> {
                        contiguous.cos.HALF
                    }

                    Pair("ucos", DType.F32) -> {
                        contiguous.cos.FLOAT
                    }

                    Pair("ucos", DType.BF16) -> {
                        contiguous.cos.BFLOAT
                    }

                    Pair("uerf", DType.F16) -> {
                        contiguous.erf.HALF
                    }

                    Pair("uerf", DType.F32) -> {
                        contiguous.erf.FLOAT
                    }

                    Pair("uerf", DType.BF16) -> {
                        contiguous.erf.BFLOAT
                    }

                    Pair("uexp", DType.F16) -> {
                        contiguous.exp.HALF
                    }

                    Pair("uexp", DType.F32) -> {
                        contiguous.exp.FLOAT
                    }

                    Pair("uexp", DType.BF16) -> {
                        contiguous.exp.BFLOAT
                    }

                    Pair("ufloor", DType.F16) -> {
                        contiguous.floor.HALF
                    }

                    Pair("ufloor", DType.F32) -> {
                        contiguous.floor.FLOAT
                    }

                    Pair("ufloor", DType.BF16) -> {
                        contiguous.floor.BFLOAT
                    }

                    Pair("ugelu_erf", DType.F16) -> {
                        contiguous.geluErf.HALF
                    }

                    Pair("ugelu_erf", DType.F32) -> {
                        contiguous.geluErf.FLOAT
                    }

                    Pair("ugelu_erf", DType.BF16) -> {
                        contiguous.geluErf.BFLOAT
                    }

                    Pair("ugelu", DType.F16) -> {
                        contiguous.gelu.HALF
                    }

                    Pair("ugelu", DType.F32) -> {
                        contiguous.gelu.FLOAT
                    }

                    Pair("ugelu", DType.BF16) -> {
                        contiguous.gelu.BFLOAT
                    }

                    Pair("ulog", DType.F16) -> {
                        contiguous.log.HALF
                    }

                    Pair("ulog", DType.F32) -> {
                        contiguous.log.FLOAT
                    }

                    Pair("ulog", DType.BF16) -> {
                        contiguous.log.BFLOAT
                    }

                    Pair("uneg", DType.F16) -> {
                        contiguous.neg.HALF
                    }

                    Pair("uneg", DType.F32) -> {
                        contiguous.neg.FLOAT
                    }

                    Pair("uneg", DType.BF16) -> {
                        contiguous.neg.BFLOAT
                    }

                    Pair("urecip", DType.F16) -> {
                        contiguous.recip.HALF
                    }

                    Pair("urecip", DType.F32) -> {
                        contiguous.recip.FLOAT
                    }

                    Pair("urecip", DType.BF16) -> {
                        contiguous.recip.BFLOAT
                    }

                    Pair("urelu", DType.F16) -> {
                        contiguous.relu.HALF
                    }

                    Pair("urelu", DType.F32) -> {
                        contiguous.relu.FLOAT
                    }

                    Pair("urelu", DType.BF16) -> {
                        contiguous.relu.BFLOAT
                    }

                    Pair("uround", DType.F16) -> {
                        contiguous.round.HALF
                    }

                    Pair("uround", DType.F32) -> {
                        contiguous.round.FLOAT
                    }

                    Pair("uround", DType.BF16) -> {
                        contiguous.round.BFLOAT
                    }

                    Pair("usilu", DType.F16) -> {
                        contiguous.silu.HALF
                    }

                    Pair("usilu", DType.F32) -> {
                        contiguous.silu.FLOAT
                    }

                    Pair("usilu", DType.BF16) -> {
                        contiguous.silu.BFLOAT
                    }

                    Pair("usin", DType.F16) -> {
                        contiguous.sin.HALF
                    }

                    Pair("usin", DType.F32) -> {
                        contiguous.sin.FLOAT
                    }

                    Pair("usin", DType.BF16) -> {
                        contiguous.sin.BFLOAT
                    }

                    Pair("usqr", DType.F16) -> {
                        contiguous.sqr.HALF
                    }

                    Pair("usqr", DType.F32) -> {
                        contiguous.sqr.FLOAT
                    }

                    Pair("usqr", DType.BF16) -> {
                        contiguous.sqr.BFLOAT
                    }

                    Pair("usqrt", DType.F16) -> {
                        contiguous.sqrt.HALF
                    }

                    Pair("usqrt", DType.F32) -> {
                        contiguous.sqrt.FLOAT
                    }

                    Pair("usqrt", DType.BF16) -> {
                        contiguous.sqrt.BFLOAT
                    }

                    Pair("utanh", DType.F16) -> {
                        contiguous.tanh.HALF
                    }

                    Pair("utanh", DType.F32) -> {
                        contiguous.tanh.FLOAT
                    }

                    Pair("utanh", DType.BF16) -> {
                        contiguous.tanh.BFLOAT
                    }

                    Pair("usign", DType.F16) -> {
                        contiguous.sign.HALF
                    }

                    Pair("usign", DType.F32) -> {
                        contiguous.sign.FLOAT
                    }

                    Pair("usign", DType.BF16) -> {
                        contiguous.sign.BFLOAT
                    }

                    Pair("usign", DType.I64) -> {
                        contiguous.sign.I64
                    }

                    else -> {
                        bail("Metal contiguous unary ${op.kernel} $dtype not implemented")
                    }
                }

            CandleMetalKernels
                .callUnaryContiguous(
                    device.device,
                    encoder,
                    device.kernels,
                    kernelName,
                    dtype.sizeInBytes(),
                    elCount,
                    src,
                    buffer,
                ).mapErr(MetalError.from)
                .getOrThrow()
        } else {
            val kernelName =
                when (Pair(op.kernel, dtype)) {
                    Pair("ucos", DType.F32) -> {
                        strided.cos.FLOAT
                    }

                    Pair("usin", DType.F32) -> {
                        strided.sin.FLOAT
                    }

                    Pair("usqr", DType.F32) -> {
                        strided.sqr.FLOAT
                    }

                    Pair("usqrt", DType.F32) -> {
                        strided.sqrt.FLOAT
                    }

                    Pair("uneg", DType.F32) -> {
                        strided.neg.FLOAT
                    }

                    Pair("uexp", DType.F32) -> {
                        strided.exp.FLOAT
                    }

                    Pair("ulog", DType.F32) -> {
                        strided.log.FLOAT
                    }

                    Pair("ugelu", DType.F32) -> {
                        strided.gelu.FLOAT
                    }

                    Pair("ugelu_erf", DType.F32) -> {
                        strided.geluErf.FLOAT
                    }

                    Pair("uerf", DType.F32) -> {
                        strided.erf.FLOAT
                    }

                    Pair("usilu", DType.F32) -> {
                        strided.silu.FLOAT
                    }

                    Pair("uabs", DType.F32) -> {
                        strided.abs.FLOAT
                    }

                    Pair("uceil", DType.F32) -> {
                        strided.ceil.FLOAT
                    }

                    Pair("ufloor", DType.F32) -> {
                        strided.floor.FLOAT
                    }

                    Pair("urelu", DType.F32) -> {
                        strided.relu.FLOAT
                    }

                    Pair("uround", DType.F32) -> {
                        strided.round.FLOAT
                    }

                    Pair("utanh", DType.F32) -> {
                        strided.tanh.FLOAT
                    }

                    Pair("ucos", DType.F16) -> {
                        strided.cos.HALF
                    }

                    Pair("usin", DType.F16) -> {
                        strided.sin.HALF
                    }

                    Pair("usqr", DType.F16) -> {
                        strided.sqr.HALF
                    }

                    Pair("usqrt", DType.F16) -> {
                        strided.sqrt.HALF
                    }

                    Pair("uneg", DType.F16) -> {
                        strided.neg.HALF
                    }

                    Pair("uexp", DType.F16) -> {
                        strided.exp.HALF
                    }

                    Pair("ulog", DType.F16) -> {
                        strided.log.HALF
                    }

                    Pair("ugelu", DType.F16) -> {
                        strided.gelu.HALF
                    }

                    Pair("ugelu_erf", DType.F16) -> {
                        strided.geluErf.HALF
                    }

                    Pair("uerf", DType.F16) -> {
                        strided.erf.HALF
                    }

                    Pair("usilu", DType.F16) -> {
                        strided.silu.HALF
                    }

                    Pair("uabs", DType.F16) -> {
                        strided.abs.HALF
                    }

                    Pair("uceil", DType.F16) -> {
                        strided.ceil.HALF
                    }

                    Pair("ufloor", DType.F16) -> {
                        strided.floor.HALF
                    }

                    Pair("urelu", DType.F16) -> {
                        strided.relu.HALF
                    }

                    Pair("uround", DType.F16) -> {
                        strided.round.HALF
                    }

                    Pair("utanh", DType.F16) -> {
                        strided.tanh.HALF
                    }

                    Pair("ucos", DType.BF16) -> {
                        strided.cos.BFLOAT
                    }

                    Pair("usin", DType.BF16) -> {
                        strided.sin.BFLOAT
                    }

                    Pair("usqr", DType.BF16) -> {
                        strided.sqr.BFLOAT
                    }

                    Pair("usqrt", DType.BF16) -> {
                        strided.sqrt.BFLOAT
                    }

                    Pair("uneg", DType.BF16) -> {
                        strided.neg.BFLOAT
                    }

                    Pair("uexp", DType.BF16) -> {
                        strided.exp.BFLOAT
                    }

                    Pair("ulog", DType.BF16) -> {
                        strided.log.BFLOAT
                    }

                    Pair("ugelu", DType.BF16) -> {
                        strided.gelu.BFLOAT
                    }

                    Pair("ugelu_erf", DType.BF16) -> {
                        strided.geluErf.BFLOAT
                    }

                    Pair("uerf", DType.BF16) -> {
                        strided.erf.BFLOAT
                    }

                    Pair("usilu", DType.BF16) -> {
                        strided.silu.BFLOAT
                    }

                    Pair("uabs", DType.BF16) -> {
                        strided.abs.BFLOAT
                    }

                    Pair("uceil", DType.BF16) -> {
                        strided.ceil.BFLOAT
                    }

                    Pair("ufloor", DType.BF16) -> {
                        strided.floor.BFLOAT
                    }

                    Pair("urelu", DType.BF16) -> {
                        strided.relu.BFLOAT
                    }

                    Pair("uround", DType.BF16) -> {
                        strided.round.BFLOAT
                    }

                    Pair("utanh", DType.BF16) -> {
                        strided.tanh.BFLOAT
                    }

                    else -> {
                        bail("Metal strided unary ${op.kernel} $dtype not implemented")
                    }
                }
            val dst = BufferOffset.zeroOffset(buffer)
            CandleMetalKernels
                .callUnaryStrided(
                    device.device,
                    encoder,
                    device.kernels,
                    kernelName,
                    layout.dims(),
                    src,
                    layout.stride(),
                    dst,
                ).mapErr(MetalError.from)
                .getOrThrow()
        }

        return Ok(Self.new(buffer, device.clone(), elCount, dtype))
    }

    fun <B : BinaryOpT> binaryImpl(
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
        op: B,
    ): Result<MetalStorage> = self.binary(op.kernel, rhs, lhsL, rhsL)

    fun whereCond(
        layout: Layout,
        t: MetalStorage,
        tL: Layout,
        f: MetalStorage,
        fL: Layout,
    ): Result<MetalStorage> {
        val device = self.device.clone()
        val shape = tL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val dtype = t.dtype
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(el, dtype)
                .withLabel("where")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        if (t.dtype() != f.dtype()) {
            bail(
                "Invalid where: different dtypes for values ${t.dtype()} != ${f.dtype()}",
            )
        }
        val name =
            when (Pair(self.dtype, t.dtype())) {
                Pair(DType.U8, DType.F32) -> {
                    "where_u8_f32"
                }

                Pair(DType.U32, DType.F32) -> {
                    "where_u32_f32"
                }

                Pair(DType.U8, DType.BF16) -> {
                    "where_u8_bf16"
                }

                Pair(DType.U8, DType.F16) -> {
                    "where_u8_f16"
                }

                Pair(DType.U8, DType.I64) -> {
                    "where_u8_i64"
                }

                Pair(DType.U8, DType.U32) -> {
                    "where_u8_u32"
                }

                Pair(DType.U8, DType.U8) -> {
                    "where_u8_u8"
                }

                else -> {
                    bail("Metal where_cond ${self.dtype} ${t.dtype()} not implemented")
                }
            }
        val src = bufferO(self.buffer, layout, self.dtype)
        val tBuf = bufferO(t.buffer, tL, t.dtype)
        val fBuf = bufferO(f.buffer, fL, f.dtype)
        CandleMetalKernels
            .callWhereCond(
                device.device,
                encoder,
                device.kernels,
                name,
                dtype.sizeInBytes(),
                dims,
                src,
                layout.stride(),
                layout.isContiguous(),
                tBuf,
                tL.stride(),
                tL.isContiguous(),
                fBuf,
                fL.stride(),
                fL.isContiguous(),
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(
            Self.new(buffer, device, el, dtype),
        )
    }

    fun conv1d(
        layout: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): Result<MetalStorage> {
        val device = self.device().clone()
        val shape = layout.shape()
        val dims = shape.dims()
        val strides = layout.stride()

        val stride = params.stride
        val dilation = params.dilation
        val padding = params.padding
        val kSize = params.kSize
        val lOut = (dims[2] + 2 * padding - dilation * (kSize - 1) - 1) / stride + 1
        val dstEl = dims[0] * lOut * dims[1] * kSize
        val dst =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("conv1d_im2col")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "im2col1d_f32"
                }

                DType.F16 -> {
                    "im2col1d_f16"
                }

                DType.BF16 -> {
                    "im2col1d_bf16"
                }

                DType.U8 -> {
                    "im2col1d_u8"
                }

                DType.U32 -> {
                    "im2col1d_u32"
                }

                else -> {
                    bail("Metal conv1d ${self.dtype} not implemented")
                }
            }
        val src = bufferO(self.buffer, layout, self.dtype)
        CandleMetalKernels
            .callIm2col1dStrided(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                layout.shape().dims(),
                strides,
                listOf(kSize, stride, padding, dilation),
                src,
                dst,
            ).mapErr(MetalError.from)
            .getOrThrow()
        drop(encoder)
        val col =
            MetalStorage(
                buffer = dst,
                device = device,
                count = dstEl,
                dtype = self.dtype,
            )
        val lOutParam = params.lOut()
        val b = params.bSize
        val n = params.cOut
        val k = params.kSize * params.cIn
        val m = lOutParam
        val colL = Layout.contiguous(listOf(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kL =
                    Layout
                        .contiguousWithOffset(listOf(1, n, k), kernelL.startOffset())
                        .transpose(1, 2)
                        .broadcastAs(listOf(b, k, n))
                        .getOrThrow()
                col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC = self.device().zerosImpl(kernelL.shape(), kernel.dtype()).getOrThrow()
                kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
                val kL =
                    Layout
                        .contiguousWithOffset(listOf(1, n, k), kernelL.startOffset())
                        .transpose(1, 2)
                        .broadcastAs(listOf(b, k, n))
                        .getOrThrow()
                col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
            }
        val resL = Layout.contiguous(listOf(b, lOutParam, n)).transpose(1, 2)
        val resT = self.device().zerosImpl(resL.shape(), res.dtype()).getOrThrow()
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        return Ok(resT)
    }

    fun convTranspose1d(
        layout: Layout,
        k: MetalStorage,
        kLayout: Layout,
        params: ParamsConvTranspose1D,
    ): Result<MetalStorage> {
        val useCol2imConv1dTr = true

        val canUseCol2im =
            kLayout.isContiguous() &&
                params.dilation == 1 &&
                params.padding == 0 &&
                params.outputPadding == 0
        val lOut = params.lOut()
        val dstEl = params.cOut * lOut * params.bSize

        val buffer =
            if (useCol2imConv1dTr && canUseCol2im) {
                val (bSize, cIn, lIn) = layout.shape().dims3().getOrThrow()
                val (cIn2, cOut, kSize) = kLayout.shape().dims3().getOrThrow()
                if (cIn != cIn2) {
                    bail(
                        "convtr1d: shape mismatch on c_in ${layout.shape()} ${kLayout.shape()}",
                    )
                }
                val buffer =
                    self
                        .device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, self.dtype)
                        .withLabel("conv_transpose1d")
                        .build()
                        .getOrThrow()

                val name =
                    when (self.dtype) {
                        DType.F32 -> "col2im1d_f32"
                        DType.F16 -> "col2im1d_f16"
                        DType.BF16 -> "col2im1d_bf16"
                        DType.U32 -> "col2im1d_u32"
                        DType.U8 -> "col2im1d_u8"
                        else -> bail("metal col2im1d ${self.dtype} not implemented")
                    }
                val col =
                    run {
                        // This merges the last two dimensions of the kernel together.
                        val kernelLMm =
                            Layout.new(
                                listOf(bSize, cIn, kSize * cOut).into(),
                                listOf(0, kSize * cOut, 1),
                                kLayout.startOffset(),
                            )
                        self
                            .matmul(
                                k,
                                listOf(bSize, lIn, cOut * kSize, cIn),
                                layout.transpose(1, 2),
                                kernelLMm,
                            ).getOrThrow()
                    }
                // It is important for the command encoder to be obtained *after* the matmul
                // kernel has run, otherwise we might use a command-buffer that has been committed
                // already resulting in the following error.
                // status < MTLCommandBufferStatusCommitted >
                // -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
                val encoder = self.device.commandEncoder().getOrThrow()
                CandleMetalKernels
                    .callCol2im1d(
                        self.device.device,
                        encoder,
                        self.device.kernels,
                        name,
                        listOf(bSize, lIn, cOut, kSize),
                        params.kSize,
                        params.stride,
                        BufferOffset.zeroOffset(col.buffer),
                        buffer,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
                buffer
            } else {
                val buffer =
                    self
                        .device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, self.dtype)
                        .withLabel("conv_transpose1d")
                        .build()
                        .getOrThrow()

                val encoder = self.device.commandEncoder().getOrThrow()
                val name =
                    when (self.dtype) {
                        DType.F32 -> {
                            "conv_transpose1d_f32"
                        }

                        DType.F16 -> {
                            "conv_transpose1d_f16"
                        }

                        DType.BF16 -> {
                            "conv_transpose1d_bf16"
                        }

                        DType.U32 -> {
                            "conv_transpose1d_u32"
                        }

                        DType.U8 -> {
                            "conv_transpose1d_u8"
                        }

                        else -> {
                            bail("Metal conv_transpose1d ${self.dtype} not implemented")
                        }
                    }
                CandleMetalKernels
                    .callConvTranspose1d(
                        self.device.device,
                        encoder,
                        self.device.kernels,
                        name,
                        params.dilation,
                        params.stride,
                        params.padding,
                        params.outputPadding,
                        params.cOut,
                        lOut,
                        params.bSize,
                        layout.dims(),
                        layout.stride(),
                        kLayout.dims(),
                        kLayout.stride(),
                        self.buffer,
                        layout.startOffset() * self.dtype.sizeInBytes(),
                        k.buffer,
                        kLayout.startOffset() * k.dtype.sizeInBytes(),
                        buffer,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
                buffer
            }
        return Ok(
            Self.new(buffer, self.device.clone(), dstEl, self.dtype),
        )
    }

    fun conv2d(
        layout: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): Result<MetalStorage> {
        val device = self.device().clone()
        val shape = layout.shape()
        val dims = shape.dims()

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

        val dst =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("conv2d_im2col")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "im2col_f32"
                }

                DType.F16 -> {
                    "im2col_f16"
                }

                DType.BF16 -> {
                    "im2col_bf16"
                }

                DType.U8 -> {
                    "im2col_u8"
                }

                DType.U32 -> {
                    "im2col_u32"
                }

                else -> {
                    bail("Metal conv2d ${self.dtype} not implemented")
                }
            }
        val src = bufferO(self.buffer, layout, self.dtype)
        CandleMetalKernels
            .callIm2colStrided(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                layout.shape().dims(),
                layout.stride(),
                listOf(hK, wK, stride, padding, dilation),
                src,
                dst,
            ).mapErr(MetalError.from)
            .getOrThrow()
        drop(encoder)
        val col =
            MetalStorage(
                buffer = dst,
                device = device,
                count = dstEl,
                dtype = self.dtype,
            )
        val hOutParam = params.outH()
        val wOutParam = params.outW()
        val b = params.bSize
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val m = hOutParam * wOutParam
        val colL = Layout.contiguous(listOf(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kL =
                    Layout
                        .contiguousWithOffset(listOf(1, n, k), kernelL.startOffset())
                        .transpose(1, 2)
                        .broadcastAs(listOf(b, k, n))
                        .getOrThrow()
                col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC = self.device().zerosImpl(kernelL.shape(), kernel.dtype()).getOrThrow()
                kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
                val kL =
                    Layout
                        .contiguousWithOffset(listOf(1, n, k), kernelL.startOffset())
                        .transpose(1, 2)
                        .broadcastAs(listOf(b, k, n))
                        .getOrThrow()
                col.matmul(kernel, listOf(b, m, n, k), colL, kL).getOrThrow()
            }
        val resL =
            Layout
                .contiguous(listOf(b, hOutParam, wOutParam, n))
                .transpose(1, 2)
                .transpose(1, 3)
        val resT = self.device().zerosImpl(resL.shape(), res.dtype()).getOrThrow()
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        return Ok(resT)
    }

    fun convTranspose2d(
        l: Layout,
        kernel: MetalStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): Result<MetalStorage> {
        // Kernel shape: (cInK, cOut, hK, wK)
        // Input shape: (bSize, cIn, hIn, wIn)
        val (outW, outH) = Pair(params.outW(), params.outH())
        val dstEl = params.cOut * outW * outH * params.bSize

        val dims = l.dims()
        if (dims.len() != 4) {
            bail("unexpected input shape for conv_transpose2d $dims, expected 4")
        }

        val kDims = kernelL.dims()
        if (kDims.len() != 4) {
            bail("unexpected kernel shape for conv_transpose2d $kDims, expected 4")
        }

        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("conv_transpose2d")
                .build()
                .getOrThrow()

        val encoder = self.device.commandEncoder().getOrThrow()

        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "conv_transpose2d_f32"
                }

                DType.F16 -> {
                    "conv_transpose2d_f16"
                }

                DType.BF16 -> {
                    "conv_transpose2d_bf16"
                }

                else -> {
                    bail("Metal conv_transpose2d ${self.dtype} not implemented")
                }
            }

        CandleMetalKernels
            .callConvTranspose2d(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                CallConvTranspose2dCfg(
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
                    inputOffset = l.startOffset() * self.dtype.sizeInBytes(),
                    kernelOffset = kernelL.startOffset() * kernel.dtype.sizeInBytes(),
                ),
                self.buffer,
                kernel.buffer,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Self.new(buffer, self.device.clone(), dstEl, self.dtype))
    }

    fun avgPool2d(
        inpL: Layout,
        k: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Result<MetalStorage> {
        val (wK, hK) = k
        val (wStride, hStride) = stride
        val shape = inpL.shape()
        val d = shape.dims4().getOrThrow()
        val bSize = d[0]
        val channels = d[1]
        val width = d[2]
        val height = d[3]
        val strides = inpL.stride()
        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "avg_pool2d_f32"
                }

                DType.F16 -> {
                    "avg_pool2d_f16"
                }

                DType.BF16 -> {
                    "avg_pool2d_bf16"
                }

                DType.U8 -> {
                    "avg_pool2d_u8"
                }

                DType.U32 -> {
                    "avg_pool2d_u32"
                }

                else -> {
                    bail("Metal avg_pool2d ${self.dtype} not implemented")
                }
            }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("avg_pool2d")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        CandleMetalKernels
            .callPool2d(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                inpL.dims(),
                strides,
                outW,
                outH,
                wK,
                hK,
                wStride,
                hStride,
                self.buffer,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(
            Self.new(buffer, self.device.clone(), dstEl, self.dtype),
        )
    }

    fun maxPool2d(
        inpL: Layout,
        k: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Result<MetalStorage> {
        val (wK, hK) = k
        val (wStride, hStride) = stride
        val shape = inpL.shape()
        val d = shape.dims4().getOrThrow()
        val bSize = d[0]
        val channels = d[1]
        val width = d[2]
        val height = d[3]
        val strides = inpL.stride()
        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "max_pool2d_f32"
                }

                DType.F16 -> {
                    "max_pool2d_f16"
                }

                DType.BF16 -> {
                    "max_pool2d_bf16"
                }

                DType.U8 -> {
                    "max_pool2d_u8"
                }

                DType.U32 -> {
                    "max_pool2d_u32"
                }

                else -> {
                    bail("Metal max_pool2d ${self.dtype} not implemented")
                }
            }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("max_pool2d")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        CandleMetalKernels
            .callPool2d(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                inpL.dims(),
                strides,
                outW,
                outH,
                wK,
                hK,
                wStride,
                hStride,
                self.buffer,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(
            Self.new(buffer, self.device.clone(), dstEl, self.dtype),
        )
    }

    fun upsampleNearest1d(
        layout: Layout,
        sz: Int,
    ): Result<MetalStorage> {
        if (layout.startOffset() < 0 || sz < 0) bail("invalid upsample_nearest1d args")
        bail("Metal upsample_nearest1d not implemented")
    }

    fun upsampleNearest2d(
        inpL: Layout,
        outW: Int,
        outH: Int,
    ): Result<MetalStorage> {
        // Slice input from start offset.
        val shape = inpL.shape()
        val dims = shape.dims()
        val strides = inpL.stride()
        if (dims.len() != 4) {
            bail("unexpected input shape for upsample $dims")
        }
        val name =
            when (self.dtype) {
                DType.F32 -> "upsample_nearest2d_f32"
                DType.F16 -> "upsample_nearest2d_f16"
                DType.BF16 -> "upsample_nearest2d_bf16"
                DType.U8 -> "upsample_nearest2d_u8"
                DType.U32 -> "upsample_nearest2d_u32"
                else -> bail("Metal upsample_nearest2d ${self.dtype} not implemented")
            }

        val dstEl = outW * outH * dims[0] * dims[1]
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("upsample_nearest2d")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, inpL, self.dtype)
        CandleMetalKernels
            .callUpsampleNearest2d(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                dims,
                strides,
                outW,
                outH,
                src,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Self.new(buffer, self.device.clone(), dstEl, self.dtype))
    }

    fun upsampleBilinear2d(
        inpL: Layout,
        outH: Int,
        outW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?,
    ): Result<MetalStorage> {
        val shape = inpL.shape()
        val dims = shape.dims()
        val strides = inpL.stride()

        if (dims.len() != 4) {
            bail("unexpected input shape for upsample_bilinear2d $dims")
        }

        val name =
            when (self.dtype) {
                DType.F32 -> {
                    "upsample_bilinear2d_f32"
                }

                DType.F16 -> {
                    "upsample_bilinear2d_f16"
                }

                DType.BF16 -> {
                    "upsample_bilinear2d_bf16"
                }

                DType.U8 -> {
                    "upsample_bilinear2d_u8"
                }

                DType.U32 -> {
                    "upsample_bilinear2d_u32"
                }

                else -> {
                    bail("Metal upsample_bilinear2d ${self.dtype} not implemented")
                }
            }

        val dstEl = outW * outH * dims[0] * dims[1]
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(dstEl, self.dtype)
                .withLabel("upsample_bilinear2d")
                .build()
                .getOrThrow()

        val encoder = self.device.commandEncoder().getOrThrow()

        val src = bufferO(self.buffer, inpL, self.dtype)
        CandleMetalKernels
            .callUpsampleBilinear2d(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                dims,
                strides,
                outW,
                outH,
                alignCorners,
                scaleH,
                scaleW,
                src,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()

        return Ok(Self.new(buffer, self.device.clone(), dstEl, self.dtype))
    }

    fun gather(
        srcL: Layout,
        ids: MetalStorage,
        idsL: Layout,
        dim: Int,
    ): Result<MetalStorage> {
        if (!idsL.isContiguous()) {
            return Err(CandleError.RequiresContiguous(op = "gather").bt())
        }
        val idsEl = idsL.dims()[dim]
        val dstEl = idsL.shape().elemCount()
        val dtype = self.dtype
        val device = self.device()
        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(dstEl, dtype)
                .withLabel("gather")
                .build()
                .getOrThrow()
        val name =
            when (Pair(ids.dtype, self.dtype)) {
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
                else -> bail("Metal gather ${ids.dtype} ${self.dtype} not implemented")
            }
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, srcL, dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        CandleMetalKernels
            .callGather(
                device.device,
                encoder,
                self.device.kernels,
                name,
                srcL.dims(),
                idsEl,
                dim,
                src,
                idsBuf,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Self.new(buffer, device.clone(), dstEl, dtype))
    }

    fun scatterSet(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int,
    ): Result<Unit> {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            return Err(CandleError.RequiresContiguous(op = "scatter").bt())
        }
        val name =
            when (Pair(ids.dtype, self.dtype)) {
                Pair(DType.U8, DType.F32) -> {
                    "s_u8_f32"
                }

                Pair(DType.U8, DType.F16) -> {
                    "s_u8_f16"
                }

                Pair(DType.U8, DType.BF16) -> {
                    "s_u8_bf16"
                }

                Pair(DType.U32, DType.U32) -> {
                    "s_u32_u32"
                }

                Pair(DType.U32, DType.F32) -> {
                    "s_u32_f32"
                }

                Pair(DType.U32, DType.F16) -> {
                    "s_u32_f16"
                }

                Pair(DType.U32, DType.BF16) -> {
                    "s_u32_bf16"
                }

                Pair(DType.I64, DType.F32) -> {
                    "s_i64_f32"
                }

                Pair(DType.I64, DType.F16) -> {
                    "s_i64_f16"
                }

                Pair(DType.I64, DType.BF16) -> {
                    "s_i64_bf16"
                }

                else -> {
                    Err(
                        MetalError.UnexpectedDType(
                            msg = "scatter ids should be u8/u32/i64",
                            expected = DType.U32,
                            got = ids.dtype(),
                        ),
                    ).getOrThrow()
                }
            }
        val encoder = self.device.commandEncoder().getOrThrow()
        val dst = bufferO(self.buffer, l, self.dtype)
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        CandleMetalKernels
            .callScatter(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                srcL.dims(),
                l.dims(),
                dim,
                srcBuf,
                idsBuf,
                dst,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Unit)
    }

    fun scatterAddSet(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int,
    ): Result<Unit> {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            return Err(CandleError.RequiresContiguous(op = "scatter-add").bt())
        }
        val name =
            when (Pair(ids.dtype, self.dtype)) {
                Pair(DType.U8, DType.F32) -> {
                    "sa_u8_f32"
                }

                Pair(DType.U8, DType.F16) -> {
                    "sa_u8_f16"
                }

                Pair(DType.U8, DType.BF16) -> {
                    "sa_u8_bf16"
                }

                Pair(DType.U32, DType.U32) -> {
                    "sa_u32_u32"
                }

                Pair(DType.U32, DType.F32) -> {
                    "sa_u32_f32"
                }

                Pair(DType.U32, DType.F16) -> {
                    "sa_u32_f16"
                }

                Pair(DType.U32, DType.BF16) -> {
                    "sa_u32_bf16"
                }

                Pair(DType.I64, DType.F32) -> {
                    "sa_i64_f32"
                }

                Pair(DType.I64, DType.F16) -> {
                    "sa_i64_f16"
                }

                Pair(DType.I64, DType.BF16) -> {
                    "sa_i64_bf16"
                }

                else -> {
                    Err(
                        MetalError.UnexpectedDType(
                            msg = "scatter-add ids should be u8/u32/i64",
                            expected = DType.U32,
                            got = ids.dtype(),
                        ),
                    ).getOrThrow()
                }
            }
        val encoder = self.device.commandEncoder().getOrThrow()
        val dst = bufferO(self.buffer, l, self.dtype)
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        CandleMetalKernels
            .callScatter(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                srcL.dims(),
                l.dims(),
                dim,
                srcBuf,
                idsBuf,
                dst,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Unit)
    }

    fun indexSelect(
        ids: MetalStorage,
        srcL: Layout,
        idsL: Layout,
        dim: Int,
    ): Result<MetalStorage> {
        if (!idsL.isContiguous()) {
            bail("Metal index_select requires contiguous ids")
        }
        val leftSize =
            srcL
                .dims()
                .subList(0, dim)
                .iter()
                .product()
        val rightSize =
            srcL
                .dims()
                .subList(dim + 1, srcL.dims().size)
                .iter()
                .product()
        val idsEl = idsL.shape().elemCount()
        val dstEl = idsEl * leftSize * rightSize
        val dtype = self.dtype
        val device = self.device()
        val buffer =
            device
                .newBufferBuilder()
                .withSizeFor(dstEl, dtype)
                .withLabel("index_select")
                .build()
                .getOrThrow()
        val name =
            when (Pair(ids.dtype, self.dtype)) {
                Pair(DType.U8, DType.U8) -> {
                    "is_u8_u8"
                }

                Pair(DType.U8, DType.U32) -> {
                    "is_u8_u32"
                }

                Pair(DType.U8, DType.I64) -> {
                    "is_u8_i64"
                }

                Pair(DType.U8, DType.BF16) -> {
                    "is_u8_bf16"
                }

                Pair(DType.U8, DType.F32) -> {
                    "is_u8_f32"
                }

                Pair(DType.U8, DType.F16) -> {
                    "is_u8_f16"
                }

                Pair(DType.U32, DType.U8) -> {
                    "is_u32_u8"
                }

                Pair(DType.U32, DType.U32) -> {
                    "is_u32_u32"
                }

                Pair(DType.U32, DType.I64) -> {
                    "is_u32_i64"
                }

                Pair(DType.U32, DType.F32) -> {
                    "is_u32_f32"
                }

                Pair(DType.U32, DType.F16) -> {
                    "is_u32_f16"
                }

                Pair(DType.U32, DType.BF16) -> {
                    "is_u32_bf16"
                }

                Pair(DType.I64, DType.U8) -> {
                    "is_i64_u8"
                }

                Pair(DType.I64, DType.U32) -> {
                    "is_i64_u32"
                }

                Pair(DType.I64, DType.I64) -> {
                    "is_i64_i64"
                }

                Pair(DType.I64, DType.F32) -> {
                    "is_i64_f32"
                }

                Pair(DType.I64, DType.F16) -> {
                    "is_i64_f16"
                }

                Pair(DType.I64, DType.BF16) -> {
                    "is_i64_bf16"
                }

                else -> {
                    bail("Metal contiguous index_select ${ids.dtype} ${self.dtype} not implemented")
                }
            }
        val encoder = self.device.commandEncoder().getOrThrow()
        val src = bufferO(self.buffer, srcL, dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        CandleMetalKernels
            .callIndexSelect(
                device.device,
                encoder,
                self.device.kernels,
                name,
                srcL.dims(),
                idsEl,
                dim,
                srcL.isContiguous(),
                srcL.dims(),
                srcL.stride(),
                src,
                idsBuf,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(Self.new(buffer, device.clone(), dstEl, dtype))
    }

    fun indexAdd(
        l: Layout,
        ids: MetalStorage,
        idsL: Layout,
        src: MetalStorage,
        srcL: Layout,
        dim: Int,
    ): Result<MetalStorage> {
        val acc = self.device.zerosImpl(l.shape(), self.dtype()).getOrThrow()
        self.copyStridedSrc(acc, 0, l).getOrThrow()
        if (!idsL.isContiguous() || !srcL.isContiguous()) {
            return Err(CandleError.RequiresContiguous(op = "index-add").bt())
        }
        val name =
            when (Pair(ids.dtype, self.dtype)) {
                Pair(DType.I64, DType.BF16) -> {
                    "ia_i64_bf16"
                }

                Pair(DType.I64, DType.F16) -> {
                    "ia_i64_f16"
                }

                Pair(DType.I64, DType.F32) -> {
                    "ia_i64_f32"
                }

                Pair(DType.I64, DType.I64) -> {
                    "ia_i64_i64"
                }

                Pair(DType.I64, DType.U32) -> {
                    "ia_i64_u32"
                }

                Pair(DType.I64, DType.U8) -> {
                    "ia_i64_u8"
                }

                Pair(DType.U32, DType.BF16) -> {
                    "ia_u32_bf16"
                }

                Pair(DType.U32, DType.F16) -> {
                    "ia_u32_f16"
                }

                Pair(DType.U32, DType.F32) -> {
                    "ia_u32_f32"
                }

                Pair(DType.U32, DType.I64) -> {
                    "ia_u32_i64"
                }

                Pair(DType.U32, DType.U32) -> {
                    "ia_u32_u32"
                }

                Pair(DType.U32, DType.U8) -> {
                    "ia_u32_u8"
                }

                Pair(DType.U8, DType.BF16) -> {
                    "ia_u8_bf16"
                }

                Pair(DType.U8, DType.F16) -> {
                    "ia_u8_f16"
                }

                Pair(DType.U8, DType.F32) -> {
                    "ia_u8_f32"
                }

                Pair(DType.U8, DType.I64) -> {
                    "ia_u8_i64"
                }

                Pair(DType.U8, DType.U32) -> {
                    "ia_u8_u32"
                }

                Pair(DType.U8, DType.U8) -> {
                    "ia_u8_u8"
                }

                else -> {
                    Err(
                        MetalError.UnexpectedDType(
                            msg = "index-add ids should be u8/u32/i64",
                            expected = DType.U32,
                            got = ids.dtype(),
                        ),
                    ).getOrThrow()
                }
            }
        val encoder = self.device.commandEncoder().getOrThrow()
        val srcBuf = bufferO(src.buffer, srcL, src.dtype)
        val idsBuf = bufferO(ids.buffer, idsL, ids.dtype)
        CandleMetalKernels
            .callIndexAdd(
                self.device.device,
                encoder,
                self.device.kernels,
                name,
                srcL.dims(),
                l.dims(),
                idsL.dims(),
                dim,
                srcBuf,
                idsBuf,
                acc.buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()
        return Ok(acc)
    }

    fun matmul(
        rhs: MetalStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): Result<MetalStorage> {
        val b = bmnk[0]
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSizeFor(b * m * n, self.dtype)
                .withLabel("matmul")
                .build()
                .getOrThrow()
        val encoder = self.device.commandEncoder().getOrThrow()
        val dtype =
            when (self.dtype) {
                DType.F32 -> {
                    CandleMetalKernels.GemmDType.F32
                }

                DType.F16 -> {
                    CandleMetalKernels.GemmDType.F16
                }

                DType.BF16 -> {
                    CandleMetalKernels.GemmDType.BF16
                }

                else -> {
                    return Err(
                        MetalError.Message(format("mlx matmul doesn't support ${self.dtype}")).into(),
                    )
                }
            }
        CandleMetalKernels
            .callMlxGemm(
                self.device.device,
                encoder,
                self.device.kernels,
                dtype,
                listOf(b, m, n, k),
                lhsL.stride(),
                lhsL.startOffset() * self.dtype.sizeInBytes(),
                self.buffer,
                rhsL.stride(),
                rhsL.startOffset() * rhs.dtype.sizeInBytes(),
                rhs.buffer,
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()

        return Ok(
            Self.new(
                buffer,
                self.device.clone(),
                b * m * n,
                self.dtype(),
            ),
        )
    }

    fun copy2d(
        dst: MetalStorage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int,
    ): Result<Unit> {
        if (self.dtype() != dst.dtype()) {
            bail(
                "copy2d with inconsistent dtypes ${self.dtype()} ${dst.dtype()}",
            )
        }
        if (srcS == d2 && dstS == d2) {
            val blit = self.device.blitCommandEncoder().getOrThrow()
            blit.setLabel("copy2d_contiguous")
            val srcOffset = srcO * self.dtype.sizeInBytes()
            val length = d1 * d2 * self.dtype.sizeInBytes()
            val dstOffset = dstO * dst.dtype().sizeInBytes()
            blit.copyFromBuffer(
                self.buffer,
                srcOffset,
                dst.buffer(),
                dstOffset,
                length,
            )
        } else {
            val elCount = d1 * d2
            if (elCount == 0) {
                return Ok(Unit)
            }
            val kernelName =
                when (self.dtype) {
                    DType.F32 -> {
                        CandleMetalKernels.copy2d.FLOAT
                    }

                    DType.F16 -> {
                        CandleMetalKernels.copy2d.HALF
                    }

                    DType.BF16 -> {
                        CandleMetalKernels.copy2d.BFLOAT
                    }

                    DType.I64 -> {
                        CandleMetalKernels.copy2d.I64
                    }

                    DType.I32 -> {
                        CandleMetalKernels.copy2d.I32
                    }

                    DType.I16 -> {
                        CandleMetalKernels.copy2d.I16
                    }

                    DType.U32 -> {
                        CandleMetalKernels.copy2d.U32
                    }

                    DType.U8 -> {
                        CandleMetalKernels.copy2d.U8
                    }

                    else -> {
                        bail("Metal copy2d ${self.dtype} not implemented")
                    }
                }
            val encoder = self.device.commandEncoder().getOrThrow()
            CandleMetalKernels
                .callCopy2d(
                    self.device.device,
                    encoder,
                    self.device.kernels,
                    kernelName,
                    self.buffer,
                    dst.buffer,
                    d1,
                    d2,
                    srcS,
                    dstS,
                    srcO * self.dtype.sizeInBytes(),
                    dstO * self.dtype.sizeInBytes(),
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Unit)
    }

    fun copyStridedSrc(
        dst: MetalStorage,
        dstOffset: Int,
        srcL: Layout,
    ): Result<Unit> {
        if (srcL.isContiguous() && self.dtype == dst.dtype()) {
            val blit = self.device.blitCommandEncoder().getOrThrow()
            blit.setLabel("copy_contiguous")
            val srcOffset = srcL.startOffset() * self.dtype.sizeInBytes()
            val length = srcL.shape().elemCount() * self.dtype.sizeInBytes()
            val dstByteOffset = dstOffset * dst.dtype().sizeInBytes()
            blit.copyFromBuffer(self.buffer, srcOffset, dst.buffer(), dstByteOffset, length)
        } else {
            val srcShape = srcL.shape()
            val elCount = srcShape.elemCount()
            if (elCount == 0) {
                return Ok(Unit)
            }
            if (self.dtype == dst.dtype()) {
                val strides = srcL.stride()
                val ndim = strides.len()
                // Inner blocks are contiguous - eligible for copy2d
                if (ndim == 1 || strides[ndim - 1] == 1) {
                    val blocks = srcL.stridedBlocks()
                    if (blocks is StridedBlocks.UniformBlocks) {
                        val startOffset = blocks.startOffset
                        val blockLen = blocks.blockLen
                        val count = blocks.count
                        val srcStride = blocks.srcStride
                        return self.copy2d(
                            dst,
                            count,
                            blockLen,
                            srcStride,
                            blockLen,
                            startOffset,
                            dstOffset,
                        )
                    }
                }
            }
            val kernelName =
                when (self.dtype) {
                    DType.F32 -> {
                        CandleMetalKernels.unary.strided.copy.FLOAT
                    }

                    DType.F16 -> {
                        CandleMetalKernels.unary.strided.copy.HALF
                    }

                    DType.BF16 -> {
                        CandleMetalKernels.unary.strided.copy.BFLOAT
                    }

                    DType.I64 -> {
                        CandleMetalKernels.unary.strided.copy.I64
                    }

                    DType.U32 -> {
                        CandleMetalKernels.unary.strided.copy.U32
                    }

                    DType.U8 -> {
                        CandleMetalKernels.unary.strided.copy.U8
                    }

                    else -> {
                        bail("Metal copy_strided ${self.dtype} not implemented")
                    }
                }
            val src = bufferO(self.buffer, srcL, self.dtype)
            val dstBuf =
                BufferOffset(
                    buffer = dst.buffer,
                    offsetInBytes = dstOffset * dst.dtype.sizeInBytes(),
                )
            val encoder = self.device.commandEncoder().getOrThrow()
            CandleMetalKernels
                .callUnaryStrided(
                    self.device.device,
                    encoder,
                    self.device.kernels,
                    kernelName,
                    srcL.dims(),
                    src,
                    srcL.stride(),
                    dstBuf,
                ).mapErr(MetalError.from)
                .getOrThrow()
        }
        return Ok(Unit)
    }

    companion object {
        fun new(
            buffer: Buffer,
            device: MetalDevice,
            count: Int,
            dtype: DType,
        ): MetalStorage =
            MetalStorage(
                buffer = buffer,
                device = device,
                count = count,
                dtype = dtype,
            )
    }

    fun buffer(): Buffer = self.buffer

    fun binary(
        op: String,
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): Result<MetalStorage> {
        fun kernelName(
            opName: String,
            dtype: DType,
            suffix: String,
        ): String = format("${opName}_${dtype.asStr()}$suffix")
        val device = self.device()
        val shape = lhsL.shape()
        val elCount = shape.elemCount()
        val encoder = device.commandEncoder().getOrThrow()
        val lhs = bufferO(self.buffer, lhsL, self.dtype)
        val rhsBuf = bufferO(rhs.buffer, rhsL, rhs.dtype)

        val dtype =
            when (op) {
                "eq", "ne", "le", "lt", "ge", "gt" -> DType.U8
                else -> self.dtype
            }
        val lhsIsScalar = lhsL.isScalarLike()
        val rhsIsScalar = rhsL.isScalarLike()
        val lhsContiguous = lhsL.isContiguous()
        val rhsContiguous = rhsL.isContiguous()

        val contiguousKernel = kernelName(op, self.dtype, "")
        val kernel =
            when {
                lhsIsScalar && rhsIsScalar -> {
                    kernelName(op, self.dtype, "_scalar")
                }

                lhsIsScalar && !rhsIsScalar && rhsContiguous -> {
                    kernelName(op, self.dtype, "_sc")
                }

                lhsIsScalar && !rhsIsScalar && !rhsContiguous -> {
                    kernelName(op, self.dtype, "_rss")
                }

                !lhsIsScalar && rhsIsScalar && lhsContiguous -> {
                    kernelName(op, self.dtype, "_cs")
                }

                !lhsIsScalar && rhsIsScalar && !lhsContiguous -> {
                    kernelName(op, self.dtype, "_lss")
                }

                !lhsIsScalar && !rhsIsScalar && lhsContiguous && rhsContiguous -> {
                    contiguousKernel.clone()
                }

                !lhsIsScalar && !rhsIsScalar && lhsContiguous && !rhsContiguous -> {
                    kernelName(op, self.dtype, "_rstrided")
                }

                !lhsIsScalar && !rhsIsScalar && !lhsContiguous && rhsContiguous -> {
                    kernelName(op, self.dtype, "_lstrided")
                }

                else -> {
                    kernelName(op, self.dtype, "_strided")
                }
            }

        val buffer =
            if (kernel == contiguousKernel) {
                val buffer =
                    device
                        .newBufferBuilder()
                        .withSizeFor(elCount, dtype)
                        .withLabel(op)
                        .build()
                        .getOrThrow()
                CandleMetalKernels
                    .callBinaryContiguous(
                        device.device,
                        encoder,
                        device.kernels,
                        kernel,
                        self.dtype.sizeInBytes(),
                        elCount,
                        lhs,
                        rhsBuf,
                        buffer,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
                buffer
            } else {
                val buffer =
                    device
                        .newBufferBuilder()
                        .withSizeFor(elCount, dtype)
                        .withLabel(op)
                        .build()
                        .getOrThrow()
                CandleMetalKernels
                    .callBinaryStrided(
                        device.device,
                        encoder,
                        device.kernels,
                        kernel,
                        self.dtype.sizeInBytes(),
                        lhsL.dims(),
                        lhs,
                        lhsL.stride(),
                        rhsBuf,
                        rhsL.stride(),
                        buffer,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
                buffer
            }
        return Ok(Self.new(buffer, device.clone(), elCount, dtype))
    }

    internal fun <T> toCpu(): Result<List<T>> {
        val size = self.count * self.dtype.sizeInBytes()
        val buffer =
            self
                .device
                .newBufferBuilder()
                .withSize(size)
                .withLabel("blit_to_cpu_dst")
                .build()
                .getOrThrow()
        run {
            val blit = self.device.blitCommandEncoder().getOrThrow()
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(self.buffer, 0, buffer, 0, size)
        }
        self.device.flushAndWaitCurrent().getOrThrow()
        return Ok(readToVec(buffer, self.count))
    }
}

class MetalBackendDevice(
    val inner: MetalDevice,
) {
    interface Storage

    companion object {
        fun new(ordinal: Int): Result<MetalDevice> {
            val device = Device.all().swapRemove(ordinal)
            val commandQueue = device.newCommandQueue().mapErr(MetalError.from).getOrThrow()
            commandQueue.setLabel(Some(NSString.fromStr("candle-metal")))
            val kernels = Arc.new(Kernels.new())
            val residencySet = Arc.new(ResidencySet.new(device))
            val seedBuf =
                device
                    .newBufferWithData(
                        listOf(299792458uL).asPtr(),
                        sizeOfU64,
                        RESOURCE_OPTIONS,
                    ).mapErr(MetalError.from)
                    .getOrThrow()
            seedBuf.setLabel("rng_seed")
            residencySet.insert(seedBuf)
            val seed = Arc.new(Mutex.new(seedBuf))
            val commands =
                Commands
                    .new(commandQueue, residencySet)
                    .mapErr(MetalError.from)
                    .getOrThrow()
            return Ok(
                MetalDevice(
                    id = DeviceId.new(),
                    device = device,
                    commands = Arc.new(commands),
                    buffers = Arc.new(RwLock.new(HashMap.newMap())),
                    privateBuffers = Arc.new(RwLock.new(HashMap.newMap())),
                    kernels = kernels,
                    seed = seed,
                    seedValue = Arc.new(RwLock.new(299792458uL)),
                    residencySet = residencySet,
                ),
            )
        }
    }

    fun location(): DeviceLocation =
        DeviceLocation.Metal(
            gpuId = self.registryId().toInt(),
        )

    fun sameDevice(rhs: MetalDevice): Boolean = self.id == rhs.id

    fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): Result<MetalStorage> {
        val buffer =
            self
                .newBufferBuilder()
                .withSizeFor(shape.elemCount(), dtype)
                .withLabel("alloc-uninit")
                .build()
                .getOrThrow()
        return Ok(
            MetalStorage.new(
                buffer,
                self.clone(),
                shape.elemCount(),
                dtype,
            ),
        )
    }

    fun zerosImpl(
        shape: Shape,
        dtype: DType,
    ): Result<MetalStorage> {
        val size = shape.elemCount() * dtype.sizeInBytes()
        val buffer =
            self
                .newBufferBuilder()
                .withZeros(size)
                .withLabel("zeros")
                .build()
                .getOrThrow()
        return Ok(
            MetalStorage.new(
                buffer,
                self.clone(),
                shape.elemCount(),
                dtype,
            ),
        )
    }

    fun <T> storageFromSlice(
        s: List<T>,
        dtype: DType,
        cpuStorageRef: (List<T>) -> CpuStorageRef,
    ): Result<MetalStorage> {
        val label = "storage_from_slice"
        val (count, buffer) =
            when (val ref = cpuStorageRef(s)) {
                is CpuStorageRef.U8 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.U32 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.I16 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.I32 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.I64 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.BF16 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.F16 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.F32 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.F64 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.F8E4M3 -> {
                    Pair(
                        ref.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(ref.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorageRef.F6E2M3,
                is CpuStorageRef.F6E3M2,
                is CpuStorageRef.F4,
                is CpuStorageRef.F8E8M0,
                -> {
                    return Err(CandleError.UnsupportedDTypeForOp(dtype, "to_dtype").bt())
                }
            }
        return Ok(Self.Storage.new(buffer.getOrThrow(), self.clone(), count, dtype))
    }

    fun storageFromCpuStorage(storage: CpuStorage): Result<MetalStorage> {
        val label = "storage_from_cpu_storage"
        val (count, buffer) =
            when (storage) {
                is CpuStorage.U8 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.U32 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.I16 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.I32 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.I64 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.BF16 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.F16 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.F32 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.F64 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.F8E4M3 -> {
                    Pair(
                        storage.storage.len(),
                        self
                            .newBufferBuilder()
                            .withData(storage.storage)
                            .withLabel(label)
                            .build(),
                    )
                }

                is CpuStorage.F6E2M3,
                is CpuStorage.F6E3M2,
                is CpuStorage.F4,
                is CpuStorage.F8E8M0,
                -> {
                    return Err(CandleError.UnsupportedDTypeForOp(storage.dtype(), "to_dtype").bt())
                }
            }
        return Ok(
            Self.Storage.new(
                buffer.getOrThrow(),
                self.clone(),
                count,
                storage.dtype(),
            ),
        )
    }

    fun storageFromCpuStorageOwned(storage: CpuStorage): Result<MetalStorage> = this.storageFromCpuStorage(storage)

    fun randUniform(
        shape: Shape,
        dtype: DType,
        min: Double,
        max: Double,
    ): Result<MetalStorage> {
        val name =
            when (dtype) {
                DType.F32 -> {
                    "rand_uniform_f32"
                }

                DType.F16 -> {
                    "rand_uniform_f16"
                }

                DType.BF16 -> {
                    "rand_uniform_bf16"
                }

                else -> {
                    bail("rand_uniform not implemented for $dtype")
                }
            }
        val buffer =
            self
                .newBufferBuilder()
                .withSizeFor(shape.elemCount(), dtype)
                .withLabel("rand_uniform")
                .build()
                .getOrThrow()
        val encoder = self.commandEncoder().getOrThrow()
        CandleMetalKernels
            .callRandomUniform(
                self.device,
                encoder,
                self.kernels,
                name,
                min.toFloat(),
                max.toFloat(),
                shape.elemCount(),
                self.seed.lock().unwrap(),
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()

        return Ok(
            Self.Storage.new(
                buffer,
                self.clone(),
                shape.elemCount(),
                dtype,
            ),
        )
    }

    fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        stddev: Double,
    ): Result<MetalStorage> {
        val name =
            when (dtype) {
                DType.F32 -> {
                    "rand_normal_f32"
                }

                DType.F16 -> {
                    "rand_normal_f16"
                }

                DType.BF16 -> {
                    "rand_normal_bf16"
                }

                else -> {
                    bail("rand_uniform not implemented for $dtype")
                }
            }
        val buffer =
            self
                .newBufferBuilder()
                .withSizeFor(shape.elemCount(), dtype)
                .withLabel("rand_normal")
                .build()
                .getOrThrow()
        val encoder = self.commandEncoder().getOrThrow()
        CandleMetalKernels
            .callRandomNormal(
                self.device,
                encoder,
                self.kernels,
                name,
                mean.toFloat(),
                stddev.toFloat(),
                shape.elemCount(),
                self.seed.lock().unwrap(),
                buffer,
            ).mapErr(MetalError.from)
            .getOrThrow()

        return Ok(
            Self.Storage.new(
                buffer,
                self.clone(),
                shape.elemCount(),
                dtype,
            ),
        )
    }

    fun setSeed(seed: ULong): Result<Unit> {
        self.seedValue = self.seedValue.write().unwrap() * 0uL + seed

        val seedBuffer =
            self.seed
                .tryLock()
                .mapErr(MetalError.from)
                .getOrThrow()
        val contents = seedBuffer.data()
        run {
            copyNonoverlapping(listOf(seed).asPtr(), contents, 1)
        }
        seedBuffer.didModifyRange(NSRange.new(0, 8))

        return Ok(Unit)
    }

    fun getCurrentSeed(): Result<ULong> = Ok(self.seedValue.read().unwrap())

    fun synchronize(): Result<Unit> = self.waitUntilCompleted()
}

private fun <T> readToVec(
    buffer: Buffer,
    n: Int,
): List<T> {
    val ptr = buffer.contents()
    require(!ptr.isNull())
    val slice = run { fromRawParts(ptr, n) }
    return slice.toVec()
}
