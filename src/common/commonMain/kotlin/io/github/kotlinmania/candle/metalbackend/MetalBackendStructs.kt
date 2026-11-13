package io.github.kotlinmania.candle.metalbackend

import io.github.kotlinmania.candle.backend.BackendStorage
import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.USE_COL2IM_CONV1D_TR
import io.github.kotlinmania.candle.cudabackend.kernelName
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.metalbackend.device.MetalDevice
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.CallConvTranspose2dCfg
import io.github.kotlinmania.candlemetalkernels.GemmDType
import io.github.kotlinmania.candlemetalkernels.callAffine
import io.github.kotlinmania.candlemetalkernels.callAffineStrided
import io.github.kotlinmania.candlemetalkernels.callBinaryContiguous
import io.github.kotlinmania.candlemetalkernels.callBinaryStrided
import io.github.kotlinmania.candlemetalkernels.callCastContiguous
import io.github.kotlinmania.candlemetalkernels.callCastStrided
import io.github.kotlinmania.candlemetalkernels.callCol2im1d
import io.github.kotlinmania.candlemetalkernels.callConstSetContiguous
import io.github.kotlinmania.candlemetalkernels.callConstSetStrided
import io.github.kotlinmania.candlemetalkernels.callConvTranspose1d
import io.github.kotlinmania.candlemetalkernels.callConvTranspose2d
import io.github.kotlinmania.candlemetalkernels.callCopy2d
import io.github.kotlinmania.candlemetalkernels.callElu
import io.github.kotlinmania.candlemetalkernels.callEluStrided
import io.github.kotlinmania.candlemetalkernels.callGather
import io.github.kotlinmania.candlemetalkernels.callIm2col1dStrided
import io.github.kotlinmania.candlemetalkernels.callIm2colStrided
import io.github.kotlinmania.candlemetalkernels.callIndexAdd
import io.github.kotlinmania.candlemetalkernels.callIndexSelect
import io.github.kotlinmania.candlemetalkernels.callMlxGemm
import io.github.kotlinmania.candlemetalkernels.callPool2d
import io.github.kotlinmania.candlemetalkernels.callPowf
import io.github.kotlinmania.candlemetalkernels.callPowfStrided
import io.github.kotlinmania.candlemetalkernels.callReduceContiguous
import io.github.kotlinmania.candlemetalkernels.callReduceStrided
import io.github.kotlinmania.candlemetalkernels.callScatter
import io.github.kotlinmania.candlemetalkernels.callUnaryContiguous
import io.github.kotlinmania.candlemetalkernels.callUnaryStrided
import io.github.kotlinmania.candlemetalkernels.callUpsampleBilinear2d
import io.github.kotlinmania.candlemetalkernels.callUpsampleNearest2d
import io.github.kotlinmania.candlemetalkernels.callWhereCond
import io.github.kotlinmania.candlemetalkernels.copy2d.BFLOAT
import io.github.kotlinmania.candlemetalkernels.copy2d.FLOAT
import io.github.kotlinmania.candlemetalkernels.copy2d.HALF
import io.github.kotlinmania.candlemetalkernels.copy2d.I16
import io.github.kotlinmania.candlemetalkernels.copy2d.I32
import io.github.kotlinmania.candlemetalkernels.copy2d.I64
import io.github.kotlinmania.candlemetalkernels.copy2d.U32
import io.github.kotlinmania.candlemetalkernels.copy2d.U8
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.unary.contiguous
import io.github.kotlinmania.candlemetalkernels.unary.strided
import io.github.kotlinmania.candlemetalkernels.utils.EncoderParam
import io.github.kotlinmania.rustlang.core.F16

public typealias Device = MetalDevice

public class MetalStorage(
    /**
     * The actual buffer containing the data.
     */
    public var buffer: Buffer = Buffer(),
    /**
     * a reference to the device owning this buffer
     */
    public var device: MetalDevice = MetalDevice(),
    /**
     * The count of allocated elements in the buffer
     */
    public var count: Int = 0,
    /**
     * The dtype is kept since buffers are untyped.
     */
    public var dtype: DType = DType.entries.first(),
) : BackendStorage<MetalDevice> {
    public fun tryClone(arg0: Layout): Result<MetalStorage> = Result.Ok(this.clone())

    public fun dtypeBackendStorage(): DType = this.dtype

    public fun deviceBackendStorage(): MetalDevice = this.device

    public fun toCpuStorage(): Result<CpuStorage> {
        return when (this.dtype) {
            DType.u8 ->
                Result.Ok(
                    CpuStorage.u8(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.u32 ->
                Result.Ok(
                    CpuStorage.u32(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.i16 ->
                Result.Ok(
                    CpuStorage.i16(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.i32 ->
                Result.Ok(
                    CpuStorage.i32(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.i64 ->
                Result.Ok(
                    CpuStorage.i64(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.f16 ->
                Result.Ok(
                    CpuStorage.f16(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.bf16 ->
                Result.Ok(
                    CpuStorage.bf16(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.f32 ->
                Result.Ok(
                    CpuStorage.f32(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.f64 ->
                Result.Ok(
                    CpuStorage.f64(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.f8e4m3 ->
                Result.Ok(
                    CpuStorage.f8e4m3(
                        when (val tryResult = this.toCpu()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                )
            DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                Result.Err(Error.UnsupportedDTypeForOp(this.dtype, "to_cpu_storage").bt())
            }
        }
    }

    public fun affine(layout: Layout, mul: Double, add: Double): Result<MetalStorage> {
        val device = this.device().clone()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = this.dtype
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(el, this.dtype)
                        .withLabel("affine")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, layout, dtype)
        if (layout.isContiguous()) {
            val name =
                when (val matchVal = this.dtype) {
                    DType.f32 -> "affine_f32"
                    DType.f16 -> "affine_f16"
                    DType.bf16 -> "affine_bf16"
                    DType.u8 -> "affine_u8"
                    DType.u32 -> "affine_u32"
                    DType.i64 -> "affine_i64"
                    else -> {
                        val dtype1 = matchVal
                        bail("Metal contiguous affine {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callAffine(device.device, encoder, device.kernels, name, this.dtype.sizeInBytes(), el, src, buffer, mul.toFloat(), add.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            val name1 =
                when (val matchVal1 = this.dtype) {
                    DType.f32 -> "affine_f32_strided"
                    DType.f16 -> "affine_f16_strided"
                    DType.bf16 -> "affine_bf16_strided"
                    DType.u8 -> "affine_u8_strided"
                    DType.u32 -> "affine_u32_strided"
                    DType.i64 -> "affine_i64_strided"
                    else -> {
                        val dtype2 = matchVal1
                        bail("Metal strided affine {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callAffineStrided(device.device, encoder, device.kernels, name1, layout.dims(), src, layout.stride(), buffer, mul.toFloat(), add.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), el, dtype))
    }

    public fun powf(layout: Layout, pow: Double): Result<MetalStorage> {
        val device = this.device().clone()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = this.dtype
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(el, this.dtype)
                        .withLabel("powf")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, layout, dtype)
        if (layout.isContiguous()) {
            val name =
                when (val matchVal = this.dtype) {
                    DType.f32 -> "powf_f32"
                    DType.f16 -> "powf_f16"
                    DType.bf16 -> "powf_bf16"
                    else -> {
                        val dtype1 = matchVal
                        bail("Metal contiguous powf {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callPowf(device.device, encoder, device.kernels, name, this.dtype.sizeInBytes(), el, src, buffer, pow.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            val name1 =
                when (val matchVal1 = this.dtype) {
                    DType.f32 -> "powf_f32_strided"
                    DType.f16 -> "powf_f16_strided"
                    DType.bf16 -> "powf_bf16_strided"
                    else -> {
                        val dtype2 = matchVal1
                        bail("Metal strided powf {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callPowfStrided(device.device, encoder, device.kernels, name1, layout.dims(), src, layout.stride(), buffer, pow.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), el, dtype))
    }

    public fun elu(layout: Layout, alpha: Double): Result<MetalStorage> {
        val device = this.device().clone()
        val shape = layout.shape()
        val el = shape.elemCount()
        val dtype = this.dtype
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(el, this.dtype)
                        .withLabel("elu")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        if (layout.isContiguous()) {
            val name =
                when (val matchVal = this.dtype) {
                    DType.f32 -> "elu_f32"
                    DType.f16 -> "elu_f16"
                    DType.bf16 -> "elu_bf16"
                    else -> {
                        val dtype1 = matchVal
                        bail("Metal contiguous elu {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callElu(device.device, encoder, device.kernels, name, this.dtype.sizeInBytes(), el, src, buffer, alpha.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            val name1 =
                when (val matchVal1 = this.dtype) {
                    DType.f32 -> "elu_f32_strided"
                    DType.f16 -> "elu_f16_strided"
                    DType.bf16 -> "elu_bf16_strided"
                    else -> {
                        val dtype2 = matchVal1
                        bail("Metal strided elu {dtype:?} not implemented")
                    }
                }
            when (val tryResult = callEluStrided(device.device, encoder, device.kernels, name1, layout.dims(), src, layout.stride(), buffer, alpha.toFloat()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), el, dtype))
    }

    public fun reduceOp(op: ReduceOp, layout: Layout, sumDims: List<Int>): Result<MetalStorage> {
        val device = this.device.clone()
        val srcStride = layout.stride()
        val srcDims = layout.shape().dims()
        // Source dims and strides with the sum dims at the end.
        val dims = mutableListOf()
        val stride = mutableListOf()
        var dstEl: Int = 1
        for ((dimIdx, d) in srcDims.iter().mapIndexed { index, item -> Pair(index, item) }) {
            if (!sumDims.contains(dimIdx)) {
                dstEl *= d
                dims.add(d)
                stride.add(srcStride[dimIdx])
            }
        }
        for (dimIdx1 in sumDims) {
            dims.add(srcDims[dimIdx1])
            stride.add(srcStride[dimIdx1])
        }
        val reductionShape = Shape.from(dims.toMutableList())
        if (layout.isContiguous() && reductionShape.isContiguous(stride)) {
            val (name, checkEmpty, returnIndex) =
                run {
                    val matchVal = Pair(op, this.dtype)
                    when {
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.F32 -> Triple("fast_sum_f32", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.F32 -> Triple("fast_min_f32", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.F32 -> Triple("fast_max_f32", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.F32 -> Triple("fast_argmin_f32", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.F32 -> Triple("fast_argmax_f32", true, true)
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.U32 -> Triple("fast_sum_u32", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.U32 -> Triple("fast_min_u32", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.U32 -> Triple("fast_max_u32", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.U32 -> Triple("fast_argmin_u32", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.U32 -> Triple("fast_argmax_u32", true, true)
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.F16 -> Triple("fast_sum_f16", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.F16 -> Triple("fast_min_f16", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.F16 -> Triple("fast_max_f16", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.F16 -> Triple("fast_argmin_f16", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.F16 -> Triple("fast_argmax_f16", true, true)
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.BF16 -> Triple("fast_sum_bf16", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.BF16 -> Triple("fast_min_bf16", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.BF16 -> Triple("fast_max_bf16", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.BF16 -> Triple("fast_argmin_bf16", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.BF16 -> Triple("fast_argmax_bf16", true, true)
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.I64 -> Triple("fast_sum_i64", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.I64 -> Triple("fast_min_i64", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.I64 -> Triple("fast_max_i64", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.I64 -> Triple("fast_argmin_i64", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.I64 -> Triple("fast_argmax_i64", true, true)
                        matchVal.first == ReduceOp.Sum && matchVal.second == DType.U8 -> Triple("fast_sum_u8", false, false)
                        matchVal.first == ReduceOp.Min && matchVal.second == DType.U8 -> Triple("fast_min_u8", true, false)
                        matchVal.first == ReduceOp.Max && matchVal.second == DType.U8 -> Triple("fast_max_u8", true, false)
                        matchVal.first == ReduceOp.ArgMin && matchVal.second == DType.U8 -> Triple("fast_argmin_u8", true, true)
                        matchVal.first == ReduceOp.ArgMax && matchVal.second == DType.U8 -> Triple("fast_argmax_u8", true, true)
                        else -> {
                            val k = matchVal.first
                            val dtype = matchVal.second
                            bail("Metal contiguous reduce op {k:?} {dtype:?} not implemented")
                        }
                    }
                }
            if (checkEmpty && layout.shape().elemCount() == 0) {
                when (val tryResult = Result.Err(Error.EmptyTensor(op = "reduce").bt())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            val dtype1 =
                if (returnIndex) {
                    DType.U32
                } else {
                    this.dtype
                }
            val buffer =
                when (
                    val tryResult =
                        device
                            .newBufferBuilder()
                            .withSizeFor(dstEl, dtype1)
                            .withLabel("reduce")
                            .build()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val encoder =
                when (val tryResult = this.device.commandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val src = bufferO(this.buffer, layout, this.dtype)
            when (val tryResult = callReduceContiguous(device.device, encoder, device.kernels, name, srcDims, dstEl, src, buffer).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            return Result.Ok(MetalStorage.new(buffer, device, dstEl, dtype1))
        }
        val (name1, checkEmpty1, returnIndex1) =
            run {
                val matchVal1 = Pair(op, this.dtype)
                when {
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.F32 -> Triple("fast_sum_f32_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.F32 -> Triple("fast_min_f32_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.F32 -> Triple("fast_max_f32_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.F32 -> Triple("fast_argmin_f32_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.F32 -> Triple("fast_argmax_f32_strided", true, true)
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.U32 -> Triple("fast_sum_u32_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.U32 -> Triple("fast_min_u32_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.U32 -> Triple("fast_max_u32_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.U32 -> Triple("fast_argmin_u32_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.U32 -> Triple("fast_argmax_u32_strided", true, true)
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.F16 -> Triple("fast_sum_f16_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.F16 -> Triple("fast_min_f16_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.F16 -> Triple("fast_max_f16_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.F16 -> Triple("fast_argmin_f16_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.F16 -> Triple("fast_argmax_f16_strided", true, true)
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.BF16 -> Triple("fast_sum_bf16_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.BF16 -> Triple("fast_min_bf16_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.BF16 -> Triple("fast_max_bf16_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.BF16 -> Triple("fast_argmin_bf16_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.BF16 -> Triple("fast_argmax_bf16_strided", true, true)
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.I64 -> Triple("fast_sum_i64_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.I64 -> Triple("fast_min_i64_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.I64 -> Triple("fast_max_i64_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.I64 -> Triple("fast_argmin_i64_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.I64 -> Triple("fast_argmax_i64_strided", true, true)
                    matchVal1.first == ReduceOp.Sum && matchVal1.second == DType.U8 -> Triple("fast_sum_u8_strided", false, false)
                    matchVal1.first == ReduceOp.Min && matchVal1.second == DType.U8 -> Triple("fast_min_u8_strided", true, false)
                    matchVal1.first == ReduceOp.Max && matchVal1.second == DType.U8 -> Triple("fast_max_u8_strided", true, false)
                    matchVal1.first == ReduceOp.ArgMin && matchVal1.second == DType.U8 -> Triple("fast_argmin_u8_strided", true, true)
                    matchVal1.first == ReduceOp.ArgMax && matchVal1.second == DType.U8 -> Triple("fast_argmax_u8_strided", true, true)
                    else -> {
                        val k1 = matchVal1.first
                        val dtype2 = matchVal1.second
                        bail("Metal strided reduce op {k:?} {dtype:?} not implemented")
                    }
                }
            }
        if (checkEmpty1 && layout.shape().elemCount() == 0) {
            when (val tryResult = Result.Err(Error.EmptyTensor(op = "reduce").bt())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        val dtype3 =
            if (returnIndex1) {
                DType.U32
            } else {
                this.dtype
            }
        val buffer1 =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, dtype3)
                        .withLabel("reduce")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder1 =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src1 = bufferO(this.buffer, layout, this.dtype)
        when (val tryResult = callReduceStrided(device.device, encoder1, device.kernels, name1, dims, stride, dstEl, src1, buffer1).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer1, device, dstEl, dtype3))
    }

    public fun cmp(op: CmpOp, rhs: MetalStorage, lhsL: Layout, rhsL: Layout): Result<MetalStorage> {
        val name =
            when (op) {
                CmpOp.Eq -> "eq"
                CmpOp.Ne -> "ne"
                CmpOp.Le -> "le"
                CmpOp.Ge -> "ge"
                CmpOp.Lt -> "lt"
                CmpOp.Gt -> "gt"
            }
        return this.binary(name, rhs, lhsL, rhsL)
    }

    public fun constSet(s: Scalar, l: Layout): Result<Unit> {
        fun <S> set(selfRef: MetalStorage, s1: S, l1: Layout): Result<Unit> where S : WithDType, S : EncoderParam {
            val device = selfRef.device()
            val dtype = selfRef.dtype
            val shape = l1.shape()
            val elCount = shape.elemCount()
            val encoder =
                when (val tryResult = device.commandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val dst = bufferO(selfRef.buffer, l1, selfRef.dtype)
            if (l1.isContiguous()) {
                val kernelName =
                    when (dtype) {
                        DType.f16 -> contiguous.HALF
                        DType.bf16 -> contiguous.BFLOAT
                        DType.f32 -> contiguous.FLOAT
                        DType.i64 -> contiguous.I64
                        DType.u32 -> contiguous.U32
                        DType.u8 -> contiguous.U8
                        DType.f8e4m3 -> bail("unsupported const-set f8e4m3")
                        DType.f64 -> bail("unsupported const-set f64")
                        DType.f4, DType.f6e2m3, DType.f6e3m2, DType.f8e8m0, DType.i16, DType.i32 -> {
                            return Result.Err(Error.UnsupportedDTypeForOp(dtype, "const-set").bt())
                        }
                    }
                when (val tryResult = callConstSetContiguous(device.device, encoder, device.kernels, kernelName, dtype.sizeInBytes(), elCount, s1, dst).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            } else {
                val kernelName1 =
                    when (dtype) {
                        DType.f16 -> strided.HALF
                        DType.bf16 -> strided.BFLOAT
                        DType.f32 -> strided.FLOAT
                        DType.i64 -> strided.I64
                        DType.u32 -> strided.U32
                        DType.u8 -> strided.U8
                        DType.f8e4m3 -> bail("unsupported const-set f8e4m3")
                        DType.f64 -> bail("unsupported const-set f64")
                        DType.f4, DType.f6e2m3, DType.f6e3m2, DType.f8e8m0, DType.i16, DType.i32 -> {
                            return Result.Err(Error.UnsupportedDTypeForOp(dtype, "const-set").bt())
                        }
                    }
                when (val tryResult = callConstSetStrided(device.device, encoder, device.kernels, kernelName1, l1.dims(), s1, l1.stride(), dst).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            return Result.Ok(Unit)
        }
        return run {
            val matchVal = Pair(this.dtype, s)
            when {
                matchVal.first == DType.U8 && matchVal.second is Scalar.U8 -> {
                    val (s1) = matchVal.second as Scalar.U8
                    set(this, s1, l)
                }
                matchVal.first == DType.U32 && matchVal.second is Scalar.U32 -> {
                    val (s2) = matchVal.second as Scalar.U32
                    set(this, s2, l)
                }
                matchVal.first == DType.I64 && matchVal.second is Scalar.I64 -> {
                    val (s3) = matchVal.second as Scalar.I64
                    set(this, s3, l)
                }
                matchVal.first == DType.F16 && matchVal.second is Scalar.F16 -> {
                    val (s4) = matchVal.second as Scalar.F16
                    set(this, s4, l)
                }
                matchVal.first == DType.BF16 && matchVal.second is Scalar.Bf16 -> {
                    val (s5) = matchVal.second as Scalar.Bf16
                    set(this, s5, l)
                }
                matchVal.first == DType.F32 && matchVal.second is Scalar.F32 -> {
                    val (s6) = matchVal.second as Scalar.F32
                    set(this, s6, l)
                }
                matchVal.first == DType.F64 && matchVal.second is Scalar.F64 -> {
                    val (s7) = matchVal.second as Scalar.F64
                    set(this, s7, l)
                }
                else -> bail("dtype mismatch, expected {:?}, got {:?}", this.dtype, s)
            }
        }
    }

    public fun toDtype(layout: Layout, dtype: DType): Result<MetalStorage> {
        val device = this.device()
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(elCount, dtype)
                        .withLabel("to_dtype")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        if (layout.isContiguous()) {
            val kernelName =
                run {
                    val matchVal = Pair(this.dtype, dtype)
                    when {
                        matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "cast_u32_bf16"
                        matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "cast_u32_f16"
                        matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "cast_u32_f32"
                        matchVal.first == DType.U32 && matchVal.second == DType.I64 -> "cast_u32_i64"
                        matchVal.first == DType.U32 && matchVal.second == DType.U8 -> "cast_u32_u8"
                        matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "cast_u8_bf16"
                        matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "cast_u8_f16"
                        matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "cast_u8_f32"
                        matchVal.first == DType.U8 && matchVal.second == DType.I64 -> "cast_u8_i64"
                        matchVal.first == DType.U8 && matchVal.second == DType.U32 -> "cast_u8_u32"
                        matchVal.first == DType.F32 && matchVal.second == DType.BF16 -> "cast_f32_bf16"
                        matchVal.first == DType.F32 && matchVal.second == DType.F16 -> "cast_f32_f16"
                        matchVal.first == DType.F32 && matchVal.second == DType.I64 -> "cast_f32_i64"
                        matchVal.first == DType.F32 && matchVal.second == DType.U32 -> "cast_f32_u32"
                        matchVal.first == DType.F32 && matchVal.second == DType.U8 -> "cast_f32_u8"
                        matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "cast_i64_bf16"
                        matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "cast_i64_f16"
                        matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "cast_i64_f32"
                        matchVal.first == DType.I64 && matchVal.second == DType.U32 -> "cast_i64_u32"
                        matchVal.first == DType.I64 && matchVal.second == DType.U8 -> "cast_i64_u8"
                        matchVal.first == DType.F16 && matchVal.second == DType.BF16 -> "cast_f16_bf16"
                        matchVal.first == DType.F16 && matchVal.second == DType.F32 -> "cast_f16_f32"
                        matchVal.first == DType.F16 && matchVal.second == DType.I64 -> "cast_f16_i64"
                        matchVal.first == DType.F16 && matchVal.second == DType.U32 -> "cast_f16_u32"
                        matchVal.first == DType.F16 && matchVal.second == DType.U8 -> "cast_f16_u8"
                        matchVal.first == DType.BF16 && matchVal.second == DType.F16 -> "cast_bf16_f16"
                        matchVal.first == DType.BF16 && matchVal.second == DType.F32 -> "cast_bf16_f32"
                        matchVal.first == DType.BF16 && matchVal.second == DType.I64 -> "cast_bf16_i64"
                        matchVal.first == DType.BF16 && matchVal.second == DType.U32 -> "cast_bf16_u32"
                        matchVal.first == DType.BF16 && matchVal.second == DType.U8 -> "cast_bf16_u8"
                        else -> {
                            val left = matchVal.first
                            val right = matchVal.second
                            bail("Metal contiguous to_dtype {left:?} {right:?} not implemented")
                        }
                    }
                }
            when (val tryResult = callCastContiguous(device.device, encoder, device.kernels, kernelName, this.dtype.sizeInBytes(), elCount, src, buffer).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            val kernelName1 =
                run {
                    val matchVal1 = Pair(this.dtype, dtype)
                    when {
                        matchVal1.first == DType.BF16 && matchVal1.second == DType.F16 -> "cast_bf16_f16_strided"
                        matchVal1.first == DType.BF16 && matchVal1.second == DType.F32 -> "cast_bf16_f32_strided"
                        matchVal1.first == DType.BF16 && matchVal1.second == DType.I64 -> "cast_bf16_i64_strided"
                        matchVal1.first == DType.BF16 && matchVal1.second == DType.U32 -> "cast_bf16_u32_strided"
                        matchVal1.first == DType.BF16 && matchVal1.second == DType.U8 -> "cast_bf16_u8_strided"
                        matchVal1.first == DType.F16 && matchVal1.second == DType.BF16 -> "cast_f16_bf16_strided"
                        matchVal1.first == DType.F16 && matchVal1.second == DType.F32 -> "cast_f16_f32_strided"
                        matchVal1.first == DType.F16 && matchVal1.second == DType.I64 -> "cast_f16_i64_strided"
                        matchVal1.first == DType.F16 && matchVal1.second == DType.U32 -> "cast_f16_u32_strided"
                        matchVal1.first == DType.F16 && matchVal1.second == DType.U8 -> "cast_f16_u8_strided"
                        matchVal1.first == DType.F32 && matchVal1.second == DType.BF16 -> "cast_f32_bf16_strided"
                        matchVal1.first == DType.F32 && matchVal1.second == DType.F16 -> "cast_f32_f16_strided"
                        matchVal1.first == DType.F32 && matchVal1.second == DType.I64 -> "cast_f32_i64_strided"
                        matchVal1.first == DType.F32 && matchVal1.second == DType.U32 -> "cast_f32_u32_strided"
                        matchVal1.first == DType.F32 && matchVal1.second == DType.U8 -> "cast_f32_u8_strided"
                        matchVal1.first == DType.I64 && matchVal1.second == DType.F32 -> "cast_i64_f32_strided"
                        matchVal1.first == DType.I64 && matchVal1.second == DType.BF16 -> "cast_i64_bf16_strided"
                        matchVal1.first == DType.I64 && matchVal1.second == DType.F16 -> "cast_i64_f16_strided"
                        matchVal1.first == DType.I64 && matchVal1.second == DType.U32 -> "cast_i64_u32_strided"
                        matchVal1.first == DType.I64 && matchVal1.second == DType.U8 -> "cast_i64_u8_strided"
                        matchVal1.first == DType.U32 && matchVal1.second == DType.BF16 -> "cast_u32_bf16_strided"
                        matchVal1.first == DType.U32 && matchVal1.second == DType.F16 -> "cast_u32_f16_strided"
                        matchVal1.first == DType.U32 && matchVal1.second == DType.F32 -> "cast_u32_f32_strided"
                        matchVal1.first == DType.U32 && matchVal1.second == DType.I64 -> "cast_u32_i64_strided"
                        matchVal1.first == DType.U32 && matchVal1.second == DType.U8 -> "cast_u32_u8_strided"
                        matchVal1.first == DType.U8 && matchVal1.second == DType.BF16 -> "cast_u8_bf16_strided"
                        matchVal1.first == DType.U8 && matchVal1.second == DType.F16 -> "cast_u8_f16_strided"
                        matchVal1.first == DType.U8 && matchVal1.second == DType.F32 -> "cast_u8_f32_strided"
                        matchVal1.first == DType.U8 && matchVal1.second == DType.I64 -> "cast_u8_i64_strided"
                        matchVal1.first == DType.U8 && matchVal1.second == DType.U32 -> "cast_u8_u32_strided"
                        else -> {
                            val left1 = matchVal1.first
                            val right1 = matchVal1.second
                            bail("Metal strided to_dtype {left:?} {right:?} not implemented")
                        }
                    }
                }
            when (val tryResult = callCastStrided(device.device, encoder, device.kernels, kernelName1, layout.dims(), src, layout.stride(), buffer).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), elCount, dtype))
    }

    public fun <B : UnaryOpT> unaryImpl(layout: Layout): Result<MetalStorage> {
        val device = this.device()
        val dtype = this.dtype
        val shape = layout.shape()
        val elCount = shape.elemCount()
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(elCount, dtype)
                        .withLabel(B.KERNEL)
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        if (layout.isContiguous()) {
            val kernelName =
                run {
                    val matchVal = Pair(B.KERNEL, dtype)
                    when {
                        matchVal.first == "uabs" && matchVal.second == DType.F16 -> contiguous.abs.HALF
                        matchVal.first == "uabs" && matchVal.second == DType.F32 -> contiguous.abs.FLOAT
                        matchVal.first == "uabs" && matchVal.second == DType.BF16 -> contiguous.abs.BFLOAT
                        matchVal.first == "uceil" && matchVal.second == DType.F16 -> contiguous.ceil.HALF
                        matchVal.first == "uceil" && matchVal.second == DType.F32 -> contiguous.ceil.FLOAT
                        matchVal.first == "uceil" && matchVal.second == DType.BF16 -> contiguous.ceil.BFLOAT
                        matchVal.first == "ucos" && matchVal.second == DType.F16 -> contiguous.cos.HALF
                        matchVal.first == "ucos" && matchVal.second == DType.F32 -> contiguous.cos.FLOAT
                        matchVal.first == "ucos" && matchVal.second == DType.BF16 -> contiguous.cos.BFLOAT
                        matchVal.first == "uerf" && matchVal.second == DType.F16 -> contiguous.erf.HALF
                        matchVal.first == "uerf" && matchVal.second == DType.F32 -> contiguous.erf.FLOAT
                        matchVal.first == "uerf" && matchVal.second == DType.BF16 -> contiguous.erf.BFLOAT
                        matchVal.first == "uexp" && matchVal.second == DType.F16 -> contiguous.exp.HALF
                        matchVal.first == "uexp" && matchVal.second == DType.F32 -> contiguous.exp.FLOAT
                        matchVal.first == "uexp" && matchVal.second == DType.BF16 -> contiguous.exp.BFLOAT
                        matchVal.first == "ufloor" && matchVal.second == DType.F16 -> contiguous.floor.HALF
                        matchVal.first == "ufloor" && matchVal.second == DType.F32 -> contiguous.floor.FLOAT
                        matchVal.first == "ufloor" && matchVal.second == DType.BF16 -> contiguous.floor.BFLOAT
                        matchVal.first == "ugelu_erf" && matchVal.second == DType.F16 -> contiguous.geluErf.HALF
                        matchVal.first == "ugelu_erf" && matchVal.second == DType.F32 -> contiguous.geluErf.FLOAT
                        matchVal.first == "ugelu_erf" && matchVal.second == DType.BF16 -> contiguous.geluErf.BFLOAT
                        matchVal.first == "ugelu" && matchVal.second == DType.F16 -> contiguous.gelu.HALF
                        matchVal.first == "ugelu" && matchVal.second == DType.F32 -> contiguous.gelu.FLOAT
                        matchVal.first == "ugelu" && matchVal.second == DType.BF16 -> contiguous.gelu.BFLOAT
                        matchVal.first == "ulog" && matchVal.second == DType.F16 -> contiguous.log.HALF
                        matchVal.first == "ulog" && matchVal.second == DType.F32 -> contiguous.log.FLOAT
                        matchVal.first == "ulog" && matchVal.second == DType.BF16 -> contiguous.log.BFLOAT
                        matchVal.first == "uneg" && matchVal.second == DType.F16 -> contiguous.neg.HALF
                        matchVal.first == "uneg" && matchVal.second == DType.F32 -> contiguous.neg.FLOAT
                        matchVal.first == "uneg" && matchVal.second == DType.BF16 -> contiguous.neg.BFLOAT
                        matchVal.first == "urecip" && matchVal.second == DType.F16 -> contiguous.recip.HALF
                        matchVal.first == "urecip" && matchVal.second == DType.F32 -> contiguous.recip.FLOAT
                        matchVal.first == "urecip" && matchVal.second == DType.BF16 -> contiguous.recip.BFLOAT
                        matchVal.first == "urelu" && matchVal.second == DType.F16 -> contiguous.relu.HALF
                        matchVal.first == "urelu" && matchVal.second == DType.F32 -> contiguous.relu.FLOAT
                        matchVal.first == "urelu" && matchVal.second == DType.BF16 -> contiguous.relu.BFLOAT
                        matchVal.first == "uround" && matchVal.second == DType.F16 -> contiguous.round.HALF
                        matchVal.first == "uround" && matchVal.second == DType.F32 -> contiguous.round.FLOAT
                        matchVal.first == "uround" && matchVal.second == DType.BF16 -> contiguous.round.BFLOAT
                        matchVal.first == "usilu" && matchVal.second == DType.F16 -> contiguous.silu.HALF
                        matchVal.first == "usilu" && matchVal.second == DType.F32 -> contiguous.silu.FLOAT
                        matchVal.first == "usilu" && matchVal.second == DType.BF16 -> contiguous.silu.BFLOAT
                        matchVal.first == "usin" && matchVal.second == DType.F16 -> contiguous.sin.HALF
                        matchVal.first == "usin" && matchVal.second == DType.F32 -> contiguous.sin.FLOAT
                        matchVal.first == "usin" && matchVal.second == DType.BF16 -> contiguous.sin.BFLOAT
                        matchVal.first == "usqr" && matchVal.second == DType.F16 -> contiguous.sqr.HALF
                        matchVal.first == "usqr" && matchVal.second == DType.F32 -> contiguous.sqr.FLOAT
                        matchVal.first == "usqr" && matchVal.second == DType.BF16 -> contiguous.sqr.BFLOAT
                        matchVal.first == "usqrt" && matchVal.second == DType.F16 -> contiguous.sqrt.HALF
                        matchVal.first == "usqrt" && matchVal.second == DType.F32 -> contiguous.sqrt.FLOAT
                        matchVal.first == "usqrt" && matchVal.second == DType.BF16 -> contiguous.sqrt.BFLOAT
                        matchVal.first == "utanh" && matchVal.second == DType.F16 -> contiguous.tanh.HALF
                        matchVal.first == "utanh" && matchVal.second == DType.F32 -> contiguous.tanh.FLOAT
                        matchVal.first == "utanh" && matchVal.second == DType.BF16 -> contiguous.tanh.BFLOAT
                        matchVal.first == "usign" && matchVal.second == DType.F16 -> contiguous.sign.HALF
                        matchVal.first == "usign" && matchVal.second == DType.F32 -> contiguous.sign.FLOAT
                        matchVal.first == "usign" && matchVal.second == DType.BF16 -> contiguous.sign.BFLOAT
                        matchVal.first == "usign" && matchVal.second == DType.I64 -> contiguous.sign.I64
                        else -> {
                            val name = matchVal.first
                            val dtype1 = matchVal.second
                            bail("Metal contiguous unary {name} {dtype:?} not implemented")
                        }
                    }
                }
            when (val tryResult = callUnaryContiguous(device.device, encoder, device.kernels, kernelName, dtype.sizeInBytes(), elCount, src, buffer).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        } else {
            val kernelName1 =
                run {
                    val matchVal1 = Pair(B.KERNEL, dtype)
                    when {
                        matchVal1.first == "ucos" && matchVal1.second == DType.F32 -> strided.cos.FLOAT
                        matchVal1.first == "usin" && matchVal1.second == DType.F32 -> strided.sin.FLOAT
                        matchVal1.first == "usqr" && matchVal1.second == DType.F32 -> strided.sqr.FLOAT
                        matchVal1.first == "usqrt" && matchVal1.second == DType.F32 -> strided.sqrt.FLOAT
                        matchVal1.first == "uneg" && matchVal1.second == DType.F32 -> strided.neg.FLOAT
                        matchVal1.first == "uexp" && matchVal1.second == DType.F32 -> strided.exp.FLOAT
                        matchVal1.first == "ulog" && matchVal1.second == DType.F32 -> strided.log.FLOAT
                        matchVal1.first == "ugelu" && matchVal1.second == DType.F32 -> strided.gelu.FLOAT
                        matchVal1.first == "ugelu_erf" && matchVal1.second == DType.F32 -> strided.geluErf.FLOAT
                        matchVal1.first == "uerf" && matchVal1.second == DType.F32 -> strided.erf.FLOAT
                        matchVal1.first == "usilu" && matchVal1.second == DType.F32 -> strided.silu.FLOAT
                        matchVal1.first == "uabs" && matchVal1.second == DType.F32 -> strided.abs.FLOAT
                        matchVal1.first == "uceil" && matchVal1.second == DType.F32 -> strided.ceil.FLOAT
                        matchVal1.first == "ufloor" && matchVal1.second == DType.F32 -> strided.floor.FLOAT
                        matchVal1.first == "urelu" && matchVal1.second == DType.F32 -> strided.relu.FLOAT
                        matchVal1.first == "uround" && matchVal1.second == DType.F32 -> strided.round.FLOAT
                        matchVal1.first == "utanh" && matchVal1.second == DType.F32 -> strided.tanh.FLOAT
                        matchVal1.first == "ucos" && matchVal1.second == DType.F16 -> strided.cos.HALF
                        matchVal1.first == "usin" && matchVal1.second == DType.F16 -> strided.sin.HALF
                        matchVal1.first == "usqr" && matchVal1.second == DType.F16 -> strided.sqr.HALF
                        matchVal1.first == "usqrt" && matchVal1.second == DType.F16 -> strided.sqrt.HALF
                        matchVal1.first == "uneg" && matchVal1.second == DType.F16 -> strided.neg.HALF
                        matchVal1.first == "uexp" && matchVal1.second == DType.F16 -> strided.exp.HALF
                        matchVal1.first == "ulog" && matchVal1.second == DType.F16 -> strided.log.HALF
                        matchVal1.first == "ugelu" && matchVal1.second == DType.F16 -> strided.gelu.HALF
                        matchVal1.first == "ugelu_erf" && matchVal1.second == DType.F16 -> strided.geluErf.HALF
                        matchVal1.first == "uerf" && matchVal1.second == DType.F16 -> strided.erf.HALF
                        matchVal1.first == "usilu" && matchVal1.second == DType.F16 -> strided.silu.HALF
                        matchVal1.first == "uabs" && matchVal1.second == DType.F16 -> strided.abs.HALF
                        matchVal1.first == "uceil" && matchVal1.second == DType.F16 -> strided.ceil.HALF
                        matchVal1.first == "ufloor" && matchVal1.second == DType.F16 -> strided.floor.HALF
                        matchVal1.first == "urelu" && matchVal1.second == DType.F16 -> strided.relu.HALF
                        matchVal1.first == "uround" && matchVal1.second == DType.F16 -> strided.round.HALF
                        matchVal1.first == "utanh" && matchVal1.second == DType.F16 -> strided.tanh.HALF
                        matchVal1.first == "ucos" && matchVal1.second == DType.BF16 -> strided.cos.BFLOAT
                        matchVal1.first == "usin" && matchVal1.second == DType.BF16 -> strided.sin.BFLOAT
                        matchVal1.first == "usqr" && matchVal1.second == DType.BF16 -> strided.sqr.BFLOAT
                        matchVal1.first == "usqrt" && matchVal1.second == DType.BF16 -> strided.sqrt.BFLOAT
                        matchVal1.first == "uneg" && matchVal1.second == DType.BF16 -> strided.neg.BFLOAT
                        matchVal1.first == "uexp" && matchVal1.second == DType.BF16 -> strided.exp.BFLOAT
                        matchVal1.first == "ulog" && matchVal1.second == DType.BF16 -> strided.log.BFLOAT
                        matchVal1.first == "ugelu" && matchVal1.second == DType.BF16 -> strided.gelu.BFLOAT
                        matchVal1.first == "ugelu_erf" && matchVal1.second == DType.BF16 -> strided.geluErf.BFLOAT
                        matchVal1.first == "uerf" && matchVal1.second == DType.BF16 -> strided.erf.BFLOAT
                        matchVal1.first == "usilu" && matchVal1.second == DType.BF16 -> strided.silu.BFLOAT
                        matchVal1.first == "uabs" && matchVal1.second == DType.BF16 -> strided.abs.BFLOAT
                        matchVal1.first == "uceil" && matchVal1.second == DType.BF16 -> strided.ceil.BFLOAT
                        matchVal1.first == "ufloor" && matchVal1.second == DType.BF16 -> strided.floor.BFLOAT
                        matchVal1.first == "urelu" && matchVal1.second == DType.BF16 -> strided.relu.BFLOAT
                        matchVal1.first == "uround" && matchVal1.second == DType.BF16 -> strided.round.BFLOAT
                        matchVal1.first == "utanh" && matchVal1.second == DType.BF16 -> strided.tanh.BFLOAT
                        else -> {
                            val name1 = matchVal1.first
                            val dtype2 = matchVal1.second
                            bail("Metal strided unary {name} {dtype:?} not implemented")
                        }
                    }
                }
            val dst = BufferOffset.zeroOffset(buffer)
            when (val tryResult = callUnaryStrided(device.device, encoder, device.kernels, kernelName1, layout.dims(), src, layout.stride(), dst).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), elCount, dtype))
    }

    public fun <B : BinaryOpT> binaryImpl(rhs: MetalStorage, lhsL: Layout, rhsL: Layout): Result<MetalStorage> = this.binary(B.KERNEL, rhs, lhsL, rhsL)

    public fun whereCond(layout: Layout, t: MetalStorage, tL: Layout, f: MetalStorage, fL: Layout): Result<MetalStorage> {
        val device = this.device.clone()
        val shape = tL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val dtype = t.dtype
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(el, dtype)
                        .withLabel("where")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (t.dtype() != f.dtype()) {
            bail("Invalid where: different dtypes for values {:?} != {:?}", t.dtype(), f.dtype())
        }
        val name =
            run {
                val matchVal = Pair(this.dtype, t.dtype())
                when {
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "where_u8_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "where_u32_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "where_u8_bf16"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "where_u8_f16"
                    matchVal.first == DType.U8 && matchVal.second == DType.I64 -> "where_u8_i64"
                    matchVal.first == DType.U8 && matchVal.second == DType.U32 -> "where_u8_u32"
                    matchVal.first == DType.U8 && matchVal.second == DType.U8 -> "where_u8_u8"
                    else -> {
                        val left = matchVal.first
                        val right = matchVal.second
                        bail("Metal where_cond {left:?} {right:?} not implemented")
                    }
                }
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        val t1 = bufferO(t.buffer, tL, t.dtype)
        val f1 = bufferO(f.buffer, fL, f.dtype)
        when (val tryResult = callWhereCond(device.device, encoder, device.kernels, name, dtype.sizeInBytes(), dims, src, layout.stride(), layout.isContiguous(), t1, tL.stride(), tL.isContiguous(), f1, fL.stride(), fL.isContiguous(), buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, device, el, dtype))
    }

    public fun conv1d(layout: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConv1D): Result<MetalStorage> {
        val device = this.device().clone()
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
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("conv1d_im2col")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "im2col1d_f32"
                DType.f16 -> "im2col1d_f16"
                DType.bf16 -> "im2col1d_bf16"
                DType.u8 -> "im2col1d_u8"
                DType.u32 -> "im2col1d_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal conv1d {dtype:?} not implemented")
                }
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        when (val tryResult = callIm2col1dStrided(this.device.device, encoder, this.device.kernels, name, layout.shape().dims(), strides, listOf(kSize, stride, padding, dilation), src, dst).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        drop(encoder)
        val col = MetalStorage(buffer = dst, device = device, count = dstEl, dtype = this.dtype)
        val lOut1 = params.lOut()
        val b = params.bSize
        val n = params.cOut
        val k = params.kSize * params.cIn
        val m = lOut1
        val colL = Layout.contiguous(Triple(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kernelL1 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC =
                    when (val tryResult = this.device().zerosImpl(kernelL.shape(), kernel.dtype())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = kernel.copyStridedSrc(kernelC, 0, kernelL)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                val kernelL2 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        val resL =
            when (val tryResult = Layout.contiguous(Triple(b, lOut1, n)).transpose(1, 2)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val resT =
            when (val tryResult = this.device().zerosImpl(resL.shape(), res.dtype())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = res.copyStridedSrc(resT, 0, resL)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(resT)
    }

    public fun convTranspose1d(layout: Layout, k: MetalStorage, kLayout: Layout, params: ParamsConvTranspose1D): Result<MetalStorage> {
        val USE_COL2IM_CONV1D_TR: Boolean = true
        val canUseCol2im = kLayout.isContiguous() && params.dilation == 1 && params.padding == 0 && params.outputPadding == 0
        val lOut = params.lOut()
        val dstEl = params.cOut * lOut * params.bSize
        val buffer2 =
            if (USE_COL2IM_CONV1D_TR && canUseCol2im) {
                val (bSize, cIn, lIn) =
                    when (val tryResult = layout.shape().dims3()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val (cIn2, cOut, kSize) =
                    when (val tryResult = kLayout.shape().dims3()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (cIn != cIn2) {
                    bail("convtr1d: shape mismatch on c_in {:?} {:?}", layout.shape(), kLayout.shape())
                }
                val buffer =
                    when (
                        val tryResult =
                            this.device
                                .newBufferBuilder()
                                .withSizeFor(dstEl, this.dtype)
                                .withLabel("conv_transpose1d")
                                .build()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val name =
                    when (val matchVal = this.dtype) {
                        DType.f32 -> "col2im1d_f32"
                        DType.f16 -> "col2im1d_f16"
                        DType.bf16 -> "col2im1d_bf16"
                        DType.u32 -> "col2im1d_u32"
                        DType.u8 -> "col2im1d_u8"
                        else -> {
                            val dtype = matchVal
                            bail("metal col2im1d {dtype:?} not implemented")
                        }
                    }
                val col =
                    run {
                        // This merges the last two dimensions of the kernel together.
                        val kernelLMm = Layout.new(Triple(bSize, cIn, kSize * cOut).into(), mutableListOf(0, kSize * cOut, 1), kLayout.startOffset())
                        when (
                            val tryResult =
                                this.matmul(
                                    k,
                                    listOf(bSize, lIn, cOut * kSize, cIn),
                                    when (val tryResult = layout.transpose(1, 2)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    },
                                    kernelLMm,
                                )
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                // It is important for the command encoder to be obtained *after* the matmul
                // kernel has run, otherwise we might use a command-buffer that has been committed
                // already resulting in the following error.
                // _status < MTLCommandBufferStatusCommitted >
                // -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
                val encoder =
                    when (val tryResult = this.device.commandEncoder()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = callCol2im1d(this.device.device, encoder, this.device.kernels, name, listOf(bSize, lIn, cOut, kSize), params.kSize, params.stride, BufferOffset.zeroOffset(col.buffer), buffer).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                buffer
            } else {
                val buffer1 =
                    when (
                        val tryResult =
                            this.device
                                .newBufferBuilder()
                                .withSizeFor(dstEl, this.dtype)
                                .withLabel("conv_transpose1d")
                                .build()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val encoder1 =
                    when (val tryResult = this.device.commandEncoder()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val name1 =
                    when (val matchVal1 = this.dtype) {
                        DType.f32 -> "conv_transpose1d_f32"
                        DType.f16 -> "conv_transpose1d_f16"
                        DType.bf16 -> "conv_transpose1d_bf16"
                        DType.u32 -> "conv_transpose1d_u32"
                        DType.u8 -> "conv_transpose1d_u8"
                        else -> {
                            val dtype1 = matchVal1
                            bail("Metal conv_transpose1d {dtype:?} not implemented")
                        }
                    }
                when (val tryResult = callConvTranspose1d(this.device.device, encoder1, this.device.kernels, name1, params.dilation, params.stride, params.padding, params.outputPadding, params.cOut, lOut, params.bSize, layout.dims(), layout.stride(), kLayout.dims(), kLayout.stride(), this.buffer, layout.startOffset() * this.dtype.sizeInBytes(), k.buffer, kLayout.startOffset() * k.dtype.sizeInBytes(), buffer1).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                buffer1
            }
        return Result.Ok(MetalStorage.new(buffer2, this.device.clone(), dstEl, this.dtype))
    }

    public fun conv2d(layout: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConv2D): Result<MetalStorage> {
        val device = this.device().clone()
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
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("conv2d_im2col")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "im2col_f32"
                DType.f16 -> "im2col_f16"
                DType.bf16 -> "im2col_bf16"
                DType.u8 -> "im2col_u8"
                DType.u32 -> "im2col_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal conv2d {dtype:?} not implemented")
                }
            }
        val src = bufferO(this.buffer, layout, this.dtype)
        when (val tryResult = callIm2colStrided(this.device.device, encoder, this.device.kernels, name, layout.shape().dims(), layout.stride(), listOf(hK, wK, stride, padding, dilation), src, dst).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        drop(encoder)
        val col = MetalStorage(buffer = dst, device = device, count = dstEl, dtype = this.dtype)
        val hOut1 = params.outH()
        val wOut1 = params.outW()
        val b = params.bSize
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val m = hOut1 * wOut1
        val colL = Layout.contiguous(Triple(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kernelL1 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC =
                    when (val tryResult = this.device().zerosImpl(kernelL.shape(), kernel.dtype())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = kernel.copyStridedSrc(kernelC, 0, kernelL)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                val kernelL2 =
                    when (
                        val tryResult =
                            when (val tryResult = Layout.contiguousWithOffset(Triple(1, n, k), kernelL.startOffset()).transpose(1, 2)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.broadcastAs(Triple(b, k, n))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = col.matmul(kernel, listOf(b, m, n, k), colL, kernelL2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        val resL =
            when (
                val tryResult =
                    when (val tryResult = Layout.contiguous(listOf(b, hOut1, wOut1, n)).transpose(1, 2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.transpose(1, 3)
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val resT =
            when (val tryResult = this.device().zerosImpl(resL.shape(), res.dtype())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = res.copyStridedSrc(resT, 0, resL)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(resT)
    }

    public fun convTranspose2d(l: Layout, kernel: MetalStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<MetalStorage> {
        // Kernel shape: (cInK, cOut, hK, wK)
        // Input shape: (bSize, cIn, hIn, wIn)
        val (outW, outH) = Pair(params.outW(), params.outH())
        val dstEl = params.cOut * outW * outH * params.bSize
        val dims = l.dims()
        if (dims.len() != 4) {
            bail("unexpected input shape for conv_transpose2d {dims:?}, expected 4")
        }
        val kDims = kernelL.dims()
        if (kDims.len() != 4) {
            bail("unexpected kernel shape for conv_transpose2d {k_dims:?}, expected 4")
        }
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("conv_transpose2d")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "conv_transpose2d_f32"
                DType.f16 -> "conv_transpose2d_f16"
                DType.bf16 -> "conv_transpose2d_bf16"
                else -> {
                    val dtype = matchVal
                    bail("Metal conv_transpose2d {dtype:?} not implemented")
                }
            }
        when (val tryResult = callConvTranspose2d(this.device.device, encoder, this.device.kernels, name, CallConvTranspose2dCfg(dilation = params.dilation, stride = params.stride, padding = params.padding, outputPadding = params.outputPadding, cOut = params.cOut, outH = outH, outW = outW, bSize = params.bSize, inputDims = l.dims(), inputStride = l.stride(), kernelDims = kernelL.dims(), kernelStride = kernelL.stride(), inputOffset = l.startOffset() * this.dtype.sizeInBytes(), kernelOffset = kernelL.startOffset() * kernel.dtype.sizeInBytes()), this.buffer, kernel.buffer, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), dstEl, this.dtype))
    }

    public fun avgPool2d(inpL: Layout, tupleArg1: Pair<Int, Int>, tupleArg2: Pair<Int, Int>): Result<MetalStorage> {
        val shape = inpL.shape()
        val (bSize, channels, width, height) =
            when (val tryResult = shape.dims4()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val strides = inpL.stride()
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "avg_pool2d_f32"
                DType.f16 -> "avg_pool2d_f16"
                DType.bf16 -> "avg_pool2d_bf16"
                DType.u8 -> "avg_pool2d_u8"
                DType.u32 -> "avg_pool2d_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal avg_pool2d {dtype:?} not implemented")
                }
            }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("avg_pool2d")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = callPool2d(this.device.device, encoder, this.device.kernels, name, inpL.dims(), strides, outW, outH, wK, hK, wStride, hStride, this.buffer, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), dstEl, this.dtype))
    }

    public fun maxPool2d(inpL: Layout, tupleArg1: Pair<Int, Int>, tupleArg2: Pair<Int, Int>): Result<MetalStorage> {
        val shape = inpL.shape()
        val (bSize, channels, width, height) =
            when (val tryResult = shape.dims4()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val strides = inpL.stride()
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "max_pool2d_f32"
                DType.f16 -> "max_pool2d_f16"
                DType.bf16 -> "max_pool2d_bf16"
                DType.u8 -> "max_pool2d_u8"
                DType.u32 -> "max_pool2d_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal max_pool2d {dtype:?} not implemented")
                }
            }
        val outW = (width - wK) / wStride + 1
        val outH = (height - hK) / hStride + 1
        val dstEl = outW * outH * bSize * channels
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("max_pool2d")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = callPool2d(this.device.device, encoder, this.device.kernels, name, inpL.dims(), strides, outW, outH, wK, hK, wStride, hStride, this.buffer, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), dstEl, this.dtype))
    }

    public fun upsampleNearest1d(arg0: Layout, arg1: Int): Result<MetalStorage> = bail("Metal upsample_nearest1d not implemented")

    public fun upsampleNearest2d(inpL: Layout, outW: Int, outH: Int): Result<MetalStorage> {
        // let inp = &inp.slice(inpL.startOffset()..);
        val shape = inpL.shape()
        val dims = shape.dims()
        val strides = inpL.stride()
        if (dims.len() != 4) {
            bail("unexpected input shape for upsample {dims:?}")
        }
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "upsample_nearest2d_f32"
                DType.f16 -> "upsample_nearest2d_f16"
                DType.bf16 -> "upsample_nearest2d_bf16"
                DType.u8 -> "upsample_nearest2d_u8"
                DType.u32 -> "upsample_nearest2d_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal upsample_nearest2d {dtype:?} not implemented")
                }
            }
        val dstEl = outW * outH * dims[0] * dims[1]
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("upsample_nearest2d")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, inpL, this.dtype)
        when (val tryResult = callUpsampleNearest2d(this.device.device, encoder, this.device.kernels, name, dims, strides, outW, outH, src, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), dstEl, this.dtype))
    }

    public fun upsampleBilinear2d(inpL: Layout, outH: Int, outW: Int, alignCorners: Boolean, scaleH: Double?, scaleW: Double?): Result<MetalStorage> {
        val shape = inpL.shape()
        val dims = shape.dims()
        val strides = inpL.stride()
        if (dims.len() != 4) {
            bail("unexpected input shape for upsample_bilinear2d {dims:?}")
        }
        val name =
            when (val matchVal = this.dtype) {
                DType.f32 -> "upsample_bilinear2d_f32"
                DType.f16 -> "upsample_bilinear2d_f16"
                DType.bf16 -> "upsample_bilinear2d_bf16"
                DType.u8 -> "upsample_bilinear2d_u8"
                DType.u32 -> "upsample_bilinear2d_u32"
                else -> {
                    val dtype = matchVal
                    bail("Metal upsample_bilinear2d {dtype:?} not implemented")
                }
            }
        val dstEl = outW * outH * dims[0] * dims[1]
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, this.dtype)
                        .withLabel("upsample_bilinear2d")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, inpL, this.dtype)
        when (val tryResult = callUpsampleBilinear2d(this.device.device, encoder, this.device.kernels, name, dims, strides, outW, outH, alignCorners, scaleH, scaleW, src, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), dstEl, this.dtype))
    }

    public fun gather(srcL: Layout, ids: MetalStorage, idsL: Layout, dim: Int): Result<MetalStorage> {
        if (!idsL.isContiguous()) {
            return Result.Err(Error.RequiresContiguous(op = "gather").bt())
        }
        val idsEl = idsL.dims()[dim]
        val dstEl = idsL.shape().elemCount()
        val dtype = this.dtype
        val device = this.device()
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, dtype)
                        .withLabel("gather")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val name =
            run {
                val matchVal = Pair(ids.dtype, this.dtype)
                when {
                    matchVal.first == DType.U8 && matchVal.second == DType.U8 -> "gather_u8_u8"
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "gather_u8_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "gather_u8_f16"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "gather_u8_bf16"
                    matchVal.first == DType.U8 && matchVal.second == DType.U32 -> "gather_u8_u32"
                    matchVal.first == DType.U8 && matchVal.second == DType.I64 -> "gather_u8_i64"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "gather_u32_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "gather_u32_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "gather_u32_bf16"
                    matchVal.first == DType.U32 && matchVal.second == DType.U32 -> "gather_u32_u32"
                    matchVal.first == DType.U32 && matchVal.second == DType.I64 -> "gather_u32_i64"
                    matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "gather_i64_f32"
                    matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "gather_i64_f16"
                    matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "gather_i64_bf16"
                    matchVal.first == DType.I64 && matchVal.second == DType.U32 -> "gather_i64_u32"
                    matchVal.first == DType.I64 && matchVal.second == DType.I64 -> "gather_i64_i64"
                    else -> {
                        val left = matchVal.first
                        val right = matchVal.second
                        bail("Metal gather {left:?} {right:?} not implemented")
                    }
                }
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, srcL, dtype)
        val ids1 = bufferO(ids.buffer, idsL, ids.dtype)
        when (val tryResult = callGather(device.device, encoder, this.device.kernels, name, srcL.dims(), idsEl, dim, src, ids1, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), dstEl, dtype))
    }

    public fun scatterSet(l: Layout, ids: MetalStorage, idsL: Layout, src: MetalStorage, srcL: Layout, dim: Int): Result<Unit> {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            return Result.Err(Error.RequiresContiguous(op = "scatter").bt())
        }
        val name =
            run {
                val matchVal = Pair(ids.dtype, this.dtype)
                when {
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "s_u8_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "s_u8_f16"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "s_u8_bf16"
                    matchVal.first == DType.U32 && matchVal.second == DType.U32 -> "s_u32_u32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "s_u32_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "s_u32_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "s_u32_bf16"
                    matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "s_i64_f32"
                    matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "s_i64_f16"
                    matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "s_i64_bf16"
                    else ->
                        when (val tryResult = Result.Err(MetalError.UnexpectedDType(msg = "scatter ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val dst = bufferO(this.buffer, l, this.dtype)
        val src1 = bufferO(src.buffer, srcL, src.dtype)
        val ids1 = bufferO(ids.buffer, idsL, ids.dtype)
        when (val tryResult = callScatter(this.device.device, encoder, this.device.kernels, name, srcL.dims(), l.dims(), dim, src1, ids1, dst).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(Unit)
    }

    public fun scatterAddSet(l: Layout, ids: MetalStorage, idsL: Layout, src: MetalStorage, srcL: Layout, dim: Int): Result<Unit> {
        if (!l.isContiguous() || !idsL.isContiguous() || !srcL.isContiguous()) {
            return Result.Err(Error.RequiresContiguous(op = "scatter-add").bt())
        }
        val name =
            run {
                val matchVal = Pair(ids.dtype, this.dtype)
                when {
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "sa_u8_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "sa_u8_f16"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "sa_u8_bf16"
                    matchVal.first == DType.U32 && matchVal.second == DType.U32 -> "sa_u32_u32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "sa_u32_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "sa_u32_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "sa_u32_bf16"
                    matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "sa_i64_f32"
                    matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "sa_i64_f16"
                    matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "sa_i64_bf16"
                    else ->
                        when (val tryResult = Result.Err(MetalError.UnexpectedDType(msg = "scatter-add ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val dst = bufferO(this.buffer, l, this.dtype)
        val src1 = bufferO(src.buffer, srcL, src.dtype)
        val ids1 = bufferO(ids.buffer, idsL, ids.dtype)
        when (val tryResult = callScatter(this.device.device, encoder, this.device.kernels, name, srcL.dims(), l.dims(), dim, src1, ids1, dst).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(Unit)
    }

    public fun indexSelect(ids: MetalStorage, srcL: Layout, idsL: Layout, dim: Int): Result<MetalStorage> {
        if (!idsL.isContiguous()) {
            bail("Metal index_select requires contiguous ids")
        }
        val leftSize: Int = ((srcL.dims()).slice(0 until dim)).iter().product()
        val rightSize: Int = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product()
        val idsEl = idsL.shape().elemCount()
        val dstEl = idsEl * leftSize * rightSize
        val dtype = this.dtype
        val device = this.device()
        val buffer =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(dstEl, dtype)
                        .withLabel("index_select")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val name =
            run {
                val matchVal = Pair(ids.dtype, this.dtype)
                when {
                    matchVal.first == DType.U8 && matchVal.second == DType.U8 -> "is_u8_u8"
                    matchVal.first == DType.U8 && matchVal.second == DType.U32 -> "is_u8_u32"
                    matchVal.first == DType.U8 && matchVal.second == DType.I64 -> "is_u8_i64"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "is_u8_bf16"
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "is_u8_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "is_u8_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.U8 -> "is_u32_u8"
                    matchVal.first == DType.U32 && matchVal.second == DType.U32 -> "is_u32_u32"
                    matchVal.first == DType.U32 && matchVal.second == DType.I64 -> "is_u32_i64"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "is_u32_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "is_u32_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "is_u32_bf16"
                    matchVal.first == DType.I64 && matchVal.second == DType.U8 -> "is_i64_u8"
                    matchVal.first == DType.I64 && matchVal.second == DType.U32 -> "is_i64_u32"
                    matchVal.first == DType.I64 && matchVal.second == DType.I64 -> "is_i64_i64"
                    matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "is_i64_f32"
                    matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "is_i64_f16"
                    matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "is_i64_bf16"
                    else -> {
                        val left = matchVal.first
                        val right = matchVal.second
                        bail("Metal contiguous index_select {left:?} {right:?} not implemented")
                    }
                }
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src = bufferO(this.buffer, srcL, dtype)
        val ids1 = bufferO(ids.buffer, idsL, ids.dtype)
        when (val tryResult = callIndexSelect(device.device, encoder, this.device.kernels, name, srcL.dims(), idsEl, dim, srcL.isContiguous(), srcL.dims(), srcL.stride(), src, ids1, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, device.clone(), dstEl, dtype))
    }

    public fun indexAdd(l: Layout, ids: MetalStorage, idsL: Layout, src: MetalStorage, srcL: Layout, dim: Int): Result<MetalStorage> {
        val acc =
            when (val tryResult = this.device.zerosImpl(l.shape(), this.dtype())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (val tryResult = this.copyStridedSrc(acc, 0, l)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        if (!idsL.isContiguous() || !srcL.isContiguous()) {
            return Result.Err(Error.RequiresContiguous(op = "index-add").bt())
        }
        val name =
            run {
                val matchVal = Pair(ids.dtype, this.dtype)
                when {
                    matchVal.first == DType.I64 && matchVal.second == DType.BF16 -> "ia_i64_bf16"
                    matchVal.first == DType.I64 && matchVal.second == DType.F16 -> "ia_i64_f16"
                    matchVal.first == DType.I64 && matchVal.second == DType.F32 -> "ia_i64_f32"
                    matchVal.first == DType.I64 && matchVal.second == DType.I64 -> "ia_i64_i64"
                    matchVal.first == DType.I64 && matchVal.second == DType.U32 -> "ia_i64_u32"
                    matchVal.first == DType.I64 && matchVal.second == DType.U8 -> "ia_i64_u8"
                    matchVal.first == DType.U32 && matchVal.second == DType.BF16 -> "ia_u32_bf16"
                    matchVal.first == DType.U32 && matchVal.second == DType.F16 -> "ia_u32_f16"
                    matchVal.first == DType.U32 && matchVal.second == DType.F32 -> "ia_u32_f32"
                    matchVal.first == DType.U32 && matchVal.second == DType.I64 -> "ia_u32_i64"
                    matchVal.first == DType.U32 && matchVal.second == DType.U32 -> "ia_u32_u32"
                    matchVal.first == DType.U32 && matchVal.second == DType.U8 -> "ia_u32_u8"
                    matchVal.first == DType.U8 && matchVal.second == DType.BF16 -> "ia_u8_bf16"
                    matchVal.first == DType.U8 && matchVal.second == DType.F16 -> "ia_u8_f16"
                    matchVal.first == DType.U8 && matchVal.second == DType.F32 -> "ia_u8_f32"
                    matchVal.first == DType.U8 && matchVal.second == DType.I64 -> "ia_u8_i64"
                    matchVal.first == DType.U8 && matchVal.second == DType.U32 -> "ia_u8_u32"
                    matchVal.first == DType.U8 && matchVal.second == DType.U8 -> "ia_u8_u8"
                    else ->
                        when (val tryResult = Result.Err(MetalError.UnexpectedDType(msg = "index-add ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val src1 = bufferO(src.buffer, srcL, src.dtype)
        val ids1 = bufferO(ids.buffer, idsL, ids.dtype)
        when (val tryResult = callIndexAdd(this.device.device, encoder, this.device.kernels, name, srcL.dims(), l.dims(), idsL.dims(), dim, src1, ids1, acc.buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(acc)
    }

    public fun matmul(rhs: MetalStorage, tupleArg1: List<Any?>, lhsL: Layout, rhsL: Layout): Result<MetalStorage> {
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSizeFor(b * m * n, this.dtype)
                        .withLabel("matmul")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val encoder =
            when (val tryResult = this.device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val dtype1 =
            when (val matchVal = this.dtype) {
                DType.f32 -> GemmDType.F32
                DType.f16 -> GemmDType.F16
                DType.bf16 -> GemmDType.BF16
                else -> {
                    val dtype = matchVal
                    return Result.Err(MetalError.Message("mlx matmul doesn't support $dtype").into())
                }
            }
        when (val tryResult = callMlxGemm(this.device.device, encoder, this.device.kernels, dtype1, listOf(b, m, n, k), lhsL.stride(), lhsL.startOffset() * this.dtype.sizeInBytes(), this.buffer, rhsL.stride(), rhsL.startOffset() * rhs.dtype.sizeInBytes(), rhs.buffer, buffer).mapErr(MetalError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(MetalStorage.new(buffer, this.device.clone(), b * m * n, this.dtype()))
    }

    public fun copy2d(dst: MetalStorage, d1: Int, d2: Int, srcS: Int, dstS: Int, srcO: Int, dstO: Int): Result<Unit> {
        if (this.dtype() != dst.dtype()) {
            bail("copy2d with inconsistent dtypes {:?} {:?}", this.dtype(), dst.dtype())
        }
        if (srcS == d2 && dstS == d2) {
            val blit =
                when (val tryResult = this.device.blitCommandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            blit.setLabel("copy2d_contiguous")
            val srcOffset = srcO * this.dtype.sizeInBytes()
            val length = d1 * d2 * this.dtype.sizeInBytes()
            val dstOffset = dstO * dst.dtype().sizeInBytes()
            blit.copyFromBuffer(this.buffer, srcOffset, dst.buffer(), dstOffset, length)
        } else {
            val elCount = d1 * d2
            if (elCount == 0) {
                return Result.Ok(Unit)
            }
            val kernelName =
                when (val matchVal = this.dtype) {
                    DType.f32 -> FLOAT
                    DType.f16 -> HALF
                    DType.bf16 -> BFLOAT
                    DType.i64 -> I64
                    DType.i32 -> I32
                    DType.i16 -> I16
                    DType.u32 -> U32
                    DType.u8 -> U8
                    else -> {
                        val dtype = matchVal
                        bail("Metal copy2d {dtype:?} not implemented")
                    }
                }
            val encoder =
                when (val tryResult = this.device.commandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = callCopy2d(this.device.device, encoder, this.device.kernels, kernelName, this.buffer, dst.buffer, d1, d2, srcS, dstS, srcO * this.dtype.sizeInBytes(), dstO * this.dtype.sizeInBytes()).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(Unit)
    }

    public fun copyStridedSrc(dst: MetalStorage, dstOffset: Int, srcL: Layout): Result<Unit> {
        if (srcL.isContiguous() && this.dtype == dst.dtype()) {
            val blit =
                when (val tryResult = this.device.blitCommandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            blit.setLabel("copy_contiguous")
            val srcOffset = srcL.startOffset() * this.dtype.sizeInBytes()
            val length = srcL.shape().elemCount() * this.dtype.sizeInBytes()
            val dstOffset1 = dstOffset * dst.dtype().sizeInBytes()
            blit.copyFromBuffer(this.buffer, srcOffset, dst.buffer(), dstOffset1, length)
        } else {
            val srcShape = srcL.shape()
            val elCount = srcShape.elemCount()
            if (elCount == 0) {
                return Result.Ok(Unit)
            }
            if (this.dtype == dst.dtype()) {
                val strides = srcL.stride()
                val ndim = strides.len()
                // Inner blocks are contiguous - eligible for copy2d
                if (ndim == 1 || strides[ndim - 1] == 1) {
                    when (val matchVal = srcL.stridedBlocks()) {
                        is StridedBlocks.UniformBlocks -> {
                            val startOffset = matchVal.startOffset
                            val blockLen = matchVal.blockLen
                            val count = matchVal.count
                            val srcStride = matchVal.srcStride
                            return this.copy2d(dst, count, blockLen, srcStride, blockLen, startOffset, dstOffset)
                        }
                        else -> {}
                    }
                }
            }
            val kernelName =
                when (val matchVal1 = this.dtype) {
                    DType.f32 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.FLOAT
                    DType.f16 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.HALF
                    DType.bf16 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.BFLOAT
                    DType.i64 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.I64
                    DType.u32 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.U32
                    DType.u8 -> io.github.kotlinmania.candlemetalkernels.unary.strided.copy.U8
                    else -> {
                        val dtype = matchVal1
                        bail("Metal copy_strided {dtype:?} not implemented")
                    }
                }
            val src = bufferO(this.buffer, srcL, this.dtype)
            val dst1 = BufferOffset(buffer = dst.buffer, offsetInBytes = dstOffset * dst.dtype.sizeInBytes())
            val encoder =
                when (val tryResult = this.device.commandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = callUnaryStrided(this.device.device, encoder, this.device.kernels, kernelName, srcL.dims(), src, srcL.stride(), dst1).mapErr(MetalError::from)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        return Result.Ok(Unit)
    }

    public fun binary(op: String, rhs: MetalStorage, lhsL: Layout, rhsL: Layout): Result<MetalStorage> {
        fun kernelName(op1: String, dtype: DType, suffix: String): String = "${op}_${dtype.asStr()}$suffix"
        val device = this.device()
        val shape = lhsL.shape()
        val elCount = shape.elemCount()
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val lhs = bufferO(this.buffer, lhsL, this.dtype)
        val rhs1 = bufferO(rhs.buffer, rhsL, rhs.dtype)
        val dtype =
            when (op) {
                "eq", "ne", "le", "lt", "ge", "gt" -> DType.U8
                else -> this.dtype
            }
        val lhsIsScalar = lhsL.isScalarLike()
        val rhsIsScalar = rhsL.isScalarLike()
        val lhsContiguous = lhsL.isContiguous()
        val rhsContiguous = rhsL.isContiguous()
        val contiguousKernel = kernelName(op, this.dtype, "")
        val kernel =
            run {
                val matchVal = listOf(lhsIsScalar, rhsIsScalar, lhsContiguous, rhsContiguous)
                when {
                    matchVal[0] == true && matchVal[1] == true -> kernelName(op, this.dtype, "_scalar")
                    matchVal[0] == true && matchVal[1] == false && matchVal[3] == true -> kernelName(op, this.dtype, "_sc")
                    matchVal[0] == true && matchVal[1] == false && matchVal[3] == false -> kernelName(op, this.dtype, "_rss")
                    matchVal[0] == false && matchVal[1] == true && matchVal[2] == true -> kernelName(op, this.dtype, "_cs")
                    matchVal[0] == false && matchVal[1] == true && matchVal[2] == false -> kernelName(op, this.dtype, "_lss")
                    matchVal[0] == false && matchVal[1] == false && matchVal[2] == true && matchVal[3] == true -> contiguousKernel.clone()
                    matchVal[0] == false && matchVal[1] == false && matchVal[2] == true && matchVal[3] == false -> kernelName(op, this.dtype, "_rstrided")
                    matchVal[0] == false && matchVal[1] == false && matchVal[2] == false && matchVal[3] == true -> kernelName(op, this.dtype, "_lstrided")
                    else -> kernelName(op, this.dtype, "_strided")
                }
            }
        val buffer2 =
            if (kernel == contiguousKernel) {
                val buffer =
                    when (
                        val tryResult =
                            device
                                .newBufferBuilder()
                                .withSizeFor(elCount, dtype)
                                .withLabel(op)
                                .build()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = callBinaryContiguous(device.device, encoder, device.kernels, kernel, this.dtype.sizeInBytes(), elCount, lhs, rhs1, buffer).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                buffer
            } else {
                val buffer1 =
                    when (
                        val tryResult =
                            device
                                .newBufferBuilder()
                                .withSizeFor(elCount, dtype)
                                .withLabel(op)
                                .build()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = callBinaryStrided(device.device, encoder, device.kernels, kernel, this.dtype.sizeInBytes(), lhsL.dims(), lhs, lhsL.stride(), rhs1, rhsL.stride(), buffer1).mapErr(MetalError::from)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                buffer1
            }
        return Result.Ok(MetalStorage.new(buffer2, device.clone(), elCount, dtype))
    }

    internal fun <T> toCpu(): Result<List<T>> {
        val size = this.count * this.dtype.sizeInBytes()
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSize(size)
                        .withLabel("blit_to_cpu_dst")
                        .build()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        run {
            val blit =
                when (val tryResult = this.device.blitCommandEncoder()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(this.buffer, 0, buffer, 0, size)
        }
        when (val tryResult = this.device.flushAndWaitCurrent()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return Result.Ok(readToVec(buffer, this.count))
    }

    public companion object {
        public fun new(buffer: Buffer, device: MetalDevice, count: Int, dtype: DType): MetalStorage = MetalStorage(buffer = buffer, device = device, count = count, dtype = dtype)
    }
}
