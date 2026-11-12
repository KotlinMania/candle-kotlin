package io.github.kotlinmania.candle.cudabackend

import io.github.kotlinmania.candle.backend.BackendStorage
import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.USE_COL2IM_CONV1D_TR
import io.github.kotlinmania.candle.cpubackend.USE_IM2COL_CONV1D
import io.github.kotlinmania.candle.cudabackend.cudnn.launchConv1d
import io.github.kotlinmania.candle.cudabackend.cudnn.launchConv2d
import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.cudabackend.device.cudaGraphHtodCacheEnabled
import io.github.kotlinmania.candle.cudabackend.error.CudaError
import io.github.kotlinmania.candle.cudabackend.utils.Map1
import io.github.kotlinmania.candle.cudabackend.utils.Map1Any
import io.github.kotlinmania.candle.cudabackend.utils.Map2
import io.github.kotlinmania.candle.cudabackend.utils.Map2Any
import io.github.kotlinmania.candle.cudabackend.utils.Map2InPlace
import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.DeviceRepr
import io.github.kotlinmania.cudarc.driver.LaunchConfig
import io.github.kotlinmania.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.candle.cudabackend.device.builderArg as barg

public typealias Device = CudaDevice

public class Clone : Map1 {
    public fun <T : DeviceRepr> f(s: CudaSlice<T>, arg1: CudaDevice, arg2: Layout): Result<CudaSlice<T>> = s.tryClone().w()

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Clone> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Clone> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Clone =
                    Clone()
                override fun writeNative(addr: Long, value: Clone) {
                }
            }
    }
}

public class Affine(
    public var field0: Double = 0.0,
    public var field1: Double = 0.0,
) : Map1 {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType {
        val shape = layout.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("affine"), AFFINE)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(el)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, dims.len())
        ds.builderArg(builder)
        builder.arg(src1)
        builder.arg(out)
        barg(builder, this.value.fromF64())
        barg(builder, this.field1.fromF64())
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Affine> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Affine> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 16L
                override fun readNative(addr: Long): Affine =
                    Affine(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr + 8L),
                    )
                override fun writeNative(addr: Long, value: Affine) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.field0)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr + 8L, value.field1)
                }
            }
    }
}

public class Elu(
    public var value: Double = 0.0,
) : Map1 {
    public var field0: Double
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType {
        val shape = layout.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("uelu"), UNARY)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(el)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, dims.len())
        ds.builderArg(builder)
        barg(builder, this.value.fromF64())
        builder.arg(src1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Elu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Elu> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): Elu =
                    Elu(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Elu) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.value)
                }
            }
    }
}

public class Im2Col1D(
    public var lK: Int = 0,
    public var stride: Int = 0,
    public var dilation: Int = 0,
    public var padding: Int = 0,
) : Map1 {
    public fun lOut(l: Int): Int = (l + 2 * this.padding - this.dilation * (this.lK - 1) - 1) / this.stride + 1

    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType {
        val shape = layout.shape()
        val dims = shape.dims()
        val lOut = this.lOut(dims[2])
        val threads = dims[0] * lOut * dims[1]
        val cfg = LaunchConfig.forNumElems(threads.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims, layout.stride()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("im2col1d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val dst =
            when (val tryResult = dev.alloc<T>(threads * this.lK)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, threads)
        barg(builder, lOut)
        barg(builder, this.lK)
        barg(builder, this.stride)
        barg(builder, this.padding)
        barg(builder, this.dilation)
        ds.builderArg(builder)
        builder.arg(src1)
        builder.arg(dst)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 32L
                override fun readNative(addr: Long): Im2Col1D =
                    Im2Col1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                    )
                override fun writeNative(addr: Long, value: Im2Col1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.lK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.dilation)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.padding)
                }
            }
    }
}

public class Im2Col(
    public var hK: Int = 0,
    public var wK: Int = 0,
    public var stride: Int = 0,
    public var dilation: Int = 0,
    public var padding: Int = 0,
) : Map1 {
    public fun hwOut(h: Int, w: Int): Pair<Int, Int> {
        val hOut = (h + 2 * this.padding - this.dilation * (this.hK - 1) - 1) / this.stride + 1
        val wOut = (w + 2 * this.padding - this.dilation * (this.wK - 1) - 1) / this.stride + 1
        return Pair(hOut, wOut)
    }

    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType {
        val shape = layout.shape()
        val dims = shape.dims()
        val (hOut, wOut) = this.hwOut(dims[2], dims[3])
        val dstEl = dims[0] * hOut * wOut * dims[1] * this.hK * this.wK
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims, layout.stride()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("im2col"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val dst =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, dstEl)
        barg(builder, hOut)
        barg(builder, wOut)
        barg(builder, this.hK)
        barg(builder, this.wK)
        barg(builder, this.stride)
        barg(builder, this.padding)
        barg(builder, this.dilation)
        ds.builderArg(builder)
        builder.arg(src1)
        builder.arg(dst)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 40L
                override fun readNative(addr: Long): Im2Col =
                    Im2Col(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 32L),
                    )
                override fun writeNative(addr: Long, value: Im2Col) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.hK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.wK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.dilation)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 32L, value.padding)
                }
            }
    }
}

public class Powf(
    public var value: Double = 0.0,
) : Map1 {
    public var field0: Double
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType {
        val shape = layout.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("upowf"), UNARY)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(el)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, dims.len())
        ds.builderArg(builder)
        barg(builder, this.value.fromF64())
        builder.arg(src1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Powf> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Powf> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): Powf =
                    Powf(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Powf) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.value)
                }
            }
    }
}

public class FastReduce(
    public var field0: List<Int> = emptyList(),
    public var field1: ReduceOp = ReduceOp.entries.first(),
) : Map1Any {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout, wrap: (CudaSlice<T>) -> S): Result<S> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val srcStride = layout.stride()
        val srcDims = layout.shape().dims()
        val srcEl: Int = srcDims.iter().product()
        // Source dims and strides with the sum dims at the end.
        val dims = mutableListOf()
        val stride = mutableListOf()
        var dstEl: Int = 1
        for ((dimIdx, d) in srcDims.iter().enumerate()) {
            if (!this.value.contains(dimIdx)) {
                dstEl *= d
                dims.add(d)
                stride.add(srcStride[dimIdx])
            }
        }
        for (dimIdx1 in this.value) {
            dims.add(srcDims[dimIdx1])
            stride.add(srcStride[dimIdx1])
        }
        val elToSumPerBlock = srcEl / dstEl
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims.asSlice(), stride.asSlice()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 = src.slice(layout.startOffset() until Int.MAX_VALUE)
        val (name, checkEmpty, returnIndex) =
            when (this.field1) {
                ReduceOp.Sum -> Triple("fast_sum", false, false)
                ReduceOp.Min -> Triple("fast_min", true, false)
                ReduceOp.Max -> Triple("fast_max", true, false)
                ReduceOp.ArgMin -> Triple("fast_argmin", true, true)
                ReduceOp.ArgMax -> Triple("fast_argmax", true, true)
            }
        // For small reductions (e.g. MoE topk sum with elToSum=8), use a one-thread-per-output
        // kernel to avoid launching millions of blocks each doing almost nothing.
        val useSmallReduce = elToSumPerBlock <= 32 && !returnIndex && name == "fast_sum"
        val (cfg, kernelNameStr) =
            if (useSmallReduce) {
                val threads = 256
                val blocks = (dstEl + threads - 1) / threads
                Pair(LaunchConfig(gridDim = Triple(blocks.toUInt(), 1, 1), blockDim = Triple(threads.toUInt(), 1, 1), sharedMemBytes = 0), "fast_sum_small")
            } else {
                val blockDim = Int.min(1024, elToSumPerBlock).nextPowerOfTwo()
                Pair(LaunchConfig(gridDim = Triple(dstEl.toUInt(), 1, 1), blockDim = Triple(blockDim.toUInt(), 1, 1), sharedMemBytes = 0), name)
            }
        if (checkEmpty && layout.shape().elemCount() == 0) {
            when (val tryResult = Err(Error.EmptyTensor(op = "reduce").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(kernelNameStr), REDUCE)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val kernelNumel =
            if (useSmallReduce) {
                dstEl
            } else {
                srcEl
            }
        return if (returnIndex) {
            // SAFETY: filled in by the follow up kernel.
            val out =
                when (val tryResult = dev.alloc<UInt>(dstEl)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val builder = func.builder()
            barg(builder, kernelNumel)
            barg(builder, elToSumPerBlock)
            barg(builder, srcDims.len())
            ds.builderArg(builder)
            builder.arg(src1)
            builder.arg(out)
            // SAFETY: ffi.
            when (val tryResult = builder.launch(cfg).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Ok(S.u32(out))
        } else {
            // SAFETY: filled in by the follow up kernel.
            val out1 =
                when (val tryResult = dev.alloc<T>(dstEl)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val builder1 = func.builder()
            barg(builder1, kernelNumel)
            barg(builder1, elToSumPerBlock)
            barg(builder1, srcDims.len())
            ds.builderArg(builder1)
            builder1.arg(src1)
            builder1.arg(out1)
            // SAFETY: ffi.
            when (val tryResult = builder1.launch(cfg).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Ok(wrap(out1))
        }
    }
}

public class IndexSelect(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
    public var field2: Int = 0,
) : Map1 {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, srcL: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val idsL = this.field1
        val (name, ids, guard) =
            when (val matchVal = this.value.slice) {
                is CudaStorageSlice.U32 -> {
                    val (slice) = matchVal
                    Pair("is_u32", slicePtr(slice, idsL.startOffset()))
                }
                is CudaStorageSlice.U8 -> {
                    val (slice1) = matchVal
                    Pair("is_u8", slicePtr(slice1, idsL.startOffset()))
                }
                is CudaStorageSlice.I64 -> {
                    val (slice2) = matchVal
                    Pair("is_i64", slicePtr(slice2, idsL.startOffset()))
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "index_select ids should be u8, u32, or i64", expected = DType.U32, got = this.value.dtype())).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val idsShape = idsL.shape()
        val idsDims = idsShape.dims()
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(idsDims, idsL.stride()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val src1 =
            when (val matchVal1 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "index-select").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal1.first
                    val o2 = matchVal1.second
                    src.slice(o1 until o2)
                }
            }
        val leftSize: Int = ((srcL.dims()).slice(0 until this.field2)).iter().product()
        val rightSize: Int = ((srcL.dims()).slice(this.field2 + 1 until srcL.dims().size)).iter().product()
        val srcDimSize = srcL.dims()[this.field2]
        val idsDimSize = idsShape.elemCount()
        val dstEl = idsShape.elemCount() * leftSize * rightSize
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), INDEXING)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, dstEl)
        barg(builder, idsDims.len())
        ds.builderArg(builder)
        barg(builder, ids)
        builder.arg(src1)
        builder.arg(out)
        barg(builder, leftSize)
        barg(builder, srcDimSize)
        barg(builder, idsDimSize)
        barg(builder, rightSize)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class Gather(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
    public var field2: Int = 0,
) : Map1 {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, srcL: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val ids = this.value
        val idsL = this.field1
        val dim = this.field2
        val (idsO1, _) =
            when (val matchVal = idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "gather").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o12 = matchVal
                    o12
                }
            }
        val (name, ids1, guard) =
            when (val matchVal1 = ids.slice) {
                is CudaStorageSlice.U32 -> {
                    val (slice) = matchVal1
                    Pair("gather_u32", slicePtr(slice, idsO1))
                }
                is CudaStorageSlice.U8 -> {
                    val (slice1) = matchVal1
                    Pair("gather_u8", slicePtr(slice1, idsO1))
                }
                is CudaStorageSlice.I64 -> {
                    val (slice2) = matchVal1
                    Pair("gather_i64", slicePtr(slice2, idsO1))
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "gather ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val el = idsL.shape().elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val src1 =
            when (val matchVal2 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "gather").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal2.first
                    val o2 = matchVal2.second
                    src.slice(o1 until o2)
                }
            }
        val leftSz: Int = ((srcL.dims()).slice(0 until dim)).iter().product()
        val rightSz: Int = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product()
        val srcDimSz = srcL.dims()[dim]
        val idsDimSz = idsL.dims()[dim]
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), INDEXING)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(el)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, ids1)
        builder.arg(src1)
        builder.arg(out)
        barg(builder, leftSz)
        barg(builder, srcDimSz)
        barg(builder, idsDimSz)
        barg(builder, rightSz)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class IndexAdd(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
    public var field2: Int = 0,
) : Map2InPlace {
    public fun <T> f(dst: CudaSlice<T>, dstL: Layout, src: CudaSlice<T>, srcL: Layout, dev: CudaDevice): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val ids = this.value
        val idsL = this.field1
        val dim = this.field2
        val (idsO1, _) =
            when (val matchVal = idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o12 = matchVal
                    o12
                }
            }
        val (name, ids1, guard) =
            when (val matchVal1 = ids.slice) {
                is CudaStorageSlice.U32 -> {
                    val (slice) = matchVal1
                    Pair("ia_u32", slicePtr(slice, idsO1))
                }
                is CudaStorageSlice.I64 -> {
                    val (slice1) = matchVal1
                    Pair("ia_i64", slicePtr(slice1, idsO1))
                }
                is CudaStorageSlice.U8 -> {
                    val (slice2) = matchVal1
                    Pair("ia_u8", slicePtr(slice2, idsO1))
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "index-add ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val dst1 =
            when (val matchVal2 = dstL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal2.first
                    val o2 = matchVal2.second
                    dst.slice(o1 until o2)
                }
            }
        val src1 =
            when (val matchVal3 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o11 = matchVal3.first
                    val o21 = matchVal3.second
                    src.slice(o11 until o21)
                }
            }
        val leftSz: Int = ((srcL.dims()).slice(0 until dim)).iter().product()
        val rightSz: Int = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product()
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val idsDimSz = idsL.dims()[0]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), INDEXING)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, ids1)
        barg(builder, idsDimSz)
        builder.arg(src1)
        builder.arg(dst1)
        barg(builder, leftSz, srcDimSz, dstDimSz, rightSz)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }
}

public class Scatter(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
    public var field2: Int = 0,
) : Map2InPlace {
    public fun <T> f(dst: CudaSlice<T>, dstL: Layout, src: CudaSlice<T>, srcL: Layout, dev: CudaDevice): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val ids = this.value
        val idsL = this.field1
        val dim = this.field2
        val (idsO1, _) =
            when (val matchVal = idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o12 = matchVal
                    o12
                }
            }
        val (name, ids1, guard) =
            when (val matchVal1 = ids.slice) {
                is CudaStorageSlice.U32 -> {
                    val (slice) = matchVal1
                    Pair("s_u32", slicePtr(slice, idsO1))
                }
                is CudaStorageSlice.I64 -> {
                    val (slice1) = matchVal1
                    Pair("s_i64", slicePtr(slice1, idsO1))
                }
                is CudaStorageSlice.U8 -> {
                    val (slice2) = matchVal1
                    Pair("s_u8", slicePtr(slice2, idsO1))
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "scatter ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val dst1 =
            when (val matchVal2 = dstL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal2.first
                    val o2 = matchVal2.second
                    dst.slice(o1 until o2)
                }
            }
        val src1 =
            when (val matchVal3 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o11 = matchVal3.first
                    val o21 = matchVal3.second
                    src.slice(o11 until o21)
                }
            }
        val leftSz: Int = ((srcL.dims()).slice(0 until dim)).iter().product()
        val rightSz: Int = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product()
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), INDEXING)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, ids1)
        builder.arg(src1)
        builder.arg(dst1)
        barg(builder, leftSz, srcDimSz, dstDimSz, rightSz)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }
}

public class ScatterAdd(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
    public var field2: Int = 0,
) : Map2InPlace {
    public fun <T> f(dst: CudaSlice<T>, dstL: Layout, src: CudaSlice<T>, srcL: Layout, dev: CudaDevice): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val ids = this.value
        val idsL = this.field1
        val dim = this.field2
        val (idsO1, _) =
            when (val matchVal = idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o12 = matchVal
                    o12
                }
            }
        val (name, ids1, guard) =
            when (val matchVal1 = ids.slice) {
                is CudaStorageSlice.U32 -> {
                    val (slice) = matchVal1
                    Pair("sa_u32", slicePtr(slice, idsO1))
                }
                is CudaStorageSlice.I64 -> {
                    val (slice1) = matchVal1
                    Pair("sa_i64", slicePtr(slice1, idsO1))
                }
                is CudaStorageSlice.U8 -> {
                    val (slice2) = matchVal1
                    Pair("sa_u8", slicePtr(slice2, idsO1))
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "scatter-add ids should be u8/u32/i64", expected = DType.U32, got = ids.dtype()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val dst1 =
            when (val matchVal2 = dstL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal2.first
                    val o2 = matchVal2.second
                    dst.slice(o1 until o2)
                }
            }
        val src1 =
            when (val matchVal3 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o11 = matchVal3.first
                    val o21 = matchVal3.second
                    src.slice(o11 until o21)
                }
            }
        val leftSz: Int = ((srcL.dims()).slice(0 until dim)).iter().product()
        val rightSz: Int = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product()
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), INDEXING)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, ids1)
        builder.arg(src1)
        builder.arg(dst1)
        barg(builder, leftSz, srcDimSz, dstDimSz, rightSz)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }
}

public class Conv1D(
    public var value: ParamsConv1D = ParamsConv1D(),
) : Map2 {
    public var field0: ParamsConv1D
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(inp: CudaSlice<T>, inpL: Layout, k: CudaSlice<T>, kL: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Kernel shape: (cOut, cInK, kSize)
        // Input shape: (bSize, cIn, lIn) or (cIn, lIn)
        val p = this.value
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val k1 = k.slice(kL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val lOut = p.lOut()
        val dstEl = p.cOut * lOut * p.bSize
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("conv1d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds =
            if (dims.len() == 3) {
                listOf(dims, inpL.stride(), kL.dims(), kL.stride()).concat()
            } else if (dims.len() == 2) {
                listOf(listOf(1), dims, listOf(1), inpL.stride(), kL.dims(), kL.stride()).concat()
            } else {
                bail("unexpected input shape for conv1d {dims:?}")
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el, lOut, p.stride, p.padding, p.dilation)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(k1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class Conv2D(
    public var value: ParamsConv2D = ParamsConv2D(),
) : Map2 {
    public var field0: ParamsConv2D
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(inp: CudaSlice<T>, inpL: Layout, k: CudaSlice<T>, kL: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Kernel shape: (cOut, cInK, hK, wK)
        // Input shape: (bSize, cIn, hIn, wIn)
        val p = this.value
        val (outW, outH) = Pair(p.outW(), p.outH())
        val dstEl = p.cOut * outW * outH * p.bSize
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val k1 = k.slice(kL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("conv2d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds =
            if (dims.len() == 4) {
                listOf(dims, inpL.stride(), kL.dims(), kL.stride()).concat()
            } else {
                bail("unexpected input shape for conv2d {dims:?}")
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el, outW, outH, p.stride, p.padding, p.dilation)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(k1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class Col2Im1D(
    public var stride: Int = 0,
) : Map1 {
    public fun <T> f(col: CudaSlice<T>, dev: CudaDevice, l: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val (bSize, lIn, cOut, kSize) =
            when (val tryResult = l.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = this.stride
        val lOut = (lIn - 1) * stride + kSize
        val dstEl = bSize * cOut * lOut
        val im =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("col2im1d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, dstEl, lOut, lIn, cOut, kSize, stride)
        builder.arg(col)
        builder.arg(im)
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(im)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Col2Im1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Col2Im1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): Col2Im1D =
                    Col2Im1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Col2Im1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.stride)
                }
            }
    }
}

public class ConvTranspose1D(
    public var value: ParamsConvTranspose1D = ParamsConvTranspose1D(),
) : Map2 {
    public var field0: ParamsConvTranspose1D
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(inp: CudaSlice<T>, inpL: Layout, k: CudaSlice<T>, kL: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Kernel shape: (cInK, cOut, lK)
        // Input shape: (bSize, cIn, lIn)
        val p = this.value
        val lOut = p.lOut()
        val dstEl = p.cOut * lOut * p.bSize
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val k1 = k.slice(kL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("conv_transpose1d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds =
            if (dims.len() == 3) {
                listOf(dims, inpL.stride(), kL.dims(), kL.stride()).concat()
            } else {
                bail("unexpected input shape for conv_transpose1d {dims:?}")
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, lOut)
        barg(builder, p.stride)
        barg(builder, p.padding)
        barg(builder, p.outputPadding)
        barg(builder, p.dilation)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(k1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class ConvTranspose2D(
    public var value: ParamsConvTranspose2D = ParamsConvTranspose2D(),
) : Map2 {
    public var field0: ParamsConvTranspose2D
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(inp: CudaSlice<T>, inpL: Layout, k: CudaSlice<T>, kL: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Kernel shape: (cInK, cOut, hK, wK)
        // Input shape: (bSize, cIn, hIn, wIn)
        val p = this.value
        val (outW, outH) = Pair(p.outW(), p.outH())
        val dstEl = p.cOut * outW * outH * p.bSize
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val k1 = k.slice(kL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("conv_transpose2d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds =
            if (dims.len() == 4) {
                listOf(dims, inpL.stride(), kL.dims(), kL.stride()).concat()
            } else {
                bail("unexpected input shape for conv_transpose2d {dims:?}")
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, outW)
        barg(builder, outH)
        barg(builder, p.stride)
        barg(builder, p.padding)
        barg(builder, p.outputPadding)
        barg(builder, p.dilation)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(k1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class Pool2D(
    public var wK: Int = 0,
    public var hK: Int = 0,
    public var wStride: Int = 0,
    public var hStride: Int = 0,
    public var op: PoolOp = PoolOp(),
) : Map1 {
    public fun <T> f(inp: CudaSlice<T>, dev: CudaDevice, inpL: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Input shape: (bSize, c, h, w)
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val ds =
            if (dims.len() == 4) {
                listOf(dims, inpL.stride()).concat()
            } else {
                bail("unexpected input shape for pool {dims:?}")
            }
        val el = shape.elemCount()
        val outW = (dims[2] - this.wK) / this.wStride + 1
        val outH = (dims[3] - this.hK) / this.hStride + 1
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val kname =
            when (this.op) {
                PoolOp.Max -> "max_pool2d"
                PoolOp.Avg -> "avg_pool2d"
            }
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(kname), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, this.wK)
        barg(builder, this.hK)
        barg(builder, this.wStride)
        barg(builder, this.hStride)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class UpsampleNearest2D(
    public var field0: Int = 0,
    public var field1: Int = 0,
) : Map1 {
    public fun <T> f(inp: CudaSlice<T>, dev: CudaDevice, inpL: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        // Input shape: (bSize, c, h, w)
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val ds =
            if (dims.len() == 4) {
                listOf(dims, inpL.stride()).concat()
            } else {
                bail("unexpected input shape for upsample {dims:?}")
            }
        val (outW, outH) = Pair(this.value, this.field1)
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("upsample_nearest2d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val scaleW = dims[2].toDouble() / outW.toDouble()
        val scaleH = dims[3].toDouble() / outH.toDouble()
        val builder = func.builder()
        barg(builder, outW)
        barg(builder, outH)
        barg(builder, scaleW)
        barg(builder, scaleH)
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest2D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest2D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 16L
                override fun readNative(addr: Long): UpsampleNearest2D =
                    UpsampleNearest2D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                    )
                override fun writeNative(addr: Long, value: UpsampleNearest2D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.field0)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.field1)
                }
            }
    }
}

public class UpsampleBilinear2D(
    public var outW: Int = 0,
    public var outH: Int = 0,
    public var alignCorners: Boolean = false,
    public var scaleHFactor: Double? = null,
    public var scaleWFactor: Double? = null,
) : Map1 {
    public fun <T> f(inp: CudaSlice<T>, dev: CudaDevice, inpL: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
        val shape = inpL.shape()
        val dims = shape.dims()
        val ds =
            if (dims.len() == 4) {
                listOf(dims, inpL.stride()).concat()
            } else {
                bail("unexpected input shape for upsample_bilinear2d {dims:?}")
            }
        val (outW, outH) = Pair(this.outW, this.outH)
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>("upsample_bilinear2d"), CONV)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(dstEl)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ds1 =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, ds)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, outW)
        barg(builder, outH)
        barg(builder, this.alignCorners)
        barg(builder, this.scaleHFactor != null)
        barg(builder, (this.scaleHFactor ?: 0.0))
        barg(builder, this.scaleWFactor != null)
        barg(builder, (this.scaleWFactor ?: 0.0))
        ds1.builderArg(builder)
        builder.arg(inp1)
        builder.arg(out)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class WhereCond(
    public var field0: CudaStorage = CudaStorage(),
    public var field1: Layout = Layout(),
) : Map2 {
    public fun <T> f(t: CudaSlice<T>, layoutT: Layout, f: CudaSlice<T>, layoutF: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val idsL = this.field1
        val (ids, guard, name) =
            when (val matchVal = this.value.slice) {
                is CudaStorageSlice.U8 -> {
                    val (slice) = matchVal
                    val ptr = slicePtr(slice, idsL.startOffset())
                    Pair(ptr, "where_u8")
                }
                is CudaStorageSlice.U32 -> {
                    val (slice1) = matchVal
                    val ptr1 = slicePtr(slice1, idsL.startOffset())
                    Pair(ptr1, "where_u32")
                }
                is CudaStorageSlice.I64 -> {
                    val (slice2) = matchVal
                    val ptr2 = slicePtr(slice2, idsL.startOffset())
                    Pair(ptr2, "where_i64")
                }
                else ->
                    when (val tryResult = Err(CudaError.UnexpectedDType(msg = "where conditions should be u8/u32/i64", expected = DType.U32, got = this.value.dtype())).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        val shape = idsL.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims, idsL.stride(), layoutT.stride(), layoutF.stride()).concat())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val t1 = t.slice(layoutT.startOffset() until Int.MAX_VALUE)
        val f1 = f.slice(layoutF.startOffset() until Int.MAX_VALUE)
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), TERNARY)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<T>(el)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, el)
        barg(builder, dims.len())
        ds.builderArg(builder)
        barg(builder, ids)
        builder.arg(t1)
        builder.arg(f1)
        builder.arg(out)
        // SAFETY: ffi
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(out)
    }
}

public class Cmp(
    public var value: CmpOp = CmpOp.entries.first(),
) : Map2Any {
    public var field0: CmpOp
        get() = value
        set(v) {
            value = v
        }

    public fun <T> f(lhs: CudaSlice<T>, lhsL: Layout, rhs: CudaSlice<T>, rhsL: Layout, dev: CudaDevice): Result<S> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
        val shape = lhsL.shape()
        val dims = shape.dims()
        val elemCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elemCount.toUInt())
        val dimsAndStrides =
            if (lhsL.isContiguous() && rhsL.isContiguous()) {
                SlicePtrOrNull.Null
            } else {
                when (val tryResult = SlicePtrOrNull.paramsFromVec(dev, listOf(dims, lhsL.stride(), rhsL.stride()).concat())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val lhs1 = lhs.slice(lhsL.startOffset() until Int.MAX_VALUE)
        val rhs1 = rhs.slice(rhsL.startOffset() until Int.MAX_VALUE)
        val name =
            when (this.value) {
                CmpOp.Eq -> "eq"
                CmpOp.Ne -> "ne"
                CmpOp.Lt -> "lt"
                CmpOp.Le -> "le"
                CmpOp.Gt -> "gt"
                CmpOp.Ge -> "ge"
            }
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName<T>(name), BINARY)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // SAFETY: Set later by running the kernel.
        val out =
            when (val tryResult = dev.alloc<UByte>(elemCount)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, elemCount)
        barg(builder, dims.len())
        dimsAndStrides.builderArg(builder)
        builder.arg(lhs1)
        builder.arg(rhs1)
        builder.arg(out)
        // SAFETY: ffi
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(S.u8(out))
    }
}

public class CudaStorage(
    public var slice: CudaStorageSlice = CudaStorageSlice(),
    public var device: CudaDevice = CudaDevice(),
) : BackendStorage {
    public fun <T : CudaDType> asCudaSlice(asCudaSliceTFn: (Any?) -> Any?): Result<CudaSlice<T>> = asCudaSliceTFn(this)

    public fun <T : CudaDType> asCudaSliceMut(asCudaSliceMutTFn: (Any?) -> Any?): Result<CudaSlice<T>> = asCudaSliceMutTFn(this)

    public fun transferToDevice(dst: CudaDevice): Result<CudaStorage> {
        val dstStream = dst.cudaStream()
        val storageSlice =
            when (this.dtype()) {
                DType.u8 -> {
                    val cudaSlice =
                        when (val tryResult = this.asCudaSlice<UByte>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u8(result)
                }
                DType.u32 -> {
                    val cudaSlice1 =
                        when (val tryResult = this.asCudaSlice<UInt>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result1 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice1).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.u32(result1)
                }
                DType.i16 -> {
                    val cudaSlice2 =
                        when (val tryResult = this.asCudaSlice<Short>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result2 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice2).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i16(result2)
                }
                DType.i32 -> {
                    val cudaSlice3 =
                        when (val tryResult = this.asCudaSlice<Int>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result3 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice3).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i32(result3)
                }
                DType.i64 -> {
                    val cudaSlice4 =
                        when (val tryResult = this.asCudaSlice<Long>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result4 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice4).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.i64(result4)
                }
                DType.bf16 -> {
                    val cudaSlice5 =
                        when (val tryResult = this.asCudaSlice<Bf16>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result5 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice5).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.bf16(result5)
                }
                DType.f16 -> {
                    val cudaSlice6 =
                        when (val tryResult = this.asCudaSlice<F16>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result6 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice6).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f16(result6)
                }
                DType.f32 -> {
                    val cudaSlice7 =
                        when (val tryResult = this.asCudaSlice<Float>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result7 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice7).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f32(result7)
                }
                DType.f64 -> {
                    val cudaSlice8 =
                        when (val tryResult = this.asCudaSlice<Double>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result8 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice8).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f64(result8)
                }
                DType.f8e4m3 -> {
                    val cudaSlice9 =
                        when (val tryResult = this.asCudaSlice<F8e4m3>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result9 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice9).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e4m3(result9)
                }
                DType.f6e2m3 -> {
                    val cudaSlice10 =
                        when (val tryResult = this.asCudaSlice<UByte>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result10 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice10).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f6e2m3(result10)
                }
                DType.f6e3m2 -> {
                    val cudaSlice11 =
                        when (val tryResult = this.asCudaSlice<UByte>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result11 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice11).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f6e3m2(result11)
                }
                DType.f4 -> {
                    val cudaSlice12 =
                        when (val tryResult = this.asCudaSlice<UByte>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result12 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice12).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f4(result12)
                }
                DType.f8e8m0 -> {
                    val cudaSlice13 =
                        when (val tryResult = this.asCudaSlice<UByte>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val result13 =
                        when (val tryResult = dstStream.cloneDtod(cudaSlice13).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    CudaStorageSlice.f8e8m0(result13)
                }
            }
        return Ok(CudaStorage(slice = storageSlice, device = dst.clone()))
    }

    public fun tryClone(layout: Layout): Result<CudaStorage> {
        val slice =
            when (val tryResult = Clone().map(this.slice, this.device(), layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val device = this.device.clone()
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun dtype(): DType =
        when (val matchVal = this.slice) {
            is CudaStorageSlice.U8 -> DType.U8
            is CudaStorageSlice.U32 -> DType.U32
            is CudaStorageSlice.I16 -> DType.I16
            is CudaStorageSlice.I32 -> DType.I32
            is CudaStorageSlice.I64 -> DType.I64
            is CudaStorageSlice.Bf16 -> DType.BF16
            is CudaStorageSlice.F16 -> DType.F16
            is CudaStorageSlice.F32 -> DType.F32
            is CudaStorageSlice.F64 -> DType.F64
            is CudaStorageSlice.F8e4m3 -> DType.F8E4M3
            is CudaStorageSlice.F6e2m3 -> DType.F6E2M3
            is CudaStorageSlice.F6e3m2 -> DType.F6E3M2
            is CudaStorageSlice.F4 -> DType.F4
            is CudaStorageSlice.F8e8m0 -> DType.F8E8M0
        }

    public fun deviceBackendStorage(): CudaDevice = this.device

    public fun constSet(s: Scalar, layout: Layout): Result<Unit> {
        val dev = this.device
        val shape = layout.shape()
        val dims = shape.dims()
        val elCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elCount.toUInt())
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val srcO = layout.startOffset()
        val (src, guardSrc, kernelName) =
            when (val matchVal = this.slice) {
                is S.U8 -> {
                    val (s1) = matchVal
                    Pair(slicePtr(s1, srcO), "const_set_u8")
                }
                is S.U32 -> {
                    val (s2) = matchVal
                    Pair(slicePtr(s2, srcO), "const_set_u32")
                }
                is S.I16 -> {
                    val (s3) = matchVal
                    Pair(slicePtr(s3, srcO), "const_set_i16")
                }
                is S.I32 -> {
                    val (s4) = matchVal
                    Pair(slicePtr(s4, srcO), "const_set_i32")
                }
                is S.I64 -> {
                    val (s5) = matchVal
                    Pair(slicePtr(s5, srcO), "const_set_i64")
                }
                is S.Bf16 -> {
                    val (s6) = matchVal
                    Pair(slicePtr(s6, srcO), "const_set_bf16")
                }
                is S.F16 -> {
                    val (s7) = matchVal
                    Pair(slicePtr(s7, srcO), "const_set_f16")
                }
                is S.F32 -> {
                    val (s8) = matchVal
                    Pair(slicePtr(s8, srcO), "const_set_f32")
                }
                is S.F64 -> {
                    val (s9) = matchVal
                    Pair(slicePtr(s9, srcO), "const_set_f64")
                }
                is S.F8e4m3 -> {
                    val (s10) = matchVal
                    Pair(slicePtr(s10, srcO), "const_set_f8_e4m3")
                }
                is S.F4 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "const_set").into())
                }
                is S.F6e2m3 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "const_set").into())
                }
                is S.F6e3m2 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "const_set").into())
                }
                is S.F8e8m0 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "const_set").into())
                }
            }
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName, FILL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val builder = func.builder()
        barg(builder, elCount)
        barg(builder, dims.len())
        ds.builderArg(builder)
        s.builderArg(builder)
        barg(builder, src)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }

    public fun toDtype(layout: Layout, dtype: DType): Result<CudaStorage> {
        val shape = layout.shape()
        val dims = shape.dims()
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val dev = this.device()
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val startO = layout.startOffset()
        // This returns an i64 rather than a &i64, this is useful to get around some temporary
        // scope issue and is safe as long as self.slice does not go out of scope before inp
        // is used.
        val (inp10, guard) =
            when (val matchVal = this.slice) {
                is CudaStorageSlice.U8 -> {
                    val (inp) = matchVal
                    slicePtr(inp, startO)
                }
                is CudaStorageSlice.U32 -> {
                    val (inp1) = matchVal
                    slicePtr(inp1, startO)
                }
                is CudaStorageSlice.I16 -> {
                    val (inp2) = matchVal
                    slicePtr(inp2, startO)
                }
                is CudaStorageSlice.I32 -> {
                    val (inp3) = matchVal
                    slicePtr(inp3, startO)
                }
                is CudaStorageSlice.I64 -> {
                    val (inp4) = matchVal
                    slicePtr(inp4, startO)
                }
                is CudaStorageSlice.Bf16 -> {
                    val (inp5) = matchVal
                    slicePtr(inp5, startO)
                }
                is CudaStorageSlice.F16 -> {
                    val (inp6) = matchVal
                    slicePtr(inp6, startO)
                }
                is CudaStorageSlice.F32 -> {
                    val (inp7) = matchVal
                    slicePtr(inp7, startO)
                }
                is CudaStorageSlice.F64 -> {
                    val (inp8) = matchVal
                    slicePtr(inp8, startO)
                }
                is CudaStorageSlice.F8e4m3 -> {
                    val (inp9) = matchVal
                    slicePtr(inp9, startO)
                }
                is CudaStorageSlice.F4 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_dtype").into())
                }
                is CudaStorageSlice.F6e2m3 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_dtype").into())
                }
                is CudaStorageSlice.F6e3m2 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_dtype").into())
                }
                is CudaStorageSlice.F8e8m0 -> {
                    return Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_dtype").into())
                }
            }
        val inp11 = inp10
        val kernelName = "cast_${this.dtype().asStr()}_${dtype.asStr()}"
        val func =
            when (val tryResult = dev.getOrLoadFunc(kernelName, CAST)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val slice =
            when (dtype) {
                DType.u8 -> {
                    val out =
                        when (val tryResult = dev.alloc<UByte>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder = func.builder()
                    barg(builder, el)
                    barg(builder, dims.len())
                    ds.builderArg(builder)
                    barg(builder, inp11)
                    builder.arg(out)
                    when (val tryResult = builder.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.u8(out)
                }
                DType.u32 -> {
                    val out1 =
                        when (val tryResult = dev.alloc<UInt>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder1 = func.builder()
                    barg(builder1, el)
                    barg(builder1, dims.len())
                    ds.builderArg(builder1)
                    barg(builder1, inp11)
                    builder1.arg(out1)
                    when (val tryResult = builder1.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.u32(out1)
                }
                DType.i64 -> {
                    val out2 =
                        when (val tryResult = dev.alloc<Long>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder2 = func.builder()
                    barg(builder2, el)
                    barg(builder2, dims.len())
                    ds.builderArg(builder2)
                    barg(builder2, inp11)
                    builder2.arg(out2)
                    when (val tryResult = builder2.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.i64(out2)
                }
                DType.bf16 -> {
                    val out3 =
                        when (val tryResult = dev.alloc<Bf16>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder3 = func.builder()
                    barg(builder3, el)
                    barg(builder3, dims.len())
                    ds.builderArg(builder3)
                    barg(builder3, inp11)
                    builder3.arg(out3)
                    when (val tryResult = builder3.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.bf16(out3)
                }
                DType.f16 -> {
                    val out4 =
                        when (val tryResult = dev.alloc<F16>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder4 = func.builder()
                    barg(builder4, el)
                    barg(builder4, dims.len())
                    ds.builderArg(builder4)
                    barg(builder4, inp11)
                    builder4.arg(out4)
                    when (val tryResult = builder4.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f16(out4)
                }
                DType.f32 -> {
                    val out5 =
                        when (val tryResult = dev.alloc<Float>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder5 = func.builder()
                    barg(builder5, el)
                    barg(builder5, dims.len())
                    ds.builderArg(builder5)
                    barg(builder5, inp11)
                    builder5.arg(out5)
                    when (val tryResult = builder5.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f32(out5)
                }
                DType.f64 -> {
                    val out6 =
                        when (val tryResult = dev.alloc<Double>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder6 = func.builder()
                    barg(builder6, el)
                    barg(builder6, dims.len())
                    ds.builderArg(builder6)
                    barg(builder6, inp11)
                    builder6.arg(out6)
                    when (val tryResult = builder6.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f64(out6)
                }
                DType.f8e4m3 -> {
                    val out7 =
                        when (val tryResult = dev.alloc<F8e4m3>(el)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    val builder7 = func.builder()
                    barg(builder7, el)
                    barg(builder7, dims.len())
                    ds.builderArg(builder7)
                    barg(builder7, inp11)
                    builder7.arg(out7)
                    when (val tryResult = builder7.launch(cfg).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                    CudaStorageSlice.f8e4m3(out7)
                }
                DType.i16, DType.i32 -> {
                    return Err(CudaError.InternalError("i16,i32 dtypes are not supported").into())
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Err(CudaError.InternalError("Dummy types not supported in CUDA backend").into())
                }
            }
        return Ok(CudaStorage(slice = slice, device = dev.clone()))
    }

    public fun affine(layout: Layout, mul: Double, add: Double): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Affine(mul, add).map(this.slice, device, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun powf(layout: Layout, e: Double): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Powf(e).map(this.slice, device, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun elu(layout: Layout, alpha: Double): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Elu(alpha).map(this.slice, device, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun reduceOp(op: ReduceOp, layout: Layout, sumDims: List<Int>): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = FastReduce(sumDims, op).map(this.slice, device, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun cmp(op: CmpOp, rhs: CudaStorage, lhsL: Layout, rhsL: Layout): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Cmp(op).map(this.slice, lhsL, rhs.slice, rhsL, device)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun <U : UnaryOpT> unaryImpl(layout: Layout): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = U.V.map(this.slice, device, layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun <B : BinaryOpT> binaryImpl(rhs: CudaStorage, lhsL: Layout, rhsL: Layout): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = B.V.map(this.slice, lhsL, rhs.slice, rhsL, device)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun toCpuStorage(): Result<CpuStorage> {
        if (cudaGraphHtodCacheEnabled()) {
            val isCapturing = (this.device.cudaStream().captureStatus()).rustEquals(Ok(CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE))
            if (isCapturing) {
                bail("to_cpu_storage during CUDA graph capture: host/device copies are not permitted while capturing")
            }
        }
        return when (val matchVal = this.slice) {
            is CudaStorageSlice.U8 -> {
                val (slice) = matchVal
                val cpuStorage =
                    when (val tryResult = slice.stream().cloneDtoh(slice).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.u8(cpuStorage))
            }
            is CudaStorageSlice.U32 -> {
                val (slice1) = matchVal
                val cpuStorage1 =
                    when (val tryResult = slice1.stream().cloneDtoh(slice1).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.u32(cpuStorage1))
            }
            is CudaStorageSlice.I16 -> {
                val (slice2) = matchVal
                val cpuStorage2 =
                    when (val tryResult = slice2.stream().cloneDtoh(slice2).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.i16(cpuStorage2))
            }
            is CudaStorageSlice.I32 -> {
                val (slice3) = matchVal
                val cpuStorage3 =
                    when (val tryResult = slice3.stream().cloneDtoh(slice3).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.i32(cpuStorage3))
            }
            is CudaStorageSlice.I64 -> {
                val (slice4) = matchVal
                val cpuStorage4 =
                    when (val tryResult = slice4.stream().cloneDtoh(slice4).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.i64(cpuStorage4))
            }
            is CudaStorageSlice.Bf16 -> {
                val (slice5) = matchVal
                val cpuStorage5 =
                    when (val tryResult = slice5.stream().cloneDtoh(slice5).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.bf16(cpuStorage5))
            }
            is CudaStorageSlice.F16 -> {
                val (slice6) = matchVal
                val cpuStorage6 =
                    when (val tryResult = slice6.stream().cloneDtoh(slice6).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.f16(cpuStorage6))
            }
            is CudaStorageSlice.F32 -> {
                val (slice7) = matchVal
                val cpuStorage7 =
                    when (val tryResult = slice7.stream().cloneDtoh(slice7).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.f32(cpuStorage7))
            }
            is CudaStorageSlice.F64 -> {
                val (slice8) = matchVal
                val cpuStorage8 =
                    when (val tryResult = slice8.stream().cloneDtoh(slice8).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.f64(cpuStorage8))
            }
            is CudaStorageSlice.F8e4m3 -> {
                val (slice9) = matchVal
                val cpuStorage9 =
                    when (val tryResult = slice9.stream().cloneDtoh(slice9).w()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(CpuStorage.f8e4m3(cpuStorage9))
            }
            is CudaStorageSlice.F4 -> Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_cpu_storage").into())
            is CudaStorageSlice.F6e2m3 -> Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_cpu_storage").into())
            is CudaStorageSlice.F6e3m2 -> Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_cpu_storage").into())
            is CudaStorageSlice.F8e8m0 -> Err(CudaError.UnsupportedDtype(dtype = this.dtype(), op = "to_cpu_storage").into())
        }
    }

    public fun whereCond(layout: Layout, t: CudaStorage, tL: Layout, f: CudaStorage, fL: Layout): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = WhereCond(this, layout).map(t.slice, tL, f.slice, fL, device)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun conv1d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConv1D): Result<CudaStorage> {
        val USE_IM2COL_CONV1D: Boolean = true
        val device = this.device().clone()
        if (USE_IM2COL_CONV1D.inv()) {
            val slice =
                when (val tryResult = Conv1D(params).map(this.slice, l, kernel.slice, kernelL, device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(CudaStorage(slice = slice, device = device))
        }
        val col =
            when (val tryResult = Im2Col1D(lK = params.kSize, stride = params.stride, dilation = params.dilation, padding = params.padding).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val col1 = CudaStorage(slice = col, device = device)
        val lOut = params.lOut()
        val b = params.bSize
        val n = params.cOut
        val k = params.kSize * params.cIn
        val m = lOut
        val colL = Layout.contiguous(Pair(b * m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kernelL1 =
                    when (val tryResult = Layout.contiguousWithOffset(Pair(n, k), kernelL.startOffset()).transpose(0, 1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col1.matmul(kernel, listOf(1, b * m, n, k), colL, kernelL1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            } else {
                // Make the kernel contiguous if not already the case.
                val kernelC =
                    when (val tryResult = this.device().allocUninit(kernelL.shape(), kernel.dtype())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = kernel.copyStridedSrc(kernelC, 0, kernelL)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                val kernelL2 =
                    when (val tryResult = Layout.contiguousWithOffset(Pair(n, k), kernelL.startOffset()).transpose(0, 1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col1.matmul(kernel, listOf(1, b * m, n, k), colL, kernelL2)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val resL =
            when (val tryResult = Layout.contiguous(Triple(b, lOut, n)).transpose(1, 2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val resT =
            when (val tryResult = this.device().allocUninit(resL.shape(), res.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = res.copyStridedSrc(resT, 0, resL)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(resT)
    }

    public fun conv1dBackendStorage(inpL: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConv1D): Result<CudaStorage> {
        val device = this.device().clone()
        if (!kernelL.isContiguous()) {
            val slice =
                when (val tryResult = Conv1D(params).map(this.slice, inpL, kernel.slice, kernelL, device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(CudaStorage(slice = slice, device = device))
        }
        val lOut = params.lOut()
        val dstEl = params.cOut * lOut * params.bSize
        val slice1 =
            run {
                val matchVal = Pair(this.slice, kernel.slice)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                        val (inp) = matchVal.first as S.U8
                        val (k) = matchVal.second as S.U8
                        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k1 = k.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out =
                            when (val tryResult = device.alloc<UByte>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv1d<UByte, UByte>(inp1, inpL, k1, out, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.u8(out)
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                        val (inp2) = matchVal.first as S.Bf16
                        val (k2) = matchVal.second as S.Bf16
                        val inp3 = inp2.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k3 = k2.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out1 =
                            when (val tryResult = device.alloc<Bf16>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                        // version.
                        // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                        when (val tryResult = launchConv1d<Bf16, Float>(inp3, inpL, k3, out1, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.bf16(out1)
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                        val (inp4) = matchVal.first as S.F16
                        val (k4) = matchVal.second as S.F16
                        val inp5 = inp4.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k5 = k4.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out2 =
                            when (val tryResult = device.alloc<F16>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv1d<F16, F16>(inp5, inpL, k5, out2, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f16(out2)
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                        val (inp6) = matchVal.first as S.F32
                        val (k6) = matchVal.second as S.F32
                        val inp7 = inp6.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k7 = k6.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out3 =
                            when (val tryResult = device.alloc<Float>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv1d<Float, Float>(inp7, inpL, k7, out3, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f32(out3)
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                        val (inp8) = matchVal.first as S.F64
                        val (k8) = matchVal.second as S.F64
                        val inp9 = inp8.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k9 = k8.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out4 =
                            when (val tryResult = device.alloc<Double>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv1d<Double, Double>(inp9, inpL, k9, out4, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f64(out4)
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 ->
                        when (val tryResult = Err(CudaError.InternalError("conv1d does not support u32"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I16 && matchVal.second is S.I16 ->
                        when (val tryResult = Err(CudaError.InternalError("conv1d does not support i16"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I32 && matchVal.second is S.I32 ->
                        when (val tryResult = Err(CudaError.InternalError("conv1d does not support i32"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I64 && matchVal.second is S.I64 ->
                        when (val tryResult = Err(CudaError.InternalError("conv1d does not support i64"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                        when (val tryResult = Err(CudaError.InternalError("conv1d does not support f8e4m3"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                    else ->
                        when (val tryResult = Err(CudaError.InternalError("dtype mismatch in conv1d"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        return Ok(CudaStorage(slice = slice1, device = device))
    }

    public fun convTranspose1d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConvTranspose1D): Result<CudaStorage> {
        val USE_COL2IM_CONV1D_TR: Boolean = true
        val device = this.device().clone()
        val canUseCol2im = kernelL.isContiguous() && params.dilation == 1 && params.padding == 0 && params.outputPadding == 0
        val slice =
            if (USE_COL2IM_CONV1D_TR && canUseCol2im) {
                val (bSize, cIn, lIn) =
                    when (val tryResult = l.shape().dims3()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val (cIn2, cOut, kSize) =
                    when (val tryResult = kernelL.shape().dims3()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                if (!kernelL.isContiguous()) {
                    bail("convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}")
                }
                if (cIn != cIn2) {
                    bail("convtr1d: shape mismatch on c_in {:?} {:?}", l.shape(), kernelL.shape())
                }
                val col =
                    run {
                        // This merges the last two dimensions of the kernel together.
                        val kernelLMm = Layout.new(Triple(bSize, cIn, kSize * cOut).into(), mutableListOf(0, kSize * cOut, 1), kernelL.startOffset())
                        when (
                            val tryResult =
                                this.matmul(
                                    kernel,
                                    listOf(bSize, lIn, cOut * kSize, cIn),
                                    when (val tryResult = l.transpose(1, 2)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    },
                                    kernelLMm,
                                )
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                val colL = Layout.contiguous(listOf(bSize, lIn, cOut, kSize))
                when (val tryResult = Col2Im1D(stride = params.stride).map(col.slice, device, colL)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            } else {
                when (val tryResult = ConvTranspose1D(params).map(this.slice, l, kernel.slice, kernelL, device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun conv2d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConv2D): Result<CudaStorage> {
        val USE_IM2COL_CONV2D: Boolean = true
        val device = this.device().clone()
        if (USE_IM2COL_CONV2D.inv()) {
            val slice =
                when (val tryResult = Conv2D(params).map(this.slice, l, kernel.slice, kernelL, device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(CudaStorage(slice = slice, device = device))
        }
        val col =
            when (val tryResult = Im2Col(hK = params.kH, wK = params.kW, stride = params.stride, dilation = params.dilation, padding = params.padding).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val col1 = CudaStorage(slice = col, device = device)
        val hOut = params.outH()
        val wOut = params.outW()
        val b = params.bSize
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val m = hOut * wOut
        val colL = Layout.contiguous(Pair(b * m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kernelL1 =
                    when (val tryResult = Layout.contiguousWithOffset(Pair(n, k), kernelL.startOffset()).transpose(0, 1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col1.matmul(kernel, listOf(1, b * m, n, k), colL, kernelL1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            } else {
                // Make the kernel contiguous if not already the case. copyStridedSrc writes the
                // materialized kernel starting at offset 0, so the matmul layout must use offset 0,
                // not the original (strided) kernel's start offset.
                val kernelC =
                    when (val tryResult = this.device().allocUninit(kernelL.shape(), kernel.dtype())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = kernel.copyStridedSrc(kernelC, 0, kernelL)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                val kernelL2 =
                    when (val tryResult = Layout.contiguousWithOffset(Pair(n, k), 0).transpose(0, 1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = col1.matmul(kernelC, listOf(1, b * m, n, k), colL, kernelL2)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val resL =
            when (
                val tryResult =
                    when (val tryResult = Layout.contiguous(listOf(b, hOut, wOut, n)).transpose(1, 2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.transpose(1, 3)
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val resT =
            when (val tryResult = this.device().allocUninit(resL.shape(), res.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = res.copyStridedSrc(resT, 0, resL)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(resT)
    }

    public fun conv2dBackendStorage(inpL: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConv2D): Result<CudaStorage> {
        val device = this.device().clone()
        if (!kernelL.isContiguous()) {
            val slice =
                when (val tryResult = Conv2D(params).map(this.slice, inpL, kernel.slice, kernelL, device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(CudaStorage(slice = slice, device = device))
        }
        val (outW, outH) = Pair(params.outW(), params.outH())
        val dstEl = params.cOut * outW * outH * params.bSize
        val slice1 =
            run {
                val matchVal = Pair(this.slice, kernel.slice)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                        val (inp) = matchVal.first as S.U8
                        val (k) = matchVal.second as S.U8
                        val inp1 = inp.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k1 = k.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out =
                            when (val tryResult = device.alloc<UByte>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv2d<UByte, UByte>(inp1, inpL, k1, out, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.u8(out)
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                        val (inp2) = matchVal.first as S.Bf16
                        val (k2) = matchVal.second as S.Bf16
                        val inp3 = inp2.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k3 = k2.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out1 =
                            when (val tryResult = device.alloc<Bf16>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                        // version.
                        // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                        when (val tryResult = launchConv2d<Bf16, Float>(inp3, inpL, k3, out1, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.bf16(out1)
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                        val (inp4) = matchVal.first as S.F16
                        val (k4) = matchVal.second as S.F16
                        val inp5 = inp4.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k5 = k4.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out2 =
                            when (val tryResult = device.alloc<F16>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv2d<F16, F16>(inp5, inpL, k5, out2, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f16(out2)
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                        val (inp6) = matchVal.first as S.F32
                        val (k6) = matchVal.second as S.F32
                        val inp7 = inp6.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k7 = k6.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out3 =
                            when (val tryResult = device.alloc<Float>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv2d<Float, Float>(inp7, inpL, k7, out3, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f32(out3)
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                        val (inp8) = matchVal.first as S.F64
                        val (k8) = matchVal.second as S.F64
                        val inp9 = inp8.slice(inpL.startOffset() until Int.MAX_VALUE)
                        val k9 = k8.slice(kernelL.startOffset() until Int.MAX_VALUE)
                        var out4 =
                            when (val tryResult = device.alloc<Double>(dstEl)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = launchConv2d<Double, Double>(inp9, inpL, k9, out4, params, device).mapErr(Error::wrap)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        S.f64(out4)
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 ->
                        when (val tryResult = Err(CudaError.InternalError("conv2d does not support u32"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I16 && matchVal.second is S.I16 ->
                        when (val tryResult = Err(CudaError.InternalError("conv2d does not support i16"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I32 && matchVal.second is S.I32 ->
                        when (val tryResult = Err(CudaError.InternalError("conv2d does not support i32"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.I64 && matchVal.second is S.I64 ->
                        when (val tryResult = Err(CudaError.InternalError("conv2d does not support i64"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                        when (val tryResult = Err(CudaError.InternalError("conv2d does not support f8e4m3"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                    else ->
                        when (val tryResult = Err(CudaError.InternalError("dtype mismatch in conv2d"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        return Ok(CudaStorage(slice = slice1, device = device))
    }

    public fun convTranspose2d(l: Layout, kernel: CudaStorage, kernelL: Layout, params: ParamsConvTranspose2D): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = ConvTranspose2D(params).map(this.slice, l, kernel.slice, kernelL, device)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun avgPool2d(l: Layout, k: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Pool2D(wK = k.first, hK = k.second, wStride = stride.first, hStride = stride.second, op = PoolOp.Avg).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun maxPool2d(l: Layout, k: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Pool2D(wK = k.first, hK = k.second, wStride = stride.first, hStride = stride.second, op = PoolOp.Max).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun upsampleNearest1d(arg0: Layout, outSz: Int): Result<CudaStorage> = bail("upsample-nearest1d is not supported on cuda")

    public fun upsampleNearest2d(l: Layout, outW: Int, outH: Int): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = UpsampleNearest2D(outW, outH).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun upsampleBilinear2d(l: Layout, outH: Int, outW: Int, alignCorners: Boolean, scaleH: Double?, scaleW: Double?): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = UpsampleBilinear2D(outW = outW, outH = outH, alignCorners = alignCorners, scaleHFactor = scaleH, scaleWFactor = scaleW).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun indexSelect(ids: CudaStorage, l: Layout, idsL: Layout, dim: Int): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = IndexSelect(ids, idsL, dim).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun gather(l: Layout, ids: CudaStorage, idsL: Layout, dim: Int): Result<CudaStorage> {
        val device = this.device().clone()
        val slice =
            when (val tryResult = Gather(ids, idsL, dim).map(this.slice, device, l)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun scatterSet(l: Layout, ids: CudaStorage, idsL: Layout, src: CudaStorage, srcL: Layout, dim: Int): Result<Unit> {
        val device = this.device().clone()
        return Scatter(ids, idsL, dim).map(this.slice, l, src.slice, srcL, device)
    }

    public fun scatterAddSet(l: Layout, ids: CudaStorage, idsL: Layout, src: CudaStorage, srcL: Layout, dim: Int): Result<Unit> {
        val device = this.device().clone()
        return ScatterAdd(ids, idsL, dim).map(this.slice, l, src.slice, srcL, device)
    }

    public fun indexAdd(l: Layout, ids: CudaStorage, idsL: Layout, src: CudaStorage, srcL: Layout, dim: Int): Result<CudaStorage> {
        val device = this.device().clone()
        val acc =
            when (val tryResult = device.allocUninit(l.shape(), this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.copyStridedSrc(acc, 0, l)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        when (val tryResult = IndexAdd(ids, idsL, dim).map(acc.slice, l, src.slice, srcL, device)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(acc)
    }

    public fun matmul(rhs: CudaStorage, tupleArg1: List<Any?>, lhsL: Layout, rhsL: Layout): Result<CudaStorage> {
        val elemCount = b * m * n
        val dev = this.device
        val slice =
            run {
                val matchVal = Pair(this.slice, rhs.slice)
                when {
                    matchVal.first is CudaStorageSlice.Bf16 && matchVal.second is CudaStorageSlice.Bf16 -> {
                        val (lhs) = matchVal.first as CudaStorageSlice.Bf16
                        val (rhs1) = matchVal.second as CudaStorageSlice.Bf16
                        val lhs1 = lhs.slice(lhsL.startOffset() until Int.MAX_VALUE)
                        val rhs2 = rhs1.slice(rhsL.startOffset() until Int.MAX_VALUE)
                        val cfg =
                            when (val tryResult = gemmConfig(ONE, ZERO, listOf(b, m, n, k), lhsL, rhsL)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        var out =
                            when (val tryResult = dev.alloc<Bf16>(elemCount)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = gemmStridedBatchedBf16(this.device.blas, cfg, rhs2, lhs1, out).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        CudaStorageSlice.bf16(out)
                    }
                    matchVal.first is CudaStorageSlice.F16 && matchVal.second is CudaStorageSlice.F16 -> {
                        val (lhs2) = matchVal.first as CudaStorageSlice.F16
                        val (rhs3) = matchVal.second as CudaStorageSlice.F16
                        val lhs3 = lhs2.slice(lhsL.startOffset() until Int.MAX_VALUE)
                        val rhs4 = rhs3.slice(rhsL.startOffset() until Int.MAX_VALUE)
                        val cfg1 =
                            when (val tryResult = gemmConfig(ONE, ZERO, listOf(b, m, n, k), lhsL, rhsL)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        var out1 =
                            when (val tryResult = dev.alloc<F16>(elemCount)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = gemmStridedBatchedF16(this.device.blas, cfg1, rhs4, lhs3, out1).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        CudaStorageSlice.f16(out1)
                    }
                    matchVal.first is CudaStorageSlice.F32 && matchVal.second is CudaStorageSlice.F32 -> {
                        val (lhs4) = matchVal.first as CudaStorageSlice.F32
                        val (rhs5) = matchVal.second as CudaStorageSlice.F32
                        val lhs5 = lhs4.slice(lhsL.startOffset() until Int.MAX_VALUE)
                        val rhs6 = rhs5.slice(rhsL.startOffset() until Int.MAX_VALUE)
                        val cfg2 =
                            when (val tryResult = gemmConfig(1.0, 0.0, listOf(b, m, n, k), lhsL, rhsL)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        var out2 =
                            when (val tryResult = dev.alloc<Float>(elemCount)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = gemmStridedBatchedF32(this.device.blas, cfg2, rhs6, lhs5, out2).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        CudaStorageSlice.f32(out2)
                    }
                    matchVal.first is CudaStorageSlice.F64 && matchVal.second is CudaStorageSlice.F64 -> {
                        val (lhs6) = matchVal.first as CudaStorageSlice.F64
                        val (rhs7) = matchVal.second as CudaStorageSlice.F64
                        val lhs7 = lhs6.slice(lhsL.startOffset() until Int.MAX_VALUE)
                        val rhs8 = rhs7.slice(rhsL.startOffset() until Int.MAX_VALUE)
                        val cfg3 =
                            when (val tryResult = gemmConfig(1.0, 0.0, listOf(b, m, n, k), lhsL, rhsL)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        var out3 =
                            when (val tryResult = dev.alloc<Double>(elemCount)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (
                            val tryResult =
                                this.device.blas
                                    .gemmStridedBatched(cfg3, rhs8, lhs7, out3)
                                    .w()
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        CudaStorageSlice.f64(out3)
                    }
                    else ->
                        when (val tryResult = Err(CudaError.InternalError("dtype mismatch in matmul op"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        val device = dev.clone()
        return Ok(CudaStorage(slice = slice, device = device))
    }

    public fun copy2d(dst: CudaStorage, d1: Int, d2: Int, srcS: Int, dstS: Int, srcO: Int, dstO: Int): Result<Unit> {
        val dev = this.device
        val d11 = d1.toUInt()
        val d21 = d2.toUInt()
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if (d11 == 0u || d21 == 0u) {
            return Ok(Unit)
        }
        val dstS1 = dstS.toUInt()
        val srcS1 = srcS.toUInt()
        val (src, guardSrc, dst1, guardDst, kname) =
            run {
                val matchVal = Pair(this.slice, dst.slice)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                        val (s) = matchVal.first as S.U8
                        val (d) = matchVal.second as S.U8
                        Triple(slicePtr(s, srcO), slicePtr(d, dstO), "copy2d_u8")
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 -> {
                        val (s1) = matchVal.first as S.U32
                        val (d1) = matchVal.second as S.U32
                        Triple(slicePtr(s1, srcO), slicePtr(d1, dstO), "copy2d_u32")
                    }
                    matchVal.first is S.I16 && matchVal.second is S.I16 -> {
                        val (s2) = matchVal.first as S.I16
                        val (d2) = matchVal.second as S.I16
                        Triple(slicePtr(s2, srcO), slicePtr(d2, dstO), "copy2d_i16")
                    }
                    matchVal.first is S.I32 && matchVal.second is S.I32 -> {
                        val (s3) = matchVal.first as S.I32
                        val (d3) = matchVal.second as S.I32
                        Triple(slicePtr(s3, srcO), slicePtr(d3, dstO), "copy2d_i32")
                    }
                    matchVal.first is S.I64 && matchVal.second is S.I64 -> {
                        val (s4) = matchVal.first as S.I64
                        val (d4) = matchVal.second as S.I64
                        Triple(slicePtr(s4, srcO), slicePtr(d4, dstO), "copy2d_i64")
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                        val (s5) = matchVal.first as S.Bf16
                        val (d5) = matchVal.second as S.Bf16
                        Triple(slicePtr(s5, srcO), slicePtr(d5, dstO), "copy2d_bf16")
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                        val (s6) = matchVal.first as S.F16
                        val (d6) = matchVal.second as S.F16
                        Triple(slicePtr(s6, srcO), slicePtr(d6, dstO), "copy2d_f16")
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                        val (s7) = matchVal.first as S.F32
                        val (d7) = matchVal.second as S.F32
                        Triple(slicePtr(s7, srcO), slicePtr(d7, dstO), "copy2d_f32")
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                        val (s8) = matchVal.first as S.F64
                        val (d8) = matchVal.second as S.F64
                        Triple(slicePtr(s8, srcO), slicePtr(d8, dstO), "copy2d_f64")
                    }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                        val (s9) = matchVal.first as S.F8e4m3
                        val (d9) = matchVal.second as S.F8e4m3
                        Triple(slicePtr(s9, srcO), slicePtr(d9, dstO), "copy2d_u8")
                    }
                    matchVal.first is S.F8e8m0 && matchVal.second is S.F8e8m0 -> {
                        val (s10) = matchVal.first as S.F8e8m0
                        val (d10) = matchVal.second as S.F8e8m0
                        Triple(slicePtr(s10, srcO), slicePtr(d10, dstO), "copy2d_u8")
                    }
                    else ->
                        when (val tryResult = Err(CudaError.InternalError("dtype mismatch in copy2d"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        val func =
            when (val tryResult = dev.getOrLoadFunc(kname, FILL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cfg = LaunchConfig.forNumElems(d11 * d21)
        val builder = func.builder()
        barg(builder, src)
        barg(builder, dst1)
        barg(builder, d11)
        barg(builder, d21)
        builder.arg(srcS1)
        builder.arg(dstS1)
        // SAFETY: ffi.
        when (val tryResult = builder.launch(cfg).w()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }

    public fun copyStridedSrc(dst: CudaStorage, dstOffset: Int, srcL: Layout): Result<Unit> {
        val srcShape = srcL.shape()
        val dims = srcShape.dims()
        val elCount = srcShape.elemCount()
        if (elCount == 0) {
            return Ok(Unit)
        }
        val cfg = LaunchConfig.forNumElems(elCount.toUInt())
        val dev = this.device
        val ds =
            when (val tryResult = SlicePtrOrNull.paramsFromLayout(dev, srcL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        run {
            val matchVal = Pair(this.slice, dst.slice)
            when {
                matchVal.first is CudaStorageSlice.Bf16 && matchVal.second is CudaStorageSlice.Bf16 -> {
                    val (src) = matchVal.first as CudaStorageSlice.Bf16
                    val (dst1) = matchVal.second as CudaStorageSlice.Bf16
                    var (src1, dst2) = sliceSrcAndDst(src, srcL, dst1, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src1, dst2)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_bf16", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder = func.builder()
                        barg(builder, elCount)
                        barg(builder, dims.len())
                        ds.builderArg(builder)
                        builder.arg(src1)
                        builder.arg(dst2)
                        // SAFETY: ffi.
                        when (val tryResult = builder.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.F16 && matchVal.second is CudaStorageSlice.F16 -> {
                    val (src2) = matchVal.first as CudaStorageSlice.F16
                    val (dst3) = matchVal.second as CudaStorageSlice.F16
                    var (src3, dst4) = sliceSrcAndDst(src2, srcL, dst3, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src3, dst4)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func1 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_f16", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder1 = func1.builder()
                        barg(builder1, elCount)
                        barg(builder1, dims.len())
                        ds.builderArg(builder1)
                        builder1.arg(src3)
                        builder1.arg(dst4)
                        // SAFETY: ffi.
                        when (val tryResult = builder1.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.F32 && matchVal.second is CudaStorageSlice.F32 -> {
                    val (src4) = matchVal.first as CudaStorageSlice.F32
                    val (dst5) = matchVal.second as CudaStorageSlice.F32
                    var (src5, dst6) = sliceSrcAndDst(src4, srcL, dst5, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src5, dst6)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func2 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_f32", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder2 = func2.builder()
                        barg(builder2, elCount)
                        barg(builder2, dims.len())
                        ds.builderArg(builder2)
                        builder2.arg(src5)
                        builder2.arg(dst6)
                        // SAFETY: ffi.
                        when (val tryResult = builder2.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.U8 && matchVal.second is CudaStorageSlice.U8 -> {
                    val (src6) = matchVal.first as CudaStorageSlice.U8
                    val (dst7) = matchVal.second as CudaStorageSlice.U8
                    var (src7, dst8) = sliceSrcAndDst(src6, srcL, dst7, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src7, dst8)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func3 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_u8", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder3 = func3.builder()
                        barg(builder3, elCount)
                        barg(builder3, dims.len())
                        ds.builderArg(builder3)
                        builder3.arg(src7)
                        builder3.arg(dst8)
                        // SAFETY: ffi.
                        when (val tryResult = builder3.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.U32 && matchVal.second is CudaStorageSlice.U32 -> {
                    val (src8) = matchVal.first as CudaStorageSlice.U32
                    val (dst9) = matchVal.second as CudaStorageSlice.U32
                    var (src9, dst10) = sliceSrcAndDst(src8, srcL, dst9, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src9, dst10)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func4 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_u32", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder4 = func4.builder()
                        barg(builder4, elCount)
                        barg(builder4, dims.len())
                        ds.builderArg(builder4)
                        builder4.arg(src9)
                        builder4.arg(dst10)
                        // SAFETY: ffi.
                        when (val tryResult = builder4.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.I16 && matchVal.second is CudaStorageSlice.I16 -> {
                    val (src10) = matchVal.first as CudaStorageSlice.I16
                    val (dst11) = matchVal.second as CudaStorageSlice.I16
                    var (src11, dst12) = sliceSrcAndDst(src10, srcL, dst11, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src11, dst12)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func5 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_i16", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder5 = func5.builder()
                        barg(builder5, elCount)
                        barg(builder5, dims.len())
                        ds.builderArg(builder5)
                        builder5.arg(src11)
                        builder5.arg(dst12)
                        // SAFETY: ffi.
                        when (val tryResult = builder5.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.I32 && matchVal.second is CudaStorageSlice.I32 -> {
                    val (src12) = matchVal.first as CudaStorageSlice.I32
                    val (dst13) = matchVal.second as CudaStorageSlice.I32
                    var (src13, dst14) = sliceSrcAndDst(src12, srcL, dst13, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src13, dst14)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func6 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_i32", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder6 = func6.builder()
                        barg(builder6, elCount)
                        barg(builder6, dims.len())
                        ds.builderArg(builder6)
                        builder6.arg(src13)
                        builder6.arg(dst14)
                        // SAFETY: ffi.
                        when (val tryResult = builder6.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.I64 && matchVal.second is CudaStorageSlice.I64 -> {
                    val (src14) = matchVal.first as CudaStorageSlice.I64
                    val (dst15) = matchVal.second as CudaStorageSlice.I64
                    var (src15, dst16) = sliceSrcAndDst(src14, srcL, dst15, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src15, dst16)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func7 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_i64", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder7 = func7.builder()
                        barg(builder7, elCount)
                        barg(builder7, dims.len())
                        ds.builderArg(builder7)
                        builder7.arg(src15)
                        builder7.arg(dst16)
                        // SAFETY: ffi.
                        when (val tryResult = builder7.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.F64 && matchVal.second is CudaStorageSlice.F64 -> {
                    val (src16) = matchVal.first as CudaStorageSlice.F64
                    val (dst17) = matchVal.second as CudaStorageSlice.F64
                    var (src17, dst18) = sliceSrcAndDst(src16, srcL, dst17, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src17, dst18)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func8 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_f64", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder8 = func8.builder()
                        barg(builder8, elCount)
                        barg(builder8, dims.len())
                        ds.builderArg(builder8)
                        builder8.arg(src17)
                        builder8.arg(dst18)
                        // SAFETY: ffi.
                        when (val tryResult = builder8.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                matchVal.first is CudaStorageSlice.F8e4m3 && matchVal.second is CudaStorageSlice.F8e4m3 -> {
                    val (src18) = matchVal.first as CudaStorageSlice.F8e4m3
                    val (dst19) = matchVal.second as CudaStorageSlice.F8e4m3
                    var (src19, dst20) = sliceSrcAndDst(src18, srcL, dst19, dstOffset)
                    if (srcL.isContiguous()) {
                        when (val tryResult = dev.memcpyDtod(src19, dst20)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    } else {
                        val func9 =
                            when (val tryResult = dev.getOrLoadFunc("ucopy_f8e4m3", UNARY)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val builder9 = func9.builder()
                        barg(builder9, elCount)
                        barg(builder9, dims.len())
                        ds.builderArg(builder9)
                        builder9.arg(src19)
                        builder9.arg(dst20)
                        // SAFETY: ffi.
                        when (val tryResult = builder9.launch(cfg).w()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
                else ->
                    when (val tryResult = Err(CudaError.InternalError("dtype mismatch in copy_strided op"))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        }
        return Ok(Unit)
    }

    public companion object {
        public fun <T : CudaDType> wrapCudaSlice(slice: CudaSlice<T>, device: CudaDevice, wrapCudaSliceTFn: (Any?) -> Any?): CudaStorage = wrapCudaSliceTFn(slice, device)
    }
}
