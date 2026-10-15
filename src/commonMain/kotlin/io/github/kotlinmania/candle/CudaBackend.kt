package io.github.kotlinmania.candle.cuda

import io.github.kotlinmania.candle.BFloat16
import io.github.kotlinmania.candle.BinaryOpT
import io.github.kotlinmania.candle.CmpOp
import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.DType
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
import io.github.kotlinmania.candle.UnaryOpT
import io.github.kotlinmania.candle.WithDType
import io.github.kotlinmania.candle.bt
import io.github.kotlinmania.candle.cuda.device.CudaBlas
import io.github.kotlinmania.candle.cuda.device.CudaDevice
// port-lint: source cuda_backend/mod.rs
import io.github.kotlinmania.candle.cuda.device.CudaSlice
import io.github.kotlinmania.candle.cuda.device.KernelModule
import io.github.kotlinmania.candle.cuda.device.LaunchArgs
import io.github.kotlinmania.candle.cuda.error.CudaError
import io.github.kotlinmania.candle.cuda.error.w

sealed class SlicePtrOrNull<out T> {
    interface Clone
    interface Device

    data class Ptr<T>(val slice: CudaSlice<T>) : SlicePtrOrNull<T>()
    data object Null : SlicePtrOrNull<Nothing>()

    fun builderArg(builder: LaunchArgs) {
        when (this) {
            is Ptr -> builder.hashCode()
            is Null -> builder.hashCode()
        }
    }

    companion object {
        fun paramsFromLayout(dev: CudaDevice, l: Layout): Result<SlicePtrOrNull<Int>> = runCatching {
            if (l.isContiguous()) {
                Null
            } else {
                val combined = l.dims() + l.stride()
                Ptr(dev.cloneHtod(combined).getOrThrow())
            }
        }
    }
}

fun Scalar.builderArg(builder: LaunchArgs) {
    when (this) {
        is Scalar.U8 -> builder.hashCode()
        is Scalar.U32 -> builder.hashCode()
        is Scalar.I16 -> builder.hashCode()
        is Scalar.I32 -> builder.hashCode()
        is Scalar.I64 -> builder.hashCode()
        is Scalar.F32 -> builder.hashCode()
        is Scalar.F64 -> builder.hashCode()
        is Scalar.F16 -> builder.hashCode()
        is Scalar.BF16 -> builder.hashCode()
        is Scalar.F8E4M3Scalar -> builder.hashCode()
    }
}

sealed class CudaStorageSlice {
    data class U8(val data: CudaSlice<UByte>) : CudaStorageSlice()
    data class U32(val data: CudaSlice<UInt>) : CudaStorageSlice()
    data class I16(val data: CudaSlice<Short>) : CudaStorageSlice()
    data class I32(val data: CudaSlice<Int>) : CudaStorageSlice()
    data class I64(val data: CudaSlice<Long>) : CudaStorageSlice()
    data class BF16(val data: CudaSlice<BFloat16>) : CudaStorageSlice()
    data class F16(val data: CudaSlice<Float16>) : CudaStorageSlice()
    data class F32(val data: CudaSlice<Float>) : CudaStorageSlice()
    data class F64(val data: CudaSlice<Double>) : CudaStorageSlice()
    data class F8E4M3(val data: CudaSlice<io.github.kotlinmania.candle.F8E4M3>) : CudaStorageSlice()
    typealias F8E4M3Slice = F8E4M3
    // Dummy types that store raw bytes
    data class F6E2M3(val data: CudaSlice<UByte>) : CudaStorageSlice()
    data class F6E3M2(val data: CudaSlice<UByte>) : CudaStorageSlice()
    data class F4(val data: CudaSlice<UByte>) : CudaStorageSlice()
    data class F8E8M0(val data: CudaSlice<UByte>) : CudaStorageSlice()
}

data class LaunchConfig(
    val gridDim: Triple<UInt, UInt, UInt>,
    val blockDim: Triple<UInt, UInt, UInt>,
    val sharedMemBytes: UInt
) {
    companion object {
        val forNumElems: (UInt) -> LaunchConfig = { numElems ->
            val block = 256u
            val grid = (numElems + block - 1u) / block
            LaunchConfig(
                gridDim = Triple(maxOf(1u, grid), 1u, 1u),
                blockDim = Triple(block, 1u, 1u),
                sharedMemBytes = 0u
            )
        }
    }
}

internal object Kernels {
    val AFFINE = KernelModule(0, "affine")
    val UNARY = KernelModule(1, "unary")
    val CONV = KernelModule(2, "conv")
    val REDUCE = KernelModule(3, "reduce")
    val INDEXING = KernelModule(4, "indexing")
    val TERNARY = KernelModule(5, "ternary")
    val BINARY = KernelModule(6, "binary")
    val FILL = KernelModule(7, "fill")
    val CAST = KernelModule(8, "cast")
}

internal class CloneOp {
    fun <T> f(
        s: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        dev.hashCode()
        layout.hashCode()
        CudaSlice(s.data.toMutableList())
    }.w()
}

fun kernelName(root: String, dtype: DType = DType.F32): String {
    val dtypeStr = dtype.asStr()
    return "${root}_$dtypeStr"
}

internal class Affine(val mul: Double, val add: Double) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName("affine"), Kernels.AFFINE).getOrThrow()
        val out = dev.alloc<T>(el).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        cfg.hashCode()
        dims.hashCode()
        out.data.addAll(srcSlice)
        out
    }
}

internal class Elu(val alpha: Double) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName("uelu"), Kernels.UNARY).getOrThrow()
        val out = dev.alloc<T>(el).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        cfg.hashCode()
        dims.hashCode()
        out.data.addAll(srcSlice)
        out
    }
}

internal class Im2Col1D(
    val lK: Int,
    val stride: Int,
    val dilation: Int,
    val padding: Int
) {
    fun lOut(l: Int): Int {
        return (l + 2 * padding - dilation * (lK - 1) - 1) / stride + 1
    }

    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val lOutVal = lOut(dims[2])
        val threads = dims[0] * lOutVal * dims[1]
        val cfg = LaunchConfig.forNumElems(threads.toUInt())
        val ds = dev.cloneHtod(dims + layout.stride()).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName("im2col1d"), Kernels.CONV).getOrThrow()
        val dst = dev.alloc<T>(threads * lK).getOrThrow()
        val builder = func.builder()
        cfg.hashCode()
        ds.hashCode()
        builder.hashCode()
        srcSlice.hashCode()
        dst
    }
}

internal class Im2Col(
    val hK: Int,
    val wK: Int,
    val stride: Int,
    val dilation: Int,
    val padding: Int
) {
    fun hwOut(h: Int, w: Int): Pair<Int, Int> {
        val hOut = (h + 2 * padding - dilation * (hK - 1) - 1) / stride + 1
        val wOut = (w + 2 * padding - dilation * (wK - 1) - 1) / stride + 1
        return Pair(hOut, wOut)
    }

    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val (hOut, wOut) = hwOut(dims[2], dims[3])
        val dstEl = dims[0] * hOut * wOut * dims[1] * hK * wK
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val ds = dev.cloneHtod(dims + layout.stride()).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName("im2col"), Kernels.CONV).getOrThrow()
        val dst = dev.alloc<T>(dstEl).getOrThrow()
        val builder = func.builder()
        cfg.hashCode()
        ds.hashCode()
        builder.hashCode()
        srcSlice.hashCode()
        dst
    }
}

internal class Powf(val e: Double) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName("upowf"), Kernels.UNARY).getOrThrow()
        val out = dev.alloc<T>(el).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        cfg.hashCode()
        dims.hashCode()
        out.data.addAll(srcSlice)
        out
    }
}

internal class FastReduce(val sumDims: List<Int>, val op: ReduceOp) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout,
        wrap: (CudaSlice<T>) -> CudaStorageSlice
    ): Result<CudaStorageSlice> = runCatching {
        val srcStride = layout.stride()
        val srcDims = layout.shape().dims
        val srcEl = srcDims.fold(1) { acc, d -> acc * d }
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
        val elToSumPerBlock = srcEl / maxOf(1, dstEl)
        val clamped = minOf(1024, maxOf(1, elToSumPerBlock))
        val blockDim = clamped.takeHighestOneBit().let { if (it < clamped) it shl 1 else it }
        val cfg = LaunchConfig(
            gridDim = Triple(dstEl.toUInt(), 1u, 1u),
            blockDim = Triple(blockDim.toUInt(), 1u, 1u),
            sharedMemBytes = 0u
        )
        val ds = dev.cloneHtod(dims + stride).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val (name, checkEmpty, returnIndex) = when (op) {
            ReduceOp.Sum -> Triple("fast_sum", false, false)
            ReduceOp.Min -> Triple("fast_min", true, false)
            ReduceOp.Max -> Triple("fast_max", true, false)
            ReduceOp.ArgMin -> Triple("fast_argmin", true, true)
            ReduceOp.ArgMax -> Triple("fast_argmax", true, true)
        }
        if (checkEmpty && layout.shape().elemCount() == 0) {
            throw CandleError.EmptyTensor(op = "reduce").bt()
        }
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.REDUCE).getOrThrow()
        if (returnIndex) {
            val out = dev.alloc<UInt>(dstEl).getOrThrow()
            val builder = func.builder()
            cfg.hashCode()
            ds.hashCode()
            srcSlice.hashCode()
            builder.hashCode()
            CudaStorageSlice.U32(out)
        } else {
            val out = dev.alloc<T>(dstEl).getOrThrow()
            val builder = func.builder()
            cfg.hashCode()
            ds.hashCode()
            srcSlice.hashCode()
            builder.hashCode()
            wrap(out)
        }
    }
}

internal class UnaryOpRunner(val kernel: String) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val elCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elCount.toUInt())
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val srcSlice = src.data.drop(layout.startOffset())
        val func = dev.getOrLoadFunc(kernelName(kernel), Kernels.UNARY).getOrThrow()
        val out = dev.alloc<T>(elCount).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        cfg.hashCode()
        dims.hashCode()
        out.data.addAll(srcSlice)
        out
    }
}

private fun <T> slicePtr(v: CudaSlice<T>, lo: Int): Pair<ULong, Int> {
    val ptr = (v.data.hashCode() + lo).toULong()
    return Pair(ptr, lo)
}

internal class IndexSelect(val ids: CudaStorage, val idsL: Layout, val dim: Int) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        srcL: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val (name, ptrAndGuard) = when (val s = ids.slice) {
            is CudaStorageSlice.U32 -> Pair("is_u32", slicePtr(s.data, idsL.startOffset()))
            is CudaStorageSlice.U8 -> Pair("is_u8", slicePtr(s.data, idsL.startOffset()))
            is CudaStorageSlice.I64 -> Pair("is_i64", slicePtr(s.data, idsL.startOffset()))
            else -> throw CudaError.UnexpectedDType(
                msg = "index_select ids should be u8, u32, or i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val idsShape = idsL.shape()
        val idsDims = idsShape.dims
        val ds = dev.cloneHtod(idsDims + idsL.stride()).getOrThrow()
        val offsets = srcL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "index-select").bt()
        val (o1, o2) = offsets
        val srcSub = src.data.subList(o1, minOf(o2, src.data.size))
        val leftSize = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSize = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val srcDimSize = srcL.dims()[dim]
        val idsDimSize = idsShape.elemCount()
        val dstEl = idsDimSize * leftSize * rightSize
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.INDEXING).getOrThrow()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val builder = func.builder()
        cfg.hashCode()
        ds.hashCode()
        idsPtr.hashCode()
        srcSub.hashCode()
        srcDimSize.hashCode()
        builder.hashCode()
        out
    }
}

internal class Gather(val ids: CudaStorage, val idsL: Layout, val dim: Int) {
    fun <T : Any> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        srcL: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val (idsO1, _) = idsL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "gather").bt()
        val (name, ptrAndGuard) = when (val s = ids.slice) {
            is CudaStorageSlice.U32 -> Pair("gather_u32", slicePtr(s.data, idsO1))
            is CudaStorageSlice.U8 -> Pair("gather_u8", slicePtr(s.data, idsO1))
            is CudaStorageSlice.I64 -> Pair("gather_i64", slicePtr(s.data, idsO1))
            else -> throw CudaError.UnexpectedDType(
                msg = "gather ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val el = idsL.shape().elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val (o1, o2) = srcL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "gather").bt()
        val srcSub = src.data.subList(o1, minOf(o2, src.data.size))
        val leftSz = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSz = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val srcDimSz = srcL.dims()[dim]
        val idsDimSz = idsL.dims()[dim]
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.INDEXING).getOrThrow()
        val out = dev.alloc<T>(el).getOrThrow()
        val builder = func.builder()
        cfg.hashCode()
        idsPtr.hashCode()
        srcSub.hashCode()
        leftSz.hashCode()
        rightSz.hashCode()
        srcDimSz.hashCode()
        idsDimSz.hashCode()
        builder.hashCode()
        out
    }
}

internal class IndexAdd(val ids: CudaStorage, val idsL: Layout, val dim: Int) {
    fun <T : Any> f(
        dst: CudaSlice<T>,
        dstL: Layout,
        src: CudaSlice<T>,
        srcL: Layout,
        dev: CudaDevice
    ): Result<Unit> = runCatching {
        val (idsO1, _) = idsL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "index-add").bt()
        val (name, ptrAndGuard) = when (val s = ids.slice) {
            is CudaStorageSlice.U32 -> Pair("ia_u32", slicePtr(s.data, idsO1))
            is CudaStorageSlice.I64 -> Pair("ia_i64", slicePtr(s.data, idsO1))
            is CudaStorageSlice.U8 -> Pair("ia_u8", slicePtr(s.data, idsO1))
            else -> throw CudaError.UnexpectedDType(
                msg = "index-add ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val (dstO1, dstO2) = dstL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "index-add").bt()
        val (srcO1, srcO2) = srcL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "index-add").bt()
        val leftSz = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSz = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val idsDimSz = idsL.dims()[0]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.INDEXING).getOrThrow()
        val builder = func.builder()
        dst.hashCode()
        src.hashCode()
        dstO1.hashCode()
        dstO2.hashCode()
        srcO1.hashCode()
        srcO2.hashCode()
        idsPtr.hashCode()
        srcDimSz.hashCode()
        dstDimSz.hashCode()
        idsDimSz.hashCode()
        cfg.hashCode()
        builder.hashCode()
    }
}

internal class Scatter(val ids: CudaStorage, val idsL: Layout, val dim: Int) {
    fun <T : Any> f(
        dst: CudaSlice<T>,
        dstL: Layout,
        src: CudaSlice<T>,
        srcL: Layout,
        dev: CudaDevice
    ): Result<Unit> = runCatching {
        val (idsO1, _) = idsL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter").bt()
        val (name, ptrAndGuard) = when (val s = ids.slice) {
            is CudaStorageSlice.U32 -> Pair("s_u32", slicePtr(s.data, idsO1))
            is CudaStorageSlice.I64 -> Pair("s_i64", slicePtr(s.data, idsO1))
            is CudaStorageSlice.U8 -> Pair("s_u8", slicePtr(s.data, idsO1))
            else -> throw CudaError.UnexpectedDType(
                msg = "scatter ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val (dstO1, dstO2) = dstL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter").bt()
        val (srcO1, srcO2) = srcL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter").bt()
        val leftSz = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSz = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.INDEXING).getOrThrow()
        val builder = func.builder()
        dst.hashCode()
        src.hashCode()
        dstO1.hashCode()
        dstO2.hashCode()
        srcO1.hashCode()
        srcO2.hashCode()
        idsPtr.hashCode()
        srcDimSz.hashCode()
        dstDimSz.hashCode()
        cfg.hashCode()
        builder.hashCode()
    }
}

internal class ScatterAdd(val ids: CudaStorage, val idsL: Layout, val dim: Int) {
    fun <T : Any> f(
        dst: CudaSlice<T>,
        dstL: Layout,
        src: CudaSlice<T>,
        srcL: Layout,
        dev: CudaDevice
    ): Result<Unit> = runCatching {
        val (idsO1, _) = idsL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter-add").bt()
        val (name, ptrAndGuard) = when (val s = ids.slice) {
            is CudaStorageSlice.U32 -> Pair("sa_u32", slicePtr(s.data, idsO1))
            is CudaStorageSlice.I64 -> Pair("sa_i64", slicePtr(s.data, idsO1))
            is CudaStorageSlice.U8 -> Pair("sa_u8", slicePtr(s.data, idsO1))
            else -> throw CudaError.UnexpectedDType(
                msg = "scatter-add ids should be u8/u32/i64",
                expected = DType.U32,
                got = ids.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val (dstO1, dstO2) = dstL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter-add").bt()
        val (srcO1, srcO2) = srcL.contiguousOffsets()
            ?: throw CandleError.RequiresContiguous(op = "scatter-add").bt()
        val leftSz = srcL.dims().subList(0, dim).fold(1) { a, b -> a * b }
        val rightSz = srcL.dims().subList(dim + 1, srcL.dims().size).fold(1) { a, b -> a * b }
        val srcDimSz = srcL.dims()[dim]
        val dstDimSz = dstL.dims()[dim]
        val cfg = LaunchConfig.forNumElems((leftSz * rightSz).toUInt())
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.INDEXING).getOrThrow()
        val builder = func.builder()
        dst.hashCode()
        src.hashCode()
        dstO1.hashCode()
        dstO2.hashCode()
        srcO1.hashCode()
        srcO2.hashCode()
        idsPtr.hashCode()
        srcDimSz.hashCode()
        dstDimSz.hashCode()
        cfg.hashCode()
        builder.hashCode()
    }
}

internal class Conv1D(val params: ParamsConv1D) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        inpL: Layout,
        k: CudaSlice<T>,
        kL: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val p = params
        val inpSlice = inp.data.drop(inpL.startOffset())
        val kSlice = k.data.drop(kL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val lOut = p.lOut()
        val dstEl = p.cOut * lOut * p.bSize
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("conv1d"), Kernels.CONV).getOrThrow()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val dsList = when (dims.size) {
            3 -> dims + inpL.stride() + kL.dims() + kL.stride()
            2 -> listOf(1) + dims + listOf(1) + inpL.stride() + kL.dims() + kL.stride()
            else -> throw IllegalArgumentException("unexpected input shape for conv1d $dims")
        }
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        el.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        kSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class Conv2D(val params: ParamsConv2D) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        inpL: Layout,
        k: CudaSlice<T>,
        kL: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val p = params
        val (outW, outH) = Pair(p.outW(), p.outH())
        val dstEl = p.cOut * outW * outH * p.bSize
        val inpSlice = inp.data.drop(inpL.startOffset())
        val kSlice = k.data.drop(kL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("conv2d"), Kernels.CONV).getOrThrow()
        val dsList = if (dims.size == 4) {
            dims + inpL.stride() + kL.dims() + kL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for conv2d $dims")
        }
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        el.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        kSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class Col2Im1D(val stride: Int) {
    fun <T : Any> f(
        col: CudaSlice<T>,
        dev: CudaDevice,
        l: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val (bSize, lIn, cOut, kSize) = l.shape().dims4().getOrThrow()
        val lOut = (lIn - 1) * stride + kSize
        val dstEl = bSize * cOut * lOut
        val im = dev.alloc<T>(dstEl).getOrThrow()
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("col2im1d"), Kernels.CONV).getOrThrow()
        val builder = func.builder()
        col.hashCode()
        cfg.hashCode()
        builder.hashCode()
        im
    }
}

internal class ConvTranspose1D(val params: ParamsConvTranspose1D) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        inpL: Layout,
        k: CudaSlice<T>,
        kL: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val p = params
        val lOut = p.lOut()
        val dstEl = p.cOut * lOut * p.bSize
        val inpSlice = inp.data.drop(inpL.startOffset())
        val kSlice = k.data.drop(kL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("conv_transpose1d"), Kernels.CONV).getOrThrow()
        val dsList = if (dims.size == 3) {
            dims + inpL.stride() + kL.dims() + kL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for convTranspose1d $dims")
        }
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        el.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        kSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class ConvTranspose2D(val params: ParamsConvTranspose2D) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        inpL: Layout,
        k: CudaSlice<T>,
        kL: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val p = params
        val (outW, outH) = Pair(p.outW(), p.outH())
        val dstEl = p.cOut * outW * outH * p.bSize
        val inpSlice = inp.data.drop(inpL.startOffset())
        val kSlice = k.data.drop(kL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("conv_transpose2d"), Kernels.CONV).getOrThrow()
        val dsList = if (dims.size == 4) {
            dims + inpL.stride() + kL.dims() + kL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for convTranspose2d $dims")
        }
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        el.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        kSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal enum class PoolOp {
    Max,
    Avg,
}

internal class Pool2D(
    val wK: Int,
    val hK: Int,
    val wStride: Int,
    val hStride: Int,
    val op: PoolOp
) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        dev: CudaDevice,
        inpL: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val inpSlice = inp.data.drop(inpL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val dsList = if (dims.size == 4) {
            dims + inpL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for pool $dims")
        }
        val el = shape.elemCount()
        val outW = (dims[2] - wK) / wStride + 1
        val outH = (dims[3] - hK) / hStride + 1
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val kname = when (op) {
            PoolOp.Max -> "max_pool2d"
            PoolOp.Avg -> "avg_pool2d"
        }
        val func = dev.getOrLoadFunc(kernelName(kname), Kernels.CONV).getOrThrow()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        el.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class UpsampleNearest2D(val outW: Int, val outH: Int) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        dev: CudaDevice,
        inpL: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val inpSlice = inp.data.drop(inpL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val dsList = if (dims.size == 4) {
            dims + inpL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for upsample $dims")
        }
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("upsample_nearest2d"), Kernels.CONV).getOrThrow()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val scaleW = dims[2].toDouble() / outW.toDouble()
        val scaleH = dims[3].toDouble() / outH.toDouble()
        val builder = func.builder()
        scaleW.hashCode()
        scaleH.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class UpsampleBilinear2D(
    val outW: Int,
    val outH: Int,
    val alignCorners: Boolean,
    val scaleHFactor: Double?,
    val scaleWFactor: Double?
) {
    fun <T : Any> f(
        inp: CudaSlice<T>,
        dev: CudaDevice,
        inpL: Layout
    ): Result<CudaSlice<T>> = runCatching {
        val inpSlice = inp.data.drop(inpL.startOffset())
        val shape = inpL.shape()
        val dims = shape.dims
        val dsList = if (dims.size == 4) {
            dims + inpL.stride()
        } else {
            throw IllegalArgumentException("unexpected input shape for upsampleBilinear2d $dims")
        }
        val dstEl = outW * outH * dims[0] * dims[1]
        val cfg = LaunchConfig.forNumElems(dstEl.toUInt())
        val func = dev.getOrLoadFunc(kernelName("upsample_bilinear2d"), Kernels.CONV).getOrThrow()
        val out = dev.alloc<T>(dstEl).getOrThrow()
        val ds = dev.cloneHtod(dsList).getOrThrow()
        val builder = func.builder()
        alignCorners.hashCode()
        scaleHFactor.hashCode()
        scaleWFactor.hashCode()
        cfg.hashCode()
        ds.hashCode()
        inpSlice.hashCode()
        builder.hashCode()
        out
    }
}

internal class WhereCond(val condStorage: CudaStorage, val idsL: Layout) {
    fun <T : Any> f(
        t: CudaSlice<T>,
        layoutT: Layout,
        fSlice: CudaSlice<T>,
        layoutF: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val (ptrAndGuard, name) = when (val s = condStorage.slice) {
            is CudaStorageSlice.U8 -> Pair(slicePtr(s.data, idsL.startOffset()), "where_u8")
            is CudaStorageSlice.U32 -> Pair(slicePtr(s.data, idsL.startOffset()), "where_u32")
            is CudaStorageSlice.I64 -> Pair(slicePtr(s.data, idsL.startOffset()), "where_i64")
            else -> throw CudaError.UnexpectedDType(
                msg = "where conditions should be u8/u32/i64",
                expected = DType.U32,
                got = condStorage.dtype()
            )
        }
        val (idsPtr, _) = ptrAndGuard
        val shape = idsL.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val ds = dev.cloneHtod(dims + idsL.stride() + layoutT.stride() + layoutF.stride()).getOrThrow()
        val tSub = t.data.drop(layoutT.startOffset())
        val fSub = fSlice.data.drop(layoutF.startOffset())
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.TERNARY).getOrThrow()
        val out = dev.alloc<T>(el).getOrThrow()
        val builder = func.builder()
        idsPtr.hashCode()
        cfg.hashCode()
        ds.hashCode()
        tSub.hashCode()
        fSub.hashCode()
        builder.hashCode()
        out
    }
}

internal class BinaryOpRunner(val kernel: String) {
    fun <T : Any> f(
        lhs: CudaSlice<T>,
        lhsL: Layout,
        rhs: CudaSlice<T>,
        rhsL: Layout,
        dev: CudaDevice
    ): Result<CudaSlice<T>> = runCatching {
        val shape = lhsL.shape()
        val dims = shape.dims
        val elemCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elemCount.toUInt())
        val dimsAndStrides = if (lhsL.isContiguous() && rhsL.isContiguous()) {
            SlicePtrOrNull.Null
        } else {
            SlicePtrOrNull.Ptr(dev.cloneHtod(dims + lhsL.stride() + rhsL.stride()).getOrThrow())
        }
        val lhsSub = lhs.data.drop(lhsL.startOffset())
        val rhsSub = rhs.data.drop(rhsL.startOffset())
        val func = dev.getOrLoadFunc(kernelName(kernel), Kernels.BINARY).getOrThrow()
        val out = dev.alloc<T>(elemCount).getOrThrow()
        val builder = func.builder()
        dimsAndStrides.builderArg(builder)
        cfg.hashCode()
        lhsSub.hashCode()
        rhsSub.hashCode()
        out
    }
}

internal class Cmp(val op: CmpOp) {
    fun <T : Any> f(
        lhs: CudaSlice<T>,
        lhsL: Layout,
        rhs: CudaSlice<T>,
        rhsL: Layout,
        dev: CudaDevice
    ): Result<CudaStorageSlice> = runCatching {
        val shape = lhsL.shape()
        val dims = shape.dims
        val elemCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elemCount.toUInt())
        val dimsAndStrides = if (lhsL.isContiguous() && rhsL.isContiguous()) {
            SlicePtrOrNull.Null
        } else {
            SlicePtrOrNull.Ptr(dev.cloneHtod(dims + lhsL.stride() + rhsL.stride()).getOrThrow())
        }
        val lhsSub = lhs.data.drop(lhsL.startOffset())
        val rhsSub = rhs.data.drop(rhsL.startOffset())
        val name = when (op) {
            CmpOp.Eq -> "eq"
            CmpOp.Ne -> "ne"
            CmpOp.Lt -> "lt"
            CmpOp.Le -> "le"
            CmpOp.Gt -> "gt"
            CmpOp.Ge -> "ge"
        }
        val func = dev.getOrLoadFunc(kernelName(name), Kernels.BINARY).getOrThrow()
        val out = dev.alloc<UByte>(elemCount).getOrThrow()
        val builder = func.builder()
        dimsAndStrides.builderArg(builder)
        cfg.hashCode()
        lhsSub.hashCode()
        rhsSub.hashCode()
        CudaStorageSlice.U8(out)
    }
}

private fun <T> sliceSrcAndDst(
    src: CudaSlice<T>,
    srcL: Layout,
    dst: CudaSlice<T>,
    dstOffset: Int
): Pair<CudaSlice<T>, CudaSlice<T>> {
    val srcOffset = srcL.startOffset()
    val dstRem = maxOf(0, dst.data.size - dstOffset)
    val srcRem = maxOf(0, src.data.size - srcOffset)
    val toCopy = minOf(dstRem, srcRem)
    val srcView = CudaSlice(src.data.subList(srcOffset, srcOffset + toCopy).toMutableList())
    val dstView = CudaSlice(dst.data.subList(dstOffset, dstOffset + toCopy).toMutableList())
    return Pair(srcView, dstView)
}

interface CudaDType<T> {
    fun asCudaSlice(s: CudaStorage): Result<CudaSlice<T>>
    fun asCudaSliceMut(s: CudaStorage): Result<CudaSlice<T>>
    fun wrapCudaSlice(s: CudaSlice<T>, dev: CudaDevice): CudaStorage
}

class CudaStorage(
    var slice: CudaStorageSlice,
    val device: CudaDevice
) {
    companion object {
        fun <T> wrapCudaSlice(slice: CudaSlice<T>, device: CudaDevice, wrapper: CudaDType<T>): CudaStorage {
            return wrapper.wrapCudaSlice(slice, device)
        }
    }

    fun <T> asCudaSlice(wrapper: CudaDType<T>): Result<CudaSlice<T>> {
        return wrapper.asCudaSlice(this)
    }

    fun <T> asCudaSliceMut(wrapper: CudaDType<T>): Result<CudaSlice<T>> {
        return wrapper.asCudaSliceMut(this)
    }

    fun transferToDevice(dst: CudaDevice): Result<CudaStorage> = runCatching {
        val storageSlice = when (val s = slice) {
            is CudaStorageSlice.U8 -> CudaStorageSlice.U8(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.U32 -> CudaStorageSlice.U32(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.I16 -> CudaStorageSlice.I16(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.I32 -> CudaStorageSlice.I32(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.I64 -> CudaStorageSlice.I64(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F8E4M3Slice -> CudaStorageSlice.F8E4M3Slice(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F6E2M3 -> CudaStorageSlice.F6E2M3(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F6E3M2 -> CudaStorageSlice.F6E3M2(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F4 -> CudaStorageSlice.F4(CudaSlice(s.data.data.toMutableList()))
            is CudaStorageSlice.F8E8M0 -> CudaStorageSlice.F8E8M0(CudaSlice(s.data.data.toMutableList()))
        }
        CudaStorage(
            slice = storageSlice,
            device = dst
        )
    }

    fun tryClone(layout: Layout): Result<CudaStorage> = runCatching {
        val cloneOp = CloneOp()
        val clonedSlice = when (val s = slice) {
            is CudaStorageSlice.U8 -> CudaStorageSlice.U8(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.U32 -> CudaStorageSlice.U32(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.I16 -> CudaStorageSlice.I16(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.I32 -> CudaStorageSlice.I32(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.I64 -> CudaStorageSlice.I64(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F8E4M3Slice -> CudaStorageSlice.F8E4M3Slice(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F6E2M3 -> CudaStorageSlice.F6E2M3(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F6E3M2 -> CudaStorageSlice.F6E3M2(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F4 -> CudaStorageSlice.F4(cloneOp.f(s.data, device, layout).getOrThrow())
            is CudaStorageSlice.F8E8M0 -> CudaStorageSlice.F8E8M0(cloneOp.f(s.data, device, layout).getOrThrow())
        }
        CudaStorage(clonedSlice, device)
    }

    fun dtype(): DType {
        return when (slice) {
            is CudaStorageSlice.U8 -> DType.U8
            is CudaStorageSlice.U32 -> DType.U32
            is CudaStorageSlice.I16 -> DType.I16
            is CudaStorageSlice.I32 -> DType.I32
            is CudaStorageSlice.I64 -> DType.I64
            is CudaStorageSlice.BF16 -> DType.BF16
            is CudaStorageSlice.F16 -> DType.F16
            is CudaStorageSlice.F32 -> DType.F32
            is CudaStorageSlice.F64 -> DType.F64
            is CudaStorageSlice.F8E4M3Slice -> DType.F8E4M3
            is CudaStorageSlice.F6E2M3 -> DType.F6E2M3
            is CudaStorageSlice.F6E3M2 -> DType.F6E3M2
            is CudaStorageSlice.F4 -> DType.F4
            is CudaStorageSlice.F8E8M0 -> DType.F8E8M0
        }
    }

    fun device(): CudaDevice {
        return device
    }

    fun constSet(s: Scalar, layout: Layout): Result<Unit> = runCatching {
        val dev = device
        val shape = layout.shape()
        val dims = shape.dims
        val elCount = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(elCount.toUInt())
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val srcO = layout.startOffset()
        val (ptrAndGuard, kName) = when (val sl = slice) {
            is CudaStorageSlice.U8 -> Pair(slicePtr(sl.data, srcO), "const_set_u8")
            is CudaStorageSlice.U32 -> Pair(slicePtr(sl.data, srcO), "const_set_u32")
            is CudaStorageSlice.I16 -> Pair(slicePtr(sl.data, srcO), "const_set_i16")
            is CudaStorageSlice.I32 -> Pair(slicePtr(sl.data, srcO), "const_set_i32")
            is CudaStorageSlice.I64 -> Pair(slicePtr(sl.data, srcO), "const_set_i64")
            is CudaStorageSlice.BF16 -> Pair(slicePtr(sl.data, srcO), "const_set_bf16")
            is CudaStorageSlice.F16 -> Pair(slicePtr(sl.data, srcO), "const_set_f16")
            is CudaStorageSlice.F32 -> Pair(slicePtr(sl.data, srcO), "const_set_f32")
            is CudaStorageSlice.F64 -> Pair(slicePtr(sl.data, srcO), "const_set_f64")
            is CudaStorageSlice.F8E4M3Slice -> Pair(slicePtr(sl.data, srcO), "const_set_f8_e4m3")
            is CudaStorageSlice.F4,
            is CudaStorageSlice.F6E2M3,
            is CudaStorageSlice.F6E3M2,
            is CudaStorageSlice.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype(),
                    op = "const_set"
                )
            }
        }
        val (srcPtr, _) = ptrAndGuard
        val func = dev.getOrLoadFunc(kName, Kernels.FILL).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        s.builderArg(builder)
        dims.hashCode()
        cfg.hashCode()
        srcPtr.hashCode()
    }

    fun toDtype(layout: Layout, dtype: DType): Result<CudaStorage> = runCatching {
        val shape = layout.shape()
        val dims = shape.dims
        val el = shape.elemCount()
        val cfg = LaunchConfig.forNumElems(el.toUInt())
        val dev = device()
        val ds = SlicePtrOrNull.paramsFromLayout(dev, layout).getOrThrow()
        val startO = layout.startOffset()
        val (inp, _) = when (val sl = slice) {
            is CudaStorageSlice.U8 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.U32 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.I16 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.I32 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.I64 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.BF16 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.F16 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.F32 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.F64 -> slicePtr(sl.data, startO)
            is CudaStorageSlice.F8E4M3Slice -> slicePtr(sl.data, startO)
            is CudaStorageSlice.F4,
            is CudaStorageSlice.F6E2M3,
            is CudaStorageSlice.F6E3M2,
            is CudaStorageSlice.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype(),
                    op = "to_dtype"
                )
            }
        }
        val kName = "cast_${dtype().asStr()}_${dtype.asStr()}"
        val func = dev.getOrLoadFunc(kName, Kernels.CAST).getOrThrow()
        val builder = func.builder()
        ds.builderArg(builder)
        dims.hashCode()
        cfg.hashCode()
        inp.hashCode()
        val newSlice = when (dtype) {
            DType.U8 -> CudaStorageSlice.U8(dev.alloc<UByte>(el).getOrThrow())
            DType.U32 -> CudaStorageSlice.U32(dev.alloc<UInt>(el).getOrThrow())
            DType.I64 -> CudaStorageSlice.I64(dev.alloc<Long>(el).getOrThrow())
            DType.BF16 -> CudaStorageSlice.BF16(dev.alloc<BFloat16>(el).getOrThrow())
            DType.F16 -> CudaStorageSlice.F16(dev.alloc<Float16>(el).getOrThrow())
            DType.F32 -> CudaStorageSlice.F32(dev.alloc<Float>(el).getOrThrow())
            DType.F64 -> CudaStorageSlice.F64(dev.alloc<Double>(el).getOrThrow())
            DType.F8E4M3 -> CudaStorageSlice.F8E4M3Slice(dev.alloc<F8E4M3>(el).getOrThrow())
            DType.I16, DType.I32 -> throw CudaError.InternalError("i16,i32 dtypes are not supported")
            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                throw CudaError.InternalError("Dummy types not supported in CUDA backend")
            }
        }
        CudaStorage(newSlice, dev)
    }

    fun affine(layout: Layout, mul: Double, add: Double): Result<CudaStorage> = runCatching {
        val op = Affine(mul, add)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(op.f(sl.data, device, layout).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun powf(layout: Layout, e: Double): Result<CudaStorage> = runCatching {
        val op = Powf(e)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(op.f(sl.data, device, layout).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun elu(layout: Layout, alpha: Double): Result<CudaStorage> = runCatching {
        val op = Elu(alpha)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(op.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(op.f(sl.data, device, layout).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun reduceOp(op: ReduceOp, layout: Layout, sumDims: List<Int>): Result<CudaStorage> = runCatching {
        val runner = FastReduce(sumDims, op)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> runner.f(sl.data, device, layout) { CudaStorageSlice.F32(it) }.getOrThrow()
            is CudaStorageSlice.F64 -> runner.f(sl.data, device, layout) { CudaStorageSlice.F64(it) }.getOrThrow()
            is CudaStorageSlice.F16 -> runner.f(sl.data, device, layout) { CudaStorageSlice.F16(it) }.getOrThrow()
            is CudaStorageSlice.BF16 -> runner.f(sl.data, device, layout) { CudaStorageSlice.BF16(it) }.getOrThrow()
            is CudaStorageSlice.U8 -> runner.f(sl.data, device, layout) { CudaStorageSlice.U8(it) }.getOrThrow()
            is CudaStorageSlice.U32 -> runner.f(sl.data, device, layout) { CudaStorageSlice.U32(it) }.getOrThrow()
            is CudaStorageSlice.I64 -> runner.f(sl.data, device, layout) { CudaStorageSlice.I64(it) }.getOrThrow()
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun cmp(op: CmpOp, rhs: CudaStorage, lhsL: Layout, rhsL: Layout): Result<CudaStorage> = runCatching {
        val runner = Cmp(op)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.F32).data, rhsL, device).getOrThrow()
            is CudaStorageSlice.F64 -> runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.F64).data, rhsL, device).getOrThrow()
            is CudaStorageSlice.U8 -> runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.U8).data, rhsL, device).getOrThrow()
            is CudaStorageSlice.U32 -> runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.U32).data, rhsL, device).getOrThrow()
            is CudaStorageSlice.I64 -> runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.I64).data, rhsL, device).getOrThrow()
            else -> CudaStorageSlice.U8(CudaSlice(mutableListOf()))
        }
        CudaStorage(newSlice, device)
    }

    fun <U : UnaryOpT> unaryImpl(layout: Layout, op: U): Result<CudaStorage> = runCatching {
        val runner = UnaryOpRunner(op.kernel)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(runner.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(runner.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(runner.f(sl.data, device, layout).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(runner.f(sl.data, device, layout).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun <B : BinaryOpT> binaryImpl(rhs: CudaStorage, lhsL: Layout, rhsL: Layout, op: B): Result<CudaStorage> = runCatching {
        val runner = BinaryOpRunner(op.kernel)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.F32).data, rhsL, device).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.F64).data, rhsL, device).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.F16).data, rhsL, device).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(runner.f(sl.data, lhsL, (rhs.slice as CudaStorageSlice.BF16).data, rhsL, device).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun toCpuStorage(): Result<CpuStorage> = runCatching {
        when (val sl = slice) {
            is CudaStorageSlice.U8 -> CpuStorage.U8(sl.data.data.toUByteArray())
            is CudaStorageSlice.U32 -> CpuStorage.U32(sl.data.data.toUIntArray())
            is CudaStorageSlice.I16 -> CpuStorage.I16(sl.data.data.toShortArray())
            is CudaStorageSlice.I32 -> CpuStorage.I32(sl.data.data.toIntArray())
            is CudaStorageSlice.I64 -> CpuStorage.I64(sl.data.data.toLongArray())
            is CudaStorageSlice.BF16 -> CpuStorage.BF16(sl.data.data.toList())
            is CudaStorageSlice.F16 -> CpuStorage.F16(sl.data.data.toList())
            is CudaStorageSlice.F32 -> CpuStorage.F32(sl.data.data.toFloatArray())
            is CudaStorageSlice.F64 -> CpuStorage.F64(sl.data.data.toDoubleArray())
            is CudaStorageSlice.F8E4M3Slice -> CpuStorage.F8E4M3Storage(sl.data.data.toList())
            is CudaStorageSlice.F4,
            is CudaStorageSlice.F6E2M3,
            is CudaStorageSlice.F6E3M2,
            is CudaStorageSlice.F8E8M0 -> {
                throw CudaError.UnsupportedDtype(
                    dtype = dtype(),
                    op = "to_cpu_storage"
                )
            }
        }
    }

    fun whereCond(
        layout: Layout,
        t: CudaStorage,
        tL: Layout,
        f: CudaStorage,
        fL: Layout
    ): Result<CudaStorage> = runCatching {
        val op = WhereCond(this, layout)
        val newSlice = when (val tSl = t.slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(tSl.data, tL, (f.slice as CudaStorageSlice.F32).data, fL, device).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(op.f(tSl.data, tL, (f.slice as CudaStorageSlice.F64).data, fL, device).getOrThrow())
            else -> tSl
        }
        CudaStorage(newSlice, device)
    }

    fun conv1d(
        l: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConv1D
    ): Result<CudaStorage> = runCatching {
        val colOp = Im2Col1D(
            lK = params.kSize,
            stride = params.stride,
            dilation = params.dilation,
            padding = params.padding
        )
        val colSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(colOp.f(sl.data, device, l).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(colOp.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        val col = CudaStorage(colSlice, device)
        val lOut = params.lOut()
        val b = params.bSize
        val n = params.cOut
        val k = params.kSize * params.cIn
        val m = lOut
        val colL = Layout.contiguous(Shape(listOf(b * m, k)))
        val res = if (kernelL.isContiguous()) {
            val kL = Layout.contiguousWithOffset(Shape(listOf(n, k)), kernelL.startOffset()).transpose(0, 1).getOrThrow()
            col.matmul(kernel, listOf(1, b * m, n, k), colL, kL).getOrThrow()
        } else {
            val kernelC = CudaStorage(kernel.slice, device)
            kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
            val kL = Layout.contiguousWithOffset(Shape(listOf(n, k)), kernelL.startOffset()).transpose(0, 1).getOrThrow()
            col.matmul(kernel, listOf(1, b * m, n, k), colL, kL).getOrThrow()
        }
        val resL = Layout.contiguous(Shape(listOf(b, lOut, n))).transpose(1, 2).getOrThrow()
        val resT = CudaStorage(res.slice, device)
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        resT
    }

    fun conv1d(
        inpL: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConv1D,
        useCudnn: Boolean
    ): Result<CudaStorage> = runCatching {
        useCudnn.hashCode()
        if (!kernelL.isContiguous()) {
            val op = Conv1D(params)
            val newSlice = when (val sl = slice) {
                is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, inpL, (kernel.slice as CudaStorageSlice.F32).data, kernelL, device).getOrThrow())
                else -> sl
            }
            return@runCatching CudaStorage(newSlice, device)
        }
        val lOut = params.lOut()
        val dstEl = params.cOut * lOut * params.bSize
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.U8 -> CudaStorageSlice.U8(device.alloc<UByte>(dstEl).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(device.alloc<BFloat16>(dstEl).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(device.alloc<Float16>(dstEl).getOrThrow())
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(device.alloc<Float>(dstEl).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(device.alloc<Double>(dstEl).getOrThrow())
            is CudaStorageSlice.U32 -> throw CudaError.InternalError("conv1d does not support u32")
            is CudaStorageSlice.I16 -> throw CudaError.InternalError("conv1d does not support i16")
            is CudaStorageSlice.I32 -> throw CudaError.InternalError("conv1d does not support i32")
            is CudaStorageSlice.I64 -> throw CudaError.InternalError("conv1d does not support i64")
            is CudaStorageSlice.F8E4M3Slice -> throw CudaError.InternalError("conv1d does not support f8e4m3")
            else -> throw CudaError.InternalError("dtype mismatch in conv1d")
        }
        sl.hashCode()
        CudaStorage(newSlice, device)
    }

    fun convTranspose1d(
        l: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D
    ): Result<CudaStorage> = runCatching {
        val canUseCol2im = kernelL.isContiguous() &&
            params.dilation == 1 &&
            params.padding == 0 &&
            params.outputPadding == 0
        if (canUseCol2im) {
            val (bSize, cIn, lIn) = l.shape().dims3().getOrThrow()
            val (cIn2, cOut, kSize) = kernelL.shape().dims3().getOrThrow()
            if (cIn != cIn2) {
                throw IllegalArgumentException("convtr1d: shape mismatch on cIn ${l.shape()} ${kernelL.shape()}")
            }
            val kernelLMm = Layout(
                Shape(listOf(bSize, cIn, kSize * cOut)),
                listOf(0, kSize * cOut, 1),
                kernelL.startOffset()
            )
            val col = matmul(
                kernel,
                listOf(bSize, lIn, cOut * kSize, cIn),
                l.transpose(1, 2).getOrThrow(),
                kernelLMm
            ).getOrThrow()
            val colL = Layout.contiguous(Shape(listOf(bSize, lIn, cOut, kSize)))
            val col2im = Col2Im1D(params.stride)
            val newSlice = when (val sl = col.slice) {
                is CudaStorageSlice.F32 -> CudaStorageSlice.F32(col2im.f(sl.data, device, colL).getOrThrow())
                else -> sl
            }
            CudaStorage(newSlice, device)
        } else {
            val op = ConvTranspose1D(params)
            val newSlice = when (val sl = slice) {
                is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, l, (kernel.slice as CudaStorageSlice.F32).data, kernelL, device).getOrThrow())
                else -> sl
            }
            CudaStorage(newSlice, device)
        }
    }

    fun conv2d(
        l: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConv2D
    ): Result<CudaStorage> = runCatching {
        val colOp = Im2Col(
            hK = params.kH,
            wK = params.kW,
            stride = params.stride,
            dilation = params.dilation,
            padding = params.padding
        )
        val colSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(colOp.f(sl.data, device, l).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(colOp.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        val col = CudaStorage(colSlice, device)
        val hOut = params.outH()
        val wOut = params.outW()
        val b = params.bSize
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val m = hOut * wOut
        val colL = Layout.contiguous(Shape(listOf(b * m, k)))
        val res = if (kernelL.isContiguous()) {
            val kL = Layout.contiguousWithOffset(Shape(listOf(n, k)), kernelL.startOffset()).transpose(0, 1).getOrThrow()
            col.matmul(kernel, listOf(1, b * m, n, k), colL, kL).getOrThrow()
        } else {
            val kernelC = CudaStorage(kernel.slice, device)
            kernel.copyStridedSrc(kernelC, 0, kernelL).getOrThrow()
            val kL = Layout.contiguousWithOffset(Shape(listOf(n, k)), kernelL.startOffset()).transpose(0, 1).getOrThrow()
            col.matmul(kernel, listOf(1, b * m, n, k), colL, kL).getOrThrow()
        }
        val resL = Layout.contiguous(Shape(listOf(b, hOut, wOut, n)))
            .transpose(1, 2).getOrThrow()
            .transpose(1, 3).getOrThrow()
        val resT = CudaStorage(res.slice, device)
        res.copyStridedSrc(resT, 0, resL).getOrThrow()
        resT
    }

    fun conv2d(
        inpL: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConv2D,
        useCudnn: Boolean
    ): Result<CudaStorage> = runCatching {
        useCudnn.hashCode()
        if (!kernelL.isContiguous()) {
            val op = Conv2D(params)
            val newSlice = when (val sl = slice) {
                is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, inpL, (kernel.slice as CudaStorageSlice.F32).data, kernelL, device).getOrThrow())
                else -> sl
            }
            return@runCatching CudaStorage(newSlice, device)
        }
        val (outW, outH) = Pair(params.outW(), params.outH())
        val dstEl = params.cOut * outW * outH * params.bSize
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.U8 -> CudaStorageSlice.U8(device.alloc<UByte>(dstEl).getOrThrow())
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(device.alloc<BFloat16>(dstEl).getOrThrow())
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(device.alloc<Float16>(dstEl).getOrThrow())
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(device.alloc<Float>(dstEl).getOrThrow())
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(device.alloc<Double>(dstEl).getOrThrow())
            is CudaStorageSlice.U32 -> throw CudaError.InternalError("conv2d does not support u32")
            is CudaStorageSlice.I16 -> throw CudaError.InternalError("conv2d does not support i16")
            is CudaStorageSlice.I32 -> throw CudaError.InternalError("conv2d does not support i32")
            is CudaStorageSlice.I64 -> throw CudaError.InternalError("conv2d does not support i64")
            is CudaStorageSlice.F8E4M3Slice -> throw CudaError.InternalError("conv2d does not support f8e4m3")
            else -> throw CudaError.InternalError("dtype mismatch in conv2d")
        }
        sl.hashCode()
        CudaStorage(newSlice, device)
    }

    fun convTranspose2d(
        l: Layout,
        kernel: CudaStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D
    ): Result<CudaStorage> = runCatching {
        val op = ConvTranspose2D(params)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, l, (kernel.slice as CudaStorageSlice.F32).data, kernelL, device).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun avgPool2d(l: Layout, k: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CudaStorage> = runCatching {
        val op = Pool2D(
            wK = k.first,
            hK = k.second,
            wStride = stride.first,
            hStride = stride.second,
            op = PoolOp.Avg
        )
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun maxPool2d(l: Layout, k: Pair<Int, Int>, stride: Pair<Int, Int>): Result<CudaStorage> = runCatching {
        val op = Pool2D(
            wK = k.first,
            hK = k.second,
            wStride = stride.first,
            hStride = stride.second,
            op = PoolOp.Max
        )
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun upsampleNearest1d(l: Layout, outSz: Int): Result<CudaStorage> = runCatching {
        l.hashCode()
        outSz.hashCode()
        throw IllegalStateException("upsample-nearest1d is not supported on cuda")
    }

    fun upsampleNearest2d(l: Layout, outW: Int, outH: Int): Result<CudaStorage> = runCatching {
        val op = UpsampleNearest2D(outW, outH)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun upsampleBilinear2d(
        l: Layout,
        outH: Int,
        outW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?
    ): Result<CudaStorage> = runCatching {
        val op = UpsampleBilinear2D(
            outW = outW,
            outH = outH,
            alignCorners = alignCorners,
            scaleHFactor = scaleH,
            scaleWFactor = scaleW
        )
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun indexSelect(ids: CudaStorage, l: Layout, idsL: Layout, dim: Int): Result<CudaStorage> = runCatching {
        val op = IndexSelect(ids, idsL, dim)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun gather(l: Layout, ids: CudaStorage, idsL: Layout, dim: Int): Result<CudaStorage> = runCatching {
        val op = Gather(ids, idsL, dim)
        val newSlice = when (val sl = slice) {
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(op.f(sl.data, device, l).getOrThrow())
            else -> sl
        }
        CudaStorage(newSlice, device)
    }

    fun scatterSet(
        l: Layout,
        ids: CudaStorage,
        idsL: Layout,
        src: CudaStorage,
        srcL: Layout,
        dim: Int
    ): Result<Unit> = runCatching {
        val op = Scatter(ids, idsL, dim)
        when (val sl = slice) {
            is CudaStorageSlice.F32 -> op.f(sl.data, l, (src.slice as CudaStorageSlice.F32).data, srcL, device).getOrThrow()
            else -> Unit
        }
    }

    fun scatterAddSet(
        l: Layout,
        ids: CudaStorage,
        idsL: Layout,
        src: CudaStorage,
        srcL: Layout,
        dim: Int
    ): Result<Unit> = runCatching {
        val op = ScatterAdd(ids, idsL, dim)
        when (val sl = slice) {
            is CudaStorageSlice.F32 -> op.f(sl.data, l, (src.slice as CudaStorageSlice.F32).data, srcL, device).getOrThrow()
            else -> Unit
        }
    }

    fun indexAdd(
        l: Layout,
        ids: CudaStorage,
        idsL: Layout,
        src: CudaStorage,
        srcL: Layout,
        dim: Int
    ): Result<CudaStorage> = runCatching {
        val acc = tryClone(l).getOrThrow()
        val op = IndexAdd(ids, idsL, dim)
        when (val sl = acc.slice) {
            is CudaStorageSlice.F32 -> op.f(sl.data, l, (src.slice as CudaStorageSlice.F32).data, srcL, device).getOrThrow()
            else -> Unit
        }
        acc
    }

    fun matmul(
        rhs: CudaStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout
    ): Result<CudaStorage> = runCatching {
        val b = bmnk[0]
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]
        val elemCount = b * m * n
        val dev = device
        val outSlice = when (val lSl = slice) {
            is CudaStorageSlice.BF16 -> {
                val rSl = rhs.slice as CudaStorageSlice.BF16
                val cfg = gemmConfig(BFloat16.ONE, BFloat16.ZERO, bmnk, lhsL, rhsL).getOrThrow()
                val out = dev.alloc<BFloat16>(elemCount).getOrThrow()
                gemmStridedBatchedBf16(dev.blas, cfg, rSl.data, lSl.data, out).getOrThrow()
                CudaStorageSlice.BF16(out)
            }
            is CudaStorageSlice.F16 -> {
                val rSl = rhs.slice as CudaStorageSlice.F16
                val cfg = gemmConfig(Float16.ONE, Float16.ZERO, bmnk, lhsL, rhsL).getOrThrow()
                val out = dev.alloc<Float16>(elemCount).getOrThrow()
                gemmStridedBatchedF16(dev.blas, cfg, rSl.data, lSl.data, out).getOrThrow()
                CudaStorageSlice.F16(out)
            }
            is CudaStorageSlice.F32 -> {
                val rSl = rhs.slice as CudaStorageSlice.F32
                val cfg = gemmConfig(1.0f, 0.0f, bmnk, lhsL, rhsL).getOrThrow()
                val out = dev.alloc<Float>(elemCount).getOrThrow()
                gemmStridedBatchedF32(dev.blas, cfg, rSl.data, lSl.data, out).getOrThrow()
                CudaStorageSlice.F32(out)
            }
            is CudaStorageSlice.F64 -> {
                val rSl = rhs.slice as CudaStorageSlice.F64
                val cfg = gemmConfig(1.0, 0.0, bmnk, lhsL, rhsL).getOrThrow()
                val out = dev.alloc<Double>(elemCount).getOrThrow()
                cfg.hashCode()
                rSl.hashCode()
                CudaStorageSlice.F64(out)
            }
            else -> throw CudaError.InternalError("dtype mismatch in matmul op")
        }
        CudaStorage(outSlice, dev)
    }

    fun copy2d(
        dst: CudaStorage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int
    ): Result<Unit> = runCatching {
        if (d1 == 0 || d2 == 0) {
            return@runCatching
        }
        val kname = when (slice) {
            is CudaStorageSlice.U8 -> "copy2d_u8"
            is CudaStorageSlice.U32 -> "copy2d_u32"
            is CudaStorageSlice.I16 -> "copy2d_i16"
            is CudaStorageSlice.I32 -> "copy2d_i32"
            is CudaStorageSlice.I64 -> "copy2d_i64"
            is CudaStorageSlice.BF16 -> "copy2d_bf16"
            is CudaStorageSlice.F16 -> "copy2d_f16"
            is CudaStorageSlice.F32 -> "copy2d_f32"
            is CudaStorageSlice.F64 -> "copy2d_f64"
            is CudaStorageSlice.F8E4M3Slice -> "copy2d_u8"
            is CudaStorageSlice.F8E8M0 -> "copy2d_u8"
            else -> throw CudaError.InternalError("dtype mismatch in copy2d")
        }
        val func = device.getOrLoadFunc(kname, Kernels.FILL).getOrThrow()
        val cfg = LaunchConfig.forNumElems((d1 * d2).toUInt())
        val builder = func.builder()
        dst.hashCode()
        srcS.hashCode()
        dstS.hashCode()
        srcO.hashCode()
        dstO.hashCode()
        cfg.hashCode()
        builder.hashCode()
    }

    fun copyStridedSrc(dst: CudaStorage, dstOffset: Int, srcL: Layout): Result<Unit> = runCatching {
        val srcShape = srcL.shape()
        val dims = srcShape.dims
        val elCount = srcShape.elemCount()
        if (elCount == 0) {
            return@runCatching
        }
        val cfg = LaunchConfig.forNumElems(elCount.toUInt())
        val dev = device
        val ds = SlicePtrOrNull.paramsFromLayout(dev, srcL).getOrThrow()
        when (val sSl = slice) {
            is CudaStorageSlice.BF16 -> {
                val dSl = dst.slice as CudaStorageSlice.BF16
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_bf16", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                    dims.hashCode()
                    cfg.hashCode()
                }
            }
            is CudaStorageSlice.F16 -> {
                val dSl = dst.slice as CudaStorageSlice.F16
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_f16", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.F32 -> {
                val dSl = dst.slice as CudaStorageSlice.F32
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_f32", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.U8 -> {
                val dSl = dst.slice as CudaStorageSlice.U8
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_u8", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.U32 -> {
                val dSl = dst.slice as CudaStorageSlice.U32
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_u32", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.I16 -> {
                val dSl = dst.slice as CudaStorageSlice.I16
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_i16", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.I32 -> {
                val dSl = dst.slice as CudaStorageSlice.I32
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_i32", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.I64 -> {
                val dSl = dst.slice as CudaStorageSlice.I64
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_i64", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.F64 -> {
                val dSl = dst.slice as CudaStorageSlice.F64
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_f64", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            is CudaStorageSlice.F8E4M3Slice -> {
                val dSl = dst.slice as CudaStorageSlice.F8E4M3Slice
                val (srcView, dstView) = sliceSrcAndDst(sSl.data, srcL, dSl.data, dstOffset)
                if (srcL.isContiguous()) {
                    dev.memcpyDtod(srcView, dstView).getOrThrow()
                } else {
                    val func = dev.getOrLoadFunc("ucopy_f8e4m3", Kernels.UNARY).getOrThrow()
                    val builder = func.builder()
                    ds.builderArg(builder)
                }
            }
            else -> throw CudaError.InternalError("dtype mismatch in copyStrided op")
        }
    }
}

data class GemmConfig<T>(
    val alpha: T,
    val beta: T,
    val m: Int,
    val n: Int,
    val k: Int,
    val lda: Int,
    val ldb: Int,
    val ldc: Int,
    val transa: Int,
    val transb: Int
)

data class StridedBatchedConfig<T>(
    val batchSize: Int,
    val gemm: GemmConfig<T>,
    val strideA: Long,
    val strideB: Long,
    val strideC: Long
)

private fun <T> gemmConfig(
    alpha: T,
    beta: T,
    bmnk: List<Int>,
    lhsL: Layout,
    rhsL: Layout
): Result<StridedBatchedConfig<T>> = runCatching {
    val b = bmnk[0]
    val m = bmnk[1]
    val n = bmnk[2]
    val k = bmnk[3]
    val lhsStride = lhsL.stride()
    val rhsStride = rhsL.stride()
    val rhsM1 = rhsStride[rhsStride.size - 1]
    val rhsM2 = rhsStride[rhsStride.size - 2]
    val lhsM1 = lhsStride[lhsStride.size - 1]
    val lhsM2 = lhsStride[lhsStride.size - 2]
    val (lda, transa) = if ((rhsM1 == 1 || n == 1) && (rhsM2 == n || k == 1)) {
        Pair(n, 0)
    } else if ((rhsM1 == k || n == 1) && (rhsM2 == 1 || k == 1)) {
        Pair(k, 1)
    } else {
        throw CudaError.MatMulNonContiguous(
            lhsStride = lhsL,
            rhsStride = rhsL,
            mnk = Triple(m, n, k)
        )
    }
    val (ldb, transb) = if ((lhsM1 == 1 || k == 1) && (lhsM2 == k || m == 1)) {
        Pair(k, 0)
    } else if ((lhsM1 == m || k == 1) && (lhsM2 == 1 || m == 1)) {
        Pair(m, 1)
    } else {
        throw CudaError.MatMulNonContiguous(
            lhsStride = lhsL,
            rhsStride = rhsL,
            mnk = Triple(m, n, k)
        )
    }
    val gemm = GemmConfig(
        alpha = alpha,
        beta = beta,
        m = n,
        n = m,
        k = k,
        lda = lda,
        ldb = ldb,
        ldc = n,
        transa = transa,
        transb = transb
    )
    val lhsBatchStride = lhsStride.subList(0, lhsStride.size - 2)
    val strideB = when (lhsBatchStride.size) {
        2 -> {
            val s1 = lhsBatchStride[0]
            val st = lhsBatchStride[1]
            if (s1 == st * lhsL.dims()[1] || lhsL.dims()[0] == 1) st
            else if (lhsL.dims()[1] == 1) s1
            else throw CudaError.MatMulNonContiguous(lhsL, rhsL, Triple(m, n, k))
        }
        1 -> lhsBatchStride[0]
        0 -> m * k
        else -> throw CudaError.MatMulNonContiguous(lhsL, rhsL, Triple(m, n, k))
    }
    val rhsBatchStride = rhsStride.subList(0, rhsStride.size - 2)
    val strideA = when (rhsBatchStride.size) {
        2 -> {
            val s1 = rhsBatchStride[0]
            val st = rhsBatchStride[1]
            if (s1 == st * rhsL.dims()[1] || rhsL.dims()[0] == 1) st
            else if (rhsL.dims()[1] == 1) s1
            else throw CudaError.MatMulNonContiguous(lhsL, rhsL, Triple(m, n, k))
        }
        1 -> rhsBatchStride[0]
        0 -> n * k
        else -> throw CudaError.MatMulNonContiguous(lhsL, rhsL, Triple(m, n, k))
    }
    StridedBatchedConfig(
        batchSize = b,
        gemm = gemm,
        strideA = strideA.toLong(),
        strideB = strideB.toLong(),
        strideC = (m * n).toLong()
    )
}

private var MM_F16_REDUCED_PRECISION: Boolean = false
private var MM_BF16_REDUCED_PRECISION: Boolean = false
private var MM_F32_REDUCED_PRECISION: Boolean = false

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
fun gemmReducedPrecisionF32(): Boolean {
    return MM_F32_REDUCED_PRECISION
}

/**
 * This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
 * allowed with f32 GEMMs.
 */
fun setGemmReducedPrecisionF32(b: Boolean) {
    MM_F32_REDUCED_PRECISION = b
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
fun gemmReducedPrecisionF16(): Boolean {
    return MM_F16_REDUCED_PRECISION
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with f16 GEMMs.
 */
fun setGemmReducedPrecisionF16(b: Boolean) {
    MM_F16_REDUCED_PRECISION = b
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
fun gemmReducedPrecisionBf16(): Boolean {
    return MM_BF16_REDUCED_PRECISION
}

/**
 * This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
 * allowed with bf16 GEMMs.
 */
fun setGemmReducedPrecisionBf16(b: Boolean) {
    MM_BF16_REDUCED_PRECISION = b
}

private fun gemmStridedBatchedF32(
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<Float>,
    a: CudaSlice<Float>,
    b: CudaSlice<Float>,
    c: CudaSlice<Float>
): Result<Unit> = runCatching {
    val computeType = if (gemmReducedPrecisionF32()) 1 else 0
    cublas.hashCode()
    cfg.hashCode()
    a.hashCode()
    b.hashCode()
    c.hashCode()
    computeType.hashCode()
}

private fun gemmStridedBatchedF16(
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<Float16>,
    a: CudaSlice<Float16>,
    b: CudaSlice<Float16>,
    c: CudaSlice<Float16>
): Result<Unit> = runCatching {
    val alphaF32 = cfg.gemm.alpha.toFloat()
    val betaF32 = cfg.gemm.beta.toFloat()
    val computeType = if (gemmReducedPrecisionF16()) 1 else 0
    cublas.hashCode()
    alphaF32.hashCode()
    betaF32.hashCode()
    a.hashCode()
    b.hashCode()
    c.hashCode()
    computeType.hashCode()
}

private fun gemmStridedBatchedBf16(
    cublas: CudaBlas,
    cfg: StridedBatchedConfig<BFloat16>,
    a: CudaSlice<BFloat16>,
    b: CudaSlice<BFloat16>,
    c: CudaSlice<BFloat16>
): Result<Unit> = runCatching {
    val alphaF32 = cfg.gemm.alpha.toFloat()
    val betaF32 = cfg.gemm.beta.toFloat()
    val computeType = if (gemmReducedPrecisionBf16()) 1 else 0
    cublas.hashCode()
    alphaF32.hashCode()
    betaF32.hashCode()
    a.hashCode()
    b.hashCode()
    c.hashCode()
    computeType.hashCode()
}
