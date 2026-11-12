// port-lint: source sort.rs
package io.github.kotlinmania.candle.sort

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.CudaSlice
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.DeviceRepr
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.LaunchConfig
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.candle.cudabackend.kernelName
import io.github.kotlinmania.candle.cudabackend.kernels
import io.github.kotlinmania.candle.customop.CustomOp1
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.dummydtype.F4
import io.github.kotlinmania.candle.dummydtype.F6e2m3
import io.github.kotlinmania.candle.dummydtype.F6e3m2
import io.github.kotlinmania.candle.dummydtype.F8e8m0
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.metalbackend.bufferO
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.rayon.prelude.*
import io.github.kotlinmania.candle.cudabackend.CudaStorageSlice as S

public open class ArgSort(
    public var asc: Boolean,
    public var lastDim: Int,
) : CustomOp1 {
    public open fun <T : WithDType> asort(vs: List<T>, layout: Layout): Result<List<UInt>> {
        val vs1 =
            when (val matchVal = layout.contiguousOffsets()) {
                null -> bail("input has to be contiguous")
                else -> {
                    val o1 = matchVal.first
                    val o2 = matchVal.second
                    vs[o1 until o2]
                }
            }
        var sortIndexes =
            run {
                val elCount = layout.shape().elemCount()
                val v = ArrayList<UInt>(elCount)
                v.setLen(elCount)
                v
            }
        if (this.asc) {
            (sortIndexes.parChunksExactMut(this.lastDim).zip(vs1.parChunksExact(this.lastDim))).forEach({ (indexes, vs2) ->
                indexes.iterMut().enumerate().forEach({ (i, v1) -> v1 = i.toUInt() })
                indexes.sortBy({ i, j ->
                    vs2[i.toInt()].partialCmp(vs2[j.toInt()]).unwrapOr(Ordering.Greater)
                })
            })
        } else {
            (sortIndexes.parChunksExactMut(this.lastDim).zip(vs.parChunksExact(this.lastDim))).forEach({ (indexes, vs2) ->
                indexes.iterMut().enumerate().forEach({ (i, v1) -> v1 = i.toUInt() })
                indexes.sortBy({ j, i ->
                    vs2[i.toInt()].partialCmp(vs2[j.toInt()]).unwrapOr(Ordering.Greater)
                })
            })
        }
        return Result.success(sortIndexes)
    }

    public open fun name(): String = "argsort"

    public open fun cpuFwd(storage: CpuStorage, layout: Layout): Result<Pair<CpuStorage, Shape>> {
        val sortIndexes =
            when (storage) {
                is CpuStorage.U8 -> {
                    val (vs) = storage
                    this.asort(vs, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.U32 -> {
                    val (vs1) = storage
                    this.asort(vs1, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.I16 -> {
                    val (vs2) = storage
                    this.asort(vs2, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.I32 -> {
                    val (vs3) = storage
                    this.asort(vs3, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.I64 -> {
                    val (vs4) = storage
                    this.asort(vs4, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.Bf16 -> {
                    val (vs5) = storage
                    this.asort(vs5, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.F16 -> {
                    val (vs6) = storage
                    this.asort(vs6, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.F32 -> {
                    val (vs7) = storage
                    this.asort(vs7, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.F64 -> {
                    val (vs8) = storage
                    this.asort(vs8, layout).getOrElse { return Result.failure(it) }
                }
                is CpuStorage.F8e4m3 -> {
                    val (vs9) = storage
                    this.asort(vs9, layout).getOrElse { return Result.failure(it) }
                }
                // Dummy types don't support sorting
                is F6e2m3 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(F6e2m3, "argsort").bt())
                }
                is F6e3m2 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(F6e3m2, "argsort").bt())
                }
                is F4 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(F4, "argsort").bt())
                }
                is F8e8m0 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(F8e8m0, "argsort").bt())
                }
            }
        val sortIndexes1 = CpuStorage.U32(sortIndexes)
        return Result.success(Pair(sortIndexes1, layout.shape().into()))
    }

    public open fun cudaFwd(storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> {
        val dev = storage.device()
        val slice = this.map(storage.slice, dev, layout).getOrElse { return Result.failure(it) }
        val dst = CudaStorage(slice = slice, device = dev.clone())
        return Result.success(Pair(dst, layout.shape().clone()))
    }

    public open fun metalFwd(storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> {
        val name = if (this.asc) {
            when (storage.dtype()) {
                DType.bf16 -> "asort_asc_bf16"
                DType.f16 -> "asort_asc_f16"
                DType.f32 -> "asort_asc_f32"
                DType.f64 -> "asort_asc_f64"
                DType.u8 -> "asort_asc_u8"
                DType.u32 -> "asort_asc_u32"
                DType.i16 -> "asort_asc_i16"
                DType.i32 -> "asort_asc_i32"
                DType.i64 -> "asort_asc_i64"
                DType.f8e4m3 -> bail("Metal device does not yet support F8E4M3.")
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(storage.dtype(), "argsort").bt())
                }
            }
        } else {
            when (storage.dtype()) {
                DType.bf16 -> "asort_desc_bf16"
                DType.f16 -> "asort_desc_f16"
                DType.f32 -> "asort_desc_f32"
                DType.f64 -> "asort_desc_f64"
                DType.u8 -> "asort_desc_u8"
                DType.u32 -> "asort_desc_u32"
                DType.i16 -> "asort_desc_i16"
                DType.i32 -> "asort_desc_i32"
                DType.i64 -> "asort_desc_i64"
                DType.f8e4m3 -> bail("Metal device does not yet support F8E4M3.")
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Result.failure(Error.UnsupportedDTypeForOp(storage.dtype(), "argsort").bt())
                }
            }
        }
        val device = storage.device()
        val kernels = device.kernels()
        val commandEncoder = device.commandEncoder().getOrElse { return Result.failure(it) }
        val el = layout.shape().elemCount()
        val ncols = this.lastDim
        val nrows = el / ncols
        val src = bufferO(storage.buffer(), layout, storage.dtype())
        val dst = device.newBufferBuilder().withSizeFor(el, DType.U32).withLabel("asort").build().getOrElse { return Result.failure(it) }
        var ncolsPad = 1
        while (ncolsPad < ncols) {
            ncolsPad *= 2
        }
        callArgSort(device.metalDevice(), commandEncoder, kernels, name, nrows, ncols, ncolsPad, src, dst).mapErr(Error::wrap).getOrElse { return Result.failure(it) }
        val dst1 = MetalStorage.new(dst, device.clone(), el, DType.U32)
        return Result.success(Pair(dst1, layout.shape().clone()))
    }
}

public fun <T> ArgSort.f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout, wrap: (CudaSlice<T>) -> S): Result<S> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits {
    val slice =
        when (val matchVal = layout.contiguousOffsets()) {
            null -> bail("input has to be contiguous")
            else -> {
                val o1 = matchVal.first
                val o2 = matchVal.second
                src.slice(o1 until o2)
            }
        }
    val elemCount = layout.shape().elemCount()
    val dst = dev.alloc<UInt>(elemCount).getOrElse { return Result.failure(it) }
    val func =
        if (this.asc) {
            dev.getOrLoadFunc(kernelName<T>("asort_asc"), kernels.SORT).getOrElse { return Result.failure(it) }
        } else {
            dev.getOrLoadFunc(kernelName<T>("asort_desc"), kernels.SORT).getOrElse { return Result.failure(it) }
        }
    val ncols = this.lastDim
    val nrows = elemCount / ncols
    val ncolsPad = nextPowerOf2(ncols)
    // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
    val blockDim = ncolsPad.min(1024)
    val cfg = LaunchConfig(gridDim = Triple(nrows.toUInt(), 1, 1), blockDim = Triple(blockDim.toUInt(), 1, 1), sharedMemBytes = (ncolsPad * sizeOf<UInt>()).toUInt())
    val stream = dev.cudaStream()
    val builder = stream.launchBuilder(func)
    val ncols1 = ncols
    val ncolsPad1 = ncolsPad
    builder.arg(slice).arg(dst).arg(ncols1).arg(ncolsPad1)
    builder.launch(cfg).w().getOrElse { return Result.failure(it) }
    return Result.success(S.u32(dst))
}

public fun nextPowerOf2(x: Int): Int {
    var n = 1
    while (n < x) {
        n *= 2
    }
    return n
}

/**
 * Returns the indices that sort the tensor along the last dimension.
 *
 * If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
 * descending order. The sort is unstable so there is no guarantees on the final order when it
 * comes to ties.
 */
public fun Tensor.argSortLastDim(asc: Boolean): Result<Tensor> {
    if (!this.isContiguous()) {
        return Result.failure(Error.RequiresContiguous(op = "arg_sort_last_dim"))
    }
    val lastDim1 =
        when (val matchVal = this.dims().last()) {
            null -> bail("empty last-dim in arg-sort")
            else -> {
                val lastDim = matchVal
                lastDim
            }
        }
    // No need for a backward pass for arg sort.
    return this.applyOp1NoBwd(ArgSort(asc = asc, lastDim = lastDim1))
}

/**
 * Sorts the tensor along the last dimension, returns the sorted tensor together with the
 * sorted indexes.
 *
 * If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
 * descending order. The sort is unstable so there is no guarantees on the final order when it
 * comes to ties.
 */
public fun Tensor.sortLastDim(asc: Boolean): Result<Pair<Tensor, Tensor>> {
    if (!this.isContiguous()) {
        return Result.failure(Error.RequiresContiguous(op = "sort_last_dim"))
    }
    val asort = this.argSortLastDim(asc).getOrElse { return Result.failure(it) }
    val sorted = this.gather(asort, D.Minus1).getOrElse { return Result.failure(it) }
    return Result.success(Pair(sorted, asort))
}
