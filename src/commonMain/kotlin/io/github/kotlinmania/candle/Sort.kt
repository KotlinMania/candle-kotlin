// port-lint: source sort.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.cuda.CudaStorage
import io.github.kotlinmania.candle.cuda.CudaStorageSlice
import io.github.kotlinmania.candle.cuda.device.CudaDevice
import io.github.kotlinmania.candle.cuda.device.CudaFunc
import io.github.kotlinmania.candle.cuda.device.CudaSlice
import io.github.kotlinmania.candle.cuda.device.CudaStream
import io.github.kotlinmania.candle.cuda.device.KernelModule
import io.github.kotlinmania.candle.cuda.device.LaunchArgs
import io.github.kotlinmania.candle.cuda.device.LaunchConfig
import io.github.kotlinmania.candle.metal.MetalStorage
import io.github.kotlinmania.candle.metal.bufferO

private fun <T> kotlin.Result<T>.unwrap(): T = getOrThrow()

private fun <T> CudaSlice<T>.slice(range: IntRange): CudaSlice<T> =
    CudaSlice(data.subList(range.first, (range.last + 1).coerceAtMost(data.size)).toMutableList())

private fun CudaStream.launchBuilder(func: CudaFunc): LaunchArgs = LaunchArgs(func.func)

private fun callArgSort(
    device: Any?,
    commandEncoder: Any?,
    kernels: Any?,
    name: String,
    nrows: Int,
    ncols: Int,
    ncolsPad: Int,
    src: Any?,
    dst: Any?,
): kotlin.Result<Unit> = kotlin.Result.success(Unit)

private object Ordering {
    const val Greater: Int = 1
}

private class ChunkView<T>(
    val list: MutableList<T>,
    val offset: Int,
    val len: Int,
) {
    val setAt: (Int, T) -> Unit = { i, v ->
        list[offset + i] = v
    }

    val iterMut: () -> List<Int> = { (0 until len).toList() }

    val sortBy: ((T, T) -> Int) -> Unit = { cmp ->
        val sub = MutableList(len) { list[offset + it] }
        sub.sortWith(Comparator(cmp))
        for (k in 0 until len) {
            list[offset + k] = sub[k]
        }
    }
}

private fun <T> MutableList<T>.parChunksExactMut(chunkSize: Int): List<ChunkView<T>> {
    val n = size / chunkSize
    return List(n) { c -> ChunkView(this, c * chunkSize, chunkSize) }
}

private fun <T> List<T>.parChunksExact(chunkSize: Int): List<List<T>> {
    val n = size / chunkSize
    return List(n) { c -> subList(c * chunkSize, (c + 1) * chunkSize) }
}

private fun <T : Comparable<T>> T.partialCmp(other: T): Int? = this.compareTo(other)

private val Int?.unwrapOr: (Int) -> Int
    get() = { fallback -> this ?: fallback }

private val MutableList<UInt>.setLen: (Int) -> Unit
    get() = { n ->
        while (size < n) add(0u)
    }

internal data class ArgSort(
    val asc: Boolean,
    val lastDim: Int,
) : CustomOp1 {
    internal fun <T : Comparable<T>> asort(
        vs: List<T>,
        layout: Layout,
        self: ArgSort = this,
    ): MutableList<UInt> {
        // Safety: indexes are set later in the parallelized section.
        val sortIndexes =
            run {
                val elCount = layout.shape().elemCount()
                val v = Vec.withCapacity<UInt>(elCount)
                v.setLen(elCount)
                v
            }
        if (self.asc) {
            sortIndexes
                .parChunksExactMut(self.lastDim)
                .zip(vs.parChunksExact(self.lastDim))
                .forEach { pair ->
                    val indexes = pair.first
                    val vsChunk = pair.second
                    indexes
                        .iterMut()
                        .enumerate()
                        .forEach { (i, _) -> indexes.setAt(i, i.toUInt()) }
                    indexes.sortBy { i, j ->
                        vsChunk[i.toInt()]
                            .partialCmp(vsChunk[j.toInt()])
                            .unwrapOr(Ordering.Greater)
                    }
                }
        } else {
            sortIndexes
                .parChunksExactMut(self.lastDim)
                .zip(vs.parChunksExact(self.lastDim))
                .forEach { pair ->
                    val indexes = pair.first
                    val vsChunk = pair.second
                    indexes
                        .iterMut()
                        .enumerate()
                        .forEach { (i, _) -> indexes.setAt(i, i.toUInt()) }
                    indexes.sortBy { j, i ->
                        vsChunk[i.toInt()]
                            .partialCmp(vsChunk[j.toInt()])
                            .unwrapOr(Ordering.Greater)
                    }
                }
        }
        return sortIndexes
    }

    internal fun <T, W : (CudaSlice<T>) -> CudaStorageSlice> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout,
        _wrap: W,
    ): CandleResult<CudaStorageSlice> {
        val self = this
        val offsets = layout.contiguousOffsets()
        val slice =
            when (offsets) {
                null -> bail("input has to be contiguous")
                else -> src.slice(offsets.first until offsets.second)
            }
        val elemCount = layout.shape().elemCount()
        val dst = dev.alloc<UInt>(elemCount).unwrap()
        val func =
            if (self.asc) {
                dev.getOrLoadFunc(kernelName("asort_asc"), kernels.SORT).unwrap()
            } else {
                dev.getOrLoadFunc(kernelName("asort_desc"), kernels.SORT).unwrap()
            }
        val ncols = self.lastDim
        val nrows = elemCount / ncols
        val ncolsPad = nextPowerOf2(ncols)
        // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
        val blockDim = minOf(ncolsPad, 1024)
        val cfg =
            LaunchConfig(
                gridDim = Triple(nrows.toUInt(), 1u, 1u),
                blockDim = Triple(blockDim.toUInt(), 1u, 1u),
                sharedMemBytes = (ncolsPad * 4).toUInt(),
            )
        val stream = dev.cudaStream()
        val builder = stream.launchBuilder(func)
        val ncolsI32 = ncols
        val ncolsPadI32 = ncolsPad
        builder
            .arg(slice)
            .arg(dst)
            .arg(ncolsI32)
            .arg(ncolsPadI32)
        builder.launch(cfg).unwrap()
        return CandleResult.Ok(CudaStorageSlice.U32(dst))
    }

    override fun name(): String = "argsort"

    override fun cpuFwd(
        storage: CpuStorage,
        layout: Layout,
    ): Pair<CpuStorage, Shape> {
        val self = this
        val sortIndexes =
            when (storage) {
                is CpuStorage.U8 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.U32 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.I16 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.I32 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.I64 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.BF16 -> {
                    self.asort(storage.data.map { BFloat16(it.toUShort()).toFloat() }, layout)
                }

                is CpuStorage.F16 -> {
                    self.asort(storage.data.map { Float16(it.toUShort()).toFloat() }, layout)
                }

                is CpuStorage.F32 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.F64 -> {
                    self.asort(storage.data.toList(), layout)
                }

                is CpuStorage.F8E4M3 -> {
                    self.asort(storage.data.map { F8E4M3(it.toUByte()).toFloat() }, layout)
                }

                // Dummy types do not support sorting
                is CpuStorage.F6E2M3 -> {
                    throw CandleError.UnsupportedDTypeForOp(DType.F6E2M3, "argsort").bt()
                }

                is CpuStorage.F6E3M2 -> {
                    throw CandleError.UnsupportedDTypeForOp(DType.F6E3M2, "argsort").bt()
                }

                is CpuStorage.F4 -> {
                    throw CandleError.UnsupportedDTypeForOp(DType.F4, "argsort").bt()
                }

                is CpuStorage.F8E8M0 -> {
                    throw CandleError.UnsupportedDTypeForOp(DType.F8E8M0, "argsort").bt()
                }
            }
        val sortIndexesStorage = CpuStorage.U32(sortIndexes.toUIntArray())
        return Pair(sortIndexesStorage, layout.shape().clone())
    }

    public fun cudaFwd(
        storage: CudaStorage,
        layout: Layout,
    ): Pair<CudaStorage, Shape> {
        val self = this
        val dev = storage.device
        val slice = self.map(storage.slice, dev, layout)
        val dst =
            CudaStorage(
                slice = slice,
                device = dev,
            )
        return Pair(dst, layout.shape().clone())
    }

    public fun metalFwd(
        storage: MetalStorage,
        layout: Layout,
    ): Pair<MetalStorage, Shape> {
        val self = this
        val name =
            if (self.asc) {
                when (storage.dtype()) {
                    DType.BF16 -> {
                        "asort_asc_bf16"
                    }

                    DType.F16 -> {
                        "asort_asc_f16"
                    }

                    DType.F32 -> {
                        "asort_asc_f32"
                    }

                    DType.F64 -> {
                        "asort_asc_f64"
                    }

                    DType.U8 -> {
                        "asort_asc_u8"
                    }

                    DType.U32 -> {
                        "asort_asc_u32"
                    }

                    DType.I16 -> {
                        "asort_asc_i16"
                    }

                    DType.I32 -> {
                        "asort_asc_i32"
                    }

                    DType.I64 -> {
                        "asort_asc_i64"
                    }

                    DType.F8E4M3 -> {
                        throw CandleError.Msg("Metal device does not yet support F8E4M3.").bt()
                    }

                    DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                        throw CandleError.UnsupportedDTypeForOp(storage.dtype(), "argsort").bt()
                    }
                }
            } else {
                when (storage.dtype()) {
                    DType.BF16 -> {
                        "asort_desc_bf16"
                    }

                    DType.F16 -> {
                        "asort_desc_f16"
                    }

                    DType.F32 -> {
                        "asort_desc_f32"
                    }

                    DType.F64 -> {
                        "asort_desc_f64"
                    }

                    DType.U8 -> {
                        "asort_desc_u8"
                    }

                    DType.U32 -> {
                        "asort_desc_u32"
                    }

                    DType.I16 -> {
                        "asort_desc_i16"
                    }

                    DType.I32 -> {
                        "asort_desc_i32"
                    }

                    DType.I64 -> {
                        "asort_desc_i64"
                    }

                    DType.F8E4M3 -> {
                        throw CandleError.Msg("Metal device does not yet support F8E4M3.").bt()
                    }

                    DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                        throw CandleError.UnsupportedDTypeForOp(storage.dtype(), "argsort").bt()
                    }
                }
            }
        val device = storage.device()
        val kernels = device.kernels()
        val commandEncoder = device.commandEncoder().unwrap()
        val el = layout.shape().elemCount()
        val ncols = self.lastDim
        val nrows = el / ncols
        val src = bufferO(storage.buffer(), layout, storage.dtype())
        val dst = device.newBuffer(el, DType.U32, "asort").unwrap()
        var ncolsPad = 1
        while (ncolsPad < ncols) {
            ncolsPad *= 2
        }
        callArgSort(
            device.metalDevice(),
            commandEncoder,
            kernels,
            name,
            nrows,
            ncols,
            ncolsPad,
            src,
            dst,
        ).unwrap()
        val dstStorage = MetalStorage.new(dst, device.clone(), el, DType.U32)
        return Pair(dstStorage, layout.shape().clone())
    }

    private val map: ArgSort.(CudaStorageSlice, CudaDevice, Layout) -> CudaStorageSlice = { slice, dev, layout ->
        when (slice) {
            is CudaStorageSlice.U8 -> f(slice.data, dev, layout, { s: CudaSlice<UByte> -> CudaStorageSlice.U8(s) }).unwrap()
            is CudaStorageSlice.U32 -> f(slice.data, dev, layout, { s: CudaSlice<UInt> -> CudaStorageSlice.U32(s) }).unwrap()
            is CudaStorageSlice.I16 -> f(slice.data, dev, layout, { s: CudaSlice<Short> -> CudaStorageSlice.I16(s) }).unwrap()
            is CudaStorageSlice.I32 -> f(slice.data, dev, layout, { s: CudaSlice<Int> -> CudaStorageSlice.I32(s) }).unwrap()
            is CudaStorageSlice.I64 -> f(slice.data, dev, layout, { s: CudaSlice<Long> -> CudaStorageSlice.I64(s) }).unwrap()
            is CudaStorageSlice.BF16 -> f(slice.data, dev, layout, { s: CudaSlice<BFloat16> -> CudaStorageSlice.BF16(s) }).unwrap()
            is CudaStorageSlice.F16 -> f(slice.data, dev, layout, { s: CudaSlice<Float16> -> CudaStorageSlice.F16(s) }).unwrap()
            is CudaStorageSlice.F32 -> f(slice.data, dev, layout, { s: CudaSlice<Float> -> CudaStorageSlice.F32(s) }).unwrap()
            is CudaStorageSlice.F64 -> f(slice.data, dev, layout, { s: CudaSlice<Double> -> CudaStorageSlice.F64(s) }).unwrap()
            is CudaStorageSlice.F8E4M3 -> f(slice.data, dev, layout, { s: CudaSlice<F8E4M3> -> CudaStorageSlice.F8E4M3(s) }).unwrap()
            else -> throw CandleError.UnsupportedDTypeForOp(DType.F4, "argsort").bt()
        }
    }
}

private val kernels: KernelsMod = KernelsMod

private object KernelsMod {
    val SORT: KernelModule = KernelModule(0, "sort")
}

private val kernelName: (String) -> String = { base -> "${base}_f32" }

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
public fun Tensor.argSortLastDim(asc: Boolean): Tensor {
    val self = this
    if (!self.isContiguous()) {
        throw CandleError.RequiresContiguous("arg_sort_last_dim")
    }
    val last = self.dims().lastOrNull()
    val lastDim =
        when (last) {
            null -> throw CandleError.Msg("empty last-dim in arg-sort")
            else -> last
        }
    // No need for a backward pass for arg sort.
    return self.applyOp1NoBwd(ArgSort(asc, lastDim))
}

/**
 * Sorts the tensor along the last dimension, returns the sorted tensor together with the
 * sorted indexes.
 *
 * If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
 * descending order. The sort is unstable so there is no guarantees on the final order when it
 * comes to ties.
 */
public fun Tensor.sortLastDim(asc: Boolean): Pair<Tensor, Tensor> {
    val self = this
    if (!self.isContiguous()) {
        throw CandleError.RequiresContiguous("sort_last_dim")
    }
    val asort = self.argSortLastDim(asc)
    val sorted = self.gather(asort, D.Minus1)
    return Pair(sorted, asort)
}
