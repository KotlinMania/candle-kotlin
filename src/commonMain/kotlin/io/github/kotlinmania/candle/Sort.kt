// port-lint: source candle-core/src/sort.rs
package io.github.kotlinmania.candle

public data class ArgSort(
    val asc: Boolean,
    val lastDim: Int,
) {
    public fun asort(vs: FloatArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: DoubleArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: IntArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: LongArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: ShortArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: UByteArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    public fun asort(vs: UIntArray, layout: Layout): UIntArray {
        val elCount = layout.shape.elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    vs[offset + i].compareTo(vs[offset + j])
                }
            } else {
                indices.sortWith { i, j ->
                    vs[offset + j].compareTo(vs[offset + i])
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }
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
public fun Tensor.argSortLastDim(asc: Boolean): Tensor {
    if (!isContiguous()) {
        throw CandleException.RequiresContiguous("arg_sort_last_dim")
    }
    val lastDim = dims().lastOrNull() ?: bail("empty last-dim in arg-sort")
    val argSort = ArgSort(asc, lastDim)
    val cont = contiguous()
    val cpu = (cont.storage() as Storage.Cpu).storage
    val sortedIndices = when (cpu) {
        is CpuStorage.F32 -> argSort.asort(cpu.data, layout())
        is CpuStorage.F64 -> argSort.asort(cpu.data, layout())
        is CpuStorage.I32 -> argSort.asort(cpu.data, layout())
        is CpuStorage.I64 -> argSort.asort(cpu.data, layout())
        is CpuStorage.I16 -> argSort.asort(cpu.data, layout())
        is CpuStorage.U32 -> argSort.asort(cpu.data, layout())
        is CpuStorage.U8 -> argSort.asort(cpu.data, layout())
        is CpuStorage.BF16 -> argSort.asort(cpu.data, layout())
        is CpuStorage.F16 -> argSort.asort(cpu.data, layout())
        is CpuStorage.F8E4M3 -> {
            val elCount = layout().shape.elemCount()
            val sortIndexes = UIntArray(elCount)
            val numChunks = elCount / lastDim
            for (chunkIdx in 0 until numChunks) {
                val offset = chunkIdx * lastDim
                val indices = Array(lastDim) { it }
                indices.sortWith { i, j ->
                    if (asc) cpu.data[offset + i].compareTo(cpu.data[offset + j])
                    else cpu.data[offset + j].compareTo(cpu.data[offset + i])
                }
                for (i in 0 until lastDim) {
                    sortIndexes[offset + i] = indices[i].toUInt()
                }
            }
            sortIndexes
        }
    }
    val newStorage = Storage.Cpu(CpuStorage.U32(sortedIndices))
    return Tensor.fromStorage(newStorage, shape(), BackpropOp.None, false)
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
    if (!isContiguous()) {
        throw CandleException.RequiresContiguous("sort_last_dim")
    }
    val asort = argSortLastDim(asc)
    val sorted = gather(asort, rank() - 1)
    return Pair(sorted, asort)
}
