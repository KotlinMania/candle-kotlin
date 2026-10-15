// port-lint: source sort.rs
package io.github.kotlinmania.candle

public data class ArgSort(
    val asc: Boolean,
    val lastDim: Int,
) : CustomOp1 {
    // Safety: indexes are set later in the parallelized section.
    public fun asort(vs: FloatArray, layout: Layout): UIntArray {
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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
        val elCount = layout.shape().elemCount()
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

    public fun asort(vs: ByteArray, layout: Layout): UIntArray {
        val elCount = layout.shape().elemCount()
        val sortIndexes = UIntArray(elCount)
        val numChunks = elCount / lastDim
        for (chunkIdx in 0 until numChunks) {
            val offset = chunkIdx * lastDim
            val indices = Array(lastDim) { it }
            if (asc) {
                indices.sortWith { i, j ->
                    F8E4M3(vs[offset + i]).toFloat().compareTo(F8E4M3(vs[offset + j]).toFloat())
                }
            } else {
                indices.sortWith { i, j ->
                    F8E4M3(vs[offset + j]).toFloat().compareTo(F8E4M3(vs[offset + i]).toFloat())
                }
            }
            for (i in 0 until lastDim) {
                sortIndexes[offset + i] = indices[i].toUInt()
            }
        }
        return sortIndexes
    }

    override fun name(): String = "argsort"

    override fun cpuFwd(storage: CpuStorage, layout: Layout): Pair<CpuStorage, Shape> {
        val sortIndexes = when (storage) {
            is CpuStorage.U8 -> asort(storage.data, layout)
            is CpuStorage.U32 -> asort(storage.data, layout)
            is CpuStorage.I16 -> asort(storage.data, layout)
            is CpuStorage.I32 -> asort(storage.data, layout)
            is CpuStorage.I64 -> asort(storage.data, layout)
            is CpuStorage.BF16 -> {
                val elCount = layout.shape().elemCount()
                val out = UIntArray(elCount)
                val numChunks = elCount / lastDim
                for (chunkIdx in 0 until numChunks) {
                    val offset = chunkIdx * lastDim
                    val indices = Array(lastDim) { it }
                    if (asc) {
                        indices.sortWith { i, j ->
                            BFloat16(storage.data[offset + i]).toFloat()
                                .compareTo(BFloat16(storage.data[offset + j]).toFloat())
                        }
                    } else {
                        indices.sortWith { i, j ->
                            BFloat16(storage.data[offset + j]).toFloat()
                                .compareTo(BFloat16(storage.data[offset + i]).toFloat())
                        }
                    }
                    for (i in 0 until lastDim) {
                        out[offset + i] = indices[i].toUInt()
                    }
                }
                out
            }
            is CpuStorage.F16 -> {
                val elCount = layout.shape().elemCount()
                val out = UIntArray(elCount)
                val numChunks = elCount / lastDim
                for (chunkIdx in 0 until numChunks) {
                    val offset = chunkIdx * lastDim
                    val indices = Array(lastDim) { it }
                    if (asc) {
                        indices.sortWith { i, j ->
                            Float16(storage.data[offset + i]).toFloat()
                                .compareTo(Float16(storage.data[offset + j]).toFloat())
                        }
                    } else {
                        indices.sortWith { i, j ->
                            Float16(storage.data[offset + j]).toFloat()
                                .compareTo(Float16(storage.data[offset + i]).toFloat())
                        }
                    }
                    for (i in 0 until lastDim) {
                        out[offset + i] = indices[i].toUInt()
                    }
                }
                out
            }
            is CpuStorage.F32 -> asort(storage.data, layout)
            is CpuStorage.F64 -> asort(storage.data, layout)
            is CpuStorage.F8E4M3 -> asort(storage.data, layout)
            is CpuStorage.F6E2M3 -> throw CandleException.UnsupportedDTypeForOp(DType.F6E2M3, "argsort")
            is CpuStorage.F6E3M2 -> throw CandleException.UnsupportedDTypeForOp(DType.F6E3M2, "argsort")
            is CpuStorage.F4 -> throw CandleException.UnsupportedDTypeForOp(DType.F4, "argsort")
            is CpuStorage.F8E8M0 -> throw CandleException.UnsupportedDTypeForOp(DType.F8E8M0, "argsort")
        }
        return Pair(CpuStorage.U32(sortIndexes), layout.shape())
    }

    override fun cudaFwd(storage: CudaStorage, layout: Layout): Pair<CudaStorage, Shape> =
        throw CandleException.NotCompiledWithCudaSupport

    override fun metalFwd(storage: MetalStorage, layout: Layout): Pair<MetalStorage, Shape> =
        throw CandleException.NotCompiledWithMetalSupport

    // Limit block dim to 1024 threads, which is the maximum on modern CUDA gpus.
    // Dummy types don't support sorting
    public fun <T> f(src: T, dev: CudaDevice, layout: Layout): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport
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
 *
 * No backward pass is needed for argument sort.
 */
public fun Tensor.argSortLastDim(asc: Boolean): Tensor {
    if (!isContiguous()) {
        throw CandleException.RequiresContiguous("arg_sort_last_dim")
    }
    val lastDim = dims().lastOrNull() ?: throw CandleException.Msg("empty last-dim in arg-sort")
    // No need for a backward pass for arg sort.
    return applyOp1NoBwd(ArgSort(asc, lastDim))
}

/**
 * Sorts the tensor along the last dimension, returns the sorted tensor together with the
 * sorted indexes.
 *
 * If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
 * descending order. The sort is unstable so there is no guarantees on the final order when it
 * comes to ties.
 *
 * Returns a pair of `(sortedTensor, sortedIndices)`.
 */
public fun Tensor.sortLastDim(asc: Boolean): Pair<Tensor, Tensor> {
    if (!isContiguous()) {
        throw CandleException.RequiresContiguous("sort_last_dim")
    }
    val asort = argSortLastDim(asc)
    val sorted = gather(asort, D.Minus1)
    return Pair(sorted, asort)
}
