// port-lint: source indexer.rs
package io.github.kotlinmania.candle

/**
 * Endpoint bound of a 1D slice range along a tensor axis.
 *
 * Models inclusive (`Included`), exclusive (`Excluded`), or open (`Unbounded`) range endpoints
 * for tensor slicing via [TensorIndexer.Narrow].
 */
public sealed class Bound<out T> {
    /** An inclusive endpoint bound containing [value]. */
    public data class Included<T>(
        val value: T,
    ) : Bound<T>()

    /** An exclusive endpoint bound stopping or starting adjacent to [value]. */
    public data class Excluded<T>(
        val value: T,
    ) : Bound<T>()

    /** An unbounded endpoint extending to the beginning (`0`) or end (`dimSize`) of the axis. */
    public object Unbounded : Bound<Nothing>()
}

/**
 * Range bounds abstraction providing start and end [Bound] endpoints for tensor axis slicing.
 */
public interface RB {
    /** Returns the lower bound of the range along the indexed dimension. */
    public fun startBound(): Bound<Int>
    /** Returns the upper bound of the range along the indexed dimension. */
    public fun endBound(): Bound<Int>
}

/**
 * Generic structure used to index a slice of the tensor
 */
public sealed class TensorIndexer {
    /**
     * This selects the elements for which an index has some specific value.
     *
     * Selecting a single index along an axis narrows that dimension to length 1
     * and squeezes the dimension out of the resulting tensor shape.
     */
    public data class Select(
        val n: Int,
    ) : TensorIndexer()

    /**
     * This is a regular slice, purely indexing a chunk of the tensor
     *
     * Preserves the indexed dimension with length equal to `stop - start`.
     */
    public data class Narrow(
        val start: Int?,
        val end: Int?,
        val leftBound: Bound<Int> = start?.let { Bound.Included(it) } ?: Bound.Unbounded,
        val rightBound: Bound<Int> = end?.let { Bound.Excluded(it) } ?: Bound.Unbounded,
    ) : TensorIndexer()

    /**
     * Indexing via a 1d tensor
     *
     * Gathers slices along the indexed axis at the positions specified by the 1D [indexes] tensor.
     */
    public data class IndexSelect(
        val indexes: Tensor,
    ) : TensorIndexer()

    /**
     * Indexing error captured while constructing a [TensorIndexer] from an index collection.
     */
    public data class Err(
        val error: CandleException,
    ) : TensorIndexer()

    public companion object {
        /** Creates a [Select] indexer for single-index selection at position `n`. */
        public val select: (Int) -> TensorIndexer = { n -> Select(n) }

        /** Creates a [Narrow] range indexer from `start` (inclusive) to `end` (exclusive). */
        public val narrow: (Int?, Int?) -> TensorIndexer = { start, end -> Narrow(start, end) }

        /** Creates an [IndexSelect] indexer using the 1D `indexes` tensor. */
        public val indexSelect: (Tensor) -> TensorIndexer = { indexes -> IndexSelect(indexes) }

        /** Unbounded range indexer (`..`) selecting the entire extent of a dimension. */
        public val full: TensorIndexer = Narrow(null, null)

        /** Converts a single integer [index] into a [Select] indexer. */
        public fun from(index: Int): TensorIndexer {
            return TensorIndexer.Select(index)
        }

        /** Converts a [UIntArray] of indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: UIntArray): TensorIndexer {
            val tensor = Tensor.new(index.map { it.toLong() }, Device.Cpu)
            return when (tensor.rank()) {
                1 -> TensorIndexer.IndexSelect(tensor)
                else -> TensorIndexer.Err(CandleException.Msg("invalid index"))
            }
        }

        /** Converts a [List] of [UInt] indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: List<UInt>): TensorIndexer {
            val len = index.size
            val tensor = Tensor.fromVec(index, Shape(len), Device.Cpu)
            return when (tensor.rank()) {
                1 -> TensorIndexer.IndexSelect(tensor)
                else -> TensorIndexer.Err(CandleException.Msg("invalid index"))
            }
        }

        /** Wraps a 1D [tensor] of integer indices as an [IndexSelect] indexer. */
        public fun from(tensor: Tensor): TensorIndexer {
            return TensorIndexer.IndexSelect(tensor.clone())
        }

        /** Converts an [RB] range specification into a [Narrow] slice indexer. */
        public fun <T : RB> from(range: T): TensorIndexer {
            val startBound = range.startBound()
            val start = when (startBound) {
                is Bound.Included -> Bound.Included(startBound.value)
                is Bound.Excluded -> Bound.Excluded(startBound.value)
                is Bound.Unbounded -> Bound.Unbounded
            }
            val endBound = range.endBound()
            val end = when (endBound) {
                is Bound.Included -> Bound.Included(endBound.value)
                is Bound.Excluded -> Bound.Excluded(endBound.value)
                is Bound.Unbounded -> Bound.Unbounded
            }
            return TensorIndexer.Narrow(null, null, start, end)
        }
    }
}

/**
 * Trait used to implement multiple signatures for ease of use of the slicing
 * of a tensor
 */
public interface IndexOp<T> {
    /**
     * Returns a slicing iterator which are the chunks of data necessary to
     * reconstruct the desired tensor.
     */
    public fun i(index: T): Tensor
}

/**
 * Intended to be used by the trait `.i()`.
 *
 * ```kotlin
 * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 *
 * val c1 = a.i(0)
 * check(c1.shape().dims() == listOf(3))
 * ```
 */
public fun Tensor.index(indexers: List<TensorIndexer>): Tensor {
    var x = this.clone()
    val dims = this.shape().dims()
    var currentDim = 0
    for (entry in indexers.withIndex()) {
        val i = entry.index
        val indexer = entry.value
        x = when (indexer) {
            is TensorIndexer.Select -> x.narrow(currentDim, indexer.n, 1).squeeze(currentDim)
            is TensorIndexer.Narrow -> {
                val leftBound = indexer.leftBound
                val start = when (leftBound) {
                    is Bound.Included -> leftBound.value
                    is Bound.Excluded -> leftBound.value + 1
                    is Bound.Unbounded -> indexer.start ?: 0
                }
                val rightBound = indexer.rightBound
                val stop = when (rightBound) {
                    is Bound.Included -> rightBound.value + 1
                    is Bound.Excluded -> rightBound.value
                    is Bound.Unbounded -> indexer.end ?: dims[i]
                }
                val out = x.narrow(currentDim, start, maxOf(0, stop - start))
                currentDim += 1
                out
            }
            is TensorIndexer.IndexSelect -> {
                if (indexer.indexes.rank() != 1) {
                    bail("multi-dimensional tensor indexing is not supported")
                }
                val out = x.indexSelect(indexer.indexes.toDevice(x.device()), currentDim)
                currentDim += 1
                out
            }
            is TensorIndexer.Err -> bail("indexing error ${indexer.error}")
        }
    }
    return x
}

/**
 * Index a tensor along the first dimension using a single [Int] index.
 */
public fun Tensor.i(index: Int): Tensor {
    return this.index(listOf(TensorIndexer.from(index)))
}

/**
 * Index a tensor along the first dimension using a single [TensorIndexer].
 */
public fun Tensor.i(a: TensorIndexer): Tensor {
    return this.index(listOf(a))
}

/**
 * Index a tensor along two dimensions using a pair of [TensorIndexer]s.
 */
public fun Tensor.i(ab: Pair<TensorIndexer, TensorIndexer>): Tensor {
    return this.index(listOf(ab.first, ab.second))
}
