// port-lint: source indexer.rs
package io.github.kotlinmania.candle

public sealed class Bound<out T> {
    public data class Included<T>(
        val value: T,
    ) : Bound<T>()

    public data class Excluded<T>(
        val value: T,
    ) : Bound<T>()

    public object Unbounded : Bound<Nothing>()
}

public interface RB {
    public fun startBound(): Bound<Int>
    public fun endBound(): Bound<Int>
}

/**
 * Generic structure used to index a slice of the tensor
 */
public sealed class TensorIndexer {
    /**
     * This selects the elements for which an index has some specific value.
     */
    public data class Select(
        val n: Int,
    ) : TensorIndexer()

    /**
     * This is a regular slice, purely indexing a chunk of the tensor
     */
    public data class Narrow(
        val start: Int?,
        val end: Int?,
        val leftBound: Bound<Int> = start?.let { Bound.Included(it) } ?: Bound.Unbounded,
        val rightBound: Bound<Int> = end?.let { Bound.Excluded(it) } ?: Bound.Unbounded,
    ) : TensorIndexer()

    /**
     * Indexing via a 1d tensor
     */
    public data class IndexSelect(
        val indexes: Tensor,
    ) : TensorIndexer()

    public data class Err(
        val error: CandleException,
    ) : TensorIndexer()

    public companion object {
        public fun select(n: Int): TensorIndexer = Select(n)

        public fun narrow(start: Int?, end: Int?): TensorIndexer = Narrow(start, end)

        public fun indexSelect(indexes: Tensor): TensorIndexer = IndexSelect(indexes)

        public val full: TensorIndexer = Narrow(null, null)

        public fun from(index: Int): TensorIndexer = Select(index)

        public fun from(index: UIntArray): TensorIndexer =
            try {
                val tensor = Tensor.fromSlice(index.toList(), Shape(index.size), Device.Cpu)
                IndexSelect(tensor)
            } catch (e: CandleException) {
                Err(e)
            }

        public fun from(index: List<UInt>): TensorIndexer {
            val len = index.size
            return try {
                val tensor = Tensor.fromVec(index, Shape(len), Device.Cpu)
                IndexSelect(tensor)
            } catch (e: CandleException) {
                Err(e)
            }
        }

        public fun from(tensor: Tensor): TensorIndexer = IndexSelect(tensor.clone())

        public fun <T : RB> from(range: T): TensorIndexer {
            val start = when (val s = range.startBound()) {
                is Bound.Included -> s.value
                is Bound.Excluded -> s.value + 1
                is Bound.Unbounded -> null
            }
            val end = when (val e = range.endBound()) {
                is Bound.Included -> e.value + 1
                is Bound.Excluded -> e.value
                is Bound.Unbounded -> null
            }
            return Narrow(start, end, range.startBound(), range.endBound())
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
 * Intended to be use by the trait `.i()`
 */
public fun Tensor.index(indexers: List<TensorIndexer>): Tensor {
    var x = this.clone()
    val dims = shape().dims()
    var currentDim = 0
    for ((i, indexer) in indexers.withIndex()) {
        x = when (indexer) {
            is TensorIndexer.Select -> x.narrow(currentDim, indexer.n, 1).squeeze(currentDim)
            is TensorIndexer.Narrow -> {
                val start = when (val lb = indexer.leftBound) {
                    is Bound.Included -> lb.value
                    is Bound.Excluded -> lb.value + 1
                    is Bound.Unbounded -> indexer.start ?: 0
                }
                val stop = when (val rb = indexer.rightBound) {
                    is Bound.Included -> rb.value + 1
                    is Bound.Excluded -> rb.value
                    is Bound.Unbounded -> indexer.end ?: dims[i]
                }
                val out = x.narrow(currentDim, start, maxOf(0, stop - start))
                currentDim += 1
                out
            }
            is TensorIndexer.IndexSelect -> {
                if (indexer.indexes.rank() != 1) {
                    throw CandleException.Msg("multi-dimensional tensor indexing is not supported")
                }
                val out = x.indexSelect(indexer.indexes.toDevice(x.device()), currentDim)
                currentDim += 1
                out
            }
            is TensorIndexer.Err -> throw CandleException.Msg("indexing error ${indexer.error}")
        }
    }
    return x
}

public fun Tensor.i(vararg indexers: TensorIndexer): Tensor = index(indexers.toList())

public fun Tensor.i(indexer: TensorIndexer): Tensor = index(listOf(indexer))

public fun Tensor.i(indexers: Pair<TensorIndexer, TensorIndexer>): Tensor =
    index(listOf(indexers.first, indexers.second))

public fun Tensor.i(indexers: Triple<TensorIndexer, TensorIndexer, TensorIndexer>): Tensor =
    index(listOf(indexers.first, indexers.second, indexers.third))

internal fun Tensor.i(range: IntRange): Tensor =
    i(TensorIndexer.Narrow(range.first, range.last + 1))

public fun Tensor.i(start: Int, end: Int): Tensor =
    i(TensorIndexer.Narrow(start, end))

public fun Tensor.i(index: Int): Tensor =
    i(TensorIndexer.Select(index))
