// port-lint: source candle-core/src/indexer.rs
package io.github.kotlinmania.candle

public sealed class TensorIndexer {
    public data class Select(
        val n: Int,
    ) : TensorIndexer()

    public data class Narrow(
        val start: Int?,
        val end: Int?,
    ) : TensorIndexer()

    public data class IndexSelect(
        val indexes: Tensor,
    ) : TensorIndexer()

    public companion object {
        public fun select(n: Int): TensorIndexer = Select(n)

        public fun narrow(start: Int?, end: Int?): TensorIndexer = Narrow(start, end)

        public fun indexSelect(indexes: Tensor): TensorIndexer = IndexSelect(indexes)

        public val full: TensorIndexer = Narrow(null, null)
    }
}

public fun Tensor.i(vararg indexers: TensorIndexer): Tensor {
    var x = this
    val dims = shape().dims
    var currentDim = 0
    for ((i, indexer) in indexers.withIndex()) {
        x =
            when (indexer) {
                is TensorIndexer.Select -> {
                    val out = x.narrow(currentDim, indexer.n, 1).squeeze(currentDim)
                    out
                }
                is TensorIndexer.Narrow -> {
                    val start = indexer.start ?: 0
                    val stop = indexer.end ?: dims[i]
                    val len = maxOf(0, stop - start)
                    val out = x.narrow(currentDim, start, len)
                    currentDim++
                    out
                }
                is TensorIndexer.IndexSelect -> {
                    if (indexer.indexes.rank() != 1) {
                        bail("multi-dimensional tensor indexing is not supported")
                    }
                    val out = x.indexSelect(indexer.indexes, currentDim)
                    currentDim++
                    out
                }
            }
    }
    return x
}

internal fun Tensor.i(range: IntRange): Tensor =
    i(TensorIndexer.Narrow(range.first, range.last + 1))

public fun Tensor.i(start: Int, end: Int): Tensor =
    i(TensorIndexer.Narrow(start, end))

public fun Tensor.i(index: Int): Tensor =
    i(TensorIndexer.Select(index))
