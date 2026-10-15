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
        /** Creates a [Select] indexer for single-index selection at position [n]. */
        public fun select(n: Int): TensorIndexer = Select(n)

        /** Creates a [Narrow] range indexer from [start] (inclusive) to [end] (exclusive). */
        public fun narrow(start: Int?, end: Int?): TensorIndexer = Narrow(start, end)

        /** Creates an [IndexSelect] indexer using the 1D [indexes] tensor. */
        public fun indexSelect(indexes: Tensor): TensorIndexer = IndexSelect(indexes)

        /** Unbounded range indexer (`..`) selecting the entire extent of a dimension. */
        public val full: TensorIndexer = Narrow(null, null)

        /** Converts a single integer [index] into a [Select] indexer. */
        public fun from(index: Int): TensorIndexer = Select(index)

        /** Converts a [UIntArray] of indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: UIntArray): TensorIndexer =
            try {
                val tensor = Tensor.fromSlice(index.toList(), Shape(index.size), Device.Cpu)
                IndexSelect(tensor)
            } catch (e: CandleException) {
                Err(e)
            }

        /** Converts a [List] of [UInt] indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: List<UInt>): TensorIndexer {
            val len = index.size
            return try {
                val tensor = Tensor.fromVec(index, Shape(len), Device.Cpu)
                IndexSelect(tensor)
            } catch (e: CandleException) {
                Err(e)
            }
        }

        /** Wraps a 1D [tensor] of integer indices as an [IndexSelect] indexer. */
        public fun from(tensor: Tensor): TensorIndexer = IndexSelect(tensor.clone())

        /** Converts an [RB] range specification into a [Narrow] slice indexer. */
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
 * Intended to be used by the trait `.i()`.
 *
 * ```kotlin
 * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
 *
 * val c0 = a.i(0..0)
 * check(c0.shape().dims() == listOf(1, 3))
 *
 * val c1 = a.i(0)
 * check(c1.shape().dims() == listOf(3))
 *
 * val c2 = a.i(Pair(TensorIndexer.full, TensorIndexer.narrow(0, 2)))
 * check(c2.shape().dims() == listOf(2, 2))
 *
 * val c3 = a.i(Pair(TensorIndexer.full, TensorIndexer.narrow(0, 3)))
 * check(c3.shape().dims() == listOf(2, 3))
 * ```
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

/**
 * Index a tensor along multiple dimensions using a vararg list of [TensorIndexer]s.
 *
 * See [Tensor.i] for detailed single-axis and multi-axis slicing examples.
 */
public fun Tensor.i(vararg indexers: TensorIndexer): Tensor = index(indexers.toList())

/**
 * Index a tensor along the first dimension using a single [TensorIndexer].
 *
 * ```kotlin
 * val a = Tensor.new(listOf(listOf(0.0, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu)
 *
 * val b = a.i(0)
 * check(b.shape().dims() == listOf(2))
 * check(b.toVec1<Double>() == listOf(0.0, 1.0))
 *
 * val c = a.i(TensorIndexer.narrow(0, 2))
 * check(c.shape().dims() == listOf(2, 2))
 * check(c.toVec2<Double>() == listOf(listOf(0.0, 1.0), listOf(2.0, 3.0)))
 *
 * val d = a.i(TensorIndexer.narrow(1, null))
 * check(d.shape().dims() == listOf(2, 2))
 * check(d.toVec2<Double>() == listOf(listOf(2.0, 3.0), listOf(4.0, 5.0)))
 * ```
 */
public fun Tensor.i(indexer: TensorIndexer): Tensor = index(listOf(indexer))

/**
 * Index a tensor along two dimensions using a pair of [TensorIndexer]s.
 *
 * ```kotlin
 * val a = Tensor.new(
 *     listOf(
 *         listOf(0f, 1f, 2f),
 *         listOf(3f, 4f, 5f),
 *         listOf(6f, 7f, 8f),
 *     ),
 *     Device.Cpu,
 * )
 *
 * val b = a.i(Pair(TensorIndexer.Select(1), TensorIndexer.Select(0)))
 * check(b.toVec0<Float>() == 3f)
 *
 * val c = a.i(Pair(TensorIndexer.narrow(0, 2), TensorIndexer.Select(1)))
 * check(c.shape().dims() == listOf(2))
 * check(c.toVec1<Float>() == listOf(1f, 4f))
 *
 * val d = a.i(Pair(TensorIndexer.narrow(2, null), TensorIndexer.full))
 * check(d.shape().dims() == listOf(1, 3))
 * check(d.toVec2<Float>() == listOf(listOf(6f, 7f, 8f)))
 * ```
 */
public fun Tensor.i(indexers: Pair<TensorIndexer, TensorIndexer>): Tensor =
    index(listOf(indexers.first, indexers.second))

/**
 * Index a tensor along three dimensions using a triple of [TensorIndexer]s.
 *
 *```kotlin
 * val a = Tensor.new(
 *     listOf(
 *         listOf(0f, 1f),
 *         listOf(2f, 3f),
 *         listOf(4f, 5f),
 *     ),
 *     Device.Cpu,
 * )
 *
 * val b = a.i(0)
 * check(b.shape().dims() == listOf(2))
 * check(b.toVec1<Float>() == listOf(0f, 1f))
 *
 * val c = a.i(TensorIndexer.narrow(0, 2))
 * check(c.shape().dims() == listOf(2, 2))
 * check(c.toVec2<Float>() == listOf(listOf(0f, 1f), listOf(2f, 3f)))
 *
 * val d = a.i(TensorIndexer.narrow(1, null))
 * check(d.shape().dims() == listOf(2, 2))
 * check(d.toVec2<Float>() == listOf(listOf(2f, 3f), listOf(4f, 5f)))
 * ```
 */
public fun Tensor.i(indexers: Triple<TensorIndexer, TensorIndexer, TensorIndexer>): Tensor =
    index(listOf(indexers.first, indexers.second, indexers.third))

/**
 * Slice a tensor along the first dimension using an [IntRange].
 *
 * See [Tensor.i] for details.
 */
internal fun Tensor.i(range: IntRange): Tensor =
    i(TensorIndexer.Narrow(range.first, range.last + 1))

/**
 * Slice a tensor along the first dimension from [start] (inclusive) to [end] (exclusive).
 *
 * See [Tensor.i] for details.
 */
public fun Tensor.i(start: Int, end: Int): Tensor =
    i(TensorIndexer.Narrow(start, end))

/**
 * Select a single index along the first dimension of a tensor.
 *
 * See [Tensor.i] for details.
 */
public fun Tensor.i(index: Int): Tensor =
    i(TensorIndexer.Select(index))
