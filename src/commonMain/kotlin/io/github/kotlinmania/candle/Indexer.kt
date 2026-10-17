// port-lint: source indexer.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.Bound.Excluded
import io.github.kotlinmania.candle.Bound.Included
import io.github.kotlinmania.candle.Bound.Unbounded

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
    ) : Bound<T>() {
        public val idx: T get() = value
        public val n: T get() = value
    }

    /** An exclusive endpoint bound stopping or starting adjacent to [value]. */
    public data class Excluded<T>(
        val value: T,
    ) : Bound<T>() {
        public val idx: T get() = value
        public val n: T get() = value
    }

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

private val UIntArray.toLongList: () -> List<Long> get() = { map { it.toLong() } }
private val <T> IndexedValue<T>.i: Int get() = index
private val <T> IndexedValue<T>.indexer: T get() = value
private val Int.saturatingSub: (Int) -> Int get() = { other -> maxOf(0, this - other) }
private val Tensor.self: Tensor get() = this
private val Tensor.isOk: Boolean get() = rank() >= 0
private val Tensor.e: CandleException get() = CandleException.Msg("invalid index")
private val Int.into: () -> TensorIndexer get() = { TensorIndexer.from(this) }
private val TensorIndexer.into: () -> TensorIndexer get() = { this }
private val Pair<TensorIndexer, TensorIndexer>.a: TensorIndexer get() = first
private val Pair<TensorIndexer, TensorIndexer>.b: TensorIndexer get() = second

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
    @ConsistentCopyVisibility
    public data class Narrow internal constructor(
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

    /**
     * Indexing error captured while constructing a [TensorIndexer] from an index collection.
     */
    public data class Err(
        val error: CandleException,
    ) : TensorIndexer() {
        public val e: CandleException get() = error
    }

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
        public fun from(index: Int): TensorIndexer = TensorIndexer.Select(index)

        /** Converts a [UIntArray] of indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: UIntArray): TensorIndexer {
            val t = Tensor.new(index, Shape(index.size), Device.Cpu)
            return when {
                t.isOk -> TensorIndexer.IndexSelect(t)
                else -> TensorIndexer.Err(CandleException.Msg("invalid index"))
            }
        }

        /** Converts a [List] of [UInt] indices into a CPU-backed [IndexSelect] indexer. */
        public fun from(index: List<UInt>): TensorIndexer {
            val len = index.len()
            val t = Tensor.new(index.toUIntArray(), Shape(len), Device.Cpu)
            return when {
                t.isOk -> TensorIndexer.IndexSelect(t)
                else -> TensorIndexer.Err(CandleException.Msg("invalid index"))
            }
        }

        /** Wraps a 1D [tensor] of integer indices as an [IndexSelect] indexer. */
        public fun from(tensor: Tensor): TensorIndexer = TensorIndexer.IndexSelect(tensor.clone())

        /** Converts an [RB] range specification into a [Narrow] slice indexer. */
        public fun <T : RB> from(range: T): TensorIndexer {
            // Bound::*
            val start =
                when (range.startBound()) {
                    is Included -> Included(range.startBound().idx)
                    is Excluded -> Excluded(range.startBound().idx)
                    is Unbounded -> Unbounded
                }
            val end =
                when (range.endBound()) {
                    is Included -> Included(range.endBound().idx)
                    is Excluded -> Excluded(range.endBound().idx)
                    is Unbounded -> Unbounded
                }
            return TensorIndexer.Narrow(null, null, start, end)
        }
    }
}

private val Bound<Int>.idx: Int get() = (this as? Included)?.value ?: (this as Excluded).value

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

private val TensorIndexer.n: Int get() = (this as TensorIndexer.Select).n
private val TensorIndexer.leftBound: Bound<Int> get() = (this as TensorIndexer.Narrow).leftBound
private val TensorIndexer.rightBound: Bound<Int> get() = (this as TensorIndexer.Narrow).rightBound
private val TensorIndexer.start: Int? get() = (this as TensorIndexer.Narrow).start
private val TensorIndexer.end: Int? get() = (this as TensorIndexer.Narrow).end
private val TensorIndexer.indexes: Tensor get() = (this as TensorIndexer.IndexSelect).indexes
private val TensorIndexer.e: CandleException get() = (this as TensorIndexer.Err).error
private val Bound<Int>.n: Int get() = (this as? Included)?.value ?: (this as Excluded).value

/**
 * Intended to be used by the trait `.i()`.
 */
public fun Tensor.index(indexers: List<TensorIndexer>): Tensor {
    var x = self.clone()
    val dims = self.shape().dims()
    var currentDim = 0
    for (entry in indexers.iter().enumerate()) {
        val indexer = entry.indexer
        x =
            when (indexer) {
                is TensorIndexer.Select -> {
                    x.narrow(currentDim, indexer.n, 1).squeeze(currentDim)
                }

                is TensorIndexer.Narrow -> {
                    val start =
                        when (indexer.leftBound) {
                            is Bound.Included -> indexer.leftBound.n
                            is Bound.Excluded -> indexer.leftBound.n + 1
                            is Bound.Unbounded -> indexer.start ?: 0
                        }
                    val stop =
                        when (indexer.rightBound) {
                            is Bound.Included -> indexer.rightBound.n + 1
                            is Bound.Excluded -> indexer.rightBound.n
                            is Bound.Unbounded -> indexer.end ?: dims[entry.i]
                        }
                    val out = x.narrow(currentDim, start, stop.saturatingSub(start))
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

                is TensorIndexer.Err -> {
                    bail("indexing error ${indexer.e}")
                }
            }
    }
    return Ok(x)
}

/**
 * Index a tensor along the first dimension using a single [Int] index.
 */
public fun Tensor.i(index: Int): Tensor = self.index(listOf(TensorIndexer.from(index)))

/**
 * Index a tensor along the first dimension using a single [TensorIndexer].
 */
public fun Tensor.i(a: TensorIndexer): Tensor = self.index(listOf(a.into()))

/**
 * Index a tensor along two dimensions using a pair of [TensorIndexer]s.
 */
public fun Tensor.i(ab: Pair<TensorIndexer, TensorIndexer>): Tensor = self.index(listOf(ab.a.into(), ab.b.into()))
