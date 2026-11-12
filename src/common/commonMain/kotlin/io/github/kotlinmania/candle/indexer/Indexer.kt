// port-lint: source indexer.rs
package io.github.kotlinmania.candle.indexer

import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.tensor.Tensor

/**
 * Intended to be use by the trait `.i()`
 *
 * ```kotlin
 * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
 * val c = a.i(0 until 1).getOrElse { return Result.failure(it) }
 * check(c.shape().dims() == listOf(1, 3))
 * val c1 = a.i(0).getOrElse { return Result.failure(it) }
 * check(c1.shape().dims() == listOf(3))
 * val c2 = a.i(Pair(0 until 0, 0 until 2)).getOrElse { return Result.failure(it) }
 * check(c2.shape().dims() == listOf(2, 2))
 * val c3 = a.i(Pair(0 until 0, 0..2)).getOrElse { return Result.failure(it) }
 * check(c3.shape().dims() == listOf(2, 3))
 * ```
 */
public fun Tensor.index(indexers: List<TensorIndexer>): Result<Tensor> {
    var x = this.clone()
    val dims = this.shape().dims()
    var currentDim = 0
    for ((i, indexer) in indexers.enumerate()) {
        x = when (indexer) {
            is TensorIndexer.Select -> {
                val (n) = indexer
                x.narrow(currentDim, n, 1).getOrElse { return Result.failure(it) }.squeeze(currentDim).getOrElse { return Result.failure(it) }
            }
            is TensorIndexer.Narrow -> {
                val (leftBound, rightBound) = indexer
                val start =
                    when (leftBound) {
                        is Bound.Included -> {
                            val (n1) = leftBound
                            n1
                        }
                        is Bound.Excluded -> {
                            val (n2) = leftBound
                            n2 + 1
                        }
                        Bound.Unbounded -> 0
                    }
                val stop =
                    when (rightBound) {
                        is Bound.Included -> {
                            val (n3) = rightBound
                            n3 + 1
                        }
                        is Bound.Excluded -> {
                            val (n4) = rightBound
                            n4
                        }
                        Bound.Unbounded -> dims[i]
                    }
                val out = x.narrow(currentDim, start, stop.saturatingSub(start)).getOrElse { return Result.failure(it) }
                currentDim += 1
                out
            }
            is TensorIndexer.IndexSelect -> {
                val (indexes) = indexer
                if (indexes.rank() != 1) {
                    bail("multi-dimensional tensor indexing is not supported")
                }
                val out1 = x.indexSelect(indexes.toDevice(x.device()).getOrElse { return Result.failure(it) }, currentDim).getOrElse { return Result.failure(it) }
                currentDim += 1
                out1
            }
            is TensorIndexer.Err -> {
                val (e) = indexer
                bail("indexing error {e:?}")
            }
        }
    }
    return Result.success(x)
}

/**
 * Generic structure used to index a slice of the tensor
 */
public sealed class TensorIndexer : From<Int> {
    /**
     * This selects the elements for which an index has some specific value.
     */
    public data class Select(public var value: Int) : TensorIndexer()
    /**
     * This is a regular slice, purely indexing a chunk of the tensor
     */
    public data class Narrow(public var field0: Bound<Int>, public var field1: Bound<Int>) : TensorIndexer()
    /**
     * Indexing via a 1d tensor
     */
    public data class IndexSelect(public var value: Tensor) : TensorIndexer()
    public data class Err(public var value: Error) : TensorIndexer()

    public companion object {
        public fun from(index: ULong): TensorIndexer = TensorIndexer.Select(index)

        public fun from(index: List<UInt>): TensorIndexer =
            run {
                val matchVal = Tensor.new(index, Device.Cpu)
                if (matchVal.isSuccess) {
                    val tensor = matchVal.getOrThrow()
                    TensorIndexer.IndexSelect(tensor)
                } else {
                    val e = matchVal.exceptionOrNull()!!
                    TensorIndexer.Err(e)
                }
            }

        public fun fromVecUInt(index: List<UInt>): TensorIndexer {
            val len = index.size
            return run {
                val matchVal = Tensor.fromVec(index, len, Device.Cpu)
                if (matchVal.isSuccess) {
                    val tensor = matchVal.getOrThrow()
                    TensorIndexer.IndexSelect(tensor)
                } else {
                    val e = matchVal.exceptionOrNull()!!
                    TensorIndexer.Err(e)
                }
            }
        }

        public fun from(tensor: Tensor): TensorIndexer = TensorIndexer.IndexSelect(tensor.clone())

        public fun <T : RB> from(range: T): TensorIndexer {
            val start =
                when (val matchVal = range.startBound()) {
                    is Included -> {
                        val (idx) = matchVal
                        Included(idx)
                    }
                    is Excluded -> {
                        val (idx1) = matchVal
                        Excluded(idx1)
                    }
                    Unbounded -> Unbounded
                }
            val end =
                when (val matchVal1 = range.endBound()) {
                    is Included -> {
                        val (idx2) = matchVal1
                        Included(idx2)
                    }
                    is Excluded -> {
                        val (idx3) = matchVal1
                        Excluded(idx3)
                    }
                    Unbounded -> Unbounded
                }
            return TensorIndexer.Narrow(start, end)
        }
    }
}

public interface RB : RangeBounds<Int>

/**
 * Trait used to implement multiple signatures for ease of use of the slicing
 * of a tensor
 */
public interface IndexOp<T> {
    /**
     * Returns a slicing iterator which are the chunks of data necessary to
     * reconstruct the desired tensor.
     */
    public fun i(index: T): Result<Tensor>
}

/**
 * ```kotlin
 * import candlecore.DType
 * import candlecore.Device
 * import candlecore.IndexOp
 * import candlecore.Tensor
 *
 * val a = Tensor.new(listOf(listOf(0.0, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
 * val b = a.i(0).getOrElse { return Result.failure(it) }
 * check(b.shape().dims() == listOf(2))
 * check(b.toVec1<Double>().getOrElse { return Result.failure(it) } == listOf(0.0, 1.0))
 * val c = a.i(0 until 2).getOrElse { return Result.failure(it) }
 * check(c.shape().dims() == listOf(2, 2))
 * check(c.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(0.0, 1.0), listOf(2.0, 3.0)))
 * val d = a.i(1 until Int.MAX_VALUE).getOrElse { return Result.failure(it) }
 * check(d.shape().dims() == listOf(2, 2))
 * check(d.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(2.0, 3.0), listOf(4.0, 5.0)))
 * ```
 */
public fun Tensor.i(index: TensorIndexer): Result<Tensor> = this.index(listOf(index))

/**
 * ```kotlin
 * import candlecore.DType
 * import candlecore.Device
 * import candlecore.IndexOp
 * import candlecore.Tensor
 *
 * val a = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
 * val b = a.i(0).getOrElse { return Result.failure(it) }
 * check(b.shape().dims() == listOf(2))
 * check(b.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(0.0, 1.0))
 * val c = a.i(0 until 2).getOrElse { return Result.failure(it) }
 * check(c.shape().dims() == listOf(2, 2))
 * check(c.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(0.0, 1.0), listOf(2.0, 3.0)))
 * val d = a.i(1 until Int.MAX_VALUE).getOrElse { return Result.failure(it) }
 * check(d.shape().dims() == listOf(2, 2))
 * check(d.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(2.0, 3.0), listOf(4.0, 5.0)))
 * ```
 */
public fun Tensor.iA(tupleArg0: TensorIndexer): Result<Tensor> = this.index(listOf(a.into()))

/**
 * ```kotlin
 * import candlecore.DType
 * import candlecore.Device
 * import candlecore.IndexOp
 * import candlecore.Tensor
 *
 * val a = Tensor.new(listOf(listOf(0.0f, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0)), Device.Cpu).getOrElse { return Result.failure(it) }
 * val b = a.i(Pair(1, 0)).getOrElse { return Result.failure(it) }
 * check(b.toVec0<Float>().getOrElse { return Result.failure(it) } == 3.0)
 * val c = a.i(Pair(0 until 2, 1)).getOrElse { return Result.failure(it) }
 * check(c.shape().dims() == listOf(2))
 * check(c.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(1.0, 4.0))
 * val d = a.i(Pair(2 until Int.MAX_VALUE, 0 until 0)).getOrElse { return Result.failure(it) }
 * check(d.shape().dims() == listOf(1, 3))
 * check(d.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(6.0, 7.0, 8.0)))
 * ```
 */
public fun Tensor.i(tupleArg0: Pair<TensorIndexer, TensorIndexer>): Result<Tensor> = this.index(listOf(a.into(), b.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.i(tupleArg0: Triple<TensorIndexer, TensorIndexer, TensorIndexer>): Result<Tensor> = this.index(listOf(A.into(), B.into(), C.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.i(tupleArg0: List<Any?>): Result<Tensor> = this.index(listOf(A.into(), B.into(), C.into(), D.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDE(tupleArg0: List<Any?>): Result<Tensor> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDEF(tupleArg0: List<Any?>): Result<Tensor> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into(), F.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDEFG(tupleArg0: List<Any?>): Result<Tensor> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into(), F.into(), G.into()))
