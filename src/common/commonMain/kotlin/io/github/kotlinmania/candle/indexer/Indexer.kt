// port-lint: source indexer.rs
package io.github.kotlinmania.candle.indexer

import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.rustlang.core.result.Result
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.ops.Bound
import io.github.kotlinmania.rustlang.std.ops.RangeBounds
import io.github.kotlinmania.rustlang.std.ops.bound.*

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
public fun Tensor.index(indexers: List<TensorIndexer>): Result<Tensor, Error> {
    var x = this.clone()
    val dims = this.shape().dims()
    var currentDim = 0
    for ((i, indexer) in indexers.enumerate()) {
        x =
            when (indexer) {
                is TensorIndexer.Select -> {
                    val (n) = indexer
                    when (
                        val tryResult =
                            when (val tryResult = x.narrow(currentDim, n, 1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.squeeze(currentDim)
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
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
                    val out =
                        when (val tryResult = x.narrow(currentDim, start, stop.saturatingSub(start))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    currentDim += 1
                    out
                }
                is TensorIndexer.IndexSelect -> {
                    val (indexes) = indexer
                    if (indexes.rank() != 1) {
                        bail("multi-dimensional tensor indexing is not supported")
                    }
                    val out1 =
                        when (
                            val tryResult =
                                x.indexSelect(
                                    when (val tryResult = indexes.toDevice(x.device())) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    },
                                    currentDim,
                                )
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    currentDim += 1
                    out1
                }
                is TensorIndexer.Err -> {
                    val (e) = indexer
                    bail("indexing error {e:?}")
                }
            }
    }
    return Ok(x)
}

/**
 * Generic structure used to index a slice of the tensor
 */
public sealed class TensorIndexer : From<Int> {
    /**
     * This selects the elements for which an index has some specific value.
     */
    public data class Select(
        public var value: Int,
    ) : TensorIndexer()

    /**
     * This is a regular slice, purely indexing a chunk of the tensor
     */
    public class Narrow internal constructor(
        field0CtorArg: Bound<Int>,
        field1CtorArg: Bound<Int>,
    ) : TensorIndexer() {
        public var field0: Bound<Int> = field0CtorArg
            internal set
        public var field1: Bound<Int> = field1CtorArg
            internal set

        public operator fun component1(): Bound<Int> = this.field0

        public operator fun component2(): Bound<Int> = this.field1

        public fun copy(field0: Bound<Int> = this.field0, field1: Bound<Int> = this.field1): Narrow = Narrow(field0CtorArg = field0, field1CtorArg = field1)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Narrow) return false
            if (this.field0 != other.field0) return false
            if (this.field1 != other.field1) return false
            return true
        }

        override fun hashCode(): Int {
            var result = this.field0.hashCode()
            result = 31 * result + this.field1.hashCode()
            return result
        }

        public companion object {
            public operator fun invoke(
                field0: Bound<Int>,
                field1: Bound<Int>,
            ): Narrow =
                Narrow(
                    field0CtorArg = field0,
                    field1CtorArg = field1,
                )
        }
    }

    /**
     * Indexing via a 1d tensor
     */
    public data class IndexSelect(
        public var value: Tensor,
    ) : TensorIndexer()
    public data class Err(
        public var value: Error,
    ) : TensorIndexer()

    public companion object {
        public fun from(index: ULong): TensorIndexer = TensorIndexer.Select(index)

        public fun from(index: List<UInt>): TensorIndexer =
            run {
                val matchVal = Tensor.new(index, Device.Cpu)
                when (matchVal) {
                    is Result.Ok -> {
                        val tensor = matchVal.value
                        TensorIndexer.IndexSelect(tensor)
                    }
                    is Result.Err -> {
                        val e = matchVal.value
                        TensorIndexer.Err(e)
                    }
                }
            }

        public fun fromVecUInt(index: List<UInt>): TensorIndexer {
            val len = index.size
            return run {
                val matchVal = Tensor.fromVec(index, len, Device.Cpu)
                when (matchVal) {
                    is Result.Ok -> {
                        val tensor = matchVal.value
                        TensorIndexer.IndexSelect(tensor)
                    }
                    is Result.Err -> {
                        val e = matchVal.value
                        TensorIndexer.Err(e)
                    }
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
    public fun i(index: T): Result<Tensor, Error>
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
public fun Tensor.i(index: TensorIndexer): Result<Tensor, Error> = this.index(listOf(index))

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
public fun Tensor.iA(tupleArg0: TensorIndexer): Result<Tensor, Error> = this.index(listOf(a.into()))

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
public fun Tensor.i(tupleArg0: Pair<TensorIndexer, TensorIndexer>): Result<Tensor, Error> = this.index(listOf(a.into(), b.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.i(tupleArg0: Triple<TensorIndexer, TensorIndexer, TensorIndexer>): Result<Tensor, Error> = this.index(listOf(A.into(), B.into(), C.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.i(tupleArg0: List<Any?>): Result<Tensor, Error> = this.index(listOf(A.into(), B.into(), C.into(), D.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDE(tupleArg0: List<Any?>): Result<Tensor, Error> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDEF(tupleArg0: List<Any?>): Result<Tensor, Error> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into(), F.into()))

/**
 * see [TensorIndex#method.i]
 */
public fun Tensor.iABCDEFG(tupleArg0: List<Any?>): Result<Tensor, Error> = this.index(listOf(A.into(), B.into(), C.into(), D.into(), E.into(), F.into(), G.into()))
