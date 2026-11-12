// port-lint: source streaming.rs
package io.github.kotlinmania.candle.streaming

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor

// StreamTensror useful for streaming ops.

public interface Dim : io.github.kotlinmania.candle.shape.Dim

/**
 * A stream tensor is used in streaming module. It can either contain an actual tensor or be
 * empty.
 */
public open class StreamTensor(
    public var value: Tensor?,
) : From<Tensor?> {
    public var field0: Tensor?
        get() = value
        set(v) {
            value = v
        }

    public open fun fmt(f: StringBuilder) {
        when (val matchVal = this.value) {
            null -> f.append("Empty")
            else -> {
                val t = matchVal
                f.append("${t.shape()}")
            }
        }
    }

    public open fun shape(): Shape? = this.value?.let({ t -> t.shape() })

    public open fun <D : Dim> cat2(rhs: StreamTensor, dim: D): Result<StreamTensor> {
        val xs3 =
            run {
                val matchVal = Pair(this.value, rhs.value)
                when {
                    matchVal.first != null && matchVal.second != null -> {
                        val lhs = matchVal.first
                        val rhs1 = matchVal.second
                        val xs = Tensor.cat(listOf(lhs, rhs1), dim).getOrElse { return Result.failure(it) }
                        xs
                    }
                    matchVal.first != null && matchVal.second == null -> {
                        val xs1 = matchVal.first
                        xs1.clone()
                    }
                    matchVal.first == null && matchVal.second != null -> {
                        val xs2 = matchVal.second
                        xs2.clone()
                    }
                    else -> null
                }
            }
        return Result.success(StreamTensor(xs3))
    }

    public open fun <D : Dim> seqLen(dim: D): Result<Int> =
        when (val matchVal = this.value) {
            null -> Result.success(0)
            else -> {
                val v = matchVal
                v.dim(dim)
            }
        }

    public open fun reset() {
        this.value = null
    }

    public open fun <D : Dim> narrow(dim: D, offset: Int, len: Int): Result<StreamTensor> {
        val t2 =
            when (val matchVal = this.value) {
                null -> null
                else -> {
                    val t = matchVal
                    val seqLen = t.dim(dim).getOrElse { return Result.failure(it) }
                    if (seqLen <= offset) {
                        null
                    } else {
                        val t1 = t.narrow(dim, offset, Int.min(len, seqLen - offset)).getOrElse { return Result.failure(it) }
                        t1
                    }
                }
            }
        return Result.success(StreamTensor(t2))
    }

    /**
     * Splits the Streaming Tensor on the time axis `dim` with the first `lhsLen` elements
     * returned in the first output and the remaining in the second output.
     */
    public open fun <D : Dim> split(dim: D, lhsLen: Int): Result<Pair<StreamTensor, StreamTensor>> {
        return when (val matchVal = this.value) {
            null -> Result.success(Pair(StreamTensor.empty(), StreamTensor.empty()))
            else -> {
                val t = matchVal
                val seqLen = t.dim(dim).getOrElse { return Result.failure(it) }
                val lhsLen1 = Int.min(seqLen, lhsLen)
                if (lhsLen1 == 0) {
                    Result.success(Pair(StreamTensor.empty(), t.clone().into()))
                } else {
                    val lhs = StreamTensor.fromTensor(t.narrow(dim, 0, lhsLen1).getOrElse { return Result.failure(it) })
                    val rhsLen = seqLen - lhsLen1
                    val rhs =
                        if (rhsLen == 0) {
                            StreamTensor.empty()
                        } else {
                            StreamTensor.fromTensor(t.narrow(dim, lhsLen1, rhsLen).getOrElse { return Result.failure(it) })
                        }
                    Result.success(Pair(lhs, rhs))
                }
            }
        }
    }

    public open fun asOption(): Tensor? = this.value

    public open fun <M : Module> apply(m: M): Result<StreamTensor> {
        return when (val matchVal = this.value) {
            null -> Result.success(StreamTensor.empty())
            else -> {
                val t = matchVal
                Result.success(StreamTensor.fromTensor(t.apply(m).getOrElse { return Result.failure(it) }))
            }
        }
    }

    public companion object {
        public fun from(value: Tensor?): StreamTensor = StreamTensor(value)

        public fun fromTensor(value: Tensor): StreamTensor = StreamTensor(value)

        public fun from(value: Unit): StreamTensor = StreamTensor(null)

        public fun empty(): StreamTensor = StreamTensor(null)
    }
}

/**
 * Streaming modules take as input a stream tensor and return a stream tensor. They may perform
 * some internal buffering so that enough data has been received for the module to be able to
 * perform some operations.
 */
public interface StreamingModule {
    public fun step(xs: StreamTensor): Result<StreamTensor>

    public fun resetState()
}

public sealed class BinOp {
    public data object Add : BinOp()
    public data object Mul : BinOp()
    public data object Sub : BinOp()
    public data object Div : BinOp()

    public fun __rustEq(other: BinOp): Boolean = this == other
}

public open class StreamingBinOp(
    public var prevLhs: StreamTensor,
    public var prevRhs: StreamTensor,
    public var op: BinOp,
    public var dim: D,
) {
    public open fun resetState() {
        this.prevLhs.reset()
        this.prevRhs.reset()
    }

    public open fun forward(lhs: Tensor, rhs: Tensor): Result<Tensor> =
        when (this.op) {
            BinOp.Add -> Tensor.add(lhs, rhs)
            BinOp.Mul -> Tensor.mul(lhs, rhs)
            BinOp.Sub -> Tensor.sub(lhs, rhs)
            BinOp.Div -> Tensor.div(lhs, rhs)
        }

    public open fun step(lhs: StreamTensor, rhs: StreamTensor): Result<StreamTensor> {
        val lhs1 = StreamTensor.cat2(this.prevLhs, lhs, this.dim).getOrElse { return Result.failure(it) }
        val rhs1 = StreamTensor.cat2(this.prevRhs, rhs, this.dim).getOrElse { return Result.failure(it) }
        val lhsLen = lhs1.seqLen(this.dim).getOrElse { return Result.failure(it) }
        val rhsLen = rhs1.seqLen(this.dim).getOrElse { return Result.failure(it) }
        val commonLen = Int.min(lhsLen, rhsLen)
        val (lhs2, prevLhs) = lhs1.split(this.dim, commonLen).getOrElse { return Result.failure(it) }
        val (rhs2, prevRhs) = rhs1.split(this.dim, commonLen).getOrElse { return Result.failure(it) }
        val ys1 =
            run {
                val matchVal = Pair(lhs2.first, rhs2.first)
                when {
                    matchVal.first != null && matchVal.second != null -> {
                        val lhs3 = matchVal.first
                        val rhs3 = matchVal.second
                        val ys = this.forward(lhs3, rhs3).getOrElse { return Result.failure(it) }
                        StreamTensor.fromTensor(ys)
                    }
                    matchVal.first == null && matchVal.second == null -> StreamTensor.empty()
                    else -> {
                        val lhs4 = matchVal.first
                        val rhs4 = matchVal.second
                        bail("INTERNAL ERROR inconsistent lhs and rhs {lhs:?} {rhs:?}")
                    }
                }
            }
        this.prevLhs = prevLhs
        this.prevRhs = prevRhs
        return Result.success(ys1)
    }

    public companion object {
        public fun new(op: BinOp, dim: D): StreamingBinOp = StreamingBinOp(prevLhs = StreamTensor.empty(), prevRhs = StreamTensor.empty(), op = op, dim = dim)
    }
}

/**
 * Simple wrapper that doesn't do any buffering.
 */
public open class Map<T : Module>(
    public var value: T,
) : StreamingModule {
    public var field0: T
        get() = value
        set(v) {
            value = v
        }

    public override fun resetState() {}

    public override fun step(xs: StreamTensor): Result<StreamTensor> = xs.apply(this.value)
}
