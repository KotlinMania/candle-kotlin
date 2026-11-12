// port-lint: source streaming.rs
package io.github.kotlinmania.candle.streaming

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.fromResidual

// StreamTensror useful for streaming ops.

public interface Dim : io.github.kotlinmania.candle.shape.Dim

/**
 * A stream tensor is used in streaming module. It can either contain an actual tensor or be
 * empty.
 */
public class StreamTensor(
    public var value: Tensor? = null,
) : From<Tensor?> {
    public var field0: Tensor?
        get() = value
        set(v) {
            value = v
        }

    public fun fmt(f: StringBuilder) {
        when (val matchVal = this.value) {
            null -> f.append("Empty")
            else -> {
                val t = matchVal
                f.append("${t.shape()}")
            }
        }
    }

    public fun shape(): Shape? = this.value?.let({ t -> t.shape() })

    public fun <D : Dim> cat2(rhs: StreamTensor, dim: D): Result<StreamTensor> {
        val xs3 =
            run {
                val matchVal = Pair(this.value, rhs.value)
                when {
                    matchVal.first != null && matchVal.second != null -> {
                        val lhs = matchVal.first
                        val rhs1 = matchVal.second
                        val xs =
                            when (val tryResult = Tensor.cat(listOf(lhs, rhs1), dim)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
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
        return Ok(StreamTensor(xs3))
    }

    public fun <D : Dim> seqLen(dim: D): Result<Int> =
        when (val matchVal = this.value) {
            null -> Ok(0)
            else -> {
                val v = matchVal
                v.dim(dim)
            }
        }

    public fun reset() {
        this.value = null
    }

    public fun <D : Dim> narrow(dim: D, offset: Int, len: Int): Result<StreamTensor> {
        val t2 =
            when (val matchVal = this.value) {
                null -> null
                else -> {
                    val t = matchVal
                    val seqLen =
                        when (val tryResult = t.dim(dim)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    if (seqLen <= offset) {
                        null
                    } else {
                        val t1 =
                            when (val tryResult = t.narrow(dim, offset, Int.min(len, seqLen - offset))) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        t1
                    }
                }
            }
        return Ok(StreamTensor(t2))
    }

    /**
     * Splits the Streaming Tensor on the time axis `dim` with the first `lhsLen` elements
     * returned in the first output and the remaining in the second output.
     */
    public fun <D : Dim> split(dim: D, lhsLen: Int): Result<Pair<StreamTensor, StreamTensor>> {
        return when (val matchVal = this.value) {
            null -> Ok(Pair(StreamTensor.empty(), StreamTensor.empty()))
            else -> {
                val t = matchVal
                val seqLen =
                    when (val tryResult = t.dim(dim)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val lhsLen1 = Int.min(seqLen, lhsLen)
                if (lhsLen1 == 0) {
                    Ok(Pair(StreamTensor.empty(), t.clone().into()))
                } else {
                    val lhs =
                        StreamTensor.fromTensor(
                            when (val tryResult = t.narrow(dim, 0, lhsLen1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        )
                    val rhsLen = seqLen - lhsLen1
                    val rhs =
                        if (rhsLen == 0) {
                            StreamTensor.empty()
                        } else {
                            StreamTensor.fromTensor(
                                when (val tryResult = t.narrow(dim, lhsLen1, rhsLen)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        }
                    Ok(Pair(lhs, rhs))
                }
            }
        }
    }

    public fun asOption(): Tensor? = this.value

    public fun <M : Module> apply(m: M): Result<StreamTensor> {
        return when (val matchVal = this.value) {
            null -> Ok(StreamTensor.empty())
            else -> {
                val t = matchVal
                Ok(
                    StreamTensor.fromTensor(
                        when (val tryResult = t.apply(m)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
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

public enum class BinOp(
    public val value: UInt,
) {
    Add(0u),
    Mul(1u),
    Sub(2u),
    Div(3u),
}

public data class StreamingBinOp(
    public var prevLhs: StreamTensor = StreamTensor(),
    public var prevRhs: StreamTensor = StreamTensor(),
    public var op: BinOp = BinOp.entries.first(),
    public var dim: D = D(),
) {
    public fun resetState() {
        this.prevLhs.reset()
        this.prevRhs.reset()
    }

    public fun forward(lhs: Tensor, rhs: Tensor): Result<Tensor> =
        when (this.op) {
            BinOp.Add -> Tensor.add(lhs, rhs)
            BinOp.Mul -> Tensor.mul(lhs, rhs)
            BinOp.Sub -> Tensor.sub(lhs, rhs)
            BinOp.Div -> Tensor.div(lhs, rhs)
        }

    public fun step(lhs: StreamTensor, rhs: StreamTensor): Result<StreamTensor> {
        val lhs1 =
            when (val tryResult = StreamTensor.cat2(this.prevLhs, lhs, this.dim)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val rhs1 =
            when (val tryResult = StreamTensor.cat2(this.prevRhs, rhs, this.dim)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lhsLen =
            when (val tryResult = lhs1.seqLen(this.dim)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val rhsLen =
            when (val tryResult = rhs1.seqLen(this.dim)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val commonLen = Int.min(lhsLen, rhsLen)
        val (lhs2, prevLhs) =
            when (val tryResult = lhs1.split(this.dim, commonLen)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val (rhs2, prevRhs) =
            when (val tryResult = rhs1.split(this.dim, commonLen)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val ys1 =
            run {
                val matchVal = Pair(lhs2.first, rhs2.first)
                when {
                    matchVal.first != null && matchVal.second != null -> {
                        val lhs3 = matchVal.first
                        val rhs3 = matchVal.second
                        val ys =
                            when (val tryResult = this.forward(lhs3, rhs3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
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
        return Ok(ys1)
    }

    public companion object {
        public fun new(op: BinOp, dim: D): StreamingBinOp = StreamingBinOp(prevLhs = StreamTensor.empty(), prevRhs = StreamTensor.empty(), op = op, dim = dim)
    }
}

/**
 * Simple wrapper that doesn't do any buffering.
 */
public class Map<T : Module>(
    public var value: T = T(),
) : StreamingModule {
    public var field0: T
        get() = value
        set(v) {
            value = v
        }

    public override fun resetState() {}

    public override fun step(xs: StreamTensor): Result<StreamTensor> = xs.apply(this.value)

    public companion object {
        public fun <T : Module> typeLayout(typeLayoutT: io.github.kotlinmania.rustlang.core.mem.TypeLayout<T>): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Map<T>> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Map<T>> {
                override val alignBytes: Long = listOf(1L, typeLayoutT.alignBytes).max()
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(typeLayoutT.sizeBytes, alignBytes)
                override fun readNative(addr: Long): Map<T> =
                    Map(
                        (typeLayoutT).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Map<T>) {
                    (typeLayoutT).writeNative(addr, value.value)
                }
            }
    }
}
