package io.github.kotlinmania.candle.streaming

import io.github.kotlinmania.candle.*

public interface Dim : ShapeDim

/**
 * StreamTensor useful for streaming ops.
 *
 * A stream tensor is used in streaming module. It can either contain an actual tensor or be
 * empty, supporting incremental concatenation, narrowing, splitting, and binary operations.
 */
// port-lint: source streaming.rs

private val None: Tensor? = null
private val Some: (Tensor) -> Tensor? = { it }
private val Self: (Tensor?) -> StreamTensor = { StreamTensor(it) }
private val asRef: (Tensor?) -> Tensor? = { it }
private val write: (StringBuilder, String, Any?) -> Unit = { f, _, v -> f.append(v.toString()) }
private val usizeMin: (Int, Int) -> Int = { a, b -> minOf(a, b) }
private val Ok: (StreamTensor) -> StreamTensor = { it }

public class StreamTensor(
    public var tensor: Tensor? = null
) {
    private val self: StreamTensor get() = this

    public fun fmt(f: StringBuilder) {
        val t = self.tensor; when (t) {
            null -> write(f, "Empty", "Empty")
            else -> write(f, "{:?}", t.shape())
        }
    }

    public fun shape(): Shape? {
        return asRef(self.tensor)?.let { t -> t.shape() }
    }

    public fun <D : ShapeDim> cat2(rhs: StreamTensor, dim: D): StreamTensor {
        val lhs = self.tensor; val r = rhs.tensor; val xs = when {
            lhs != null && r != null -> {
                val xs = Tensor.cat(listOf(lhs, r), dim)
                Some(xs)
            }
            lhs != null || r != null -> Some((lhs ?: r)!!.clone())
            else -> None
        }
        return Ok(Self(xs))
    }

    public fun <D : ShapeDim> seqLen(dim: D): Int {
        val v = self.tensor; return when (v) {
            null -> 0
            else -> v.dim(dim)
        }
    }

    public fun reset() {
        self.tensor = None
    }

    public fun <D : ShapeDim> narrow(dim: D, offset: Int, len: Int): StreamTensor {
        val cur = self.tensor; val t = when (cur) {
            null -> None
            else -> {
                val seqLen = cur.dim(dim)
                if (seqLen <= offset) {
                    None
                } else {
                    val t = cur.narrow(dim, offset, usizeMin(len, seqLen - offset))
                    Some(t)
                }
            }
        }
        return Ok(Self(t))
    }

    /**
     * Splits the Streaming Tensor on the time axis dim with the first lhsLen elements
     * returned in the first output and the remaining in the second output.
     */
    public fun <D : ShapeDim> split(dim: D, lhsLen: Int): Pair<StreamTensor, StreamTensor> {
        val t = self.tensor; return when (t) {
            null -> Pair(empty(), empty())
            else -> {
                val seqLen = t.dim(dim)
                val lhsLen = usizeMin(seqLen, lhsLen)
                if (lhsLen == 0) {
                    Pair(empty(), from(t.clone()))
                } else {
                    val lhs = fromTensor(t.narrow(dim, 0, lhsLen))
                    val rhsLen = seqLen - lhsLen
                    val rhs = if (rhsLen == 0) {
                        empty()
                    } else {
                        fromTensor(t.narrow(dim, lhsLen, rhsLen))
                    }
                    Pair(lhs, rhs)
                }
            }
        }
    }

    public fun asOption(): Tensor? {
        return asRef(self.tensor)
    }

    public fun <M : Module> apply(m: M): StreamTensor {
        val t = self.tensor; return when (t) {
            null -> Ok(empty())
            else -> Ok(fromTensor(t.apply(m)))
        }
    }

    public companion object {
        public fun from(value: Tensor?): StreamTensor {
            return Self(value)
        }

        public fun from(value: Unit): StreamTensor {
            return if (value == Unit) Self(None) else Self(None)
        }

        public fun empty(): StreamTensor {
            return Self(None)
        }

        public fun fromTensor(tensor: Tensor): StreamTensor {
            return Self(Some(tensor))
        }
    }
}

/**
 * Streaming modules take as input a stream tensor and return a stream tensor. They may perform
 * some internal buffering so that enough data has been received for the module to be able to
 * perform some operations.
 */
public interface StreamingModule {
    public fun step(xs: StreamTensor): StreamTensor
    public fun resetState()
}

public enum class BinOp {
    Add,
    Mul,
    Sub,
    Div
}

public class StreamingBinOp(
    private var prevLhs: StreamTensor,
    private var prevRhs: StreamTensor,
    public var op: BinOp,
    public var dim: D
) {
    public constructor(op: BinOp, dim: D) : this(StreamTensor.empty(), StreamTensor.empty(), op, dim)
    public constructor(op: BinOp, dim: Int) : this(op, D.Custom(dim))
    private val self: StreamingBinOp get() = this

    public fun resetState() {
        self.prevLhs.reset()
        self.prevRhs.reset()
    }

    public fun forward(lhs: Tensor, rhs: Tensor): Tensor {
        return when (self.op) {
            BinOp.Add -> lhs.add(rhs)
            BinOp.Mul -> lhs.mul(rhs)
            BinOp.Sub -> lhs.sub(rhs)
            BinOp.Div -> lhs.div(rhs)
        }
    }

    public fun step(lhs: StreamTensor, rhs: StreamTensor): StreamTensor {
        val lhs = self.prevLhs.cat2(lhs, self.dim)
        val rhs = self.prevRhs.cat2(rhs, self.dim)
        val lhsLen = lhs.seqLen(self.dim)
        val rhsLen = rhs.seqLen(self.dim)
        val commonLen = usizeMin(lhsLen, rhsLen)
        val splitLhs = lhs.split(self.dim, commonLen)
        val splitRhs = rhs.split(self.dim, commonLen)
        val l = splitLhs.first.tensor; val r = splitRhs.first.tensor
        val ys = when {
            l != null && r != null -> {
                val ys = self.forward(l, r)
                StreamTensor.fromTensor(ys)
            }
            l == null && r == null -> StreamTensor.empty()
            else -> bail("INTERNAL ERROR inconsistent lhs and rhs $l $r")
        }
        self.prevLhs = splitLhs.second
        self.prevRhs = splitRhs.second
        return Ok(ys)
    }

    public companion object {
        public fun new(op: BinOp, dim: D): StreamingBinOp {
            return StreamingBinOp(
                prevLhs = StreamTensor.empty(),
                prevRhs = StreamTensor.empty(),
                op = op,
                dim = dim
            )
        }
    }
}

/**
 * Simple wrapper that doesn't do any buffering.
 */
public class Map<T : Module>(
    public val module: T
) : StreamingModule {
    private val self: Map<T> get() = this

    override fun resetState() { self }

    override fun step(xs: StreamTensor): StreamTensor {
        return xs.apply(self.module)
    }
}
