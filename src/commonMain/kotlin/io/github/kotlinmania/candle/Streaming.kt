// port-lint: source streaming.rs
package io.github.kotlinmania.candle

/**
 * A stream tensor is used in streaming module. It can either contain an actual tensor or be
 * empty.
 */
public class StreamTensor(
    public var tensor: Tensor? = null,
) {
    override fun toString(): String =
        when (val t = tensor) {
            null -> "Empty"
            else -> t.shape().toString()
        }

    public fun shape(): Shape? = tensor?.shape()

    public fun cat2(rhs: StreamTensor, dim: Int): StreamTensor {
        val l = tensor
        val r = rhs.tensor
        val xs = when {
            l != null && r != null -> Tensor.cat(listOf(l, r), dim)
            l != null -> l
            r != null -> r
            else -> null
        }
        return StreamTensor(xs)
    }

    public fun cat2(rhs: StreamTensor, dim: ShapeDim): StreamTensor {
        val l = tensor
        val r = rhs.tensor
        val dimIdx = (l?.shape() ?: r?.shape())?.let { dim.toIndex(it, "cat") } ?: 0
        return cat2(rhs, dimIdx)
    }

    public fun seqLen(dim: Int): Int = tensor?.dim(dim) ?: 0

    public fun seqLen(dim: ShapeDim): Int = tensor?.dim(dim) ?: 0

    public fun reset() {
        tensor = null
    }

    public fun narrow(dim: Int, offset: Int, len: Int): StreamTensor {
        val t = tensor ?: return StreamTensor(null)
        val seqLen = t.dim(dim)
        return if (seqLen <= offset) {
            StreamTensor(null)
        } else {
            val narrowed = t.narrow(dim, offset, minOf(len, seqLen - offset))
            StreamTensor(narrowed)
        }
    }

    public fun narrow(dim: ShapeDim, offset: Int, len: Int): StreamTensor {
        val t = tensor ?: return StreamTensor(null)
        val dimIdx = dim.toIndex(t.shape(), "narrow")
        return narrow(dimIdx, offset, len)
    }

    /**
     * Splits the Streaming Tensor on the time axis dim with the first lhsLen elements
     * returned in the first output and the remaining in the second output.
     */
    public fun split(dim: Int, lhsLen: Int): Pair<StreamTensor, StreamTensor> {
        val t = tensor ?: return Pair(empty(), empty())
        val seqLen = t.dim(dim)
        val actualLhsLen = minOf(seqLen, lhsLen)
        if (actualLhsLen == 0) {
            return Pair(empty(), fromTensor(t))
        }
        val lhs = fromTensor(t.narrow(dim, 0, actualLhsLen))
        val rhsLen = seqLen - actualLhsLen
        val rhs = if (rhsLen == 0) {
            empty()
        } else {
            fromTensor(t.narrow(dim, actualLhsLen, rhsLen))
        }
        return Pair(lhs, rhs)
    }

    public fun split(dim: ShapeDim, lhsLen: Int): Pair<StreamTensor, StreamTensor> {
        val t = tensor ?: return Pair(empty(), empty())
        val dimIdx = dim.toIndex(t.shape(), "split")
        return split(dimIdx, lhsLen)
    }

    public fun asOption(): Tensor? = tensor

    public fun apply(m: Module): StreamTensor =
        when (val t = tensor) {
            null -> empty()
            else -> fromTensor(t.apply(m))
        }

    public companion object {
        public fun empty(): StreamTensor = StreamTensor(null)

        public fun fromTensor(tensor: Tensor): StreamTensor = StreamTensor(tensor)

        public fun from(value: Tensor?): StreamTensor = StreamTensor(value)

        public fun from(value: Tensor): StreamTensor = StreamTensor(value)

        public fun from(): StreamTensor = StreamTensor(null)

        public fun cat2(lhs: StreamTensor, rhs: StreamTensor, dim: Int): StreamTensor = lhs.cat2(rhs, dim)

        public fun cat2(lhs: StreamTensor, rhs: StreamTensor, dim: ShapeDim): StreamTensor = lhs.cat2(rhs, dim)
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
    Div,
}

public class StreamingBinOp(
    public var op: BinOp,
    public var dim: ShapeDim,
) {
    public constructor(op: BinOp, dim: Int) : this(op, ShapeDimIndex(dim))

    private var prevLhs: StreamTensor = StreamTensor.empty()
    private var prevRhs: StreamTensor = StreamTensor.empty()

    public fun resetState() {
        prevLhs.reset()
        prevRhs.reset()
    }

    public fun forward(lhs: Tensor, rhs: Tensor): Tensor =
        when (op) {
            BinOp.Add -> Tensor.add(lhs, rhs)
            BinOp.Mul -> Tensor.mul(lhs, rhs)
            BinOp.Sub -> Tensor.sub(lhs, rhs)
            BinOp.Div -> Tensor.div(lhs, rhs)
        }

    public fun step(lhs: StreamTensor, rhs: StreamTensor): StreamTensor {
        val catLhs = StreamTensor.cat2(prevLhs, lhs, dim)
        val catRhs = StreamTensor.cat2(prevRhs, rhs, dim)
        val lhsLen = catLhs.seqLen(dim)
        val rhsLen = catRhs.seqLen(dim)
        val commonLen = minOf(lhsLen, rhsLen)
        val (newLhs, newPrevLhs) = catLhs.split(dim, commonLen)
        val (newRhs, newPrevRhs) = catRhs.split(dim, commonLen)
        val l = newLhs.asOption()
        val r = newRhs.asOption()
        val ys = if (l != null && r != null) {
            val out = forward(l, r)
            StreamTensor.fromTensor(out)
        } else if (l == null && r == null) {
            StreamTensor.empty()
        } else {
            bail("INTERNAL ERROR inconsistent lhs and rhs $newLhs $newRhs")
        }
        prevLhs = newPrevLhs
        prevRhs = newPrevRhs
        return ys
    }

    public companion object {
        public fun new(op: BinOp, dim: ShapeDim): StreamingBinOp = StreamingBinOp(op, dim)

        public fun new(op: BinOp, dim: Int): StreamingBinOp = StreamingBinOp(op, dim)
    }
}

/**
 * Simple wrapper that doesn't do any buffering.
 */
public class Map<T : Module>(
    public val module: T,
) : StreamingModule {
    override fun resetState() {}

    override fun step(xs: StreamTensor): StreamTensor = xs.apply(module)
}
