// port-lint: source candle-core/src/shape.rs
package io.github.kotlinmania.candle

public data class MatmulBroadcastShapes(
    val lhs: Shape,
    val rhs: Shape,
)

public data class Dims2(
    val d0: Int,
    val d1: Int,
)

public data class Dims3(
    val d0: Int,
    val d1: Int,
    val d2: Int,
)

public data class Dims4(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
)

public data class Dims5(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
    val d4: Int,
)

public data class Shape(
    val dims: List<Int>,
) {
    public constructor(vararg dims: Int) : this(dims.toList())

    public val rank: Int get() = dims.size

    public fun intoDims(): List<Int> = dims

    public fun dim(dimIndex: Int): Int {
        val idx = toIndex(dimIndex, this, "dim")
        return dims[idx]
    }

    public fun elemCount(): Int {
        if (dims.isEmpty()) return 1
        var prod = 1
        for (d in dims) {
            prod *= d
        }
        return prod
    }

    public fun strideContiguous(): List<Int> {
        if (dims.isEmpty()) return emptyList()
        val result = ArrayList<Int>(dims.size)
        var acc = 1
        for (i in dims.indices.reversed()) {
            result.add(acc)
            acc *= dims[i]
        }
        result.reverse()
        return result
    }

    public fun isContiguous(stride: List<Int>): Boolean {
        if (dims.size != stride.size) return false
        var acc = 1
        for (i in dims.indices.reversed()) {
            val dim = dims[i]
            val s = stride[i]
            if (dim > 1 && s != acc) {
                return false
            }
            acc *= dim
        }
        return true
    }

    public fun isFortranContiguous(stride: List<Int>): Boolean {
        if (dims.size != stride.size) return false
        var acc = 1
        for (i in dims.indices) {
            val dim = dims[i]
            val s = stride[i]
            if (dim > 1 && s != acc) {
                return false
            }
            acc *= dim
        }
        return true
    }

    public fun extend(additionalDims: List<Int>): Shape = Shape(dims + additionalDims)

    public fun broadcastShapeBinaryOp(rhs: Shape, op: String): Shape {
        val lhsDims = dims
        val rhsDims = rhs.dims
        val lhsNdims = lhsDims.size
        val rhsNdims = rhsDims.size
        val bcastNdims = maxOf(lhsNdims, rhsNdims)
        val bcastDims = ArrayList<Int>(bcastNdims)
        for (idx in 0 until bcastNdims) {
            val revIdx = bcastNdims - idx
            val lValue = if (lhsNdims < revIdx) 1 else lhsDims[lhsNdims - revIdx]
            val rValue = if (rhsNdims < revIdx) 1 else rhsDims[rhsNdims - revIdx]
            val bcastValue =
                if (lValue == rValue) {
                    lValue
                } else if (lValue == 1) {
                    rValue
                } else if (rValue == 1) {
                    lValue
                } else {
                    throw CandleException.ShapeMismatchBinaryOp(this, rhs, op)
                }
            bcastDims.add(bcastValue)
        }
        return Shape(bcastDims)
    }

    public fun broadcastShapeMatmul(rhs: Shape): MatmulBroadcastShapes {
        val lhsDims = dims
        val rhsDims = rhs.dims
        if (lhsDims.size < 2 || rhsDims.size < 2) {
            bail("only 2d matrixes are supported $this $rhs")
        }
        val m = lhsDims[lhsDims.size - 2]
        val lhsK = lhsDims[lhsDims.size - 1]
        val rhsK = rhsDims[rhsDims.size - 2]
        val n = rhsDims[rhsDims.size - 1]
        if (lhsK != rhsK) {
            bail("different inner dimensions in broadcast matmul $this $rhs")
        }

        val lhsB = Shape(lhsDims.subList(0, lhsDims.size - 2))
        val rhsB = Shape(rhsDims.subList(0, rhsDims.size - 2))
        val bcast = lhsB.broadcastShapeBinaryOp(rhsB, "broadcast_matmul")
        val bcastDims = bcast.dims

        val bcastLhs = bcastDims + listOf(m, lhsK)
        val bcastRhs = bcastDims + listOf(rhsK, n)
        return MatmulBroadcastShapes(Shape(bcastLhs), Shape(bcastRhs))
    }

    public fun dims0() {
        if (dims.isNotEmpty()) {
            throw CandleException.UnexpectedNumberOfDims(0, dims.size, this)
        }
    }

    public fun dims1(): Int {
        if (dims.size != 1) {
            throw CandleException.UnexpectedNumberOfDims(1, dims.size, this)
        }
        return dims[0]
    }

    public fun dims2(): Dims2 {
        if (dims.size != 2) {
            throw CandleException.UnexpectedNumberOfDims(2, dims.size, this)
        }
        return Dims2(dims[0], dims[1])
    }

    public fun dims3(): Dims3 {
        if (dims.size != 3) {
            throw CandleException.UnexpectedNumberOfDims(3, dims.size, this)
        }
        return Dims3(dims[0], dims[1], dims[2])
    }

    public fun dims4(): Dims4 {
        if (dims.size != 4) {
            throw CandleException.UnexpectedNumberOfDims(4, dims.size, this)
        }
        return Dims4(dims[0], dims[1], dims[2], dims[3])
    }

    public fun dims5(): Dims5 {
        if (dims.size != 5) {
            throw CandleException.UnexpectedNumberOfDims(5, dims.size, this)
        }
        return Dims5(dims[0], dims[1], dims[2], dims[3], dims[4])
    }

    override fun toString(): String = dims.toString()

    public companion object {
        public val SCALAR: Shape = Shape(emptyList())

        public fun fromDims(dims: List<Int>): Shape = Shape(dims)

        public fun fromDims(vararg dims: Int): Shape = Shape(dims.toList())

        public fun toIndex(dim: Int, shape: Shape, op: String): Int {
            val rank = shape.rank
            val normalized = if (dim < 0) rank + dim else dim
            if (normalized < 0 || normalized >= rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            return normalized
        }

        public fun toIndexPlusOne(dim: Int, shape: Shape, op: String): Int {
            val rank = shape.rank
            val normalized = if (dim < 0) rank + 1 + dim else dim
            if (normalized < 0 || normalized > rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            return normalized
        }
    }
}

public sealed interface ShapeDim {
    public fun toIndex(shape: Shape, op: String): Int
    public fun toIndexPlusOne(shape: Shape, op: String): Int
}

public object ShapeDimMinus1 : ShapeDim {
    override fun toIndex(shape: Shape, op: String): Int =
        if (shape.rank >= 1) shape.rank - 1 else throw CandleException.DimOutOfRange(shape, -1, op)

    override fun toIndexPlusOne(shape: Shape, op: String): Int = shape.rank
}

public object ShapeDimMinus2 : ShapeDim {
    override fun toIndex(shape: Shape, op: String): Int =
        if (shape.rank >= 2) shape.rank - 2 else throw CandleException.DimOutOfRange(shape, -2, op)

    override fun toIndexPlusOne(shape: Shape, op: String): Int =
        if (shape.rank >= 1) shape.rank - 1 else throw CandleException.DimOutOfRange(shape, -2, op)
}

public data class ShapeDimMinus(
    val u: Int,
) : ShapeDim {
    override fun toIndex(shape: Shape, op: String): Int =
        if (u > 0 && shape.rank >= u) shape.rank - u else throw CandleException.DimOutOfRange(shape, -u, op)

    override fun toIndexPlusOne(shape: Shape, op: String): Int =
        if (u > 0 && shape.rank + 1 >= u) shape.rank + 1 - u else throw CandleException.DimOutOfRange(shape, -u, op)
}

public fun List<Int>.intoShapeWithHole(elCount: Int): Shape {
    var holeIdx = -1
    var knownProduct = 1
    for (i in indices) {
        val d = this[i]
        if (d == -1) {
            if (holeIdx != -1) {
                bail("multiple holes in shape: $this")
            }
            holeIdx = i
        } else {
            if (d < 0) bail("invalid dimension $d in shape $this")
            knownProduct *= d
        }
    }
    if (holeIdx == -1) {
        if (knownProduct != elCount && elCount > 0) {
            bail("cannot reshape tensor with $elCount elements to $this")
        }
        return Shape(this)
    }
    if (knownProduct == 0 || elCount % knownProduct != 0) {
        bail("cannot reshape tensor with $elCount elements to $this")
    }
    val holeSize = elCount / knownProduct
    val result = toMutableList()
    result[holeIdx] = holeSize
    return Shape(result)
}
