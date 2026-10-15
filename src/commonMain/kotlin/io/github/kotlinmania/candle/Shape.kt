// port-lint: source shape.rs
package io.github.kotlinmania.candle

/**
 * The shape of a tensor is a tuple with the size of each of its dimensions.
 */

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

    /**
     * The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
     */
    public val rank: Int get() = dims.size

    public fun rank(): Int = dims.size

    public fun intoDims(): List<Int> = dims

    /**
     * The dimensions as a slice of integers.
     */
    public fun dims(): List<Int> = dims

    /**
     * The dimension size for a specified dimension index.
     */
    public fun dim(dimIndex: Int): Int {
        val idx = toIndex(dimIndex, this, "dim")
        return dims[idx]
    }

    public fun dim(dim: Dim): Int {
        val idx = dim.toIndex(this, "dim")
        return dims[idx]
    }

    /**
     * The total number of elements, this is the product of all dimension sizes.
     */
    public fun elemCount(): Int {
        if (dims.isEmpty()) return 1
        var prod = 1
        for (d in dims) {
            prod *= d
        }
        return prod
    }

    /**
     * The strides given in number of elements for a contiguous n-dimensional
     * arrays using this shape.
     */
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

    /**
     * Returns true if the strides are C contiguous (aka row major).
     */
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

    /**
     * Returns true if the strides are Fortran contiguous (aka column major).
     */
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

    /**
     * Modifies the shape by adding a list of additional dimensions at the end of the existing
     * dimensions.
     */
    public fun extend(additionalDims: List<Int>): Shape = Shape(dims + additionalDims)

    /**
     * Check whether the two shapes are compatible for broadcast, and if it is the case return the
     * broadcasted shape. This is to be used for binary pointwise ops.
     */
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

    public fun fmt(): String = toString()

    override fun toString(): String = dims.toString()

    public companion object {
        public val SCALAR: Shape = Shape(emptyList())

        public fun from(dims: List<Int>): Shape = Shape(dims)
        public fun from(vararg dims: Int): Shape = Shape(dims.toList())
        public fun from(shape: Shape): Shape = Shape(shape.dims)
        public fun from(d1: Int): Shape = Shape(listOf(d1))
        public fun from(d1: Int, d2: Int): Shape = Shape(listOf(d1, d2))
        public fun from(d1: Int, d2: Int, d3: Int): Shape = Shape(listOf(d1, d2, d3))
        public fun from(d1: Int, d2: Int, d3: Int, d4: Int): Shape = Shape(listOf(d1, d2, d3, d4))
        public fun from(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int): Shape = Shape(listOf(d1, d2, d3, d4, d5))
        public fun from(d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int): Shape = Shape(listOf(d1, d2, d3, d4, d5, d6))

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

public interface Dim {
    public fun toIndex(shape: Shape, op: String): Int
    public fun toIndexPlusOne(shape: Shape, op: String): Int
}

public class DimInt(public val dim: Int) : Dim {
    override fun toIndex(shape: Shape, op: String): Int = Shape.toIndex(dim, shape, op)
    override fun toIndexPlusOne(shape: Shape, op: String): Int = Shape.toIndexPlusOne(dim, shape, op)
}

public sealed class D : Dim {
    public object Minus1 : D() {
        override fun toIndex(shape: Shape, op: String): Int =
            if (shape.rank >= 1) shape.rank - 1 else throw outOfRange(shape, op)

        override fun toIndexPlusOne(shape: Shape, op: String): Int = shape.rank
    }

    public object Minus2 : D() {
        override fun toIndex(shape: Shape, op: String): Int =
            if (shape.rank >= 2) shape.rank - 2 else throw outOfRange(shape, op)

        override fun toIndexPlusOne(shape: Shape, op: String): Int =
            if (shape.rank >= 1) shape.rank - 1 else throw outOfRange(shape, op)
    }

    public data class Minus(val u: Int) : D() {
        override fun toIndex(shape: Shape, op: String): Int =
            if (u > 0 && shape.rank >= u) shape.rank - u else throw outOfRange(shape, op)

        override fun toIndexPlusOne(shape: Shape, op: String): Int =
            if (u > 0 && shape.rank + 1 >= u) shape.rank + 1 - u else throw outOfRange(shape, op)
    }

    public fun outOfRange(shape: Shape, op: String): CandleException {
        val dim = when (this) {
            is Minus1 -> -1
            is Minus2 -> -2
            is Minus -> -u
        }
        return CandleException.DimOutOfRange(shape, dim, op)
    }
}

public interface Dims {
    public fun toIndexesInternal(shape: Shape, op: String): List<Int>

    public fun toIndexes(shape: Shape, op: String): List<Int> {
        val dims = toIndexesInternal(shape, op)
        for (i in dims.indices) {
            val dim = dims[i]
            if (dims.subList(0, i).contains(dim)) {
                throw CandleException.DuplicateDimIndex(shape, dims, op)
            }
            if (dim >= shape.rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
        }
        return dims
    }
}

public class DimsList(public val dims: List<Int>) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> = dims
}

public class DimsArray(public val dims: IntArray) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> = dims.toList()
}

public class DimsSlice(public val dims: IntArray) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> = dims.toList()
}

public object DimsEmpty : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> = emptyList()
}

public class DimsDim(public val dim: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(dim.toIndex(shape, op))
}

public class DimsTuple1(public val d1: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(d1.toIndex(shape, op))
}

public class DimsTuple2(public val d1: Dim, public val d2: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(d1.toIndex(shape, op), d2.toIndex(shape, op))
}

public class DimsTuple3(public val d1: Dim, public val d2: Dim, public val d3: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(d1.toIndex(shape, op), d2.toIndex(shape, op), d3.toIndex(shape, op))
}

public class DimsTuple4(public val d1: Dim, public val d2: Dim, public val d3: Dim, public val d4: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(d1.toIndex(shape, op), d2.toIndex(shape, op), d3.toIndex(shape, op), d4.toIndex(shape, op))
}

public class DimsTuple5(
    public val d1: Dim,
    public val d2: Dim,
    public val d3: Dim,
    public val d4: Dim,
    public val d5: Dim,
) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(
            d1.toIndex(shape, op),
            d2.toIndex(shape, op),
            d3.toIndex(shape, op),
            d4.toIndex(shape, op),
            d5.toIndex(shape, op),
        )
}

public class DimsTuple6(
    public val d1: Dim,
    public val d2: Dim,
    public val d3: Dim,
    public val d4: Dim,
    public val d5: Dim,
    public val d6: Dim,
) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> =
        listOf(
            d1.toIndex(shape, op),
            d2.toIndex(shape, op),
            d3.toIndex(shape, op),
            d4.toIndex(shape, op),
            d5.toIndex(shape, op),
            d6.toIndex(shape, op),
        )
}

public interface ShapeWithOneHole {
    public fun intoShape(elCount: Int): Shape
}

public class ShapeWithOneHoleShape(public val shape: Shape) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = shape
}

public object ShapeWithOneHole1 : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(elCount)
}

public fun holeSize(elCount: Int, prodD: Int, s: Any): Int {
    if (prodD == 0) {
        bail("cannot reshape tensor of $elCount elements to $s")
    }
    if (elCount % prodD != 0) {
        bail("cannot reshape tensor with $elCount elements to $s")
    }
    return elCount / prodD
}

public data class ShapeWithOneHole2Hole0(public val d1: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(holeSize(elCount, d1, this), d1)
}

public data class ShapeWithOneHole2Hole1(public val d1: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, holeSize(elCount, d1, this))
}

public data class ShapeWithOneHole3Hole0(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(holeSize(elCount, d1 * d2, this), d1, d2)
}

public data class ShapeWithOneHole3Hole1(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, holeSize(elCount, d1 * d2, this), d2)
}

public data class ShapeWithOneHole3Hole2(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, holeSize(elCount, d1 * d2, this))
}

public data class ShapeWithOneHole4Hole0(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(holeSize(elCount, d1 * d2 * d3, this), d1, d2, d3)
}

public data class ShapeWithOneHole4Hole1(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, holeSize(elCount, d1 * d2 * d3, this), d2, d3)
}

public data class ShapeWithOneHole4Hole2(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, holeSize(elCount, d1 * d2 * d3, this), d3)
}

public data class ShapeWithOneHole4Hole3(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, d3, holeSize(elCount, d1 * d2 * d3, this))
}

public data class ShapeWithOneHole5Hole0(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(holeSize(elCount, d1 * d2 * d3 * d4, this), d1, d2, d3, d4)
}

public data class ShapeWithOneHole5Hole1(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, holeSize(elCount, d1 * d2 * d3 * d4, this), d2, d3, d4)
}

public data class ShapeWithOneHole5Hole2(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, holeSize(elCount, d1 * d2 * d3 * d4, this), d3, d4)
}

public data class ShapeWithOneHole5Hole3(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, d3, holeSize(elCount, d1 * d2 * d3 * d4, this), d4)
}

public data class ShapeWithOneHole5Hole4(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Shape.from(d1, d2, d3, d4, holeSize(elCount, d1 * d2 * d3 * d4, this))
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
    val hSize = elCount / knownProduct
    val result = toMutableList()
    result[holeIdx] = hSize
    return Shape(result)
}
