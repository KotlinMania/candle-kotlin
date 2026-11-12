// port-lint: source shape.rs
package io.github.kotlinmania.candle

import kotlin.math.max

/**
 * The shape of a tensor is a tuple with the size of each of its dimensions.
 */
public class Shape(
    internal val dims: List<Int>,
) {
    public constructor(vararg dims: Int) : this(dims.toList())

    private val self: List<Int> get() = dims

    internal val rank: Int get() = dims.size

    public fun copy(dims: List<Int> = this.dims): Shape = Shape(dims)

    override fun equals(other: Any?): Boolean = other is Shape && dims == other.dims

    override fun hashCode(): Int = dims.hashCode()

    public fun fmt(f: String = ""): String = write(f, "{:?}", this.dims())

    public companion object {
        public val Self: Companion get() = this
        public val SCALAR: Shape = Shape(emptyList())

        public fun from(vararg dims: Int): Shape = Self(dims.toVec())

        public fun from(shape: Shape): Shape = Self(shape.dims.toVec())

        public fun from(u: Unit): Shape = Self(vec(), u)

        public fun from(d1: Int): Shape = Self(vec(d1))

        public fun from(dims: List<Int>): Shape = Self(dims)

        public fun fromDims(dims: List<Int>): Shape = Self(dims.toVec())

        public fun fromDims(vararg dims: Int): Shape = Self(dims.toVec())

        public fun toIndex(
            dim: Int,
            shape: Shape,
            op: String,
        ): Int {
            val rank = shape.rank
            val normalized = if (dim < 0) rank + dim else dim
            if (normalized < 0 || normalized >= rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            return normalized
        }

        public fun toIndexPlusOne(
            dim: Int,
            shape: Shape,
            op: String,
        ): Int {
            val rank = shape.rank
            val normalized = if (dim < 0) rank + 1 + dim else dim
            if (normalized < 0 || normalized > rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            return normalized
        }
    }

    /**
     * The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
     */
    public fun rank(): Int = self.len()

    public fun intoDims(): List<Int> = self

    /**
     * The dimensions as a slice of integers.
     */
    public fun dims(): List<Int> = self

    /**
     * The dimension size for a specified dimension index.
     */
    public fun <D> dim(dim: D): Int {
        val dimIdx = dim.toIndex(this, "dim")
        return Ok(this.dims()[dimIdx])
    }

    /**
     * The total number of elements, this is the product of all dimension sizes.
     */
    public fun elemCount(): Int = self.iter().product()

    /**
     * The strides given in number of elements for a contiguous n-dimensional
     * arrays using this shape.
     */
    public fun strideContiguous(): List<Int> {
        val stride: MutableList<Int> =
            this
                .self
                .iter()
                .rev()
                .scan(1) { prod, u ->
                    val prodPreMult = prod[0]
                    prod[0] *= u
                    Some(prodPreMult)
                }.collect()
        stride.reverse()
        return stride
    }

    /**
     * Returns true if the strides are C contiguous (aka row major).
     */
    public fun isContiguous(stride: List<Int>): Boolean {
        if (self.len() != stride.len()) {
            return false
        }
        var acc = 1
        for ((s, dim) in stride.iter().zip(self.iter()).rev()) {
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
        if (self.len() != stride.len()) {
            return false
        }
        var acc = 1
        for ((s, dim) in stride.iter().zip(self.iter())) {
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
    public fun extend(additionalDims: List<Int>): Shape {
        val s = self.extend(additionalDims)
        return Shape(s)
    }

    /**
     * Check whether the two shapes are compatible for broadcast, and if it is the case return the
     * broadcasted shape. This is to be used for binary pointwise ops.
     */
    public fun broadcastShapeBinaryOp(
        rhs: Shape,
        op: String,
    ): Shape {
        val lhs = this
        val lhsDims = lhs.dims()
        val rhsDims = rhs.dims()
        val lhsNdims = lhsDims.len()
        val rhsNdims = rhsDims.len()
        val bcastNdims = max(lhsNdims, rhsNdims)
        val bcastDims = MutableList(bcastNdims) { 0 }
        for (idx in 0 until bcastNdims) {
            val revIdx = bcastNdims - idx
            val lValue =
                if (lhsNdims < revIdx) {
                    1
                } else {
                    lhsDims[lhsNdims - revIdx]
                }
            val rValue =
                if (rhsNdims < revIdx) {
                    1
                } else {
                    rhsDims[rhsNdims - revIdx]
                }
            bcastDims[idx] =
                if (lValue == rValue) {
                    lValue
                } else if (lValue == 1) {
                    rValue
                } else if (rValue == 1) {
                    lValue
                } else {
                    throw CandleException
                        .ShapeMismatchBinaryOp(
                            lhs = lhs.clone(),
                            rhs = rhs.clone(),
                            op = op,
                        ).bt()
                }
        }
        return Ok(Shape.from(bcastDims))
    }

    public fun broadcastShapeMatmul(rhs: Shape): MatmulBroadcastShapes {
        val lhs = this
        val lhsDims = lhs.dims()
        val rhsDims = rhs.dims()
        if (lhsDims.len() < 2 || rhsDims.len() < 2) {
            bail("only 2d matrixes are supported $lhs $rhs")
        }
        val (m, lhsK) = Pair(lhsDims[lhsDims.len() - 2], lhsDims[lhsDims.len() - 1])
        val (rhsK, n) = Pair(rhsDims[rhsDims.len() - 2], rhsDims[rhsDims.len() - 1])
        if (lhsK != rhsK) {
            bail("different inner dimensions in broadcast matmul $lhs $rhs")
        }

        val lhsB = Self.from(lhsDims.subList(0, lhsDims.len() - 2))
        val rhsB = Self.from(rhsDims.subList(0, rhsDims.len() - 2))
        val bcast = lhsB.broadcastShapeBinaryOp(rhsB, "broadcast_matmul")
        val bcastDims = bcast.dims()

        val bcastLhs = listOf(bcastDims, listOf(m, lhsK)).concat()
        val bcastRhs = listOf(bcastDims, listOf(rhsK, n)).concat()
        return Ok(MatmulBroadcastShapes(Shape.from(bcastLhs), Shape.from(bcastRhs)))
    }
}

public interface Dim {
    public fun toIndex(
        shape: Shape,
        op: String,
    ): Int

    public fun toIndexPlusOne(
        shape: Shape,
        op: String,
    ): Int
}

public class DimInt(
    public val dim: Int,
) : Dim {
    private val self: Int get() = dim

    override fun toIndex(
        shape: Shape,
        op: String,
    ): Int {
        val dim = self
        return if (dim >= shape.dims().len()) {
            throw CandleException
                .DimOutOfRange(
                    shape = shape.clone(),
                    dim = dim,
                    op = op,
                ).bt()
        } else {
            Ok(dim)
        }
    }

    override fun toIndexPlusOne(
        shape: Shape,
        op: String,
    ): Int {
        val dim = self
        return if (dim > shape.dims().len()) {
            throw CandleException
                .DimOutOfRange(
                    shape = shape.clone(),
                    dim = dim,
                    op = op,
                ).bt()
        } else {
            Ok(dim)
        }
    }
}

public sealed class D : Dim {
    private val self: D = this

    public object Minus1 : D()

    public object Minus2 : D()

    public data class Minus(
        val u: Int,
    ) : D()

    public data class Custom(
        val u: Int,
    ) : D()

    public fun outOfRange(
        shape: Shape,
        op: String,
    ): CandleException {
        val dim =
            when (self) {
                is Minus1 -> -1
                is Minus2 -> -2
                is Minus -> -self.u
                is Custom -> self.u
            }
        return CandleException
            .DimOutOfRange(
                shape = shape.clone(),
                dim = dim,
                op = op,
            ).bt()
    }

    override fun toIndex(
        shape: Shape,
        op: String,
    ): Int {
        val rank = shape.rank()
        return when {
            self is Minus1 && rank >= 1 -> Ok(rank - 1)
            self is Minus2 && rank >= 2 -> Ok(rank - 2)
            self is Minus && self.u > 0 && rank >= self.u -> Ok(rank - self.u)
            self is Custom && self.u >= 0 && self.u < rank -> Ok(self.u)
            else -> throw self.outOfRange(shape, op)
        }
    }

    override fun toIndexPlusOne(
        shape: Shape,
        op: String,
    ): Int {
        val rank = shape.rank()
        return when {
            self is Minus1 -> Ok(rank)
            self is Minus2 && rank >= 1 -> Ok(rank - 1)
            self is Minus && self.u > 0 && rank + 1 >= self.u -> Ok(rank + 1 - self.u)
            self is Custom && self.u >= 0 && self.u <= rank -> Ok(self.u)
            else -> throw self.outOfRange(shape, op)
        }
    }
}

public interface Dims {
    public fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int>

    public fun toIndexes(
        shape: Shape,
        op: String,
    ): List<Int> {
        val dims = this.toIndexesInternal(shape, op)
        for ((i, dim) in dims.iter().enumerate()) {
            if (dims.subList(0, i).contains(dim)) {
                throw CandleException
                    .DuplicateDimIndex(
                        shape = shape.clone(),
                        dims = dims.clone(),
                        op = op,
                    ).bt()
            }
            if (dim >= shape.rank()) {
                throw CandleException
                    .DimOutOfRange(
                        shape = shape.clone(),
                        dim = dim,
                        op = op,
                    ).bt()
            }
        }
        return Ok(dims)
    }
}

public class DimsList(
    public val dims: List<Int>,
) : Dims {
    private val self: List<Int> get() = dims

    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> = Ok(self, shape, op)
}

public class DimsArray(
    public val dims: IntArray,
) : Dims {
    private val self: IntArray get() = dims

    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> = Ok(self.toVec(), shape, op)
}

public class DimsSlice(
    public val dims: IntArray,
) : Dims {
    private val self: IntArray get() = dims

    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> = Ok(self.toVec(), shape, op)
}

public object DimsEmpty : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> = Ok(vec(), shape, op)
}

public class DimsDim(
    public val dim: Dim,
) : Dims {
    private val self: Dim get() = dim

    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val dim = self.toIndex(shape, op)
        return Ok(vec(dim))
    }
}

public class DimsTuple1(
    public val v0: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val dim = v0.toIndex(shape, op)
        return Ok(vec(dim))
    }
}

public class DimsTuple2(
    public val v0: Dim,
    public val v1: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val d0 = v0.toIndex(shape, op)
        val d1 = v1.toIndex(shape, op)
        return Ok(vec(d0, d1))
    }
}

public class DimsTuple3(
    public val v0: Dim,
    public val v1: Dim,
    public val v2: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val d0 = v0.toIndex(shape, op)
        val d1 = v1.toIndex(shape, op)
        val d2 = v2.toIndex(shape, op)
        return Ok(vec(d0, d1, d2))
    }
}

public class DimsTuple4(
    public val v0: Dim,
    public val v1: Dim,
    public val v2: Dim,
    public val v3: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val d0 = v0.toIndex(shape, op)
        val d1 = v1.toIndex(shape, op)
        val d2 = v2.toIndex(shape, op)
        val d3 = v3.toIndex(shape, op)
        return Ok(vec(d0, d1, d2, d3))
    }
}

public class DimsTuple5(
    public val v0: Dim,
    public val v1: Dim,
    public val v2: Dim,
    public val v3: Dim,
    public val v4: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val d0 = v0.toIndex(shape, op)
        val d1 = v1.toIndex(shape, op)
        val d2 = v2.toIndex(shape, op)
        val d3 = v3.toIndex(shape, op)
        val d4 = v4.toIndex(shape, op)
        return Ok(vec(d0, d1, d2, d3, d4))
    }
}

public class DimsTuple6(
    public val v0: Dim,
    public val v1: Dim,
    public val v2: Dim,
    public val v3: Dim,
    public val v4: Dim,
    public val v5: Dim,
) : Dims {
    override fun toIndexesInternal(
        shape: Shape,
        op: String,
    ): List<Int> {
        val d0 = v0.toIndex(shape, op)
        val d1 = v1.toIndex(shape, op)
        val d2 = v2.toIndex(shape, op)
        val d3 = v3.toIndex(shape, op)
        val d4 = v4.toIndex(shape, op)
        val d5 = v5.toIndex(shape, op)
        return Ok(vec(d0, d1, d2, d3, d4, d5))
    }
}

public interface ShapeWithOneHole {
    public fun intoShape(elCount: Int): Shape
}

public class ShapeWithOneHoleShape(
    public val shape: Shape,
) : ShapeWithOneHole {
    private val self: Shape get() = shape

    override fun intoShape(elCount: Int): Shape = Ok(self.into(), elCount)
}

public object ShapeWithOneHole1 : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape = Ok(elCount.into())
}

public fun holeSize(
    elCount: Int,
    prodD: Int,
    s: ShapeWithOneHole,
): Int {
    if (prodD == 0) {
        bail("cannot reshape tensor of $elCount elements to $s")
    }
    if (!elCount.isMultipleOf(prodD)) {
        bail("cannot reshape tensor with $elCount elements to $s")
    }
    return Ok(elCount / prodD)
}

public data class ShapeWithOneHole2Hole0(
    public val d1: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole2Hole0 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Pair(holeSize(elCount, d1, self), d1).into())
    }
}

public data class ShapeWithOneHole2Hole1(
    public val d1: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole2Hole1 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Pair(d1, holeSize(elCount, d1, self)).into())
    }
}

public data class ShapeWithOneHole3Hole0(
    public val d1: Int,
    public val d2: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole0 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(holeSize(elCount, d1 * d2, self), d1, d2).into())
    }
}

public data class ShapeWithOneHole3Hole1(
    public val d1: Int,
    public val d2: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole1 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(d1, holeSize(elCount, d1 * d2, self), d2).into())
    }
}

public data class ShapeWithOneHole3Hole2(
    public val d1: Int,
    public val d2: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole2 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(d1, d2, holeSize(elCount, d1 * d2, self)).into())
    }
}

public data class ShapeWithOneHole4Hole0(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole0 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d, d1, d2, d3).into())
    }
}

public data class ShapeWithOneHole4Hole1(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole1 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d1, d, d2, d3).into())
    }
}

public data class ShapeWithOneHole4Hole2(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole2 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d1, d2, d, d3).into())
    }
}

public data class ShapeWithOneHole4Hole3(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole3 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d1, d2, d3, d).into())
    }
}

public data class ShapeWithOneHole5Hole0(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole5Hole0 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3 * d4, self)
        return Ok(listOf(d, d1, d2, d3, d4).into())
    }
}

public data class ShapeWithOneHole5Hole1(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole5Hole1 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3 * d4, self)
        return Ok(listOf(d1, d, d2, d3, d4).into())
    }
}

public data class ShapeWithOneHole5Hole2(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole5Hole2 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3 * d4, self)
        return Ok(listOf(d1, d2, d, d3, d4).into())
    }
}

public data class ShapeWithOneHole5Hole3(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole5Hole3 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3 * d4, self)
        return Ok(listOf(d1, d2, d3, d, d4).into())
    }
}

public data class ShapeWithOneHole5Hole4(
    public val d1: Int,
    public val d2: Int,
    public val d3: Int,
    public val d4: Int,
) : ShapeWithOneHole {
    private val self: ShapeWithOneHole5Hole4 get() = this

    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3 * d4, self)
        return Ok(listOf(d1, d2, d3, d4, d).into())
    }
}

public data class MatmulBroadcastShapes(
    val lhs: Shape,
    val rhs: Shape,
) {
    public val first: Shape get() = lhs
    public val second: Shape get() = rhs

    public inline fun unwrapOrElse(f: (CandleException) -> MatmulBroadcastShapes): MatmulBroadcastShapes = this

    public fun getOrThrow(): MatmulBroadcastShapes = this
}

public data class Dims2(
    val d0: Int,
    val d1: Int,
) {
    public val first: Int get() = d0
    public val second: Int get() = d1

    public inline fun unwrapOrElse(f: (CandleException) -> Dims2): Dims2 = this

    public inline fun getOrElse(f: (CandleException) -> Dims2): Dims2 = this

    public fun getOrThrow(): Dims2 = this
}

public data class Dims3(
    val d0: Int,
    val d1: Int,
    val d2: Int,
) {
    public val first: Int get() = d0
    public val second: Int get() = d1
    public val third: Int get() = d2

    public inline fun unwrapOrElse(f: (CandleException) -> Dims3): Dims3 = this

    public inline fun getOrElse(f: (CandleException) -> Dims3): Dims3 = this

    public fun getOrThrow(): Dims3 = this
}

public data class Dims4(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
) {
    public inline fun unwrapOrElse(f: (CandleException) -> Dims4): Dims4 = this

    public inline fun getOrElse(f: (CandleException) -> Dims4): Dims4 = this

    public fun getOrThrow(): Dims4 = this
}

public data class Dims5(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
    val d4: Int,
) {
    public inline fun unwrapOrElse(f: (CandleException) -> Dims5): Dims5 = this

    public inline fun getOrElse(f: (CandleException) -> Dims5): Dims5 = this

    public fun getOrThrow(): Dims5 = this
}

public fun Shape.dims0() {
    if (dims.isNotEmpty()) throw CandleException.UnexpectedNumberOfDims(0, dims.size, this)
}

public fun Shape.dims1(): Int {
    if (dims.size != 1) throw CandleException.UnexpectedNumberOfDims(1, dims.size, this)
    return dims[0]
}

public fun Shape.dims2(): Dims2 {
    if (dims.size != 2) throw CandleException.UnexpectedNumberOfDims(2, dims.size, this)
    return Dims2(dims[0], dims[1])
}

public fun Shape.dims3(): Dims3 {
    if (dims.size != 3) throw CandleException.UnexpectedNumberOfDims(3, dims.size, this)
    return Dims3(dims[0], dims[1], dims[2])
}

public fun Shape.dims4(): Dims4 {
    if (dims.size != 4) throw CandleException.UnexpectedNumberOfDims(4, dims.size, this)
    return Dims4(dims[0], dims[1], dims[2], dims[3])
}

public fun Shape.dims5(): Dims5 {
    if (dims.size != 5) throw CandleException.UnexpectedNumberOfDims(5, dims.size, this)
    return Dims5(dims[0], dims[1], dims[2], dims[3], dims[4])
}

public fun dims2(dims: List<Int>): Dims2 = Shape(dims).dims2()

public fun dims3(dims: List<Int>): Dims3 = Shape(dims).dims3()

public fun dims4(dims: List<Int>): Dims4 = Shape(dims).dims4()

public fun dims5(dims: List<Int>): Dims5 = Shape(dims).dims5()

public val Shape.Companion.toIndex: (Int, Shape, String) -> Int
    get() = { dim, shape, op ->
        val rank = shape.rank
        val normalized = if (dim < 0) rank + dim else dim
        if (normalized < 0 || normalized >= rank) {
            throw CandleException.DimOutOfRange(shape, dim, op)
        }
        normalized
    }

public val Shape.Companion.toIndexPlusOne: (Int, Shape, String) -> Int
    get() = { dim, shape, op ->
        val rank = shape.rank
        val normalized = if (dim < 0) rank + 1 + dim else dim
        if (normalized < 0 || normalized > rank) {
            throw CandleException.DimOutOfRange(shape, dim, op)
        }
        normalized
    }

internal fun Shape.Companion.fromDims(dims: IntArray): Shape = Shape(dims.toList())

internal fun Shape.Companion.Self(dims: List<Int>): Shape = Shape(dims)

public val List<Int>.intoShapeWithHole: (Int) -> Shape
    get() = { elCount ->
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
            Shape(this)
        } else {
            if (knownProduct == 0 || elCount % knownProduct != 0) {
                bail("cannot reshape tensor with $elCount elements to $this")
            }
            val hSize = elCount / knownProduct
            val result = toMutableList()
            result[holeIdx] = hSize
            Shape(result)
        }
    }
