// port-lint: source shape.rs
package io.github.kotlinmania.candle

/**
 * The shape of a tensor is a tuple with the size of each of its dimensions.
 */

public data class MatmulBroadcastShapes(
    val lhs: Shape,
    val rhs: Shape,
) {
    public val first: Shape get() = lhs
    public val second: Shape get() = rhs
    public val unwrapOrElse: ((CandleError) -> MatmulBroadcastShapes) -> MatmulBroadcastShapes get() = { this }
    public val getOrThrow: () -> MatmulBroadcastShapes get() = { this }
}

public data class Dims2(
    val d0: Int,
    val d1: Int,
) {
    public val first: Int get() = d0
    public val second: Int get() = d1
    public val unwrapOrElse: ((CandleError) -> Dims2) -> Dims2 get() = { this }
    public val getOrThrow: () -> Dims2 get() = { this }
}

public data class Dims3(
    val d0: Int,
    val d1: Int,
    val d2: Int,
) {
    public val first: Int get() = d0
    public val second: Int get() = d1
    public val third: Int get() = d2
    public val unwrapOrElse: ((CandleError) -> Dims3) -> Dims3 get() = { this }
    public val getOrThrow: () -> Dims3 get() = { this }
}

public data class Dims4(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
) {
    public val unwrapOrElse: ((CandleError) -> Dims4) -> Dims4 get() = { this }
    public val getOrThrow: () -> Dims4 get() = { this }
}

public data class Dims5(
    val d0: Int,
    val d1: Int,
    val d2: Int,
    val d3: Int,
    val d4: Int,
) {
    public val unwrapOrElse: ((CandleError) -> Dims5) -> Dims5 get() = { this }
    public val getOrThrow: () -> Dims5 get() = { this }
}

private val <T> List<T>.len: () -> Int get() = { size }
private val <T> List<T>.iter: () -> List<T> get() = { this }
private val <T> List<T>.rev: () -> List<T> get() = { reversed() }
private val List<Int>.product: () -> Int get() = { fold(1) { a, b -> a * b } }
private val IntArray.toVec: () -> List<Int> get() = { toList() }
private val <T> List<T>.toVec: () -> List<T> get() = { toList() }
private val toDimsList: (Any, IntArray) -> List<Int>
    get() = { dims, rest ->
        when (dims) {
            is List<*> -> dims.map { it as Int }
            is IntArray -> dims.toList()
            is Int -> listOf(dims) + rest.toList()
            else -> emptyList()
        }
    }
private val <D> D.toIndex: (Shape, String) -> Int
    get() = { shape, op ->
        when (this) {
            is Dim -> this.toIndex(shape, op)
            is Int -> Shape.toIndex(this, shape, op)
            else -> Shape.toIndex(0, shape, op)
        }
    }
private val <T> T.into: () -> Shape
    get() = {
        when (this) {
            is Shape -> this
            is Int -> Shape(listOf(this))
            is Pair<*, *> -> Shape(listOf(this.first as Int, this.second as Int))
            is Triple<*, *, *> -> Shape(listOf(this.first as Int, this.second as Int, this.third as Int))
            is List<*> -> Shape(this.map { it as Int })
            else -> Shape(emptyList())
        }
    }
private val <T> T.Ok: (T) -> T get() = { it }
private val <T> T.Ok2: (T, Any?) -> T get() = { v, _ -> v }
private val <T> T.Ok3: (T, Any?, Any?) -> T get() = { v, _, _ -> v }
private val <T> T.Some: (T) -> T get() = { it }
private val List<Int>.scan: (Int, (IntArray, Int) -> Int) -> List<Int>
    get() = { init, f ->
        val state = intArrayOf(init)
        val out = ArrayList<Int>(size)
        for (u in this) {
            out.add(f(state, u))
        }
        out
    }
private val <T> List<T>.collect: () -> MutableList<T> get() = { toMutableList() }
private val write: (String, String, Any?) -> String get() = { f, _, v -> f + v.toString() }

public data class Shape(
    val dims: List<Int>,
) {
    public constructor(vararg dims: Int) : this(dims.toList())

    private val self: List<Int> get() = dims

    public val rank: Int get() = dims.size

    public fun fmt(f: String = ""): String {
        return write(f, "{:?}", this.dims())
    }

    /**
     * The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
     */
    public fun rank(): Int {
        return self.len()
    }

    public fun intoDims(): List<Int> {
        return self.toVec()
    }

    /**
     * The dimensions as a slice of integers.
     */
    public fun dims(): List<Int> {
        return self.iter()
    }

    /**
     * The dimension size for a specified dimension index.
     */
    public fun <D> dim(dim: D): Int {
        val idx = dim.toIndex(this, "dim")
        return Ok(this.dims()[idx])
    }

    /**
     * The total number of elements, this is the product of all dimension sizes.
     */
    public fun elemCount(): Int {
        return self.iter().product()
    }

    /**
     * The strides given in number of elements for a contiguous n-dimensional
     * arrays using this shape.
     */
    public fun strideContiguous(): List<Int> {
        val stride: MutableList<Int> = this
            .self
            .iter()
            .rev()
            .scan(1) { prod, u ->
                val prodPreMult = prod[0]
                prod[0] *= u
                Some(prodPreMult)
            }
            .collect()
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
        for (dim in stride.iter().zip(self.iter()).rev()) {
            if (dim.second > 1 && dim.first != acc) {
                return false
            }
            acc *= dim.second
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
        for (dim in stride.iter().zip(self.iter())) {
            if (dim.second > 1 && dim.first != acc) {
                return false
            }
            acc *= dim.second
        }
        return true
    }

    /**
     * Modifies the shape by adding a list of additional dimensions at the end of the existing
     * dimensions.
     */
    public fun extend(additionalDims: List<Int>): Shape {
        val extended = self + additionalDims
        return Shape(extended)
    }

    /**
     * Check whether the two shapes are compatible for broadcast, and if it is the case return the
     * broadcasted shape. This is to be used for binary pointwise ops.
     */
    public fun broadcastShapeBinaryOp(rhs: Shape, op: String): Shape {
        val lhs = this
        val lhsDims = lhs.dims()
        val rhsDims = rhs.dims()
        val lhsNdims = lhsDims.len()
        val rhsNdims = rhsDims.len()
        val bcastNdims = maxOf(lhsNdims, rhsNdims)
        val bcastDims = MutableList(bcastNdims) { 0 }
        for (idx in 0 until bcastNdims) {
            val revIdx = bcastNdims - idx
            val lValue = if (lhsNdims < revIdx) {
                1
            } else {
                lhsDims[lhsNdims - revIdx]
            }
            val rValue = if (rhsNdims < revIdx) {
                1
            } else {
                rhsDims[rhsNdims - revIdx]
            }
            bcastDims[idx] = if (lValue == rValue) {
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
                    )
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
        val m = lhsDims[lhsDims.len() - 2]
        val lhsK = lhsDims[lhsDims.len() - 1]
        val rhsK = rhsDims[rhsDims.len() - 2]
        val n = rhsDims[rhsDims.len() - 1]
        if (lhsK != rhsK) {
            bail("different inner dimensions in broadcast matmul $lhs $rhs")
        }
        val lhsB = from(lhsDims.subList(0, lhsDims.len() - 2))
        val rhsB = from(rhsDims.subList(0, rhsDims.len() - 2))
        val bcast = lhsB.broadcastShapeBinaryOp(rhsB, "broadcast_matmul")
        val bcastDims = bcast.dims()
        val bcastLhs = bcastDims + listOf(m, lhsK)
        val bcastRhs = bcastDims + listOf(rhsK, n)
        return MatmulBroadcastShapes(Shape.from(bcastLhs), Shape.from(bcastRhs))
    }

    public val clone: () -> Shape get() = { Shape(dims.toList()) }

    public val dims0: () -> Unit
        get() = {
            if (dims.isNotEmpty()) {
                throw CandleException.UnexpectedNumberOfDims(0, dims.size, this)
            }
        }

    public val dims1: () -> Int
        get() = {
            if (dims.size != 1) {
                throw CandleException.UnexpectedNumberOfDims(1, dims.size, this)
            }
            dims[0]
        }

    public val dims2: () -> Dims2
        get() = {
            if (dims.size != 2) {
                throw CandleException.UnexpectedNumberOfDims(2, dims.size, this)
            }
            Dims2(dims[0], dims[1])
        }

    public val dims3: () -> Dims3
        get() = {
            if (dims.size != 3) {
                throw CandleException.UnexpectedNumberOfDims(3, dims.size, this)
            }
            Dims3(dims[0], dims[1], dims[2])
        }

    public val dims4: () -> Dims4
        get() = {
            if (dims.size != 4) {
                throw CandleException.UnexpectedNumberOfDims(4, dims.size, this)
            }
            Dims4(dims[0], dims[1], dims[2], dims[3])
        }

    public val dims5: () -> Dims5
        get() = {
            if (dims.size != 5) {
                throw CandleException.UnexpectedNumberOfDims(5, dims.size, this)
            }
            Dims5(dims[0], dims[1], dims[2], dims[3], dims[4])
        }

    public companion object {
        public val SCALAR: Shape = Shape(emptyList())

        public fun from(dims: IntArray, c: Int = dims.size): Shape {
            return Ok2(Shape(dims.toVec()), c)
        }

        public fun from(vararg dims: Int): Shape {
            return Shape(dims.toVec())
        }

        public fun from(shape: Shape): Shape {
            return Shape(shape.dims.toVec())
        }

        public fun from(unit: Unit): Shape {
            return Ok2(Shape(emptyList()), unit)
        }

        public fun from(d1: Int): Shape {
            return Shape(listOf(d1))
        }

        public fun from(dims: List<Int>): Shape {
            return Shape(dims)
        }

        public fun fromDims(dims: Any, vararg rest: Int): Shape {
            return Shape(toDimsList(dims, rest))
        }

        public val toIndex: (Int, Shape, String) -> Int = { dim, shape, op ->
            val rank = shape.rank
            val normalized = if (dim < 0) rank + dim else dim
            if (normalized < 0 || normalized >= rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            normalized
        }

        public val toIndexPlusOne: (Int, Shape, String) -> Int = { dim, shape, op ->
            val rank = shape.rank
            val normalized = if (dim < 0) rank + 1 + dim else dim
            if (normalized < 0 || normalized > rank) {
                throw CandleException.DimOutOfRange(shape, dim, op)
            }
            normalized
        }
    }
}

public interface Dim {
    public fun toIndex(shape: Shape, op: String): Int
    public fun toIndexPlusOne(shape: Shape, op: String): Int
}

public class DimInt(public val self: Int) : Dim {
    public val dim: Int get() = self

    override fun toIndex(shape: Shape, op: String): Int {
        val dim = self
        return if (dim >= shape.dims().len()) {
            throw CandleException.DimOutOfRange(
                shape = shape.clone(),
                dim = dim,
                op = op,
            )
        } else {
            val res = Ok(dim)
            res
        }
    }

    override fun toIndexPlusOne(shape: Shape, op: String): Int {
        val dim = self
        return if (dim > shape.dims().len()) {
            throw CandleException.DimOutOfRange(
                shape = shape.clone(),
                dim = dim,
                op = op,
            )
        } else {
            val res = Ok(dim)
            res
        }
    }
}

public sealed class D : Dim {
    private val self: D get() = this

    public object Minus1 : D()
    public object Minus2 : D()
    public data class Minus(val u: Int) : D()

    public fun outOfRange(shape: Shape, op: String): CandleException {
        val dim = when (self) {
            is Minus1 -> -1
            is Minus2 -> -2
            is Minus -> -self.u
        }
        val err = CandleException.DimOutOfRange(
            shape = shape.clone(),
            dim = dim,
            op = op,
        )
        return err
    }

    override fun toIndex(shape: Shape, op: String): Int {
        val rank = shape.rank()
        return when {
            self is Minus1 && rank >= 1 -> Ok(rank - 1)
            self is Minus2 && rank >= 2 -> Ok(rank - 2)
            self is Minus && self.u > 0 && rank >= self.u -> Ok(rank - self.u)
            else -> throw self.outOfRange(shape, op)
        }
    }

    override fun toIndexPlusOne(shape: Shape, op: String): Int {
        val rank = shape.rank()
        return when {
            self is Minus1 -> Ok(rank)
            self is Minus2 && rank >= 1 -> Ok(rank - 1)
            self is Minus && self.u > 0 && rank + 1 >= self.u -> Ok(rank + 1 - self.u)
            else -> throw self.outOfRange(shape, op)
        }
    }
}

public interface Dims {
    public fun toIndexesInternal(shape: Shape, op: String): List<Int>

    public fun toIndexes(shape: Shape, op: String): List<Int> {
        val dims = this.toIndexesInternal(shape, op)
        for (i in dims.indices) {
            val dim = dims[i]
            if (dims.subList(0, i).contains(dim)) {
                throw CandleException.DuplicateDimIndex(
                    shape = shape.clone(),
                    dims = dims.toVec(),
                    op = op,
                )
            }
            if (dim >= shape.rank()) {
                throw CandleException.DimOutOfRange(
                    shape = shape.clone(),
                    dim = dim,
                    op = op,
                )
            }
        }
        val res = Ok(dims)
        return res
    }
}

public class DimsList(public val self: List<Int>) : Dims {
    public val dims: List<Int> get() = self
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        return Ok3(self, shape, op)
    }
}

public class DimsArray(public val self: IntArray) : Dims {
    public val dims: IntArray get() = self
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        return Ok3(self.toVec(), shape, op)
    }
}

public class DimsSlice(public val self: IntArray) : Dims {
    public val dims: IntArray get() = self
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        return Ok3(self.toVec(), shape, op)
    }
}

public object DimsEmpty : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        return Ok3(emptyList(), shape, op)
    }
}

public class DimsDim(public val self: Dim) : Dims {
    public val dim: Dim get() = self
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val dim = self.toIndex(shape, op)
        return Ok(listOf(dim))
    }
}

public class DimsTuple1(public val d1: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val dim = d1.toIndex(shape, op)
        return Ok(listOf(dim))
    }
}

public class DimsTuple2(public val d1: Dim, public val d2: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val d0 = d1.toIndex(shape, op)
        val d1Idx = d2.toIndex(shape, op)
        return Ok(listOf(d0, d1Idx))
    }
}

public class DimsTuple3(public val d1: Dim, public val d2: Dim, public val d3: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val d0 = d1.toIndex(shape, op)
        val d1Idx = d2.toIndex(shape, op)
        val d2Idx = d3.toIndex(shape, op)
        return Ok(listOf(d0, d1Idx, d2Idx))
    }
}

public class DimsTuple4(public val d1: Dim, public val d2: Dim, public val d3: Dim, public val d4: Dim) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val d0 = d1.toIndex(shape, op)
        val d1Idx = d2.toIndex(shape, op)
        val d2Idx = d3.toIndex(shape, op)
        val d3Idx = d4.toIndex(shape, op)
        return Ok(listOf(d0, d1Idx, d2Idx, d3Idx))
    }
}

public class DimsTuple5(
    public val d1: Dim,
    public val d2: Dim,
    public val d3: Dim,
    public val d4: Dim,
    public val d5: Dim,
) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val d0 = d1.toIndex(shape, op)
        val d1Idx = d2.toIndex(shape, op)
        val d2Idx = d3.toIndex(shape, op)
        val d3Idx = d4.toIndex(shape, op)
        val d4Idx = d5.toIndex(shape, op)
        return Ok(listOf(d0, d1Idx, d2Idx, d3Idx, d4Idx))
    }
}

public class DimsTuple6(
    public val d1: Dim,
    public val d2: Dim,
    public val d3: Dim,
    public val d4: Dim,
    public val d5: Dim,
    public val d6: Dim,
) : Dims {
    override fun toIndexesInternal(shape: Shape, op: String): List<Int> {
        val d0 = d1.toIndex(shape, op)
        val d1Idx = d2.toIndex(shape, op)
        val d2Idx = d3.toIndex(shape, op)
        val d3Idx = d4.toIndex(shape, op)
        val d4Idx = d5.toIndex(shape, op)
        val d5Idx = d6.toIndex(shape, op)
        return Ok(listOf(d0, d1Idx, d2Idx, d3Idx, d4Idx, d5Idx))
    }
}

public interface ShapeWithOneHole {
    public fun intoShape(elCount: Int): Shape
}

public class ShapeWithOneHoleShape(public val self: Shape) : ShapeWithOneHole {
    public val shape: Shape get() = self
    override fun intoShape(elCount: Int): Shape {
        return Ok2(self.into(), elCount)
    }
}

public object ShapeWithOneHole1 : ShapeWithOneHole {
    override fun intoShape(elCount: Int): Shape {
        return Ok(elCount.into())
    }
}

public fun holeSize(elCount: Int, prodD: Int, s: Any): Int {
    if (prodD == 0) {
        bail("cannot reshape tensor of $elCount elements to $s")
    }
    if (elCount % prodD != 0) {
        bail("cannot reshape tensor with $elCount elements to $s")
    }
    return Ok(elCount / prodD)
}

public data class ShapeWithOneHole2Hole0(public val d1: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole2Hole0 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Pair(holeSize(elCount, d1, self), d1).into())
    }
}

public data class ShapeWithOneHole2Hole1(public val d1: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole2Hole1 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Pair(d1, holeSize(elCount, d1, self)).into())
    }
}

public data class ShapeWithOneHole3Hole0(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole0 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(holeSize(elCount, d1 * d2, self), d1, d2).into())
    }
}

public data class ShapeWithOneHole3Hole1(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole1 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(d1, holeSize(elCount, d1 * d2, self), d2).into())
    }
}

public data class ShapeWithOneHole3Hole2(public val d1: Int, public val d2: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole3Hole2 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        return Ok(Triple(d1, d2, holeSize(elCount, d1 * d2, self)).into())
    }
}

public data class ShapeWithOneHole4Hole0(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole0 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d, d1, d2, d3).into())
    }
}

public data class ShapeWithOneHole4Hole1(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole1 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d1, d, d2, d3).into())
    }
}

public data class ShapeWithOneHole4Hole2(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
    private val self: ShapeWithOneHole4Hole2 get() = this
    override fun intoShape(elCount: Int): Shape {
        val d1 = self.d1
        val d = holeSize(elCount, d1 * d2 * d3, self)
        return Ok(listOf(d1, d2, d, d3).into())
    }
}

public data class ShapeWithOneHole4Hole3(public val d1: Int, public val d2: Int, public val d3: Int) : ShapeWithOneHole {
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
