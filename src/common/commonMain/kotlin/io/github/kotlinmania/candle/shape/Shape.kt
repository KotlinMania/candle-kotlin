// port-lint: source shape.rs
package io.github.kotlinmania.candle.shape

import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.tensor.Tensor

// The shape of a tensor is a tuple with the size of each of its dimensions.

public open class Shape(
    public var value: List<Int>,
) : From<List<Int>>, TryInto<Unit> {
    public var field0: List<Int>
        get() = value
        set(v) {
            value = v
        }

    public open fun fmt(f: StringBuilder) {
        f.append("${this.dims()}")
    }

    /**
     * The rank is the number of dimensions, 0 for a scalar value, 1 for a vector, etc.
     */
    public open fun rank(): Int = this.value.size

    public open fun intoDims(): List<Int> = this.value

    /**
     * The dimensions as a slice of `Int`.
     */
    public open fun dims(): List<Int> = this.value

    /**
     * The dimension size for a specified dimension index.
     */
    public open fun <D : Dim> dim(dim: D): Result<Int> {
        val dim1 = dim.toIndex(this, "dim").getOrElse { return Result.failure(it) }
        return Result.success(this.dims()[dim1])
    }

    /**
     * The total number of elements, this is the product of all dimension sizes.
     */
    public open fun elemCount(): Int = this.value.iter().product()

    /**
     * The strides given in number of elements for a contiguous n-dimensional
     * arrays using this shape.
     */
    internal open fun strideContiguous(): List<Int> {
        var stride: Vec =
            this.value.iter().rev().scan(1, { prod, u ->
                val prodPreMult = prod
                prod *= u
                prodPreMult
            }).collect()
        stride.reverse()
        return stride
    }

    /**
     * Returns true if the strides are C contiguous (aka row major).
     */
    public open fun isContiguous(stride: List<Int>): Boolean {
        if (this.value.size != stride.size) {
            return false
        }
        var acc = 1
        for ((stride1, dim) in (stride.zip(this.value)).rev()) {
            if (dim > 1 && stride1 != acc) {
                return false
            }
            acc *= dim
        }
        return true
    }

    /**
     * Returns true if the strides are Fortran contiguous (aka column major).
     */
    public open fun isFortranContiguous(stride: List<Int>): Boolean {
        if (this.value.size != stride.size) {
            return false
        }
        var acc = 1
        for ((stride1, dim) in (stride.zip(this.value))) {
            if (dim > 1 && stride1 != acc) {
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
    public open fun extend(additionalDims: List<Int>): Shape {
        this.value.extend(additionalDims)
        return this
    }

    /**
     * Check whether the two shapes are compatible for broadcast, and if it is the case return the
     * broadcasted shape. This is to be used for binary pointwise ops.
     */
    public open fun broadcastShapeBinaryOp(rhs: Shape, op: String): Result<Shape> {
        val lhs = this
        val lhsDims = lhs.dims()
        val rhsDims = rhs.dims()
        val lhsNdims = lhsDims.size
        val rhsNdims = rhsDims.size
        val bcastNdims = Int.max(lhsNdims, rhsNdims)
        val bcastDims = mutableListOf(0, bcastNdims)
        for ((idx, bcastValue) in ArrayDeque(bcastDims).enumerate()) {
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
            bcastValue = if (lValue == rValue) {
                lValue
            } else if (lValue == 1) {
                rValue
            } else if (rValue == 1) {
                lValue
            } else {
                Result.failure(Error.ShapeMismatchBinaryOp(lhs = lhs.clone(), rhs = rhs.clone(), op = op).bt()).getOrElse { return Result.failure(it) }
            }
        }
        return Result.success(Shape.from(bcastDims))
    }

    internal open fun broadcastShapeMatmul(rhs: Shape): Result<Pair<Shape, Shape>> {
        val lhs = this
        val lhsDims = lhs.dims()
        val rhsDims = rhs.dims()
        if (lhsDims.size < 2 || rhsDims.size < 2) {
            bail("only 2d matrixes are supported {lhs:?} {rhs:?}")
        }
        val (m, lhsK) = Pair(lhsDims[lhsDims.size - 2], lhsDims[lhsDims.size - 1])
        val (rhsK, n) = Pair(rhsDims[rhsDims.size - 2], rhsDims[rhsDims.size - 1])
        if (lhsK != rhsK) {
            bail("different inner dimensions in broadcast matmul {lhs:?} {rhs:?}")
        }
        val lhsB = Shape.from(lhsDims[0 until lhsDims.size - 2])
        val rhsB = Shape.from(rhsDims[0 until rhsDims.size - 2])
        val bcast = lhsB.broadcastShapeBinaryOp(rhsB, "broadcast_matmul").getOrElse { return Result.failure(it) }
        val bcastDims = bcast.dims()
        val bcastLhs = listOf(bcastDims, listOf(m, lhsK)).concat()
        val bcastRhs = listOf(bcastDims, listOf(rhsK, n)).concat()
        return Result.success(Pair(Shape.from(bcastLhs), Shape.from(bcastRhs)))
    }

    public open fun dims0(): Result<Unit> = dims0(this.value.asSlice())

    public open fun tryInto(): Result<Unit> = this.dims0()

    public open fun dims1(): Result<Int> = dims1(this.value.asSlice())

    public open fun tryIntoInt(): Result<Int> = this.dims1()

    public open fun dims2(): Result<Pair<Int, Int>> = dims2(this.value.asSlice())

    public open fun tryIntoIntInt(): Result<Pair<Int, Int>> = this.dims2()

    public open fun dims3(): Result<Triple<Int, Int, Int>> = dims3(this.value.asSlice())

    public open fun tryIntoIntIntInt(): Result<Triple<Int, Int, Int>> = this.dims3()

    public open fun dims4(): Result<List<Any?>> = dims4(this.value.asSlice())

    public open fun tryIntoIntIntIntInt(): Result<List<Any?>> = this.dims4()

    public open fun dims5(): Result<List<Any?>> = dims5(this.value.asSlice())

    public open fun tryIntoIntIntIntIntInt(): Result<List<Any?>> = this.dims5()

    public companion object {
        public fun from(dims: List<ULong>, c: Int): Shape = Shape(dims.toVec())

        public fun from(dims: List<ULong>): Shape = Shape(dims.toVec())

        public fun from(shape: Shape): Shape = Shape(shape.value.toVec())

        public fun from(arg0: Unit): Shape = Shape(mutableListOf())

        public fun from(d1: ULong): Shape = Shape(mutableListOf(d1))

        public fun fromInt(d: ULong): Shape = Shape(mutableListOf(d.value))

        public fun from(d: Pair<ULong, ULong>): Shape = Shape(mutableListOf(d.value, d.field1))

        public fun from(d: Triple<ULong, ULong, ULong>): Shape = Shape(mutableListOf(d.value, d.field1, d.field2))

        public fun fromIntIntIntInt(d: List<Any?>): Shape = Shape(mutableListOf(d.value, d.field1, d.field2, d.field3))

        public fun fromIntIntIntIntInt(d: List<Any?>): Shape = Shape(mutableListOf(d.value, d.field1, d.field2, d.field3, d.field4))

        public fun fromIntIntIntIntIntInt(d: List<Any?>): Shape = Shape(mutableListOf(d.value, d.field1, d.field2, d.field3, d.field4, d.field5))

        public fun fromVecInt(dims: List<ULong>): Shape = Shape(dims)

        public fun fromDims(dims: List<ULong>): Shape = Shape(dims.toVec())
    }
}

public val SCALAR: Shape = Shape(mutableListOf())

public interface Dim {
    public fun toIndex(shape: Shape, op: String): Result<Int>

    public fun toIndexPlusOne(shape: Shape, op: String): Result<Int>
}

public fun Int.toIndex(shape: Shape, op: String): Result<Int> {
    val dim = this
    return if (dim >= shape.dims().size) {
        Result.failure(Error.DimOutOfRange(shape = shape.clone(), dim = dim, op = op).bt()).getOrElse { return Result.failure(it) }
    } else {
        Result.success(dim)
    }
}

public fun Int.toIndexPlusOne(shape: Shape, op: String): Result<Int> {
    val dim = this
    return if (dim > shape.dims().size) {
        Result.failure(Error.DimOutOfRange(shape = shape.clone(), dim = dim, op = op).bt()).getOrElse { return Result.failure(it) }
    } else {
        Result.success(dim)
    }
}

public sealed class D : Dim {
    public data object Minus1 : D()
    public data object Minus2 : D()
    public data class Minus(public var value: Int) : D()

    public fun __rustEq(other: D): Boolean = this == other

    public open fun outOfRange(shape: Shape, op: String): Error {
        val dim =
            when (this) {
                D.Minus1 -> -1
                D.Minus2 -> -2
                is Minus -> {
                    val (u) = this
                    -(u.toInt())
                }
            }
        return Error.DimOutOfRange(shape = shape.clone(), dim = dim, op = op).bt()
    }

    public override fun toIndex(shape: Shape, op: String): Result<Int> {
        val rank = shape.rank()
        return when {
            this == D.Minus1 && rank >= 1 -> Result.success(rank - 1)
            this == D.Minus2 && rank >= 2 -> Result.success(rank - 2)
            this is Minus && (this as Minus).value > 0 && rank >= (this as Minus).value -> {
                val (u) = this as Minus
                Result.success(rank - u)
            }
            else -> Result.failure(this.outOfRange(shape, op))
        }
    }

    public override fun toIndexPlusOne(shape: Shape, op: String): Result<Int> {
        val rank = shape.rank()
        return when {
            this == D.Minus1 -> Result.success(rank)
            this == D.Minus2 && rank >= 1 -> Result.success(rank - 1)
            this is Minus && (this as Minus).value > 0 && rank + 1 >= (this as Minus).value -> {
                val (u) = this as Minus
                Result.success(rank + 1 - u)
            }
            else -> Result.failure(this.outOfRange(shape, op))
        }
    }
}

public interface Dims {
    public fun toIndexesInternal(shape: Shape, op: String): Result<List<Int>>

    public fun toIndexes(shape: Shape, op: String): Result<List<Int>> {
        val dims = this.toIndexesInternal(shape, op).getOrElse { return Result.failure(it) }
        for ((i, dim) in dims.iter().enumerate()) {
            if (dims[0 until i].contains(dim)) {
                Result.failure(Error.DuplicateDimIndex(shape = shape.clone(), dims = dims.clone(), op = op).bt()).getOrElse { return Result.failure(it) }
            }
            if (dim >= shape.rank()) {
                Result.failure(Error.DimOutOfRange(shape = shape.clone(), dim = dim.toInt(), op = op).bt()).getOrElse { return Result.failure(it) }
            }
        }
        return Result.success(dims)
    }
}

public fun List<Int>.toIndexesInternal(arg0: Shape, arg1: String): Result<List<ULong>> = Result.success(this)

public fun List<Int>.toIndexesInternalFromIntArrayN(arg0: Shape, arg1: String, n: Int): Result<List<ULong>> = Result.success(this.toVec())

public fun Unit.toIndexesInternal(arg0: Shape, arg1: String): Result<List<ULong>> = Result.success(mutableListOf())

public fun <D : Dim> D.toIndexesInternal(shape: Shape, op: String): Result<List<Int>> {
    val dim = this.toIndex(shape, op).getOrElse { return Result.failure(it) }
    return Result.success(mutableListOf(dim))
}

public fun <D1 : Dim, D2 : Dim> Pair<D1, D2>.toIndexesInternal(shape: Shape, op: String): Result<List<Int>> {
    val d0 = this.value.toIndex(shape, op).getOrElse { return Result.failure(it) }
    val d1 = this.field1.toIndex(shape, op).getOrElse { return Result.failure(it) }
    return Result.success(mutableListOf(d0, d1))
}

public fun <D1 : Dim, D2 : Dim, D3 : Dim> Triple<D1, D2, D3>.toIndexesInternal(shape: Shape, op: String): Result<List<Int>> {
    val d0 = this.value.toIndex(shape, op).getOrElse { return Result.failure(it) }
    val d1 = this.field1.toIndex(shape, op).getOrElse { return Result.failure(it) }
    val d2 = this.field2.toIndex(shape, op).getOrElse { return Result.failure(it) }
    return Result.success(mutableListOf(d0, d1, d2))
}

public fun dims0(dims: List<Int>): Result<Unit> =
    if (dims.size != 0) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 0, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ Unit(dims) })
    }

public fun Tensor.dims0(): Result<Unit> = this.shape().dims0()

public fun dims1(dims: List<Int>): Result<Int> =
    if (dims.size != 1) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 1, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ d: List<Int> -> d[0](dims) })
    }

public fun Tensor.dims1(): Result<Int> = this.shape().dims1()

public fun dims2(dims: List<Int>): Result<Pair<Int, Int>> =
    if (dims.size != 2) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 2, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ d: List<Int> -> Pair(d[0], d[1])(dims) })
    }

public fun Tensor.dims2(): Result<Pair<Int, Int>> = this.shape().dims2()

public fun dims3(dims: List<Int>): Result<Triple<Int, Int, Int>> =
    if (dims.size != 3) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 3, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ d: List<Int> -> Triple(d[0], d[1], d[2])(dims) })
    }

public fun Tensor.dims3(): Result<Triple<Int, Int, Int>> = this.shape().dims3()

public fun dims4(dims: List<Int>): Result<List<Any?>> =
    if (dims.size != 4) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 4, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ d: List<Int> -> listOf(d[0], d[1], d[2], d[3])(dims) })
    }

public fun Tensor.dims4(): Result<List<Any?>> = this.shape().dims4()

public fun dims5(dims: List<Int>): Result<List<Any?>> =
    if (dims.size != 5) {
        Result.failure(Error.UnexpectedNumberOfDims(expected = 5, got = dims.size, shape = Shape.from(dims)).bt())
    } else {
        Result.success({ d: List<Int> -> listOf(d[0], d[1], d[2], d[3], d[4])(dims) })
    }

public fun Tensor.dims5(): Result<List<Any?>> = this.shape().dims5()

public interface ShapeWithOneHole {
    public fun intoShape(elCount: Int): Result<Shape>
}

public fun S.intoShape(elCount: Int): Result<Shape> {
    val shape = this.into()
    if (shape.elemCount() != elCount) {
        return Result.failure(Error.ShapeMismatch(bufferSize = elCount, shape = shape).bt())
    }
    return Result.success(shape)
}

public fun Unit.intoShape(elCount: ULong): Result<Shape> = Result.success(elCount.into())

public fun holeSize(elCount: Int, prodD: Int, s: Any): Result<Int> {
    if (prodD == 0) {
        bail("cannot reshape tensor of {el_count} elements to {s:?}")
    }
    if (!elCount.isMultipleOf(prodD)) {
        bail("cannot reshape tensor with {el_count} elements to {s:?}")
    }
    return Result.success(elCount / prodD)
}

public fun Pair<Unit, Int>.intoShape(elCount: Int): Result<Shape> {
    val (_, d1) = this
    return Result.success(Pair(holeSize(elCount, d1, this).getOrElse { return Result.failure(it) }, d1).into())
}

public fun Triple<Unit, Int, Int>.intoShape(elCount: Int): Result<Shape> {
    val (_, d1, d2) = this
    return Result.success(Triple(holeSize(elCount, d1 * d2, this).getOrElse { return Result.failure(it) }, d1, d2).into())
}

public fun List<Any?>.intoShape(elCount: Int): Result<Shape> {
    val (_, d1, d2, d3) = this
    val d = holeSize(elCount, d1 * d2 * d3, this).getOrElse { return Result.failure(it) }
    return Result.success(listOf(d, d1, d2, d3).into())
}
