// port-lint: source tensor.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.byteorder.BigEndian
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import kotlin.math.floor
import kotlin.math.pow

/**
 * Unique identifier for a tensor.
 */
public class TensorId private constructor(
    public val id: Long,
) {
    public companion object {
        private var counter: Long = 1L

        public fun next(): TensorId = TensorId(counter++)
    }

    override fun equals(other: Any?): Boolean = other is TensorId && id == other.id

    override fun hashCode(): Int = id.hashCode()

    override fun toString(): String = "TensorId($id)"
}

/**
 * Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
 */
internal fun fromStorage(
    storage: Storage,
    shape: Shape,
    op: BackpropOp = BackpropOp.None,
    isVariable: Boolean = false,
): Tensor = Tensor.fromStorage(storage, shape, op, isVariable)

/**
 * Tensors are N-dimensional matrixes of elements using a single data type.
 *
 * The core class for manipulating tensors.
 *
 * ```kotlin
 * val a = Tensor.arange(0f, 6f, Device.Cpu).reshape(2, 3)
 * val b = Tensor.arange(0f, 12f, Device.Cpu).reshape(3, 4)
 *
 * val c = a.matmul(b)
 * ```
 *
 * Tensors share underlying storage views so cloning and reshaping them is cheap.
 */
public class Tensor internal constructor(
    private val id: TensorId,
    private var storage: Storage,
    private val layout: Layout,
    private val op: BackpropOp,
    private val isVariable: Boolean,
) {
    public interface Target

    public interface Output

    /**
     * The unique identifier for this tensor.
     */
    public fun id(): TensorId = id

    /**
     * The tensor shape, i.e. dimension sizes on each axis.
     */
    public fun shape(): Shape = layout.shape

    /**
     * The dimension size for this tensor on each axis.
     */
    public fun dims(): List<Int> = layout.dims

    /**
     * Extracts the 0D scalar dimensions of this tensor, verifying that the rank is 0.
     */
    public fun dims0() {
        shape().dims0()
    }

    /**
     * Extracts the single dimension size of a 1D tensor.
     */
    public fun dims1(): Int = shape().dims1()

    /**
     * Extracts the two dimension sizes of a 2D tensor.
     */
    public fun dims2(): Dims2 = shape().dims2()

    /**
     * Extracts the three dimension sizes of a 3D tensor.
     */
    public fun dims3(): Dims3 = shape().dims3()

    /**
     * Extracts the four dimension sizes of a 4D tensor.
     */
    public fun dims4(): Dims4 = shape().dims4()

    /**
     * Extracts the five dimension sizes of a 5D tensor.
     */
    public fun dims5(): Dims5 = shape().dims5()

    /**
     * Returns the start and end offsets if the tensor layout is contiguous.
     */
    public fun contiguousOffsets(): ContiguousOffsets? = layout.contiguousOffsets()

    /**
     * The dimension size for a specified dimension index.
     */
    public fun dim(d: Int): Int = layout.dim(d)

    /**
     * The dimension size for a specified dimension index.
     */
    public fun dim(d: Dim): Int = layout.dim(d.toIndex(shape(), "dim"))

    /**
     * Returns a new tensor that is a narrowed version of the input, the dimension `dim`
     * ranges from `start` to `start + len`.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(3, 3), Device.Cpu)
     * val b = a.narrow(0, 1, 2)
     * val c = a.narrow(1, 1, 1)
     * ```
     */
    public fun narrow(
        dim: Dim,
        start: Int,
        len: Int,
    ): Tensor = narrow(dim.toIndex(shape(), "narrow"), start, len)

    /**
     * Run the `forward` method of `m` on `self`.
     */
    public fun apply(m: Module): Tensor = m.forward(this)

    /**
     * Run the `forward` method of `m` on `self` with train mode.
     */
    public fun applyT(
        m: ModuleT,
        train: Boolean,
    ): Tensor = m.forwardT(this, train)

    /**
     * Checks that the specified dimension index is within the valid bounds of this tensor.
     */
    public fun checkDim(
        dim: Int,
        op: String,
    ) {
        if (dim < 0 || dim >= dims().size) {
            throw CandleException.DimOutOfRange(shape(), dim, op)
        }
    }

    /**
     * The number of dimensions for this tensor, 0 for a scalar tensor, 1 for a 1D tensor, etc.
     */
    public fun rank(): Int = layout.shape.rank

    /**
     * The number of elements stored in this tensor.
     */
    public fun elemCount(): Int = layout.shape.elemCount()

    /**
     * The dtype for the elements stored in the input tensor.
     */
    public fun dtype(): DType = storage.dtype()

    /**
     * The device on which the input tensor is located.
     */
    public fun device(): Device = storage.device()

    /**
     * The layout of the input tensor, this stores both the shape of the tensor as well as the
     * strides and the start offset to apply to the underlying storage.
     */
    public fun layout(): Layout = layout

    /**
     * The stride of the input tensor along each dimension.
     */
    public fun stride(): List<Int> = layout.stride

    /**
     * Returns true if the data is stored in a C contiguous (aka row major) way.
     */
    public fun isContiguous(): Boolean = layout.isContiguous()

    /**
     * Returns true if the data is stored in a Fortran contiguous (aka column major) way.
     */
    public fun isFortranContiguous(): Boolean = layout.isFortranContiguous()

    /**
     * Whether this tensor is a variable or not. A variable is a tensor for which gradient is
     * tracked and on which backpropagation can be performed.
     */
    public fun isVariable(): Boolean = isVariable

    /**
     * Returns a read reference to the underlying storage of this tensor.
     */
    public fun storage(): Storage = storage

    /**
     * Returns a mutable reference to the underlying storage of this tensor.
     */
    public fun storageMut(): Storage = storage

    /**
     * The storage used by this tensor, together with the layout to use to access it safely.
     */
    public fun storageAndLayout(): Pair<Storage, Layout> = Pair(storage, layout)

    /**
     * The mutable storage used by this tensor, together with the layout to use to access it safely.
     */
    public fun storageMutAndLayout(): Pair<Storage, Layout> = Pair(storage, layout)

    /**
     * Returns the backpropagation operation associated with this tensor in the computation graph.
     */
    public fun op(): Op? = op.toOp()

    /**
     * Returns true if the computation graph should track this op, that is if it is
     * a variable or if it has some variable as dependencies.
     */
    public fun trackOp(): Boolean = isVariable || op.toOp() != null

    /**
     * Sets constant value across all elements of the tensor.
     */
    public fun constSet(value: Scalar) {
        storage.constSet(value, layout)
    }

    /**
     * Sets zero value across all elements of the tensor.
     */
    public fun zeroSet() {
        constSet(Scalar.zero(dtype()))
    }

    /**
     * Sets one value across all elements of the tensor.
     */
    public fun oneSet() {
        constSet(Scalar.one(dtype()))
    }

    /**
     * Creates a new tensor filled with zeros with same shape, dtype, and device as the other
     * tensor.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val b = a.zerosLike()
     * // b is on CPU f32.
     * ```
     */
    public fun zerosLike(): Tensor = zeros(shape(), dtype(), device())

    /**
     * Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val b = a.onesLike()
     * // b == a + 1
     * ```
     */
    public fun onesLike(): Tensor = ones(shape(), dtype(), device())

    /**
     * Creates a new tensor filled with uninitialized memory of the same shape, dtype, and device as the other
     * tensor.
     *
     * # Safety
     * This returns uninitialized memory.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val b = a.emptyLike()
     * ```
     */
    public fun emptyLike(): Tensor = empty(shape(), dtype(), device())

    /**
     * Creates a new tensor initialized with values sampled uniformly between `lo` and `up`
     * with the same shape, dtype, and device as this tensor.
     */
    public fun randLike(
        lo: Double,
        up: Double,
    ): Tensor = randF64Impl(lo, up, shape(), dtype(), device(), false)

    /**
     * Creates a new tensor initialized with values sampled from a normal distribution with the
     * specified `mean` and standard deviation `stdev` with the same shape, dtype, and device.
     */
    public fun randnLike(
        mean: Double,
        stdev: Double,
    ): Tensor = randnF64Impl(mean, stdev, shape(), dtype(), device(), false)

    /**
     * Returns a new tensor detached from the current graph, gradient are not propagated through
     * this new node. The storage of this tensor is shared with the initial tensor.
     *
     * If the tensor is already detached from the computation graph, the same tensor is returned.
     */
    public fun detach(): Tensor = Tensor(TensorId.next(), storage, layout, BackpropOp.None, false)

    internal fun sameStorage(rhs: Tensor): Boolean = this.storage === rhs.storage

    internal fun sameShapeBinaryOp(
        rhs: Tensor,
        op: String,
    ): Shape {
        val lhs = shape()
        val r = rhs.shape()
        if (lhs != r) {
            throw CandleException.ShapeMismatchBinaryOp(lhs, r, op)
        }
        return lhs
    }

    // --- Unary Ops ---

    /**
     * Computes the exponential of each element of the input tensor.
     */
    public fun exp(): Tensor = unaryOp(UnaryOp.Exp)

    /**
     * Computes the natural logarithm of each element of the input tensor.
     */
    public fun log(): Tensor = unaryOp(UnaryOp.Log)

    /**
     * Computes the sine of each element of the input tensor.
     */
    public fun sin(): Tensor = unaryOp(UnaryOp.Sin)

    /**
     * Computes the cosine of each element of the input tensor.
     */
    public fun cos(): Tensor = unaryOp(UnaryOp.Cos)

    /**
     * Computes the absolute value of each element of the input tensor.
     */
    public fun abs(): Tensor = unaryOp(UnaryOp.Abs)

    /**
     * Negates each element of the input tensor.
     */
    public fun neg(): Tensor = unaryOp(UnaryOp.Neg)

    /**
     * Unary minus operator negating each element of the input tensor.
     */
    public operator fun unaryMinus(): Tensor = neg()

    /**
     * Computes the reciprocal `1 / x` of each element of the input tensor.
     */
    public fun recip(): Tensor = unaryOp(UnaryOp.Recip)

    /**
     * Computes the square `x * x` of each element of the input tensor.
     */
    public fun sqr(): Tensor = unaryOp(UnaryOp.Sqr)

    /**
     * Computes the square root of each element of the input tensor.
     */
    public fun sqrt(): Tensor = unaryOp(UnaryOp.Sqrt)

    /**
     * Applies the Gaussian Error Linear Unit (GELU) tanh approximation function on each element.
     */
    public fun gelu(): Tensor = unaryOp(UnaryOp.Gelu)

    /**
     * Applies the exact Gaussian Error Linear Unit (GELU) function using the error function `erf`.
     */
    public fun geluErf(): Tensor = unaryOp(UnaryOp.GeluErf)

    /**
     * Computes the Gauss error function `erf` of each element of the input tensor.
     */
    public fun erf(): Tensor = unaryOp(UnaryOp.Erf)

    /**
     * Applies the Rectified Linear Unit (ReLU) activation `max(0, x)` on each element.
     */
    public fun relu(): Tensor = unaryOp(UnaryOp.Relu)

    /**
     * Applies the Sigmoid Linear Unit (SiLU / Swish) activation `x / (1 + exp(-x))` on each element.
     */
    public fun silu(): Tensor = unaryOp(UnaryOp.Silu)

    /**
     * Computes the hyperbolic tangent `tanh` of each element of the input tensor.
     */
    public fun tanh(): Tensor = unaryOp(UnaryOp.Tanh)

    /**
     * Returns the largest integer less than or equal to each element of the input tensor.
     */
    public fun floor(): Tensor = unaryOp(UnaryOp.Floor)

    /**
     * Returns the smallest integer greater than or equal to each element of the input tensor.
     */
    public fun ceil(): Tensor = unaryOp(UnaryOp.Ceil)

    /**
     * Rounds each element of the input tensor to the nearest integer.
     */
    public fun round(): Tensor = unaryOp(UnaryOp.Round)

    /**
     * Returns the sign of each element of the input tensor (`1`, `-1`, or `0`).
     */
    public fun sign(): Tensor = unaryOp(UnaryOp.Sign)

    private fun unaryOp(uOp: UnaryOp): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.unaryImpl(layout, uOp)
        val newOp = BackpropOp.new1(this) { Op.Unary(it, uOp) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Round element of the input tensor to the nearest integer.
     *
     * If the number of decimals is negative, it specifies the number of positions to the left of
     * the decimal point.
     */
    public fun roundTo(decimals: Int): Tensor {
        val mult = 10.0.pow(decimals.toDouble())
        return (affine(mult, 0.0)).round().affine(1.0 / mult, 0.0)
    }

    /**
     * Repeat this tensor along the specified dimensions.
     */
    public fun repeat(shape: Shape): Tensor {
        val repeats = shape.dims
        var inp =
            if (rank() < repeats.size) {
                val padCount = repeats.size - rank()
                val newDims = List(padCount) { 1 } + dims()
                reshape(Shape(newDims))
            } else {
                this
            }
        for (idx in repeats.indices) {
            val repeat = repeats[idx]
            if (repeat > 1) {
                val list = List(repeat) { inp }
                inp = cat(list, idx)
            }
        }
        return inp
    }

    /**
     * This operation multiplies the input tensor by `mul` then adds `add` and return the result.
     * The input values `mul` and `add` are casted to the appropriate type so some rounding might
     * be performed.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f), Shape(2, 2), Device.Cpu)
     * val res = a.affine(4.0, -2.0)
     * // [[-2.0, 2.0], [6.0, 10.0]]
     * ```
     */
    public fun affine(
        mul: Double,
        add: Double,
    ): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.affine(layout, mul, add)
        val newOp = BackpropOp.new1(this) { Op.Affine(it, mul, add) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Raise the tensor to some float exponent `e`.
     */
    public fun powf(e: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.powf(layout, e)
        val newOp = BackpropOp.new1(this) { Op.Powf(it, e) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Applies the Exponential Linear Unit (ELU) function on each element of the input tensor.
     */
    public fun elu(alpha: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.elu(layout, alpha)
        val newOp = BackpropOp.new1(this) { Op.Elu(it, alpha) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Split a tensor into the specified number of chunks, this may return less chunks than
     * specified.
     */
    public fun chunk(
        chunks: Int,
        dim: Int,
    ): List<Tensor> {
        val dimIdx = Shape.toIndex(dim, shape(), "chunk")
        val size = dims()[dimIdx]
        return if (size < chunks) {
            (0 until size).map { narrow(dimIdx, it, 1) }
        } else {
            val chunkSize = size / chunks
            val cntAdditional = size % chunks
            val tensors = ArrayList<Tensor>(chunks)
            var sumChunkSize = 0
            for (i in 0 until chunks) {
                val currentChunkSize = if (i < cntAdditional) chunkSize + 1 else chunkSize
                val tensor = narrow(dimIdx, sumChunkSize, currentChunkSize)
                tensors.add(tensor)
                sumChunkSize += currentChunkSize
            }
            tensors
        }
    }

    /**
     * Split a tensor into the specified number of chunks, this may return less chunks than
     * specified.
     */
    public fun chunk(
        chunks: Int,
        dim: Dim,
    ): List<Tensor> = chunk(chunks, dim.toIndex(shape(), "chunk"))

    internal fun squeezeDims(dims: List<Int>): Tensor =
        when (dims.size) {
            0 -> {
                this
            }

            1 -> {
                squeeze(dims[0])
            }

            else -> {
                val remainingDims = dims().filterIndexed { idx, _ -> idx !in dims }
                reshape(Shape(remainingDims))
            }
        }

    internal fun reduceImpl(
        dim: Int,
        keepdim: Boolean,
        op: ReduceOp,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), op.name)
        val newStorage = storage.reduceOp(op, layout, listOf(dimIdx))
        val newDims = dims().toMutableList()
        newDims[dimIdx] = 1
        val newOp =
            when (op) {
                ReduceOp.Sum, ReduceOp.Min, ReduceOp.Max -> {
                    BackpropOp.new1(this) { Op.Reduce(it, op, newDims) }
                }

                ReduceOp.ArgMin, ReduceOp.ArgMax -> {
                    BackpropOp.None
                }
            }
        val res = fromStorage(newStorage, Shape(newDims), newOp, false)
        return if (keepdim) res else res.squeezeDims(listOf(dimIdx))
    }

    internal fun sumImpl(
        sumDims: List<Int>,
        keepdim: Boolean,
    ): Tensor {
        val normalizedDims = sumDims.map { Shape.toIndex(it, shape(), "sum") }
        val newStorage = storage.reduceOp(ReduceOp.Sum, layout, normalizedDims)
        val newDims = dims().toMutableList()
        for (d in normalizedDims) {
            newDims[d] = 1
        }
        val newOp = BackpropOp.new1(this) { Op.Reduce(it, ReduceOp.Sum, newDims) }
        val sum = fromStorage(newStorage, Shape(newDims), newOp, false)
        return if (keepdim) sum else sum.squeezeDims(normalizedDims)
    }

    /**
     * Roll the tensor input along the given dimension.
     * Elements that are shifted beyond the last position are re-introduced at the first position.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val rolledDown = tensor.roll(1, 0)
     * // [[4.0, 5.0], [0.0, 1.0], [2.0, 3.0]]
     * val rolledUp = tensor.roll(-1, 0)
     * // [[2.0, 3.0], [4.0, 5.0], [0.0, 1.0]]
     * ```
     */
    public fun roll(
        shift: Int,
        dim: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "roll")
        val dimSize = dims()[dimIdx]
        val rem = shift.rem(dimSize)
        val normalizedShift = if (rem < 0) rem + dimSize else rem
        if (normalizedShift == 0) return this
        val a = narrow(dimIdx, 0, dimSize - normalizedShift)
        val b = narrow(dimIdx, dimSize - normalizedShift, normalizedShift)
        return cat(listOf(b, a), dimIdx)
    }

    /**
     * Roll the tensor input along the given dimension.
     * Elements that are shifted beyond the last position are re-introduced at the first position.
     */
    public fun roll(
        shift: Int,
        dim: Dim,
    ): Tensor = roll(shift, dim.toIndex(shape(), "roll"))

    // --- Binary Ops ---

    /**
     * Element-wise addition of two tensors.
     */
    public fun add(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Add)

    /**
     * Adds a scalar value to each element of the tensor.
     */
    public fun add(rhs: Double): Tensor = affine(1.0, rhs)

    /**
     * Operator overload for element-wise tensor addition.
     */
    public operator fun plus(rhs: Tensor): Tensor = add(rhs)

    /**
     * Element-wise subtraction of two tensors.
     */
    public fun sub(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Sub)

    /**
     * Subtracts a scalar value from each element of the tensor.
     */
    public fun sub(rhs: Double): Tensor = affine(1.0, -rhs)

    /**
     * Operator overload for element-wise tensor subtraction.
     */
    public operator fun minus(rhs: Tensor): Tensor = sub(rhs)

    /**
     * Element-wise multiplication of two tensors.
     */
    public fun mul(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Mul)

    /**
     * Multiplies each element of the tensor by a scalar value.
     */
    public fun mul(rhs: Double): Tensor = affine(rhs, 0.0)

    /**
     * Operator overload for element-wise tensor multiplication.
     */
    public operator fun times(rhs: Tensor): Tensor = mul(rhs)

    /**
     * Element-wise division of two tensors.
     */
    public operator fun div(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Div)

    /**
     * Divides each element of the tensor by a scalar value.
     */
    public operator fun div(rhs: Double): Tensor = affine(1.0 / rhs, 0.0)

    /**
     * Computes the element-wise maximum of two tensors.
     */
    public fun maximum(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Maximum)

    /**
     * Computes the element-wise minimum of two tensors.
     */
    public fun minimum(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Minimum)

    private fun binaryOp(
        rhs: Tensor,
        bOp: BinaryOp,
    ): Tensor {
        val targetShape = shape().broadcastShapeBinaryOp(rhs.shape(), bOp.name)
        val lhsTensor = if (shape() != targetShape) broadcastAs(targetShape) else this
        val rhsTensor = if (rhs.shape() != targetShape) rhs.broadcastAs(targetShape) else rhs
        val newStorage = lhsTensor.storage.binaryImpl(rhsTensor.storage, lhsTensor.layout, rhsTensor.layout, bOp)
        val newOp = BackpropOp.new2(lhsTensor, rhsTensor) { l, r -> Op.Binary(l, r, bOp) }
        return fromStorage(newStorage, targetShape, newOp, false)
    }

    /**
     * Element-wise addition with broadcasting support.
     */
    public fun broadcastAdd(rhs: Tensor): Tensor = add(rhs)

    /**
     * Element-wise subtraction with broadcasting support.
     */
    public fun broadcastSub(rhs: Tensor): Tensor = sub(rhs)

    /**
     * Element-wise multiplication with broadcasting support.
     */
    public fun broadcastMul(rhs: Tensor): Tensor = mul(rhs)

    /**
     * Element-wise division with broadcasting support.
     */
    public fun broadcastDiv(rhs: Tensor): Tensor = div(rhs)

    /**
     * Element-wise maximum with broadcasting support.
     */
    public fun broadcastMaximum(rhs: Tensor): Tensor = maximum(rhs)

    /**
     * Element-wise minimum with broadcasting support.
     */
    public fun broadcastMinimum(rhs: Tensor): Tensor = minimum(rhs)

    // --- Reductions ---

    /**
     * Returns the sum of all elements in the input tensor. The sum is performed over all the
     * input dimensions and compared to `sumKeepdim` these dimensions are squeezed rather than
     * kept.
     */
    public fun sum(dims: List<Int>): Tensor = sumImpl(dims, false)

    /**
     * Returns the sum of all elements in the input tensor over the specified dimensions.
     */
    public fun sum(vararg dims: Int): Tensor = sum(dims.toList())

    /**
     * Returns the sum of all elements in the input tensor. The sum is performed over all the
     * input dimensions.
     *
     * The resulting tensor has a shape that is similar to the shape of the input tensor, except
     * that the number of elements for each dimension index in `dim` is 1.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f), Shape(2, 2), Device.Cpu)
     * val s0 = a.sumKeepdim(0)
     * // [[2.0, 4.0]]
     * val s1 = a.sumKeepdim(1)
     * // [[1.0], [5.0]]
     * ```
     */
    public fun sumKeepdim(dim: Int): Tensor = sumImpl(listOf(dim), true)

    /**
     * Returns the sum of all elements in the input tensor preserving reduced dimensions as 1.
     */
    public fun sumKeepdim(dims: List<Int>): Tensor = sumImpl(dims, true)

    /**
     * Computes the sum of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val sum = tensor.sumAll()
     * // 15.0
     * ```
     */
    public fun sumAll(): Tensor = sum((0 until rank()).toList())

    /**
     * Similar to `minKeepdim` but the target dimension is squeezed.
     */
    public fun min(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.Min)

    /**
     * Gathers the minimum value across the selected dimension. The resulting shape has the same
     * number of dimensions as the original tensor and the select dimension has a single element.
     */
    public fun minKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.Min)

    /**
     * Computes the min of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val min = tensor.minAll()
     * // 0.0
     * ```
     */
    public fun minAll(): Tensor = if (rank() == 0) this else flattenAll().min(0)

    /**
     * Similar to `maxKeepdim` but the target dimension is squeezed.
     */
    public fun max(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.Max)

    /**
     * Gathers the maximum value across the selected dimension. The resulting shape has the same
     * number of dimensions as the original tensor and the select dimension has a single element.
     */
    public fun maxKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.Max)

    /**
     * Computes the max of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val max = tensor.maxAll()
     * // 5.0
     * ```
     */
    public fun maxAll(): Tensor = if (rank() == 0) this else flattenAll().max(0)

    /**
     * Similar to `argminKeepdim` but the target dimension is squeezed.
     */
    public fun argmin(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.ArgMin)

    /**
     * Returns the indices of the minimum values across the selected dimension keeping dimension as 1.
     */
    public fun argminKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.ArgMin)

    /**
     * Similar to `argmaxKeepdim` but the target dimension is squeezed.
     */
    public fun argmax(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.ArgMax)

    /**
     * Returns the indices of the maximum values across the selected dimension keeping dimension as 1.
     */
    public fun argmaxKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.ArgMax)

    /**
     * Returns the mean of all elements in the input tensor. The mean is performed over all the
     * input dimensions and compared to `meanKeepdim` these dimensions are squeezed rather than
     * kept.
     */
    public fun mean(dim: Int): Tensor = mean(listOf(dim))

    /**
     * Returns the mean of all elements in the input tensor over the specified dimensions.
     */
    public fun mean(meanDims: List<Int>): Tensor {
        val normalizedDims = meanDims.map { Shape.toIndex(it, shape(), "mean") }
        val reducedDim = normalizedDims.fold(1) { acc, d -> acc * dims()[d] }
        val scale = 1.0 / reducedDim.toDouble()
        return sumImpl(normalizedDims, false).affine(scale, 0.0)
    }

    /**
     * Returns the mean of all elements in the input tensor. The mean is performed over all the
     * input dimensions.
     *
     * The resulting tensor has a shape that is similar to the shape of the input tensor, except
     * that the number of elements for each dimension index in `dim` is 1.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f), Shape(2, 2), Device.Cpu)
     * val m0 = a.meanKeepdim(0)
     * // [[1.0, 2.0]]
     * val m1 = a.meanKeepdim(1)
     * // [[0.5], [2.5]]
     * ```
     */
    public fun meanKeepdim(dim: Int): Tensor = meanKeepdim(listOf(dim))

    /**
     * Returns the mean of all elements across the specified dimensions keeping reduced dimensions as 1.
     */
    public fun meanKeepdim(meanDims: List<Int>): Tensor {
        val normalizedDims = meanDims.map { Shape.toIndex(it, shape(), "mean-keepdim") }
        val reducedDim = normalizedDims.fold(1) { acc, d -> acc * dims()[d] }
        val scale = 1.0 / reducedDim.toDouble()
        return sumImpl(normalizedDims, true).affine(scale, 0.0)
    }

    /**
     * Computes the mean of all elements in the tensor.
     */
    public fun meanAll(): Tensor {
        val s = sumAll()
        return s.affine(1.0 / elemCount().toDouble(), 0.0)
    }

    /**
     * Returns the unbiased variance over the selected dimension keeping reduced dimension as 1.
     */
    public fun varKeepdim(dim: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "var")
        val meanVal = meanKeepdim(dimIdx)
        val squares = (this - meanVal).sqr()
        val dimSize = dims()[dimIdx]
        return squares.sumImpl(listOf(dimIdx), true).affine(1.0 / (dimSize - 1).toDouble(), 0.0)
    }

    /**
     * Returns the unbiased variance over the selected dimension.
     */
    public fun `var`(dim: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "var")
        return varKeepdim(dimIdx).squeeze(dimIdx)
    }

    /**
     * Returns the unbiased variance over the selected dimension.
     */
    public fun var_(dim: Dim): Tensor = `var`(dim.toIndex(shape(), "var"))

    // --- Comparisons ---

    /**
     * Element-wise comparison between two tensors, e.g. equality, greater than, ... The actual
     * comparison operation is specified by the `op` argument.
     *
     * The returned tensor has the same shape as the original tensors and uses `u8` elements.
     */
    public fun cmp(
        rhs: Tensor,
        op: CmpOp,
    ): Tensor {
        val targetShape = sameShapeBinaryOp(rhs, "cmp")
        val newStorage = storage.cmp(op, rhs.storage, layout, rhs.layout)
        val newOp = BackpropOp.new1(this) { Op.Cmp(it, op) }
        return fromStorage(newStorage, targetShape, newOp, false)
    }

    /**
     * Element-wise comparison with a tensor or scalar value.
     */
    public fun cmp(
        rhs: TensorOrScalar,
        op: CmpOp,
    ): Tensor {
        val r =
            when (val ts = rhs.toTensorScalar().getOrThrow()) {
                is TensorScalar.Tensor -> {
                    ts.tensor
                }

                is TensorScalar.Scalar -> {
                    ts.tensor
                        .toDtype(dtype())
                        .toDevice(device())
                        .broadcastAs(shape())
                }
            }
        return cmp(r, op)
    }

    /**
     * Element-wise equality.
     */
    public fun eq(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Eq)

    /**
     * Element-wise non-equality.
     */
    public fun ne(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Ne)

    /**
     * Element-wise comparison with lower-equal, the returned tensor uses value 1 where `self <=
     * rhs` and 0 otherwise.
     */
    public fun le(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Le)

    /**
     * Element-wise comparison with greater-equal, the returned tensor uses value 1 where `self >=
     * rhs` and 0 otherwise.
     */
    public fun ge(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Ge)

    /**
     * Element-wise comparison with lower-than, the returned tensor uses value 1 where `self <
     * rhs` and 0 otherwise.
     */
    public fun lt(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Lt)

    /**
     * Element-wise comparison with greater-than, the returned tensor uses value 1 where `self >
     * rhs` and 0 otherwise.
     */
    public fun gt(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Gt)

    /**
     * Element-wise equality with broadcasting support.
     */
    public fun broadcastEq(rhs: Tensor): Tensor = eq(rhs)

    /**
     * Element-wise non-equality with broadcasting support.
     */
    public fun broadcastNe(rhs: Tensor): Tensor = ne(rhs)

    /**
     * Element-wise lower-than comparison with broadcasting support.
     */
    public fun broadcastLt(rhs: Tensor): Tensor = lt(rhs)

    /**
     * Element-wise lower-equal comparison with broadcasting support.
     */
    public fun broadcastLe(rhs: Tensor): Tensor = le(rhs)

    /**
     * Element-wise greater-than comparison with broadcasting support.
     */
    public fun broadcastGt(rhs: Tensor): Tensor = gt(rhs)

    /**
     * Element-wise greater-equal comparison with broadcasting support.
     */
    public fun broadcastGe(rhs: Tensor): Tensor = ge(rhs)

    /**
     * Clamp the tensor values to be between `min` and `max`.
     */
    public fun clamp(
        min: Tensor,
        max: Tensor,
    ): Tensor = maximum(min).minimum(max)

    /**
     * Clamp the tensor values to be between `min` and `max` scalar values.
     */
    public fun clamp(
        min: Double,
        max: Double,
    ): Tensor {
        val minT =
            fromStorage(
                Storage.Cpu(CpuStorage.F64(doubleArrayOf(min))),
                Shape(listOf(1)),
                BackpropOp.None,
                false,
            ).broadcastAs(shape())
        val maxT =
            fromStorage(
                Storage.Cpu(CpuStorage.F64(doubleArrayOf(max))),
                Shape(listOf(1)),
                BackpropOp.None,
                false,
            ).broadcastAs(shape())
        return clamp(minT, maxT)
    }

    // --- Interpolation & Pooling ---

    /**
     * Interpolate the input tensor to the `targetSize` size, taking the value of the nearest element.
     *
     * The input tensor should have three dimensions, `(batch, channels, l)`, the returned
     * tensor also has three dimensions, `(batch, channels, targetSize)`.
     */
    public fun interpolate1d(targetSize: Int): Tensor {
        val d = dims3()
        val op = BackpropOp.new1(this) { Op.UpsampleNearest1D(it, targetSize) }
        val newStorage = storage.upsampleNearest1d(layout, targetSize)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, targetSize)), op, false)
    }

    /**
     * Alias for `interpolate1d`.
     */
    public fun upsampleNearest1d(targetSize: Int): Tensor = interpolate1d(targetSize)

    /**
     * Interpolate the input tensor to the `(targetH, targetW)` size, taking the value of the
     * nearest element.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, targetH, targetW)`.
     */
    public fun interpolate2d(
        targetH: Int,
        targetW: Int,
    ): Tensor {
        val d = dims4()
        val op = BackpropOp.new1(this) { Op.UpsampleNearest2D(it, targetH, targetW) }
        val newStorage = storage.upsampleNearest2d(layout, targetH, targetW)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, targetH, targetW)), op, false)
    }

    /**
     * Alias for `interpolate2d`.
     */
    public fun upsampleNearest2d(
        targetH: Int,
        targetW: Int,
    ): Tensor = interpolate2d(targetH, targetW)

    /**
     * Bilinear interpolation to resize the input tensor to the specified size.
     *
     * The input tensor should have four dimensions: `(batch, channels, h, w)`.
     * The returned tensor also has four dimensions: `(batch, channels, targetH, targetW)`.
     *
     * # Arguments
     *
     * * `targetH` - Target height
     * * `targetW` - Target width
     * * `alignCorners` - If true, corner pixels are aligned. If false (default),
     *   pixels are treated as areas (matches PyTorch default behavior).
     *
     * # Example
     *
     * ```kotlin
     * val t = Tensor.arange(0f, 16f, Device.Cpu).reshape(1, 1, 4, 4)
     * val upsampled = t.upsampleBilinear2d(8, 8, false)
     * // dims: [1, 1, 8, 8]
     * ```
     */
    public fun upsampleBilinear2d(
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean = false,
    ): Tensor {
        val d = dims4()
        val op = BackpropOp.new1(this) { Op.UpsampleBilinear2D(it, targetH, targetW, alignCorners) }
        val newStorage = storage.upsampleBilinear2d(layout, targetH, targetW, alignCorners, null, null)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, targetH, targetW)), op, false)
    }

    /**
     * Bilinear interpolation using scale factors.
     *
     * Similar to `upsampleBilinear2d` but uses scale factors instead of absolute sizes.
     * This matches PyTorch's `interpolate(scaleFactor=...)` behavior.
     *
     * # Arguments
     *
     * * `scaleH` - Height scaling factor
     * * `scaleW` - Width scaling factor
     * * `alignCorners` - If true, corner pixels are aligned
     *
     * # Example
     *
     * ```kotlin
     * val t = Tensor.arange(0f, 16f, Device.Cpu).reshape(1, 1, 4, 4)
     * // Scale by 2x in both dimensions
     * val upsampled = t.upsampleBilinear2dWithScale(2.0, 2.0, false)
     * // dims: [1, 1, 8, 8]
     * ```
     */
    public fun upsampleBilinear2dWithScale(
        scaleH: Double,
        scaleW: Double,
        alignCorners: Boolean = false,
    ): Tensor {
        val d = dims4()
        val heightOut = floor(d.d2.toDouble() * scaleH).toInt()
        val widthOut = floor(d.d3.toDouble() * scaleW).toInt()
        if (d.d2 == heightOut && d.d3 == widthOut) return this
        val op = BackpropOp.new1(this) { Op.UpsampleBilinear2D(it, heightOut, widthOut, alignCorners) }
        val newStorage = storage.upsampleBilinear2d(layout, heightOut, widthOut, alignCorners, scaleH, scaleW)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, heightOut, widthOut)), op, false)
    }

    /**
     * 2D average pooling over an input tensor with multiple channels.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
     * the two last dimensions using a kernel of size `sz`. The returned element is the average
     * value over the kernel window.
     */
    public fun avgPool2d(sz: Int): Tensor = avgPool2dWithStride(Pair(sz, sz), Pair(sz, sz))

    /**
     * 2D average pooling over an input tensor with multiple channels using a pair of kernel dimensions.
     */
    public fun avgPool2d(sz: Pair<Int, Int>): Tensor = avgPool2dWithStride(sz, sz)

    /**
     * Same as `avgPool2d` but with a `stride` that can be set to a value different from the
     * kernel size.
     */
    public fun avgPool2dWithStride(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Tensor {
        val d = dims4()
        if (d.d2 < kernelSize.first || d.d3 < kernelSize.second) {
            bail("kernel-size $kernelSize is larger than the input size ${d.d2},${d.d3}")
        }
        val hOut = (d.d2 - kernelSize.first) / stride.first + 1
        val wOut = (d.d3 - kernelSize.second) / stride.second + 1
        val op = BackpropOp.new1(this) { Op.AvgPool2D(it, kernelSize, stride) }
        val newStorage = storage.avgPool2d(layout, kernelSize, stride)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, hOut, wOut)), op, false)
    }

    /**
     * 2D max pooling over an input tensor with multiple channels.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
     * the two last dimensions using a kernel of size `sz`, the returned element is the maximum
     * value over the kernel window.
     */
    public fun maxPool2d(sz: Int): Tensor = maxPool2dWithStride(Pair(sz, sz), Pair(sz, sz))

    /**
     * 2D max pooling over an input tensor with multiple channels using a pair of kernel dimensions.
     */
    public fun maxPool2d(sz: Pair<Int, Int>): Tensor = maxPool2dWithStride(sz, sz)

    /**
     * Same as `maxPool2d` but with a `stride` that can be set to a value different from the
     * kernel size.
     */
    public fun maxPool2dWithStride(
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Tensor {
        val d = dims4()
        if (d.d2 < kernelSize.first || d.d3 < kernelSize.second) {
            bail("kernel-size $kernelSize is larger than the input size ${d.d2},${d.d3}")
        }
        val hOut = (d.d2 - kernelSize.first) / stride.first + 1
        val wOut = (d.d3 - kernelSize.second) / stride.second + 1
        val op = BackpropOp.new1(this) { Op.MaxPool2D(it, kernelSize, stride) }
        val newStorage = storage.maxPool2d(layout, kernelSize, stride)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, hOut, wOut)), op, false)
    }

    /**
     * Computes the dot product of two 1D tensors.
     *
     * - If inputs are 1D vectors (`[n]`), returns their scalar dot product.
     * - Throws if shapes are not compatible
     * - Not supported for integer dtypes
     *
     * # Example (vectors)
     * ```kotlin
     * val t1 = Tensor.new(doubleArrayOf(1.0, 2.0, 3.0), Shape(3), Device.Cpu)
     * val t2 = Tensor.new(doubleArrayOf(4.0, 5.0, 6.0), Shape(3), Device.Cpu)
     * val res = t1.dot(t2)
     * // 32.0
     * ```
     */
    public fun dot(rhs: Tensor): Tensor {
        if (dims().size != 1 || rhs.dims().size != 1) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), rhs.shape(), "dot")
        }
        return (this * rhs).sumAll()
    }

    /**
     * Computes the **Frobenius norm** (L2 norm of all elements) of the tensor.
     * - Output is `sqrt(sum(x^2))`.
     * - Always returns a scalar (`[]` shape).
     *
     * # Example
     * ```kotlin
     * val t = Tensor.new(doubleArrayOf(3.0, 4.0, 0.0, 0.0), Shape(2, 2), Device.Cpu)
     * val norm = t.norm()
     * // 5.0
     * ```
     */
    public fun norm(): Tensor {
        if (dtype().isInt()) {
            bail("norm not supported for integer dtypes")
        }
        return sqr().sumAll().sqrt()
    }

    /**
     * Performs strict matrix-vector multiplication (`[m, n] * [n] = [m]`).
     *
     * - If `self` is a matrix (`[m, n]`) and `rhs` is a vector (`[n]`), returns a vector (`[m]`).
     * - **No broadcasting**: Throws if `self` is not 2D or if `rhs` is not 1D with matching size.
     *
     * # Example
     * ```kotlin
     * val mat = Tensor.new(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Shape(2, 3), Device.Cpu)
     * val vec = Tensor.new(doubleArrayOf(1.0, 1.0, 1.0), Shape(3), Device.Cpu)
     * val res = mat.mv(vec)
     * // [6.0, 15.0]
     * ```
     */
    public fun mv(rhs: Tensor): Tensor {
        val lhsDims = dims()
        val rhsDims = rhs.dims()
        if (lhsDims.size != 2 || rhsDims.size != 1 || lhsDims[1] != rhsDims[0]) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), rhs.shape(), "mv")
        }
        return matmul(rhs.unsqueeze(1)).squeeze(1)
    }

    // --- Matrix Ops ---

    /**
     * Returns the matrix-multiplication of the input tensor with the other provided tensor.
     *
     * # Arguments
     *
     * * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
     * * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
     *
     * The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
     */
    public fun matmul(rhs: Tensor): Tensor {
        val (lhsBcastShape, rhsBcastShape) = shape().broadcastShapeMatmul(rhs.shape())
        val lhsTensor = if (shape() != lhsBcastShape) broadcastAs(lhsBcastShape) else this
        val rhsTensor = if (rhs.shape() != rhsBcastShape) rhs.broadcastAs(rhsBcastShape) else rhs

        val lhsDims = lhsBcastShape.dims
        val rhsDims = rhsBcastShape.dims
        val m = lhsDims[lhsDims.size - 2]
        val k = lhsDims[lhsDims.size - 1]
        val n = rhsDims[rhsDims.size - 1]
        val b = lhsDims.subList(0, lhsDims.size - 2).fold(1) { acc, d -> acc * d }

        val newStorage = lhsTensor.storage.matmul(rhsTensor.storage, listOf(b, m, n, k), lhsTensor.layout, rhsTensor.layout)
        val outDims = lhsDims.subList(0, lhsDims.size - 2) + listOf(m, n)
        val outShape = Shape(outDims)
        val newOp = BackpropOp.new2(lhsTensor, rhsTensor) { l, r -> Op.Matmul(l, r) }
        return fromStorage(newStorage, outShape, newOp, false)
    }

    /**
     * Matrix-multiplication with broadcasting support.
     *
     * Compared to `matmul` the two matrixes are allowed to have different dimensions as long as
     * they are compatible for broadcast. E.g. if `self` has shape `(j, 1, n, k)` and `rhs` has
     * shape `(l, k, m)`, the output will have shape `(j, l, n, m)`.
     */
    public fun broadcastMatmul(rhs: Tensor): Tensor = matmul(rhs)

    /**
     * Returns a tensor with the same shape as the input tensor, the values are taken from
     * `onTrue` if the input tensor value is not zero, and `onFalse` at the positions where the
     * input tensor is equal to zero.
     */
    public fun whereCond(
        onTrue: Tensor,
        onFalse: Tensor,
    ): Tensor {
        val targetShape =
            shape()
                .broadcastShapeBinaryOp(onTrue.shape(), "where_cond")
                .broadcastShapeBinaryOp(onFalse.shape(), "where_cond")
        val condL = layout.broadcastAs(targetShape)
        val trueL = onTrue.layout.broadcastAs(targetShape)
        val falseL = onFalse.layout.broadcastAs(targetShape)
        val newStorage = storage.whereCond(condL, onTrue.storage, trueL, onFalse.storage, falseL)
        val newOp = BackpropOp.new3(this, onTrue, onFalse) { t1, t2, t3 -> Op.WhereCond(t1, t2, t3) }
        return fromStorage(newStorage, targetShape, newOp, false)
    }

    /**
     * Returns a tensor with the values from the `self` tensor at the index corresponding to the
     * values hold in the `ids` tensor.
     *
     * # Arguments
     *
     * * `self` - A tensor with dimensions `v, h`.
     * * `ids` - A tensor with dimensions `s` and with integer values between 0 and v (exclusive).
     *
     * The resulting tensor has dimensions `s, h`. `s` is called the sequence length, `v` the
     * vocabulary size, and `h` the hidden size.
     *
     * ```kotlin
     * val values = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val ids = Tensor.new(uintArrayOf(2u, 1u, 2u), Shape(3), Device.Cpu)
     * val emb = values.embedding(ids)
     * // [[4.0, 5.0], [2.0, 3.0], [4.0, 5.0]]
     * ```
     */
    public fun embedding(ids: Tensor): Tensor {
        if (rank() != 2 || ids.rank() != 1) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), ids.shape(), "embedding")
        }
        return indexSelect(ids, 0)
    }

    internal fun scatterChecks(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ) {
        val sourceDims = source.dims()
        val selfDims = dims()
        val mismatch =
            if (sourceDims.size != selfDims.size) {
                true
            } else {
                var m = false
                for (i in selfDims.indices) {
                    if (i != dim && selfDims[i] != sourceDims[i]) {
                        m = true
                        break
                    }
                }
                m
            }
        if (mismatch) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), source.shape(), "scatter (self, src)")
        }
        if (indexes.dims() != source.dims()) {
            throw CandleException.ShapeMismatchBinaryOp(indexes.shape(), source.shape(), "scatter (indexes, src)")
        }
    }

    /**
     * Scatters values from `source` into a copy of `self` at the specified `indexes` along `dim`.
     */
    public fun scatter(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "scatter")
        scatterChecks(indexes, source, dimIdx)
        val sh = shape()
        val newStorage = storage.tryClone(layout)
        val l = Layout.contiguous(sh)
        newStorage.scatterSet(l, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
        val newOp = BackpropOp.new3(this, indexes, source) { t1, t2, t3 -> Op.Scatter(t1, t2, t3, dimIdx) }
        return fromStorage(newStorage, sh, newOp, false)
    }

    /**
     * Scatters values from `source` into a copy of `self` at the specified `indexes` along `dim`.
     */
    public fun scatter(
        indexes: Tensor,
        source: Tensor,
        dim: Dim,
    ): Tensor = scatter(indexes, source, dim.toIndex(shape(), "scatter"))

    /**
     * Scatters values from `source` in-place into `self` at the specified `indexes` along `dim`.
     */
    public fun scatterSet(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ) {
        if (sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-set")
        scatterChecks(indexes, source, dimIdx)
        storage.scatterSet(layout, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
    }

    /**
     * Scatters values from `source` in-place into `self` at the specified `indexes` along `dim`.
     */
    public fun scatterSet(
        indexes: Tensor,
        source: Tensor,
        dim: Dim,
    ) = scatterSet(indexes, source, dim.toIndex(shape(), "scatter-set"))

    /**
     * Scatters and adds values from `source` into a copy of `self` at `indexes` along `dim`.
     */
    public fun scatterAdd(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-add")
        scatterChecks(indexes, source, dimIdx)
        val sh = shape()
        val newStorage = storage.tryClone(layout)
        val l = Layout.contiguous(sh)
        newStorage.scatterAdd(l, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
        val newOp = BackpropOp.new3(this, indexes, source) { t1, t2, t3 -> Op.ScatterAdd(t1, t2, t3, dimIdx) }
        return fromStorage(newStorage, sh, newOp, false)
    }

    /**
     * Scatters and adds values from `source` into a copy of `self` at `indexes` along `dim`.
     */
    public fun scatterAdd(
        indexes: Tensor,
        source: Tensor,
        dim: Dim,
    ): Tensor = scatterAdd(indexes, source, dim.toIndex(shape(), "scatter-add"))

    /**
     * Scatters and adds values from `source` in-place into `self` at `indexes` along `dim`.
     */
    public fun scatterAddSet(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ) {
        if (sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-add-set")
        scatterChecks(indexes, source, dimIdx)
        storage.scatterAdd(layout, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
    }

    /**
     * Scatters and adds values from `source` in-place into `self` at `indexes` along `dim`.
     */
    public fun scatterAddSet(
        indexes: Tensor,
        source: Tensor,
        dim: Dim,
    ) = scatterAddSet(indexes, source, dim.toIndex(shape(), "scatter-add-set"))

    /**
     * Embeds the values of the `src` tensor into the `self` tensor on the first dimension.
     */
    public fun sliceScatter0(
        src: Tensor,
        start: Int,
    ): Tensor {
        if (dtype() != src.dtype()) {
            throw CandleException.DTypeMismatchBinaryOp(dtype(), src.dtype(), "slice-scatter")
        }
        if (device().location() != src.device().location()) {
            throw CandleException.DeviceMismatchBinaryOp(device().location(), src.device().location(), "slice-scatter")
        }
        if (rank() != src.rank()) {
            throw CandleException.UnexpectedNumberOfDims(rank(), src.rank(), src.shape())
        }
        val shapeOk =
            dims().indices.all { dimIdx ->
                if (dimIdx == 0) {
                    src.dims()[dimIdx] + start <= dims()[dimIdx]
                } else {
                    dims()[dimIdx] == src.dims()[dimIdx]
                }
            }
        if (!shapeOk) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), src.shape(), "slice-scatter (self, src)")
        }
        val newStorage = storage.tryClone(layout)
        val offset = start * src.dims().drop(1).fold(1) { acc, v -> acc * v }
        src.storage.copyStridedSrc(newStorage, offset, src.layout)
        val op = BackpropOp.new2(this, src) { t1, t2 -> Op.SliceScatter0(t1, t2, start) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Embeds the values of the `src` tensor into the `self` tensor on the specified dimension.
     */
    public fun sliceScatter(
        src: Tensor,
        dim: Int,
        start: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "slice-scatter")
        return if (dimIdx == 0) {
            sliceScatter0(src, start)
        } else {
            transpose(0, dimIdx)
                .sliceScatter0(src.transpose(0, dimIdx), start)
                .transpose(0, dimIdx)
        }
    }

    /**
     * Embeds the values of the `src` tensor into the `self` tensor on the specified dimension.
     */
    public fun sliceScatter(
        src: Tensor,
        dim: Dim,
        start: Int,
    ): Tensor = sliceScatter(src, dim.toIndex(shape(), "slice-scatter"), start)

    /**
     * Accumulate element from `source` at indexes `indexes` and add them to `self`.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val s = Tensor.new(floatArrayOf(1f, 2f, 3f), Shape(1, 3), Device.Cpu)
     * val ids = Tensor.new(intArrayOf(1), Shape(1), Device.Cpu)
     * val res = a.indexAdd(ids, s, 0)
     * // [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]
     * ```
     */
    public fun indexAdd(
        indexes: Tensor,
        source: Tensor,
        dim: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "index-add")
        val sourceDims = source.dims()
        val selfDims = dims()
        var mismatch = sourceDims.size != selfDims.size
        if (!mismatch) {
            for (i in selfDims.indices) {
                if (i != dimIdx && selfDims[i] != sourceDims[i]) {
                    mismatch = true
                    break
                }
            }
        }
        if (mismatch) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), source.shape(), "index-add (self, source)")
        }
        val indexesLen = indexes.dims1()
        if (sourceDims[dimIdx] != indexesLen) {
            throw CandleException.ShapeMismatchBinaryOp(indexes.shape(), source.shape(), "index-add (ids, source))")
        }
        val newStorage = storage.indexAdd(layout, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
        val op = BackpropOp.new3(this, indexes, source) { t1, t2, t3 -> Op.IndexAdd(t1, t2, t3, dimIdx) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Accumulate element from `source` at indexes `indexes` and add them to `self`.
     */
    public fun indexAdd(
        indexes: Tensor,
        source: Tensor,
        dim: Dim,
    ): Tensor = indexAdd(indexes, source, dim.toIndex(shape(), "index-add"))

    // --- Indexing ---

    /**
     * Gather values across the target dimension.
     *
     * # Arguments
     *
     * * `self` - The input tensor.
     * * `indexes` - The indices of elements to gather, this should have same number of dimensions as `self`
     *   and `indexes.dims()[d] <= self.dims()[d]` for all dimensions `d != dim`
     * * `dim` - the target dimension.
     *
     * The resulting tensor has the same shape as `indexes` and use values from `self` indexed on
     * dimension `dim` by the values in `indexes`.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(2, 2), Device.Cpu)
     * val b = Tensor.new(uintArrayOf(0u, 0u, 1u, 0u), Shape(2, 2), Device.Cpu)
     * val c = a.gather(b, 1)
     * // [[1.0, 1.0], [4.0, 3.0]]
     * ```
     */
    public fun gather(
        indexes: Tensor,
        dim: Int,
    ): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "gather")
        val selfDims = dims()
        val indexesDims = indexes.dims()
        var mismatch = indexesDims.size != selfDims.size
        if (!mismatch) {
            for (i in 0 until selfDims.size) {
                if (i != normalizedDim && selfDims[i] < indexesDims[i]) {
                    mismatch = true
                    break
                }
            }
        }
        if (mismatch) {
            throw CandleException.ShapeMismatch(shape().elemCount(), indexes.shape())
        }
        val newStorage = storage.gather(layout, indexes.storage, indexes.layout, normalizedDim)
        val op = BackpropOp.new2(this, indexes) { t1, t2 -> Op.Gather(t1, t2, normalizedDim) }
        return fromStorage(newStorage, indexes.shape(), op, false)
    }

    /**
     * Gather values across the target dimension.
     */
    public fun gather(
        indexes: Tensor,
        dim: Dim,
    ): Tensor {
        val normalizedDim = dim.toIndex(shape(), "gather")
        return gather(indexes, normalizedDim)
    }

    /**
     * Select values for the input tensor at the target indexes across the specified dimension.
     *
     * The `indexes` argument is an int tensor with a single dimension.
     * The output has the same number of dimension as the `self` input. The target dimension of
     * the output has length the length of `indexes` and the values are taken from `self` using
     * the index from `indexes`. Other dimensions have the same number of elements as the input
     * tensor.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(2, 2), Device.Cpu)
     * val b = Tensor.new(uintArrayOf(0u, 1u, 0u), Shape(3), Device.Cpu)
     * val c = a.indexSelect(b, 0)
     * // [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]]
     * ```
     */
    public fun indexSelect(
        indexes: Tensor,
        dim: Int,
    ): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "index_select")
        val newStorage = storage.indexSelect(indexes.storage, layout, indexes.layout, normalizedDim)
        val outDims = dims().toMutableList()
        outDims[normalizedDim] = indexes.elemCount()
        val op = BackpropOp.new2(this, indexes) { t, i -> Op.IndexSelect(t, i, normalizedDim) }
        return fromStorage(newStorage, Shape(outDims), op, false)
    }

    /**
     * Select values for the input tensor at the target indexes across the specified dimension.
     */
    public fun indexSelect(
        indexes: Tensor,
        dim: Dim,
    ): Tensor = indexSelect(indexes, dim.toIndex(shape(), "index-select"))

    /**
     * Returns an iterator over position of the elements in the storage when ranging over the
     * index tuples in lexicographic order.
     */
    public fun stridedIndex(): StridedIndex = layout.stridedIndex()

    /**
     * Similar to `stridedIndex` but returns the position of the start of each contiguous block
     * as well as the length of the contiguous blocks. For a contiguous tensor, the index iterator
     * will only return the start offset and the size would be the number of elements in the
     * tensor.
     */
    public fun stridedBlocks(): StridedBlocks = layout.stridedBlocks()

    // --- Data Accessors ---

    /**
     * Returns a reference to this tensor.
     */
    public fun asRef(): Tensor = this

    /**
     * Dereferences this tensor wrapper to its underlying tensor representation.
     */
    public fun deref(): Tensor = this

    public inline fun <reified S> convertCpuScalar(
        cpu: CpuStorage,
        index: Int,
    ): S {
        val v: Any =
            when (S::class) {
                Float::class -> cpu.getAsFloat(index)
                Double::class -> cpu.getAsDouble(index)
                Long::class -> cpu.getAsLong(index)
                Int::class -> cpu.getAsLong(index).toInt()
                Short::class -> cpu.getAsLong(index).toShort()
                Byte::class -> cpu.getAsLong(index).toByte()
                UInt::class -> cpu.getAsLong(index).toUInt()
                UShort::class -> cpu.getAsLong(index).toUShort()
                UByte::class -> cpu.getAsLong(index).toUByte()
                ULong::class -> cpu.getAsLong(index).toULong()
                F16::class -> F16.fromFloat(cpu.getAsFloat(index))
                BF16::class -> BF16.fromFloat(cpu.getAsFloat(index))
                else -> cpu.getAsDouble(index)
            }
        return v as S
    }

    /**
     * Returns the data contained in a 1D tensor as a list of scalar values.
     */
    public inline fun <reified S> toVec1(): List<S> {
        val count = elemCount()
        if (rank() != 1) throw CandleException.UnexpectedNumberOfDims(1, rank(), shape())
        val cont = contiguous()
        val cpu = (cont.storage() as Storage.Cpu).storage
        val list = ArrayList<S>(count)
        for (i in 0 until count) {
            list.add(convertCpuScalar<S>(cpu, i))
        }
        return list
    }

    /**
     * Returns the data contained in a 2D tensor as a list of lists of scalar values.
     */
    public inline fun <reified S> toVec2(): List<List<S>> {
        val (dim1, dim2) = dims2()
        val cont = contiguous()
        val cpu = (cont.storage() as Storage.Cpu).storage
        val result = ArrayList<List<S>>(dim1)
        for (i in 0 until dim1) {
            val row = ArrayList<S>(dim2)
            for (j in 0 until dim2) {
                val idx = i * dim2 + j
                row.add(convertCpuScalar<S>(cpu, idx))
            }
            result.add(row)
        }
        return result
    }

    /**
     * Returns the data contained in a 3D tensor as nested lists of scalar values.
     */
    public inline fun <reified S> toVec3(): List<List<List<S>>> {
        val (dim1, dim2, dim3) = dims3()
        val cont = contiguous()
        val cpu = (cont.storage() as Storage.Cpu).storage
        val result = ArrayList<List<List<S>>>(dim1)
        for (i in 0 until dim1) {
            val page = ArrayList<List<S>>(dim2)
            for (j in 0 until dim2) {
                val row = ArrayList<S>(dim3)
                for (k in 0 until dim3) {
                    val idx = (i * dim2 + j) * dim3 + k
                    row.add(convertCpuScalar<S>(cpu, idx))
                }
                page.add(row)
            }
            result.add(page)
        }
        return result
    }

    /**
     * Retrieves the single scalar value held in the tensor. If the tensor contains multiple
     * dimensions, an error is returned instead.
     */
    public inline fun <reified S> toScalar(): S {
        if (rank() != 0) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        val cont = contiguous()
        val cpu = (cont.storage() as Storage.Cpu).storage
        return convertCpuScalar<S>(cpu, 0)
    }

    /**
     * An alias for `toScalar`.
     */
    public inline fun <reified S> toVec0(): S = toScalar<S>()

    /**
     * Returns the data contained in a 1D tensor as a float array.
     */
    public fun toVec1F32(): FloatArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = FloatArray(count)
        for (i in 0 until count) res[i] = cpu.getAsFloat(i)
        return res
    }

    /**
     * Returns the data contained in a 1D tensor as a double array.
     */
    public fun toVec1F64(): DoubleArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = DoubleArray(count)
        for (i in 0 until count) res[i] = cpu.getAsDouble(i)
        return res
    }

    /**
     * Retrieves the single 32-bit float scalar value held in the tensor.
     */
    public fun toScalarF32(): Float {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1F32()[0]
    }

    /**
     * Retrieves the single 64-bit float scalar value held in the tensor.
     */
    public fun toScalarF64(): Double {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1F64()[0]
    }

    /**
     * Returns the data contained in a 1D tensor as a 64-bit integer array.
     */
    public fun toVec1I64(): LongArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = LongArray(count)
        for (i in 0 until count) res[i] = cpu.getAsLong(i)
        return res
    }

    /**
     * Retrieves the single 64-bit integer scalar value held in the tensor.
     */
    public fun toScalarI64(): Long {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1I64()[0]
    }

    // --- Shape manipulations ---

    /**
     * Reshapes this tensor to the target shape.
     *
     * If the tensor is contiguous, this requires no data copy. Otherwise the tensor is first
     * made contiguous before reshaping. One dimension may be specified as `()` / `-1` via
     * `intoShapeWithHole` to infer its size from the total element count.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val c = a.reshape(Shape(1, 6))
     * assertEquals(listOf(1, 6), c.shape().dims)
     * ```
     */
    public fun reshape(shape: Shape): Tensor {
        if (shape.elemCount() != elemCount()) {
            throw CandleException.ShapeMismatch(elemCount(), shape)
        }
        if (isContiguous()) {
            val newLayout = Layout.contiguousWithOffset(shape, layout.startOffset)
            val op = BackpropOp.new1(this) { Op.Reshape(it) }
            return Tensor(TensorId.next(), storage, newLayout, op, false)
        } else {
            return contiguous().reshape(shape)
        }
    }

    /**
     * Reshapes this tensor to the specified dimensions, inferring at most one `-1` dimension.
     */
    public fun reshape(vararg dims: Int): Tensor = reshape(dims.toList().intoShapeWithHole(elemCount()))

    internal fun flatten_(
        startDim: Int?,
        endDim: Int?,
    ): Tensor {
        if (rank() == 0) {
            return reshape(1)
        }
        val start = startDim?.let { Shape.toIndex(it, shape(), "flatten") } ?: 0
        val end = endDim?.let { Shape.toIndex(it, shape(), "flatten") } ?: (rank() - 1)
        return if (start < end) {
            val d = dims()
            val dstDims = ArrayList<Int>()
            dstDims.addAll(d.subList(0, start))
            dstDims.add(d.subList(start, end + 1).fold(1) { acc, v -> acc * v })
            if (end + 1 < d.size) {
                dstDims.addAll(d.subList(end + 1, d.size))
            }
            reshape(Shape(dstDims))
        } else {
            this
        }
    }

    /**
     * Flattens the input tensor by reshaping it into a one dimension tensor.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val b = a.flattenAll()
     * // [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
     * ```
     */
    public fun flattenAll(): Tensor = flatten_(null, null)

    /**
     * Flattens the input tensor on dimension indexes from `startDim` (inclusive) to the last
     * dimension.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3, 4), DType.F32, Device.Cpu)
     * val c = a.flattenFrom(1)
     * assertEquals(listOf(2, 12), c.shape().dims)
     * ```
     */
    public fun flattenFrom(dim: Int): Tensor = flatten_(dim, null)

    /**
     * Flattens the input tensor on dimension indexes from `dim` (inclusive) to the last dimension.
     */
    public fun flattenFrom(dim: Dim): Tensor = flattenFrom(dim.toIndex(shape(), "flatten_from"))

    /**
     * Flattens the input tensor on dimension indexes from `0` to `endDim` (inclusive).
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3, 4), DType.F32, Device.Cpu)
     * val c = a.flattenTo(1)
     * assertEquals(listOf(6, 4), c.shape().dims)
     * ```
     */
    public fun flattenTo(dim: Int): Tensor = flatten_(null, dim)

    /**
     * Flattens the input tensor on dimension indexes from `0` to `dim` (inclusive).
     */
    public fun flattenTo(dim: Dim): Tensor = flattenTo(dim.toIndex(shape(), "flatten_to"))

    /**
     * Flattens the input tensor on dimension indexes from `startDim` to `endDim` (both
     * inclusive).
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3, 4, 5), DType.F32, Device.Cpu)
     * val c = a.flatten(1, 2)
     * assertEquals(listOf(2, 12, 5), c.shape().dims)
     * ```
     */
    public fun flatten(
        startDim: Int,
        endDim: Int,
    ): Tensor = flatten_(startDim, endDim)

    /**
     * Flattens the input tensor on dimension indexes from `startDim` to `endDim` (both inclusive).
     */
    public fun flatten(
        startDim: Dim,
        endDim: Dim,
    ): Tensor = flatten(startDim.toIndex(shape(), "flatten"), endDim.toIndex(shape(), "flatten"))

    /**
     * Creates a new tensor with the specified dimension removed if its size was one.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 1, 3), DType.F32, Device.Cpu)
     * val c = a.squeeze(1)
     * assertEquals(listOf(2, 3), c.shape().dims)
     * ```
     */
    public fun squeeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "squeeze")
        if (dims()[normalizedDim] != 1) return this
        val newDims = dims().toMutableList()
        newDims.removeAt(normalizedDim)
        val newStride = stride().toMutableList()
        newStride.removeAt(normalizedDim)
        val op = BackpropOp.new1(this) { Op.Reshape(it) }
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), op, false)
    }

    /**
     * Creates a new tensor with the specified dimension removed if its size was one.
     */
    public fun squeeze(dim: Dim): Tensor = squeeze(dim.toIndex(shape(), "squeeze"))

    /**
     * Creates a new tensor with a dimension of size one inserted at the specified position.
     *
     * ```kotlin
     * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
     * val c = a.unsqueeze(0)
     * assertEquals(listOf(1, 2, 3), c.shape().dims)
     * ```
     */
    public fun unsqueeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndexPlusOne(dim, shape(), "unsqueeze")
        val newDims = dims().toMutableList()
        newDims.add(normalizedDim, 1)
        val newStride = stride().toMutableList()
        val s = if (normalizedDim < stride().size) stride()[normalizedDim] else 1
        newStride.add(normalizedDim, s)
        val op = BackpropOp.new1(this) { Op.Reshape(it) }
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), op, false)
    }

    /**
     * Creates a new tensor with a dimension of size one inserted at the specified position.
     */
    public fun unsqueeze(dim: Dim): Tensor = unsqueeze(dim.toIndexPlusOne(shape(), "unsqueeze"))

    /**
     * Returns a tensor that is a transposed version of the input, the given dimensions are
     * swapped.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val transposedTensor = tensor.transpose(0, 1)
     * // [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]
     * ```
     */
    public fun transpose(
        dim1: Int,
        dim2: Int,
    ): Tensor {
        if (dim1 == dim2) return this
        val newLayout = layout.transpose(dim1, dim2)
        val op = BackpropOp.new1(this) { Op.Transpose(it, dim1, dim2) }
        return Tensor(TensorId.next(), storage, newLayout, op, false)
    }

    /**
     * Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
     */
    public fun transpose(
        dim1: Dim,
        dim2: Dim,
    ): Tensor = transpose(dim1.toIndex(shape(), "transpose"), dim2.toIndex(shape(), "transpose"))

    /**
     * Returns a tensor that is a transposed version of the input, the two last dimensions of the
     * input are swapped.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val transposedTensor = tensor.t()
     * // [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]
     * ```
     */
    public fun t(): Tensor {
        if (rank() < 2) {
            throw CandleException.UnexpectedNumberOfDims(2, rank(), shape())
        }
        return transpose(rank() - 2, rank() - 1)
    }

    /**
     * Returns a tensor with the same data as the input where the dimensions have been permuted.
     * dims must be a permutation of 0, 1, ..., n-1 where n is the number of dimensions of the
     * input.
     *
     * ```kotlin
     * val tensor = Tensor.arange(0f, 24f, Device.Cpu).reshape(Shape(2, 3, 4))
     * val permuted = tensor.permute(listOf(2, 0, 1))
     * assertEquals(listOf(4, 2, 3), permuted.dims())
     * ```
     */
    public fun permute(idxs: List<Int>): Tensor {
        val newLayout = layout.permute(idxs)
        val op = BackpropOp.new1(this) { Op.Permute(it, idxs) }
        return Tensor(TensorId.next(), storage, newLayout, op, false)
    }

    /**
     * Returns a tensor with the same data where dimensions have been permuted.
     */
    public fun permute(vararg idxs: Int): Tensor = permute(idxs.toList())

    /**
     * Broadcast the input tensor to the target shape. This returns an error if the input shape is
     * not compatible with the target shape.
     *
     * If the input shape is `i1, i2, ... ik`, the target shape has to have `k` dimensions or
     * more and shape `j1, ..., jl, t1, t2, ..., tk`. The dimensions `j1` to `jl` can have
     * any value, the dimension `ta` must be equal to `ia` if `ia` is different from 1. If
     * `ia` is equal to 1, any value can be used.
     */
    public fun broadcastAs(targetShape: Shape): Tensor {
        val newLayout = layout.broadcastAs(targetShape)
        val op = BackpropOp.new1(this) { Op.Broadcast(it) }
        return Tensor(TensorId.next(), storage, newLayout, op, false)
    }

    /**
     * An alias for broadcastAs.
     */
    public fun expand(shape: Shape): Tensor = broadcastAs(shape)

    /**
     * Broadcasts the input tensor to the target shape on the left dimensions.
     *
     * For example if the input tensor has dimensions `(c, d)` and `leftShape` is `(a, b)`, the
     * returned tensor has dimensions `(a, b, c, d)`.
     *
     * ```kotlin
     * val a = Tensor.ones(Shape(2, 3), DType.F32, Device.Cpu)
     * val b = a.broadcastLeft(Shape(4, 1))
     * assertEquals(listOf(4, 1, 2, 3), b.dims())
     * ```
     */
    public fun broadcastLeft(leftShape: Shape): Tensor {
        val newDims = leftShape.dims + dims()
        return broadcastAs(Shape(newDims))
    }

    /**
     * Returns a new tensor that is a narrowed version of the input, the dimension `dim`
     * ranges from `start` to `start + len`.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), Shape(3, 3), Device.Cpu)
     * val b = a.narrow(0, 1, 2)
     * // [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
     * val c = a.narrow(1, 1, 1)
     * // [[1.0], [4.0], [7.0]]
     * ```
     */
    public fun narrow(
        dim: Int,
        start: Int,
        len: Int,
    ): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "narrow")
        val newLayout = layout.narrow(normalizedDim, start, len)
        val op = BackpropOp.new1(this) { Op.Narrow(it, normalizedDim, start, len) }
        return Tensor(TensorId.next(), storage, newLayout, op, false)
    }

    /**
     * Returns a tensor that is in row major order. This is the same as the original tensor if it
     * was already contiguous, otherwise a copy is triggered.
     */
    public fun contiguous(): Tensor {
        if (isContiguous() && layout.startOffset == 0) return this
        val newStorage = storage.tryClone(layout)
        val op = BackpropOp.new1(this) { Op.Copy(it) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Returns a tensor that is in row major order. This always makes a copy of the underlying
     * tensor.
     */
    public fun forceContiguous(): Tensor = copy()

    /**
     * Copies the actual storage of the tensor; compared to `clone()` this triggers a deep copy of
     * the underlying storage buffer.
     */
    public fun copy(): Tensor {
        val newStorage = storage.tryClone(layout)
        val op = BackpropOp.new1(this) { Op.Copy(it) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Create a variable based on the values currently stored in this tensor. The storage is always
     * copied.
     */
    public fun makeVar(): Tensor {
        val newStorage = storage.tryClone(layout)
        return fromStorage(newStorage, shape(), BackpropOp.None, true)
    }

    /**
     * Casts the input tensor to the target `dtype`.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val cast = tensor.toDtype(DType.F64)
     * assertEquals(DType.F64, cast.dtype())
     * ```
     */
    public fun toDtype(targetDType: DType): Tensor {
        if (dtype() == targetDType) return this
        val newStorage = storage.toDtype(layout, targetDType)
        val op = BackpropOp.new1(this) { Op.ToDType(it) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Returns a tensor that is on the target device. This creates a copy on the target device if
     * the original tensor is not already on the target device.
     */
    public fun toDevice(targetDevice: Device): Tensor {
        if (device().sameDevice(targetDevice)) return this
        val currentStorage = storage
        val newStorage: Storage =
            when (currentStorage) {
                is Storage.Cpu -> {
                    when (targetDevice) {
                        is Device.Cpu -> Storage.Cpu(currentStorage.storage)
                        is Device.Cuda -> Storage.Cuda(targetDevice.device.storageFromCpuStorage(currentStorage.storage))
                        is Device.Metal -> Storage.Metal(targetDevice.device.storageFromCpuStorage(currentStorage.storage))
                    }
                }

                is Storage.Cuda -> {
                    when (targetDevice) {
                        is Device.Cpu -> {
                            Storage.Cpu(currentStorage.storage.toCpuStorage())
                        }

                        is Device.Cuda -> {
                            Storage.Cuda(currentStorage.storage.transferToDevice(targetDevice.device))
                        }

                        is Device.Metal -> {
                            val cpuStorage = currentStorage.storage.toCpuStorage()
                            Storage.Metal(targetDevice.device.storageFromCpuStorage(cpuStorage))
                        }
                    }
                }

                is Storage.Metal -> {
                    when (targetDevice) {
                        is Device.Cpu -> {
                            Storage.Cpu(currentStorage.storage.toCpuStorage())
                        }

                        is Device.Cuda -> {
                            val cpuStorage = currentStorage.storage.toCpuStorage()
                            Storage.Cuda(targetDevice.device.storageFromCpuStorage(cpuStorage))
                        }

                        is Device.Metal -> {
                            val cpuStorage = currentStorage.storage.toCpuStorage()
                            Storage.Metal(targetDevice.device.storageFromCpuStorage(cpuStorage))
                        }
                    }
                }
            }
        val op = BackpropOp.new1(this) { Op.ToDevice(it) }
        return Tensor(TensorId.next(), newStorage, layout.clone(), op, false)
    }

    /**
     * Returns the sub-tensor fixing the index at `i` on the first dimension.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val t0 = tensor.get(0)
     * // [0.0, 1.0]
     * val t1 = tensor.get(1)
     * // [2.0, 3.0]
     * ```
     */
    public fun get(i: Int): Tensor {
        val dims = dims()
        return if (dims.isEmpty()) {
            this
        } else {
            narrow(0, i, 1).reshape(Shape(dims.subList(1, dims.size)))
        }
    }

    /**
     * Returns the sub-tensor fixing the index at `index` on the dimension `dim`.
     *
     * ```kotlin
     * val tensor = Tensor.new(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), Shape(3, 2), Device.Cpu)
     * val c1 = tensor.getOnDim(1, 1)
     * // [1.0, 3.0, 5.0]
     * val r0 = tensor.getOnDim(0, 0)
     * // [0.0, 1.0]
     * ```
     */
    public fun getOnDim(
        dim: Int,
        index: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "get_on_dim")
        val dims = dims()
        val targetDims = dims.subList(0, dimIdx) + dims.subList(dimIdx + 1, dims.size)
        return narrow(dimIdx, index, 1).reshape(Shape(targetDims))
    }

    /**
     * Returns the sub-tensor fixing the index at `index` on the dimension `dim`.
     */
    public fun getOnDim(
        dim: Dim,
        index: Int,
    ): Tensor = getOnDim(dim.toIndex(shape(), "get_on_dim"), index)

    /**
     * Pad the input tensor using 0s along dimension `dim`. This adds `left` elements before the
     * input tensor values and `right` elements after.
     */
    public fun padWithZeros(
        dim: Int,
        left: Int,
        right: Int,
    ): Tensor {
        if (left == 0 && right == 0) return this
        val dimIdx = Shape.toIndex(dim, shape(), "pad_with_zeros")
        return if (left == 0) {
            val d = dims().toMutableList()
            d[dimIdx] = right
            val r = zeros(Shape(d), dtype(), device())
            cat(listOf(this, r), dimIdx)
        } else if (right == 0) {
            val d = dims().toMutableList()
            d[dimIdx] = left
            val l = zeros(Shape(d), dtype(), device())
            cat(listOf(l, this), dimIdx)
        } else {
            val dL = dims().toMutableList()
            dL[dimIdx] = left
            val l = zeros(Shape(dL), dtype(), device())
            val dR = dims().toMutableList()
            dR[dimIdx] = right
            val r = zeros(Shape(dR), dtype(), device())
            cat(listOf(l, this, r), dimIdx)
        }
    }

    /**
     * Pad the input tensor using 0s along dimension `dim`.
     */
    public fun padWithZeros(
        dim: Dim,
        left: Int,
        right: Int,
    ): Tensor = padWithZeros(dim.toIndex(shape(), "pad_with_zeros"), left, right)

    /**
     * Pad the input tensor using same values along dimension `dim`. This adds `left` elements
     * before the input tensor values and `right` elements after.
     */
    public fun padWithSame(
        dim: Int,
        left: Int,
        right: Int,
    ): Tensor {
        if (left == 0 && right == 0) return this
        if (elemCount() == 0) bail("cannot use pad_with_same on an empty tensor")
        val dimIdx = Shape.toIndex(dim, shape(), "pad_with_same")
        val dimLen = dims()[dimIdx]
        return if (left == 0) {
            val r = narrow(dimIdx, dimLen - 1, 1)
            val list = ArrayList<Tensor>(1 + right)
            list.add(this)
            repeat(right) { list.add(r) }
            cat(list, dimIdx)
        } else if (right == 0) {
            val l = narrow(dimIdx, 0, 1)
            val list = ArrayList<Tensor>(left + 1)
            repeat(left) { list.add(l) }
            list.add(this)
            cat(list, dimIdx)
        } else {
            val l = narrow(dimIdx, 0, 1)
            val r = narrow(dimIdx, dimLen - 1, 1)
            val list = ArrayList<Tensor>(left + 1 + right)
            repeat(left) { list.add(l) }
            list.add(this)
            repeat(right) { list.add(r) }
            cat(list, dimIdx)
        }
    }

    /**
     * Pad the input tensor using same values along dimension `dim`.
     */
    public fun padWithSame(
        dim: Dim,
        left: Int,
        right: Int,
    ): Tensor = padWithSame(dim.toIndex(shape(), "pad_with_same"), left, right)

    /**
     * Normalize a relative axis value: negative values count from the back.
     */
    public fun normalizeAxis(axis: Long): Int {
        val r = rank().toLong()
        return if (r <= axis) {
            bail("axis $axis is too large, tensor rank $r")
        } else if (axis >= 0) {
            axis.toInt()
        } else {
            val naxis = r + axis
            if (naxis < 0) {
                bail("axis $axis is too small, tensor rank $r")
            }
            naxis.toInt()
        }
    }

    /**
     * Returns the cumulative sum of elements of the input tensor along the dimension `dim`.
     *
     * ```kotlin
     * val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), Shape(2, 3), Device.Cpu)
     * val b = a.cumsum(1)
     * // [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]]
     * ```
     */
    public fun cumsum(dim: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "cumsum")
        val r = rank()
        if (r == 0) return this
        val nAxis = dims()[dimIdx]
        val triu = triu2(nAxis, dtype(), device())
        return if (r == 1) {
            unsqueeze(0).matmul(triu).squeeze(0)
        } else {
            val last = r - 1
            val t = transpose(dimIdx, last)
            val res = t.broadcastMatmul(triu)
            res.transpose(dimIdx, last)
        }
    }

    /**
     * Returns the cumulative sum of elements along the specified dimension.
     */
    public fun cumsum(dim: Dim): Tensor = cumsum(dim.toIndex(shape(), "cumsum"))

    /**
     * Returns a copy of `self` where the values within `ranges` have been replaced with the
     * content of `src`.
     *
     * ```kotlin
     * val a = Tensor.arange(0f, 12f, Device.Cpu).reshape(Shape(4, 3))
     * val b = Tensor.new(floatArrayOf(-1f, -2f, -3f, -4f), Shape(2, 2), Device.Cpu)
     * val c = a.sliceAssign(listOf(1..2, 0..1), b)
     * ```
     */
    public fun sliceAssign(
        ranges: List<IntRange>,
        src: Tensor,
    ): Tensor {
        val srcDims = src.dims()
        val selfDims = dims()
        if (selfDims.size != srcDims.size) {
            bail("slice-assign requires input with the same rank ${selfDims.size} <> ${srcDims.size}")
        }
        if (selfDims.size != ranges.size) {
            bail("slice-assign requires input with the same rank as there are ranges ${selfDims.size} <> ${ranges.size}")
        }
        var curSrc = src
        var mask = ones(src.shape(), DType.U8, src.device())
        for (i in ranges.indices) {
            val range = ranges[i]
            val startIncluded = range.first
            val endExcluded = range.last + 1
            if (endExcluded <= startIncluded) {
                bail("slice-assign: empty range for dim $i, $startIncluded $endExcluded")
            }
            if (selfDims[i] < endExcluded) {
                bail("slice-assign: upper bound is out of range for dim $i, $endExcluded ${selfDims[i]}")
            }
            if (endExcluded - startIncluded != srcDims[i]) {
                bail("slice-assign: the range for dim $i ($startIncluded..$endExcluded) does not match the size of src ${srcDims[i]}")
            }
            curSrc = curSrc.padWithZeros(i, startIncluded, selfDims[i] - endExcluded)
            mask = mask.padWithZeros(i, startIncluded, selfDims[i] - endExcluded)
        }
        return mask.whereCond(curSrc, this)
    }

    /**
     * Returns the log-sum-exp of elements of the input tensor along the specified dimensions.
     *
     * Computes `log(sum(exp(tensor), dim))` in a numerically stable way by subtracting the
     * maximum along each reduced dimension before exponentiating.
     */
    public fun logSumExp(sumDims: List<Int>): Tensor {
        val normalizedDims = sumDims.map { Shape.toIndex(it, shape(), "log-sum-exp") }
        if (normalizedDims.isEmpty()) return this
        var maxVal = maxKeepdim(normalizedDims[0])
        for (i in 1 until normalizedDims.size) {
            maxVal = maxVal.maxKeepdim(normalizedDims[i])
        }
        val expVal = (this - maxVal).exp()
        val sumVal = expVal.sum(normalizedDims)
        return sumVal.log() + maxVal.squeezeDims(normalizedDims)
    }

    /**
     * Pointwise pow operation.
     */
    public fun pow(rhs: Tensor): Tensor = (rhs * log()).exp()

    /**
     * Broadcasting version of `pow`.
     */
    public fun broadcastPow(rhs: Tensor): Tensor = (rhs.broadcastMul(log())).exp()

    /**
     * Returns a new tensor with the order of elements reversed along the specified dimensions.
     * This function makes a copy of the tensor's data.
     */
    public fun flip(dims: List<Int>): Tensor {
        var result = this
        for (dim in dims) {
            val size = result.dims()[dim]
            val indices = LongArray(size) { (size - 1 - it).toLong() }
            val indicesTensor = new(indices, Shape(listOf(size)), result.device())
            result = result.indexSelect(indicesTensor, dim)
        }
        return result
    }

    /**
     * Returns all slices of size `size` from `self` in dimension `dim` stepped by `step`.
     */
    public fun unfold(
        dim: Int,
        size: Int,
        step: Int,
    ): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "unfold")
        val sizes = dims().toMutableList()
        val strides = stride().toMutableList()
        val maxLen = if (dims().isEmpty()) 1 else sizes[dimIdx]
        if (size > maxLen) {
            bail("unfold: maximum size for tensor at dimension $dimIdx is $maxLen but size is $size")
        }
        sizes.add(size)
        strides.add(if (dims().isEmpty()) 1 else strides[dimIdx])
        if (dims().isNotEmpty()) {
            sizes[dimIdx] = ((sizes[dimIdx].toFloat() - size.toFloat()) / step.toFloat() + 1.0f).toInt()
            strides[dimIdx] *= step
        }
        val newLayout = Layout(Shape(sizes), strides, layout.startOffset)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.new1(this) { Op.Reshape(it) }, false)
    }

    /**
     * Returns all slices of size `size` from `self` in dimension `dim` stepped by `step`.
     */
    public fun unfold(
        dim: Dim,
        size: Int,
        step: Int,
    ): Tensor = unfold(dim.toIndex(shape(), "unfold"), size, step)

    override fun toString(): String = "Tensor[${shape()}, ${dtype()}, ${device()}]"

    public companion object {
        /**
         * Creates a fresh tensor structure based on a storage and a shape.
         *
         * # Note
         * - This uses contiguous strides
         * - Ensure the shape is compatible with the shape of the storage.
         */
        public fun fromStorage(
            storage: Storage,
            shape: Shape,
            op: BackpropOp,
            isVariable: Boolean,
        ): Tensor =
            Tensor(
                TensorId.next(),
                storage,
                Layout.contiguous(shape),
                op,
                isVariable,
            )

        internal fun onesImpl(
            shape: Shape,
            dtype: DType,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val t = zerosImpl(shape, dtype, device, isVariable)
            t.storage.constSet(Scalar.one(dtype), t.layout)
            return t
        }

        /**
         * Creates a new tensor filled with ones.
         *
         * ```kotlin
         * val a = Tensor.ones(Shape(2, 3), DType.F32, Device.Cpu)
         * val b = Tensor.fromSlice(floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f), Shape(2, 3), Device.Cpu)
         * // a == b
         * ```
         */
        public fun ones(
            shape: Shape,
            dtype: DType,
            device: Device,
        ): Tensor = onesImpl(shape, dtype, device, false)

        internal fun zerosImpl(
            shape: Shape,
            dtype: DType,
            device: Device,
            isVariable: Boolean,
        ): Tensor = fromStorage(device.zeros(shape, dtype), shape, BackpropOp.None, isVariable)

        /**
         * Creates a new tensor filled with zeros.
         *
         * ```kotlin
         * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
         * val b = Tensor.fromSlice(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f), Shape(2, 3), Device.Cpu)
         * // a == b
         * ```
         */
        public fun zeros(
            shape: Shape,
            dtype: DType,
            device: Device,
        ): Tensor = zerosImpl(shape, dtype, device, false)

        internal fun emptyImpl(
            shape: Shape,
            dtype: DType,
            device: Device,
            isVariable: Boolean,
        ): Tensor = zerosImpl(shape, dtype, device, isVariable)

        /**
         * Creates a new tensor filled with uninitialized memory.
         *
         * ```kotlin
         * val a = Tensor.empty(Shape(2, 3), DType.F32, Device.Cpu)
         * ```
         */
        public fun empty(
            shape: Shape,
            dtype: DType,
            device: Device,
        ): Tensor = emptyImpl(shape, dtype, device, false)

        internal fun randImpl(
            lo: Float,
            up: Float,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.randUniform(lo.toDouble(), up.toDouble(), shape, DType.F32)
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        internal fun randF64Impl(
            lo: Double,
            up: Double,
            shape: Shape,
            dtype: DType,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.randUniformF64(lo, up, shape, dtype)
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        /**
         * Creates a new tensor initialized with values sampled uniformly between `min` and `max`.
         */
        public fun rand(
            min: Float,
            max: Float,
            shape: Shape,
            device: Device,
        ): Tensor = randImpl(min, max, shape, device, false)

        internal fun randnImpl(
            mean: Float,
            std: Float,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.randNormal(mean.toDouble(), std.toDouble(), shape, DType.F32)
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        internal fun randnF64Impl(
            mean: Double,
            std: Double,
            shape: Shape,
            dtype: DType,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.randNormalF64(mean, std, shape, dtype)
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        /**
         * Creates a new tensor initialized with values sampled from a normal distribution with the
         * specified `mean` and standard deviation `std`.
         */
        public fun randn(
            mean: Float,
            std: Float,
            shape: Shape,
            device: Device,
        ): Tensor = randnImpl(mean, std, shape, device, false)

        internal fun newImpl(
            data: FloatArray,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.F32(data.copyOf())), shape, BackpropOp.None, isVariable)
        }

        internal fun newImpl(
            array: NdArray,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.storage(array)
            return fromStorage(storage, shape, BackpropOp.none(), isVariable)
        }

        internal fun <D> newImpl(
            array: List<D>,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.storageFromSlice(array)
            return fromStorage(storage, shape, BackpropOp.none(), isVariable)
        }

        /**
         * Creates a zero-filled tensor from shape with default F32 dtype and CPU device.
         */
        public fun from(shape: Shape): Tensor = zeros(shape, DType.F32, Device.Cpu)

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         */
        public fun new(
            data: FloatArray,
            shape: Shape,
            device: Device,
        ): Tensor = newImpl(data, shape, device, false)

        /**
         * Creates a new tensor on the specified device using 64-bit float values and shape.
         */
        public fun new(
            data: DoubleArray,
            shape: Shape,
            device: Device,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.F64(data.copyOf())), shape, BackpropOp.None, false)
        }

        /**
         * Creates a new tensor on the specified device using 32-bit integer values and shape.
         */
        public fun new(
            data: IntArray,
            shape: Shape,
            device: Device,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.I32(data.copyOf())), shape, BackpropOp.None, false)
        }

        /**
         * Creates a new tensor on the specified device using 64-bit integer values and shape.
         */
        public fun new(
            data: LongArray,
            shape: Shape,
            device: Device,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.I64(data.copyOf())), shape, BackpropOp.None, false)
        }

        /**
         * Creates a new tensor on the specified device using unsigned 32-bit integer values and shape.
         */
        public fun new(
            data: UIntArray,
            shape: Shape,
            device: Device,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.U32(data.copyOf())), shape, BackpropOp.None, false)
        }

        /**
         * Creates a new tensor on the specified device using unsigned 8-bit integer values and shape.
         */
        public fun new(
            data: UByteArray,
            shape: Shape,
            device: Device,
        ): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(device.storageFromCpuStorageOwned(CpuStorage.U8(data.copyOf())), shape, BackpropOp.None, false)
        }

        public fun new(data: Double, device: Device = Device.Cpu): Tensor =
            new(doubleArrayOf(data), Shape.SCALAR, device)

        public fun new(data: Float, device: Device = Device.Cpu): Tensor =
            new(floatArrayOf(data), Shape.SCALAR, device)

        public fun new(data: Long, device: Device = Device.Cpu): Tensor =
            new(longArrayOf(data), Shape.SCALAR, device)

        public fun new(data: Int, device: Device = Device.Cpu): Tensor =
            new(intArrayOf(data), Shape.SCALAR, device)

        public fun new(data: UInt, device: Device = Device.Cpu): Tensor =
            new(uintArrayOf(data), Shape.SCALAR, device)

        public fun new(data: UByte, device: Device = Device.Cpu): Tensor =
            new(ubyteArrayOf(data), Shape.SCALAR, device)

        public fun new(data: DoubleArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        public fun new(data: FloatArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        public fun new(data: LongArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        public fun new(data: IntArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        public fun new(data: UIntArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        public fun new(data: UByteArray, device: Device = Device.Cpu): Tensor =
            new(data, Shape(data.size), device)

        /**
         * Returns a new tensor with all the elements having the same specified value.
         *
         * ```kotlin
         * val a = Tensor.full(3.5, Shape(2, 4), Device.Cpu)
         * // [[3.5, 3.5, 3.5, 3.5], [3.5, 3.5, 3.5, 3.5]]
         * ```
         */
        public fun full(
            value: Double,
            shape: Shape,
            device: Device,
        ): Tensor {
            val t = zeros(shape, DType.F64, device)
            t.storage.constSet(Scalar.F64(value), t.layout)
            return t
        }

        /**
         * Creates a new 1D tensor from an iterator.
         *
         * ```kotlin
         * val a = Tensor.fromIter(listOf(1.0f, 2.0f, 3.0f, 4.0f), Device.Cpu)
         * // [1.0, 2.0, 3.0, 4.0]
         * ```
         */
        public fun fromIter(
            iter: Iterable<Float>,
            device: Device,
        ): Tensor {
            val list = iter.toList()
            return new(list.toFloatArray(), Shape(listOf(list.size)), device)
        }

        /**
         * Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
         * difference `1` from `start`.
         *
         * ```kotlin
         * val a = Tensor.arange(2.0f, 5.0f, Device.Cpu)
         * // [2.0, 3.0, 4.0]
         * ```
         */
        public fun arange(
            start: Float,
            end: Float,
            device: Device,
        ): Tensor = arangeStep(start, end, 1.0f, device)

        /**
         * Creates a new 1D integer tensor with values from the interval `[start, end)` taken with
         * step `1`.
         */
        public fun arange(
            start: Int,
            end: Int,
            device: Device,
        ): Tensor {
            val count = maxOf(0, end - start)
            val data = IntArray(count)
            for (i in 0 until count) {
                data[i] = start + i
            }
            return new(data, Shape(listOf(count)), device)
        }

        /**
         * Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
         * difference `step` from `start`.
         *
         * ```kotlin
         * val a = Tensor.arangeStep(2.0f, 4.0f, 0.5f, Device.Cpu)
         * // [2.0, 2.5, 3.0, 3.5]
         * ```
         */
        public fun arangeStep(
            start: Float,
            end: Float,
            step: Float,
            device: Device,
        ): Tensor {
            if (step == 0.0f) bail("step cannot be zero")
            val data = ArrayList<Float>()
            var current = start
            if (step > 0.0f) {
                while (current < end) {
                    data.add(current)
                    current += step
                }
            } else {
                while (current > end) {
                    data.add(current)
                    current += step
                }
            }
            return new(data.toFloatArray(), Shape(listOf(data.size)), device)
        }

        internal fun fromVecImpl(
            data: FloatArray,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor = newImpl(data, shape, device, isVariable)

        internal fun <D> fromVecImpl(
            data: List<D>,
            shape: Shape,
            device: Device,
            isVariable: Boolean,
        ): Tensor {
            val storage = device.storageOwned(data)
            return fromStorage(storage, shape, BackpropOp.none(), isVariable)
        }

        /**
         * Creates a new tensor initialized with values from the input slice. The number of elements
         * in this slice must be the same as the number of elements defined by the shape.
         *
         * ```kotlin
         * val values = floatArrayOf(2f, 3f, 4f, 5f, 6f, 7f)
         * val a = Tensor.fromSlice(values, Shape(2, 3), Device.Cpu)
         * // [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
         * ```
         */
        public fun fromSlice(
            data: FloatArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new 64-bit float tensor initialized with values from the input slice.
         */
        public fun fromSlice(
            data: DoubleArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new 32-bit integer tensor initialized with values from the input slice.
         */
        public fun fromSlice(
            data: IntArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new 64-bit integer tensor initialized with values from the input slice.
         */
        public fun fromSlice(
            data: LongArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new unsigned 32-bit integer tensor initialized with values from the input slice.
         */
        public fun fromSlice(
            data: UIntArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new unsigned 8-bit integer tensor initialized with values from the input slice.
         */
        public fun fromSlice(
            data: UByteArray,
            shape: Shape,
            device: Device = Device.Cpu,
        ): Tensor = new(data, shape, device)

        /**
         * Creates a new tensor initialized with values from the input list. The number of elements
         * in this list must be the same as the number of elements defined by the shape.
         *
         * ```kotlin
         * val a = Tensor.fromVec(listOf(1f, 2f, 3f, 4f, 5f, 6f), Shape(2, 3), Device.Cpu)
         * // [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
         * ```
         */
        public fun fromVec(
            data: List<Float>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toFloatArray(), shape, device)

        public fun fromVec(
            data: List<Double>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toDoubleArray(), shape, device)

        public fun fromVec(
            data: List<Long>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toLongArray(), shape, device)

        public fun fromVec(
            data: List<Int>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toIntArray(), shape, device)

        public fun fromVec(
            data: List<UInt>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toUIntArray(), shape, device)

        public fun fromVec(
            data: List<UByte>,
            shape: Shape,
            device: Device,
        ): Tensor = new(data.toUByteArray(), shape, device)

        /**
         * Repeat this tensor along the specified dimensions.
         */
        public fun repeat(
            tensor: Tensor,
            shape: Shape,
        ): Tensor = tensor.repeat(shape)

        /**
         * Creates grids of coordinates specified by the 1D inputs.
         *
         * # Arguments
         *
         * * `args` - A list of 1D tensors.
         * * `xyIndexing` - Whether to use xy indexing or ij indexing. If xy is selected, the
         *   first dimension corresponds to the cardinality of the second input and the second
         *   dimension corresponds to the cardinality of the first input. If ij is selected, the
         *   dimensions are in the same order as the cardinality of the inputs.
         *
         * ```kotlin
         * val x = Tensor.new(floatArrayOf(1f, 2f, 3f), Shape(3), Device.Cpu)
         * val y = Tensor.new(floatArrayOf(4f, 5f, 6f), Shape(3), Device.Cpu)
         * val gridsXy = Tensor.meshgrid(listOf(x, y), true)
         * val gridsIj = Tensor.meshgrid(listOf(x, y), false)
         * ```
         */
        public fun meshgrid(
            args: List<Tensor>,
            xyIndexing: Boolean,
        ): List<Tensor> {
            if (args.size <= 1) {
                throw CandleException.OpRequiresAtLeastTwoTensors("meshgrid")
            }
            val orderedArgs = if (xyIndexing) args.reversed() else args
            val shape = orderedArgs.map { it.dims1() }
            val grids = ArrayList<Tensor>(orderedArgs.size)
            for (idx in orderedArgs.indices) {
                val ones = MutableList(orderedArgs.size) { 1 }
                ones[idx] = shape[idx]
                val arg = orderedArgs[idx].reshape(Shape(ones))
                val repeats = shape.toMutableList()
                repeats[idx] = 1
                val repeated = arg.repeat(Shape(repeats))
                grids.add(repeated)
            }
            if (xyIndexing) {
                grids.reverse()
            }
            return grids
        }

        /**
         * Stacks two or more tensors along a particular dimension.
         *
         * All tensors must have the same rank, and the output has one additional rank.
         *
         * ```kotlin
         * val a = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
         * val b = Tensor.zeros(Shape(2, 3), DType.F32, Device.Cpu)
         * val c = Tensor.stack(listOf(a, b), 0)
         * assertEquals(listOf(2, 2, 3), c.shape().dims)
         * ```
         */
        public fun stack(
            tensors: List<Tensor>,
            dim: Int,
        ): Tensor {
            if (tensors.isEmpty()) {
                throw CandleException.OpRequiresAtLeastOneTensor("stack")
            }
            val unsqueezed = tensors.map { it.unsqueeze(dim) }
            return cat(unsqueezed, dim)
        }

        /**
         * Stacks two or more tensors along a particular dimension.
         */
        public fun stack(
            tensors: List<Tensor>,
            dim: Dim,
        ): Tensor = stack(tensors, dim.toIndexPlusOne(tensors[0].shape(), "stack"))

        /**
         * Returns a lower triangular matrix of ones of size `n` by `n`.
         */
        public fun tril2(
            n: Int,
            dtype: DType,
            device: Device,
        ): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.le(t2).toDtype(dtype)
        }

        /**
         * Returns an upper triangular matrix of ones of size `n` by `n`.
         */
        public fun triu2(
            n: Int,
            dtype: DType,
            device: Device,
        ): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.ge(t2).toDtype(dtype)
        }

        /**
         * Returns a matrix with a diagonal of ones of size `n` by `n`.
         */
        public fun eye(
            n: Int,
            dtype: DType,
            device: Device,
        ): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.eq(t2).toDtype(dtype)
        }
    }
}

public interface Read {
    public val readExact: (ByteArray) -> CandleResult<Unit>
    public val readU16Into: (UShortArray) -> CandleResult<Unit>
    public val readF32Into: (FloatArray) -> CandleResult<Unit>
    public val readF64Into: (DoubleArray) -> CandleResult<Unit>
    public val readU32Into: (UIntArray) -> CandleResult<Unit>
    public val readI16Into: (ShortArray) -> CandleResult<Unit>
    public val readI32Into: (IntArray) -> CandleResult<Unit>
    public val readI64Into: (LongArray) -> CandleResult<Unit>
}

public interface Write {
    public val writeAll: (ByteArray) -> CandleResult<Unit>
    public val writeU8: (UByte) -> CandleResult<Unit>
    public val writeU16: (UShort) -> CandleResult<Unit>
    public val writeI16: (Short) -> CandleResult<Unit>
    public val writeU32: (UInt) -> CandleResult<Unit>
    public val writeI32: (Int) -> CandleResult<Unit>
    public val writeI64: (Long) -> CandleResult<Unit>
    public val writeF32: (Float) -> CandleResult<Unit>
    public val writeF64: (Double) -> CandleResult<Unit>
}

public sealed class SeekFrom {
    public data class Start(
        public val pos: ULong,
    ) : SeekFrom()

    public data class End(
        public val offset: Long,
    ) : SeekFrom()

    public data class Current(
        public val offset: Long,
    ) : SeekFrom()
}

public open class NpyByteReader(
    internal val data: ByteArray,
    internal var pos: Int = 0,
    internal val sourcePath: String = "",
) : Read {
    public fun streamPosition(): CandleResult<ULong> = CandleResult.Ok(pos.toULong())

    internal fun readToEnd(buf: MutableList<Byte>): CandleResult<Int> {
        val count = data.size - pos
        for (i in pos until data.size) {
            buf.add(data[i])
        }
        pos = data.size
        return CandleResult.Ok(count)
    }

    internal fun readUntil(
        delim: Byte,
        buf: MutableList<Byte>,
    ): CandleResult<Int> {
        val start = pos
        while (pos < data.size) {
            val b = data[pos++]
            buf.add(b)
            if (b == delim) break
        }
        return CandleResult.Ok(pos - start)
    }

    public fun byRef(): NpyByteReader = this

    public fun take(limit: ULong): NpyByteReader {
        val count = minOf(limit.toInt(), data.size - pos)
        val slice = data.copyOfRange(pos, pos + count)
        pos += count
        return NpyByteReader(slice)
    }

    public fun seek(seekFrom: SeekFrom): CandleResult<ULong> {
        val next =
            when (seekFrom) {
                is SeekFrom.Start -> seekFrom.pos.toInt()
                is SeekFrom.End -> (data.size.toLong() + seekFrom.offset).toInt()
                is SeekFrom.Current -> (pos.toLong() + seekFrom.offset).toInt()
            }
        if (next < 0 || next > data.size) {
            return CandleResult.Err(CandleException.Io("invalid seek position: $next"))
        }
        pos = next
        return CandleResult.Ok(pos.toULong())
    }

    public fun readU8(): CandleResult<UByte> {
        val buf = ByteArray(1)
        val r = readExact(buf)
        return if (r is CandleResult.Err) r else CandleResult.Ok(buf[0].toUByte())
    }

    public fun readI8(): CandleResult<Byte> {
        val buf = ByteArray(1)
        val r = readExact(buf)
        return if (r is CandleResult.Err) r else CandleResult.Ok(buf[0])
    }

    public fun <B> readU16(): CandleResult<UShort> {
        val dst = UShortArray(1)
        val r = readU16Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readI16(): CandleResult<Short> {
        val dst = ShortArray(1)
        val r = readI16Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readU32(): CandleResult<UInt> {
        val dst = UIntArray(1)
        val r = readU32Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readI32(): CandleResult<Int> {
        val dst = IntArray(1)
        val r = readI32Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readU64(): CandleResult<ULong> {
        val dst = LongArray(1)
        val r = readI64Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0].toULong())
    }

    public fun <B> readI64(): CandleResult<Long> {
        val dst = LongArray(1)
        val r = readI64Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readF32(): CandleResult<Float> {
        val dst = FloatArray(1)
        val r = readF32Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public inline fun <reified B> readF64(): CandleResult<Double> {
        if (B::class == io.github.kotlinmania.byteorder.BigEndian::class) {
            val bytes = ByteArray(8)
            val r = readExact(bytes)
            if (r is CandleResult.Err) return r
            var bits = 0L
            for (b in 0 until 8) {
                bits = (bits shl 8) or (bytes[b].toLong() and 0xFFL)
            }
            return CandleResult.Ok(Double.fromBits(bits))
        }
        val dst = DoubleArray(1)
        val r = readF64Into.invoke(dst)
        return if (r is CandleResult.Err) r else CandleResult.Ok(dst[0])
    }

    public fun <B> readU32Into(dst: UIntArray): CandleResult<Unit> = readU32Into.invoke(dst)

    public fun <B> readU64Into(dst: ULongArray): CandleResult<Unit> {
        val tmp = LongArray(dst.size)
        val r = readI64Into.invoke(tmp)
        if (r is CandleResult.Err) return r
        for (i in dst.indices) dst[i] = tmp[i].toULong()
        return CandleResult.Ok(Unit)
    }

    override val readExact: (ByteArray) -> CandleResult<Unit> = { buf ->
        if (pos + buf.size > data.size) {
            CandleResult.Err(CandleException.Io("unexpected EOF"))
        } else {
            data.copyInto(buf, 0, pos, pos + buf.size)
            pos += buf.size
            CandleResult.Ok(Unit)
        }
    }

    override val readU16Into: (UShortArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 2)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val b0 = bytes[i * 2].toInt() and 0xFF
                val b1 = bytes[i * 2 + 1].toInt() and 0xFF
                dst[i] = ((b1 shl 8) or b0).toUShort()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readF32Into: (FloatArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                val bits =
                    (bytes[base].toInt() and 0xFF) or
                        ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                        ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                        ((bytes[base + 3].toInt() and 0xFF) shl 24)
                dst[i] = Float.fromBits(bits)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readF64Into: (DoubleArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 8)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 8
                var bits = 0L
                for (b in 0 until 8) {
                    bits = bits or ((bytes[base + b].toLong() and 0xFFL) shl (b * 8))
                }
                dst[i] = Double.fromBits(bits)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readU32Into: (UIntArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                val bits =
                    (bytes[base].toInt() and 0xFF) or
                        ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                        ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                        ((bytes[base + 3].toInt() and 0xFF) shl 24)
                dst[i] = bits.toUInt()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI16Into: (ShortArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 2)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val b0 = bytes[i * 2].toInt() and 0xFF
                val b1 = bytes[i * 2 + 1].toInt() and 0xFF
                dst[i] = ((b1 shl 8) or b0).toShort()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI32Into: (IntArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                dst[i] = (bytes[base].toInt() and 0xFF) or
                    ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[base + 3].toInt() and 0xFF) shl 24)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI64Into: (LongArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 8)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 8
                var bits = 0L
                for (b in 0 until 8) {
                    bits = bits or ((bytes[base + b].toLong() and 0xFFL) shl (b * 8))
                }
                dst[i] = bits
            }
            CandleResult.Ok(Unit)
        }
    }
}

public class NpyByteWriter : Write {
    private var buf: ByteArray = ByteArray(1024)
    private var size: Int = 0

    public fun streamPosition(): CandleResult<ULong> = CandleResult.Ok(size.toULong())

    public fun flush(): CandleResult<Unit> = CandleResult.Ok(Unit)

    public fun writeI8(v: Byte): CandleResult<Unit> = writeU8.invoke(v.toUByte())

    public fun <B> writeU16(v: UShort): CandleResult<Unit> = writeU16.invoke(v)

    public fun <B> writeI16(v: Short): CandleResult<Unit> = writeI16.invoke(v)

    public fun <B> writeU32(v: UInt): CandleResult<Unit> = writeU32.invoke(v)

    public fun <B> writeI32(v: Int): CandleResult<Unit> = writeI32.invoke(v)

    public fun <B> writeU64(v: ULong): CandleResult<Unit> = writeI64.invoke(v.toLong())

    public fun <B> writeI64(v: Long): CandleResult<Unit> = writeI64.invoke(v)

    public fun <B> writeF32(v: Float): CandleResult<Unit> = writeF32.invoke(v)

    public fun <B> writeF64(v: Double): CandleResult<Unit> = writeF64.invoke(v)

    private val ensureCapacity: (Int) -> Unit = { additional ->
        if (size + additional > buf.size) {
            val newSize = maxOf(buf.size * 2, size + additional)
            buf = buf.copyOf(newSize)
        }
    }

    override val writeAll: (ByteArray) -> CandleResult<Unit> = { bytes ->
        ensureCapacity(bytes.size)
        bytes.copyInto(buf, size)
        size += bytes.size
        CandleResult.Ok(Unit)
    }

    override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
        ensureCapacity(1)
        buf[size++] = v.toByte()
        CandleResult.Ok(Unit)
    }

    override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
        val bits = v.toInt()
        ensureCapacity(2)
        buf[size++] = (bits and 0xFF).toByte()
        buf[size++] = ((bits ushr 8) and 0xFF).toByte()
        CandleResult.Ok(Unit)
    }

    override val writeI16: (Short) -> CandleResult<Unit> = { v ->
        writeU16.invoke(v.toUShort())
    }

    override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
        val bits = v.toInt()
        ensureCapacity(4)
        buf[size++] = (bits and 0xFF).toByte()
        buf[size++] = ((bits ushr 8) and 0xFF).toByte()
        buf[size++] = ((bits ushr 16) and 0xFF).toByte()
        buf[size++] = ((bits ushr 24) and 0xFF).toByte()
        CandleResult.Ok(Unit)
    }

    override val writeI32: (Int) -> CandleResult<Unit> = { v ->
        writeU32.invoke(v.toUInt())
    }

    override val writeI64: (Long) -> CandleResult<Unit> = { v ->
        ensureCapacity(8)
        for (i in 0 until 8) {
            buf[size++] = ((v ushr (i * 8)) and 0xFFL).toByte()
        }
        CandleResult.Ok(Unit)
    }

    override val writeF32: (Float) -> CandleResult<Unit> = { v ->
        writeI32.invoke(v.toRawBits())
    }

    override val writeF64: (Double) -> CandleResult<Unit> = { v ->
        writeI64.invoke(v.toRawBits())
    }

    public val toByteArray: () -> ByteArray = {
        buf.copyOf(size)
    }
}

private val virtualFiles = mutableMapOf<String, ByteArray>()
private val virtualZips = mutableMapOf<String, Map<String, ByteArray>>()

internal fun platformReadFileBytes(path: String): ByteArray = virtualFiles[path] ?: ByteArray(0)

internal fun platformWriteFileBytes(
    path: String,
    bytes: ByteArray,
) {
    virtualFiles[path] = bytes
}

internal fun platformReadZipEntries(path: String): Map<String, ByteArray> = virtualZips[path].orEmpty()

internal fun platformWriteZipEntries(
    path: String,
    entries: Map<String, ByteArray>,
) {
    virtualZips[path] = entries.toMap()
}

internal class File(
    val path: String,
    readerData: ByteArray = ByteArray(0),
    private val writer: NpyByteWriter = NpyByteWriter(),
    private val isWrite: Boolean = false,
) : NpyByteReader(readerData, 0, path),
    Write {
    override val writeAll: (ByteArray) -> CandleResult<Unit> = { bytes ->
        val r = writer.writeAll.invoke(bytes)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
        val r = writer.writeU8.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
        val r = writer.writeU16.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI16: (Short) -> CandleResult<Unit> = { v ->
        val r = writer.writeI16.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
        val r = writer.writeU32.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI32: (Int) -> CandleResult<Unit> = { v ->
        val r = writer.writeI32.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI64: (Long) -> CandleResult<Unit> = { v ->
        val r = writer.writeI64.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeF32: (Float) -> CandleResult<Unit> = { v ->
        val r = writer.writeF32.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeF64: (Double) -> CandleResult<Unit> = { v ->
        val r = writer.writeF64.invoke(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }

    companion object {
        fun <P : Any> open(p: P): CandleResult<File> {
            val s = p.toString()
            return CandleResult.Ok(File(s, readerData = platformReadFileBytes(s)))
        }

        fun <P : Any> create(p: P): CandleResult<File> {
            val s = p.toString()
            return CandleResult.Ok(File(s, isWrite = true))
        }
    }
}

internal class BufReader(
    val inner: NpyByteReader,
) : NpyByteReader(inner.data, inner.pos, inner.sourcePath) {
    companion object {
        val new: (NpyByteReader) -> BufReader = { r -> BufReader(r) }
    }
}

internal class ZipEntryReader(
    private val entryName: String,
    bytes: ByteArray,
) : NpyByteReader(bytes) {
    val name: () -> String = { entryName }
}

internal object Cursor {
    fun new(buf: List<Byte>): NpyByteReader = NpyByteReader(buf.toByteArray())

    fun new(buf: ByteArray): NpyByteReader = NpyByteReader(buf)
}

internal class HashMap<K, V>(
    private val inner: kotlin.collections.HashMap<K, V> = kotlin.collections.HashMap(),
) : MutableMap<K, V> by inner {
    companion object {
        fun <K, V> new(): HashMap<K, V> = HashMap()

        fun <K, V> withCapacity(cap: Int): HashMap<K, V> = HashMap(kotlin.collections.HashMap(cap))
    }
}

internal object PathBuf {
    fun from(s: String): String = s
}

internal fun ioSink(): NpyByteWriter = NpyByteWriter()

internal fun ioCopy(
    reader: NpyByteReader,
    writer: NpyByteWriter,
): CandleResult<ULong> {
    val buf = mutableListOf<Byte>()
    val r = reader.readToEnd(buf)
    if (r is CandleResult.Err) return r
    val count = (r as CandleResult.Ok).value
    val wr = writer.writeAll(buf.toByteArray())
    if (wr is CandleResult.Err) return wr
    return CandleResult.Ok(count.toULong())
}

internal fun format(
    fmt: String,
    vararg args: Any?,
): String {
    var res = fmt
    for (arg in args) {
        val idx = res.indexOf("{}")
        if (idx >= 0) {
            res = res.substring(0, idx) + arg.toString() + res.substring(idx + 2)
        }
    }
    return res
}

internal fun <T> T.clone(): T = this

internal fun String.asRef(): String = this

internal fun String.asStr(): String = this

internal fun String.display(): String = this

internal fun String.toStringLossy(): String = this

internal fun String.intoOwned(): String = this

internal fun String.toOwned(): String = this

internal fun String.asBytes(): ByteArray = this.encodeToByteArray()

internal fun String.Companion.fromUtf8(bytes: ByteArray): CandleResult<String> = CandleResult.Ok(bytes.decodeToString())

internal fun String.Companion.fromUtf8Lossy(bytes: ByteArray): String = bytes.decodeToString()

internal fun <K, V> MutableMap<K, V>.insert(
    k: K,
    v: V,
): V? = this.put(k, v)

internal fun <T> List<T>.asSlice(): List<T> = this

internal fun <T> MutableList<T>.pop(): T? = if (this.isEmpty()) null else this.removeAt(this.size - 1)

internal fun <T> MutableList<T>.append(other: MutableList<T>) {
    this.addAll(other)
    other.clear()
}

internal fun <T> MutableList<T>.extend(other: Iterable<T>) {
    this.addAll(other)
}

internal fun <T> MutableList<T>.extend(other: List<T>) {
    this.addAll(other)
}

internal fun <T> MutableList<T>.lastMut(): T? = this.lastOrNull()

internal fun <T> MutableList<T>.splitOff(at: Int): MutableList<T> {
    val tail = this.subList(at, this.size).toMutableList()
    while (this.size > at) {
        this.removeAt(this.size - 1)
    }
    return tail
}

internal fun <T> T?.context(msg: String): CandleResult<T> =
    if (this != null) CandleResult.Ok(this) else CandleResult.Err(CandleException.Msg(msg))

internal fun <T> T?.okOrElse(errFn: () -> CandleException): CandleResult<T> =
    if (this != null) CandleResult.Ok(this) else CandleResult.Err(errFn())

internal fun eprintln(msg: String) {
    println(msg)
}

internal data class Quadruple<out A, out B, out C, out D>(
    val first: A,
    val second: B,
    val third: C,
    val fourth: D,
)

internal fun ULong.saturatingSub(other: ULong): ULong = if (this > other) this - other else 0uL

internal fun ULong.saturatingAdd(other: ULong): ULong {
    val res = this + other
    return if (res < this) ULong.MAX_VALUE else res
}

internal fun ULong.saturatingMul(other: ULong): ULong {
    if (this == 0uL || other == 0uL) return 0uL
    val res = this * other
    return if (res / this != other) ULong.MAX_VALUE else res
}

internal fun ULong.divCeil(rhs: ULong): ULong = (this + rhs - 1uL) / rhs

internal fun UByte.Companion.from(b: Boolean): UByte = if (b) 1u else 0u

internal fun <T> Set<T>.len(): Int = this.size

internal fun <T> Set<T>.intoIter(): Iterator<T> = this.iterator()

internal fun UIntArray.intoIter(): List<UInt> = this.toList()

private fun UIntArray.iter(): List<UInt> = this.toList()

internal fun ULongArray.intoIter(): List<ULong> = this.toList()

internal fun <T> List<T>.intoIter(): List<T> = this

internal fun <T> List<T>.iter(): List<T> = this

internal fun <T> List<T>.rev(): List<T> = this.reversed()

internal fun <T> List<T>.enumerate(): List<IndexedValue<T>> = this.withIndex().toList()

internal fun <T> List<T>.collect(): MutableList<T> = this.toMutableList()

internal fun <T> MutableList<T>.push(v: T) {
    this.add(v)
}

internal fun <K, V> List<Pair<K, V>>.collectMap(): Map<K, V> = this.toMap()

internal fun <T> List<T>.cloned(): List<T> = this.toList()

internal fun IntProgression.collect(): List<Int> = this.toList()

internal fun List<Int>.product(): Int = this.fold(1) { acc, x -> acc * x }

internal object Zip {
    enum class CompressionMethod {
        Stored,
        ;

        companion object
    }

    object write {
        class FileOptions<T> {
            val compressionMethod: (CompressionMethod) -> FileOptions<T> = { this }

            companion object {
                val default: () -> FileOptions<Unit> = { FileOptions() }
            }
        }
    }

    class ZipArchive(
        private val entries: List<Pair<String, ByteArray>>,
    ) {
        val len: () -> Int = { entries.size }

        fun fileNames(): List<String> = entries.map { it.first }

        val byIndex: (Int) -> CandleResult<ZipEntryReader> = { i ->
            val e = entries[i]
            CandleResult.Ok(ZipEntryReader(e.first, e.second))
        }
        val byName: (String) -> CandleResult<ZipEntryReader> = { target ->
            val found = entries.firstOrNull { it.first == target }
            if (found != null) {
                CandleResult.Ok(ZipEntryReader(found.first, found.second))
            } else {
                CandleResult.Err(CandleException.Zip("entry not found: $target"))
            }
        }

        companion object {
            val new: (BufReader) -> CandleResult<ZipArchive> = { br ->
                val map = platformReadZipEntries(br.sourcePath)
                CandleResult.Ok(ZipArchive(map.entries.map { entry -> kotlin.Pair(entry.key, entry.value) }))
            }
        }
    }

    class ZipWriter(
        private val file: File,
    ) : Write {
        val write = Zip.write
        val CompressionMethod = Zip.CompressionMethod
        private val entries = linkedMapOf<String, ByteArray>()
        private var currentName: String = ""
        private var currentWriter: NpyByteWriter = NpyByteWriter()

        private val flushCurrent: () -> Unit = {
            if (currentName.isNotEmpty()) {
                entries[currentName] = currentWriter.toByteArray()
                platformWriteZipEntries(file.path, entries)
            }
        }

        val startFile: (String, write.FileOptions<Unit>) -> CandleResult<Unit> = { name, _ ->
            flushCurrent()
            currentName = name
            currentWriter = NpyByteWriter()
            CandleResult.Ok(Unit)
        }

        override val writeAll: (ByteArray) -> CandleResult<Unit> = { b ->
            val r = currentWriter.writeAll.invoke(b)
            flushCurrent()
            r
        }
        override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU8.invoke(v)
            flushCurrent()
            r
        }
        override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU16.invoke(v)
            flushCurrent()
            r
        }
        override val writeI16: (Short) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI16.invoke(v)
            flushCurrent()
            r
        }
        override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU32.invoke(v)
            flushCurrent()
            r
        }
        override val writeI32: (Int) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI32.invoke(v)
            flushCurrent()
            r
        }
        override val writeI64: (Long) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI64.invoke(v)
            flushCurrent()
            r
        }
        override val writeF32: (Float) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeF32.invoke(v)
            flushCurrent()
            r
        }
        override val writeF64: (Double) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeF64.invoke(v)
            flushCurrent()
            r
        }

        companion object {
            val new: (File) -> ZipWriter = { f -> ZipWriter(f) }
        }
    }
}
