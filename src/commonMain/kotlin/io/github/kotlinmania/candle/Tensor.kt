// port-lint: source tensor.rs
package io.github.kotlinmania.candle

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

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
 * Multi-dimensional array supporting mathematical operations, backpropagation, and execution on devices.
 */
public class Tensor internal constructor(
    private val id: TensorId,
    private var storage: Storage,
    private val layout: Layout,
    private val op: BackpropOp,
    private val isVariable: Boolean,
) {
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

    public fun dims0() {
        shape().dims0()
    }

    public fun dims1(): Int = shape().dims1()

    public fun dims2(): Dims2 = shape().dims2()

    public fun dims3(): Dims3 = shape().dims3()

    public fun dims4(): Dims4 = shape().dims4()

    public fun dims5(): Dims5 = shape().dims5()

    public fun contiguousOffsets(): ContiguousOffsets? = layout.contiguousOffsets()

    /**
     * The dimension size for a specified dimension index.
     */
    public fun dim(d: Int): Int = layout.dim(d)

    /**
     * The dimension size for a specified dimension index.
     */
    public fun dim(d: ShapeDim): Int = layout.dim(d.toIndex(shape(), "dim"))

    /**
     * Returns a new tensor that is a narrowed version of the input, the dimension dim
     * ranges from start to start + len.
     */
    public fun narrow(dim: ShapeDim, start: Int, len: Int): Tensor =
        narrow(dim.toIndex(shape(), "narrow"), start, len)

    /**
     * Run the forward method of module on self.
     */
    public fun apply(m: Module): Tensor = m.forward(this)

    /**
     * Run the forward method of module on self with train mode.
     */
    public fun applyT(m: ModuleT, train: Boolean): Tensor = m.forwardT(this, train)

    public fun checkDim(dim: Int, op: String) {
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

    public fun storage(): Storage = storage

    public fun storageMut(): Storage = storage

    /**
     * The storage used by this tensor, together with the layout to use to access it safely.
     */
    public fun storageAndLayout(): Pair<Storage, Layout> = Pair(storage, layout)

    public fun storageMutAndLayout(): Pair<Storage, Layout> = Pair(storage, layout)

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
     * Creates a new tensor filled with zeros with same shape, dtype, and device as the other tensor.
     */
    public fun zerosLike(): Tensor = zeros(shape(), dtype(), device())

    /**
     * Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
     */
    public fun onesLike(): Tensor = ones(shape(), dtype(), device())

    /**
     * Creates a new tensor filled with uninitialized memory of the same shape, dtype, and device.
     */
    public fun emptyLike(): Tensor = empty(shape(), dtype(), device())

    /**
     * Returns a new tensor initialized with values sampled uniformly between lo and up.
     */
    public fun randLike(lo: Double, up: Double): Tensor =
        randF64Impl(lo, up, shape(), dtype(), device(), false)

    /**
     * Returns a new tensor initialized with values sampled from a normal distribution with mean and stdev.
     */
    public fun randnLike(mean: Double, stdev: Double): Tensor =
        randnF64Impl(mean, stdev, shape(), dtype(), device(), false)

    /**
     * Returns a new tensor detached from the current graph, gradient are not propagated through
     * this new node. The storage of this tensor is shared with the initial tensor.
     */
    public fun detach(): Tensor = Tensor(TensorId.next(), storage, layout, BackpropOp.None, false)

    internal fun sameStorage(rhs: Tensor): Boolean = this.storage === rhs.storage

    internal fun sameShapeBinaryOp(rhs: Tensor, op: String): Shape {
        val lhs = shape()
        val r = rhs.shape()
        if (lhs != r) {
            throw CandleException.ShapeMismatchBinaryOp(lhs, r, op)
        }
        return lhs
    }

    // --- Unary Ops ---
    public fun exp(): Tensor = unaryOp(UnaryOp.Exp)

    public fun log(): Tensor = unaryOp(UnaryOp.Log)

    public fun sin(): Tensor = unaryOp(UnaryOp.Sin)

    public fun cos(): Tensor = unaryOp(UnaryOp.Cos)

    public fun abs(): Tensor = unaryOp(UnaryOp.Abs)

    public fun neg(): Tensor = unaryOp(UnaryOp.Neg)

    public operator fun unaryMinus(): Tensor = neg()

    public fun recip(): Tensor = unaryOp(UnaryOp.Recip)

    public fun sqr(): Tensor = unaryOp(UnaryOp.Sqr)

    public fun sqrt(): Tensor = unaryOp(UnaryOp.Sqrt)

    public fun gelu(): Tensor = unaryOp(UnaryOp.Gelu)

    public fun geluErf(): Tensor = unaryOp(UnaryOp.GeluErf)

    public fun erf(): Tensor = unaryOp(UnaryOp.Erf)

    public fun relu(): Tensor = unaryOp(UnaryOp.Relu)

    public fun silu(): Tensor = unaryOp(UnaryOp.Silu)

    public fun tanh(): Tensor = unaryOp(UnaryOp.Tanh)

    public fun floor(): Tensor = unaryOp(UnaryOp.Floor)

    public fun ceil(): Tensor = unaryOp(UnaryOp.Ceil)

    public fun round(): Tensor = unaryOp(UnaryOp.Round)

    public fun sign(): Tensor = unaryOp(UnaryOp.Sign)

    private fun unaryOp(uOp: UnaryOp): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.unaryImpl(layout, uOp)
        val newOp = BackpropOp.new1(this) { Op.Unary(it, uOp) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Round element of the input tensor to the nearest integer.
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
        var inp = if (rank() < repeats.size) {
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
     * This operation multiplies the input tensor by mul then adds add and return the result.
     */
    public fun affine(mul: Double, add: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.affine(layout, mul, add)
        val newOp = BackpropOp.new1(this) { Op.Affine(it, mul, add) }
        return fromStorage(newStorage, layout.shape, newOp, false)
    }

    /**
     * Raise the tensor to some float exponent e.
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
     * Split a tensor into the specified number of chunks, this may return less chunks than specified.
     */
    public fun chunk(chunks: Int, dim: Int): List<Tensor> {
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
     * Split a tensor into the specified number of chunks along dimension dim.
     */
    public fun chunk(chunks: Int, dim: ShapeDim): List<Tensor> =
        chunk(chunks, dim.toIndex(shape(), "chunk"))

    internal fun squeezeDims(dims: List<Int>): Tensor =
        when (dims.size) {
            0 -> this
            1 -> squeeze(dims[0])
            else -> {
                val remainingDims = dims().filterIndexed { idx, _ -> idx !in dims }
                reshape(Shape(remainingDims))
            }
        }

    internal fun reduceImpl(dim: Int, keepdim: Boolean, op: ReduceOp): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), op.name)
        val newStorage = storage.reduceOp(op, layout, listOf(dimIdx))
        val newDims = dims().toMutableList()
        newDims[dimIdx] = 1
        val newOp = when (op) {
            ReduceOp.Sum, ReduceOp.Min, ReduceOp.Max ->
                BackpropOp.new1(this) { Op.Reduce(it, op, newDims) }
            ReduceOp.ArgMin, ReduceOp.ArgMax -> BackpropOp.None
        }
        val res = fromStorage(newStorage, Shape(newDims), newOp, false)
        return if (keepdim) res else res.squeezeDims(listOf(dimIdx))
    }

    internal fun sumImpl(sumDims: List<Int>, keepdim: Boolean): Tensor {
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
     * Elements shifted beyond the last position are re-introduced at the first position.
     */
    public fun roll(shift: Int, dim: Int): Tensor {
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
     */
    public fun roll(shift: Int, dim: ShapeDim): Tensor =
        roll(shift, dim.toIndex(shape(), "roll"))

    // --- Binary Ops ---
    public fun add(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Add)

    public fun add(rhs: Double): Tensor = affine(1.0, rhs)

    public operator fun plus(rhs: Tensor): Tensor = add(rhs)

    public fun sub(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Sub)

    public fun sub(rhs: Double): Tensor = affine(1.0, -rhs)

    public operator fun minus(rhs: Tensor): Tensor = sub(rhs)

    public fun mul(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Mul)

    public fun mul(rhs: Double): Tensor = affine(rhs, 0.0)

    public operator fun times(rhs: Tensor): Tensor = mul(rhs)

    public operator fun div(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Div)

    public operator fun div(rhs: Double): Tensor = affine(1.0 / rhs, 0.0)

    public fun maximum(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Maximum)

    public fun minimum(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Minimum)

    private fun binaryOp(rhs: Tensor, bOp: BinaryOp): Tensor {
        val targetShape = shape().broadcastShapeBinaryOp(rhs.shape(), bOp.name)
        val lhsL = layout.broadcastAs(targetShape)
        val rhsL = rhs.layout.broadcastAs(targetShape)
        val newStorage = storage.binaryImpl(rhs.storage, lhsL, rhsL, bOp)
        val newOp = BackpropOp.new2(this, rhs) { l, r -> Op.Binary(l, r, bOp) }
        return fromStorage(newStorage, targetShape, newOp, false)
    }

    public fun broadcastAdd(rhs: Tensor): Tensor = add(rhs)

    public fun broadcastSub(rhs: Tensor): Tensor = sub(rhs)

    public fun broadcastMul(rhs: Tensor): Tensor = mul(rhs)

    public fun broadcastDiv(rhs: Tensor): Tensor = div(rhs)

    public fun broadcastMaximum(rhs: Tensor): Tensor = maximum(rhs)

    public fun broadcastMinimum(rhs: Tensor): Tensor = minimum(rhs)

    // --- Reductions ---
    /**
     * Returns the sum of all elements in the input tensor over the specified dimensions.
     */
    public fun sum(dims: List<Int>): Tensor = sumImpl(dims, false)

    public fun sum(vararg dims: Int): Tensor = sum(dims.toList())

    /**
     * Returns the sum of all elements in the input tensor preserving reduced dimensions as 1.
     */
    public fun sumKeepdim(dim: Int): Tensor = sumImpl(listOf(dim), true)

    public fun sumKeepdim(dims: List<Int>): Tensor = sumImpl(dims, true)

    /**
     * Computes the sum of all the elements in this tensor.
     */
    public fun sumAll(): Tensor = sum((0 until rank()).toList())

    /**
     * Similar to minKeepdim but the target dimension is squeezed.
     */
    public fun min(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.Min)

    /**
     * Gathers the minimum value across the selected dimension keeping reduced dimension as 1.
     */
    public fun minKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.Min)

    /**
     * Computes the min of all the elements in this tensor.
     */
    public fun minAll(): Tensor = if (rank() == 0) this else flattenAll().min(0)

    /**
     * Similar to maxKeepdim but the target dimension is squeezed.
     */
    public fun max(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.Max)

    /**
     * Gathers the maximum value across the selected dimension keeping reduced dimension as 1.
     */
    public fun maxKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.Max)

    /**
     * Computes the max of all the elements in this tensor.
     */
    public fun maxAll(): Tensor = if (rank() == 0) this else flattenAll().max(0)

    /**
     * Similar to argminKeepdim but the target dimension is squeezed.
     */
    public fun argmin(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.ArgMin)

    /**
     * Returns the indices of the minimum values across the selected dimension keeping dimension as 1.
     */
    public fun argminKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.ArgMin)

    /**
     * Similar to argmaxKeepdim but the target dimension is squeezed.
     */
    public fun argmax(dim: Int): Tensor = reduceImpl(dim, false, ReduceOp.ArgMax)

    /**
     * Returns the indices of the maximum values across the selected dimension keeping dimension as 1.
     */
    public fun argmaxKeepdim(dim: Int): Tensor = reduceImpl(dim, true, ReduceOp.ArgMax)

    /**
     * Returns the mean of all elements in the input tensor over the specified dimension.
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
     * Returns the mean of all elements across the specified dimensions keeping reduced dimensions as 1.
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

    public fun var_(dim: ShapeDim): Tensor = `var`(dim.toIndex(shape(), "var"))

    // --- Comparisons ---
    /**
     * Element-wise comparison between two tensors with the specified comparison op.
     */
    public fun cmp(rhs: Tensor, op: CmpOp): Tensor {
        val targetShape = sameShapeBinaryOp(rhs, "cmp")
        val newStorage = storage.cmp(op, rhs.storage, layout, rhs.layout)
        val newOp = BackpropOp.new1(this) { Op.Cmp(it, op) }
        return fromStorage(newStorage, targetShape, newOp, false)
    }

    /**
     * Element-wise comparison with TensorOrScalar.
     */
    public fun cmp(rhs: TensorOrScalar, op: CmpOp): Tensor {
        val r = when (val ts = rhs.toTensorScalar()) {
            is TensorScalar.TensorVal -> ts.value
            is TensorScalar.ScalarVal -> ts.value.toDtype(dtype()).toDevice(device()).broadcastAs(shape())
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
     * Element-wise comparison with lower-equal.
     */
    public fun le(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Le)

    /**
     * Element-wise comparison with greater-equal.
     */
    public fun ge(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Ge)

    /**
     * Element-wise comparison with lower-than.
     */
    public fun lt(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Lt)

    /**
     * Element-wise comparison with greater-than.
     */
    public fun gt(rhs: Tensor): Tensor = cmp(rhs, CmpOp.Gt)

    public fun broadcastEq(rhs: Tensor): Tensor = eq(rhs)

    public fun broadcastNe(rhs: Tensor): Tensor = ne(rhs)

    public fun broadcastLt(rhs: Tensor): Tensor = lt(rhs)

    public fun broadcastLe(rhs: Tensor): Tensor = le(rhs)

    public fun broadcastGt(rhs: Tensor): Tensor = gt(rhs)

    public fun broadcastGe(rhs: Tensor): Tensor = ge(rhs)

    /**
     * Clamp the tensor values to be between min and max.
     */
    public fun clamp(min: Tensor, max: Tensor): Tensor = maximum(min).minimum(max)

    /**
     * Clamp the tensor values to be between min and max scalar values.
     */
    public fun clamp(min: Double, max: Double): Tensor {
        val minT = fromStorage(Storage.Cpu(CpuStorage.F64(doubleArrayOf(min))), Shape(listOf(1)), BackpropOp.None, false).broadcastAs(shape())
        val maxT = fromStorage(Storage.Cpu(CpuStorage.F64(doubleArrayOf(max))), Shape(listOf(1)), BackpropOp.None, false).broadcastAs(shape())
        return clamp(minT, maxT)
    }

    // --- Interpolation & Pooling ---
    /**
     * Interpolate the input tensor to the targetSize size, taking the value of the nearest element.
     */
    public fun interpolate1d(targetSize: Int): Tensor {
        val d = dims3()
        val op = BackpropOp.new1(this) { Op.UpsampleNearest1D(it, targetSize) }
        val newStorage = storage.upsampleNearest1d(layout, targetSize)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, targetSize)), op, false)
    }

    /**
     * Alias for interpolate1d.
     */
    public fun upsampleNearest1d(targetSize: Int): Tensor = interpolate1d(targetSize)

    /**
     * Interpolate the input tensor to the (targetH, targetW) size using nearest neighbor.
     */
    public fun interpolate2d(targetH: Int, targetW: Int): Tensor {
        val d = dims4()
        val op = BackpropOp.new1(this) { Op.UpsampleNearest2D(it, targetH, targetW) }
        val newStorage = storage.upsampleNearest2d(layout, targetH, targetW)
        return fromStorage(newStorage, Shape(listOf(d.d0, d.d1, targetH, targetW)), op, false)
    }

    /**
     * Alias for interpolate2d.
     */
    public fun upsampleNearest2d(targetH: Int, targetW: Int): Tensor =
        interpolate2d(targetH, targetW)

    /**
     * Bilinear interpolation to resize the input tensor to the specified size.
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
     */
    public fun avgPool2d(sz: Int): Tensor = avgPool2dWithStride(Pair(sz, sz), Pair(sz, sz))

    public fun avgPool2d(sz: Pair<Int, Int>): Tensor = avgPool2dWithStride(sz, sz)

    /**
     * 2D average pooling with custom kernel size and stride.
     */
    public fun avgPool2dWithStride(kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Tensor {
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
     */
    public fun maxPool2d(sz: Int): Tensor = maxPool2dWithStride(Pair(sz, sz), Pair(sz, sz))

    public fun maxPool2d(sz: Pair<Int, Int>): Tensor = maxPool2dWithStride(sz, sz)

    /**
     * 2D max pooling with custom kernel size and stride.
     */
    public fun maxPool2dWithStride(kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Tensor {
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
     */
    public fun dot(rhs: Tensor): Tensor {
        if (dims().size != 1 || rhs.dims().size != 1) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), rhs.shape(), "dot")
        }
        return (this * rhs).sumAll()
    }

    /**
     * Computes the Frobenius norm of the tensor.
     */
    public fun norm(): Tensor {
        if (dtype().isInt()) {
            bail("norm not supported for integer dtypes")
        }
        return sqr().sumAll().sqrt()
    }

    /**
     * Performs strict matrix-vector multiplication.
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
     */
    public fun matmul(rhs: Tensor): Tensor {
        val (lhsBcastShape, rhsBcastShape) = shape().broadcastShapeMatmul(rhs.shape())
        val lhsL = layout.broadcastAs(lhsBcastShape)
        val rhsL = rhs.layout.broadcastAs(rhsBcastShape)

        val lhsDims = lhsBcastShape.dims
        val rhsDims = rhsBcastShape.dims
        val m = lhsDims[lhsDims.size - 2]
        val k = lhsDims[lhsDims.size - 1]
        val n = rhsDims[rhsDims.size - 1]
        val b = lhsDims.subList(0, lhsDims.size - 2).fold(1) { acc, d -> acc * d }

        val newStorage = storage.matmul(rhs.storage, listOf(b, m, n, k), lhsL, rhsL)
        val outDims = lhsDims.subList(0, lhsDims.size - 2) + listOf(m, n)
        val outShape = Shape(outDims)
        val newOp = BackpropOp.new2(this, rhs) { l, r -> Op.Matmul(l, r) }
        return fromStorage(newStorage, outShape, newOp, false)
    }

    /**
     * Matrix-multiplication with broadcasting support.
     */
    public fun broadcastMatmul(rhs: Tensor): Tensor = matmul(rhs)

    /**
     * Returns a tensor with elements from onTrue if self is not zero, and onFalse where self is zero.
     */
    public fun whereCond(onTrue: Tensor, onFalse: Tensor): Tensor {
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
     * Returns a tensor with values from self at the indices corresponding to ids.
     */
    public fun embedding(ids: Tensor): Tensor {
        if (rank() != 2 || ids.rank() != 1) {
            throw CandleException.ShapeMismatchBinaryOp(shape(), ids.shape(), "embedding")
        }
        return indexSelect(ids, 0)
    }

    internal fun scatterChecks(indexes: Tensor, source: Tensor, dim: Int) {
        val sourceDims = source.dims()
        val selfDims = dims()
        val mismatch = if (sourceDims.size != selfDims.size) {
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
     * Scatter source values into a clone of self based on indexes along dimension dim.
     */
    public fun scatter(indexes: Tensor, source: Tensor, dim: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "scatter")
        scatterChecks(indexes, source, dimIdx)
        val sh = shape()
        val newStorage = storage.tryClone(layout)
        val l = Layout.contiguous(sh)
        newStorage.scatterSet(l, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
        val newOp = BackpropOp.new3(this, indexes, source) { t1, t2, t3 -> Op.Scatter(t1, t2, t3, dimIdx) }
        return fromStorage(newStorage, sh, newOp, false)
    }

    public fun scatter(indexes: Tensor, source: Tensor, dim: ShapeDim): Tensor =
        scatter(indexes, source, dim.toIndex(shape(), "scatter"))

    /**
     * Scatter source values directly into self based on indexes along dimension dim.
     */
    public fun scatterSet(indexes: Tensor, source: Tensor, dim: Int) {
        if (sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-set")
        scatterChecks(indexes, source, dimIdx)
        storage.scatterSet(layout, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
    }

    public fun scatterSet(indexes: Tensor, source: Tensor, dim: ShapeDim) =
        scatterSet(indexes, source, dim.toIndex(shape(), "scatter-set"))

    /**
     * Accumulate source values into self based on indexes along dimension dim.
     */
    public fun scatterAdd(indexes: Tensor, source: Tensor, dim: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-add")
        scatterChecks(indexes, source, dimIdx)
        val sh = shape()
        val newStorage = storage.tryClone(layout)
        val l = Layout.contiguous(sh)
        newStorage.scatterAdd(l, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
        val newOp = BackpropOp.new3(this, indexes, source) { t1, t2, t3 -> Op.ScatterAdd(t1, t2, t3, dimIdx) }
        return fromStorage(newStorage, sh, newOp, false)
    }

    public fun scatterAdd(indexes: Tensor, source: Tensor, dim: ShapeDim): Tensor =
        scatterAdd(indexes, source, dim.toIndex(shape(), "scatter-add"))

    /**
     * Accumulate source values directly into self based on indexes along dimension dim.
     */
    public fun scatterAddSet(indexes: Tensor, source: Tensor, dim: Int) {
        if (sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dimIdx = Shape.toIndex(dim, shape(), "scatter-add-set")
        scatterChecks(indexes, source, dimIdx)
        storage.scatterAdd(layout, indexes.storage, indexes.layout, source.storage, source.layout, dimIdx)
    }

    public fun scatterAddSet(indexes: Tensor, source: Tensor, dim: ShapeDim) =
        scatterAddSet(indexes, source, dim.toIndex(shape(), "scatter-add-set"))

    /**
     * Embeds the values of the src tensor into the self tensor on the first dimension.
     */
    public fun sliceScatter0(src: Tensor, start: Int): Tensor {
        if (dtype() != src.dtype()) {
            throw CandleException.DTypeMismatchBinaryOp(dtype(), src.dtype(), "slice-scatter")
        }
        if (device().location() != src.device().location()) {
            throw CandleException.DeviceMismatchBinaryOp(device().location(), src.device().location(), "slice-scatter")
        }
        if (rank() != src.rank()) {
            throw CandleException.UnexpectedNumberOfDims(rank(), src.rank(), src.shape())
        }
        val shapeOk = dims().indices.all { dimIdx ->
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
     * Embeds the values of the src tensor into the self tensor on the specified dimension.
     */
    public fun sliceScatter(src: Tensor, dim: Int, start: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "slice-scatter")
        return if (dimIdx == 0) {
            sliceScatter0(src, start)
        } else {
            transpose(0, dimIdx)
                .sliceScatter0(src.transpose(0, dimIdx), start)
                .transpose(0, dimIdx)
        }
    }

    public fun sliceScatter(src: Tensor, dim: ShapeDim, start: Int): Tensor =
        sliceScatter(src, dim.toIndex(shape(), "slice-scatter"), start)

    /**
     * Accumulate elements from source at indices indexes and add them to self.
     */
    public fun indexAdd(indexes: Tensor, source: Tensor, dim: Int): Tensor {
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

    public fun indexAdd(indexes: Tensor, source: Tensor, dim: ShapeDim): Tensor =
        indexAdd(indexes, source, dim.toIndex(shape(), "index-add"))

    // --- Indexing ---
    /**
     * Gather values across the target dimension.
     */
    public fun gather(indexes: Tensor, dim: Int): Tensor {
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

    public fun gather(indexes: Tensor, dim: ShapeDim): Tensor {
        val normalizedDim = dim.toIndex(shape(), "gather")
        return gather(indexes, normalizedDim)
    }

    /**
     * Select values for the input tensor at the target indexes across the specified dimension.
     */
    public fun indexSelect(indexes: Tensor, dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "index_select")
        val newStorage = storage.indexSelect(indexes.storage, layout, indexes.layout, normalizedDim)
        val outDims = dims().toMutableList()
        outDims[normalizedDim] = indexes.elemCount()
        return fromStorage(newStorage, Shape(outDims), BackpropOp.None, false)
    }

    public fun indexSelect(indexes: Tensor, dim: ShapeDim): Tensor =
        indexSelect(indexes, dim.toIndex(shape(), "index-select"))

    /**
     * Returns an iterator over position of the elements in the storage.
     */
    public fun stridedIndex(): StridedIndex = layout.stridedIndex()

    /**
     * Returns the position of the start and length of contiguous blocks.
     */
    public fun stridedBlocks(): StridedBlocks = layout.stridedBlocks()

    // --- Data Accessors ---
    public fun asRef(): Tensor = this

    public fun deref(): Tensor = this

    public fun <S> toVec1(): List<S> {
        val count = elemCount()
        if (rank() != 1) throw CandleException.UnexpectedNumberOfDims(1, rank(), shape())
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val list = ArrayList<S>(count)
        for (i in 0 until count) {
            val v: Any = cpu.getAsDouble(i)
            list.add(v as S)
        }
        return list
    }

    public fun <S> toVec2(): List<List<S>> {
        val (dim1, dim2) = dims2()
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val result = ArrayList<List<S>>(dim1)
        for (i in 0 until dim1) {
            val row = ArrayList<S>(dim2)
            for (j in 0 until dim2) {
                val idx = i * dim2 + j
                val v: Any = cpu.getAsDouble(idx)
                row.add(v as S)
            }
            result.add(row)
        }
        return result
    }

    public fun <S> toVec3(): List<List<List<S>>> {
        val (dim1, dim2, dim3) = dims3()
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val result = ArrayList<List<List<S>>>(dim1)
        for (i in 0 until dim1) {
            val page = ArrayList<List<S>>(dim2)
            for (j in 0 until dim2) {
                val row = ArrayList<S>(dim3)
                for (k in 0 until dim3) {
                    val idx = (i * dim2 + j) * dim3 + k
                    val v: Any = cpu.getAsDouble(idx)
                    row.add(v as S)
                }
                page.add(row)
            }
            result.add(page)
        }
        return result
    }

    public fun <S> toScalar(): S {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1<S>()[0]
    }

    public fun <S> toVec0(): S = toScalar<S>()

    public fun toVec1F32(): FloatArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = FloatArray(count)
        for (i in 0 until count) res[i] = cpu.getAsFloat(i)
        return res
    }

    public fun toVec1F64(): DoubleArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = DoubleArray(count)
        for (i in 0 until count) res[i] = cpu.getAsDouble(i)
        return res
    }

    public fun toScalarF32(): Float {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1F32()[0]
    }

    public fun toScalarF64(): Double {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1F64()[0]
    }

    public fun toVec1I64(): LongArray {
        val cont = contiguous()
        val cpu = (cont.storage as Storage.Cpu).storage
        val count = elemCount()
        val res = LongArray(count)
        for (i in 0 until count) res[i] = cpu.getAsLong(i)
        return res
    }

    public fun toScalarI64(): Long {
        if (elemCount() != 1) throw CandleException.UnexpectedNumberOfDims(0, rank(), shape())
        return toVec1I64()[0]
    }

    // --- Shape manipulations ---
    /**
     * Reshape returns a tensor with the target shape provided element count matches.
     */
    public fun reshape(shape: Shape): Tensor {
        if (shape.elemCount() != elemCount()) {
            throw CandleException.ShapeMismatch(elemCount(), shape)
        }
        if (isContiguous()) {
            val newLayout = Layout.contiguousWithOffset(shape, layout.startOffset)
            return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
        } else {
            return contiguous().reshape(shape)
        }
    }

    public fun reshape(vararg dims: Int): Tensor =
        reshape(dims.toList().intoShapeWithHole(elemCount()))

    internal fun flatten_(startDim: Int?, endDim: Int?): Tensor {
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
     */
    public fun flattenAll(): Tensor = flatten_(null, null)

    /**
     * Flattens the input tensor on dimension indexes from startDim (inclusive) to the last dimension.
     */
    public fun flattenFrom(dim: Int): Tensor = flatten_(dim, null)

    public fun flattenFrom(dim: ShapeDim): Tensor = flattenFrom(dim.toIndex(shape(), "flatten_from"))

    /**
     * Flattens the input tensor on dimension indexes from 0 to endDim (inclusive).
     */
    public fun flattenTo(dim: Int): Tensor = flatten_(null, dim)

    public fun flattenTo(dim: ShapeDim): Tensor = flattenTo(dim.toIndex(shape(), "flatten_to"))

    /**
     * Flattens the input tensor on dimension indexes from startDim to endDim (both inclusive).
     */
    public fun flatten(startDim: Int, endDim: Int): Tensor = flatten_(startDim, endDim)

    public fun flatten(startDim: ShapeDim, endDim: ShapeDim): Tensor =
        flatten(startDim.toIndex(shape(), "flatten"), endDim.toIndex(shape(), "flatten"))

    /**
     * Creates a new tensor with the specified dimension removed if its size was one.
     */
    public fun squeeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "squeeze")
        if (dims()[normalizedDim] != 1) return this
        val newDims = dims().toMutableList()
        newDims.removeAt(normalizedDim)
        val newStride = stride().toMutableList()
        newStride.removeAt(normalizedDim)
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), BackpropOp.None, false)
    }

    public fun squeeze(dim: ShapeDim): Tensor = squeeze(dim.toIndex(shape(), "squeeze"))

    /**
     * Creates a new tensor with a dimension of size one inserted at the specified position.
     */
    public fun unsqueeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndexPlusOne(dim, shape(), "unsqueeze")
        val newDims = dims().toMutableList()
        newDims.add(normalizedDim, 1)
        val newStride = stride().toMutableList()
        val s = if (normalizedDim < stride().size) stride()[normalizedDim] else 1
        newStride.add(normalizedDim, s)
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), BackpropOp.None, false)
    }

    public fun unsqueeze(dim: ShapeDim): Tensor = unsqueeze(dim.toIndexPlusOne(shape(), "unsqueeze"))

    /**
     * Returns a tensor that is a transposed version of the input, the given dimensions are swapped.
     */
    public fun transpose(dim1: Int, dim2: Int): Tensor {
        val newLayout = layout.transpose(dim1, dim2)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun transpose(dim1: ShapeDim, dim2: ShapeDim): Tensor =
        transpose(dim1.toIndex(shape(), "transpose"), dim2.toIndex(shape(), "transpose"))

    /**
     * Returns a tensor that is a transposed version of the input, swapping the two last dimensions.
     */
    public fun t(): Tensor {
        if (rank() < 2) {
            throw CandleException.UnexpectedNumberOfDims(2, rank(), shape())
        }
        return transpose(rank() - 2, rank() - 1)
    }

    /**
     * Returns a tensor with the same data where dimensions have been permuted.
     */
    public fun permute(idxs: List<Int>): Tensor {
        val newLayout = layout.permute(idxs)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun permute(vararg idxs: Int): Tensor = permute(idxs.toList())

    /**
     * Broadcast the input tensor to the target shape.
     */
    public fun broadcastAs(targetShape: Shape): Tensor {
        val newLayout = layout.broadcastAs(targetShape)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    /**
     * An alias for broadcastAs.
     */
    public fun expand(shape: Shape): Tensor = broadcastAs(shape)

    /**
     * Returns a new tensor duplicating data from the original tensor, with new dimensions on left.
     */
    public fun broadcastLeft(leftShape: Shape): Tensor {
        val newDims = leftShape.dims + dims()
        return broadcastAs(Shape(newDims))
    }

    /**
     * Returns a narrowed version of the input along dimension dim from start to start + len.
     */
    public fun narrow(dim: Int, start: Int, len: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "narrow")
        val newLayout = layout.narrow(normalizedDim, start, len)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    /**
     * Returns a tensor that is in row major order.
     */
    public fun contiguous(): Tensor {
        if (isContiguous() && layout.startOffset == 0) return this
        val newStorage = storage.tryClone(layout)
        return fromStorage(newStorage, shape(), BackpropOp.None, false)
    }

    /**
     * Returns a tensor that is in row major order, always copying underlying storage.
     */
    public fun forceContiguous(): Tensor = copy()

    /**
     * Copies the actual storage of the tensor.
     */
    public fun copy(): Tensor {
        val newStorage = storage.tryClone(layout)
        val op = BackpropOp.new1(this) { Op.Copy(it) }
        return fromStorage(newStorage, shape(), op, false)
    }

    /**
     * Create a variable based on the values currently stored in this tensor.
     */
    public fun makeVar(): Tensor {
        val newStorage = storage.tryClone(layout)
        return fromStorage(newStorage, shape(), BackpropOp.None, true)
    }

    /**
     * Casts the input tensor to the target dtype.
     */
    public fun toDtype(targetDType: DType): Tensor {
        if (dtype() == targetDType) return this
        val newStorage = storage.toDtype(layout, targetDType)
        return fromStorage(newStorage, shape(), BackpropOp.None, false)
    }

    /**
     * Transfers tensor to target device.
     */
    public fun toDevice(targetDevice: Device): Tensor {
        if (device().sameDevice(targetDevice)) return this
        if (targetDevice.isCpu()) {
            return fromStorage(Storage.Cpu(storage.toCpuStorage()), shape(), BackpropOp.None, false)
        }
        throw CandleException.NotCompiledWithCudaSupport
    }

    /**
     * Returns the sub-tensor fixing the index at i on the first dimension.
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
     * Returns the sub-tensor fixing the index at index on the dimension dim.
     */
    public fun getOnDim(dim: Int, index: Int): Tensor {
        val dimIdx = Shape.toIndex(dim, shape(), "get_on_dim")
        val dims = dims()
        val targetDims = dims.subList(0, dimIdx) + dims.subList(dimIdx + 1, dims.size)
        return narrow(dimIdx, index, 1).reshape(Shape(targetDims))
    }

    public fun getOnDim(dim: ShapeDim, index: Int): Tensor =
        getOnDim(dim.toIndex(shape(), "get_on_dim"), index)

    /**
     * Pad the input tensor using 0s along dimension dim.
     */
    public fun padWithZeros(dim: Int, left: Int, right: Int): Tensor {
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

    public fun padWithZeros(dim: ShapeDim, left: Int, right: Int): Tensor =
        padWithZeros(dim.toIndex(shape(), "pad_with_zeros"), left, right)

    /**
     * Pad the input tensor using same values along dimension dim.
     */
    public fun padWithSame(dim: Int, left: Int, right: Int): Tensor {
        if (left == 0 && right == 0) return this
        if (elemCount() == 0) bail("cannot use pad_with_same on an empty tensor")
        val dimIdx = Shape.toIndex(dim, shape(), "pad_with_same")
        val dimLen = dims()[dimIdx]
        return if (left == 0) {
            val r = narrow(dimIdx, dimLen - 1, 1)
            val list = ArrayList<Tensor>(1 + right)
            list.add(this)
            for (i in 0 until right) list.add(r)
            cat(list, dimIdx)
        } else if (right == 0) {
            val l = narrow(dimIdx, 0, 1)
            val list = ArrayList<Tensor>(left + 1)
            for (i in 0 until left) list.add(l)
            list.add(this)
            cat(list, dimIdx)
        } else {
            val l = narrow(dimIdx, 0, 1)
            val r = narrow(dimIdx, dimLen - 1, 1)
            val list = ArrayList<Tensor>(left + 1 + right)
            for (i in 0 until left) list.add(l)
            list.add(this)
            for (i in 0 until right) list.add(r)
            cat(list, dimIdx)
        }
    }

    public fun padWithSame(dim: ShapeDim, left: Int, right: Int): Tensor =
        padWithSame(dim.toIndex(shape(), "pad_with_same"), left, right)

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
     * Returns the cumulative sum of elements along the specified dimension.
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

    public fun cumsum(dim: ShapeDim): Tensor = cumsum(dim.toIndex(shape(), "cumsum"))

    /**
     * Returns a copy of self where values within ranges have been replaced with src.
     */
    public fun sliceAssign(ranges: List<IntRange>, src: Tensor): Tensor {
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
     * Returns log(sum(exp(tensor), dim)).
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
     * Pointwise pow operation: rhs * log(self).exp().
     */
    public fun pow(rhs: Tensor): Tensor = (rhs * log()).exp()

    /**
     * Broadcasting version of pow.
     */
    public fun broadcastPow(rhs: Tensor): Tensor = (rhs.broadcastMul(log())).exp()

    /**
     * Returns a new tensor with the order of elements reversed along the specified dimensions.
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
     * Returns all slices of size size from self in dimension dim stepped by step.
     */
    public fun unfold(dim: Int, size: Int, step: Int): Tensor {
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

    public fun unfold(dim: ShapeDim, size: Int, step: Int): Tensor =
        unfold(dim.toIndex(shape(), "unfold"), size, step)

    override fun toString(): String = "Tensor[${shape()}, ${dtype()}, ${device()}]"

    public companion object {
        /**
         * Creates a fresh tensor structure based on storage, shape, op, and variable flag.
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

        internal fun onesImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Tensor {
            val t = zerosImpl(shape, dtype, device, isVariable)
            t.storage.constSet(Scalar.one(dtype), t.layout)
            return t
        }

        /**
         * Creates a new tensor filled with ones.
         */
        public fun ones(shape: Shape, dtype: DType, device: Device): Tensor =
            onesImpl(shape, dtype, device, false)

        internal fun zerosImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            val count = shape.elemCount()
            val cpuStorage = when (dtype) {
                DType.U8 -> CpuStorage.U8(UByteArray(count))
                DType.U32 -> CpuStorage.U32(UIntArray(count))
                DType.I16 -> CpuStorage.I16(ShortArray(count))
                DType.I32 -> CpuStorage.I32(IntArray(count))
                DType.I64 -> CpuStorage.I64(LongArray(count))
                DType.BF16 -> CpuStorage.BF16(ShortArray(count))
                DType.F16 -> CpuStorage.F16(ShortArray(count))
                DType.F32 -> CpuStorage.F32(FloatArray(count))
                DType.F64 -> CpuStorage.F64(DoubleArray(count))
                DType.F8E4M3 -> CpuStorage.F8E4M3(ByteArray(count))
            }
            return fromStorage(Storage.Cpu(cpuStorage), shape, BackpropOp.None, isVariable)
        }

        /**
         * Creates a new tensor filled with zeros.
         */
        public fun zeros(shape: Shape, dtype: DType, device: Device): Tensor =
            zerosImpl(shape, dtype, device, false)

        internal fun emptyImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Tensor =
            zerosImpl(shape, dtype, device, isVariable)

        /**
         * Creates a new tensor filled with uninitialized memory.
         */
        public fun empty(shape: Shape, dtype: DType, device: Device): Tensor =
            emptyImpl(shape, dtype, device, false)

        internal fun randImpl(lo: Float, up: Float, shape: Shape, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            val count = shape.elemCount()
            val data = FloatArray(count)
            val diff = up - lo
            for (i in 0 until count) {
                data[i] = lo + Random.nextFloat() * diff
            }
            val storage = Storage.Cpu(CpuStorage.F32(data))
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        internal fun randF64Impl(lo: Double, up: Double, shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            val count = shape.elemCount()
            val data = DoubleArray(count)
            val diff = up - lo
            for (i in 0 until count) {
                data[i] = lo + Random.nextDouble() * diff
            }
            val t = fromStorage(Storage.Cpu(CpuStorage.F64(data)), shape, BackpropOp.None, isVariable)
            return if (dtype == DType.F64) t else t.toDtype(dtype)
        }

        /**
         * Creates a new tensor initialized with values sampled uniformly between lo and up.
         */
        public fun rand(min: Float, max: Float, shape: Shape, device: Device): Tensor =
            randImpl(min, max, shape, device, false)

        internal fun randnImpl(mean: Float, std: Float, shape: Shape, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            val count = shape.elemCount()
            val data = FloatArray(count)
            var i = 0
            while (i < count) {
                val u1 = Random.nextDouble().coerceAtLeast(1e-15)
                val u2 = Random.nextDouble()
                val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
                val z1 = sqrt(-2.0 * ln(u1)) * sin(2.0 * PI * u2)
                data[i++] = (mean + std * z0).toFloat()
                if (i < count) {
                    data[i++] = (mean + std * z1).toFloat()
                }
            }
            val storage = Storage.Cpu(CpuStorage.F32(data))
            return fromStorage(storage, shape, BackpropOp.None, isVariable)
        }

        internal fun randnF64Impl(mean: Double, std: Double, shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            val count = shape.elemCount()
            val data = DoubleArray(count)
            var i = 0
            while (i < count) {
                val u1 = Random.nextDouble().coerceAtLeast(1e-15)
                val u2 = Random.nextDouble()
                val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
                val z1 = sqrt(-2.0 * ln(u1)) * sin(2.0 * PI * u2)
                data[i++] = mean + std * z0
                if (i < count) {
                    data[i++] = mean + std * z1
                }
            }
            val t = fromStorage(Storage.Cpu(CpuStorage.F64(data)), shape, BackpropOp.None, isVariable)
            return if (dtype == DType.F64) t else t.toDtype(dtype)
        }

        /**
         * Creates a new tensor initialized with values sampled from normal distribution.
         */
        public fun randn(mean: Float, std: Float, shape: Shape, device: Device): Tensor =
            randnImpl(mean, std, shape, device, false)

        internal fun newImpl(data: FloatArray, shape: Shape, device: Device, isVariable: Boolean): Tensor {
            device.hashCode()
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.F32(data.copyOf())), shape, BackpropOp.None, isVariable)
        }

        /**
         * Creates a zero-filled tensor from shape with default F32 dtype and CPU device.
         */
        public fun from(shape: Shape): Tensor = zeros(shape, DType.F32, Device.Cpu)

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         */
        public fun new(data: FloatArray, shape: Shape, device: Device): Tensor =
            newImpl(data, shape, device, false)

        public fun new(data: DoubleArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.F64(data.copyOf())), shape, BackpropOp.None, false)
        }

        public fun new(data: IntArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.I32(data.copyOf())), shape, BackpropOp.None, false)
        }

        public fun new(data: LongArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.I64(data.copyOf())), shape, BackpropOp.None, false)
        }

        public fun new(data: UIntArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.U32(data.copyOf())), shape, BackpropOp.None, false)
        }

        public fun new(data: UByteArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.U8(data.copyOf())), shape, BackpropOp.None, false)
        }

        /**
         * Returns a new tensor with all elements having the same specified value.
         */
        public fun full(value: Double, shape: Shape, device: Device): Tensor {
            val t = zeros(shape, DType.F64, device)
            t.storage.constSet(Scalar.F64(value), t.layout)
            return t
        }

        /**
         * Creates a new 1D tensor from an iterator.
         */
        public fun fromIter(iter: Iterable<Float>, device: Device): Tensor {
            val list = iter.toList()
            return new(list.toFloatArray(), Shape(listOf(list.size)), device)
        }

        /**
         * Creates a new 1D tensor with values from interval [start, end) with difference 1.
         */
        public fun arange(start: Float, end: Float, device: Device): Tensor =
            arangeStep(start, end, 1.0f, device)

        public fun arange(start: Int, end: Int, device: Device): Tensor {
            val count = maxOf(0, end - start)
            val data = IntArray(count)
            for (i in 0 until count) {
                data[i] = start + i
            }
            return new(data, Shape(listOf(count)), device)
        }

        /**
         * Creates a new 1D tensor with values from interval [start, end) with difference step.
         */
        public fun arangeStep(start: Float, end: Float, step: Float, device: Device): Tensor {
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

        internal fun fromVecImpl(data: FloatArray, shape: Shape, device: Device, isVariable: Boolean): Tensor =
            newImpl(data, shape, device, isVariable)

        /**
         * Creates a new tensor initialized with values from input slice.
         */
        public fun fromSlice(data: FloatArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)
        public fun fromSlice(data: DoubleArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)
        public fun fromSlice(data: IntArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)
        public fun fromSlice(data: LongArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)
        public fun fromSlice(data: UIntArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)
        public fun fromSlice(data: UByteArray, shape: Shape, device: Device = Device.Cpu): Tensor = new(data, shape, device)

        /**
         * Creates a new tensor initialized with values from input list.
         */
        public fun fromVec(data: List<Float>, shape: Shape, device: Device): Tensor =
            new(data.toFloatArray(), shape, device)

        /**
         * Repeat this tensor along the specified dimensions.
         */
        public fun repeat(tensor: Tensor, shape: Shape): Tensor = tensor.repeat(shape)

        /**
         * Creates grids of coordinates specified by 1D inputs.
         */
        public fun meshgrid(args: List<Tensor>, xyIndexing: Boolean): List<Tensor> {
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
         */
        public fun stack(tensors: List<Tensor>, dim: Int): Tensor {
            if (tensors.isEmpty()) {
                throw CandleException.OpRequiresAtLeastOneTensor("stack")
            }
            val unsqueezed = tensors.map { it.unsqueeze(dim) }
            return cat(unsqueezed, dim)
        }

        public fun stack(tensors: List<Tensor>, dim: ShapeDim): Tensor =
            stack(tensors, dim.toIndexPlusOne(tensors[0].shape(), "stack"))

        /**
         * Returns a lower triangular matrix of ones of size n by n.
         */
        public fun tril2(n: Int, dtype: DType, device: Device): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.le(t2).toDtype(dtype)
        }

        /**
         * Returns an upper triangular matrix of ones of size n by n.
         */
        public fun triu2(n: Int, dtype: DType, device: Device): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.ge(t2).toDtype(dtype)
        }

        /**
         * Returns a matrix with a diagonal of ones of size n by n.
         */
        public fun eye(n: Int, dtype: DType, device: Device): Tensor {
            val t = arange(0, n, device)
            val t1 = t.reshape(Shape(listOf(1, n))).broadcastAs(Shape(listOf(n, n)))
            val t2 = t.reshape(Shape(listOf(n, 1))).broadcastAs(Shape(listOf(n, n)))
            return t1.eq(t2).toDtype(dtype)
        }

        private fun getReducedShape(shape: Shape, reduceDims: List<Int>, keepdim: Boolean): Shape {
            val srcDims = shape.dims
            val outDims = ArrayList<Int>()
            for (i in srcDims.indices) {
                if (i in reduceDims) {
                    if (keepdim) outDims.add(1)
                } else {
                    outDims.add(srcDims[i])
                }
            }
            return Shape(outDims)
        }
    }
}
