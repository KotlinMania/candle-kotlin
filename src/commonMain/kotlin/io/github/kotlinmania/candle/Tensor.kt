// port-lint: source candle-core/src/tensor.rs
package io.github.kotlinmania.candle

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

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

public class Tensor internal constructor(
    private val id: TensorId,
    private var storage: Storage,
    private val layout: Layout,
    private val op: BackpropOp,
    private val isVariable: Boolean,
) {
    public fun id(): TensorId = id

    public fun shape(): Shape = layout.shape

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

    public fun dim(d: Int): Int = layout.dim(d)

    public fun rank(): Int = layout.shape.rank

    public fun elemCount(): Int = layout.shape.elemCount()

    public fun dtype(): DType = storage.dtype()

    public fun device(): Device = storage.device()

    public fun layout(): Layout = layout

    public fun stride(): List<Int> = layout.stride

    public fun isContiguous(): Boolean = layout.isContiguous()

    public fun isVariable(): Boolean = isVariable

    public fun storage(): Storage = storage

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

    public fun affine(mul: Double, add: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.affine(layout, mul, add)
        return fromStorage(newStorage, layout.shape, BackpropOp.None, false)
    }

    public fun powf(e: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.powf(layout, e)
        return fromStorage(newStorage, layout.shape, BackpropOp.None, false)
    }

    public fun elu(alpha: Double): Tensor {
        if (elemCount() == 0) return this
        val newStorage = storage.elu(layout, alpha)
        return fromStorage(newStorage, layout.shape, BackpropOp.None, false)
    }

    // --- Binary Ops ---
    public fun add(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Add)

    public operator fun plus(rhs: Tensor): Tensor = add(rhs)

    public fun sub(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Sub)

    public operator fun minus(rhs: Tensor): Tensor = sub(rhs)

    public fun mul(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Mul)

    public operator fun times(rhs: Tensor): Tensor = mul(rhs)

    public operator fun div(rhs: Tensor): Tensor = binaryOp(rhs, BinaryOp.Div)

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

    // --- Comparisons ---
    public fun eq(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Eq)

    public fun ne(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Ne)

    public fun le(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Le)

    public fun ge(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Ge)

    public fun lt(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Lt)

    public fun gt(rhs: Tensor): Tensor = cmpOp(rhs, CmpOp.Gt)

    private fun cmpOp(rhs: Tensor, cOp: CmpOp): Tensor {
        val targetShape = shape().broadcastShapeBinaryOp(rhs.shape(), cOp.name)
        val lhsL = layout.broadcastAs(targetShape)
        val rhsL = rhs.layout.broadcastAs(targetShape)
        val newStorage = storage.cmp(cOp, rhs.storage, lhsL, rhsL)
        return fromStorage(newStorage, targetShape, BackpropOp.None, false)
    }

    // --- Matrix Ops ---
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

    public fun broadcastMatmul(rhs: Tensor): Tensor = matmul(rhs)

    // --- Reductions ---
    public fun sum(dims: List<Int>): Tensor {
        val normalizedDims = dims.map { Shape.toIndex(it, shape(), "sum") }.distinct().sorted()
        val newStorage = storage.reduceOp(ReduceOp.Sum, layout, normalizedDims)
        val outShape = getReducedShape(shape(), normalizedDims, false)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun sum(vararg dims: Int): Tensor = sum(dims.toList())

    public fun sumKeepdim(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "sum_keepdim")
        val newStorage = storage.reduceOp(ReduceOp.Sum, layout, listOf(normalizedDim))
        val outShape = getReducedShape(shape(), listOf(normalizedDim), true)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun sumAll(): Tensor = sum((0 until rank()).toList())

    public fun min(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "min")
        val newStorage = storage.reduceOp(ReduceOp.Min, layout, listOf(normalizedDim))
        val outShape = getReducedShape(shape(), listOf(normalizedDim), false)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun max(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "max")
        val newStorage = storage.reduceOp(ReduceOp.Max, layout, listOf(normalizedDim))
        val outShape = getReducedShape(shape(), listOf(normalizedDim), false)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun argmin(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "argmin")
        val newStorage = storage.reduceOp(ReduceOp.ArgMin, layout, listOf(normalizedDim))
        val outShape = getReducedShape(shape(), listOf(normalizedDim), false)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun argmax(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "argmax")
        val newStorage = storage.reduceOp(ReduceOp.ArgMax, layout, listOf(normalizedDim))
        val outShape = getReducedShape(shape(), listOf(normalizedDim), false)
        return fromStorage(newStorage, outShape, BackpropOp.None, false)
    }

    public fun mean(dim: Int): Tensor {
        val s = sum(dim)
        val dimSize = dims()[Shape.toIndex(dim, shape(), "mean")]
        return s.affine(1.0 / dimSize.toDouble(), 0.0)
    }

    public fun meanAll(): Tensor {
        val s = sumAll()
        return s.affine(1.0 / elemCount().toDouble(), 0.0)
    }

    // --- Indexing ---
    public fun indexSelect(indexes: Tensor, dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "index_select")
        val newStorage = storage.indexSelect(indexes.storage, layout, indexes.layout, normalizedDim)
        val outDims = dims().toMutableList()
        outDims[normalizedDim] = indexes.elemCount()
        return fromStorage(newStorage, Shape(outDims), BackpropOp.None, false)
    }

    public fun whereCond(onTrue: Tensor, onFalse: Tensor): Tensor {
        val targetShape =
            shape()
                .broadcastShapeBinaryOp(onTrue.shape(), "where_cond")
                .broadcastShapeBinaryOp(onFalse.shape(), "where_cond")
        val condL = layout.broadcastAs(targetShape)
        val trueL = onTrue.layout.broadcastAs(targetShape)
        val falseL = onFalse.layout.broadcastAs(targetShape)
        val newStorage = storage.whereCond(condL, onTrue.storage, trueL, onFalse.storage, falseL)
        return fromStorage(newStorage, targetShape, BackpropOp.None, false)
    }

    // --- Shape manipulations ---
    public fun reshape(shape: Shape): Tensor {
        if (shape.elemCount() != elemCount()) {
            throw CandleException.ShapeMismatch(elemCount(), shape)
        }
        val newLayout =
            if (isContiguous()) {
                Layout.contiguousWithOffset(shape, layout.startOffset)
            } else {
                val cont = contiguous()
                Layout.contiguous(shape)
            }
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun reshape(vararg dims: Int): Tensor =
        reshape(dims.toList().intoShapeWithHole(elemCount()))

    public fun flattenAll(): Tensor = reshape(Shape(listOf(elemCount())))

    public fun flattenFrom(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "flatten_from")
        val d = dims()
        val outer = d.subList(0, normalizedDim)
        val inner = d.subList(normalizedDim, d.size).fold(1) { a, b -> a * b }
        return reshape(Shape(outer + listOf(inner)))
    }

    public fun flattenTo(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "flatten_to")
        val d = dims()
        val outer = d.subList(0, normalizedDim + 1).fold(1) { a, b -> a * b }
        val inner = d.subList(normalizedDim + 1, d.size)
        return reshape(Shape(listOf(outer) + inner))
    }

    public fun squeeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "squeeze")
        if (dims()[normalizedDim] != 1) return this
        val newDims = dims().toMutableList()
        newDims.removeAt(normalizedDim)
        val newStride = stride().toMutableList()
        newStride.removeAt(normalizedDim)
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), BackpropOp.None, false)
    }

    public fun unsqueeze(dim: Int): Tensor {
        val normalizedDim = Shape.toIndexPlusOne(dim, shape(), "unsqueeze")
        val newDims = dims().toMutableList()
        newDims.add(normalizedDim, 1)
        val newStride = stride().toMutableList()
        val s = if (normalizedDim < stride().size) stride()[normalizedDim] else 1
        newStride.add(normalizedDim, s)
        return Tensor(TensorId.next(), storage, Layout(Shape(newDims), newStride, layout.startOffset), BackpropOp.None, false)
    }

    public fun transpose(dim1: Int, dim2: Int): Tensor {
        val newLayout = layout.transpose(dim1, dim2)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun t(): Tensor {
        if (rank() < 2) {
            throw CandleException.UnexpectedNumberOfDims(2, rank(), shape())
        }
        return transpose(rank() - 2, rank() - 1)
    }

    public fun permute(idxs: List<Int>): Tensor {
        val newLayout = layout.permute(idxs)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun permute(vararg idxs: Int): Tensor = permute(idxs.toList())

    public fun broadcastAs(targetShape: Shape): Tensor {
        val newLayout = layout.broadcastAs(targetShape)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun narrow(dim: Int, start: Int, len: Int): Tensor {
        val normalizedDim = Shape.toIndex(dim, shape(), "narrow")
        val newLayout = layout.narrow(normalizedDim, start, len)
        return Tensor(TensorId.next(), storage, newLayout, BackpropOp.None, false)
    }

    public fun contiguous(): Tensor {
        if (isContiguous() && layout.startOffset == 0) return this
        val newStorage = storage.tryClone(layout)
        return fromStorage(newStorage, shape(), BackpropOp.None, false)
    }

    // --- Conversions ---
    public fun toDtype(targetDType: DType): Tensor {
        if (dtype() == targetDType) return this
        val newStorage = storage.toDtype(layout, targetDType)
        return fromStorage(newStorage, shape(), BackpropOp.None, false)
    }

    public fun toDevice(targetDevice: Device): Tensor {
        if (device().sameDevice(targetDevice)) return this
        if (targetDevice.isCpu()) {
            return fromStorage(Storage.Cpu(storage.toCpuStorage()), shape(), BackpropOp.None, false)
        }
        throw CandleException.NotCompiledWithCudaSupport
    }

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

    override fun toString(): String = "Tensor[${shape()}, ${dtype()}, ${device()}]"

    public companion object {
        internal fun fromStorage(
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

        public fun zeros(shape: Shape, dtype: DType, device: Device): Tensor {
            val count = shape.elemCount()
            val cpuStorage =
                when (dtype) {
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
                    else -> bail("unsupported dtype for zeros $dtype")
                }
            return fromStorage(Storage.Cpu(cpuStorage), shape, BackpropOp.None, false)
        }

        public fun ones(shape: Shape, dtype: DType, device: Device): Tensor {
            val t = zeros(shape, dtype, device)
            t.storage.constSet(Scalar.one(dtype), t.layout)
            return t
        }

        public fun new(data: FloatArray, shape: Shape, device: Device): Tensor {
            if (data.size != shape.elemCount()) {
                throw CandleException.ShapeMismatch(data.size, shape)
            }
            return fromStorage(Storage.Cpu(CpuStorage.F32(data.copyOf())), shape, BackpropOp.None, false)
        }

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

        public fun fromVec(data: List<Float>, shape: Shape, device: Device): Tensor =
            new(data.toFloatArray(), shape, device)

        public fun arange(start: Float, end: Float, device: Device): Tensor {
            val step = 1.0f
            val count = maxOf(0, ceil((end - start) / step).toInt())
            val data = FloatArray(count)
            for (i in 0 until count) {
                data[i] = start + i * step
            }
            return new(data, Shape(listOf(count)), device)
        }

        public fun arange(start: Int, end: Int, device: Device): Tensor {
            val count = maxOf(0, end - start)
            val data = IntArray(count)
            for (i in 0 until count) {
                data[i] = start + i
            }
            return new(data, Shape(listOf(count)), device)
        }

        public fun rand(min: Float, max: Float, shape: Shape, device: Device): Tensor {
            val count = shape.elemCount()
            val data = FloatArray(count)
            val diff = max - min
            for (i in 0 until count) {
                data[i] = min + Random.nextFloat() * diff
            }
            return new(data, shape, device)
        }

        public fun randn(mean: Float, std: Float, shape: Shape, device: Device): Tensor {
            val count = shape.elemCount()
            val data = FloatArray(count)
            var i = 0
            while (i < count) {
                // Box-Muller transform
                val u1 = Random.nextDouble().coerceAtLeast(1e-15)
                val u2 = Random.nextDouble()
                val z0 = sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
                val z1 = sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.sin(2.0 * kotlin.math.PI * u2)
                data[i++] = (mean + std * z0).toFloat()
                if (i < count) {
                    data[i++] = (mean + std * z1).toFloat()
                }
            }
            return new(data, shape, device)
        }

        public fun cat(tensors: List<Tensor>, dim: Int): Tensor {
            if (tensors.isEmpty()) {
                throw CandleException.OpRequiresAtLeastOneTensor("cat")
            }
            val first = tensors[0]
            if (tensors.size == 1) return first
            val normalizedDim = Shape.toIndex(dim, first.shape(), "cat")

            val targetDType = first.dtype()
            val targetDevice = first.device()

            var totalDimSize = 0
            for ((idx, t) in tensors.withIndex()) {
                if (t.dtype() != targetDType) {
                    throw CandleException.DTypeMismatchBinaryOp(targetDType, t.dtype(), "cat")
                }
                if (t.rank() != first.rank()) {
                    throw CandleException.ShapeMismatchCat(normalizedDim, first.shape(), idx + 1, t.shape())
                }
                for (d in 0 until first.rank()) {
                    if (d != normalizedDim && t.dim(d) != first.dim(d)) {
                        throw CandleException.ShapeMismatchCat(normalizedDim, first.shape(), idx + 1, t.shape())
                    }
                }
                totalDimSize += t.dim(normalizedDim)
            }

            val outDims = first.dims().toMutableList()
            outDims[normalizedDim] = totalDimSize
            val outShape = Shape(outDims)
            val result = zeros(outShape, targetDType, targetDevice)

            var offset = 0
            for (t in tensors) {
                val len = t.dim(normalizedDim)
                val destNarrow = result.narrow(normalizedDim, offset, len)
                val cont = t.contiguous()
                val srcCpu = (cont.storage as Storage.Cpu).storage
                val dstCpu = (destNarrow.storage as Storage.Cpu).storage
                val destStrided = destNarrow.layout.stridedIndex()
                var srcIdx = 0
                while (destStrided.hasNext()) {
                    val dIdx = destStrided.next()
                    setStorageVal(dstCpu, dIdx, srcCpu, srcIdx++)
                }
                offset += len
            }
            return result
        }

        public fun stack(tensors: List<Tensor>, dim: Int): Tensor {
            if (tensors.isEmpty()) {
                throw CandleException.OpRequiresAtLeastOneTensor("stack")
            }
            val unsqueezed = tensors.map { it.unsqueeze(dim) }
            return cat(unsqueezed, dim)
        }

        private fun setStorageVal(dst: CpuStorage, dstIdx: Int, src: CpuStorage, srcIdx: Int) {
            when (dst) {
                is CpuStorage.F32 -> dst.data[dstIdx] = src.getAsFloat(srcIdx)
                is CpuStorage.F64 -> dst.data[dstIdx] = src.getAsDouble(srcIdx)
                is CpuStorage.I32 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt()
                is CpuStorage.I64 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toLong()
                is CpuStorage.U32 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toLong().toUInt()
                is CpuStorage.U8 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toUByte()
                is CpuStorage.I16 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toShort()
                is CpuStorage.F16 -> dst.data[dstIdx] = Float16Utils.floatToF16Bits(src.getAsFloat(srcIdx))
                is CpuStorage.BF16 -> dst.data[dstIdx] = Float16Utils.floatToBF16Bits(src.getAsFloat(srcIdx))
                is CpuStorage.F8E4M3 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toByte()
            }
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
