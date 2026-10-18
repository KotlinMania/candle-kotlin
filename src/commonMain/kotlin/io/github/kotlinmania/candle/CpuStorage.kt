package io.github.kotlinmania.candle

import io.github.kotlinmania.float8.F8E4M3
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.round
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh
import io.github.kotlinmania.float8.F8E4M3 as Float8

/**
 * Implementation of Backend Fns for CPU
 */

public const val useIm2colConv1d: Boolean = true
public const val useCol2imConv1dTr: Boolean = true

internal class VecList<T>(
    private val capacity: Int,
) : AbstractMutableList<T>() {
    private val buf: kotlin.collections.HashMap<Int, T> = kotlin.collections.HashMap(capacity)
    override var size: Int = 0
        private set

    override fun get(index: Int): T = buf.getValue(index)

    override fun set(
        index: Int,
        element: T,
    ): T {
        val prev = buf.put(index, element)
        if (index >= size) size = index + 1
        return prev ?: element
    }

    override fun add(
        index: Int,
        element: T,
    ) {
        buf[index] = element
        if (index >= size) size = index + 1
    }

    override fun removeAt(index: Int): T {
        val prev = buf.remove(index) ?: throw IndexOutOfBoundsException()
        if (size > 0) size -= 1
        return prev
    }

    fun spareCapacityMut(): MutableList<T> {
        size = capacity
        return this
    }

    fun setLen(len: Int) {
        while (buf.size > len) buf.remove(buf.size - 1)
        size = len
    }
}

// Maybe we should not implement Clone here and instead have an explicit allocator +
// intercept the oom errors to avoid panicking and provide a proper error.
public sealed class CpuStorage {
    public interface Device

    public interface Storage

    public data class U8(
        val data: UByteArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(UByteArray(r.size) { r[it].toInt().toUByte() })
    }

    public data class U32(
        val data: UIntArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(UIntArray(r.size) { r[it].toLong().toUInt() })
    }

    public data class I16(
        val data: ShortArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(ShortArray(r.size) { r[it].toInt().toShort() })
    }

    public data class I32(
        val data: IntArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(IntArray(r.size) { r[it].toInt() })
    }

    public data class I64(
        val data: LongArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(LongArray(r.size) { r[it].toLong() })
    }

    public data class BF16(
        val data: ShortArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(ShortArray(r.size) { Float16Utils.floatToBF16Bits(r[it].toFloat()) })
    }

    public data class F16(
        val data: ShortArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(ShortArray(r.size) { Float16Utils.floatToF16Bits(r[it].toFloat()) })
    }

    public data class F32(
        val data: FloatArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(FloatArray(r.size) { r[it].toFloat() })
    }

    public data class F64(
        val data: DoubleArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(DoubleArray(r.size) { r[it] })
    }

    public data class F8E4M3(
        val data: ByteArray,
    ) : CpuStorage() {
        public constructor(r: List<Double>) : this(ByteArray(r.size) { r[it].toInt().toByte() })
    }

    public fun applyInPlace(
        m: Map2InPlace,
        l1: Layout,
        v2: CpuStorage,
        l2: Layout,
    ) {
        val m1 = asSlice().toMutableList()
        m.f(m1, l1, v2.asSlice(), l2)
        when (this) {
            is U8 -> for (i in m1.indices) data[i] = m1[i].toInt().toUByte()
            is U32 -> for (i in m1.indices) data[i] = m1[i].toLong().toUInt()
            is I16 -> for (i in m1.indices) data[i] = m1[i].toInt().toShort()
            is I32 -> for (i in m1.indices) data[i] = m1[i].toInt()
            is I64 -> for (i in m1.indices) data[i] = m1[i].toLong()
            is BF16 -> for (i in m1.indices) data[i] = Float16Utils.floatToBF16Bits(m1[i].toFloat())
            is F16 -> for (i in m1.indices) data[i] = Float16Utils.floatToF16Bits(m1[i].toFloat())
            is F32 -> for (i in m1.indices) data[i] = m1[i].toFloat()
            is F64 -> for (i in m1.indices) data[i] = m1[i]
            is F8E4M3 -> for (i in m1.indices) data[i] = m1[i].toInt().toByte()
            is F6E2M3, is F6E3M2, is F4, is F8E8M0 -> Unit
        }
    }

    // Dummy types that store raw bytes
    public data class F6E2M3(
        val data: ByteArray,
    ) : CpuStorage()

    public data class F6E3M2(
        val data: ByteArray,
    ) : CpuStorage()

    public data class F4(
        val data: ByteArray,
    ) : CpuStorage()

    public data class F8E8M0(
        val data: ByteArray,
    ) : CpuStorage()

    public fun asSlice(): List<Double> {
        val len =
            when (this) {
                is U8 -> data.size
                is U32 -> data.size
                is I16 -> data.size
                is I32 -> data.size
                is I64 -> data.size
                is BF16 -> data.size
                is F16 -> data.size
                is F32 -> data.size
                is F64 -> data.size
                is F8E4M3 -> data.size
                is F6E2M3 -> data.size
                is F6E3M2 -> data.size
                is F4 -> data.size
                is F8E8M0 -> data.size
            }
        return List(len) { getAsDouble(it) }
    }

    public fun dtype(): DType =
        when (this) {
            is U8 -> DType.U8
            is U32 -> DType.U32
            is I16 -> DType.I16
            is I32 -> DType.I32
            is I64 -> DType.I64
            is BF16 -> DType.BF16
            is F16 -> DType.F16
            is F32 -> DType.F32
            is F64 -> DType.F64
            is F8E4M3 -> DType.F8E4M3
            is F6E2M3 -> DType.F6E2M3
            is F6E3M2 -> DType.F6E3M2
            is F4 -> DType.F4
            is F8E8M0 -> DType.F8E8M0
        }

    public fun device(): CpuDevice = CpuDevice

    public fun toCpuStorage(): CpuStorage = this

    public fun tryClone(layout: Layout): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (this) {
            is U8 -> {
                val dst = UByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                U8(dst)
            }

            is U32 -> {
                val dst = UIntArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                U32(dst)
            }

            is I16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                I16(dst)
            }

            is I32 -> {
                val dst = IntArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                I32(dst)
            }

            is I64 -> {
                val dst = LongArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                I64(dst)
            }

            is BF16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                BF16(dst)
            }

            is F16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F16(dst)
            }

            is F32 -> {
                val dst = FloatArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F32(dst)
            }

            is F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F64(dst)
            }

            is F8E4M3 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F8E4M3(dst)
            }

            is F6E2M3 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F6E2M3(dst)
            }

            is F6E3M2 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F6E3M2(dst)
            }

            is F4 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F4(dst)
            }

            is F8E8M0 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx]
                F8E8M0(dst)
            }
        }
    }

    public fun constSet(
        v: Scalar,
        layout: Layout,
    ) {
        fun <T> set(
            src: MutableList<T>,
            l: Layout,
            s: T,
        ) {
            for (idx in l.stridedIndex()) {
                if (idx < src.size) src[idx] = s
            }
        }
        val strided = layout.stridedIndex()
        when (this) {
            is U8 -> {
                val value = (v as? Scalar.U8)?.value ?: (v.toF64().toInt().toUByte())
                for (idx in strided) data[idx] = value
            }

            is U32 -> {
                val value = (v as? Scalar.U32)?.value ?: (v.toF64().toLong().toUInt())
                for (idx in strided) data[idx] = value
            }

            is I16 -> {
                val value = (v as? Scalar.I16)?.value ?: (v.toF64().toInt().toShort())
                for (idx in strided) data[idx] = value
            }

            is I32 -> {
                val value = (v as? Scalar.I32)?.value ?: (v.toF64().toInt())
                for (idx in strided) data[idx] = value
            }

            is I64 -> {
                val value = (v as? Scalar.I64)?.value ?: (v.toF64().toLong())
                for (idx in strided) data[idx] = value
            }

            is BF16 -> {
                val value = (v as? Scalar.BF16)?.value ?: Float16Utils.floatToBF16Bits(v.toF64().toFloat())
                for (idx in strided) data[idx] = value
            }

            is F16 -> {
                val value = (v as? Scalar.F16)?.value ?: Float16Utils.floatToF16Bits(v.toF64().toFloat())
                for (idx in strided) data[idx] = value
            }

            is F32 -> {
                val value = (v as? Scalar.F32)?.value ?: (v.toF64().toFloat())
                for (idx in strided) data[idx] = value
            }

            is F64 -> {
                val value = v.toF64()
                for (idx in strided) data[idx] = value
            }

            is F8E4M3 -> {
                val value = (v as? Scalar.F8E4M3)?.value ?: 0.toByte()
                for (idx in strided) data[idx] = value
            }

            is F6E2M3 -> {
                bail("const_set not supported for dummy type F6E2M3")
            }

            is F6E3M2 -> {
                bail("const_set not supported for dummy type F6E3M2")
            }

            is F4 -> {
                bail("const_set not supported for dummy type F4")
            }

            is F8E8M0 -> {
                bail("const_set not supported for dummy type F8E8M0")
            }
        }
    }

    public fun affine(
        layout: Layout,
        mul: Double,
        add: Double,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (this) {
            is F32 -> {
                val dst = FloatArray(count)
                val m = mul.toFloat()
                val a = add.toFloat()
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx] * m + a
                F32(dst)
            }

            is F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx] * mul + add
                F64(dst)
            }

            is I32 -> {
                val dst = IntArray(count)
                val m = mul.toInt()
                val a = add.toInt()
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx] * m + a
                I32(dst)
            }

            is I64 -> {
                val dst = LongArray(count)
                val m = mul.toLong()
                val a = add.toLong()
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx] * m + a
                I64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                val m = mul.toFloat()
                val a = add.toFloat()
                var idx = 0
                for (srcIdx in strided) {
                    val v = getAsFloat(srcIdx)
                    dst[idx++] = v * m + a
                }
                F32(dst).toDType(Layout.contiguous(layout.shape), dtype())
            }
        }
    }

    public fun powf(
        layout: Layout,
        e: Double,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (this) {
            is F32 -> {
                val dst = FloatArray(count)
                val expF = e.toFloat()
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx].pow(expF)
                F32(dst)
            }

            is F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = data[srcIdx].pow(e)
                F64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                val expF = e.toFloat()
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsFloat(srcIdx).pow(expF)
                F32(dst).toDType(Layout.contiguous(layout.shape), dtype())
            }
        }
    }

    public fun elu(
        layout: Layout,
        alpha: Double,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (this) {
            is F32 -> {
                val dst = FloatArray(count)
                val a = alpha.toFloat()
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] = if (v > 0f) v else (a * (exp(v) - 1f))
                }
                F32(dst)
            }

            is F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] = if (v > 0.0) v else (alpha * (exp(v) - 1.0))
                }
                F64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                val a = alpha.toFloat()
                var idx = 0
                for (srcIdx in strided) {
                    val v = getAsFloat(srcIdx)
                    dst[idx++] = if (v > 0f) v else (a * (exp(v) - 1f))
                }
                F32(dst).toDType(Layout.contiguous(layout.shape), dtype())
            }
        }
    }

    public fun unaryImpl(
        layout: Layout,
        op: UnaryOp,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (this) {
            is F32 -> {
                val dst = FloatArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] = applyUnaryFloat(v, op)
                }
                F32(dst)
            }

            is F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] = applyUnaryDouble(v, op)
                }
                F64(dst)
            }

            is I32 -> {
                val dst = IntArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] =
                        when (op) {
                            UnaryOp.Abs -> abs(v)
                            UnaryOp.Neg -> -v
                            UnaryOp.Sqr -> v * v
                            UnaryOp.Sign -> sign(v.toFloat()).toInt()
                            UnaryOp.Relu -> if (v > 0) v else 0
                            else -> applyUnaryFloat(v.toFloat(), op).toInt()
                        }
                }
                I32(dst)
            }

            is I64 -> {
                val dst = LongArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = data[srcIdx]
                    dst[idx++] =
                        when (op) {
                            UnaryOp.Abs -> abs(v)
                            UnaryOp.Neg -> -v
                            UnaryOp.Sqr -> v * v
                            UnaryOp.Sign -> sign(v.toFloat()).toLong()
                            UnaryOp.Relu -> if (v > 0L) v else 0L
                            else -> applyUnaryDouble(v.toDouble(), op).toLong()
                        }
                }
                I64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                var idx = 0
                for (srcIdx in strided) {
                    val v = getAsFloat(srcIdx)
                    dst[idx++] = applyUnaryFloat(v, op)
                }
                F32(dst).toDType(Layout.contiguous(layout.shape), dtype())
            }
        }
    }

    public fun binaryImpl(
        rhs: CpuStorage,
        lhsLayout: Layout,
        rhsLayout: Layout,
        op: BinaryOp,
    ): CpuStorage {
        val count = lhsLayout.shape.elemCount()
        val lhsStrided = lhsLayout.stridedIndex()
        val rhsStrided = rhsLayout.stridedIndex()
        return when (this) {
            is F32 -> {
                val rhsData = (rhs as F32).data
                val dst = FloatArray(count)
                var idx = 0
                while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
                    val l = data[lhsStrided.next()]
                    val r = rhsData[rhsStrided.next()]
                    dst[idx++] =
                        when (op) {
                            BinaryOp.Add -> l + r
                            BinaryOp.Sub -> l - r
                            BinaryOp.Mul -> l * r
                            BinaryOp.Div -> l / r
                            BinaryOp.Maximum -> maxOf(l, r)
                            BinaryOp.Minimum -> minOf(l, r)
                        }
                }
                F32(dst)
            }

            is F64 -> {
                val rhsData = (rhs as F64).data
                val dst = DoubleArray(count)
                var idx = 0
                while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
                    val l = data[lhsStrided.next()]
                    val r = rhsData[rhsStrided.next()]
                    dst[idx++] =
                        when (op) {
                            BinaryOp.Add -> l + r
                            BinaryOp.Sub -> l - r
                            BinaryOp.Mul -> l * r
                            BinaryOp.Div -> l / r
                            BinaryOp.Maximum -> maxOf(l, r)
                            BinaryOp.Minimum -> minOf(l, r)
                        }
                }
                F64(dst)
            }

            is I32 -> {
                val rhsData = (rhs as I32).data
                val dst = IntArray(count)
                var idx = 0
                while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
                    val l = data[lhsStrided.next()]
                    val r = rhsData[rhsStrided.next()]
                    dst[idx++] =
                        when (op) {
                            BinaryOp.Add -> l + r
                            BinaryOp.Sub -> l - r
                            BinaryOp.Mul -> l * r
                            BinaryOp.Div -> l / r
                            BinaryOp.Maximum -> maxOf(l, r)
                            BinaryOp.Minimum -> minOf(l, r)
                        }
                }
                I32(dst)
            }

            is I64 -> {
                val rhsData = (rhs as I64).data
                val dst = LongArray(count)
                var idx = 0
                while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
                    val l = data[lhsStrided.next()]
                    val r = rhsData[rhsStrided.next()]
                    dst[idx++] =
                        when (op) {
                            BinaryOp.Add -> l + r
                            BinaryOp.Sub -> l - r
                            BinaryOp.Mul -> l * r
                            BinaryOp.Div -> l / r
                            BinaryOp.Maximum -> maxOf(l, r)
                            BinaryOp.Minimum -> minOf(l, r)
                        }
                }
                I64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                var idx = 0
                while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
                    val l = getAsFloat(lhsStrided.next())
                    val r = rhs.getAsFloat(rhsStrided.next())
                    dst[idx++] =
                        when (op) {
                            BinaryOp.Add -> l + r
                            BinaryOp.Sub -> l - r
                            BinaryOp.Mul -> l * r
                            BinaryOp.Div -> l / r
                            BinaryOp.Maximum -> maxOf(l, r)
                            BinaryOp.Minimum -> minOf(l, r)
                        }
                }
                F32(dst).toDType(Layout.contiguous(lhsLayout.shape), dtype())
            }
        }
    }

    public fun cmp(
        op: CmpOp,
        rhs: CpuStorage,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): CpuStorage {
        val count = lhsLayout.shape.elemCount()
        val lhsStrided = lhsLayout.stridedIndex()
        val rhsStrided = rhsLayout.stridedIndex()
        val dst = UByteArray(count)
        var idx = 0
        while (lhsStrided.hasNext() && rhsStrided.hasNext()) {
            val l = getAsDouble(lhsStrided.next())
            val r = rhs.getAsDouble(rhsStrided.next())
            val cond =
                when (op) {
                    CmpOp.Eq -> l == r
                    CmpOp.Ne -> l != r
                    CmpOp.Le -> l <= r
                    CmpOp.Ge -> l >= r
                    CmpOp.Lt -> l < r
                    CmpOp.Gt -> l > r
                }
            dst[idx++] = if (cond) 1u else 0u
        }
        return U8(dst)
    }

    public fun whereCond(
        layout: Layout,
        onTrue: CpuStorage,
        trueLayout: Layout,
        onFalse: CpuStorage,
        falseLayout: Layout,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val condStrided = layout.stridedIndex()
        val trueStrided = trueLayout.stridedIndex()
        val falseStrided = falseLayout.stridedIndex()

        val condData = (this as U8).data
        val targetDType = onTrue.dtype()
        return when (targetDType) {
            DType.F32 -> {
                val tData = (onTrue as F32).data
                val fData = (onFalse as F32).data
                val dst = FloatArray(count)
                var idx = 0
                while (condStrided.hasNext()) {
                    val cond = condData[condStrided.next()] != 0u.toUByte()
                    val tVal = tData[trueStrided.next()]
                    val fVal = fData[falseStrided.next()]
                    dst[idx++] = if (cond) tVal else fVal
                }
                F32(dst)
            }

            DType.F64 -> {
                val tData = (onTrue as F64).data
                val fData = (onFalse as F64).data
                val dst = DoubleArray(count)
                var idx = 0
                while (condStrided.hasNext()) {
                    val cond = condData[condStrided.next()] != 0u.toUByte()
                    val tVal = tData[trueStrided.next()]
                    val fVal = fData[falseStrided.next()]
                    dst[idx++] = if (cond) tVal else fVal
                }
                F64(dst)
            }

            DType.I32 -> {
                val tData = (onTrue as I32).data
                val fData = (onFalse as I32).data
                val dst = IntArray(count)
                var idx = 0
                while (condStrided.hasNext()) {
                    val cond = condData[condStrided.next()] != 0u.toUByte()
                    val tVal = tData[trueStrided.next()]
                    val fVal = fData[falseStrided.next()]
                    dst[idx++] = if (cond) tVal else fVal
                }
                I32(dst)
            }

            DType.I64 -> {
                val tData = (onTrue as I64).data
                val fData = (onFalse as I64).data
                val dst = LongArray(count)
                var idx = 0
                while (condStrided.hasNext()) {
                    val cond = condData[condStrided.next()] != 0u.toUByte()
                    val tVal = tData[trueStrided.next()]
                    val fVal = fData[falseStrided.next()]
                    dst[idx++] = if (cond) tVal else fVal
                }
                I64(dst)
            }

            else -> {
                val dst = FloatArray(count)
                var idx = 0
                while (condStrided.hasNext()) {
                    val cond = condData[condStrided.next()] != 0u.toUByte()
                    val tVal = onTrue.getAsFloat(trueStrided.next())
                    val fVal = onFalse.getAsFloat(falseStrided.next())
                    dst[idx++] = if (cond) tVal else fVal
                }
                F32(dst).toDType(Layout.contiguous(layout.shape), targetDType)
            }
        }
    }

    public fun toDType(
        layout: Layout,
        target: DType,
    ): CpuStorage {
        val count = layout.shape.elemCount()
        val strided = layout.stridedIndex()
        return when (target) {
            DType.U8 -> {
                val dst = UByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toInt().toUByte()
                U8(dst)
            }

            DType.U32 -> {
                val dst = UIntArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toLong().toUInt()
                U32(dst)
            }

            DType.I16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toInt().toShort()
                I16(dst)
            }

            DType.I32 -> {
                val dst = IntArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toInt()
                I32(dst)
            }

            DType.I64 -> {
                val dst = LongArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toLong()
                I64(dst)
            }

            DType.BF16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = Float16Utils.floatToBF16Bits(getAsFloat(srcIdx))
                BF16(dst)
            }

            DType.F16 -> {
                val dst = ShortArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = Float16Utils.floatToF16Bits(getAsFloat(srcIdx))
                F16(dst)
            }

            DType.F32 -> {
                val dst = FloatArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsFloat(srcIdx)
                F32(dst)
            }

            DType.F64 -> {
                val dst = DoubleArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx)
                F64(dst)
            }

            DType.F8E4M3 -> {
                val dst = ByteArray(count)
                var idx = 0
                for (srcIdx in strided) dst[idx++] = getAsDouble(srcIdx).toInt().toByte()
                F8E4M3(dst)
            }

            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                bail("unsupported dtype $target")
            }
        }
    }

    public fun matmul(
        rhs: CpuStorage,
        bmnk: List<Int>,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): CpuStorage {
        val b = bmnk[0]
        val m = bmnk[1]
        val n = bmnk[2]
        val k = bmnk[3]

        val count = b * m * n
        val outShape = if (b > 1 || lhsLayout.shape.rank > 2) Shape(listOf(b, m, n)) else Shape(listOf(m, n))

        val lhsDims = lhsLayout.dims
        val rhsDims = rhsLayout.dims
        val lhsStride = lhsLayout.stride
        val rhsStride = rhsLayout.stride

        val lhsBatchStride = if (lhsDims.size > 2) lhsStride[lhsDims.size - 3] else 0
        val rhsBatchStride = if (rhsDims.size > 2) rhsStride[rhsDims.size - 3] else 0

        val lhsMStride = lhsStride[lhsDims.size - 2]
        val lhsKStride = lhsStride[lhsDims.size - 1]

        val rhsKStride = rhsStride[rhsDims.size - 2]
        val rhsNStride = rhsStride[rhsDims.size - 1]

        val dst = FloatArray(count)
        var dstIdx = 0

        for (bIdx in 0 until b) {
            val lhsBOffset = lhsLayout.startOffset + bIdx * lhsBatchStride
            val rhsBOffset = rhsLayout.startOffset + bIdx * rhsBatchStride
            for (mIdx in 0 until m) {
                val lhsRowOffset = lhsBOffset + mIdx * lhsMStride
                for (nIdx in 0 until n) {
                    val rhsColOffset = rhsBOffset + nIdx * rhsNStride
                    var sum = 0.0f
                    for (kIdx in 0 until k) {
                        val lVal = getAsFloat(lhsRowOffset + kIdx * lhsKStride)
                        val rVal = rhs.getAsFloat(rhsColOffset + kIdx * rhsKStride)
                        sum += lVal * rVal
                    }
                    dst[dstIdx++] = sum
                }
            }
        }
        return F32(dst).toDType(Layout.contiguous(outShape), dtype())
    }

    public fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        reduceDims: List<Int>,
    ): CpuStorage {
        for (dim in reduceDims) {
            if (dim !in layout.dims.indices) throw CandleException.DimOutOfRange(layout.shape, dim, "reduce")
        }
        if (reduceDims.distinct().size != reduceDims.size) {
            throw CandleException.DuplicateDimIndex(layout.shape, reduceDims, "reduce")
        }
        if (op != ReduceOp.Sum && reduceDims.size != 1) {
            throw CandleException.OnlySingleDimension(op.name.lowercase(), reduceDims)
        }
        if (op != ReduceOp.Sum && layout.shape.elemCount() == 0) throw CandleException.EmptyTensor("reduce")
        return when (this) {
            is U8 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0.toUByte(), { a, b -> (a + b).toUByte() }, { a, b -> a < b }) {
                    U8(it.toUByteArray())
                }
            }

            is U32 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0u, { a, b -> a + b }, { a, b -> a < b }) { U32(it.toUIntArray()) }
            }

            is I16 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0.toShort(), { a, b -> (a + b).toShort() }, { a, b -> a < b }) {
                    I16(it.toShortArray())
                }
            }

            is I32 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0, { a, b -> a + b }, { a, b -> a < b }) { I32(it.toIntArray()) }
            }

            is I64 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0L, { a, b -> a + b }, { a, b -> a < b }) { I64(it.toLongArray()) }
            }

            is F32 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0f, { a, b -> a + b }, { a, b -> a < b }) { F32(it.toFloatArray()) }
            }

            is F64 -> {
                reduceTyped(op, layout, reduceDims, data.asList(), 0.0, { a, b -> a + b }, { a, b -> a < b }) { F64(it.toDoubleArray()) }
            }

            is BF16 -> {
                reduceTyped(
                    op,
                    layout,
                    reduceDims,
                    data.asList(),
                    0.toShort(),
                    { a, b -> Float16Utils.floatToBF16Bits(Float16Utils.bf16BitsToFloat(a) + Float16Utils.bf16BitsToFloat(b)) },
                    { a, b -> Float16Utils.bf16BitsToFloat(a) < Float16Utils.bf16BitsToFloat(b) },
                ) { BF16(it.toShortArray()) }
            }

            is F16 -> {
                reduceTyped(
                    op,
                    layout,
                    reduceDims,
                    data.asList(),
                    0.toShort(),
                    { a, b -> Float16Utils.floatToF16Bits(Float16Utils.f16BitsToFloat(a) + Float16Utils.f16BitsToFloat(b)) },
                    { a, b -> Float16Utils.f16BitsToFloat(a) < Float16Utils.f16BitsToFloat(b) },
                ) { F16(it.toShortArray()) }
            }

            is F8E4M3 -> {
                reduceTyped(
                    op,
                    layout,
                    reduceDims,
                    data.asList(),
                    0.toByte(),
                    { a, b -> (Float8.fromBits(a.toUByte()) + Float8.fromBits(b.toUByte())).bits },
                    { a, b -> Float8.fromBits(a.toUByte()).toF32() < Float8.fromBits(b.toUByte()).toF32() },
                ) { F8E4M3(it.toByteArray()) }
            }

            is F6E2M3, is F6E3M2, is F4, is F8E8M0 -> {
                throw CandleException.UnsupportedDTypeForOp(dtype(), "reduce")
            }
        }
    }

    private fun <T> reduceTyped(
        op: ReduceOp,
        layout: Layout,
        reduceDims: List<Int>,
        values: List<T>,
        zero: T,
        add: (T, T) -> T,
        less: (T, T) -> Boolean,
        wrap: (List<T>) -> CpuStorage,
    ): CpuStorage {
        if (op == ReduceOp.Sum) {
            val dims = reduceDims.sorted()
            val dstShape = Shape(layout.dims.mapIndexed { i, size -> if (i in dims) 1 else size })
            val dimsAndStride = dims.map { dim -> layout.dims[dim] to layout.dims.drop(dim + 1).fold(1, Int::times) }
            return wrap(ReduceSum(dstShape, dims.toIntArray(), dimsAndStride).foldTyped(values, layout, zero, add))
        }
        val useMin = op == ReduceOp.Min || op == ReduceOp.ArgMin
        val returnIndex = op == ReduceOp.ArgMin || op == ReduceOp.ArgMax
        val reducer = ReduceIndex(reduceDims.single(), useMin, returnIndex)
        val replace: (T, T) -> Boolean = { best, value -> if (useMin) less(value, best) else less(best, value) }
        return if (returnIndex) {
            U32(reducer.foldTyped(values, layout, replace) { _, index -> index.toUInt() }.toUIntArray())
        } else {
            wrap(reducer.foldTyped(values, layout, replace) { value, _ -> value })
        }
    }

    public fun indexSelect(
        index: CpuStorage,
        layout: Layout,
        indexLayout: Layout,
        dim: Int,
    ): CpuStorage {
        val srcShape = layout.shape
        val srcDims = srcShape.dims
        val indexCount = indexLayout.shape.elemCount()
        val outDims = srcDims.toMutableList()
        outDims[dim] = indexCount
        val outShape = Shape(outDims)
        val outCount = outShape.elemCount()

        val dst = FloatArray(outCount)
        val indexValues = IntArray(indexCount)
        val indexStrided = indexLayout.stridedIndex()
        var iIdx = 0
        while (indexStrided.hasNext()) {
            indexValues[iIdx++] = index.getAsDouble(indexStrided.next()).toInt()
        }

        val dimStride = layout.stride[dim]
        val outerCount = srcDims.subList(0, dim).fold(1) { a, b -> a * b }
        val innerCount = srcDims.subList(dim + 1, srcDims.size).fold(1) { a, b -> a * b }

        var dstPos = 0
        for (o in 0 until outerCount) {
            val outerOffset = layout.startOffset + o * (srcDims[dim] * innerCount)
            for (idx in 0 until indexCount) {
                val selectedDimIdx = indexValues[idx]
                if (selectedDimIdx < 0 || selectedDimIdx >= srcDims[dim]) {
                    throw CandleException.InvalidIndex("index_select", selectedDimIdx, srcDims[dim])
                }
                val selectedOffset = outerOffset + selectedDimIdx * dimStride
                for (inn in 0 until innerCount) {
                    dst[dstPos++] = getAsFloat(selectedOffset + inn)
                }
            }
        }
        return F32(dst).toDType(Layout.contiguous(outShape), dtype())
    }

    public fun gather(
        ids: CpuStorage,
        srcL: Layout,
        idsL: Layout,
        dim: Int,
    ): CpuStorage {
        val idsOffsets = idsL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("gather")
        val srcOffsets = srcL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("gather")
        val idsDims = idsL.shape.dims
        val srcDims = srcL.shape.dims
        val dstLen = idsDims.fold(1) { a, b -> a * b }
        var dstLeftLen = 1
        for (i in 0 until dim) dstLeftLen *= idsDims[i]
        val dstDimLen = idsDims[dim]
        var dstRightLen = 1
        for (i in (dim + 1) until idsDims.size) dstRightLen *= idsDims[i]

        val srcDimLen = srcDims[dim]
        var srcRightLen = 1
        for (i in (dim + 1) until srcDims.size) srcRightLen *= srcDims[i]

        val dst = FloatArray(dstLen)
        for (leftI in 0 until dstLeftLen) {
            val startSrcIdx = srcOffsets.start + leftI * srcRightLen * srcDimLen
            val startDstIdx = leftI * dstRightLen * dstDimLen
            for (i in 0 until dstDimLen) {
                val curDstIdx = startDstIdx + i * dstRightLen
                for (rightI in 0 until dstRightLen) {
                    val dstIdx = curDstIdx + rightI
                    val idIdx = idsOffsets.start + dstIdx
                    val indexVal = ids.getAsDouble(idIdx).toLong()
                    if (indexVal < 0 || indexVal >= srcDimLen) {
                        throw CandleException.InvalidIndex("gather", indexVal.toInt(), srcDimLen)
                    }
                    val srcIdx = startSrcIdx + indexVal.toInt() * srcRightLen + rightI
                    dst[dstIdx] = getAsFloat(srcIdx)
                }
            }
        }
        return F32(dst).toDType(Layout.contiguous(idsL.shape), dtype())
    }

    public fun getAsFloat(index: Int): Float =
        when (this) {
            is U8 -> data[index].toFloat()
            is U32 -> data[index].toFloat()
            is I16 -> data[index].toFloat()
            is I32 -> data[index].toFloat()
            is I64 -> data[index].toFloat()
            is BF16 -> Float16Utils.bf16BitsToFloat(data[index])
            is F16 -> Float16Utils.f16BitsToFloat(data[index])
            is F32 -> data[index]
            is F64 -> data[index].toFloat()
            is F8E4M3 -> data[index].toFloat()
            is F6E2M3 -> data[index].toFloat()
            is F6E3M2 -> data[index].toFloat()
            is F4 -> data[index].toFloat()
            is F8E8M0 -> data[index].toFloat()
        }

    public fun getAsDouble(index: Int): Double =
        when (this) {
            is U8 -> data[index].toDouble()
            is U32 -> data[index].toDouble()
            is I16 -> data[index].toDouble()
            is I32 -> data[index].toDouble()
            is I64 -> data[index].toDouble()
            is BF16 -> Float16Utils.bf16BitsToFloat(data[index]).toDouble()
            is F16 -> Float16Utils.f16BitsToFloat(data[index]).toDouble()
            is F32 -> data[index].toDouble()
            is F64 -> data[index]
            is F8E4M3 -> data[index].toDouble()
            is F6E2M3 -> data[index].toDouble()
            is F6E3M2 -> data[index].toDouble()
            is F4 -> data[index].toDouble()
            is F8E8M0 -> data[index].toDouble()
        }

    public fun getAsLong(index: Int): Long =
        when (this) {
            is U8 -> data[index].toLong() and 0xFFL
            is U32 -> data[index].toLong() and 0xFFFFFFFFL
            is I16 -> data[index].toLong()
            is I32 -> data[index].toLong()
            is I64 -> data[index]
            is BF16 -> Float16Utils.bf16BitsToFloat(data[index]).toLong()
            is F16 -> Float16Utils.f16BitsToFloat(data[index]).toLong()
            is F32 -> data[index].toLong()
            is F64 -> data[index].toLong()
            is F8E4M3 -> data[index].toLong()
            is F6E2M3 -> data[index].toLong()
            is F6E3M2 -> data[index].toLong()
            is F4 -> data[index].toLong()
            is F8E8M0 -> data[index].toLong()
        }

    public fun setDouble(
        index: Int,
        value: Double,
    ) {
        when (this) {
            is U8 -> data[index] = value.toInt().toUByte()
            is U32 -> data[index] = value.toLong().toUInt()
            is I16 -> data[index] = value.toInt().toShort()
            is I32 -> data[index] = value.toInt()
            is I64 -> data[index] = value.toLong()
            is BF16 -> data[index] = Float16Utils.floatToBF16Bits(value.toFloat())
            is F16 -> data[index] = Float16Utils.floatToF16Bits(value.toFloat())
            is F32 -> data[index] = value.toFloat()
            is F64 -> data[index] = value
            is F8E4M3 -> data[index] = value.toInt().toByte()
            is F6E2M3 -> data[index] = value.toInt().toByte()
            is F6E3M2 -> data[index] = value.toInt().toByte()
            is F4 -> data[index] = value.toInt().toByte()
            is F8E8M0 -> data[index] = value.toInt().toByte()
        }
    }

    private fun applyUnaryFloat(
        v: Float,
        op: UnaryOp,
    ): Float =
        when (op) {
            UnaryOp.Exp -> exp(v)
            UnaryOp.Log -> ln(v)
            UnaryOp.Sin -> sin(v)
            UnaryOp.Cos -> cos(v)
            UnaryOp.Abs -> abs(v)
            UnaryOp.Neg -> -v
            UnaryOp.Recip -> 1.0f / v
            UnaryOp.Sqr -> v * v
            UnaryOp.Sqrt -> sqrt(v)
            UnaryOp.Gelu -> MathOps.gelu(v)
            UnaryOp.GeluErf -> MathOps.geluErf(v)
            UnaryOp.Erf -> MathOps.erf(v)
            UnaryOp.Relu -> MathOps.relu(v)
            UnaryOp.Silu -> MathOps.silu(v)
            UnaryOp.Tanh -> tanh(v)
            UnaryOp.Floor -> floor(v)
            UnaryOp.Ceil -> ceil(v)
            UnaryOp.Round -> round(v)
            UnaryOp.Sign -> sign(v)
        }

    private fun applyUnaryDouble(
        v: Double,
        op: UnaryOp,
    ): Double =
        when (op) {
            UnaryOp.Exp -> exp(v)
            UnaryOp.Log -> ln(v)
            UnaryOp.Sin -> sin(v)
            UnaryOp.Cos -> cos(v)
            UnaryOp.Abs -> abs(v)
            UnaryOp.Neg -> -v
            UnaryOp.Recip -> 1.0 / v
            UnaryOp.Sqr -> v * v
            UnaryOp.Sqrt -> sqrt(v)
            UnaryOp.Gelu -> MathOps.gelu(v)
            UnaryOp.GeluErf -> MathOps.geluErf(v)
            UnaryOp.Erf -> MathOps.erf(v)
            UnaryOp.Relu -> MathOps.relu(v)
            UnaryOp.Silu -> MathOps.silu(v)
            UnaryOp.Tanh -> tanh(v)
            UnaryOp.Floor -> floor(v)
            UnaryOp.Ceil -> ceil(v)
            UnaryOp.Round -> round(v)
            UnaryOp.Sign -> sign(v)
        }

    public fun copy2d(
        dst: CpuStorage,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        when (this) {
            is U8 -> copy2dByteArray(data, (dst as U8).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is U32 -> copy2dUIntArray(data, (dst as U32).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is I16 -> copy2dShortArray(data, (dst as I16).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is I32 -> copy2dIntArray(data, (dst as I32).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is I64 -> copy2dLongArray(data, (dst as I64).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is BF16 -> copy2dShortArray(data, (dst as BF16).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F16 -> copy2dShortArray(data, (dst as F16).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F32 -> copy2dFloatArray(data, (dst as F32).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F64 -> copy2dDoubleArray(data, (dst as F64).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F8E4M3 -> copy2dRawByteArray(data, (dst as F8E4M3).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F6E2M3 -> copy2dRawByteArray(data, (dst as F6E2M3).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F6E3M2 -> copy2dRawByteArray(data, (dst as F6E3M2).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F4 -> copy2dRawByteArray(data, (dst as F4).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
            is F8E8M0 -> copy2dRawByteArray(data, (dst as F8E8M0).data, d1, d2, srcStride1, dstStride1, srcOffset, dstOffset)
        }
    }

    public fun copyStridedSrc(
        dst: CpuStorage,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (this) {
            is U8 -> copyStridedSrcByteArray(data, (dst as U8).data, dstOffset, srcLayout)
            is U32 -> copyStridedSrcUIntArray(data, (dst as U32).data, dstOffset, srcLayout)
            is I16 -> copyStridedSrcShortArray(data, (dst as I16).data, dstOffset, srcLayout)
            is I32 -> copyStridedSrcIntArray(data, (dst as I32).data, dstOffset, srcLayout)
            is I64 -> copyStridedSrcLongArray(data, (dst as I64).data, dstOffset, srcLayout)
            is BF16 -> copyStridedSrcShortArray(data, (dst as BF16).data, dstOffset, srcLayout)
            is F16 -> copyStridedSrcShortArray(data, (dst as F16).data, dstOffset, srcLayout)
            is F32 -> copyStridedSrcFloatArray(data, (dst as F32).data, dstOffset, srcLayout)
            is F64 -> copyStridedSrcDoubleArray(data, (dst as F64).data, dstOffset, srcLayout)
            is F8E4M3 -> copyStridedSrcRawByteArray(data, (dst as F8E4M3).data, dstOffset, srcLayout)
            is F6E2M3 -> copyStridedSrcRawByteArray(data, (dst as F6E2M3).data, dstOffset, srcLayout)
            is F6E3M2 -> copyStridedSrcRawByteArray(data, (dst as F6E3M2).data, dstOffset, srcLayout)
            is F4 -> copyStridedSrcRawByteArray(data, (dst as F4).data, dstOffset, srcLayout)
            is F8E8M0 -> copyStridedSrcRawByteArray(data, (dst as F8E8M0).data, dstOffset, srcLayout)
        }
    }

    private fun copy2dByteArray(
        src: UByteArray,
        dst: UByteArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dUIntArray(
        src: UIntArray,
        dst: UIntArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dShortArray(
        src: ShortArray,
        dst: ShortArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dIntArray(
        src: IntArray,
        dst: IntArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dLongArray(
        src: LongArray,
        dst: LongArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dFloatArray(
        src: FloatArray,
        dst: FloatArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dDoubleArray(
        src: DoubleArray,
        dst: DoubleArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dRawByteArray(
        src: ByteArray,
        dst: ByteArray,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        repeat(d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copyStridedSrcByteArray(
        src: UByteArray,
        dst: UByteArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dByteArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcUIntArray(
        src: UIntArray,
        dst: UIntArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dUIntArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcShortArray(
        src: ShortArray,
        dst: ShortArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dShortArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcIntArray(
        src: IntArray,
        dst: IntArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dIntArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcLongArray(
        src: LongArray,
        dst: LongArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dLongArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcFloatArray(
        src: FloatArray,
        dst: FloatArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dFloatArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcDoubleArray(
        src: DoubleArray,
        dst: DoubleArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dDoubleArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcRawByteArray(
        src: ByteArray,
        dst: ByteArray,
        dstOffset: Int,
        srcLayout: Layout,
    ) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> {
                src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            }

            is StridedBlocks.UniformBlocks -> {
                copy2dRawByteArray(
                    src,
                    dst,
                    blocks.count,
                    blocks.blockLen,
                    blocks.srcStride,
                    blocks.blockLen,
                    blocks.startOffset,
                    dstOffset,
                )
            }

            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for ((i, srcIndex) in blockStartIndex.asSequence().withIndex()) dst[dstOffset + i] = src[srcIndex]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex.asSequence()) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    public fun conv1d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): CpuStorage {
        if (!useIm2colConv1d) {
            return Conv1D(params).map(this, l, kernel, kernelL)
        }
        val op =
            Im2Col1D(
                lK = params.kSize,
                padding = params.padding,
                stride = params.stride,
                dilation = params.dilation,
            )
        val col = op.map(this, l)
        val b = params.bSize
        val n = params.cOut
        val lOut = params.lOut()
        val k = op.lK * params.cIn
        val m = lOut
        val colL = Layout.contiguous(Shape(b, m, k))
        val res =
            if (kernelL.isContiguous()) {
                val kL =
                    Layout
                        .contiguousWithOffset(Shape(1, n, k), kernelL.startOffset)
                        .transpose(1, 2)
                        .broadcastAs(Shape(b, k, n))
                col.matmul(kernel, listOf(b, m, n, k), colL, kL)
            } else {
                val kernelC = device().allocUninit(kernelL.shape, kernel.dtype())
                kernel.copyStridedSrc(kernelC, 0, kernelL)
                val kL =
                    Layout
                        .contiguousWithOffset(Shape(1, n, k), 0)
                        .transpose(1, 2)
                        .broadcastAs(Shape(b, k, n))
                col.matmul(kernelC, listOf(b, m, n, k), colL, kL)
            }
        val resL = Layout.contiguous(Shape(b, lOut, params.cOut)).transpose(1, 2)
        val resT = device().allocUninit(resL.shape, res.dtype())
        res.copyStridedSrc(resT, 0, resL)
        return resT
    }

    public fun convTranspose1d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): CpuStorage {
        val canUseCol2im =
            kernelL.isContiguous() &&
                params.dilation == 1 &&
                params.padding == 0 &&
                params.outputPadding == 0
        return if (useCol2imConv1dTr && canUseCol2im) {
            val (bSize, cIn, lIn) = l.shape.dims3()
            val (cIn2, cOut, kSize) = kernelL.shape.dims3()
            if (!kernelL.isContiguous()) {
                bail("convtr1d: the second argument (kernel) has to be contiguous $kernelL")
            }
            if (cIn != cIn2) {
                bail("convtr1d: shape mismatch on c_in ${l.shape} ${kernelL.shape}")
            }
            val col =
                run {
                    val kernelLMm =
                        Layout(
                            Shape(bSize, cIn, kSize * cOut),
                            listOf(0, kSize * cOut, 1),
                            kernelL.startOffset,
                        )
                    matmul(
                        kernel,
                        listOf(bSize, lIn, cOut * kSize, cIn),
                        l.transpose(1, 2),
                        kernelLMm,
                    )
                }
            val colL = Layout.contiguous(Shape(bSize, lIn, cOut, kSize))
            Col2Im1D(stride = params.stride).map(col, colL)
        } else {
            ConvTranspose1D(params).map(this, l, kernel, kernelL)
        }
    }

    public fun conv2d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): CpuStorage = Conv2D(params).map(this, l, kernel, kernelL)

    public fun convTranspose2d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): CpuStorage = ConvTranspose2D(params).map(this, l, kernel, kernelL)

    public fun avgPool2d(
        layout: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): CpuStorage {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val h = d.d2
        val w = d.d3
        val s = layout.stride
        val (strideH, strideW) = Pair(s[2], s[3])
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset
        val count = bSz * c * hOut * wOut
        val scale = 1.0 / (kH * kW).toDouble()
        val dst = FloatArray(count)
        for (bIdx in 0 until bSz) {
            val bOffset = bIdx * c * hOut * wOut
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val cOffset = bOffset + cIdx * hOut * wOut
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var sum = 0.0
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val mIdx = sH * hIdx + m
                                val nIdx = sW * wIdx + n
                                sum += getAsDouble(srcC + mIdx * strideH + nIdx * strideW)
                            }
                        }
                        dst[cOffset + hIdx * wOut + wIdx] = (sum * scale).toFloat()
                    }
                }
            }
        }
        return F32(dst)
    }

    public fun maxPool2d(
        layout: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): CpuStorage {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val h = d.d2
        val w = d.d3
        val s = layout.stride
        val (strideH, strideW) = Pair(s[2], s[3])
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset
        val count = bSz * c * hOut * wOut
        val dst = FloatArray(count)
        for (bIdx in 0 until bSz) {
            val bOffset = bIdx * c * hOut * wOut
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val cOffset = bOffset + cIdx * hOut * wOut
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var largest = getAsDouble(srcC + sH * hIdx * strideH + sW * wIdx * strideW)
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val mIdx = sH * hIdx + m
                                val nIdx = sW * wIdx + n
                                val v = getAsDouble(srcC + mIdx * strideH + nIdx * strideW)
                                if (v > largest) largest = v
                            }
                        }
                        dst[cOffset + hIdx * wOut + wIdx] = largest.toFloat()
                    }
                }
            }
        }
        return F32(dst)
    }

    public fun upsampleNearest1d(
        layout: Layout,
        dstSz: Int,
    ): CpuStorage {
        val (bSz, c, srcSz) = layout.shape.dims3()
        val s = layout.stride
        val strideSz = s[2]
        val srcIndex = layout.startOffset
        val scaleSz = srcSz.toDouble() / dstSz.toDouble()
        val count = bSz * c * dstSz
        val dst = FloatArray(count)
        val srcIdxs =
            IntArray(dstSz) { idx ->
                minOf(srcSz - 1, (idx.toDouble() * scaleSz).toInt())
            }
        for (bIdx in 0 until bSz) {
            val bOffset = bIdx * c * dstSz
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val cOffset = bOffset + cIdx * dstSz
                val srcC = srcB + cIdx * s[1]
                for (idx in 0 until dstSz) {
                    dst[cOffset + idx] = getAsFloat(srcC + srcIdxs[idx] * strideSz)
                }
            }
        }
        return F32(dst)
    }

    public fun upsampleNearest2d(
        layout: Layout,
        dstH: Int,
        dstW: Int,
    ): CpuStorage {
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val srcH = d.d2
        val srcW = d.d3
        val s = layout.stride
        val (strideH, strideW) = Pair(s[2], s[3])
        val srcIndex = layout.startOffset
        val scaleH = srcH.toDouble() / dstH.toDouble()
        val scaleW = srcW.toDouble() / dstW.toDouble()
        val count = bSz * c * dstH * dstW
        val dst = FloatArray(count)
        val srcHIdxs =
            IntArray(dstH) { hIdx ->
                minOf(srcH - 1, (hIdx.toDouble() * scaleH).toInt())
            }
        val srcWIdxs =
            IntArray(dstW) { wIdx ->
                minOf(srcW - 1, (wIdx.toDouble() * scaleW).toInt())
            }
        for (bIdx in 0 until bSz) {
            val bOffset = bIdx * c * dstH * dstW
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val cOffset = bOffset + cIdx * dstH * dstW
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until dstH) {
                    val srcHIdx = srcHIdxs[hIdx]
                    for (wIdx in 0 until dstW) {
                        val srcWIdx = srcWIdxs[wIdx]
                        dst[cOffset + hIdx * dstW + wIdx] = getAsFloat(srcC + srcHIdx * strideH + srcWIdx * strideW)
                    }
                }
            }
        }
        return F32(dst)
    }

    public fun upsampleBilinear2d(
        layout: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleHFactor: Double? = null,
        scaleWFactor: Double? = null,
    ): CpuStorage {
        val d = layout.shape.dims4()
        val batch = d.d0
        val channels = d.d1
        val heightIn = d.d2
        val widthIn = d.d3
        val heightOut = targetH
        val widthOut = targetW

        if (heightIn == heightOut && widthIn == widthOut) {
            return tryClone(layout)
        }

        val s = layout.stride
        val srcOffset = layout.startOffset

        val scaleH =
            if (alignCorners) {
                if (heightOut > 1) (heightIn - 1).toDouble() / (heightOut - 1).toDouble() else 0.0
            } else {
                scaleHFactor?.let { 1.0 / it } ?: (heightIn.toDouble() / heightOut.toDouble())
            }

        val scaleW =
            if (alignCorners) {
                if (widthOut > 1) (widthIn - 1).toDouble() / (widthOut - 1).toDouble() else 0.0
            } else {
                scaleWFactor?.let { 1.0 / it } ?: (widthIn.toDouble() / widthOut.toDouble())
            }

        val hIndices =
            Array(heightOut) { hOut ->
                val srcH = if (alignCorners) scaleH * hOut.toDouble() else scaleH * (hOut.toDouble() + 0.5) - 0.5
                val srcHClamped = maxOf(0.0, srcH)
                val h0 = srcHClamped.toInt()
                val h1 = minOf(heightIn - 1, h0 + 1)
                val weightH = (srcHClamped - h0.toDouble()).coerceIn(0.0, 1.0)
                Triple(h0, h1, weightH)
            }

        val wIndices =
            Array(widthOut) { wOut ->
                val srcW = if (alignCorners) scaleW * wOut.toDouble() else scaleW * (wOut.toDouble() + 0.5) - 0.5
                val srcWClamped = maxOf(0.0, srcW)
                val w0 = srcWClamped.toInt()
                val w1 = minOf(widthIn - 1, w0 + 1)
                val weightW = (srcWClamped - w0.toDouble()).coerceIn(0.0, 1.0)
                Triple(w0, w1, weightW)
            }

        val dst = FloatArray(batch * channels * heightOut * widthOut)
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                val baseIdx = srcOffset + b * s[0] + c * s[1]
                val dstBase = (b * channels + c) * heightOut * widthOut

                for (hOut in 0 until heightOut) {
                    val (h0, h1, weightH) = hIndices[hOut]
                    for (wOut in 0 until widthOut) {
                        val (w0, w1, weightW) = wIndices[wOut]
                        val idx00 = baseIdx + h0 * s[2] + w0 * s[3]
                        val idx10 = baseIdx + h0 * s[2] + w1 * s[3]
                        val idx01 = baseIdx + h1 * s[2] + w0 * s[3]
                        val idx11 = baseIdx + h1 * s[2] + w1 * s[3]

                        val v00 = getAsDouble(idx00)
                        val v10 = getAsDouble(idx10)
                        val v01 = getAsDouble(idx01)
                        val v11 = getAsDouble(idx11)

                        val vTop = v00 * (1.0 - weightW) + v10 * weightW
                        val vBottom = v01 * (1.0 - weightW) + v11 * weightW
                        val value = vTop * (1.0 - weightH) + vBottom * weightH

                        dst[dstBase + hOut * widthOut + wOut] = value.toFloat()
                    }
                }
            }
        }
        return F32(dst)
    }

    public fun scatterSet(
        l: Layout,
        indexes: CpuStorage,
        indexesL: Layout,
        source: CpuStorage,
        sourceL: Layout,
        d: Int,
    ) = scatterUpdate(l, indexes, indexesL, source, sourceL, d, false)

    public fun scatterAddSet(
        l: Layout,
        indexes: CpuStorage,
        indexesL: Layout,
        source: CpuStorage,
        sourceL: Layout,
        d: Int,
    ) = scatterUpdate(l, indexes, indexesL, source, sourceL, d, true)

    private fun scatterUpdate(
        l: Layout,
        indexes: CpuStorage,
        indexesL: Layout,
        source: CpuStorage,
        sourceL: Layout,
        d: Int,
        add: Boolean,
    ) {
        val op = if (add) "scatter-add" else "scatter"
        if (dtype() != source.dtype()) throw CandleException.DTypeMismatchBinaryOp(dtype(), source.dtype(), op)
        if (this is F6E2M3 || this is F6E3M2 || this is F4 || this is F8E8M0) {
            throw CandleException.UnsupportedDTypeForOp(dtype(), op)
        }
        val ids = indexes.scatterIndexes(op, add)
        Scatter(ids, indexesL, d).forEachOffset(l, sourceL) { dstIndex, srcIndex ->
            updateElement(source, dstIndex, srcIndex, add)
        }
    }

    private fun scatterIndexes(op: String, signedSmall: Boolean): List<Long> =
        when (this) {
            is U8 -> data.map { if (it == UByte.MAX_VALUE) Long.MAX_VALUE else it.toLong() }
            is U32 -> data.map { if (it == UInt.MAX_VALUE) Long.MAX_VALUE else it.toLong() }
            is I64 -> data.asList()
            is I16 -> if (signedSmall) data.map { if (it == Short.MAX_VALUE) Long.MAX_VALUE else it.toLong() } else throw CandleException.UnsupportedDTypeForOp(dtype(), op)
            is I32 -> if (signedSmall) data.map { if (it == Int.MAX_VALUE) Long.MAX_VALUE else it.toLong() } else throw CandleException.UnsupportedDTypeForOp(dtype(), op)
            else -> throw CandleException.UnsupportedDTypeForOp(dtype(), op)
        }

    private fun updateElement(source: CpuStorage, dst: Int, src: Int, add: Boolean) {
        when (this) {
            is U8 -> {
                val v = (source as U8).data[src]
                data[dst] = if (add) (data[dst] + v).toUByte() else v
            }

            is U32 -> {
                val v = (source as U32).data[src]
                data[dst] = if (add) data[dst] + v else v
            }

            is I16 -> {
                val v = (source as I16).data[src]
                data[dst] = if (add) (data[dst] + v).toShort() else v
            }

            is I32 -> {
                val v = (source as I32).data[src]
                data[dst] = if (add) data[dst] + v else v
            }

            is I64 -> {
                val v = (source as I64).data[src]
                data[dst] = if (add) data[dst] + v else v
            }

            is F32 -> {
                val v = (source as F32).data[src]
                data[dst] = if (add) data[dst] + v else v
            }

            is F64 -> {
                val v = (source as F64).data[src]
                data[dst] = if (add) data[dst] + v else v
            }

            is BF16 -> {
                val v = (source as BF16).data[src]
                data[dst] = if (add) Float16Utils.floatToBF16Bits(Float16Utils.bf16BitsToFloat(data[dst]) + Float16Utils.bf16BitsToFloat(v)) else v
            }

            is F16 -> {
                val v = (source as F16).data[src]
                data[dst] = if (add) Float16Utils.floatToF16Bits(Float16Utils.f16BitsToFloat(data[dst]) + Float16Utils.f16BitsToFloat(v)) else v
            }

            is F8E4M3 -> {
                val v = (source as F8E4M3).data[src]
                data[dst] = if (add) (Float8.fromBits(data[dst].toUByte()) + Float8.fromBits(v.toUByte())).bits else v
            }

            is F6E2M3, is F6E3M2, is F4, is F8E8M0 -> {
                throw CandleException.UnsupportedDTypeForOp(dtype(), "scatter")
            }
        }
    }

    public fun indexAdd(
        l: Layout,
        indexes: CpuStorage,
        indexesL: Layout,
        source: CpuStorage,
        sourceL: Layout,
        d: Int,
    ): CpuStorage {
        val op = "index-add"
        if (dtype() != source.dtype()) throw CandleException.DTypeMismatchBinaryOp(dtype(), source.dtype(), op)
        if (this is F6E2M3 || this is F6E3M2 || this is F4 || this is F8E8M0) {
            throw CandleException.UnsupportedDTypeForOp(dtype(), op)
        }
        val offsets = indexesL.contiguousOffsets() ?: throw CandleException.RequiresContiguous(op)
        val ids = indexes.scatterIndexes(op, true).subList(offsets.start, offsets.end)
        val res = tryClone(l)
        IndexAdd(ids, d).forEachOffset(l, sourceL) { dstIndex, srcIndex ->
            res.updateElement(source, dstIndex, srcIndex, true)
        }
        return res
    }

    public companion object {
        public fun concat(storages: List<CpuStorage>): CpuStorage {
            if (storages.isEmpty()) bail("cannot concat empty storages")
            val storage0 = storages[0]
            return when (storage0) {
                is U8 -> {
                    val total = storages.sumOf { (it as? U8)?.data?.size ?: bail("dtype mismatch") }
                    val dst = UByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as U8).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    U8(dst)
                }

                is U32 -> {
                    val total = storages.sumOf { (it as? U32)?.data?.size ?: bail("dtype mismatch") }
                    val dst = UIntArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as U32).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    U32(dst)
                }

                is I16 -> {
                    val total = storages.sumOf { (it as? I16)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ShortArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as I16).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    I16(dst)
                }

                is I32 -> {
                    val total = storages.sumOf { (it as? I32)?.data?.size ?: bail("dtype mismatch") }
                    val dst = IntArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as I32).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    I32(dst)
                }

                is I64 -> {
                    val total = storages.sumOf { (it as? I64)?.data?.size ?: bail("dtype mismatch") }
                    val dst = LongArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as I64).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    I64(dst)
                }

                is BF16 -> {
                    val total = storages.sumOf { (it as? BF16)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ShortArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as BF16).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    BF16(dst)
                }

                is F16 -> {
                    val total = storages.sumOf { (it as? F16)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ShortArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F16).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F16(dst)
                }

                is F32 -> {
                    val total = storages.sumOf { (it as? F32)?.data?.size ?: bail("dtype mismatch") }
                    val dst = FloatArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F32).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F32(dst)
                }

                is F64 -> {
                    val total = storages.sumOf { (it as? F64)?.data?.size ?: bail("dtype mismatch") }
                    val dst = DoubleArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F64).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F64(dst)
                }

                is F8E4M3 -> {
                    val total = storages.sumOf { (it as? F8E4M3)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F8E4M3).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F8E4M3(dst)
                }

                is F6E2M3 -> {
                    val total = storages.sumOf { (it as? F6E2M3)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F6E2M3).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F6E2M3(dst)
                }

                is F6E3M2 -> {
                    val total = storages.sumOf { (it as? F6E3M2)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F6E3M2).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F6E3M2(dst)
                }

                is F4 -> {
                    val total = storages.sumOf { (it as? F4)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F4).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F4(dst)
                }

                is F8E8M0 -> {
                    val total = storages.sumOf { (it as? F8E8M0)?.data?.size ?: bail("dtype mismatch") }
                    val dst = ByteArray(total)
                    var offset = 0
                    for (s in storages) {
                        val arr = (s as F8E8M0).data
                        arr.copyInto(dst, offset)
                        offset += arr.size
                    }
                    F8E8M0(dst)
                }
            }
        }
    }
}

public sealed class CpuStorageRef {
    public data class U8(
        val data: UByteArray,
    ) : CpuStorageRef()

    public data class U32(
        val data: UIntArray,
    ) : CpuStorageRef()

    public data class I16(
        val data: ShortArray,
    ) : CpuStorageRef()

    public data class I32(
        val data: IntArray,
    ) : CpuStorageRef()

    public data class I64(
        val data: LongArray,
    ) : CpuStorageRef()

    public data class BF16(
        val data: ShortArray,
    ) : CpuStorageRef()

    public data class F16(
        val data: ShortArray,
    ) : CpuStorageRef()

    public data class F32(
        val data: FloatArray,
    ) : CpuStorageRef()

    public data class F64(
        val data: DoubleArray,
    ) : CpuStorageRef()

    public data class F8E4M3(
        val data: ByteArray,
    ) : CpuStorageRef()

    public data class F6E2M3(
        val data: ByteArray,
    ) : CpuStorageRef()

    public data class F6E3M2(
        val data: ByteArray,
    ) : CpuStorageRef()

    public data class F4(
        val data: ByteArray,
    ) : CpuStorageRef()

    public data class F8E8M0(
        val data: ByteArray,
    ) : CpuStorageRef()
}

public class Cmp(
    public val opKind: CmpOp,
) : Map2U8 {
    override val op: String = "cmp"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): UByteArray {
        val dst =
            when (opKind) {
                CmpOp.Eq -> binaryMap(l1, l2, v1, v2) { x, y -> if (x == y) 1u else 0u }
                CmpOp.Ne -> binaryMap(l1, l2, v1, v2) { x, y -> if (x != y) 1u else 0u }
                CmpOp.Lt -> binaryMap(l1, l2, v1, v2) { x, y -> if (x < y) 1u else 0u }
                CmpOp.Le -> binaryMap(l1, l2, v1, v2) { x, y -> if (x <= y) 1u else 0u }
                CmpOp.Gt -> binaryMap(l1, l2, v1, v2) { x, y -> if (x > y) 1u else 0u }
                CmpOp.Ge -> binaryMap(l1, l2, v1, v2) { x, y -> if (x >= y) 1u else 0u }
            }
        return UByteArray(dst.size) { dst[it].toUByte() }
    }
}

public class WCond(
    public val pred: List<Long>,
    public val predL: Layout,
) : Map2 {
    override val op: String = "where"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val oPred = predL.contiguousOffsets()
        val oT = l1.contiguousOffsets()
        val oF = l2.contiguousOffsets()
        if (oPred != null && oT != null && oF != null) {
            val pSlice = pred.subList(oPred.start, oPred.end)
            val tSlice = v1.subList(oT.start, oT.end)
            val fSlice = v2.subList(oF.start, oF.end)
            return List(pSlice.size) { i ->
                if (pSlice[i] != 0L) tSlice[i] else fSlice[i]
            }
        }
        val pIter = predL.stridedIndex()
        val tIter = l1.stridedIndex()
        val fIter = l2.stridedIndex()
        val out = ArrayList<Double>(predL.shape.elemCount())
        while (pIter.hasNext() && tIter.hasNext() && fIter.hasNext()) {
            val iP = pIter.next()
            val iT = tIter.next()
            val iF = fIter.next()
            out.add(if (pred[iP] != 0L) v1[iT] else v2[iF])
        }
        return out
    }
}

public class ReduceIndex(
    public val reduceDimIndex: Int,
    public val useMin: Boolean,
    public val returnIndex: Boolean,
) : Map1Any {
    // The value gets replaced if f(s[currentAcc], s[i]) returns true.
    public fun <U> foldImpl(
        src: List<Double>,
        srcL: Layout,
        f: (Double, Double) -> Boolean,
        g: (Double, Int) -> U,
    ): List<U> = foldTyped(src, srcL, f, g)

    internal fun <T, U> foldTyped(
        src: List<T>,
        srcL: Layout,
        f: (T, T) -> Boolean,
        g: (T, Int) -> U,
    ): List<U> {
        val reduceDimSize = srcL.dims[reduceDimIndex]
        val reduceDimStride = srcL.stride[reduceDimIndex]
        val dstLen = srcL.shape.elemCount() / reduceDimSize
        val dst = ArrayList<U>(dstLen)
        val offsets = srcL.contiguousOffsets()
        if (offsets != null) {
            val s = src.subList(offsets.start, offsets.end)
            if (reduceDimStride == 1) {
                for (startSrcIdx in 0 until dstLen) {
                    val startSrcI = startSrcIdx * reduceDimSize
                    var acc = 0
                    var value = s[startSrcI]
                    for (srcI in 0 until reduceDimSize) {
                        val cur = s[startSrcI + srcI]
                        if (f(value, cur)) {
                            acc = srcI
                            value = cur
                        }
                    }
                    dst.add(g(value, acc))
                }
            } else {
                for (startSrcIdx in 0 until dstLen) {
                    val p = startSrcIdx / reduceDimStride
                    val q = startSrcIdx % reduceDimStride
                    val startSrcI = p * reduceDimStride * reduceDimSize + q
                    var acc = 0
                    var value = s[startSrcI]
                    for (srcI in 0 until reduceDimSize) {
                        val cur = s[startSrcI + srcI * reduceDimStride]
                        if (f(value, cur)) {
                            acc = srcI
                            value = cur
                        }
                    }
                    dst.add(g(value, acc))
                }
            }
        } else {
            val l = srcL.narrow(reduceDimIndex, 0, 1)
            for (srcIndex in l.stridedIndex()) {
                var acc = 0
                var value = src[srcIndex]
                for (srcI in 0 until reduceDimSize) {
                    val cur = src[srcIndex + srcI * reduceDimStride]
                    if (f(value, cur)) {
                        acc = srcI
                        value = cur
                    }
                }
                dst.add(g(value, acc))
            }
        }
        return dst
    }

    override fun f(
        vs: List<Double>,
        layout: Layout,
        wrap: (List<Double>) -> CpuStorage,
    ): CpuStorage {
        if (layout.shape.elemCount() == 0) {
            throw CandleException.EmptyTensor("reduce")
        }
        return when {
            !returnIndex && useMin -> {
                wrap(foldImpl(vs, layout, { x, y -> x > y }, { v, _ -> v }))
            }

            !returnIndex && !useMin -> {
                wrap(foldImpl(vs, layout, { x, y -> x < y }, { v, _ -> v }))
            }

            returnIndex && useMin -> {
                val idxs = foldImpl(vs, layout, { x, y -> x > y }, { _, i -> i.toUInt() })
                CpuStorage.U32(UIntArray(idxs.size) { idxs[it] })
            }

            else -> {
                val idxs = foldImpl(vs, layout, { x, y -> x < y }, { _, i -> i.toUInt() })
                CpuStorage.U32(UIntArray(idxs.size) { idxs[it] })
            }
        }
    }
}

public class ReduceSum internal constructor(
    public val dstShape: Shape,
    public val reduceDims: IntArray,
    public val reduceDimsAndStride: List<Pair<Int, Int>>,
) : Map1 {
    public fun foldImpl(
        src: List<Double>,
        srcL: Layout,
        startElt: Double,
    ): List<Double> = foldTyped(src, srcL, startElt) { a, b -> a + b }

    internal fun <T> foldTyped(
        src: List<T>,
        srcL: Layout,
        startElt: T,
        add: (T, T) -> T,
    ): List<T> {
        val dst = MutableList(dstShape.elemCount()) { startElt }
        val offsets = srcL.contiguousOffsets()
        if (offsets != null) {
            val s = src.subList(offsets.start, offsets.end)
            var reduceOverLastDims = true
            for (i in reduceDims.indices) {
                val v = reduceDims[reduceDims.size - 1 - i]
                if (v != srcL.shape.rank - 1 - i) {
                    reduceOverLastDims = false
                    break
                }
            }
            if (reduceOverLastDims) {
                var reduceSz = 1
                for ((u, _) in reduceDimsAndStride) reduceSz *= u
                for (dstI in 0 until dst.size) {
                    val srcI = dstI * reduceSz
                    var acc = startElt
                    for (k in 0 until reduceSz) acc = add(acc, s[srcI + k])
                    dst[dstI] = acc
                }
                return dst
            }
            for (unstrIndex in s.indices) {
                var dstIndex = unstrIndex
                for ((dim, stride) in reduceDimsAndStride) {
                    val pre = dstIndex / stride
                    val post = dstIndex % stride
                    dstIndex = (pre / dim) * stride + post
                }
                dst[dstIndex] = add(dst[dstIndex], s[unstrIndex])
            }
        } else {
            var unstrIndex = 0
            for (srcIndex in srcL.stridedIndex()) {
                var dstIndex = unstrIndex++
                for ((dim, stride) in reduceDimsAndStride) {
                    val pre = dstIndex / stride
                    val post = dstIndex % stride
                    dstIndex = (pre / dim) * stride + post
                }
                dst[dstIndex] = add(dst[dstIndex], src[srcIndex])
            }
        }
        return dst
    }

    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> = foldImpl(vs, layout, 0.0)
}

public class Affine(
    public val mul: Double,
    public val add: Double,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> = unaryMap(vs, layout) { v -> v * mul + add }
}

public class AvgPool2D internal constructor(
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int>,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val h = d.d2
        val w = d.d3
        val s = layout.stride
        val strideH = s[2]
        val strideW = s[3]
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset
        val dst = MutableList(bSz * c * hOut * wOut) { 0.0 }
        val scale = 1.0 / (kH * kW).toDouble()
        for (bIdx in 0 until bSz) {
            val dstB = bIdx * c * hOut * wOut
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val dstC = dstB + cIdx * hOut * wOut
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var sum = 0.0
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val mPos = sH * hIdx + m
                                val nPos = sW * wIdx + n
                                sum += vs[srcC + mPos * strideH + nPos * strideW]
                            }
                        }
                        dst[dstC + hIdx * wOut + wIdx] = sum * scale
                    }
                }
            }
        }
        return dst
    }
}

public class MaxPool2D internal constructor(
    public val kernelSize: Pair<Int, Int>,
    public val stride: Pair<Int, Int>,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val h = d.d2
        val w = d.d3
        val s = layout.stride
        val strideH = s[2]
        val strideW = s[3]
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset
        val dst = MutableList(bSz * c * hOut * wOut) { 0.0 }
        for (bIdx in 0 until bSz) {
            val dstB = bIdx * c * hOut * wOut
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val dstC = dstB + cIdx * hOut * wOut
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var largest = vs[srcC + sH * hIdx * strideH + sW * wIdx * strideW]
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val mPos = sH * hIdx + m
                                val nPos = sW * wIdx + n
                                val cur = vs[srcC + mPos * strideH + nPos * strideW]
                                if (largest < cur) largest = cur
                            }
                        }
                        dst[dstC + hIdx * wOut + wIdx] = largest
                    }
                }
            }
        }
        return dst
    }
}

public class UpsampleNearest1D(
    public val targetSize: Int,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val dstSz = targetSize
        val (bSz, c, srcSz) = layout.shape.dims3()
        val s = layout.stride
        val strideSz = s[2]
        val srcIndex = layout.startOffset
        val scaleSz = srcSz.toDouble() / dstSz.toDouble()
        val dst = MutableList(bSz * c * dstSz) { 0.0 }
        val srcIdxs = IntArray(dstSz) { idx -> minOf(srcSz - 1, (idx.toDouble() * scaleSz).toInt()) }
        for (bIdx in 0 until bSz) {
            val dstB = bIdx * c * dstSz
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val dstC = dstB + cIdx * dstSz
                val srcC = srcB + cIdx * s[1]
                for (idx in 0 until dstSz) {
                    dst[dstC + idx] = vs[srcC + srcIdxs[idx] * strideSz]
                }
            }
        }
        return dst
    }
}

public class UpsampleNearest2D(
    public val targetH: Int,
    public val targetW: Int,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val dstH = targetH
        val dstW = targetW
        val d = layout.shape.dims4()
        val bSz = d.d0
        val c = d.d1
        val srcH = d.d2
        val srcW = d.d3
        val s = layout.stride
        val strideH = s[2]
        val strideW = s[3]
        val srcIndex = layout.startOffset
        val scaleH = srcH.toDouble() / dstH.toDouble()
        val scaleW = srcW.toDouble() / dstW.toDouble()
        val dst = MutableList(bSz * c * dstH * dstW) { 0.0 }
        val srcHIdxs = IntArray(dstH) { hIdx -> minOf(srcH - 1, (hIdx.toDouble() * scaleH).toInt()) }
        val srcWIdxs = IntArray(dstW) { wIdx -> minOf(srcW - 1, (wIdx.toDouble() * scaleW).toInt()) }
        for (bIdx in 0 until bSz) {
            val dstB = bIdx * c * dstH * dstW
            val srcB = srcIndex + bIdx * s[0]
            for (cIdx in 0 until c) {
                val dstC = dstB + cIdx * dstH * dstW
                val srcC = srcB + cIdx * s[1]
                for (hIdx in 0 until dstH) {
                    val srcHIdx = srcHIdxs[hIdx]
                    for (wIdx in 0 until dstW) {
                        val srcWIdx = srcWIdxs[wIdx]
                        dst[dstC + hIdx * dstW + wIdx] = vs[srcC + srcHIdx * strideH + srcWIdx * strideW]
                    }
                }
            }
        }
        return dst
    }
}

public class UpsampleBilinear2D(
    public val targetH: Int,
    public val targetW: Int,
    public val alignCorners: Boolean,
    public val scaleHFactor: Double?,
    public val scaleWFactor: Double?,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val d = layout.shape.dims4()
        val batch = d.d0
        val channels = d.d1
        val heightIn = d.d2
        val widthIn = d.d3
        val heightOut = targetH
        val widthOut = targetW
        if (heightIn == heightOut && widthIn == widthOut) {
            return vs.toList()
        }
        val s = layout.stride
        val srcOffset = layout.startOffset
        val scaleH =
            if (alignCorners) {
                if (heightOut > 1) (heightIn - 1).toDouble() / (heightOut - 1).toDouble() else 0.0
            } else {
                scaleHFactor?.let { 1.0 / it } ?: (heightIn.toDouble() / heightOut.toDouble())
            }
        val scaleW =
            if (alignCorners) {
                if (widthOut > 1) (widthIn - 1).toDouble() / (widthOut - 1).toDouble() else 0.0
            } else {
                scaleWFactor?.let { 1.0 / it } ?: (widthIn.toDouble() / widthOut.toDouble())
            }
        val hIndices =
            Array(heightOut) { hOut ->
                val srcH = if (alignCorners) scaleH * hOut.toDouble() else scaleH * (hOut.toDouble() + 0.5) - 0.5
                val srcHClamped = maxOf(0.0, srcH)
                val h0 = floor(srcHClamped).toInt()
                val h1 = minOf(heightIn - 1, h0 + 1)
                val weightH = (srcHClamped - h0.toDouble()).coerceIn(0.0, 1.0)
                Triple(h0, h1, weightH)
            }
        val wIndices =
            Array(widthOut) { wOut ->
                val srcW = if (alignCorners) scaleW * wOut.toDouble() else scaleW * (wOut.toDouble() + 0.5) - 0.5
                val srcWClamped = maxOf(0.0, srcW)
                val w0 = floor(srcWClamped).toInt()
                val w1 = minOf(widthIn - 1, w0 + 1)
                val weightW = (srcWClamped - w0.toDouble()).coerceIn(0.0, 1.0)
                Triple(w0, w1, weightW)
            }
        val dst = MutableList(batch * channels * heightOut * widthOut) { 0.0 }
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                val baseIdx = srcOffset + b * s[0] + c * s[1]
                val dstBase = (b * channels + c) * heightOut * widthOut
                for (hOut in 0 until heightOut) {
                    val (h0, h1, weightH) = hIndices[hOut]
                    for (wOut in 0 until widthOut) {
                        val (w0, w1, weightW) = wIndices[wOut]
                        val v00 = vs[baseIdx + h0 * s[2] + w0 * s[3]]
                        val v10 = vs[baseIdx + h0 * s[2] + w1 * s[3]]
                        val v01 = vs[baseIdx + h1 * s[2] + w0 * s[3]]
                        val v11 = vs[baseIdx + h1 * s[2] + w1 * s[3]]
                        val vTop = v00 * (1.0 - weightW) + v10 * weightW
                        val vBottom = v01 * (1.0 - weightW) + v11 * weightW
                        dst[dstBase + hOut * widthOut + wOut] = vTop * (1.0 - weightH) + vBottom * weightH
                    }
                }
            }
        }
        return dst
    }
}

public class Gather(
    public val ids: List<Long>,
    public val idsL: Layout,
    public val dim: Int,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val idsOffsets = idsL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("gather")
        val srcOffsets = layout.contiguousOffsets() ?: throw CandleException.RequiresContiguous("gather")
        val idsSlice = ids.subList(idsOffsets.start, idsOffsets.end)
        val srcSlice = vs.subList(srcOffsets.start, srcOffsets.end)
        val idsDims = idsL.dims
        val srcDims = layout.dims
        val dstLen = idsDims.fold(1) { a, b -> a * b }
        val dstLeftLen = idsDims.subList(0, dim).fold(1) { a, b -> a * b }
        val dstDimLen = idsDims[dim]
        val dstRightLen = idsDims.subList(dim + 1, idsDims.size).fold(1) { a, b -> a * b }
        val srcDimLen = srcDims[dim]
        val srcRightLen = srcDims.subList(dim + 1, srcDims.size).fold(1) { a, b -> a * b }
        val dst = MutableList(dstLen) { 0.0 }
        for (leftI in 0 until dstLeftLen) {
            val startSrcIdx = leftI * srcRightLen * srcDimLen
            val startDstIdx = leftI * dstRightLen * dstDimLen
            for (i in 0 until dstDimLen) {
                val curDstIdx = startDstIdx + i * dstRightLen
                for (rightI in 0 until dstRightLen) {
                    val dstIdx = curDstIdx + rightI
                    val index = idsSlice[dstIdx]
                    if (index == Long.MAX_VALUE) {
                        dst[dstIdx] = 0.0
                    } else {
                        val idxInt = index.toInt()
                        if (idxInt < 0 || idxInt >= srcDimLen) {
                            throw CandleException.InvalidIndex("gather", idxInt, srcDimLen)
                        }
                        val srcIdx = startSrcIdx + idxInt * srcRightLen + rightI
                        dst[dstIdx] = srcSlice[srcIdx]
                    }
                }
            }
        }
        return dst
    }
}

public class IndexSelect(
    public val ids: List<Long>,
    public val idsL: Layout,
    public val dim: Int,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val srcOffsets = layout.contiguousOffsets() ?: throw CandleException.RequiresContiguous("index-select")
        val srcSlice = vs.subList(srcOffsets.start, srcOffsets.end)
        val idsDims = idsL.dims
        if (idsDims.size != 1) {
            throw CandleException.UnexpectedNumberOfDims(1, idsDims.size, idsL.shape)
        }
        val nIds = idsDims[0]
        val strideIds = idsL.stride[0]
        val dstDims = layout.dims.toMutableList()
        val srcDim = dstDims[dim]
        dstDims[dim] = nIds
        val dstLen = dstDims.fold(1) { a, b -> a * b }
        val leftLen = dstDims.subList(0, dim).fold(1) { a, b -> a * b }
        val rightLen = dstDims.subList(dim + 1, dstDims.size).fold(1) { a, b -> a * b }
        val dst = MutableList(dstLen) { 0.0 }
        for (leftI in 0 until leftLen) {
            val startSrcIdx = leftI * rightLen * srcDim
            val startDstIdx = leftI * rightLen * nIds
            for (i in 0 until nIds) {
                val curDstIdx = startDstIdx + i * rightLen
                val index = ids[idsL.startOffset + strideIds * i]
                if (index == Long.MAX_VALUE) {
                    for (r in 0 until rightLen) dst[curDstIdx + r] = 0.0
                } else {
                    val idxInt = index.toInt()
                    if (idxInt < 0 || idxInt >= srcDim) {
                        throw CandleException.InvalidIndex("index-select", idxInt, srcDim)
                    }
                    val curSrcIdx = startSrcIdx + idxInt * rightLen
                    for (r in 0 until rightLen) {
                        dst[curDstIdx + r] = srcSlice[curSrcIdx + r]
                    }
                }
            }
        }
        return dst
    }
}

public interface ElemUpdate {
    public fun f(
        dst: MutableList<Double>,
        idx: Int,
        src: Double,
    )

    public object Set : ElemUpdate {
        override fun f(
            dst: MutableList<Double>,
            idx: Int,
            src: Double,
        ) {
            dst[idx] = src
        }
    }

    public object Add : ElemUpdate {
        override fun f(
            dst: MutableList<Double>,
            idx: Int,
            src: Double,
        ) {
            dst[idx] = dst[idx] + src
        }
    }
}

public class Scatter(
    public val ids: List<Long>,
    public val idsL: Layout,
    public val dim: Int,
    public val update: ElemUpdate = ElemUpdate.Set,
) : Map2InPlace {
    override val op: String = "scatter"

    override fun f(
        v1: MutableList<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ) {
        forEachOffset(l1, l2) { dstIdx, srcIdx -> update.f(v1, dstIdx, v2[srcIdx]) }
    }

    internal fun forEachOffset(l1: Layout, l2: Layout, apply: (Int, Int) -> Unit) {
        if (dim !in l1.dims.indices) throw CandleException.DimOutOfRange(l1.shape, dim, op)
        if (l1.dims.size != l2.dims.size || l1.dims.indices.any { it != dim && l1.dims[it] != l2.dims[it] }) {
            throw CandleException.ShapeMismatchBinaryOp(l1.shape, l2.shape, op)
        }
        if (idsL.dims != l2.dims) throw CandleException.ShapeMismatchBinaryOp(idsL.shape, l2.shape, op)
        val dstOffsets = l1.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter")
        val srcOffsets = l2.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter")
        val idsOffsets = idsL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter")
        val idsDims = idsL.dims
        val dstDims = l1.dims
        val dstDimLen = dstDims[dim]
        val dstRightLen = dstDims.subList(dim + 1, dstDims.size).fold(1) { a, b -> a * b }
        val idsLeftLen = idsDims.subList(0, dim).fold(1) { a, b -> a * b }
        val idsDimLen = idsDims[dim]
        val idsRightLen = idsDims.subList(dim + 1, idsDims.size).fold(1) { a, b -> a * b }
        for (leftI in 0 until idsLeftLen) {
            val startIdsIdx = leftI * idsRightLen * idsDimLen
            val startDstIdx = leftI * dstRightLen * dstDimLen
            for (i in 0 until idsDimLen) {
                val curIdsIdx = startIdsIdx + i * idsRightLen
                for (rightI in 0 until dstRightLen) {
                    val idsIdx = curIdsIdx + rightI
                    val index = ids[idsOffsets.start + idsIdx]
                    if (index == Long.MAX_VALUE) continue
                    if (index < 0 || index >= dstDimLen.toLong()) {
                        throw CandleException.InvalidIndex(op, index.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()).toInt(), dstDimLen)
                    }
                    val dstIdx = dstOffsets.start + startDstIdx + index.toInt() * dstRightLen + rightI
                    apply(dstIdx, srcOffsets.start + idsIdx)
                }
            }
        }
    }

    public companion object {
        public fun new(
            ids: List<Long>,
            idsL: Layout,
            dim: Int,
            update: ElemUpdate = ElemUpdate.Set,
        ): Scatter = Scatter(ids, idsL, dim, update)
    }
}

public class IndexAdd(
    public val ids: List<Long>,
    public val dim: Int,
) : Map2 {
    override val op: String = "index-add"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val dstLen = l1.shape.elemCount()
        val dst = MutableList(dstLen) { 0.0 }
        copyStridedSrc(v1, dst, 0, l1)
        forEachOffset(l1, l2) { dstIndex, srcIndex -> dst[dstIndex] = dst[dstIndex] + v2[srcIndex] }
        return dst
    }

    internal fun forEachOffset(l1: Layout, l2: Layout, apply: (Int, Int) -> Unit) {
        if (dim !in l1.dims.indices) throw CandleException.DimOutOfRange(l1.shape, dim, op)
        if (l1.dims.size != l2.dims.size || l1.dims.indices.any { it != dim && l1.dims[it] != l2.dims[it] }) {
            throw CandleException.ShapeMismatchBinaryOp(l1.shape, l2.shape, op)
        }
        if (ids.size != l2.dims[dim]) throw CandleException.ShapeMismatchBinaryOp(Shape(ids.size), l2.shape, op)
        val srcOffsets = l2.contiguousOffsets() ?: throw CandleException.RequiresContiguous(op)
        val maxIdx = l1.dims[dim]
        val preDim = l2.dims.take(dim).fold(1, Int::times)
        val srcDimSz = l2.dims[dim]
        val postDim = l2.dims.drop(dim + 1).fold(1, Int::times)
        for ((srcIdx, rawDstIdx) in ids.withIndex()) {
            if (rawDstIdx == Long.MAX_VALUE) continue
            if (rawDstIdx < 0 || rawDstIdx >= maxIdx.toLong()) {
                throw CandleException.InvalidIndex(op, rawDstIdx.coerceIn(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong()).toInt(), maxIdx)
            }
            val dstIdx = rawDstIdx.toInt()
            for (preI in 0 until preDim) {
                val preSrcI = srcOffsets.start + (preI * srcDimSz + srcIdx) * postDim
                val preDstI = (preI * maxIdx + dstIdx) * postDim
                for (k in 0 until postDim) apply(preDstI + k, preSrcI + k)
            }
        }
    }
}

public fun copy2d(
    src: List<Double>,
    dst: MutableList<Double>,
    d1: Int,
    d2: Int,
    srcStride1: Int,
    dstStride1: Int,
    srcOffset: Int,
    dstOffset: Int,
) {
    if (srcStride1 == d2 && dstStride1 == d2) {
        for (i in 0 until d1 * d2) {
            dst[dstOffset + i] = src[srcOffset + i]
        }
        return
    }
    var srcIdx = srcOffset
    var dstIdx = dstOffset
    repeat(d1) {
        for (j in 0 until d2) {
            dst[dstIdx + j] = src[srcIdx + j]
        }
        srcIdx += srcStride1
        dstIdx += dstStride1
    }
}

public fun copyStridedSrc(
    src: List<Double>,
    dst: MutableList<Double>,
    dstOffset: Int,
    srcL: Layout,
) {
    when (val blocks = srcL.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            for (i in 0 until blocks.len) {
                dst[dstOffset + i] = src[blocks.startOffset + i]
            }
        }

        is StridedBlocks.UniformBlocks -> {
            copy2d(
                src,
                dst,
                blocks.count,
                blocks.blockLen,
                blocks.srcStride,
                blocks.blockLen,
                blocks.startOffset,
                dstOffset,
            )
        }

        is StridedBlocks.MultipleBlocks -> {
            if (blocks.blockLen == 1) {
                for ((i, srcIndex) in blocks.blockStartIndex.withIndex()) {
                    dst[dstOffset + i] = src[srcIndex]
                }
            } else {
                var dstIndex = dstOffset
                for (srcIndex in blocks.blockStartIndex) {
                    for (j in 0 until blocks.blockLen) {
                        dst[dstIndex + j] = src[srcIndex + j]
                    }
                    dstIndex += blocks.blockLen
                }
            }
        }
    }
}

public class Conv1D(
    public val params: ParamsConv1D,
) : Map2 {
    override val op: String = "conv1d"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val p = params
        val inp = v1.subList(l1.startOffset, v1.size)
        val k = v2.subList(l2.startOffset, v2.size)
        val (inpS0, inpS1, inpS2) = dims3(l1.stride)
        val (kS0, kS1, kS2) = dims3(l2.stride)
        val lOut = p.lOut()
        val dstElems = p.cOut * lOut * p.bSize
        val dst = MutableList(dstElems) { 0.0 }
        val inpCont = DoubleArray(p.bSize * p.cIn * p.lIn)
        for (bIdx in 0 until p.bSize) {
            for (srcL in 0 until p.lIn) {
                for (srcCIdx in 0 until p.cIn) {
                    val inpIdx = bIdx * inpS0 + srcCIdx * inpS1 + srcL * inpS2
                    inpCont[bIdx * p.lIn * p.cIn + srcL * p.cIn + srcCIdx] = inp[inpIdx]
                }
            }
        }
        for (offset in 0 until p.kSize) {
            for (dstCIdx in 0 until p.cOut) {
                val dstIdxBase = dstCIdx * lOut
                val kCont = DoubleArray(p.cIn) { cInIdx -> k[dstCIdx * kS0 + cInIdx * kS1 + offset * kS2] }
                for (bIdx in 0 until p.bSize) {
                    val dstIdxB = dstIdxBase + bIdx * p.cOut * lOut
                    for (dstL in 0 until lOut) {
                        val dstIdx = dstIdxB + dstL
                        val srcL = p.stride * dstL + offset * p.dilation
                        if (srcL < p.padding || srcL >= p.padding + p.lIn) continue
                        val srcLAdj = srcL - p.padding
                        val inpBase = bIdx * p.lIn * p.cIn + srcLAdj * p.cIn
                        var d = 0.0
                        for (i in 0 until p.cIn) {
                            d += inpCont[inpBase + i] * kCont[i]
                        }
                        dst[dstIdx] = dst[dstIdx] + d
                    }
                }
            }
        }
        return dst
    }
}

public class Im2Col1D(
    public val lK: Int,
    public val stride: Int,
    public val dilation: Int,
    public val padding: Int,
) : Map1 {
    public fun lOut(l: Int): Int = (l + 2 * padding - dilation * (lK - 1) - 1) / stride + 1

    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val (b, c, l) = layout.shape.dims3()
        val lOutVal = lOut(l)
        val src = vs.subList(layout.startOffset, vs.size)
        val dst = MutableList(b * lOutVal * c * lK) { 0.0 }
        val s = layout.stride
        val srcS0 = s[0]
        val srcS1 = s[1]
        val srcS2 = s[2]
        for (bIdx in 0 until b) {
            val srcIdxB = bIdx * srcS0
            val dstIdxB = bIdx * lOutVal * c * lK
            for (lIdx in 0 until lOutVal) {
                val dstIdxL = dstIdxB + lIdx * c * lK
                for (cIdx in 0 until c) {
                    val dstIdxC = dstIdxL + cIdx * lK
                    val srcIdxC = cIdx * srcS1 + srcIdxB
                    for (lKIdx in 0 until lK) {
                        val srcL = lIdx * stride + lKIdx * dilation
                        if (padding != 0 && (srcL < padding || srcL >= l + padding)) continue
                        val srcLAdj = srcL - padding
                        val srcIdx = srcIdxC + srcLAdj * srcS2
                        val dstIdx = dstIdxC + lKIdx
                        dst[dstIdx] = src[srcIdx]
                    }
                }
            }
        }
        return dst
    }
}

public class Im2Col(
    public val hK: Int,
    public val wK: Int,
    public val stride: Int,
    public val dilation: Int,
    public val padding: Int,
) : Map1 {
    public fun hwOut(
        h: Int,
        w: Int,
    ): Pair<Int, Int> {
        val hOut = (h + 2 * padding - dilation * (hK - 1) - 1) / stride + 1
        val wOut = (w + 2 * padding - dilation * (wK - 1) - 1) / stride + 1
        return Pair(hOut, wOut)
    }

    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val d = layout.shape.dims4()
        val b = d.d0
        val c = d.d1
        val h = d.d2
        val w = d.d3
        val (hOut, wOut) = hwOut(h, w)
        val src = vs.subList(layout.startOffset, vs.size)
        val dst = MutableList(b * hOut * wOut * c * hK * wK) { 0.0 }
        val s = layout.stride
        val srcS0 = s[0]
        val srcS1 = s[1]
        val srcS2 = s[2]
        val srcS3 = s[3]
        for (bIdx in 0 until b) {
            val srcIdxB = bIdx * srcS0
            val dstIdxB = bIdx * hOut * wOut * c * hK * wK
            for (hIdx in 0 until hOut) {
                val dstIdxH = dstIdxB + hIdx * wOut * c * hK * wK
                for (wIdx in 0 until wOut) {
                    val dstIdxW = dstIdxH + wIdx * c * hK * wK
                    for (cIdx in 0 until c) {
                        val dstIdxC = dstIdxW + cIdx * hK * wK
                        val srcIdxC = cIdx * srcS1 + srcIdxB
                        for (hKIdx in 0 until hK) {
                            val srcH = hIdx * stride + hKIdx * dilation
                            if (padding != 0 && (srcH < padding || srcH >= h + padding)) continue
                            val srcHAdj = srcH - padding
                            val srcIdxH = srcIdxC + srcHAdj * srcS2
                            val dstIdxHK = dstIdxC + hKIdx * wK
                            for (wKIdx in 0 until wK) {
                                val srcW = wIdx * stride + wKIdx * dilation
                                if (padding != 0 && (srcW < padding || srcW >= w + padding)) continue
                                val srcWAdj = srcW - padding
                                val srcIdx = srcIdxH + srcWAdj * srcS3
                                val dstIdx = dstIdxHK + wKIdx
                                dst[dstIdx] = src[srcIdx]
                            }
                        }
                    }
                }
            }
        }
        return dst
    }
}

public class Col2Im1D(
    public val stride: Int,
) : Map1 {
    override fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double> {
        val d = layout.shape.dims4()
        val bSize = d.d0
        val lIn = d.d1
        val cOut = d.d2
        val kSize = d.d3
        val lOut = (lIn - 1) * stride + kSize
        val im = MutableList(bSize * cOut * lOut) { 0.0 }
        val dstS0 = cOut * lOut
        val dstS1 = lOut
        val srcS0 = cOut * kSize * lIn
        val srcS1 = cOut * kSize
        val srcS2 = kSize
        for (lInI in 0 until lIn) {
            for (kI in 0 until kSize) {
                val lOutI = lInI * stride + kI
                for (bI in 0 until bSize) {
                    for (cI in 0 until cOut) {
                        val dstIdx = bI * dstS0 + cI * dstS1 + lOutI
                        val srcIdx = bI * srcS0 + lInI * srcS1 + cI * srcS2 + kI
                        im[dstIdx] = im[dstIdx] + vs[srcIdx]
                    }
                }
            }
        }
        return im
    }
}

public class ConvTranspose1D(
    public val params: ParamsConvTranspose1D,
) : Map2 {
    override val op: String = "conv_transpose1d"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val p = params
        val inp = v1.subList(l1.startOffset, v1.size)
        val k = v2.subList(l2.startOffset, v2.size)
        val (inpS0, inpS1, inpS2) = dims3(l1.stride)
        val (kS0, kS1, kS2) = dims3(l2.stride)
        val lOut = p.lOut()
        val dstElems = p.cOut * lOut * p.bSize
        val dst = MutableList(dstElems) { 0.0 }
        val dstS0 = p.cOut * lOut
        val dstS1 = lOut
        val dstS2 = 1
        val inpCont = DoubleArray(p.bSize * p.cIn * p.lIn)
        val contS0 = p.lIn * p.cIn
        val contS1 = p.cIn
        for (bIdx in 0 until p.bSize) {
            for (lIdx in 0 until p.lIn) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = bIdx * inpS0 + cIdx * inpS1 + lIdx * inpS2
                    val dstIdx = bIdx * contS0 + lIdx * contS1 + cIdx
                    inpCont[dstIdx] = inp[srcIdx]
                }
            }
        }
        for (kIdx in 0 until p.kSize) {
            for (dstCIdx in 0 until p.cOut) {
                val kCont = DoubleArray(p.cIn) { cInIdx -> k[cInIdx * kS0 + dstCIdx * kS1 + kIdx * kS2] }
                for (bIdx in 0 until p.bSize) {
                    for (lIdx in 0 until p.lIn) {
                        val outIdxRaw = lIdx * p.stride + kIdx * p.dilation
                        if (outIdxRaw < p.padding) continue
                        val outIdx = outIdxRaw - p.padding
                        if (outIdx < lOut) {
                            val inpBase = bIdx * contS0 + lIdx * contS1
                            val dstIdx = bIdx * dstS0 + outIdx * dstS2 + dstCIdx * dstS1
                            var d = 0.0
                            for (i in 0 until p.cIn) {
                                d += inpCont[inpBase + i] * kCont[i]
                            }
                            dst[dstIdx] = dst[dstIdx] + d
                        }
                    }
                }
            }
        }
        return dst
    }
}

public class ConvTranspose2D(
    public val params: ParamsConvTranspose2D,
) : Map2 {
    override val op: String = "conv_transpose2d"

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val p = params
        val inp = v1.subList(l1.startOffset, v1.size)
        val inpStrides = dims4(l1.stride)
        val inpS0 = inpStrides.d0
        val inpS1 = inpStrides.d1
        val inpS2 = inpStrides.d2
        val inpS3 = inpStrides.d3
        val k = v2.subList(l2.startOffset, v2.size)
        val kStrides = dims4(l2.stride)
        val kS0 = kStrides.d0
        val kS1 = kStrides.d1
        val kS2 = kStrides.d2
        val kS3 = kStrides.d3
        val outH = p.outH()
        val outW = p.outW()
        val dst = MutableList(p.bSize * p.cOut * outH * outW) { 0.0 }
        val dstS0 = p.cOut * outH * outW
        val dstS1 = outH * outW
        val dstS2 = outW
        val dstS3 = 1
        val inpCont = DoubleArray(p.bSize * p.cIn * p.iH * p.iW)
        val contS0 = p.iH * p.iW * p.cIn
        val contS1 = p.iW * p.cIn
        val contS2 = p.cIn
        for (bIdx in 0 until p.bSize) {
            for (hIdx in 0 until p.iH) {
                for (wIdx in 0 until p.iW) {
                    for (cIdx in 0 until p.cIn) {
                        val srcIdx = bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                        val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                        inpCont[dstIdx] = inp[srcIdx]
                    }
                }
            }
        }
        for (kY in 0 until p.kH) {
            for (kX in 0 until p.kW) {
                for (dstCIdx in 0 until p.cOut) {
                    val kCont = DoubleArray(p.cIn) { cInIdx -> k[cInIdx * kS0 + dstCIdx * kS1 + kY * kS2 + kX * kS3] }
                    for (bIdx in 0 until p.bSize) {
                        for (inpY in 0 until p.iH) {
                            for (inpX in 0 until p.iW) {
                                val outXRaw = inpX * p.stride + kX * p.dilation
                                val outYRaw = inpY * p.stride + kY * p.dilation
                                if (outXRaw < p.padding || outYRaw < p.padding) continue
                                val outX = outXRaw - p.padding
                                val outY = outYRaw - p.padding
                                if (outX < outW && outY < outH) {
                                    val inpBase = bIdx * contS0 + inpY * contS1 + inpX * contS2
                                    val dstIdx = bIdx * dstS0 + outY * dstS2 + outX * dstS3 + dstCIdx * dstS1
                                    var d = 0.0
                                    for (i in 0 until p.cIn) {
                                        d += inpCont[inpBase + i] * kCont[i]
                                    }
                                    dst[dstIdx] = dst[dstIdx] + d
                                }
                            }
                        }
                    }
                }
            }
        }
        return dst
    }
}

public class MatMul(
    public val bmnk: Dims4,
) : Map2 {
    override val op: String = "mat_mul"

    public fun stridingError(
        lhsL: Layout,
        rhsL: Layout,
        msg: String,
    ): CandleException =
        CandleException.MatMulUnexpectedStriding(
            MatMulUnexpectedStriding(
                lhsL = lhsL,
                rhsL = rhsL,
                bmnk = bmnk,
                msg = msg,
            ),
        )

    public fun abSkip(
        lhsL: Layout,
        rhsL: Layout,
    ): Pair<Int, Int> {
        val lhsStride = lhsL.stride
        val rhsStride = rhsL.stride
        val rank = lhsStride.size
        val m = bmnk.d1
        val n = bmnk.d2
        val k = bmnk.d3
        val lhsBatchStride = lhsStride.subList(0, rank - 2)
        val aSkip =
            when (lhsBatchStride.size) {
                2 -> {
                    val s1 = lhsBatchStride[0]
                    val stride = lhsBatchStride[1]
                    when {
                        s1 == stride * lhsL.dims[1] -> stride
                        lhsL.dims[0] == 1 -> stride
                        lhsL.dims[1] == 1 -> s1
                        else -> throw stridingError(lhsL, rhsL, "non-contiguous lhs")
                    }
                }

                1 -> {
                    lhsBatchStride[0]
                }

                0 -> {
                    m * k
                }

                else -> {
                    throw stridingError(lhsL, rhsL, "non-contiguous lhs")
                }
            }
        val rhsBatchStride = rhsStride.subList(0, rank - 2)
        val bSkip =
            when (rhsBatchStride.size) {
                2 -> {
                    val s1 = rhsBatchStride[0]
                    val stride = rhsBatchStride[1]
                    when {
                        s1 == stride * rhsL.dims[1] -> stride
                        rhsL.dims[0] == 1 -> stride
                        rhsL.dims[1] == 1 -> s1
                        else -> throw stridingError(lhsL, rhsL, "non-contiguous rhs")
                    }
                }

                1 -> {
                    rhsBatchStride[0]
                }

                0 -> {
                    n * k
                }

                else -> {
                    throw stridingError(lhsL, rhsL, "non-contiguous rhs")
                }
            }
        return Pair(aSkip, bSkip)
    }

    override fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double> {
        val b = bmnk.d0
        val m = bmnk.d1
        val n = bmnk.d2
        val k = bmnk.d3
        val lhs = v1.subList(l1.startOffset, v1.size)
        val rhs = v2.subList(l2.startOffset, v2.size)
        val lhsStride = l1.stride
        val rhsStride = l2.stride
        val rank = lhsStride.size
        val lhsCs = lhsStride[rank - 1]
        val lhsRs = lhsStride[rank - 2]
        val rhsCs = rhsStride[rank - 1]
        val rhsRs = rhsStride[rank - 2]
        val (aSkip, bSkip) = abSkip(l1, l2)
        val cSkip = m * n
        val dst = MutableList(b * m * n) { 0.0 }
        for (step in 0 until b) {
            val lhsBase = step * aSkip
            val rhsBase = step * bSkip
            val dstBase = step * cSkip
            for (mIdx in 0 until m) {
                for (nIdx in 0 until n) {
                    var sum = 0.0
                    for (kIdx in 0 until k) {
                        val lVal = lhs[lhsBase + mIdx * lhsRs + kIdx * lhsCs]
                        val rVal = rhs[rhsBase + kIdx * rhsRs + nIdx * rhsCs]
                        sum += lVal * rVal
                    }
                    dst[dstBase + mIdx * n + nIdx] = sum
                }
            }
        }
        return dst
    }

    public fun f(
        lhs: FloatArray,
        lhsL: Layout,
        rhs: FloatArray,
        rhsL: Layout,
    ): FloatArray {
        val res = f(lhs.map { it.toDouble() }, lhsL, rhs.map { it.toDouble() }, rhsL)
        return FloatArray(res.size) { res[it].toFloat() }
    }

    public fun f(
        lhs: DoubleArray,
        lhsL: Layout,
        rhs: DoubleArray,
        rhsL: Layout,
    ): DoubleArray {
        val res = f(lhs.toList(), lhsL, rhs.toList(), rhsL)
        return DoubleArray(res.size) { res[it] }
    }
}

public fun elu(
    v: Double,
    alpha: Double,
): Double = if (v >= 0.0) v else (exp(v) - 1.0) * alpha

internal val CandleException.bt: () -> CandleException get() = { this }

internal val Map1.self: Map1 get() = this

internal fun Map1.f(
    vs: CpuStorage,
    layout: Layout,
): List<Double> = this.f(vs.asSlice(), layout)

internal fun <T> Map1.Ok(v: T): T = v

internal fun Map1.Err(e: CandleException): Nothing = throw e

internal val Map1Any.self: Map1Any get() = this

internal fun Map1Any.f(
    vs: CpuStorage,
    layout: Layout,
    wrap: (List<Double>) -> CpuStorage,
): CpuStorage = this.f(vs.asSlice(), layout, wrap)

internal fun <T> Map1Any.Ok(v: T): T = v

internal fun Map1Any.Err(e: CandleException): Nothing = throw e

internal val Map2.self: Map2 get() = this

internal fun Map2.f(
    v1: CpuStorage,
    l1: Layout,
    v2: CpuStorage,
    l2: Layout,
): List<Double> = this.f(v1.asSlice(), l1, v2.asSlice(), l2)

internal val Map2.OP: String get() = op

internal fun <T> Map2.Ok(v: T): T = v

internal fun Map2.Err(e: CandleException): Nothing = throw e

internal val Map2InPlace.self: Map2InPlace get() = this

internal fun Map2InPlace.f(
    v1: CpuStorage,
    l1: Layout,
    v2: CpuStorage,
    l2: Layout,
) {
    v1.applyInPlace(this, l1, v2, l2)
}

internal val Map2InPlace.OP: String get() = op

internal fun <T> Map2InPlace.Ok(v: T): T = v

internal fun Map2InPlace.Err(e: CandleException): Nothing = throw e

internal val Map2U8.self: Map2U8 get() = this

internal fun Map2U8.f(
    v1: CpuStorage,
    l1: Layout,
    v2: CpuStorage,
    l2: Layout,
): UByteArray = this.f(v1.asSlice(), l1, v2.asSlice(), l2)

internal val Map2U8.OP: String get() = op

internal fun <T> Map2U8.Ok(v: T): T = v

internal fun Map2U8.Err(e: CandleException): Nothing = throw e

internal object Vec {
    fun <T> new(): VecList<T> = VecList(0)

    fun <T> withCapacity(cap: Int): VecList<T> = VecList(cap)

    val tryFrom1: (Tensor) -> CandleResult<List<Any?>> = { tensor ->
        CandleResult.Ok(tensor.toVec1<Any?>())
    }
    val tryFrom2: (Tensor) -> CandleResult<List<List<Any?>>> = { tensor ->
        CandleResult.Ok(tensor.toVec2<Any?>())
    }
    val tryFrom3: (Tensor) -> CandleResult<List<List<List<Any?>>>> = { tensor ->
        CandleResult.Ok(tensor.toVec3<Any?>())
    }
}

internal fun <T> List<T>.getUnchecked(i: Int): T = this[i]

internal object Bf16Const {
    const val ZERO: Short = 0
    val ONE: Short get() = Float16Utils.floatToBF16Bits(1.0f)
}

internal val bf16: Bf16Const = Bf16Const

internal object F16Const {
    const val ZERO: Short = 0
    val ONE: Short get() = Float16Utils.floatToF16Bits(1.0f)
}

internal val f16: F16Const = F16Const

internal object F8e4m3Const {
    const val ZERO: Byte = 0
    const val ONE: Byte = 1
}

internal val f8e4m3: F8e4m3Const = F8e4m3Const

internal fun panic(msg: String): Nothing = throw IllegalStateException(msg)

internal fun Any.toScalar(): Scalar =
    when (this) {
        is UByte -> Scalar.U8(this)
        is UInt -> Scalar.U32(this)
        is Short -> Scalar.I16(this)
        is Int -> Scalar.I32(this)
        is Long -> Scalar.I64(this)
        is Float -> Scalar.F32(this)
        is Double -> Scalar.F64(this)
        is Byte -> Scalar.F8E4M3(this)
        is Scalar -> this
        else -> Scalar.F64(0.0)
    }

internal fun Scalar.U8.toDouble(): Double = v.toDouble()

internal fun Scalar.U32.toDouble(): Double = v.toDouble()

internal fun Scalar.I16.toDouble(): Double = v.toDouble()

internal fun Scalar.I32.toDouble(): Double = v.toDouble()

internal fun Scalar.I64.toDouble(): Double = v.toDouble()

internal fun Scalar.F32.toDouble(): Double = v.toDouble()

internal fun <T : Any> Tensor.Companion.new(
    v: T,
    dev: Device,
): CandleResult<Tensor> {
    val t: Tensor =
        when (v) {
            is Tensor -> v
            is Double -> Tensor.new(doubleArrayOf(v), Shape(), dev)
            is Float -> Tensor.new(floatArrayOf(v), Shape(), dev)
            is Int -> Tensor.new(intArrayOf(v), Shape(), dev)
            is Long -> Tensor.new(longArrayOf(v), Shape(), dev)
            else -> Tensor.new(doubleArrayOf((v as Number).toDouble()), Shape(), dev)
        }
    return CandleResult.Ok(t)
}

internal fun <T> MutableList<T>.swap(
    i: Int,
    j: Int,
) {
    val tmp = this[i]
    this[i] = this[j]
    this[j] = tmp
}

internal fun Shape.into(): Shape = this

internal fun Shape.clone(): Shape = this

internal fun <T> List<T>.clone(): List<T> = this.toList()

internal fun <T> List<T>.len(): Int = this.size

internal fun <T> List<T>.toVec(): MutableList<T> = this.toMutableList()

internal fun <T> Sequence<T>.rev(): List<T> = this.toList().reversed()

internal fun <T> Sequence<T>.enumerate(): Sequence<IndexedValue<T>> = this.withIndex()

internal fun <T, R> Sequence<T>.zip(other: List<R>): Sequence<Pair<T, R>> = this.zip(other.asSequence())

internal fun IntRange.rev(): IntProgression = this.reversed()

internal fun Int.toIndex(
    shape: Shape,
    op: String,
): Int = if (this < 0 || this >= shape.dims.size) throw CandleException.DimOutOfRange(shape, this, op) else this

internal fun <T> Sequence<T>.skip(n: Int): Sequence<T> = this.drop(n)

internal fun Sequence<Int>.product(): Int = this.fold(1) { acc, v -> acc * v }

internal fun debugAssert(
    cond: Boolean,
    msg: String = "",
    a: Int = 0,
    b: Int = 0,
) {
    require(cond) { "$msg: $a, $b" }
}

internal fun <T> debugAssertEq(
    a: T,
    b: T,
) {
    require(a == b)
}

internal fun <T> Some(v: T): T = v

internal fun <T> Iterator<T>.collect(): List<T> = this.asSequence().toList()

internal fun <T> vec(vararg elements: T): List<T> = elements.toList()

internal fun NdIter.nextOrNull(): IntArray? = if (this.hasNext()) this.next() else null

internal fun NdIter.len(): Int = this.remaining

internal fun Layout.Companion.contiguous(): Layout = contiguous(Shape())

internal fun Layout.Companion.contiguous(vararg dims: Int): Layout = contiguous(Shape(*dims))

internal fun Layout.Companion.contiguousWithOffset(
    dim: Int,
    startOffset: Int,
): Layout = contiguousWithOffset(Shape(dim), startOffset)

internal fun NdIter.Companion.new(vararg layouts: Layout): NdIter = new(layouts.toList())

internal fun IntArray.toVec(): List<Int> = this.toList()

internal fun assertEq(
    actual: Any?,
    expected: Any?,
) {
    if (actual is IntArray && expected is IntArray) {
        require(actual.contentEquals(expected))
    } else if (actual is List<*> && expected is List<*> && actual.size == expected.size &&
        actual.indices.all { i -> actual[i] is IntArray && expected[i] is IntArray }
    ) {
        require(actual.indices.all { i -> (actual[i] as IntArray).contentEquals(expected[i] as IntArray) })
    } else {
        require(actual == expected)
    }
}

internal fun Int.into(): Shape = Shape(listOf(this))

internal fun Pair<Int, Int>.into(): Shape = Shape(listOf(first, second))

internal fun Triple<Int, Int, Int>.into(): Shape = Shape(listOf(first, second, third))

internal fun List<Int>.into(): Shape = Shape(this)

internal fun <D> D.toIndex(
    shape: Shape,
    op: String,
): Int =
    when (this) {
        is Dim -> this.toIndex(shape, op)
        is Int -> Shape.toIndex(this, shape, op)
        else -> Shape.toIndex(0, shape, op)
    }

internal fun List<Int>.scan(
    init: Int,
    f: (IntArray, Int) -> Int,
): Sequence<Int> =
    sequence {
        val state = intArrayOf(init)
        for (u in this@scan) {
            yield(f(state, u))
        }
    }

internal fun <T> Sequence<T>.collect(): MutableList<T> = this.toMutableList()

internal fun <T> List<T>.extend(other: List<T>): List<T> = this + other

internal fun Int.isMultipleOf(other: Int): Boolean = this % other == 0

internal fun write(
    f: String,
    fmt: String,
    v: Any?,
): String = if (fmt.isEmpty()) f else f + v.toString()

internal fun write(
    f: Formatter,
    fmt: String,
    v: Any?,
): CandleResult<Unit> = if (fmt.isEmpty() && v == null) CandleResult.Ok(Unit) else CandleResult.Ok(Unit)

internal fun <T> List<List<T>>.concat(): List<T> = this.flatten()

internal fun Shape.Companion.Self(
    dims: List<Int>,
    u: Unit,
): Shape = if (u == Unit) Shape(dims) else Shape(dims)

internal fun Ok(): Unit = Unit

internal fun <T> Ok(v: T): T = v

internal fun <T> Ok(
    v: T,
    a: Any?,
): T = if (a == null) v else v

internal fun <T> Ok(
    v: T,
    a: Any?,
    b: Any?,
): T = if (a == null && b == null) v else v

internal val <T> WithDType<T>.self: WithDType<T> get() = this

internal annotation class Test

internal fun assertEquals(
    expected: Any?,
    actual: Any?,
) = assertEq(actual, expected)

internal val DType.Companion.U8: DType get() = DType.U8
internal val DType.Companion.U32: DType get() = DType.U32
internal val DType.Companion.I16: DType get() = DType.I16
internal val DType.Companion.I32: DType get() = DType.I32
internal val DType.Companion.I64: DType get() = DType.I64
internal val DType.Companion.BF16: DType get() = DType.BF16
internal val DType.Companion.F16: DType get() = DType.F16
internal val DType.Companion.F32: DType get() = DType.F32
internal val DType.Companion.F64: DType get() = DType.F64
internal val DType.Companion.F8E4M3: DType get() = DType.F8E4M3
internal val DType.Companion.F6E2M3: DType get() = DType.F6E2M3
internal val DType.Companion.F6E3M2: DType get() = DType.F6E3M2
internal val DType.Companion.F4: DType get() = DType.F4
internal val DType.Companion.F8E8M0: DType get() = DType.F8E8M0
