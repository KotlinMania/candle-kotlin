// port-lint: source cpu_backend/mod.rs
package io.github.kotlinmania.candle

import kotlin.math.PI
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
import kotlin.random.Random

public sealed class CpuStorage {
    public data class U8(
        val data: UByteArray,
    ) : CpuStorage()

    public data class U32(
        val data: UIntArray,
    ) : CpuStorage()

    public data class I16(
        val data: ShortArray,
    ) : CpuStorage()

    public data class I32(
        val data: IntArray,
    ) : CpuStorage()

    public data class I64(
        val data: LongArray,
    ) : CpuStorage()

    public data class BF16(
        val data: ShortArray,
    ) : CpuStorage()

    public data class F16(
        val data: ShortArray,
    ) : CpuStorage()

    public data class F32(
        val data: FloatArray,
    ) : CpuStorage()

    public data class F64(
        val data: DoubleArray,
    ) : CpuStorage()

    public data class F8E4M3(
        val data: ByteArray,
    ) : CpuStorage()

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
        }
    }

    public fun constSet(v: Scalar, layout: Layout) {
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
        }
    }

    public fun affine(layout: Layout, mul: Double, add: Double): CpuStorage {
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

    public fun powf(layout: Layout, e: Double): CpuStorage {
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

    public fun elu(layout: Layout, alpha: Double): CpuStorage {
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

    public fun unaryImpl(layout: Layout, op: UnaryOp): CpuStorage {
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

    public fun toDType(layout: Layout, target: DType): CpuStorage {
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
            DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
                bail("unsupported dtype $target")
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
        val srcShape = layout.shape
        val srcDims = srcShape.dims
        val outDims = ArrayList<Int>()
        for (i in srcDims.indices) {
            if (i in reduceDims) {
                outDims.add(1)
            } else {
                outDims.add(srcDims[i])
            }
        }
        val outShape = Shape(outDims)
        val outCount = outShape.elemCount()
        val outLayout = Layout.contiguous(outShape)

        val outData = FloatArray(outCount)
        when (op) {
            ReduceOp.Sum -> {
                val strided = layout.stridedIndex()
                var srcIdx = 0
                while (strided.hasNext()) {
                    val sIdx = strided.next()
                    val outIdx = getReducedIndex(srcIdx++, srcDims, reduceDims)
                    outData[outIdx] += getAsFloat(sIdx)
                }
            }
            ReduceOp.Min -> {
                outData.fill(Float.POSITIVE_INFINITY)
                val strided = layout.stridedIndex()
                var srcIdx = 0
                while (strided.hasNext()) {
                    val sIdx = strided.next()
                    val outIdx = getReducedIndex(srcIdx++, srcDims, reduceDims)
                    val v = getAsFloat(sIdx)
                    if (v < outData[outIdx]) outData[outIdx] = v
                }
            }
            ReduceOp.Max -> {
                outData.fill(Float.NEGATIVE_INFINITY)
                val strided = layout.stridedIndex()
                var srcIdx = 0
                while (strided.hasNext()) {
                    val sIdx = strided.next()
                    val outIdx = getReducedIndex(srcIdx++, srcDims, reduceDims)
                    val v = getAsFloat(sIdx)
                    if (v > outData[outIdx]) outData[outIdx] = v
                }
            }
            ReduceOp.ArgMin, ReduceOp.ArgMax -> {
                // ArgMin/ArgMax along single dimension
                val dim = reduceDims[0]
                val dimSize = srcDims[dim]
                val argOut = UIntArray(outCount)
                val bestVal = FloatArray(outCount) { if (op == ReduceOp.ArgMin) Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY }
                val strided = layout.stridedIndex()
                var srcIdx = 0
                while (strided.hasNext()) {
                    val sIdx = strided.next()
                    val dimPos = (srcIdx / getDimStride(srcIdx, srcDims, dim)) % dimSize
                    val outIdx = getReducedIndex(srcIdx++, srcDims, reduceDims)
                    val v = getAsFloat(sIdx)
                    if (op == ReduceOp.ArgMin && v < bestVal[outIdx]) {
                        bestVal[outIdx] = v
                        argOut[outIdx] = dimPos.toUInt()
                    } else if (op == ReduceOp.ArgMax && v > bestVal[outIdx]) {
                        bestVal[outIdx] = v
                        argOut[outIdx] = dimPos.toUInt()
                    }
                }
                return U32(argOut)
            }
        }
        return F32(outData).toDType(outLayout, dtype())
    }

    private fun getDimStride(srcIdx: Int, dims: List<Int>, targetDim: Int): Int {
        var stride = 1
        for (i in (dims.size - 1) downTo (targetDim + 1)) {
            stride *= dims[i]
        }
        return stride
    }

    private fun getReducedIndex(srcFlatIdx: Int, dims: List<Int>, reduceDims: List<Int>): Int {
        var remaining = srcFlatIdx
        var outIdx = 0
        var outStride = 1
        for (i in dims.indices.reversed()) {
            val dimSize = dims[i]
            val coord = remaining % dimSize
            remaining /= dimSize
            val outCoord = if (i in reduceDims) 0 else coord
            val outDimSize = if (i in reduceDims) 1 else dimSize
            outIdx += outCoord * outStride
            outStride *= outDimSize
        }
        return outIdx
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
        }

    public fun setDouble(index: Int, value: Double) {
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
        }
    }

    private fun applyUnaryFloat(v: Float, op: UnaryOp): Float =
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

    private fun applyUnaryDouble(v: Double, op: UnaryOp): Double =
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
        }
    }

    private fun copy2dByteArray(src: UByteArray, dst: UByteArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dUIntArray(src: UIntArray, dst: UIntArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dShortArray(src: ShortArray, dst: ShortArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dIntArray(src: IntArray, dst: IntArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dLongArray(src: LongArray, dst: LongArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dFloatArray(src: FloatArray, dst: FloatArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dDoubleArray(src: DoubleArray, dst: DoubleArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copy2dRawByteArray(src: ByteArray, dst: ByteArray, d1: Int, d2: Int, srcStride1: Int, dstStride1: Int, srcOffset: Int, dstOffset: Int) {
        if (srcStride1 == d2 && dstStride1 == d2) {
            src.copyInto(dst, dstOffset, srcOffset, srcOffset + d1 * d2)
            return
        }
        var srcIdx = srcOffset
        var dstIdx = dstOffset
        for (i in 0 until d1) {
            src.copyInto(dst, dstIdx, srcIdx, srcIdx + d2)
            srcIdx += srcStride1
            dstIdx += dstStride1
        }
    }

    private fun copyStridedSrcByteArray(src: UByteArray, dst: UByteArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dByteArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcUIntArray(src: UIntArray, dst: UIntArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dUIntArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcShortArray(src: ShortArray, dst: ShortArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dShortArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcIntArray(src: IntArray, dst: IntArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dIntArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcLongArray(src: LongArray, dst: LongArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dLongArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcFloatArray(src: FloatArray, dst: FloatArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dFloatArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcDoubleArray(src: DoubleArray, dst: DoubleArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dDoubleArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
                        src.copyInto(dst, curDstIndex, srcIndex, srcIndex + blockLen)
                        curDstIndex += blockLen
                    }
                }
            }
        }
    }

    private fun copyStridedSrcRawByteArray(src: ByteArray, dst: ByteArray, dstOffset: Int, srcLayout: Layout) {
        when (val blocks = srcLayout.stridedBlocks()) {
            is StridedBlocks.SingleBlock -> src.copyInto(dst, dstOffset, blocks.startOffset, blocks.startOffset + blocks.len)
            is StridedBlocks.UniformBlocks -> copy2dRawByteArray(src, dst, blocks.count, blocks.blockLen, blocks.srcStride, blocks.blockLen, blocks.startOffset, dstOffset)
            is StridedBlocks.MultipleBlocks -> {
                val blockStartIndex = blocks.blockStartIndex
                val blockLen = blocks.blockLen
                if (blockLen == 1) {
                    for (i in blockStartIndex.indices) dst[dstOffset + i] = src[blockStartIndex[i]]
                } else {
                    var curDstIndex = dstOffset
                    for (srcIndex in blockStartIndex) {
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
        val b = params.bSize
        val m = params.lOut()
        val n = params.cOut
        val k = params.kSize * params.cIn
        val colL = Layout.contiguous(Shape(b, m, k))
        val kernelTransposed = kernelL.transpose(1, 2)
        val kernelBroadcast = kernelTransposed.broadcastAs(Shape(b, k, n))
        return matmul(kernel, listOf(b, m, n, k), colL, kernelBroadcast)
    }

    public fun convTranspose1d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): CpuStorage {
        val b = params.bSize
        val m = params.lOut()
        val n = params.cOut
        val k = params.kSize * params.cIn
        val colL = Layout.contiguous(Shape(b, m, k))
        val kernelTransposed = kernelL.transpose(1, 2)
        val kernelBroadcast = kernelTransposed.broadcastAs(Shape(b, k, n))
        return matmul(kernel, listOf(b, m, n, k), colL, kernelBroadcast)
    }

    public fun conv2d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): CpuStorage {
        val b = params.bSize
        val m = params.outH() * params.outW()
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val colL = Layout.contiguous(Shape(b, m, k))
        val kernelTransposed = kernelL.transpose(1, 2)
        val kernelBroadcast = kernelTransposed.broadcastAs(Shape(b, k, n))
        return matmul(kernel, listOf(b, m, n, k), colL, kernelBroadcast)
    }

    public fun convTranspose2d(
        l: Layout,
        kernel: CpuStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): CpuStorage {
        val b = params.bSize
        val m = params.outH() * params.outW()
        val n = params.cOut
        val k = params.kH * params.kW * params.cIn
        val colL = Layout.contiguous(Shape(b, m, k))
        val kernelTransposed = kernelL.transpose(1, 2)
        val kernelBroadcast = kernelTransposed.broadcastAs(Shape(b, k, n))
        return matmul(kernel, listOf(b, m, n, k), colL, kernelBroadcast)
    }

    public fun avgPool2d(
        layout: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): CpuStorage {
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (bSz, c, h, w) = layout.shape.dims4()
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
        val (bSz, c, h, w) = layout.shape.dims4()
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

    public fun upsampleNearest1d(layout: Layout, dstSz: Int): CpuStorage {
        val (bSz, c, srcSz) = layout.shape.dims3()
        val s = layout.stride
        val strideSz = s[2]
        val srcIndex = layout.startOffset
        val scaleSz = srcSz.toDouble() / dstSz.toDouble()
        val count = bSz * c * dstSz
        val dst = FloatArray(count)
        val srcIdxs = IntArray(dstSz) { idx ->
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

    public fun upsampleNearest2d(layout: Layout, dstH: Int, dstW: Int): CpuStorage {
        val (bSz, c, srcH, srcW) = layout.shape.dims4()
        val s = layout.stride
        val (strideH, strideW) = Pair(s[2], s[3])
        val srcIndex = layout.startOffset
        val scaleH = srcH.toDouble() / dstH.toDouble()
        val scaleW = srcW.toDouble() / dstW.toDouble()
        val count = bSz * c * dstH * dstW
        val dst = FloatArray(count)
        val srcHIdxs = IntArray(dstH) { hIdx ->
            minOf(srcH - 1, (hIdx.toDouble() * scaleH).toInt())
        }
        val srcWIdxs = IntArray(dstW) { wIdx ->
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
        val (batch, channels, heightIn, widthIn) = layout.shape.dims4()
        val heightOut = targetH
        val widthOut = targetW

        if (heightIn == heightOut && widthIn == widthOut) {
            return tryClone(layout)
        }

        val s = layout.stride
        val srcOffset = layout.startOffset

        val scaleH = if (alignCorners) {
            if (heightOut > 1) (heightIn - 1).toDouble() / (heightOut - 1).toDouble() else 0.0
        } else {
            scaleHFactor?.let { 1.0 / it } ?: (heightIn.toDouble() / heightOut.toDouble())
        }

        val scaleW = if (alignCorners) {
            if (widthOut > 1) (widthIn - 1).toDouble() / (widthOut - 1).toDouble() else 0.0
        } else {
            scaleWFactor?.let { 1.0 / it } ?: (widthIn.toDouble() / widthOut.toDouble())
        }

        val hIndices = Array(heightOut) { hOut ->
            val srcH = if (alignCorners) scaleH * hOut.toDouble() else scaleH * (hOut.toDouble() + 0.5) - 0.5
            val srcHClamped = maxOf(0.0, srcH)
            val h0 = srcHClamped.toInt()
            val h1 = minOf(heightIn - 1, h0 + 1)
            val weightH = (srcHClamped - h0.toDouble()).coerceIn(0.0, 1.0)
            Triple(h0, h1, weightH)
        }

        val wIndices = Array(widthOut) { wOut ->
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
    ) {
        val srcOffsets = sourceL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-set")
        val idsOffsets = indexesL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-set")
        val lOffsets = l.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-set")
        val count = idsOffsets.end - idsOffsets.start
        for (i in 0 until count) {
            val idx = indexes.getAsDouble(idsOffsets.start + i).toInt()
            val srcVal = source.getAsDouble(srcOffsets.start + i)
            setDouble(lOffsets.start + idx, srcVal)
        }
    }

    public fun scatterAddSet(
        l: Layout,
        indexes: CpuStorage,
        indexesL: Layout,
        source: CpuStorage,
        sourceL: Layout,
        d: Int,
    ) {
        val srcOffsets = sourceL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-add")
        val idsOffsets = indexesL.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-add")
        val lOffsets = l.contiguousOffsets() ?: throw CandleException.RequiresContiguous("scatter-add")
        val count = idsOffsets.end - idsOffsets.start
        for (i in 0 until count) {
            val idx = indexes.getAsDouble(idsOffsets.start + i).toInt()
            val cur = getAsDouble(lOffsets.start + idx)
            val srcVal = source.getAsDouble(srcOffsets.start + i)
            setDouble(lOffsets.start + idx, cur + srcVal)
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
        val res = tryClone(l)
        res.scatterAddSet(l, indexes, indexesL, source, sourceL, d)
        return res
    }
}

public object CpuDevice {
    public fun location(): DeviceLocation = DeviceLocation.Cpu

    public fun sameDevice(other: CpuDevice): Boolean = true

    public fun storageFromSlice(s: FloatArray): CpuStorage = CpuStorage.F32(s.copyOf())
    public fun storageFromSlice(s: DoubleArray): CpuStorage = CpuStorage.F64(s.copyOf())
    public fun storageFromSlice(s: IntArray): CpuStorage = CpuStorage.I32(s.copyOf())
    public fun storageFromSlice(s: LongArray): CpuStorage = CpuStorage.I64(s.copyOf())
    public fun storageFromSlice(s: ShortArray): CpuStorage = CpuStorage.I16(s.copyOf())
    public fun storageFromSlice(s: UIntArray): CpuStorage = CpuStorage.U32(s.copyOf())
    public fun storageFromSlice(s: UByteArray): CpuStorage = CpuStorage.U8(s.copyOf())

    public fun storageFromCpuStorage(s: CpuStorage): CpuStorage = s

    public fun storageFromCpuStorageOwned(s: CpuStorage): CpuStorage = s

    public fun new(ordinal: Int = 0): CpuDevice = this

    public fun setSeed(seed: ULong) {
        bail("cannot seed the CPU rng with set_seed")
    }

    public fun getCurrentSeed(): ULong {
        bail("cannot get the CPU rng seed with get_current_seed")
    }

    public fun randUniform(shape: Shape, dtype: DType, min: Double, max: Double): CpuStorage {
        val elemCount = shape.elemCount()
        val diff = max - min
        return when (dtype) {
            DType.BF16 -> {
                val data = ShortArray(elemCount) {
                    Float16Utils.floatToBF16Bits((min + Random.nextDouble() * diff).toFloat())
                }
                CpuStorage.BF16(data)
            }
            DType.F16 -> {
                val data = ShortArray(elemCount) {
                    Float16Utils.floatToF16Bits((min + Random.nextDouble() * diff).toFloat())
                }
                CpuStorage.F16(data)
            }
            DType.F32 -> {
                val data = FloatArray(elemCount) {
                    (min + Random.nextDouble() * diff).toFloat()
                }
                CpuStorage.F32(data)
            }
            DType.F64 -> {
                val data = DoubleArray(elemCount) {
                    min + Random.nextDouble() * diff
                }
                CpuStorage.F64(data)
            }
            DType.F8E4M3 -> {
                val data = ByteArray(elemCount) {
                    (min + Random.nextDouble() * diff).toInt().toByte()
                }
                CpuStorage.F8E4M3(data)
            }
            else -> bail("Unsupported dtype for randUniform $dtype")
        }
    }

    public fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): CpuStorage {
        val elemCount = shape.elemCount()
        val normalValues = DoubleArray(elemCount)
        var i = 0
        while (i < elemCount) {
            val u1 = Random.nextDouble().coerceAtLeast(1e-15)
            val u2 = Random.nextDouble()
            val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
            val z1 = sqrt(-2.0 * ln(u1)) * sin(2.0 * PI * u2)
            normalValues[i++] = mean + std * z0
            if (i < elemCount) {
                normalValues[i++] = mean + std * z1
            }
        }
        return when (dtype) {
            DType.BF16 -> {
                val data = ShortArray(elemCount) { Float16Utils.floatToBF16Bits(normalValues[it].toFloat()) }
                CpuStorage.BF16(data)
            }
            DType.F16 -> {
                val data = ShortArray(elemCount) { Float16Utils.floatToF16Bits(normalValues[it].toFloat()) }
                CpuStorage.F16(data)
            }
            DType.F32 -> {
                val data = FloatArray(elemCount) { normalValues[it].toFloat() }
                CpuStorage.F32(data)
            }
            DType.F64 -> {
                CpuStorage.F64(normalValues)
            }
            DType.F8E4M3 -> {
                val data = ByteArray(elemCount) { normalValues[it].toInt().toByte() }
                CpuStorage.F8E4M3(data)
            }
            else -> bail("Unsupported dtype for randNormal $dtype")
        }
    }

    public fun allocUninit(shape: Shape, dtype: DType): CpuStorage = zerosImpl(shape, dtype)

    public fun zerosImpl(shape: Shape, dtype: DType): CpuStorage {
        val elemCount = shape.elemCount()
        return when (dtype) {
            DType.U8 -> CpuStorage.U8(UByteArray(elemCount))
            DType.U32 -> CpuStorage.U32(UIntArray(elemCount))
            DType.I16 -> CpuStorage.I16(ShortArray(elemCount))
            DType.I32 -> CpuStorage.I32(IntArray(elemCount))
            DType.I64 -> CpuStorage.I64(LongArray(elemCount))
            DType.BF16 -> CpuStorage.BF16(ShortArray(elemCount))
            DType.F16 -> CpuStorage.F16(ShortArray(elemCount))
            DType.F32 -> CpuStorage.F32(FloatArray(elemCount))
            DType.F64 -> CpuStorage.F64(DoubleArray(elemCount))
            DType.F8E4M3 -> CpuStorage.F8E4M3(ByteArray(elemCount))
            else -> bail("Unsupported dtype for zeros $dtype")
        }
    }

    public fun synchronize() {}
}

