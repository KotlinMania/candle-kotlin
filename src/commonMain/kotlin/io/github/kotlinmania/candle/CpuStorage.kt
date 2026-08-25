// port-lint: source candle-core/src/cpu_backend/mod.rs
package io.github.kotlinmania.candle

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
}
