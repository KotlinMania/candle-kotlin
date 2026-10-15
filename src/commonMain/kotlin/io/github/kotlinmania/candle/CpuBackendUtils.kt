// port-lint: source cpu_backend/utils.rs
package io.github.kotlinmania.candle

/**
 * Helper functions to write CPU kernels.
 */
public interface Map1 {
    public fun f(vs: List<Double>, layout: Layout): List<Double>

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage =
        when (vs) {
            is CpuStorage.U8 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.U8(UByteArray(res.size) { res[it].toInt().toUByte() })
            }
            is CpuStorage.U32 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.U32(UIntArray(res.size) { res[it].toLong().toUInt() })
            }
            is CpuStorage.I16 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.I16(ShortArray(res.size) { res[it].toInt().toShort() })
            }
            is CpuStorage.I32 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.I32(IntArray(res.size) { res[it].toInt() })
            }
            is CpuStorage.I64 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.I64(LongArray(res.size) { res[it].toLong() })
            }
            is CpuStorage.BF16 -> {
                val res = f(FloatArray(vs.data.size) { bfloat16BitsToFloat(vs.data[it]) }.map { it.toDouble() }, layout)
                CpuStorage.BF16(UShortArray(res.size) { floatToBFloat16Bits(res[it].toFloat()) })
            }
            is CpuStorage.F16 -> {
                val res = f(FloatArray(vs.data.size) { halfBitsToFloat(vs.data[it]) }.map { it.toDouble() }, layout)
                CpuStorage.F16(UShortArray(res.size) { floatToHalfBits(res[it].toFloat()) })
            }
            is CpuStorage.F32 -> {
                val res = f(vs.data.map { it.toDouble() }, layout)
                CpuStorage.F32(FloatArray(res.size) { res[it].toFloat() })
            }
            is CpuStorage.F64 -> {
                val res = f(vs.data.toList(), layout)
                CpuStorage.F64(DoubleArray(res.size) { res[it] })
            }
            is CpuStorage.F8E4M3 -> {
                val res = f(FloatArray(vs.data.size) { f8e4m3ToFloat(vs.data[it]) }.map { it.toDouble() }, layout)
                CpuStorage.F8E4M3(ByteArray(res.size) { floatToF8e4m3(res[it].toFloat()) })
            }
            // Dummy types don't support Map1 operations
            is CpuStorage.F6E2M3 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F6E3M2 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F4 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F8E8M0 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
        }
}

public interface Map1Any {
    public fun f(
        vs: List<Double>,
        layout: Layout,
        wrap: (List<Double>) -> CpuStorage,
    ): CpuStorage

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage =
        when (vs) {
            is CpuStorage.U8 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.U8(UByteArray(res.size) { res[it].toInt().toUByte() }) }
            is CpuStorage.U32 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.U32(UIntArray(res.size) { res[it].toLong().toUInt() }) }
            is CpuStorage.I16 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.I16(ShortArray(res.size) { res[it].toInt().toShort() }) }
            is CpuStorage.I32 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.I32(IntArray(res.size) { res[it].toInt() }) }
            is CpuStorage.I64 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.I64(LongArray(res.size) { res[it].toLong() }) }
            is CpuStorage.BF16 -> f(
                FloatArray(vs.data.size) { bfloat16BitsToFloat(vs.data[it]) }.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.BF16(UShortArray(res.size) { floatToBFloat16Bits(res[it].toFloat()) }) }
            is CpuStorage.F16 -> f(
                FloatArray(vs.data.size) { halfBitsToFloat(vs.data[it]) }.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.F16(UShortArray(res.size) { floatToHalfBits(res[it].toFloat()) }) }
            is CpuStorage.F32 -> f(
                vs.data.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.F32(FloatArray(res.size) { res[it].toFloat() }) }
            is CpuStorage.F64 -> f(
                vs.data.toList(),
                layout,
            ) { res -> CpuStorage.F64(DoubleArray(res.size) { res[it] }) }
            is CpuStorage.F8E4M3 -> f(
                FloatArray(vs.data.size) { f8e4m3ToFloat(vs.data[it]) }.map { it.toDouble() },
                layout,
            ) { res -> CpuStorage.F8E4M3(ByteArray(res.size) { floatToF8e4m3(res[it].toFloat()) }) }
            // Dummy types don't support Map1Any operations
            is CpuStorage.F6E2M3 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F6E3M2 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F4 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F8E8M0 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
        }
}

public interface Map2 {
    public val op: String

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): List<Double>

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage =
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.U8(UByteArray(res.size) { res[it].toInt().toUByte() })
            }
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.U32(UIntArray(res.size) { res[it].toLong().toUInt() })
            }
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.I16(ShortArray(res.size) { res[it].toInt().toShort() })
            }
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.I32(IntArray(res.size) { res[it].toInt() })
            }
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.I64(LongArray(res.size) { res[it].toLong() })
            }
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                val d1 = FloatArray(v1.data.size) { bfloat16BitsToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { bfloat16BitsToFloat(v2.data[it]) }.map { it.toDouble() }
                val res = f(d1, l1, d2, l2)
                CpuStorage.BF16(UShortArray(res.size) { floatToBFloat16Bits(res[it].toFloat()) })
            }
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                val d1 = FloatArray(v1.data.size) { halfBitsToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { halfBitsToFloat(v2.data[it]) }.map { it.toDouble() }
                val res = f(d1, l1, d2, l2)
                CpuStorage.F16(UShortArray(res.size) { floatToHalfBits(res[it].toFloat()) })
            }
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> {
                val res = f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2)
                CpuStorage.F32(FloatArray(res.size) { res[it].toFloat() })
            }
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> {
                val res = f(v1.data.toList(), l1, v2.data.toList(), l2)
                CpuStorage.F64(DoubleArray(res.size) { res[it] })
            }
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                val d1 = FloatArray(v1.data.size) { f8e4m3ToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { f8e4m3ToFloat(v2.data[it]) }.map { it.toDouble() }
                val res = f(d1, l1, d2, l2)
                CpuStorage.F8E4M3(ByteArray(res.size) { floatToF8e4m3(res[it].toFloat()) })
            }
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = op,
            )
        }
}

public interface Map2InPlace {
    public val op: String

    public fun f(v1: MutableList<Double>, l1: Layout, v2: List<Double>, l2: Layout)

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout) {
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toInt().toUByte()
            }
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toLong().toUInt()
            }
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toInt().toShort()
            }
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toInt()
            }
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toLong()
            }
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                val m1 = FloatArray(v1.data.size) { bfloat16BitsToFloat(v1.data[it]) }.map { it.toDouble() }.toMutableList()
                val d2 = FloatArray(v2.data.size) { bfloat16BitsToFloat(v2.data[it]) }.map { it.toDouble() }
                f(m1, l1, d2, l2)
                for (i in m1.indices) v1.data[i] = floatToBFloat16Bits(m1[i].toFloat())
            }
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                val m1 = FloatArray(v1.data.size) { halfBitsToFloat(v1.data[it]) }.map { it.toDouble() }.toMutableList()
                val d2 = FloatArray(v2.data.size) { halfBitsToFloat(v2.data[it]) }.map { it.toDouble() }
                f(m1, l1, d2, l2)
                for (i in m1.indices) v1.data[i] = floatToHalfBits(m1[i].toFloat())
            }
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> {
                val m1 = v1.data.map { it.toDouble() }.toMutableList()
                f(m1, l1, v2.data.map { it.toDouble() }, l2)
                for (i in m1.indices) v1.data[i] = m1[i].toFloat()
            }
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> {
                val m1 = v1.data.toMutableList()
                f(m1, l1, v2.data.toList(), l2)
                for (i in m1.indices) v1.data[i] = m1[i]
            }
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                val m1 = FloatArray(v1.data.size) { f8e4m3ToFloat(v1.data[it]) }.map { it.toDouble() }.toMutableList()
                val d2 = FloatArray(v2.data.size) { f8e4m3ToFloat(v2.data[it]) }.map { it.toDouble() }
                f(m1, l1, d2, l2)
                for (i in m1.indices) v1.data[i] = floatToF8e4m3(m1[i].toFloat())
            }
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = op,
            )
        }
    }
}

public interface Map2U8 {
    public val op: String

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): UByteArray

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage =
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                val d1 = FloatArray(v1.data.size) { bfloat16BitsToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { bfloat16BitsToFloat(v2.data[it]) }.map { it.toDouble() }
                CpuStorage.U8(f(d1, l1, d2, l2))
            }
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                val d1 = FloatArray(v1.data.size) { halfBitsToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { halfBitsToFloat(v2.data[it]) }.map { it.toDouble() }
                CpuStorage.U8(f(d1, l1, d2, l2))
            }
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 ->
                CpuStorage.U8(f(v1.data.map { it.toDouble() }, l1, v2.data.map { it.toDouble() }, l2))
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 ->
                CpuStorage.U8(f(v1.data.toList(), l1, v2.data.toList(), l2))
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                val d1 = FloatArray(v1.data.size) { f8e4m3ToFloat(v1.data[it]) }.map { it.toDouble() }
                val d2 = FloatArray(v2.data.size) { f8e4m3ToFloat(v2.data[it]) }.map { it.toDouble() }
                CpuStorage.U8(f(d1, l1, d2, l2))
            }
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = op,
            )
        }
}

public fun <T, U> binaryMap(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> U,
): List<U> {
    val elCount = lhsL.shape().elemCount()
    val result = ArrayList<U>(elCount)

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val innerLs = ndIter.innerStrides[0]
    val innerRs = ndIter.innerStrides[1]

    for (offsets in ndIter) {
        val lhsOff = offsets[0]
        val rhsOff = offsets[1]
        for (i in 0 until innerSize) {
            result.add(f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs]))
        }
    }

    return result
}

// Similar to binaryMap but with vectorized variants.
public fun <T> binaryMapVec(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> T,
    fVec: (List<T>, List<T>, MutableList<T>) -> Unit,
    fScalarVec: (T, List<T>, MutableList<T>) -> Unit,
): List<T> {
    val elCount = lhsL.shape().elemCount()
    if (elCount == 0) return emptyList()
    val ys = MutableList(elCount) { lhs[0] }

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val innerLs = ndIter.innerStrides[0]
    val innerRs = ndIter.innerStrides[1]

    var dstOff = 0

    for (offsets in ndIter) {
        val lhsOff = offsets[0]
        val rhsOff = offsets[1]
        when {
            innerLs == 1 && innerRs == 1 -> fVec(
                lhs.subList(lhsOff, lhsOff + innerSize),
                rhs.subList(rhsOff, rhsOff + innerSize),
                ys.subList(dstOff, dstOff + innerSize),
            )
            innerLs == 1 && innerRs == 0 -> {
                val r = rhs[rhsOff]
                fScalarVec(
                    r,
                    lhs.subList(lhsOff, lhsOff + innerSize),
                    ys.subList(dstOff, dstOff + innerSize),
                )
            }
            innerLs == 0 && innerRs == 1 -> {
                val l = lhs[lhsOff]
                for (i in 0 until innerSize) {
                    ys[dstOff + i] = f(l, rhs[rhsOff + i])
                }
            }
            else -> {
                for (i in 0 until innerSize) {
                    ys[dstOff + i] = f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs])
                }
            }
        }
        dstOff += innerSize
    }

    // All elCount elements have been written in the dispatch loop above.
    return ys
}

public fun <T, U> unaryMap(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
): List<U> =
    when (val blocks = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val startOffset = blocks.startOffset
            val len = blocks.len
            val result = ArrayList<U>(len)
            for (i in startOffset until (startOffset + len)) {
                result.add(f(vs[i]))
            }
            result
        }
        is StridedBlocks.UniformBlocks -> {
            val startOffset = blocks.startOffset
            val blockLen = blocks.blockLen
            val count = blocks.count
            val srcStride = blocks.srcStride
            val result = ArrayList<U>(count * blockLen)
            if (blockLen == 1) {
                for (i in 0 until count) {
                    val v = vs[startOffset + i * srcStride]
                    result.add(f(v))
                }
            } else {
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    for (offset in 0 until blockLen) {
                        val v = vs[srcStart + offset]
                        result.add(f(v))
                    }
                }
            }
            result
        }
        is StridedBlocks.MultipleBlocks -> {
            val blockStartIndex = blocks.blockStartIndex
            val blockLen = blocks.blockLen
            val result = ArrayList<U>(layout.shape().elemCount())
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                for (index in blockStartIndex) {
                    val v = vs[index]
                    result.add(f(v))
                }
            } else {
                for (index in blockStartIndex) {
                    for (offset in 0 until blockLen) {
                        val v = vs[index + offset]
                        result.add(f(v))
                    }
                }
            }
            result
        }
    }

public fun <T, U> unaryMapVec(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
    fVec: (List<T>, MutableList<U>) -> Unit,
): List<U> =
    when (val blocks = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val startOffset = blocks.startOffset
            val len = blocks.len
            if (len == 0) {
                emptyList()
            } else {
                val initVal = f(vs[startOffset])
                val ys = MutableList(len) { initVal }
                fVec(vs.subList(startOffset, startOffset + len), ys)
                ys
            }
        }
        is StridedBlocks.UniformBlocks -> {
            val startOffset = blocks.startOffset
            val blockLen = blocks.blockLen
            val count = blocks.count
            val srcStride = blocks.srcStride
            val elCount = count * blockLen
            if (blockLen == 1) {
                val result = ArrayList<U>(count)
                for (i in 0 until count) {
                    val v = vs[startOffset + i * srcStride]
                    result.add(f(v))
                }
                result
            } else if (elCount == 0) {
                emptyList()
            } else {
                val initVal = f(vs[startOffset])
                val ys = MutableList(elCount) { initVal }
                var dstIndex = 0
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    fVec(
                        vs.subList(srcStart, srcStart + blockLen),
                        ys.subList(dstIndex, dstIndex + blockLen),
                    )
                    dstIndex += blockLen
                }
                ys
            }
        }
        is StridedBlocks.MultipleBlocks -> {
            val blockStartIndex = blocks.blockStartIndex
            val blockLen = blocks.blockLen
            val elCount = layout.shape().elemCount()
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                val result = ArrayList<U>(elCount)
                for (index in blockStartIndex) {
                    val v = vs[index]
                    result.add(f(v))
                }
                result
            } else if (elCount == 0 || vs.isEmpty()) {
                emptyList()
            } else {
                val initVal = f(vs[0])
                val ys = MutableList(elCount) { initVal }
                var dstIndex = 0
                for (srcIndex in blockStartIndex) {
                    val srcSlice = vs.subList(srcIndex, srcIndex + blockLen)
                    val dstSlice = ys.subList(dstIndex, dstIndex + blockLen)
                    fVec(srcSlice, dstSlice)
                    dstIndex += blockLen
                }
                ys
            }
        }
    }
