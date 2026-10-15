// port-lint: source cpu_backend/utils.rs
package io.github.kotlinmania.candle

/**
 * Helper functions to write CPU kernels.
 */
internal object C {
    val vs: (CpuStorage) -> List<Double> = { s ->
        when (s) {
            is CpuStorage.U8 -> s.data.map { it.toDouble() }
            is CpuStorage.U32 -> s.data.map { it.toDouble() }
            is CpuStorage.I16 -> s.data.map { it.toDouble() }
            is CpuStorage.I32 -> s.data.map { it.toDouble() }
            is CpuStorage.I64 -> s.data.map { it.toDouble() }
            is CpuStorage.BF16 -> FloatArray(s.data.size) { Float16Utils.bf16BitsToFloat(s.data[it]) }.map { it.toDouble() }
            is CpuStorage.F16 -> FloatArray(s.data.size) { Float16Utils.f16BitsToFloat(s.data[it]) }.map { it.toDouble() }
            is CpuStorage.F32 -> s.data.map { it.toDouble() }
            is CpuStorage.F64 -> s.data.toList()
            is CpuStorage.F8E4M3 -> s.data.map { it.toDouble() }
            else -> emptyList()
        }
    }
    val U8: (List<Double>) -> CpuStorage = { r -> CpuStorage.U8(UByteArray(r.size) { r[it].toInt().toUByte() }) }
    val U32: (List<Double>) -> CpuStorage = { r -> CpuStorage.U32(UIntArray(r.size) { r[it].toLong().toUInt() }) }
    val I16: (List<Double>) -> CpuStorage = { r -> CpuStorage.I16(ShortArray(r.size) { r[it].toInt().toShort() }) }
    val I32: (List<Double>) -> CpuStorage = { r -> CpuStorage.I32(IntArray(r.size) { r[it].toInt() }) }
    val I64: (List<Double>) -> CpuStorage = { r -> CpuStorage.I64(LongArray(r.size) { r[it].toLong() }) }
    val BF16: (List<Double>) -> CpuStorage = { r -> CpuStorage.BF16(ShortArray(r.size) { Float16Utils.floatToBF16Bits(r[it].toFloat()) }) }
    val F16: (List<Double>) -> CpuStorage = { r -> CpuStorage.F16(ShortArray(r.size) { Float16Utils.floatToF16Bits(r[it].toFloat()) }) }
    val F32: (List<Double>) -> CpuStorage = { r -> CpuStorage.F32(FloatArray(r.size) { r[it].toFloat() }) }
    val F64: (List<Double>) -> CpuStorage = { r -> CpuStorage.F64(DoubleArray(r.size) { r[it] }) }
    val F8E4M3: (List<Double>) -> CpuStorage = { r -> CpuStorage.F8E4M3(ByteArray(r.size) { r[it].toInt().toByte() }) }
    val applyInPlace: (Map2InPlace, CpuStorage, Layout, CpuStorage, Layout) -> Unit = { m, v1, l1, v2, l2 ->
        val m1 = vs(v1).toMutableList()
        m.f(m1, l1, vs(v2), l2)
        when (v1) {
            is CpuStorage.U8 -> for (i in m1.indices) v1.data[i] = m1[i].toInt().toUByte()
            is CpuStorage.U32 -> for (i in m1.indices) v1.data[i] = m1[i].toLong().toUInt()
            is CpuStorage.I16 -> for (i in m1.indices) v1.data[i] = m1[i].toInt().toShort()
            is CpuStorage.I32 -> for (i in m1.indices) v1.data[i] = m1[i].toInt()
            is CpuStorage.I64 -> for (i in m1.indices) v1.data[i] = m1[i].toLong()
            is CpuStorage.BF16 -> for (i in m1.indices) v1.data[i] = Float16Utils.floatToBF16Bits(m1[i].toFloat())
            is CpuStorage.F16 -> for (i in m1.indices) v1.data[i] = Float16Utils.floatToF16Bits(m1[i].toFloat())
            is CpuStorage.F32 -> for (i in m1.indices) v1.data[i] = m1[i].toFloat()
            is CpuStorage.F64 -> for (i in m1.indices) v1.data[i] = m1[i]
            is CpuStorage.F8E4M3 -> for (i in m1.indices) v1.data[i] = m1[i].toInt().toByte()
            else -> Unit
        }
    }
}

public interface Map1 {
    public val self: Map1 get() = this

    public fun f(vs: List<Double>, layout: Layout): List<Double>

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage {
        return when (vs) {
            is CpuStorage.U8 -> C.U8(self.f(C.vs(vs), layout))
            is CpuStorage.U32 -> C.U32(self.f(C.vs(vs), layout))
            is CpuStorage.I16 -> C.I16(self.f(C.vs(vs), layout))
            is CpuStorage.I32 -> C.I32(self.f(C.vs(vs), layout))
            is CpuStorage.I64 -> C.I64(self.f(C.vs(vs), layout))
            is CpuStorage.BF16 -> C.BF16(self.f(C.vs(vs), layout))
            is CpuStorage.F16 -> C.F16(self.f(C.vs(vs), layout))
            is CpuStorage.F32 -> C.F32(self.f(C.vs(vs), layout))
            is CpuStorage.F64 -> C.F64(self.f(C.vs(vs), layout))
            is CpuStorage.F8E4M3 -> C.F8E4M3(self.f(C.vs(vs), layout))
            is CpuStorage.F6E2M3 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F6E3M2 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F4 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
            is CpuStorage.F8E8M0 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1")
        }
    }
}

public interface Map1Any {
    public val self: Map1Any get() = this

    public fun f(
        vs: List<Double>,
        layout: Layout,
        wrap: (List<Double>) -> CpuStorage,
    ): CpuStorage

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage {
        return when (vs) {
            is CpuStorage.U8 -> self.f(C.vs(vs), layout, C.U8)
            is CpuStorage.U32 -> self.f(C.vs(vs), layout, C.U32)
            is CpuStorage.I16 -> self.f(C.vs(vs), layout, C.I16)
            is CpuStorage.I32 -> self.f(C.vs(vs), layout, C.I32)
            is CpuStorage.I64 -> self.f(C.vs(vs), layout, C.I64)
            is CpuStorage.BF16 -> self.f(C.vs(vs), layout, C.BF16)
            is CpuStorage.F16 -> self.f(C.vs(vs), layout, C.F16)
            is CpuStorage.F32 -> self.f(C.vs(vs), layout, C.F32)
            is CpuStorage.F64 -> self.f(C.vs(vs), layout, C.F64)
            is CpuStorage.F8E4M3 -> self.f(C.vs(vs), layout, C.F8E4M3)
            is CpuStorage.F6E2M3 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F6E3M2 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F4 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
            is CpuStorage.F8E8M0 -> throw CandleException.UnsupportedDTypeForOp(vs.dtype(), "map1any")
        }
    }
}

public interface Map2 {
    public val op: String
    public val self: Map2 get() = this

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): List<Double>

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage {
        return when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> C.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> C.U32(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> C.I16(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> C.I32(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> C.I64(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> C.BF16(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> C.F16(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> C.F32(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> C.F64(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> C.F8E4M3(self.f(C.vs(v1), l1, C.vs(v2), l2))
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = self.op,
            )
        }
    }
}

public interface Map2InPlace {
    public val op: String
    public val self: Map2InPlace get() = this

    public fun f(v1: MutableList<Double>, l1: Layout, v2: List<Double>, l2: Layout)

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout) {
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> C.applyInPlace(self, v1, l1, v2, l2)
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> C.applyInPlace(self, v1, l1, v2, l2)
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = self.op,
            )
        }
    }
}

public interface Map2U8 {
    public val op: String
    public val self: Map2U8 get() = this

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): UByteArray

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage {
        return when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> CpuStorage.U8(self.f(C.vs(v1), l1, C.vs(v2), l2))
            else -> throw CandleException.DTypeMismatchBinaryOp(
                lhs = v1.dtype(),
                rhs = v2.dtype(),
                op = self.op,
            )
        }
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
