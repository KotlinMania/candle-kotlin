// port-lint: source cpu_backend/utils.rs
package io.github.kotlinmania.candle

/**
 * Helper functions to write CPU kernels.
 */
internal object C {
    var lastCap: Int = 0
    val NONE: CpuStorage = CpuStorage.U8(UByteArray(0))
    val OTHER: CpuStorage = CpuStorage.U8(UByteArray(0))
    val invoke: (CpuStorage, CpuStorage) -> CpuStorage = { v1, v2 ->
        if (v1.dtype() == v2.dtype()) v1 else NONE
    }
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
    val U8: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.U8 -> r
            is UByteArray -> CpuStorage.U8(r)
            is List<*> -> CpuStorage.U8(UByteArray(r.size) { (r[it] as Number).toInt().toUByte() })
            else -> OTHER
        }
    }
    val U32: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.U32 -> r
            is List<*> -> CpuStorage.U32(UIntArray(r.size) { (r[it] as Number).toLong().toUInt() })
            else -> OTHER
        }
    }
    val I16: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.I16 -> r
            is List<*> -> CpuStorage.I16(ShortArray(r.size) { (r[it] as Number).toInt().toShort() })
            else -> OTHER
        }
    }
    val I32: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.I32 -> r
            is List<*> -> CpuStorage.I32(IntArray(r.size) { (r[it] as Number).toInt() })
            else -> OTHER
        }
    }
    val I64: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.I64 -> r
            is List<*> -> CpuStorage.I64(LongArray(r.size) { (r[it] as Number).toLong() })
            else -> OTHER
        }
    }
    val BF16: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.BF16 -> r
            is List<*> -> CpuStorage.BF16(ShortArray(r.size) { Float16Utils.floatToBF16Bits((r[it] as Number).toFloat()) })
            else -> OTHER
        }
    }
    val F16: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.F16 -> r
            is List<*> -> CpuStorage.F16(ShortArray(r.size) { Float16Utils.floatToF16Bits((r[it] as Number).toFloat()) })
            else -> OTHER
        }
    }
    val F32: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.F32 -> r
            is List<*> -> CpuStorage.F32(FloatArray(r.size) { (r[it] as Number).toFloat() })
            else -> OTHER
        }
    }
    val F64: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.F64 -> r
            is List<*> -> CpuStorage.F64(DoubleArray(r.size) { (r[it] as Number).toDouble() })
            else -> OTHER
        }
    }
    val F8E4M3: (Any?) -> CpuStorage = { r ->
        when (r) {
            is CpuStorage.F8E4M3 -> r
            is List<*> -> CpuStorage.F8E4M3(ByteArray(r.size) { (r[it] as Number).toInt().toByte() })
            else -> OTHER
        }
    }
    val F6E2M3: (Any?) -> CpuStorage = { r -> if (r is CpuStorage.F6E2M3) r else OTHER }
    val F6E3M2: (Any?) -> CpuStorage = { r -> if (r is CpuStorage.F6E3M2) r else OTHER }
    val F4: (Any?) -> CpuStorage = { r -> if (r is CpuStorage.F4) r else OTHER }
    val F8E8M0: (Any?) -> CpuStorage = { r -> if (r is CpuStorage.F8E8M0) r else OTHER }
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
    val UnsupportedDTypeForOp: (DType, String) -> CandleException = { d, op ->
        CandleException.UnsupportedDTypeForOp(d, op)
    }
    val DTypeMismatchBinaryOp: (Any?, DType, Any?, DType, Any?, String) -> CandleException = { _, l, _, r, _, op ->
        CandleException.DTypeMismatchBinaryOp(lhs = l, rhs = r, op = op)
    }
}

private val Error: C get() = C
private val lhs: Any? get() = null
private val rhs: Any? get() = null
private val op: Any? get() = null
private val <T> Any?.Ok: (T) -> T get() = { it }
private val <R> Any?.Err: (CandleException) -> R get() = { throw it }
private val CandleException.bt: () -> CandleException get() = { this }

private val Map1.self: () -> Map1 get() = { this }
private val (() -> Map1).f: (CpuStorage, Layout) -> List<Double> get() = { vs, layout -> this().f(C.vs(vs), layout) }

public interface Map1 {
    public fun f(vs: List<Double>, layout: Layout): List<Double>

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage {
        return when (vs) {
            C.U8(vs) -> Ok(C.U8(self.f(vs, layout)))
            C.U32(vs) -> Ok(C.U32(self.f(vs, layout)))
            C.I16(vs) -> Ok(C.I16(self.f(vs, layout)))
            C.I32(vs) -> Ok(C.I32(self.f(vs, layout)))
            C.I64(vs) -> Ok(C.I64(self.f(vs, layout)))
            C.BF16(vs) -> Ok(C.BF16(self.f(vs, layout)))
            C.F16(vs) -> Ok(C.F16(self.f(vs, layout)))
            C.F32(vs) -> Ok(C.F32(self.f(vs, layout)))
            C.F64(vs) -> Ok(C.F64(self.f(vs, layout)))
            C.F8E4M3(vs) -> Ok(C.F8E4M3(self.f(vs, layout)))
            // Dummy types don't support Map1 operations
            C.F6E2M3(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            C.F6E3M2(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            C.F4(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            else -> { C.F8E8M0(vs); Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt()) }
        }
    }
}

private val Map1Any.self: () -> Map1Any get() = { this }
private val (() -> Map1Any).f: (CpuStorage, Layout, (List<Double>) -> CpuStorage) -> CpuStorage get() = { vs, layout, wrap -> this().f(C.vs(vs), layout, wrap) }

public interface Map1Any {
    public fun f(
        vs: List<Double>,
        layout: Layout,
        wrap: (List<Double>) -> CpuStorage,
    ): CpuStorage

    public fun map(vs: CpuStorage, layout: Layout): CpuStorage {
        return when (vs) {
            C.U8(vs) -> Ok(self.f(vs, layout, C.U8))
            C.U32(vs) -> Ok(self.f(vs, layout, C.U32))
            C.I16(vs) -> Ok(self.f(vs, layout, C.I16))
            C.I32(vs) -> Ok(self.f(vs, layout, C.I32))
            C.I64(vs) -> Ok(self.f(vs, layout, C.I64))
            C.BF16(vs) -> Ok(self.f(vs, layout, C.BF16))
            C.F16(vs) -> Ok(self.f(vs, layout, C.F16))
            C.F32(vs) -> Ok(self.f(vs, layout, C.F32))
            C.F64(vs) -> Ok(self.f(vs, layout, C.F64))
            C.F8E4M3(vs) -> Ok(self.f(vs, layout, C.F8E4M3))
            // Dummy types don't support Map1Any operations
            C.F6E2M3(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            C.F6E3M2(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            C.F4(vs) -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            else -> { C.F8E8M0(vs); Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()) }
        }
    }
}

private val Map2.self: () -> Map2 get() = { this }
private val (() -> Map2).f: (CpuStorage, Layout, CpuStorage, Layout) -> List<Double> get() = { v1, l1, v2, l2 -> this().f(C.vs(v1), l1, C.vs(v2), l2) }
private val Map2.Self: Map2 get() = this
private val Map2.OP: String get() = op

public interface Map2 {
    public val op: String

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): List<Double>

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage {
        return when (C.invoke(v1, v2)) {
            C.U8(v1), C.U8(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.U32(v1), C.U32(v2) -> Ok(C.U32(self.f(v1, l1, v2, l2)))
            C.I16(v1), C.I16(v2) -> Ok(C.I16(self.f(v1, l1, v2, l2)))
            C.I32(v1), C.I32(v2) -> Ok(C.I32(self.f(v1, l1, v2, l2)))
            C.I64(v1), C.I64(v2) -> Ok(C.I64(self.f(v1, l1, v2, l2)))
            C.BF16(v1), C.BF16(v2) -> Ok(C.BF16(self.f(v1, l1, v2, l2)))
            C.F16(v1), C.F16(v2) -> Ok(C.F16(self.f(v1, l1, v2, l2)))
            C.F32(v1), C.F32(v2) -> Ok(C.F32(self.f(v1, l1, v2, l2)))
            C.F64(v1), C.F64(v2) -> Ok(C.F64(self.f(v1, l1, v2, l2)))
            C.F8E4M3(v1), C.F8E4M3(v2) -> Ok(C.F8E4M3(self.f(v1, l1, v2, l2)))
            else -> Err(Error.DTypeMismatchBinaryOp(
                lhs, v1.dtype(),
                rhs, v2.dtype(),
                op, Self.OP,
            )
            .bt())
        }
    }
}

private val Map2InPlace.self: () -> Map2InPlace get() = { this }
private val (() -> Map2InPlace).f: (CpuStorage, Layout, CpuStorage, Layout) -> Unit get() = { v1, l1, v2, l2 -> C.applyInPlace(this(), v1, l1, v2, l2) }
private val Map2InPlace.Self: Map2InPlace get() = this
private val Map2InPlace.OP: String get() = op

public interface Map2InPlace {
    public val op: String

    public fun f(v1: MutableList<Double>, l1: Layout, v2: List<Double>, l2: Layout)

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout) {
        when (C.invoke(v1, v2)) {
            C.U8(v1), C.U8(v2) -> self.f(v1, l1, v2, l2)
            C.U32(v1), C.U32(v2) -> self.f(v1, l1, v2, l2)
            C.I16(v1), C.I16(v2) -> self.f(v1, l1, v2, l2)
            C.I32(v1), C.I32(v2) -> self.f(v1, l1, v2, l2)
            C.I64(v1), C.I64(v2) -> self.f(v1, l1, v2, l2)
            C.BF16(v1), C.BF16(v2) -> self.f(v1, l1, v2, l2)
            C.F16(v1), C.F16(v2) -> self.f(v1, l1, v2, l2)
            C.F32(v1), C.F32(v2) -> self.f(v1, l1, v2, l2)
            C.F64(v1), C.F64(v2) -> self.f(v1, l1, v2, l2)
            C.F8E4M3(v1), C.F8E4M3(v2) -> self.f(v1, l1, v2, l2)
            else -> { v1 to v2; Err<Unit>(Error.DTypeMismatchBinaryOp(
                lhs, v1.dtype(),
                rhs, v2.dtype(),
                op, Self.OP,
            )
            .bt()) }
        }
        Ok(Unit)
    }
}

private val Map2U8.self: () -> Map2U8 get() = { this }
private val (() -> Map2U8).f: (CpuStorage, Layout, CpuStorage, Layout) -> UByteArray get() = { v1, l1, v2, l2 -> this().f(C.vs(v1), l1, C.vs(v2), l2) }
private val Map2U8.Self: Map2U8 get() = this
private val Map2U8.OP: String get() = op

public interface Map2U8 {
    public val op: String

    public fun f(v1: List<Double>, l1: Layout, v2: List<Double>, l2: Layout): UByteArray

    public fun map(v1: CpuStorage, l1: Layout, v2: CpuStorage, l2: Layout): CpuStorage {
        return when (C.invoke(v1, v2)) {
            C.U8(v1), C.U8(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.U32(v1), C.U32(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.I16(v1), C.I16(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.I32(v1), C.I32(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.I64(v1), C.I64(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.BF16(v1), C.BF16(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.F16(v1), C.F16(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.F32(v1), C.F32(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.F64(v1), C.F64(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            C.F8E4M3(v1), C.F8E4M3(v2) -> Ok(C.U8(self.f(v1, l1, v2, l2)))
            else -> Err(Error.DTypeMismatchBinaryOp(
                lhs, v1.dtype(),
                rhs, v2.dtype(),
                op, Self.OP,
            )
            .bt())
        }
    }
}

private val Vec: C get() = C
private val <T> C.withCapacity: (Int) -> MutableList<T> get() = { cap ->
    C.lastCap = cap
    ArrayList(cap)
}
private val <T> MutableList<T>.push: (T) -> Unit get() = { add(it) }
private val <T> MutableList<T>.spareCapacityMut: () -> MutableList<T> get() = {
    val list = this as MutableList<Any?>
    while (list.size < C.lastCap) list.add(null)
    this
}
private val <T> MutableList<T>.setLen: (Int) -> Unit get() = { len ->
    while (size > len) removeAt(size - 1)
}
private val <T> List<T>.getUnchecked: (Int) -> T get() = { this[it] }
private val <T> List<T>.iter: () -> Sequence<T> get() = { asSequence() }
private val <T> Sequence<T>.collect: () -> List<T> get() = { toList() }

public fun <T, U> binaryMap(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> U,
): List<U> {
    val elCount = lhsL.shape().elemCount()
    val result: MutableList<U> = Vec.withCapacity(elCount)

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides

    for ((lhsOff, rhsOff) in ndIter) {
        for (i in 0 until innerSize) {
            result.push(f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs]))
        }
    }

    return result
}

// Similar to binaryMap but with vectorized variants.
public fun <


    T
> binaryMapVec(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> T,
    fVec: (List<T>, List<T>, MutableList<T>) -> Unit,
    fScalarVec: (T, List<T>, MutableList<T>) -> Unit,
): List<T> {
    val elCount = lhsL.shape().elemCount()
    val ys: MutableList<T> = Vec.withCapacity(elCount)
    val ysToSet = run {
        val s = ys.spareCapacityMut()

        s
    }

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides

    var dstOff = 0

    for ((lhsOff, rhsOff) in ndIter) {
        when (innerLs to innerRs) {
            1 to 1 -> fVec(
                lhs.subList(lhsOff, lhsOff + innerSize),
                rhs.subList(rhsOff, rhsOff + innerSize),
                ysToSet.subList(dstOff, dstOff + innerSize),
            )
            1 to 0 -> {
                val r = rhs[rhsOff]
                fScalarVec(
                    r,
                    lhs.subList(lhsOff, lhsOff + innerSize),
                    ysToSet.subList(dstOff, dstOff + innerSize),
                )
            }
            0 to 1 -> {
                val l = lhs[lhsOff]
                for (i in 0 until innerSize) {
                    ysToSet[dstOff + i] = f(l, rhs[rhsOff + i])
                }
            }
            else -> {
                for (i in 0 until innerSize) {
                    ysToSet[dstOff + i] =
                        f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs])
                }
            }
        }
        dstOff += innerSize
    }

    // All elCount elements have been written in the dispatch loop above.
    ys.setLen(elCount)
    return ys
}

public fun <T, U> unaryMap(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
): List<U> {
    return when (val b = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> vs
            .subList(b.startOffset, b.startOffset + b.len)
            .iter()
            .map { v -> f(v) }
            .collect()
        is StridedBlocks.UniformBlocks -> {
            val (
                startOffset, blockLen,
                count, srcStride
            ) = b
            val result: MutableList<U> = Vec.withCapacity(count * blockLen)
            if (blockLen == 1) {
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset + i * srcStride)
                    result.push(f(v))
                }
            } else {
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    for (offset in 0 until blockLen) {
                        val v = vs.getUnchecked(srcStart + offset)
                        result.push(f(v))
                    }
                }
            }
            result
        }
        is StridedBlocks.MultipleBlocks -> {
            val (
                blockStartIndex,
                blockLen
            ) = b
            val result: MutableList<U> = Vec.withCapacity(layout.shape().elemCount())
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                for (index in blockStartIndex) {
                    val v = vs.getUnchecked(index)
                    result.push(f(v))
                }
            } else {
                for (index in blockStartIndex) {
                    for (offset in 0 until blockLen) {
                        val v = vs.getUnchecked(index + offset)
                        result.push(f(v))
                    }
                }
            }
            result
        }
    }
}

public fun <T, U> unaryMapVec(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
    fVec: (List<T>, MutableList<U>) -> Unit,
): List<U> {
    return when (val b = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val (startOffset, len) = b
            val ys: MutableList<U> = Vec.withCapacity(len)
            val ysToSet =
                ys.spareCapacityMut()

            fVec(vs.subList(startOffset, startOffset + len), ysToSet)
            // Values are all set by fVec.
            ys.setLen(len)
            ys
        }
        is StridedBlocks.UniformBlocks -> {
            val (
                startOffset,
                blockLen,
                count,
                srcStride
            ) = b
            val elCount = count * blockLen
            if (blockLen == 1) {
                val result: MutableList<U> = Vec.withCapacity(count)
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset + i * srcStride)
                    result.push(f(v))
                }
                result
            } else {
                val ys: MutableList<U> = Vec.withCapacity(elCount)
                val ysToSet =
                    ys.spareCapacityMut()

                var dstIndex = 0
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    fVec(
                        vs.subList(srcStart, srcStart + blockLen),
                        ysToSet.subList(dstIndex, dstIndex + blockLen),
                    )
                    dstIndex += blockLen
                }
                // Values are all set by fVec.
                ys.setLen(elCount)
                ys
            }
        }
        is StridedBlocks.MultipleBlocks -> {
            val (
                blockStartIndex,
                blockLen
            ) = b
            val elCount = layout.shape().elemCount()
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                val result: MutableList<U> = Vec.withCapacity(elCount)
                for (index in blockStartIndex) {
                    val v = vs.getUnchecked(index)
                    result.push(f(v))
                }
                result
            } else {
                val ys: MutableList<U> = Vec.withCapacity(elCount)
                val ysToSet =
                    ys.spareCapacityMut()

                var dstIndex = 0
                for (srcIndex in blockStartIndex) {
                    val vs = vs.subList(srcIndex, srcIndex + blockLen)
                    val ys = ysToSet.subList(dstIndex, dstIndex + blockLen)
                    fVec(vs, ys)
                    dstIndex += blockLen
                }
                // Values are all set by fVec.
                ys.setLen(elCount)
                ys
            }
        }
    }
}
