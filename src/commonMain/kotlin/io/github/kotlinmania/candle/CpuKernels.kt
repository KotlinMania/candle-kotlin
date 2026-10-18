// port-lint: source cpu/kernels.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16

/**
 * Vectorized and scalar CPU kernel operations over contiguous slices.
 */
public interface VecOps<T> {
    public val zero: () -> T
    public val add: (T, T) -> T
    public val mul: (T, T) -> T

    public fun min(
        lhs: T,
        rhs: T,
    ): T

    public fun max(
        lhs: T,
        rhs: T,
    ): T

    /**
     * Element-wise addition of two slices into a third.
     */
    public fun vecAdd(
        lhs: List<T>,
        rhs: List<T>,
        res: MutableList<T>,
    ) {
        lhs
            .iter()
            .zip(rhs)
            .zip(res.indices)
            .forEach { entry -> res[entry.second] = add(entry.first.first, entry.first.second) }
    }

    /**
     * Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
     */
    public fun scalarAdd(
        scalar: T,
        xs: List<T>,
        ys: MutableList<T>,
    ) {
        xs.iter().zip(ys.indices).forEach { entry -> ys[entry.second] = add(entry.first, scalar) }
    }

    /**
     * Dot-product of two vectors.
     *
     * # Safety
     *
     * The length of `lhs` and `rhs` have to be at least `len`. `res` has to point to a valid
     * element.
     */
    public fun vecDot(
        lhs: List<T>,
        rhs: List<T>,
        res: MutableList<T>,
        len: Int,
    ) {
        res[0] = zero()
        for (i in 0 until len) {
            res[0] = add(res[0], mul(lhs.add(i), rhs.add(i)))
        }
    }

    /**
     * Sum of all elements in a vector.
     *
     * # Safety
     *
     * The length of `xs` must be at least `len`. `res` has to point to a valid
     * element.
     */
    public fun vecReduceSum(
        xs: List<T>,
        res: MutableList<T>,
        len: Int,
    ) {
        res[0] = zero()
        for (i in 0 until len) {
            res[0] = add(res[0], xs.add(i))
        }
    }

    /**
     * Maximum element in a non-empty vector.
     *
     * # Safety
     *
     * The length of `xs` must be at least `len` and positive. `res` has to point to a valid
     * element.
     */
    public fun vecReduceMax(
        xs: List<T>,
        res: MutableList<T>,
        len: Int,
    ) {
        res[0] = xs[0]
        for (i in 1 until len) {
            res[0] = max(res[0], xs.add(i))
        }
    }

    /**
     * Minimum element in a non-empty vector.
     *
     * # Safety
     *
     * The length of `xs` must be at least `len` and positive. `res` has to point to a valid
     * element.
     */
    public fun vecReduceMin(
        xs: List<T>,
        res: MutableList<T>,
        len: Int,
    ) {
        res[0] = xs[0]
        for (i in 1 until len) {
            res[0] = min(res[0], xs.add(i))
        }
    }
}

public object F32VecOps : VecOps<Float> {
    override val zero: () -> Float = { 0.0f }
    override val add: (Float, Float) -> Float = { a, b -> a + b }
    override val mul: (Float, Float) -> Float = { a, b -> a * b }

    override fun min(
        lhs: Float,
        rhs: Float,
    ): Float = Self.min(lhs, rhs)

    override fun max(
        lhs: Float,
        rhs: Float,
    ): Float = Self.max(lhs, rhs)

    override fun vecAdd(
        lhs: List<Float>,
        rhs: List<Float>,
        res: MutableList<Float>,
    ) {
        if (hasMkl()) mklVsAdd(lhs, rhs, res)
        if (hasAccelerate() && !hasMkl()) accelerateVsAdd(lhs, rhs, res)
        if (!hasMkl() && !hasAccelerate()) {
            lhs
                .iter()
                .zip(rhs)
                .zip(res.indices)
                .forEach { entry -> res[entry.second] = entry.first.first + entry.first.second }
        }
    }

    override fun vecDot(
        lhs: List<Float>,
        rhs: List<Float>,
        res: MutableList<Float>,
        len: Int,
    ) {
        vecDotF32(lhs, rhs, res, len)
    }

    override fun vecReduceSum(
        xs: List<Float>,
        res: MutableList<Float>,
        len: Int,
    ) {
        vecSum(xs, res, len)
    }
}

public object F16VecOps : VecOps<io.github.kotlinmania.half.F16> {
    override val zero: () -> io.github.kotlinmania.half.F16 = {
        io.github.kotlinmania.half.F16
            .fromFloat(0.0f)
    }
    override val add: (io.github.kotlinmania.half.F16, io.github.kotlinmania.half.F16) -> io.github.kotlinmania.half.F16 = { a, b ->
        io.github.kotlinmania.half.F16
            .fromFloat(a.toFloat() + b.toFloat())
    }
    override val mul: (io.github.kotlinmania.half.F16, io.github.kotlinmania.half.F16) -> io.github.kotlinmania.half.F16 = { a, b ->
        io.github.kotlinmania.half.F16
            .fromFloat(a.toFloat() * b.toFloat())
    }

    override fun min(
        lhs: io.github.kotlinmania.half.F16,
        rhs: io.github.kotlinmania.half.F16,
    ): io.github.kotlinmania.half.F16 = Self.min(lhs, rhs)

    override fun max(
        lhs: io.github.kotlinmania.half.F16,
        rhs: io.github.kotlinmania.half.F16,
    ): io.github.kotlinmania.half.F16 = Self.max(lhs, rhs)

    override fun vecAdd(
        lhs: List<io.github.kotlinmania.half.F16>,
        rhs: List<io.github.kotlinmania.half.F16>,
        res: MutableList<io.github.kotlinmania.half.F16>,
    ) {
        vecAddF16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.len())
    }

    override fun scalarAdd(
        scalar: io.github.kotlinmania.half.F16,
        xs: List<io.github.kotlinmania.half.F16>,
        ys: MutableList<io.github.kotlinmania.half.F16>,
    ) {
        vecScalarAddF16(scalar, xs.asPtr(), ys.asMutPtr(), xs.len())
    }

    override fun vecDot(
        lhs: List<io.github.kotlinmania.half.F16>,
        rhs: List<io.github.kotlinmania.half.F16>,
        res: MutableList<io.github.kotlinmania.half.F16>,
        len: Int,
    ) {
        val resF32 = FloatArray(1) { 0f }
        vecDotF16(lhs, rhs, resF32, len)
        res[0] =
            io.github.kotlinmania.half.F16
                .fromF32(resF32[0])
    }
}

public object F64VecOps : VecOps<Double> {
    override val zero: () -> Double = { 0.0 }
    override val add: (Double, Double) -> Double = { a, b -> a + b }
    override val mul: (Double, Double) -> Double = { a, b -> a * b }

    override fun min(
        lhs: Double,
        rhs: Double,
    ): Double = Self.min(lhs, rhs)

    override fun max(
        lhs: Double,
        rhs: Double,
    ): Double = Self.max(lhs, rhs)

    override fun vecAdd(
        lhs: List<Double>,
        rhs: List<Double>,
        res: MutableList<Double>,
    ) {
        if (hasMkl()) mklVdAdd(lhs, rhs, res)
        if (hasAccelerate() && !hasMkl()) accelerateVdAdd(lhs, rhs, res)
        if (!hasMkl() && !hasAccelerate()) {
            lhs
                .iter()
                .zip(rhs)
                .zip(res.indices)
                .forEach { entry -> res[entry.second] = entry.first.first + entry.first.second }
        }
    }
}

public object BF16VecOps : VecOps<io.github.kotlinmania.half.BF16> {
    override val zero: () -> io.github.kotlinmania.half.BF16 = {
        io.github.kotlinmania.half.BF16
            .fromFloat(0.0f)
    }
    override val add: (io.github.kotlinmania.half.BF16, io.github.kotlinmania.half.BF16) -> io.github.kotlinmania.half.BF16 = { a, b ->
        io.github.kotlinmania.half.BF16
            .fromFloat(a.toFloat() + b.toFloat())
    }
    override val mul: (io.github.kotlinmania.half.BF16, io.github.kotlinmania.half.BF16) -> io.github.kotlinmania.half.BF16 = { a, b ->
        io.github.kotlinmania.half.BF16
            .fromFloat(a.toFloat() * b.toFloat())
    }

    override fun min(
        lhs: io.github.kotlinmania.half.BF16,
        rhs: io.github.kotlinmania.half.BF16,
    ): io.github.kotlinmania.half.BF16 = Self.min(lhs, rhs)

    override fun max(
        lhs: io.github.kotlinmania.half.BF16,
        rhs: io.github.kotlinmania.half.BF16,
    ): io.github.kotlinmania.half.BF16 = Self.max(lhs, rhs)

    override fun vecAdd(
        lhs: List<io.github.kotlinmania.half.BF16>,
        rhs: List<io.github.kotlinmania.half.BF16>,
        res: MutableList<io.github.kotlinmania.half.BF16>,
    ) {
        vecAddBf16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.len())
    }

    override fun scalarAdd(
        scalar: io.github.kotlinmania.half.BF16,
        xs: List<io.github.kotlinmania.half.BF16>,
        ys: MutableList<io.github.kotlinmania.half.BF16>,
    ) {
        vecScalarAddBf16(scalar, xs.asPtr(), ys.asMutPtr(), xs.len())
    }

    override fun vecDot(
        lhs: List<io.github.kotlinmania.half.BF16>,
        rhs: List<io.github.kotlinmania.half.BF16>,
        res: MutableList<io.github.kotlinmania.half.BF16>,
        len: Int,
    ) {
        val resF32 = FloatArray(1) { 0f }
        vecDotBf16(lhs, rhs, resF32, len)
        res[0] =
            io.github.kotlinmania.half.BF16
                .fromF32(resF32[0])
    }
}

public object U8VecOps : VecOps<UByte> {
    override val zero: () -> UByte = { 0u }
    override val add: (UByte, UByte) -> UByte = { a, b -> (a + b).toUByte() }
    override val mul: (UByte, UByte) -> UByte = { a, b -> (a * b).toUByte() }

    override fun min(
        lhs: UByte,
        rhs: UByte,
    ): UByte = Ord.min(lhs, rhs)

    override fun max(
        lhs: UByte,
        rhs: UByte,
    ): UByte = Ord.max(lhs, rhs)
}

public object U32VecOps : VecOps<UInt> {
    override val zero: () -> UInt = { 0u }
    override val add: (UInt, UInt) -> UInt = { a, b -> a + b }
    override val mul: (UInt, UInt) -> UInt = { a, b -> a * b }

    override fun min(
        lhs: UInt,
        rhs: UInt,
    ): UInt = Ord.min(lhs, rhs)

    override fun max(
        lhs: UInt,
        rhs: UInt,
    ): UInt = Ord.max(lhs, rhs)
}

public object I16VecOps : VecOps<Short> {
    override val zero: () -> Short = { 0 }
    override val add: (Short, Short) -> Short = { a, b -> (a + b).toShort() }
    override val mul: (Short, Short) -> Short = { a, b -> (a * b).toShort() }

    override fun min(
        lhs: Short,
        rhs: Short,
    ): Short = Ord.min(lhs, rhs)

    override fun max(
        lhs: Short,
        rhs: Short,
    ): Short = Ord.max(lhs, rhs)
}

public object I32VecOps : VecOps<Int> {
    override val zero: () -> Int = { 0 }
    override val add: (Int, Int) -> Int = { a, b -> a + b }
    override val mul: (Int, Int) -> Int = { a, b -> a * b }

    override fun min(
        lhs: Int,
        rhs: Int,
    ): Int = Ord.min(lhs, rhs)

    override fun max(
        lhs: Int,
        rhs: Int,
    ): Int = Ord.max(lhs, rhs)
}

public object I64VecOps : VecOps<Long> {
    override val zero: () -> Long = { 0L }
    override val add: (Long, Long) -> Long = { a, b -> a + b }
    override val mul: (Long, Long) -> Long = { a, b -> a * b }

    override fun min(
        lhs: Long,
        rhs: Long,
    ): Long = Ord.min(lhs, rhs)

    override fun max(
        lhs: Long,
        rhs: Long,
    ): Long = Ord.max(lhs, rhs)
}

public object F8E4M3VecOps : VecOps<Byte> {
    override val zero: () -> Byte = { 0 }
    override val add: (Byte, Byte) -> Byte = { a, b -> (a + b).toByte() }
    override val mul: (Byte, Byte) -> Byte = { a, b -> (a * b).toByte() }

    override fun min(
        lhs: Byte,
        rhs: Byte,
    ): Byte = Self.min(lhs, rhs)

    override fun max(
        lhs: Byte,
        rhs: Byte,
    ): Byte = Self.max(lhs, rhs)
}

public fun parForEach(
    nThreads: Int,
    f: (Int) -> Unit,
) {
    if (nThreads == 1) {
        f(0)
    } else {
        rayonScope { s ->
            for (threadIdx in 0 until nThreads) {
                val funcRef = f
                s.spawn { _ -> funcRef(threadIdx) }
            }
        }
    }
}

public fun parRange(
    lo: Int,
    up: Int,
    nThreads: Int,
    f: (Int) -> Unit,
) {
    if (nThreads == 1) {
        for (i in lo until up) {
            f(i)
        }
    } else {
        rayonScope { s ->
            for (threadIdx in 0 until nThreads) {
                s.spawn { _ ->
                    for (i in (threadIdx until up).stepBy(nThreads)) {
                        if (i >= lo) f(i)
                    }
                }
            }
        }
    }
}

private fun <T> List<T>.add(idx: Int): T = this[idx]

private fun <T> List<T>.asPtr(): List<T> = this

private fun <T> MutableList<T>.asMutPtr(): MutableList<T> = this

private val IntProgression.stepBy: (Int) -> IntProgression get() = { stepSize -> this step stepSize }
private val Self: U8VecOps get() = U8VecOps
private val Ord: U8VecOps get() = U8VecOps

private fun <T : Comparable<T>> U8VecOps.min(
    a: T,
    b: T,
): T = minOf(a, b)

private fun <T : Comparable<T>> U8VecOps.max(
    a: T,
    b: T,
): T = maxOf(a, b)

private val mklVsAdd: (List<Float>, List<Float>, MutableList<Float>) -> Unit = { _, _, _ -> }
private val accelerateVsAdd: (List<Float>, List<Float>, MutableList<Float>) -> Unit = { _, _, _ -> }
private val mklVdAdd: (List<Double>, List<Double>, MutableList<Double>) -> Unit = { _, _, _ -> }
private val accelerateVdAdd: (List<Double>, List<Double>, MutableList<Double>) -> Unit = { _, _, _ -> }
private val vecDotF32: (List<Float>, List<Float>, MutableList<Float>, Int) -> Unit = { lhs, rhs, res, len ->
    val out = FloatArray(1)
    io.github.kotlinmania.candle
        .vecDotF32(lhs.toFloatArray(), rhs.toFloatArray(), out, len)
    res[0] = out[0]
}
private val vecSum: (List<Float>, MutableList<Float>, Int) -> Unit = { xs, res, len ->
    val out = FloatArray(1)
    io.github.kotlinmania.candle
        .vecSum(xs.toFloatArray(), out, len)
    res[0] = out[0]
}
private val vecAddF16: (List<io.github.kotlinmania.half.F16>, List<io.github.kotlinmania.half.F16>, MutableList<io.github.kotlinmania.half.F16>, Int) -> Unit = { lhs, rhs, res, len ->
    for (i in 0 until len) {
        res[i] =
            io.github.kotlinmania.half.F16
                .fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
    }
}
private val vecScalarAddF16: (io.github.kotlinmania.half.F16, List<io.github.kotlinmania.half.F16>, MutableList<io.github.kotlinmania.half.F16>, Int) -> Unit = { scalar, xs, ys, len ->
    for (i in 0 until len) {
        ys[i] =
            io.github.kotlinmania.half.F16
                .fromFloat(xs[i].toFloat() + scalar.toFloat())
    }
}
private val vecDotF16: (List<io.github.kotlinmania.half.F16>, List<io.github.kotlinmania.half.F16>, FloatArray, Int) -> Unit = { lhs, rhs, res, len ->
    var acc = 0f
    for (i in 0 until len) acc += lhs[i].toFloat() * rhs[i].toFloat()
    res[0] = acc
}
private val vecAddBf16: (List<io.github.kotlinmania.half.BF16>, List<io.github.kotlinmania.half.BF16>, MutableList<io.github.kotlinmania.half.BF16>, Int) -> Unit = { lhs, rhs, res, len ->
    for (i in 0 until len) {
        res[i] =
            io.github.kotlinmania.half.BF16
                .fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
    }
}
private val vecScalarAddBf16: (io.github.kotlinmania.half.BF16, List<io.github.kotlinmania.half.BF16>, MutableList<io.github.kotlinmania.half.BF16>, Int) -> Unit = { scalar, xs, ys, len ->
    for (i in 0 until len) {
        ys[i] =
            io.github.kotlinmania.half.BF16
                .fromFloat(xs[i].toFloat() + scalar.toFloat())
    }
}
private val vecDotBf16: (List<io.github.kotlinmania.half.BF16>, List<io.github.kotlinmania.half.BF16>, FloatArray, Int) -> Unit = { lhs, rhs, res, len ->
    var acc = 0f
    for (i in 0 until len) acc += lhs[i].toFloat() * rhs[i].toFloat()
    res[0] = acc
}
private val U8VecOps.spawn: ((Int) -> Unit) -> Unit get() = { block -> block(0) }
private val rayonScope: ((U8VecOps) -> Unit) -> Unit = { block -> block(U8VecOps) }
