// port-lint: source cpu/kernels.rs
package io.github.kotlinmania.candle

/**
 * Vectorized and scalar CPU kernel operations over contiguous slices.
 */
public interface VecOps<T> {
    public val zero: () -> T
    public val add: (T, T) -> T
    public val mul: (T, T) -> T

    public fun min(self: T, rhs: T): T
    public fun max(self: T, rhs: T): T

    /**
     * Element-wise addition of two slices into a third.
     */
    public fun vecAdd(lhs: List<T>, rhs: List<T>, res: MutableList<T>) {
        lhs.iter()
            .zip(rhs)
            .zip(res.indices)
            .forEach { entry -> res[entry.second] = add(entry.first.first, entry.first.second) }
    }

    /**
     * Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
     */
    public fun scalarAdd(scalar: T, xs: List<T>, ys: MutableList<T>) {
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
    public fun vecDot(lhs: List<T>, rhs: List<T>, res: MutableList<T>, len: Int) {
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
    public fun vecReduceSum(xs: List<T>, res: MutableList<T>, len: Int) {
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
    public fun vecReduceMax(xs: List<T>, res: MutableList<T>, len: Int) {
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
    public fun vecReduceMin(xs: List<T>, res: MutableList<T>, len: Int) {
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

    override fun min(self: Float, rhs: Float): Float {
        return Self.min(self, rhs)
    }

    override fun max(self: Float, rhs: Float): Float {
        return Self.max(self, rhs)
    }

    override fun vecAdd(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>) {
        if (hasMkl()) mklVsAdd(lhs, rhs, res)
        if (hasAccelerate() && !hasMkl()) accelerateVsAdd(lhs, rhs, res)
        if (!hasMkl() && !hasAccelerate()) {
            lhs.iter()
                .zip(rhs)
                .zip(res.indices)
                .forEach { entry -> res[entry.second] = entry.first.first + entry.first.second }
        }
    }

    override fun vecDot(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>, len: Int) {
        vecDotF32(lhs, rhs, res, len)
    }

    override fun vecReduceSum(xs: List<Float>, res: MutableList<Float>, len: Int) {
        vecSum(xs, res, len)
    }
}

public object F16VecOps : VecOps<Float16> {
    override val zero: () -> Float16 = { Float16.fromFloat(0.0f) }
    override val add: (Float16, Float16) -> Float16 = { a, b -> Float16.fromFloat(a.toFloat() + b.toFloat()) }
    override val mul: (Float16, Float16) -> Float16 = { a, b -> Float16.fromFloat(a.toFloat() * b.toFloat()) }

    override fun min(self: Float16, rhs: Float16): Float16 {
        return Self.min(self, rhs)
    }

    override fun max(self: Float16, rhs: Float16): Float16 {
        return Self.max(self, rhs)
    }

    override fun vecAdd(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>) {
        vecAddF16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.len())
    }

    override fun scalarAdd(scalar: Float16, xs: List<Float16>, ys: MutableList<Float16>) {
        vecScalarAddF16(scalar, xs.asPtr(), ys.asMutPtr(), xs.len())
    }

    override fun vecDot(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>, len: Int) {
        val resF32 = FloatArray(1) { 0f }
        vecDotF16(lhs, rhs, resF32, len)
        res[0] = Float16.fromF32(resF32[0])
    }
}

public object F64VecOps : VecOps<Double> {
    override val zero: () -> Double = { 0.0 }
    override val add: (Double, Double) -> Double = { a, b -> a + b }
    override val mul: (Double, Double) -> Double = { a, b -> a * b }

    override fun min(self: Double, rhs: Double): Double {
        return Self.min(self, rhs)
    }

    override fun max(self: Double, rhs: Double): Double {
        return Self.max(self, rhs)
    }

    override fun vecAdd(lhs: List<Double>, rhs: List<Double>, res: MutableList<Double>) {
        if (hasMkl()) mklVdAdd(lhs, rhs, res)
        if (hasAccelerate() && !hasMkl()) accelerateVdAdd(lhs, rhs, res)
        if (!hasMkl() && !hasAccelerate()) {
            lhs.iter()
                .zip(rhs)
                .zip(res.indices)
                .forEach { entry -> res[entry.second] = entry.first.first + entry.first.second }
        }
    }
}

public object BF16VecOps : VecOps<BFloat16> {
    override val zero: () -> BFloat16 = { BFloat16.fromFloat(0.0f) }
    override val add: (BFloat16, BFloat16) -> BFloat16 = { a, b -> BFloat16.fromFloat(a.toFloat() + b.toFloat()) }
    override val mul: (BFloat16, BFloat16) -> BFloat16 = { a, b -> BFloat16.fromFloat(a.toFloat() * b.toFloat()) }

    override fun min(self: BFloat16, rhs: BFloat16): BFloat16 {
        return Self.min(self, rhs)
    }

    override fun max(self: BFloat16, rhs: BFloat16): BFloat16 {
        return Self.max(self, rhs)
    }

    override fun vecAdd(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>) {
        vecAddBf16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.len())
    }

    override fun scalarAdd(scalar: BFloat16, xs: List<BFloat16>, ys: MutableList<BFloat16>) {
        vecScalarAddBf16(scalar, xs.asPtr(), ys.asMutPtr(), xs.len())
    }

    override fun vecDot(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>, len: Int) {
        val resF32 = FloatArray(1) { 0f }
        vecDotBf16(lhs, rhs, resF32, len)
        res[0] = BFloat16.fromF32(resF32[0])
    }
}

public object U8VecOps : VecOps<UByte> {
    override val zero: () -> UByte = { 0u }
    override val add: (UByte, UByte) -> UByte = { a, b -> (a + b).toUByte() }
    override val mul: (UByte, UByte) -> UByte = { a, b -> (a * b).toUByte() }

    override fun min(self: UByte, rhs: UByte): UByte {
        return Ord.min(self, rhs)
    }

    override fun max(self: UByte, rhs: UByte): UByte {
        return Ord.max(self, rhs)
    }
}

public object U32VecOps : VecOps<UInt> {
    override val zero: () -> UInt = { 0u }
    override val add: (UInt, UInt) -> UInt = { a, b -> a + b }
    override val mul: (UInt, UInt) -> UInt = { a, b -> a * b }

    override fun min(self: UInt, rhs: UInt): UInt {
        return Ord.min(self, rhs)
    }

    override fun max(self: UInt, rhs: UInt): UInt {
        return Ord.max(self, rhs)
    }
}

public object I16VecOps : VecOps<Short> {
    override val zero: () -> Short = { 0 }
    override val add: (Short, Short) -> Short = { a, b -> (a + b).toShort() }
    override val mul: (Short, Short) -> Short = { a, b -> (a * b).toShort() }

    override fun min(self: Short, rhs: Short): Short {
        return Ord.min(self, rhs)
    }

    override fun max(self: Short, rhs: Short): Short {
        return Ord.max(self, rhs)
    }
}

public object I32VecOps : VecOps<Int> {
    override val zero: () -> Int = { 0 }
    override val add: (Int, Int) -> Int = { a, b -> a + b }
    override val mul: (Int, Int) -> Int = { a, b -> a * b }

    override fun min(self: Int, rhs: Int): Int {
        return Ord.min(self, rhs)
    }

    override fun max(self: Int, rhs: Int): Int {
        return Ord.max(self, rhs)
    }
}

public object I64VecOps : VecOps<Long> {
    override val zero: () -> Long = { 0L }
    override val add: (Long, Long) -> Long = { a, b -> a + b }
    override val mul: (Long, Long) -> Long = { a, b -> a * b }

    override fun min(self: Long, rhs: Long): Long {
        return Ord.min(self, rhs)
    }

    override fun max(self: Long, rhs: Long): Long {
        return Ord.max(self, rhs)
    }
}

public object F8E4M3VecOps : VecOps<Byte> {
    override val zero: () -> Byte = { 0 }
    override val add: (Byte, Byte) -> Byte = { a, b -> (a + b).toByte() }
    override val mul: (Byte, Byte) -> Byte = { a, b -> (a * b).toByte() }

    override fun min(self: Byte, rhs: Byte): Byte {
        return Self.min(self, rhs)
    }

    override fun max(self: Byte, rhs: Byte): Byte {
        return Self.max(self, rhs)
    }
}

public fun parForEach(nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        func(0)
    } else {
        rayonScope { s ->
            for (threadIdx in 0 until nThreads) {
                val funcRef = func
                s.spawn { _ -> funcRef(threadIdx) }
            }
        }
    }
}

public fun parRange(lo: Int, up: Int, nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        for (i in lo until up) {
            func(i)
        }
    } else {
        rayonScope { s ->
            for (threadIdx in 0 until nThreads) {
                val funcRef = func
                s.spawn { _ ->
                    for (i in (threadIdx until up).stepBy(nThreads)) {
                        if (i >= lo) func(i)
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
private fun <T : Comparable<T>> U8VecOps.min(a: T, b: T): T = minOf(a, b)
private fun <T : Comparable<T>> U8VecOps.max(a: T, b: T): T = maxOf(a, b)
private val mklVsAdd: (List<Float>, List<Float>, MutableList<Float>) -> Unit = { _, _, _ -> }
private val accelerateVsAdd: (List<Float>, List<Float>, MutableList<Float>) -> Unit = { _, _, _ -> }
private val mklVdAdd: (List<Double>, List<Double>, MutableList<Double>) -> Unit = { _, _, _ -> }
private val accelerateVdAdd: (List<Double>, List<Double>, MutableList<Double>) -> Unit = { _, _, _ -> }
private val vecDotF32: (List<Float>, List<Float>, MutableList<Float>, Int) -> Unit = { lhs, rhs, res, len ->
    val out = FloatArray(1)
    io.github.kotlinmania.candle.vecDotF32(lhs.toFloatArray(), rhs.toFloatArray(), out, len)
    res[0] = out[0]
}
private val vecSum: (List<Float>, MutableList<Float>, Int) -> Unit = { xs, res, len ->
    val out = FloatArray(1)
    io.github.kotlinmania.candle.vecSum(xs.toFloatArray(), out, len)
    res[0] = out[0]
}
private val vecAddF16: (List<Float16>, List<Float16>, MutableList<Float16>, Int) -> Unit = { lhs, rhs, res, len ->
    for (i in 0 until len) res[i] = Float16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
}
private val vecScalarAddF16: (Float16, List<Float16>, MutableList<Float16>, Int) -> Unit = { scalar, xs, ys, len ->
    for (i in 0 until len) ys[i] = Float16.fromFloat(xs[i].toFloat() + scalar.toFloat())
}
private val vecDotF16: (List<Float16>, List<Float16>, FloatArray, Int) -> Unit = { lhs, rhs, res, len ->
    var acc = 0f
    for (i in 0 until len) acc += lhs[i].toFloat() * rhs[i].toFloat()
    res[0] = acc
}
private val vecAddBf16: (List<BFloat16>, List<BFloat16>, MutableList<BFloat16>, Int) -> Unit = { lhs, rhs, res, len ->
    for (i in 0 until len) res[i] = BFloat16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
}
private val vecScalarAddBf16: (BFloat16, List<BFloat16>, MutableList<BFloat16>, Int) -> Unit = { scalar, xs, ys, len ->
    for (i in 0 until len) ys[i] = BFloat16.fromFloat(xs[i].toFloat() + scalar.toFloat())
}
private val vecDotBf16: (List<BFloat16>, List<BFloat16>, FloatArray, Int) -> Unit = { lhs, rhs, res, len ->
    var acc = 0f
    for (i in 0 until len) acc += lhs[i].toFloat() * rhs[i].toFloat()
    res[0] = acc
}
private val U8VecOps.spawn: ((Int) -> Unit) -> Unit get() = { block -> block(0) }
private val rayonScope: ((U8VecOps) -> Unit) -> Unit = { block -> block(U8VecOps) }
