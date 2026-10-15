// port-lint: source cpu/kernels.rs
package io.github.kotlinmania.candle

/**
 * Vectorized and scalar CPU kernel operations over contiguous slices.
 */
public interface VecOps<T> {
    public val zero: () -> T
    public val add: (T, T) -> T
    public val mul: (T, T) -> T

    /**
     * Element-wise minimum of two scalar values.
     */
    public fun min(self: T, rhs: T): T

    /**
     * Element-wise maximum of two scalar values.
     */
    public fun max(self: T, rhs: T): T

    /**
     * Element-wise addition of two slices into a third.
     */
    public fun vecAdd(lhs: List<T>, rhs: List<T>, res: MutableList<T>) {
        lhs.zip(rhs).forEachIndexed { idx, (a, b) ->
            res[idx] = add(a, b)
        }
    }

    /**
     * Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
     */
    public fun scalarAdd(scalar: T, xs: List<T>, ys: MutableList<T>) {
        xs.forEachIndexed { idx, x ->
            ys[idx] = add(x, scalar)
        }
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
            res[0] = add(res[0], mul(lhs[i], rhs[i]))
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
            res[0] = add(res[0], xs[i])
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
            res[0] = max(res[0], xs[i])
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
            res[0] = min(res[0], xs[i])
        }
    }
}

public object F32VecOps : VecOps<Float> {
    override val zero: () -> Float = { 0.0f }
    override val add: (Float, Float) -> Float = { a, b -> a + b }
    override val mul: (Float, Float) -> Float = { a, b -> a * b }

    override fun min(self: Float, rhs: Float): Float = minOf(self, rhs)

    override fun max(self: Float, rhs: Float): Float = maxOf(self, rhs)

    override fun vecAdd(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>) {
        lhs.zip(rhs).forEachIndexed { idx, (a, b) ->
            res[idx] = a + b
        }
    }

    override fun vecDot(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>, len: Int) {
        val out = FloatArray(1)
        vecDotF32(lhs.toFloatArray(), rhs.toFloatArray(), out, len)
        res[0] = out[0]
    }

    override fun vecReduceSum(xs: List<Float>, res: MutableList<Float>, len: Int) {
        val out = FloatArray(1)
        vecSum(xs.toFloatArray(), out, len)
        res[0] = out[0]
    }
}

public object F16VecOps : VecOps<Float16> {
    override val zero: () -> Float16 = { Float16.fromFloat(0.0f) }
    override val add: (Float16, Float16) -> Float16 = { a, b -> Float16.fromFloat(a.toFloat() + b.toFloat()) }
    override val mul: (Float16, Float16) -> Float16 = { a, b -> Float16.fromFloat(a.toFloat() * b.toFloat()) }

    override fun min(self: Float16, rhs: Float16): Float16 =
        Float16.fromFloat(minOf(self.toFloat(), rhs.toFloat()))

    override fun max(self: Float16, rhs: Float16): Float16 =
        Float16.fromFloat(maxOf(self.toFloat(), rhs.toFloat()))

    override fun vecAdd(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>) {
        for (i in lhs.indices) {
            res[i] = Float16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
        }
    }

    override fun scalarAdd(scalar: Float16, xs: List<Float16>, ys: MutableList<Float16>) {
        for (i in xs.indices) {
            ys[i] = Float16.fromFloat(xs[i].toFloat() + scalar.toFloat())
        }
    }

    override fun vecDot(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>, len: Int) {
        var resF32 = 0.0f
        for (i in 0 until len) {
            resF32 += lhs[i].toFloat() * rhs[i].toFloat()
        }
        res[0] = Float16.fromFloat(resF32)
    }
}

public object F64VecOps : VecOps<Double> {
    override val zero: () -> Double = { 0.0 }
    override val add: (Double, Double) -> Double = { a, b -> a + b }
    override val mul: (Double, Double) -> Double = { a, b -> a * b }

    override fun min(self: Double, rhs: Double): Double = minOf(self, rhs)

    override fun max(self: Double, rhs: Double): Double = maxOf(self, rhs)

    override fun vecAdd(lhs: List<Double>, rhs: List<Double>, res: MutableList<Double>) {
        lhs.zip(rhs).forEachIndexed { idx, (a, b) ->
            res[idx] = a + b
        }
    }
}

public object BF16VecOps : VecOps<BFloat16> {
    override val zero: () -> BFloat16 = { BFloat16.fromFloat(0.0f) }
    override val add: (BFloat16, BFloat16) -> BFloat16 = { a, b -> BFloat16.fromFloat(a.toFloat() + b.toFloat()) }
    override val mul: (BFloat16, BFloat16) -> BFloat16 = { a, b -> BFloat16.fromFloat(a.toFloat() * b.toFloat()) }

    override fun min(self: BFloat16, rhs: BFloat16): BFloat16 =
        BFloat16.fromFloat(minOf(self.toFloat(), rhs.toFloat()))

    override fun max(self: BFloat16, rhs: BFloat16): BFloat16 =
        BFloat16.fromFloat(maxOf(self.toFloat(), rhs.toFloat()))

    override fun vecAdd(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>) {
        for (i in lhs.indices) {
            res[i] = BFloat16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
        }
    }

    override fun scalarAdd(scalar: BFloat16, xs: List<BFloat16>, ys: MutableList<BFloat16>) {
        for (i in xs.indices) {
            ys[i] = BFloat16.fromFloat(xs[i].toFloat() + scalar.toFloat())
        }
    }

    override fun vecDot(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>, len: Int) {
        var resF32 = 0.0f
        for (i in 0 until len) {
            resF32 += lhs[i].toFloat() * rhs[i].toFloat()
        }
        res[0] = BFloat16.fromFloat(resF32)
    }
}

public object U8VecOps : VecOps<UByte> {
    override val zero: () -> UByte = { 0u }
    override val add: (UByte, UByte) -> UByte = { a, b -> (a + b).toUByte() }
    override val mul: (UByte, UByte) -> UByte = { a, b -> (a * b).toUByte() }

    override fun min(self: UByte, rhs: UByte): UByte = minOf(self, rhs)

    override fun max(self: UByte, rhs: UByte): UByte = maxOf(self, rhs)
}

public object U32VecOps : VecOps<UInt> {
    override val zero: () -> UInt = { 0u }
    override val add: (UInt, UInt) -> UInt = { a, b -> a + b }
    override val mul: (UInt, UInt) -> UInt = { a, b -> a * b }

    override fun min(self: UInt, rhs: UInt): UInt = minOf(self, rhs)

    override fun max(self: UInt, rhs: UInt): UInt = maxOf(self, rhs)
}

public object I16VecOps : VecOps<Short> {
    override val zero: () -> Short = { 0 }
    override val add: (Short, Short) -> Short = { a, b -> (a + b).toShort() }
    override val mul: (Short, Short) -> Short = { a, b -> (a * b).toShort() }

    override fun min(self: Short, rhs: Short): Short = minOf(self, rhs)

    override fun max(self: Short, rhs: Short): Short = maxOf(self, rhs)
}

public object I32VecOps : VecOps<Int> {
    override val zero: () -> Int = { 0 }
    override val add: (Int, Int) -> Int = { a, b -> a + b }
    override val mul: (Int, Int) -> Int = { a, b -> a * b }

    override fun min(self: Int, rhs: Int): Int = minOf(self, rhs)

    override fun max(self: Int, rhs: Int): Int = maxOf(self, rhs)
}

public object I64VecOps : VecOps<Long> {
    override val zero: () -> Long = { 0L }
    override val add: (Long, Long) -> Long = { a, b -> a + b }
    override val mul: (Long, Long) -> Long = { a, b -> a * b }

    override fun min(self: Long, rhs: Long): Long = minOf(self, rhs)

    override fun max(self: Long, rhs: Long): Long = maxOf(self, rhs)
}

public object F8E4M3VecOps : VecOps<Byte> {
    override val zero: () -> Byte = { 0 }
    override val add: (Byte, Byte) -> Byte = { a, b -> (a + b).toByte() }
    override val mul: (Byte, Byte) -> Byte = { a, b -> (a * b).toByte() }

    override fun min(self: Byte, rhs: Byte): Byte = minOf(self, rhs)

    override fun max(self: Byte, rhs: Byte): Byte = maxOf(self, rhs)
}

public fun parForEach(nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        func(0)
    } else {
        for (threadIdx in 0 until nThreads) {
            val f = func
            f(threadIdx)
        }
    }
}

public fun parRange(lo: Int, up: Int, nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        for (i in lo until up) {
            func(i)
        }
    } else {
        for (threadIdx in 0 until nThreads) {
            val f = func
            for (i in (threadIdx until up) step nThreads) {
                if (i >= lo) {
                    f(i)
                }
            }
        }
    }
}
