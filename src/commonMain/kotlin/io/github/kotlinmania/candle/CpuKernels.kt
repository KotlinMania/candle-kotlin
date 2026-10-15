// port-lint: source cpu/kernels.rs
package io.github.kotlinmania.candle

/**
 * Vectorized and scalar CPU kernel operations over contiguous slices.
 */
public interface VecOps<T> {
    /**
     * Element-wise minimum of two scalar values.
     */
    public fun min(lhs: T, rhs: T): T

    /**
     * Element-wise maximum of two scalar values.
     */
    public fun max(lhs: T, rhs: T): T

    /**
     * Element-wise addition of two slices into a third.
     *
     * # Safety
     *
     * All slices must have matching lengths.
     */
    public fun vecAdd(lhs: List<T>, rhs: List<T>, res: MutableList<T>)

    /**
     * Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
     *
     * # Safety
     *
     * Input and output slices must have matching lengths.
     */
    public fun scalarAdd(scalar: T, xs: List<T>, ys: MutableList<T>)

    /**
     * Dot-product of two vectors.
     *
     * # Safety
     *
     * The length of `lhs` and `rhs` have to be at least `len`. `res` has to point to a valid
     * element.
     */
    public fun vecDot(lhs: List<T>, rhs: List<T>, res: MutableList<T>, len: Int)

    /**
     * Sum of all elements in a vector.
     *
     * # Safety
     *
     * The length of `xs` must be at least `len`. `res` has to point to a valid
     * element.
     */
    public fun vecReduceSum(xs: List<T>, res: MutableList<T>, len: Int)

    /**
     * Maximum element in a non-empty vector.
     *
     * # Safety
     *
     * The length of `xs` must be at least `len` and positive. `res` has to point to a valid
     * element.
     */
    public fun vecReduceMax(xs: List<T>, res: MutableList<T>, len: Int) {
        var acc = xs[0]
        for (i in 1 until len) {
            acc = max(acc, xs[i])
        }
        res[0] = acc
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
        var acc = xs[0]
        for (i in 1 until len) {
            acc = min(acc, xs[i])
        }
        res[0] = acc
    }
}

public object F32VecOps : VecOps<Float> {
    override fun min(lhs: Float, rhs: Float): Float = minOf(lhs, rhs)
    override fun max(lhs: Float, rhs: Float): Float = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>) {
        for (i in lhs.indices) {
            res[i] = lhs[i] + rhs[i]
        }
    }

    override fun scalarAdd(scalar: Float, xs: List<Float>, ys: MutableList<Float>) {
        for (i in xs.indices) {
            ys[i] = xs[i] + scalar
        }
    }

    override fun vecDot(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += lhs[i] * rhs[i]
        }
        res[0] = acc
    }

    override fun vecReduceSum(xs: List<Float>, res: MutableList<Float>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += xs[i]
        }
        res[0] = acc
    }
}

public object F16VecOps : VecOps<Float16> {
    override fun min(lhs: Float16, rhs: Float16): Float16 = Float16.fromFloat(minOf(lhs.toFloat(), rhs.toFloat()))
    override fun max(lhs: Float16, rhs: Float16): Float16 = Float16.fromFloat(maxOf(lhs.toFloat(), rhs.toFloat()))

    override fun vecAdd(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>) {
        for (i in lhs.indices) {
            res[i] = Float16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
        }
    }

    override fun scalarAdd(scalar: Float16, xs: List<Float16>, ys: MutableList<Float16>) {
        val sv = scalar.toFloat()
        for (i in xs.indices) {
            ys[i] = Float16.fromFloat(xs[i].toFloat() + sv)
        }
    }

    override fun vecDot(lhs: List<Float16>, rhs: List<Float16>, res: MutableList<Float16>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += lhs[i].toFloat() * rhs[i].toFloat()
        }
        res[0] = Float16.fromFloat(acc)
    }

    override fun vecReduceSum(xs: List<Float16>, res: MutableList<Float16>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += xs[i].toFloat()
        }
        res[0] = Float16.fromFloat(acc)
    }
}

public object F64VecOps : VecOps<Double> {
    override fun min(lhs: Double, rhs: Double): Double = minOf(lhs, rhs)
    override fun max(lhs: Double, rhs: Double): Double = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Double>, rhs: List<Double>, res: MutableList<Double>) {
        for (i in lhs.indices) {
            res[i] = lhs[i] + rhs[i]
        }
    }

    override fun scalarAdd(scalar: Double, xs: List<Double>, ys: MutableList<Double>) {
        for (i in xs.indices) {
            ys[i] = xs[i] + scalar
        }
    }

    override fun vecDot(lhs: List<Double>, rhs: List<Double>, res: MutableList<Double>, len: Int) {
        var acc = 0.0
        for (i in 0 until len) {
            acc += lhs[i] * rhs[i]
        }
        res[0] = acc
    }

    override fun vecReduceSum(xs: List<Double>, res: MutableList<Double>, len: Int) {
        var acc = 0.0
        for (i in 0 until len) {
            acc += xs[i]
        }
        res[0] = acc
    }
}

public object BF16VecOps : VecOps<BFloat16> {
    override fun min(lhs: BFloat16, rhs: BFloat16): BFloat16 = BFloat16.fromFloat(minOf(lhs.toFloat(), rhs.toFloat()))
    override fun max(lhs: BFloat16, rhs: BFloat16): BFloat16 = BFloat16.fromFloat(maxOf(lhs.toFloat(), rhs.toFloat()))

    override fun vecAdd(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>) {
        for (i in lhs.indices) {
            res[i] = BFloat16.fromFloat(lhs[i].toFloat() + rhs[i].toFloat())
        }
    }

    override fun scalarAdd(scalar: BFloat16, xs: List<BFloat16>, ys: MutableList<BFloat16>) {
        val sv = scalar.toFloat()
        for (i in xs.indices) {
            ys[i] = BFloat16.fromFloat(xs[i].toFloat() + sv)
        }
    }

    override fun vecDot(lhs: List<BFloat16>, rhs: List<BFloat16>, res: MutableList<BFloat16>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += lhs[i].toFloat() * rhs[i].toFloat()
        }
        res[0] = BFloat16.fromFloat(acc)
    }

    override fun vecReduceSum(xs: List<BFloat16>, res: MutableList<BFloat16>, len: Int) {
        var acc = 0.0f
        for (i in 0 until len) {
            acc += xs[i].toFloat()
        }
        res[0] = BFloat16.fromFloat(acc)
    }
}

public object U8VecOps : VecOps<UByte> {
    override fun min(lhs: UByte, rhs: UByte): UByte = minOf(lhs, rhs)
    override fun max(lhs: UByte, rhs: UByte): UByte = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<UByte>, rhs: List<UByte>, res: MutableList<UByte>) {
        for (i in lhs.indices) res[i] = (lhs[i] + rhs[i]).toUByte()
    }

    override fun scalarAdd(scalar: UByte, xs: List<UByte>, ys: MutableList<UByte>) {
        for (i in xs.indices) ys[i] = (xs[i] + scalar).toUByte()
    }

    override fun vecDot(lhs: List<UByte>, rhs: List<UByte>, res: MutableList<UByte>, len: Int) {
        var acc = 0u
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc.toUByte()
    }

    override fun vecReduceSum(xs: List<UByte>, res: MutableList<UByte>, len: Int) {
        var acc = 0u
        for (i in 0 until len) acc += xs[i]
        res[0] = acc.toUByte()
    }
}

public object U32VecOps : VecOps<UInt> {
    override fun min(lhs: UInt, rhs: UInt): UInt = minOf(lhs, rhs)
    override fun max(lhs: UInt, rhs: UInt): UInt = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<UInt>, rhs: List<UInt>, res: MutableList<UInt>) {
        for (i in lhs.indices) res[i] = lhs[i] + rhs[i]
    }

    override fun scalarAdd(scalar: UInt, xs: List<UInt>, ys: MutableList<UInt>) {
        for (i in xs.indices) ys[i] = xs[i] + scalar
    }

    override fun vecDot(lhs: List<UInt>, rhs: List<UInt>, res: MutableList<UInt>, len: Int) {
        var acc = 0u
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc
    }

    override fun vecReduceSum(xs: List<UInt>, res: MutableList<UInt>, len: Int) {
        var acc = 0u
        for (i in 0 until len) acc += xs[i]
        res[0] = acc
    }
}

public object I16VecOps : VecOps<Short> {
    override fun min(lhs: Short, rhs: Short): Short = minOf(lhs, rhs)
    override fun max(lhs: Short, rhs: Short): Short = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Short>, rhs: List<Short>, res: MutableList<Short>) {
        for (i in lhs.indices) res[i] = (lhs[i] + rhs[i]).toShort()
    }

    override fun scalarAdd(scalar: Short, xs: List<Short>, ys: MutableList<Short>) {
        for (i in xs.indices) ys[i] = (xs[i] + scalar).toShort()
    }

    override fun vecDot(lhs: List<Short>, rhs: List<Short>, res: MutableList<Short>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc.toShort()
    }

    override fun vecReduceSum(xs: List<Short>, res: MutableList<Short>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += xs[i]
        res[0] = acc.toShort()
    }
}

public object I32VecOps : VecOps<Int> {
    override fun min(lhs: Int, rhs: Int): Int = minOf(lhs, rhs)
    override fun max(lhs: Int, rhs: Int): Int = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Int>, rhs: List<Int>, res: MutableList<Int>) {
        for (i in lhs.indices) res[i] = lhs[i] + rhs[i]
    }

    override fun scalarAdd(scalar: Int, xs: List<Int>, ys: MutableList<Int>) {
        for (i in xs.indices) ys[i] = xs[i] + scalar
    }

    override fun vecDot(lhs: List<Int>, rhs: List<Int>, res: MutableList<Int>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc
    }

    override fun vecReduceSum(xs: List<Int>, res: MutableList<Int>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += xs[i]
        res[0] = acc
    }
}

public object I64VecOps : VecOps<Long> {
    override fun min(lhs: Long, rhs: Long): Long = minOf(lhs, rhs)
    override fun max(lhs: Long, rhs: Long): Long = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Long>, rhs: List<Long>, res: MutableList<Long>) {
        for (i in lhs.indices) res[i] = lhs[i] + rhs[i]
    }

    override fun scalarAdd(scalar: Long, xs: List<Long>, ys: MutableList<Long>) {
        for (i in xs.indices) ys[i] = xs[i] + scalar
    }

    override fun vecDot(lhs: List<Long>, rhs: List<Long>, res: MutableList<Long>, len: Int) {
        var acc = 0L
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc
    }

    override fun vecReduceSum(xs: List<Long>, res: MutableList<Long>, len: Int) {
        var acc = 0L
        for (i in 0 until len) acc += xs[i]
        res[0] = acc
    }
}

public object F8E4M3VecOps : VecOps<Byte> {
    override fun min(lhs: Byte, rhs: Byte): Byte = minOf(lhs, rhs)
    override fun max(lhs: Byte, rhs: Byte): Byte = maxOf(lhs, rhs)

    override fun vecAdd(lhs: List<Byte>, rhs: List<Byte>, res: MutableList<Byte>) {
        for (i in lhs.indices) res[i] = (lhs[i] + rhs[i]).toByte()
    }

    override fun scalarAdd(scalar: Byte, xs: List<Byte>, ys: MutableList<Byte>) {
        for (i in xs.indices) ys[i] = (xs[i] + scalar).toByte()
    }

    override fun vecDot(lhs: List<Byte>, rhs: List<Byte>, res: MutableList<Byte>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += lhs[i] * rhs[i]
        res[0] = acc.toByte()
    }

    override fun vecReduceSum(xs: List<Byte>, res: MutableList<Byte>, len: Int) {
        var acc = 0
        for (i in 0 until len) acc += xs[i]
        res[0] = acc.toByte()
    }
}

public fun parForEach(nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        func(0)
    } else {
        for (threadIdx in 0 until nThreads) {
            func(threadIdx)
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
            var i = threadIdx
            while (i < up) {
                if (i >= lo) {
                    func(i)
                }
                i += nThreads
            }
        }
    }
}
