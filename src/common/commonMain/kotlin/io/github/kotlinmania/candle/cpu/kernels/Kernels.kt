// port-lint: source cpu/kernels.rs
package io.github.kotlinmania.candle.cpu.kernels

import io.github.kotlinmania.candle.cpu.vecAddBf16
import io.github.kotlinmania.candle.cpu.vecAddF16
import io.github.kotlinmania.candle.cpu.vecDotBf16
import io.github.kotlinmania.candle.cpu.vecDotF16
import io.github.kotlinmania.candle.cpu.vecDotF32
import io.github.kotlinmania.candle.cpu.vecScalarAddBf16
import io.github.kotlinmania.candle.cpu.vecScalarAddF16
import io.github.kotlinmania.candle.cpu.vecSum
import io.github.kotlinmania.candle.vdAdd
import io.github.kotlinmania.candle.vsAdd
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.half.f16.fromF32
import io.github.kotlinmania.numtraits.NumAssign
import io.github.kotlinmania.rayon.scope
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public interface VecOps<T> : NumAssign {
    public fun min(rhs: T): T

    public fun max(rhs: T): T

    /**
     * Element-wise addition of two slices into a third.
     */
    public fun vecAdd(lhs: List<T>, rhs: List<T>, res: MutableList<T>) {
        for (i in 0 until minOf(lhs.size, rhs.size, res.size)) {
            res[i] = lhs[i] + rhs[i]
        }
    }

    public fun vecMul(lhs: List<T>, rhs: List<T>, res: MutableList<T>) {
        for (i in 0 until minOf(lhs.size, rhs.size, res.size)) {
            res[i] = lhs[i] * rhs[i]
        }
    }

    public fun scalarMul(scalar: T, xs: List<T>, ys: MutableList<T>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = xs[i] * scalar
        }
    }

    /**
     * Add a broadcast scalar to every element of a slice: `ys[i] = xs[i] + scalar`.
     */
    public fun scalarAdd(scalar: T, xs: List<T>, ys: MutableList<T>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = xs[i] + scalar
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
    public fun vecDot(lhs: ConstPtr<T>, rhs: ConstPtr<T>, res: MutPtr<T>, len: Int) {
        val res1 = res
        (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, T.zero())
        for (i in 0 until len) {
            (res1).let { targetPtr ->
                val targetLayout = ObjectSlotLayout<T>()
                var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                currentValue += ((ObjectSlotLayout<T>()).readNative((ConstPtr((lhs).rawAddress + ((i).toLong() * (ObjectSlotLayout<T>()).strideBytes))).rawAddress) * (ObjectSlotLayout<T>()).readNative((ConstPtr((rhs).rawAddress + ((i).toLong() * (ObjectSlotLayout<T>()).strideBytes))).rawAddress))
                targetLayout.writeNative(targetPtr.rawAddress, currentValue)
            }
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
    public fun vecReduceSum(xs: ConstPtr<T>, res: MutPtr<T>, len: Int) {
        val res1 = res
        (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, T.zero())
        for (i in 0 until len) {
            (res1).let { targetPtr ->
                val targetLayout = ObjectSlotLayout<T>()
                var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                currentValue += ((ObjectSlotLayout<T>()).readNative((ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<T>()).strideBytes))).rawAddress))
                targetLayout.writeNative(targetPtr.rawAddress, currentValue)
            }
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
    public fun vecReduceMax(xs: ConstPtr<T>, res: MutPtr<T>, len: Int) {
        val res1 = res
        (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, (ObjectSlotLayout<T>()).readNative((xs).rawAddress))
        for (i in 1 until len) {
            (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, (ObjectSlotLayout<T>()).readNative((res1).rawAddress).max((ObjectSlotLayout<T>()).readNative((ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<T>()).strideBytes))).rawAddress)))
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
    public fun vecReduceMin(xs: ConstPtr<T>, res: MutPtr<T>, len: Int) {
        val res1 = res
        (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, (ObjectSlotLayout<T>()).readNative((xs).rawAddress))
        for (i in 1 until len) {
            (ObjectSlotLayout<T>()).writeNative((res1).rawAddress, (ObjectSlotLayout<T>()).readNative((res1).rawAddress).min((ObjectSlotLayout<T>()).readNative((ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<T>()).strideBytes))).rawAddress)))
        }
    }
}

public fun Float.min(other: Float): Float = Float.min(this, other)

public fun Float.max(other: Float): Float = Float.max(this, other)

public fun vecOpsFloat(lhs: List<Float>, rhs: List<Float>, res: MutableList<Float>) {
    for (i in 0 until minOf(lhs.size, rhs.size, res.size)) {
        res[i] = lhs[i] + rhs[i]
    }
}

public fun vecOpsFloat(lhs: ConstPtr<Float>, rhs: ConstPtr<Float>, res: MutPtr<Float>, len: ULong) {
    vecDotF32(lhs, rhs, res, len.toInt())
}

public fun vecOpsFloat(xs: ConstPtr<Float>, res: MutPtr<Float>, len: ULong) {
    vecSum(xs, res, len.toInt())
}

public fun F16.min(other: F16): F16 = F16.min(this, other)

public fun F16.max(other: F16): F16 = F16.max(this, other)

public fun vecOpsF16(lhs: List<F16>, rhs: List<F16>, res: List<F16>) {
    vecAddF16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.size)
}

public fun vecOpsF16(scalar: F16, xs: List<F16>, ys: List<F16>) {
    vecScalarAddF16(scalar, xs.asPtr(), ys.asMutPtr(), xs.size)
}

public fun vecOpsF16(lhs: ConstPtr<F16>, rhs: ConstPtr<F16>, res: MutPtr<F16>, len: Int) {
    val res1 = res
    val resF32 = 0.0f
    vecDotF16(lhs, rhs, resF32, len)
    (ObjectSlotLayout<F16>()).writeNative((res1).rawAddress, fromF32(resF32))
}

public fun Double.min(other: Double): Double = Double.min(this, other)

public fun Double.max(other: Double): Double = Double.max(this, other)

public fun vecOpsDouble(lhs: List<Double>, rhs: List<Double>, res: MutableList<Double>) {
    for (i in 0 until minOf(lhs.size, rhs.size, res.size)) {
        res[i] = lhs[i] + rhs[i]
    }
}

public fun Bf16.min(other: Bf16): Bf16 = Bf16.min(this, other)

public fun Bf16.max(other: Bf16): Bf16 = Bf16.max(this, other)

public fun vecOpsBf16(lhs: List<Bf16>, rhs: List<Bf16>, res: List<Bf16>) {
    vecAddBf16(lhs.asPtr(), rhs.asPtr(), res.asMutPtr(), lhs.size)
}

public fun vecOpsBf16(scalar: Bf16, xs: List<Bf16>, ys: List<Bf16>) {
    vecScalarAddBf16(scalar, xs.asPtr(), ys.asMutPtr(), xs.size)
}

public fun vecOpsBf16(lhs: ConstPtr<Bf16>, rhs: ConstPtr<Bf16>, res: MutPtr<Bf16>, len: Int) {
    val res1 = res
    val resF32 = 0.0f
    vecDotBf16(lhs, rhs, resF32, len)
    (ObjectSlotLayout<Bf16>()).writeNative(
        (res1).rawAddress,
        io.github.kotlinmania.half.bf16
            .fromF32(resF32),
    )
}

public fun UByte.min(other: UByte): UByte = UByte.min(this, other)

public fun UByte.max(other: UByte): UByte = UByte.max(this, other)

public fun UInt.min(other: UInt): UInt = UInt.min(this, other)

public fun UInt.max(other: UInt): UInt = UInt.max(this, other)

public fun Short.min(other: Short): Short = Short.min(this, other)

public fun Short.max(other: Short): Short = Short.max(this, other)

public fun Int.min(other: Int): Int = Int.min(this, other)

public fun Int.max(other: Int): Int = Int.max(this, other)

public fun Long.min(other: Long): Long = Long.min(this, other)

public fun Long.max(other: Long): Long = Long.max(this, other)

public fun F8e4m3.min(other: F8e4m3): F8e4m3 = F8e4m3.min(this, other)

public fun F8e4m3.max(other: F8e4m3): F8e4m3 = F8e4m3.max(this, other)

public fun parForEach(nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        func(0)
    } else {
        scope({ s ->
            for (threadIdx in 0 until nThreads) {
                val func1 = func
                s.spawn({ func1(threadIdx) })
            }
        })
    }
}

public fun parRange(lo: Int, up: Int, nThreads: Int, func: (Int) -> Unit) {
    if (nThreads == 1) {
        for (i in lo until up) {
            func(i)
        }
    } else {
        scope({ s ->
            for (threadIdx in 0 until nThreads) {
                val func1 = func
                s.spawn({
                    for (i1 in (threadIdx until up).stepBy(nThreads)) {
                        func1(i1)
                    }
                })
            }
        })
    }
}
