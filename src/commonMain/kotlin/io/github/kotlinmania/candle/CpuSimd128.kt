// port-lint: source cpu/simd128.rs
package io.github.kotlinmania.candle

public object CpuSimd128 {
    public object AssociatedTypes {
        public interface Unit
        public interface Array
    }

    public const val step: Int = 16
    public const val epr: Int = 4
    public const val arr: Int = step / epr

    public class CurrentCpu : Cpu<FloatArray, kotlin.Array<FloatArray>> {
        override val step: Int = CpuSimd128.step
        override val epr: Int = CpuSimd128.epr
        override val arr: Int = CpuSimd128.arr

        override fun zero(): FloatArray = FloatArray(epr)

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

        override fun load(memAddr: FloatArray, offset: Int): FloatArray =
            FloatArray(epr) { memAddr[offset + it] }

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            FloatArray(epr) { a[it] + b[it] }

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            FloatArray(epr) { b[it] * c[it] + a[it] }

        override fun vecStore(memAddr: FloatArray, offset: Int, a: FloatArray) {
            for (i in 0 until epr) {
                memAddr[offset + i] = a[i]
            }
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
            for (i in 0 until arr / 2) {
                x[2 * i] = vecAdd(x[2 * i], x[2 * i + 1])
            }
            for (i in 0 until arr / 4) {
                x[4 * i] = vecAdd(x[4 * i], x[4 * i + 2])
            }
            for (i in 0 until arr / 8) {
                x[8 * i] = vecAdd(x[8 * i], x[8 * i + 4])
            }
            y[yOffset] = x[0][0] + x[0][1] + x[0][2] + x[0][3]
        }
    }
}
