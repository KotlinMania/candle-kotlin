// port-lint: source cpu/simd128.rs
package io.github.kotlinmania.candle

private val f32x4Splat: (Float) -> FloatArray = { v -> FloatArray(4) { v } }
private val v128Load: (FloatArray, Int) -> FloatArray = { mem, off -> FloatArray(4) { mem[off + it] } }
private val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] + b[it] } }
private val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] * b[it] } }
private val v128Store: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    for (i in 0 until 4) mem[off + i] = a[i]
}
private val f32x4ExtractLane: (FloatArray, Int) -> Float = { x, lane -> x[lane] }

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

        override fun zero(): FloatArray =
            f32x4Splat(0.0f)

        override fun zeroArray(): Array<FloatArray> =
            Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray =
            f32x4Splat(v)

        override fun load(memAddr: FloatArray, offset: Int): FloatArray =
            v128Load(memAddr, offset)

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            f32x4Add(a, b)

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            f32x4Add(f32x4Mul(b, c), a)

        override fun vecStore(memAddr: FloatArray, offset: Int, a: FloatArray) {
            v128Store(memAddr, offset, a)
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
            for (i in 0 until arr / 2) {
                x[2 * i] = f32x4Add(x[2 * i], x[2 * i + 1])
            }
            for (i in 0 until arr / 4) {
                x[4 * i] = f32x4Add(x[4 * i], x[4 * i + 2])
            }
            for (i in 0 until arr / 8) {
                x[8 * i] = f32x4Add(x[8 * i], x[8 * i + 4])
            }
            y[yOffset] =
                f32x4ExtractLane(x[0], 0) +
                f32x4ExtractLane(x[0], 1) +
                f32x4ExtractLane(x[0], 2) +
                f32x4ExtractLane(x[0], 3)
        }
    }
}
