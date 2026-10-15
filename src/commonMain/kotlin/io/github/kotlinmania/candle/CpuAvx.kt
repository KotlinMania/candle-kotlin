// port-lint: source cpu/avx.rs
package io.github.kotlinmania.candle

public object CpuAvx {
    public const val step: Int = 32
    public const val epr: Int = 8
    public const val arr: Int = step / epr

    public class CurrentCpu : Cpu<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr

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
            var sum = 0.0f
            for (v in x[0]) sum += v
            y[yOffset] = sum
        }
    }

    public class CurrentCpuF16 : CpuF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr
        override val flushInterval: Int = Int.MAX_VALUE

        override fun zero(): FloatArray = FloatArray(epr)

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

        override fun load(memAddr: ShortArray, offset: Int): FloatArray =
            FloatArray(epr) { Float16Utils.f16BitsToFloat(memAddr[offset + it]) }

        public fun load(memAddr: ShortArray): FloatArray = load(memAddr, 0)

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            FloatArray(epr) { a[it] + b[it] }

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            FloatArray(epr) { b[it] * c[it] + a[it] }

        override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
            for (i in 0 until epr) {
                memAddr[offset + i] = Float16Utils.floatToF16Bits(a[i])
            }
        }

        public fun vecStore(memAddr: ShortArray, a: FloatArray) {
            vecStore(memAddr, 0, a)
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
            var offset = arr shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            var sum = 0.0f
            for (v in x[0]) sum += v
            y[yOffset] = sum
        }
    }

    public class CurrentCpuBF16 : CpuBF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr

        override fun zero(): FloatArray = FloatArray(epr)

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

        override fun load(memAddr: ShortArray, offset: Int): FloatArray =
            FloatArray(epr) { Float16Utils.bf16BitsToFloat(memAddr[offset + it]) }

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            FloatArray(epr) { a[it] + b[it] }

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            FloatArray(epr) { b[it] * c[it] + a[it] }

        override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
            for (i in 0 until epr) {
                memAddr[offset + i] = Float16Utils.floatToBF16Bits(a[i])
            }
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
            var offset = arr shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = vecAdd(x[i], x[offset + i])
            }
            var sum = 0.0f
            for (v in x[0]) sum += v
            y[yOffset] = sum
        }
    }
}
