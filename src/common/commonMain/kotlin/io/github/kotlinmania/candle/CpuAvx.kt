// port-lint: source cpu/avx.rs
package io.github.kotlinmania.candle

internal expect val mm256SetzeroPs: () -> FloatArray
internal expect val mm256Set1Ps: (Float) -> FloatArray
internal expect val mm256LoaduPs: (FloatArray, Int) -> FloatArray
internal expect val mm256AddPs: (FloatArray, FloatArray) -> FloatArray
internal expect val mm256FmaPs: (FloatArray, FloatArray, FloatArray) -> FloatArray
internal expect val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit
internal expect val mm256ReduceOnePs: (FloatArray) -> Float
internal expect val mm256CvtphPs: (ShortArray, Int) -> FloatArray
internal expect val mm256CvtpsPh: (FloatArray, ShortArray, Int) -> Unit
internal expect val mm256CvtpbhPs: (ShortArray, Int) -> FloatArray
internal expect val mm256CvtnepsPbh: (FloatArray, ShortArray, Int) -> Unit
private val isX86FeatureDetected: (String) -> Boolean = { _ -> true }

public object CpuAvx {
    public object AssociatedTypes {
        public interface Unit

        public interface Array
    }

    public const val step: Int = 32
    public const val epr: Int = 8
    public const val arr: Int = step / epr

    public class CurrentCpu : Cpu<FloatArray, kotlin.Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr

        override fun zero(): FloatArray = mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = mm256Set1Ps(v)

        override fun load(
            memAddr: FloatArray,
            offset: Int,
        ): FloatArray = mm256LoaduPs(memAddr, offset)

        override fun vecAdd(
            a: FloatArray,
            b: FloatArray,
        ): FloatArray = mm256AddPs(a, b)

        override fun vecFma(
            a: FloatArray,
            b: FloatArray,
            c: FloatArray,
        ): FloatArray = mm256FmaPs(a, b, c)

        override fun vecStore(
            memAddr: FloatArray,
            offset: Int,
            a: FloatArray,
        ) {
            mm256StoreuPs(memAddr, offset, a)
        }

        override fun vecReduce(
            x: Array<FloatArray>,
            y: FloatArray,
            yOffset: Int,
        ) {
            for (i in 0 until arr / 2) {
                x[2 * i] = mm256AddPs(x[2 * i], x[2 * i + 1])
            }
            for (i in 0 until arr / 4) {
                x[4 * i] = mm256AddPs(x[4 * i], x[4 * i + 2])
            }
            for (i in 0 until arr / 8) {
                x[8 * i] = mm256AddPs(x[8 * i], x[8 * i + 4])
            }
            y[yOffset] = mm256ReduceOnePs(x[0])
        }
    }

    public class CurrentCpuF16 : CpuF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr
        override val flushInterval: Int = Int.MAX_VALUE

        override fun zero(): FloatArray = mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = mm256Set1Ps(v)

        override fun load(
            memAddr: ShortArray,
            offset: Int,
        ): FloatArray = mm256CvtphPs(memAddr, offset)

        public fun load(memAddr: ShortArray): FloatArray = mm256CvtphPs(memAddr, 0)

        override fun vecAdd(
            a: FloatArray,
            b: FloatArray,
        ): FloatArray = mm256AddPs(a, b)

        override fun vecFma(
            a: FloatArray,
            b: FloatArray,
            c: FloatArray,
        ): FloatArray = mm256FmaPs(a, b, c)

        override fun vecStore(
            memAddr: ShortArray,
            offset: Int,
            a: FloatArray,
        ) {
            mm256CvtpsPh(a, memAddr, offset)
        }

        public fun vecStore(
            memAddr: ShortArray,
            a: FloatArray,
        ) {
            mm256CvtpsPh(a, memAddr, 0)
        }

        override fun vecReduce(
            x: Array<FloatArray>,
            y: FloatArray,
            yOffset: Int,
        ) {
            var offset = arr shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            y[yOffset] = mm256ReduceOnePs(x[0])
        }
    }

    public class CurrentCpuBF16 : CpuBF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr

        override fun zero(): FloatArray = mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray = mm256Set1Ps(v)

        override fun load(
            memAddr: ShortArray,
            offset: Int,
        ): FloatArray =
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mm256CvtpbhPs(memAddr, offset)
            } else {
                mm256CvtpbhPs(memAddr, offset)
            }

        override fun vecAdd(
            a: FloatArray,
            b: FloatArray,
        ): FloatArray = mm256AddPs(a, b)

        override fun vecFma(
            a: FloatArray,
            b: FloatArray,
            c: FloatArray,
        ): FloatArray = mm256FmaPs(a, b, c)

        override fun vecStore(
            memAddr: ShortArray,
            offset: Int,
            a: FloatArray,
        ) {
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mm256CvtnepsPbh(a, memAddr, offset)
            } else {
                mm256CvtnepsPbh(a, memAddr, offset)
            }
        }

        override fun vecReduce(
            x: Array<FloatArray>,
            y: FloatArray,
            yOffset: Int,
        ) {
            var offset = arr shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            y[yOffset] = mm256ReduceOnePs(x[0])
        }
    }
}
