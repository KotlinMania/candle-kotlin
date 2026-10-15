// port-lint: source cpu/avx.rs
package io.github.kotlinmania.candle

private val mm256SetzeroPs: () -> FloatArray = { FloatArray(8) }
private val mm256Set1Ps: (Float) -> FloatArray = { v -> FloatArray(8) { v } }
private val mm256LoaduPs: (FloatArray, Int) -> FloatArray = { mem, off -> FloatArray(8) { mem[off + it] } }
internal expect val mm256AddPs: (FloatArray, FloatArray) -> FloatArray
private val mm256MulPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(8) { a[it] * b[it] } }
private val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    for (i in 0 until 8) mem[off + i] = a[i]
}
private val mm256Castps256Ps128: (FloatArray) -> FloatArray = { a -> FloatArray(4) { a[it] } }
private val mm256Extractf128Ps: (FloatArray, Int) -> FloatArray = { a, imm ->
    FloatArray(4) { a[imm * 4 + it] }
}
private val mmAddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] + b[it] } }
private val mmHaddPs: (FloatArray, FloatArray) -> FloatArray = { a, b ->
    floatArrayOf(a[0] + a[1], a[2] + a[3], b[0] + b[1], b[2] + b[3])
}
private val mmCvtssF32: (FloatArray) -> Float = { a -> a[0] }
private val mmLoaduSi128: (ShortArray, Int) -> ShortArray = { mem, off -> ShortArray(8) { mem[off + it] } }
private val mmStoreuSi128: (ShortArray, Int, ShortArray) -> Unit = { mem, off, a ->
    for (i in 0 until 8) mem[off + i] = a[i]
}
internal expect val mm256CvtphPs: (ShortArray) -> FloatArray
private val mm256CvtpsPh: (FloatArray, Int) -> ShortArray = { a, _ ->
    ShortArray(8) { Float16Utils.floatToF16Bits(a[it]) }
}
private val mm256CvtpbhPs: (ShortArray) -> FloatArray = { a ->
    FloatArray(8) { Float16Utils.bf16BitsToFloat(a[it]) }
}
private val mm256CvtnepsPbh: (FloatArray) -> ShortArray = { a ->
    ShortArray(8) { Float16Utils.floatToBF16Bits(a[it]) }
}
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

        override fun zero(): FloatArray =
            mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> =
            Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray =
            mm256Set1Ps(v)

        override fun load(memAddr: FloatArray, offset: Int): FloatArray =
            mm256LoaduPs(memAddr, offset)

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            mm256AddPs(a, b)

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            mm256AddPs(mm256MulPs(b, c), a)

        override fun vecStore(memAddr: FloatArray, offset: Int, a: FloatArray) {
            mm256StoreuPs(memAddr, offset, a)
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
            for (i in 0 until arr / 2) {
                x[2 * i] = mm256AddPs(x[2 * i], x[2 * i + 1])
            }
            for (i in 0 until arr / 4) {
                x[4 * i] = mm256AddPs(x[4 * i], x[4 * i + 2])
            }
            for (i in 0 until arr / 8) {
                x[8 * i] = mm256AddPs(x[8 * i], x[8 * i + 4])
            }
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            y[yOffset] = mmCvtssF32(mmHaddPs(t1, t1))
        }
    }

    public class CurrentCpuF16 : CpuF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr
        override val flushInterval: Int = Int.MAX_VALUE

        override fun zero(): FloatArray =
            mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> =
            Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray =
            mm256Set1Ps(v)

        override fun load(memAddr: ShortArray, offset: Int): FloatArray =
            mm256CvtphPs(mmLoaduSi128(memAddr, offset))

        public fun load(memAddr: ShortArray): FloatArray {
            val tmp = FloatArray(8)
            for (i in 0 until 8) {
                tmp[i] = Float16Utils.f16BitsToFloat(memAddr[i])
            }
            return mm256LoaduPs(tmp, 0)
        }

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            mm256AddPs(a, b)

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            mm256AddPs(mm256MulPs(b, c), a)

        override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
            mmStoreuSi128(memAddr, offset, mm256CvtpsPh(a, 0))
        }

        public fun vecStore(memAddr: ShortArray, a: FloatArray) {
            val tmp = FloatArray(8)
            mm256StoreuPs(tmp, 0, a)
            for (i in 0 until 8) {
                memAddr[i] = Float16Utils.floatToF16Bits(tmp[i])
            }
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
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
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            y[yOffset] = mmCvtssF32(mmHaddPs(t1, t1))
        }
    }

    public class CurrentCpuBF16 : CpuBF16<FloatArray, Array<FloatArray>> {
        override val step: Int = CpuAvx.step
        override val epr: Int = CpuAvx.epr
        override val arr: Int = CpuAvx.arr

        override fun zero(): FloatArray =
            mm256SetzeroPs()

        override fun zeroArray(): Array<FloatArray> =
            Array(arr) { zero() }

        override fun fromF32(v: Float): FloatArray =
            mm256Set1Ps(v)

        override fun load(memAddr: ShortArray, offset: Int): FloatArray =
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mm256CvtpbhPs(mmLoaduSi128(memAddr, offset))
            } else {
                val tmp = FloatArray(8)
                for (i in 0 until 8) {
                    tmp[i] = Float16Utils.bf16BitsToFloat(memAddr[offset + i])
                }
                mm256LoaduPs(tmp, 0)
            }

        override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
            mm256AddPs(a, b)

        override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
            mm256AddPs(mm256MulPs(b, c), a)

        override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mmStoreuSi128(memAddr, offset, mm256CvtnepsPbh(a))
            } else {
                val tmp = FloatArray(8)
                mm256StoreuPs(tmp, 0, a)
                for (i in 0 until 8) {
                    memAddr[offset + i] = Float16Utils.floatToBF16Bits(tmp[i])
                }
            }
        }

        override fun vecReduce(x: Array<FloatArray>, y: FloatArray, yOffset: Int) {
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
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            y[yOffset] = mmCvtssF32(mmHaddPs(t1, t1))
        }
    }
}
