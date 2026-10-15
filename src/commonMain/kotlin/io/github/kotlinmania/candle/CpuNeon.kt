// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

public object CpuNeon {
    public object AssociatedTypes {
        public interface Unit
        public interface Array
    }

    public class CurrentCpu : Cpu<FloatArray, kotlin.Array<FloatArray>> {
        override val step: Int = 32
        override val epr: Int = 4
        override val arr: Int = step / epr

        public fun reduceOne(x: FloatArray): Float =
            x[0] + x[1] + x[2] + x[3]

        public fun reduceOne(x: FloatArray, offset: Int): Float =
            x[offset] + x[offset + 1] + x[offset + 2] + x[offset + 3]

        override fun zero(): FloatArray = FloatArray(epr)

        override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

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
            y[yOffset] = reduceOne(x[0])
        }
    }

    public object Fp16 {
        public object Fallback {
            public class CurrentCpuF16 : CpuF16<FloatArray, Array<FloatArray>> {
                override val step: Int = 32
                override val epr: Int = 4
                override val arr: Int = step / epr
                override val flushInterval: Int = Int.MAX_VALUE

                override fun zero(): FloatArray = FloatArray(epr)

                override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

                override fun load(memAddr: ShortArray, offset: Int): FloatArray =
                    FloatArray(epr) { Float16Utils.f16BitsToFloat(memAddr[offset + it]) }

                override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
                    FloatArray(epr) { a[it] + b[it] }

                override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
                    FloatArray(epr) { b[it] * c[it] + a[it] }

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

                override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

                override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
                    for (i in 0 until epr) {
                        memAddr[offset + i] = Float16Utils.floatToF16Bits(a[i])
                    }
                }
            }
        }

        public object Optimized {
            public class CurrentCpuF16 : CpuF16<ShortArray, Array<ShortArray>> {
                override val step: Int = 32
                override val epr: Int = 8
                override val arr: Int = step / epr
                override val flushInterval: Int = 16

                public fun reduceOne(x: ShortArray): Float {
                    var sum = 0.0f
                    for (v in x) sum += Float16Utils.f16BitsToFloat(v)
                    return sum
                }

                override fun zero(): ShortArray = ShortArray(epr)

                override fun zeroArray(): Array<ShortArray> = Array(arr) { zero() }

                override fun load(memAddr: ShortArray, offset: Int): ShortArray =
                    ShortArray(epr) { memAddr[offset + it] }

                override fun vecAdd(a: ShortArray, b: ShortArray): ShortArray =
                    ShortArray(epr) {
                        Float16Utils.floatToF16Bits(
                            Float16Utils.f16BitsToFloat(a[it]) + Float16Utils.f16BitsToFloat(b[it]),
                        )
                    }

                override fun vecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray =
                    ShortArray(epr) {
                        Float16Utils.floatToF16Bits(
                            Float16Utils.f16BitsToFloat(a[it]) +
                                Float16Utils.f16BitsToFloat(b[it]) * Float16Utils.f16BitsToFloat(c[it]),
                        )
                    }

                override fun vecReduce(x: Array<ShortArray>, y: FloatArray, yOffset: Int) {
                    var sum = 0.0f
                    for (item in x) {
                        sum += reduceOne(item)
                    }
                    y[yOffset] = sum
                }

                override fun fromF32(v: Float): ShortArray {
                    val bits = Float16Utils.floatToF16Bits(v)
                    return ShortArray(epr) { bits }
                }

                override fun vecStore(memAddr: ShortArray, offset: Int, a: ShortArray) {
                    for (i in 0 until epr) {
                        memAddr[offset + i] = a[i]
                    }
                }
            }
        }
    }

    public object Bf16 {
        public object Fallback {
            public class CurrentCpuBF16 : CpuBF16<FloatArray, Array<FloatArray>> {
                override val step: Int = 32
                override val epr: Int = 4
                override val arr: Int = step / epr

                override fun zero(): FloatArray = FloatArray(epr)

                override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

                override fun load(memAddr: ShortArray, offset: Int): FloatArray =
                    FloatArray(epr) { Float16Utils.bf16BitsToFloat(memAddr[offset + it]) }

                override fun vecAdd(a: FloatArray, b: FloatArray): FloatArray =
                    FloatArray(epr) { a[it] + b[it] }

                override fun vecFma(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray =
                    FloatArray(epr) { b[it] * c[it] + a[it] }

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

                override fun fromF32(v: Float): FloatArray = FloatArray(epr) { v }

                override fun vecStore(memAddr: ShortArray, offset: Int, a: FloatArray) {
                    for (i in 0 until epr) {
                        memAddr[offset + i] = Float16Utils.floatToBF16Bits(a[i])
                    }
                }
            }
        }

        public object Optimized {
            public class CurrentCpuBF16 : CpuBF16<ShortArray, Array<ShortArray>> {
                override val step: Int = 32
                override val epr: Int = 8
                override val arr: Int = step / epr

                override fun zero(): ShortArray = ShortArray(epr)

                override fun zeroArray(): Array<ShortArray> = Array(arr) { zero() }

                override fun load(memAddr: ShortArray, offset: Int): ShortArray =
                    ShortArray(epr) { memAddr[offset + it] }

                override fun vecAdd(a: ShortArray, b: ShortArray): ShortArray =
                    ShortArray(epr) {
                        Float16Utils.floatToBF16Bits(
                            Float16Utils.bf16BitsToFloat(a[it]) + Float16Utils.bf16BitsToFloat(b[it]),
                        )
                    }

                override fun vecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray =
                    ShortArray(epr) {
                        Float16Utils.floatToBF16Bits(
                            Float16Utils.bf16BitsToFloat(a[it]) +
                                Float16Utils.bf16BitsToFloat(b[it]) * Float16Utils.bf16BitsToFloat(c[it]),
                        )
                    }

                override fun vecReduce(x: Array<ShortArray>, y: FloatArray, yOffset: Int) {
                    var sum = 0.0f
                    for (item in x) {
                        for (v in item) {
                            sum += Float16Utils.bf16BitsToFloat(v)
                        }
                    }
                    y[yOffset] = sum
                }

                override fun fromF32(v: Float): ShortArray {
                    val bits = Float16Utils.floatToBF16Bits(v)
                    return ShortArray(epr) { bits }
                }

                override fun vecStore(memAddr: ShortArray, offset: Int, a: ShortArray) {
                    for (i in 0 until epr) {
                        memAddr[offset + i] = a[i]
                    }
                }
            }
        }
    }
}
