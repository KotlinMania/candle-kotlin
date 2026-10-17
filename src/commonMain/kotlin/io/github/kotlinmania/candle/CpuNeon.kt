// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

internal expect val vdupqNF32: (Float) -> FloatArray
internal expect val vld1qF32: (FloatArray, Int) -> FloatArray
internal expect val vaddqF32: (FloatArray, FloatArray) -> FloatArray
internal expect val vfmaqF32: (FloatArray, FloatArray, FloatArray) -> FloatArray
internal expect val vst1qF32: (FloatArray, Int, FloatArray) -> Unit
internal expect val vaddvqF32: (FloatArray) -> Float
internal expect val vgetqLaneF32: (FloatArray, Int) -> Float
private val vld1U16: (ShortArray, Int) -> ShortArray = { mem, off -> ShortArray(4) { mem[off + it] } }
private val vst1U16: (ShortArray, Int, ShortArray) -> Unit = { mem, off, bits ->
    for (i in 0 until 4) mem[off + i] = bits[i]
}
private val vld1qU16: (ShortArray, Int) -> ShortArray = { mem, off -> ShortArray(8) { mem[off + it] } }
private val vst1qU16: (ShortArray, Int, ShortArray) -> Unit = { mem, off, a ->
    for (i in 0 until 8) mem[off + i] = a[i]
}
internal expect val fcvtl4s4h: (ShortArray) -> FloatArray
internal expect val fcvtn4h4s: (FloatArray) -> ShortArray
private val vshllNU16: (ShortArray, Int) -> IntArray = { v, shift ->
    IntArray(4) { (v[it].toInt() and 0xFFFF) shl shift }
}
private val vreinterpretqF32U32: (IntArray) -> FloatArray = { s ->
    FloatArray(4) { Float.fromBits(s[it]) }
}
private val vreinterpretqU32F32: (FloatArray) -> IntArray = { a ->
    IntArray(4) { a[it].toRawBits() }
}
private val vshrnNU32: (IntArray, Int) -> ShortArray = { s, shift ->
    ShortArray(4) { (s[it] ushr shift).toShort() }
}
private val vreinterpretqU16F32: (FloatArray) -> ShortArray = { a ->
    ShortArray(8) { i ->
        val bits = a[i / 2].toRawBits()
        if (i % 2 == 0) (bits and 0xFFFF).toShort() else (bits ushr 16).toShort()
    }
}
private val reinterpretArrayF32: (Array<ShortArray>) -> Array<FloatArray> = { x ->
    Array(x.size) { row ->
        FloatArray(4) { lane ->
            val lo = x[row][2 * lane].toInt() and 0xFFFF
            val hi = x[row][2 * lane + 1].toInt() and 0xFFFF
            Float.fromBits(lo or (hi shl 16))
        }
    }
}
internal expect val asmMovi8hZero: () -> ShortArray
internal expect val asmFadd8h: (ShortArray, ShortArray) -> ShortArray
internal expect val asmFmla8h: (ShortArray, ShortArray, ShortArray) -> ShortArray
internal expect val asmDup8hFromF32: (Float) -> ShortArray
internal expect val asmReduceF16x8: (ShortArray) -> Float
internal expect val asmFadd4sOnU16x8: (ShortArray, ShortArray) -> ShortArray
internal expect val asmBfdot4s8h: (ShortArray, ShortArray, ShortArray) -> ShortArray

public object CpuNeon {
    public object AssociatedTypes {
        public interface Unit

        public interface Array
    }

    public class CurrentCpu : Cpu<FloatArray, kotlin.Array<FloatArray>> {
        override val step: Int = 32
        override val epr: Int = 4
        override val arr: Int = step / epr

        public fun reduceOne(x: FloatArray): Float = vaddvqF32(x)

        public fun reduceOne(
            x: FloatArray,
            offset: Int,
        ): Float =
            vgetqLaneF32(x, offset + 0) +
                vgetqLaneF32(x, offset + 1) +
                vgetqLaneF32(x, offset + 2) +
                vgetqLaneF32(x, offset + 3)

        override fun zero(): FloatArray = vdupqNF32(0.0f)

        override fun fromF32(v: Float): FloatArray = vdupqNF32(v)

        override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

        override fun load(
            memAddr: FloatArray,
            offset: Int,
        ): FloatArray = vld1qF32(memAddr, offset)

        override fun vecAdd(
            a: FloatArray,
            b: FloatArray,
        ): FloatArray = vaddqF32(a, b)

        override fun vecFma(
            a: FloatArray,
            b: FloatArray,
            c: FloatArray,
        ): FloatArray = vfmaqF32(a, b, c)

        override fun vecStore(
            memAddr: FloatArray,
            offset: Int,
            a: FloatArray,
        ) {
            vst1qF32(memAddr, offset, a)
        }

        override fun vecReduce(
            x: Array<FloatArray>,
            y: FloatArray,
            yOffset: Int,
        ) {
            for (i in 0 until arr / 2) {
                x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
            }
            for (i in 0 until arr / 4) {
                x[4 * i] = vaddqF32(x[4 * i], x[4 * i + 2])
            }
            for (i in 0 until arr / 8) {
                x[8 * i] = vaddqF32(x[8 * i], x[8 * i + 4])
            }
            y[yOffset] = reduceOne(x[0])
        }
    }

    public object Fp16 {
        // Fallback: widen f16 to f32 on load using FCVTL, accumulate in f32.
        public object Fallback {
            public class CurrentCpuF16 : CpuF16<FloatArray, Array<FloatArray>> {
                override val step: Int = 32
                override val epr: Int = 4
                override val arr: Int = step / epr
                override val flushInterval: Int = Int.MAX_VALUE

                override fun zero(): FloatArray = vdupqNF32(0.0f)

                override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

                override fun load(
                    memAddr: ShortArray,
                    offset: Int,
                ): FloatArray {
                    val bits = vld1U16(memAddr, offset)
                    val result = fcvtl4s4h(bits)
                    return result
                }

                override fun vecAdd(
                    a: FloatArray,
                    b: FloatArray,
                ): FloatArray = vaddqF32(a, b)

                override fun vecFma(
                    a: FloatArray,
                    b: FloatArray,
                    c: FloatArray,
                ): FloatArray = vfmaqF32(a, b, c)

                override fun vecReduce(
                    x: Array<FloatArray>,
                    y: FloatArray,
                    yOffset: Int,
                ) {
                    for (i in 0 until arr / 2) {
                        x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
                    }
                    for (i in 0 until arr / 4) {
                        x[4 * i] = vaddqF32(x[4 * i], x[4 * i + 2])
                    }
                    for (i in 0 until arr / 8) {
                        x[8 * i] = vaddqF32(x[8 * i], x[8 * i + 4])
                    }
                    y[yOffset] = vaddvqF32(x[0])
                }

                override fun fromF32(v: Float): FloatArray = vdupqNF32(v)

                override fun vecStore(
                    memAddr: ShortArray,
                    offset: Int,
                    a: FloatArray,
                ) {
                    val bits = fcvtn4h4s(a)
                    vst1U16(memAddr, offset, bits)
                }
            }
        }

        // Optimized: accumulate in f16 using FMLA.8h, flush to f32 periodically.
        public object Optimized {
            public class CurrentCpuF16 : CpuF16<ShortArray, Array<ShortArray>> {
                override val step: Int = 32
                override val epr: Int = 8
                override val arr: Int = step / epr

                // Flush f16 accumulators to f32 every FLUSH_INTERVAL steps to bound rounding error.
                override val flushInterval: Int = 16

                public fun reduceOne(x: ShortArray): Float {
                    // 1. Widen bottom 4 lanes of f16x8 to a full f32x4 vector
                    // 2. Widen top 4 lanes of f16x8 to a second f32x4 vector
                    // 3. Perform a parallel element-wise vector addition
                    // 4. Pairwise add 4 lanes down to 2 lanes: [A+B, C+D, ?, ?]
                    // 5. Pairwise add 2 lanes down to 1 scalar: [(A+B)+(C+D), ?, ?, ?]
                    // 6. Extract the scalar from the lowest lane of the vector
                    val result = asmReduceF16x8(x)
                    return result
                }

                override fun zero(): ShortArray {
                    val result = asmMovi8hZero()
                    return result
                }

                override fun zeroArray(): Array<ShortArray> = Array(arr) { zero() }

                override fun load(
                    memAddr: ShortArray,
                    offset: Int,
                ): ShortArray = vld1qU16(memAddr, offset)

                override fun vecAdd(
                    a: ShortArray,
                    b: ShortArray,
                ): ShortArray {
                    val result = asmFadd8h(a, b)
                    return result
                }

                override fun vecFma(
                    a: ShortArray,
                    b: ShortArray,
                    c: ShortArray,
                ): ShortArray {
                    val result = asmFmla8h(a, b, c)
                    return result
                }

                override fun vecReduce(
                    x: Array<ShortArray>,
                    y: FloatArray,
                    yOffset: Int,
                ) {
                    var sum = 0.0f
                    for (item in x) {
                        sum += reduceOne(item)
                    }
                    y[yOffset] = sum
                }

                override fun fromF32(v: Float): ShortArray {
                    val result = asmDup8hFromF32(v)
                    return result
                }

                override fun vecStore(
                    memAddr: ShortArray,
                    offset: Int,
                    a: ShortArray,
                ) {
                    vst1qU16(memAddr, offset, a)
                }
            }
        }
    }

    public object Bf16 {
        // Fallback: accumulate in f32 using SHLL to widen bf16 to f32 on load.
        public object Fallback {
            public class CurrentCpuBF16 : CpuBF16<FloatArray, Array<FloatArray>> {
                override val step: Int = 32
                override val epr: Int = 4
                override val arr: Int = step / epr

                override fun zero(): FloatArray = vdupqNF32(0.0f)

                override fun zeroArray(): Array<FloatArray> = Array(arr) { zero() }

                override fun load(
                    memAddr: ShortArray,
                    offset: Int,
                ): FloatArray {
                    // bf16 is the top 16 bits of f32; shift left by 16 to reconstruct f32 bits.
                    val i = vld1U16(memAddr, offset)
                    val s = vshllNU16(i, 16)
                    return vreinterpretqF32U32(s)
                }

                override fun vecAdd(
                    a: FloatArray,
                    b: FloatArray,
                ): FloatArray = vaddqF32(a, b)

                override fun vecFma(
                    a: FloatArray,
                    b: FloatArray,
                    c: FloatArray,
                ): FloatArray = vfmaqF32(a, b, c)

                override fun vecReduce(
                    x: Array<FloatArray>,
                    y: FloatArray,
                    yOffset: Int,
                ) {
                    for (i in 0 until arr / 2) {
                        x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
                    }
                    for (i in 0 until arr / 4) {
                        x[4 * i] = vaddqF32(x[4 * i], x[4 * i + 2])
                    }
                    for (i in 0 until arr / 8) {
                        x[8 * i] = vaddqF32(x[8 * i], x[8 * i + 4])
                    }
                    y[yOffset] = vaddvqF32(x[0])
                }

                override fun fromF32(v: Float): FloatArray = vdupqNF32(v)

                override fun vecStore(
                    memAddr: ShortArray,
                    offset: Int,
                    a: FloatArray,
                ) {
                    // Extract top 16 bits of each f32 lane, which is the bf16 representation.
                    val s = vreinterpretqU32F32(a)
                    val h = vshrnNU32(s, 16)
                    vst1U16(memAddr, offset, h)
                }
            }
        }

        // Optimized: BFDOT accumulates bf16 pairs directly into f32 lanes.
        public object Optimized {
            public class CurrentCpuBF16 : CpuBF16<ShortArray, Array<ShortArray>> {
                override val step: Int = 32
                override val epr: Int = 8
                override val arr: Int = step / epr

                override fun zero(): ShortArray = vreinterpretqU16F32(vdupqNF32(0.0f))

                override fun zeroArray(): Array<ShortArray> = Array(arr) { zero() }

                override fun load(
                    memAddr: ShortArray,
                    offset: Int,
                ): ShortArray = vld1qU16(memAddr, offset)

                override fun vecAdd(
                    a: ShortArray,
                    b: ShortArray,
                ): ShortArray {
                    val result = asmFadd4sOnU16x8(a, b)
                    return result
                }

                override fun vecFma(
                    a: ShortArray,
                    b: ShortArray,
                    c: ShortArray,
                ): ShortArray {
                    // IMPORTANT: a is not bf16x8 here. It is 128 bits used as f32x4 (.4s) to accumulate
                    // results from bfdot. b and c are bf16x8 inputs (.8h).
                    // bfdot: acc[i] += (f32)b[2i]*c[2i] + (f32)b[2i+1]*c[2i+1]
                    val result = asmBfdot4s8h(a, b, c)
                    return result
                }

                override fun vecReduce(
                    x: Array<ShortArray>,
                    y: FloatArray,
                    yOffset: Int,
                ) {
                    // IMPORTANT: x is not bf16x8 here. It is 128 bits used as f32x4.
                    // This means that if you use vecReduce where x is actually bf16x8 you will get
                    // entirely wrong results.
                    val xf = reinterpretArrayF32(x)
                    for (i in 0 until arr / 2) {
                        xf[2 * i] = vaddqF32(xf[2 * i], xf[2 * i + 1])
                    }
                    for (i in 0 until arr / 4) {
                        xf[4 * i] = vaddqF32(xf[4 * i], xf[4 * i + 2])
                    }
                    y[yOffset] = vaddvqF32(xf[0])
                }

                override fun fromF32(v: Float): ShortArray = vreinterpretqU16F32(vdupqNF32(v))

                override fun vecStore(
                    memAddr: ShortArray,
                    offset: Int,
                    a: ShortArray,
                ) {
                    vst1qU16(memAddr, offset, a)
                }
            }
        }
    }
}
