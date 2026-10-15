// port-lint: source quantized/neon.rs
package io.github.kotlinmania.candle.quantized.neon

import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ8K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.QK80
import io.github.kotlinmania.candle.QKK

private val vdupqNF32: (Float) -> FloatArray = { v -> FloatArray(4) { v } }
private val vdupqNS32: (Int) -> IntArray = { v -> IntArray(4) { v } }
private val vdupqNU8: (Int) -> ByteArray = { v -> ByteArray(16) { v.toByte() } }
private val vdupqNS8: (Int) -> ByteArray = { v -> ByteArray(16) { v.toByte() } }
private fun vld1qU8(src: UByteArray, off: Int): ByteArray = ByteArray(16) { i -> src[off + i].toByte() }
private fun vld1qU8(src: ByteArray, off: Int): ByteArray = ByteArray(16) { i -> src[off + i] }
private val vld1qS8: (ByteArray, Int) -> ByteArray = { src, off -> ByteArray(16) { i -> src[off + i] } }
private val vld1qS16: (ShortArray, Int) -> ShortArray = { src, off -> ShortArray(8) { i -> src[off + i] } }
private val vld1qS16X2: (ShortArray, Int) -> Array<ShortArray> = { src, off ->
    arrayOf(
        ShortArray(8) { i -> src[off + i] },
        ShortArray(8) { i -> src[off + 8 + i] },
    )
}
private val vld1qU8X2: (ByteArray, Int) -> Array<ByteArray> = { src, off ->
    arrayOf(
        ByteArray(16) { i -> src[off + i] },
        ByteArray(16) { i -> src[off + 16 + i] },
    )
}
private val vld1qS8X2: (ByteArray, Int) -> Array<ByteArray> = { src, off ->
    arrayOf(
        ByteArray(16) { i -> src[off + i] },
        ByteArray(16) { i -> src[off + 16 + i] },
    )
}
private val vld1qU8X4: (ByteArray, Int) -> Array<ByteArray> = { src, off ->
    arrayOf(
        ByteArray(16) { i -> src[off + i] },
        ByteArray(16) { i -> src[off + 16 + i] },
        ByteArray(16) { i -> src[off + 32 + i] },
        ByteArray(16) { i -> src[off + 48 + i] },
    )
}
private val vld1qS8X4: (ByteArray, Int) -> Array<ByteArray> = { src, off ->
    arrayOf(
        ByteArray(16) { i -> src[off + i] },
        ByteArray(16) { i -> src[off + 16 + i] },
        ByteArray(16) { i -> src[off + 32 + i] },
        ByteArray(16) { i -> src[off + 48 + i] },
    )
}
private val vst1qU8: (UByteArray, Int, ByteArray) -> Unit = { dst, off, v ->
    for (i in 0 until 16) dst[off + i] = v[i].toUByte()
}
private val vgetLowS8: (ByteArray) -> ByteArray = { a -> ByteArray(8) { i -> a[i] } }
private val vgetHighS8: (ByteArray) -> ByteArray = { a -> ByteArray(8) { i -> a[i + 8] } }
private val vgetLowU8: (ByteArray) -> ByteArray = { a -> ByteArray(8) { i -> a[i] } }
private val vgetHighU8: (ByteArray) -> ByteArray = { a -> ByteArray(8) { i -> a[i + 8] } }
private val vgetLowS16: (ShortArray) -> ShortArray = { a -> ShortArray(4) { i -> a[i] } }
private val vgetHighS16: (ShortArray) -> ShortArray = { a -> ShortArray(4) { i -> a[i + 4] } }
internal expect val vmullS8: (ByteArray, ByteArray) -> ShortArray
internal expect val vmullS16: (ShortArray, ShortArray) -> IntArray
internal expect val vpaddlqS16: (ShortArray) -> IntArray
internal expect val vpaddqS16: (ShortArray, ShortArray) -> ShortArray
private val vmovlS8: (ByteArray) -> ShortArray = { a -> ShortArray(8) { i -> a[i].toShort() } }
private val vmovlU8: (ByteArray) -> ShortArray = { a -> ShortArray(8) { i -> (a[i].toInt() and 0xFF).toShort() } }
private val vandqU8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() and b[i].toInt()).toByte() }
}
private val vorrqU8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() or b[i].toInt()).toByte() }
}
private val vbicqU8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() and b[i].toInt().inv()).toByte() }
}
private val vshrqNU8: (ByteArray, Int) -> ByteArray = { a, shift ->
    ByteArray(16) { i -> ((a[i].toInt() and 0xFF) ushr shift).toByte() }
}
private val vshlqNU8: (ByteArray, Int) -> ByteArray = { a, shift ->
    ByteArray(16) { i -> ((a[i].toInt() and 0xFF) shl shift).toByte() }
}
private val vsubqS8: (ByteArray, ByteArray) -> ByteArray = { a, b ->
    ByteArray(16) { i -> (a[i].toInt() - b[i].toInt()).toByte() }
}
private val vreinterpretqS8U8: (ByteArray) -> ByteArray = { a -> a }
private val vreinterpretqS16U16: (ShortArray) -> ShortArray = { a -> a }
private val vaddqS32: (IntArray, IntArray) -> IntArray = { a, b ->
    IntArray(4) { i -> a[i] + b[i] }
}
private val vcvtqF32S32: (IntArray) -> FloatArray = { a ->
    FloatArray(4) { i -> a[i].toFloat() }
}
internal expect val vmlaqNF32: (FloatArray, FloatArray, Float) -> FloatArray
private val vaddvqF32: (FloatArray) -> Float = { a -> a[0] + a[1] + a[2] + a[3] }
internal expect val vaddvqS32: (IntArray) -> Int
private val readU32Into: (ByteArray, IntArray, Int) -> Unit = { src, dst, count ->
    for (w in 0 until count) {
        val b0 = src[w * 4].toInt() and 0xFF
        val b1 = src[w * 4 + 1].toInt() and 0xFF
        val b2 = src[w * 4 + 2].toInt() and 0xFF
        val b3 = src[w * 4 + 3].toInt() and 0xFF
        dst[w] = b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
}
private val writeU32Into: (IntArray, UByteArray) -> Unit = { src, dst ->
    for (w in src.indices) {
        val word = src[w]
        dst[w * 4] = (word and 0xFF).toUByte()
        dst[w * 4 + 1] = ((word ushr 8) and 0xFF).toUByte()
        dst[w * 4 + 2] = ((word ushr 16) and 0xFF).toUByte()
        dst[w * 4 + 3] = ((word ushr 24) and 0xFF).toUByte()
    }
}
private val vld1U8Words: (Int, Int) -> ByteArray = { w0, w1 ->
    val words = intArrayOf(w0, w1)
    val out = ByteArray(8)
    for (w in 0 until 2) {
        out[w * 4] = (words[w] and 0xFF).toByte()
        out[w * 4 + 1] = ((words[w] ushr 8) and 0xFF).toByte()
        out[w * 4 + 2] = ((words[w] ushr 16) and 0xFF).toByte()
        out[w * 4 + 3] = ((words[w] ushr 24) and 0xFF).toByte()
    }
    out
}
private val utmpToI8: (IntArray) -> ByteArray = { src ->
    val out = ByteArray(16)
    for (w in 0 until 4) {
        val word = src[w]
        out[w * 4] = (word and 0xFF).toByte()
        out[w * 4 + 1] = ((word ushr 8) and 0xFF).toByte()
        out[w * 4 + 2] = ((word ushr 16) and 0xFF).toByte()
        out[w * 4 + 3] = ((word ushr 24) and 0xFF).toByte()
    }
    out
}

private fun vdotqS32(a: ByteArray, b: ByteArray): IntArray {
    val p0 = vmullS8(vgetLowS8(a), vgetLowS8(b))
    val p1 = vmullS8(vgetHighS8(a), vgetHighS8(b))
    return vaddqS32(vpaddlqS16(p0), vpaddlqS16(p1))
}

internal fun vecDotQ40Q80(n: Int, xs: List<BlockQ40>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ40Q80: $n is not divisible by $QK80"
    }
    val nb = n / QK80
    var sumv0 = vdupqNF32(0.0f)
    for (i in 0 until nb) {
        val x0 = xs[i]
        val y0 = ys[i]

        val m4b = vdupqNU8(0x0F)
        val s8b = vdupqNS8(0x8)

        val v00 = vld1qU8(x0.qs, 0)

        // 4-bit -> 8-bit
        val v00l = vreinterpretqS8U8(vandqU8(v00, m4b))
        val v00h = vreinterpretqS8U8(vshrqNU8(v00, 4))

        // sub 8
        val v00ls = vsubqS8(v00l, s8b)
        val v00hs = vsubqS8(v00h, s8b)

        // load y
        val v10l = vld1qS8(y0.qs, 0)
        val v10h = vld1qS8(y0.qs, 16)

        val pl0 = vdotqS32(v00ls, v10l)
        val ph0 = vdotqS32(v00hs, v10h)
        sumv0 = vmlaqNF32(
            sumv0,
            vcvtqF32S32(vaddqS32(pl0, ph0)),
            x0.d.toFloat() * y0.d.toFloat(),
        )
    }
    return vaddvqF32(sumv0)
}

internal fun vecDotQ80Q80(n: Int, xs: List<BlockQ80>, ys: List<BlockQ80>): Float {
    require(n % QK80 == 0) {
        "vecDotQ80Q80: $n is not divisible by $QK80"
    }
    val nb = n / QK80
    var sumv0 = vdupqNF32(0.0f)
    for (i in 0 until nb) {
        val x0 = xs[i]
        val y0 = ys[i]

        val x00 = vld1qS8(x0.qs, 0)
        val x01 = vld1qS8(x0.qs, 16)

        // load y
        val y00 = vld1qS8(y0.qs, 0)
        val y01 = vld1qS8(y0.qs, 16)

        val p0 = vdotqS32(x00, y00)
        val p1 = vdotqS32(x01, y01)

        sumv0 = vmlaqNF32(
            sumv0,
            vcvtqF32S32(vaddqS32(p0, p1)),
            x0.d.toFloat() * y0.d.toFloat(),
        )
    }
    return vaddvqF32(sumv0)
}

internal fun vecDotQ8kQ8k(n: Int, xs: List<BlockQ8K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ8kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    for ((x, y) in xs.zip(ys)) {
        var sumI = vdupqNS32(0)
        val scale = x.d * y.d
        val xQs = x.qs
        val yQs = y.qs
        for (i in 0 until QKK step 16) {
            val xv = vld1qS8(xQs, i)
            val yv = vld1qS8(yQs, i)
            val xy = vdotqS32(xv, yv)
            sumI = vaddqS32(sumI, xy)
        }
        sumf += vaddvqS32(sumI).toFloat() * scale
    }
    return sumf
}

internal fun vecDotQ6kQ8k(n: Int, xs: List<BlockQ6K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ6kQ8k: $n is not divisible by $QKK"
    }
    var sum = 0.0f
    val m4b = vdupqNU8(0xF)
    val mone = vdupqNU8(3)

    for ((x, y) in xs.zip(ys)) {
        val dAll = x.d.toFloat()

        var q6 = 0
        var qh = 0
        var q8 = 0
        var scale = 0

        val q8sums = vld1qS16X2(y.bsums, 0)
        val scales = vld1qS8(x.scales, 0)
        val q6scales = arrayOf(
            vmovlS8(vgetLowS8(scales)),
            vmovlS8(vgetHighS8(scales)),
        )

        val prod = vaddqS32(
            vaddqS32(
                vmullS16(vgetLowS16(q8sums[0]), vgetLowS16(q6scales[0])),
                vmullS16(vgetHighS16(q8sums[0]), vgetHighS16(q6scales[0])),
            ),
            vaddqS32(
                vmullS16(vgetLowS16(q8sums[1]), vgetLowS16(q6scales[1])),
                vmullS16(vgetHighS16(q8sums[1]), vgetHighS16(q6scales[1])),
            ),
        )
        val isumMins = vaddvqS32(prod)

        var isum = 0

        for (j in 0 until QKK / 128) {
            val qhbits = vld1qU8X2(x.qh, qh)
            qh += 32
            val q6bits = vld1qU8X4(x.ql, q6)
            q6 += 64
            var q8bytes = vld1qS8X4(y.qs, q8)
            q8 += 64

            var q6h0 = vshlqNU8(vandqU8(mone, qhbits[0]), 4)
            var q6h1 = vshlqNU8(vandqU8(mone, qhbits[1]), 4)
            var shifted = vshrqNU8(qhbits[0], 2)
            var q6h2 = vshlqNU8(vandqU8(mone, shifted), 4)
            shifted = vshrqNU8(qhbits[1], 2)
            var q6h3 = vshlqNU8(vandqU8(mone, shifted), 4)

            var q6bytes0 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits[0], m4b), q6h0))
            var q6bytes1 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits[1], m4b), q6h1))
            var q6bytes2 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits[2], m4b), q6h2))
            var q6bytes3 = vreinterpretqS8U8(vorrqU8(vandqU8(q6bits[3], m4b), q6h3))

            var p0 = vdotqS32(q6bytes0, q8bytes[0])
            var p1 = vdotqS32(q6bytes1, q8bytes[1])
            var scale0 = x.scales[scale].toInt()
            var scale1 = x.scales[scale + 1].toInt()
            isum += vaddvqS32(p0) * scale0 + vaddvqS32(p1) * scale1
            scale += 2

            var p2 = vdotqS32(q6bytes2, q8bytes[2])
            var p3 = vdotqS32(q6bytes3, q8bytes[3])
            scale0 = x.scales[scale].toInt()
            scale1 = x.scales[scale + 1].toInt()
            isum += vaddvqS32(p2) * scale0 + vaddvqS32(p3) * scale1
            scale += 2

            q8bytes = vld1qS8X4(y.qs, q8)
            q8 += 64

            shifted = vshrqNU8(qhbits[0], 4)
            q6h0 = vshlqNU8(vandqU8(mone, shifted), 4)
            shifted = vshrqNU8(qhbits[1], 4)
            q6h1 = vshlqNU8(vandqU8(mone, shifted), 4)
            shifted = vshrqNU8(qhbits[0], 6)
            q6h2 = vshlqNU8(vandqU8(mone, shifted), 4)
            shifted = vshrqNU8(qhbits[1], 6)
            q6h3 = vshlqNU8(vandqU8(mone, shifted), 4)

            q6bytes0 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits[0], 4), q6h0))
            q6bytes1 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits[1], 4), q6h1))
            q6bytes2 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits[2], 4), q6h2))
            q6bytes3 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q6bits[3], 4), q6h3))

            p0 = vdotqS32(q6bytes0, q8bytes[0])
            p1 = vdotqS32(q6bytes1, q8bytes[1])
            scale0 = x.scales[scale].toInt()
            scale1 = x.scales[scale + 1].toInt()
            isum += vaddvqS32(p0) * scale0 + vaddvqS32(p1) * scale1
            scale += 2

            p2 = vdotqS32(q6bytes2, q8bytes[2])
            p3 = vdotqS32(q6bytes3, q8bytes[3])
            scale0 = x.scales[scale].toInt()
            scale1 = x.scales[scale + 1].toInt()
            isum += vaddvqS32(p2) * scale0 + vaddvqS32(p3) * scale1
            scale += 2
        }
        sum += dAll * y.d * (isum - 32 * isumMins).toFloat()
    }
    return sum
}

internal fun vecDotQ5kQ8k(n: Int, xs: List<BlockQ5K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ5kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val m4b = vdupqNU8(0xF)
    val mone = vdupqNU8(1)
    val mtwo = vdupqNU8(2)

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

        val q8sums = vpaddqS16(
            vld1qS16(y.bsums, 0),
            vld1qS16(y.bsums, 8),
        )

        readU32Into(x.scales, utmp, 3)

        utmp[3] = ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4)
        val uaux = utmp[1] and kmask1
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[2] = uaux
        utmp[0] = utmp[0] and kmask1

        val mins8 = vld1U8Words(utmp[2], utmp[3])
        val mins = vreinterpretqS16U16(vmovlU8(mins8))
        val prod = vaddqS32(
            vmullS16(vgetLowS16(q8sums), vgetLowS16(mins)),
            vmullS16(vgetHighS16(q8sums), vgetHighS16(mins)),
        )
        val sumiMins = vaddvqS32(prod)

        val scales = vld1U8Words(utmp[0], utmp[1])
        var scaleIdx = 0

        var q5 = 0
        var q8 = 0

        val qhbits = vld1qU8X2(x.qh, 0)

        var sumi = 0

        for (j in 0 until QKK / 64) {
            val q5bits = vld1qU8X2(x.qs, q5)
            q5 += 32
            val q8bytes = vld1qS8X4(y.qs, q8)
            q8 += 64

            val q5h0 = vshlqNU8(vandqU8(mone, qhbits[0]), 4)
            val q5h1 = vshlqNU8(vandqU8(mone, qhbits[1]), 4)
            val q5h2 = vshlqNU8(vandqU8(mtwo, qhbits[0]), 3)
            val q5h3 = vshlqNU8(vandqU8(mtwo, qhbits[1]), 3)
            qhbits[0] = vshrqNU8(qhbits[0], 2)
            qhbits[1] = vshrqNU8(qhbits[1], 2)

            val q5bytes0 = vreinterpretqS8U8(vorrqU8(vandqU8(q5bits[0], m4b), q5h0))
            val q5bytes1 = vreinterpretqS8U8(vorrqU8(vandqU8(q5bits[1], m4b), q5h1))
            val q5bytes2 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5bits[0], 4), q5h2))
            val q5bytes3 = vreinterpretqS8U8(vorrqU8(vshrqNU8(q5bits[1], 4), q5h3))

            val p0 = vdotqS32(q5bytes0, q8bytes[0])
            val p1 = vdotqS32(q5bytes1, q8bytes[1])
            sumi += vaddvqS32(vaddqS32(p0, p1)) * (scales[scaleIdx].toInt() and 0xFF)
            scaleIdx += 1

            val p2 = vdotqS32(q5bytes2, q8bytes[2])
            val p3 = vdotqS32(q5bytes3, q8bytes[3])
            sumi += vaddvqS32(vaddqS32(p2, p3)) * (scales[scaleIdx].toInt() and 0xFF)
            scaleIdx += 1
        }
        sumf += d * sumi.toFloat() - dmin * sumiMins.toFloat()
    }
    return sumf
}

internal fun vecDotQ4kQ8k(n: Int, xs: List<BlockQ4K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ4kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val scales = UByteArray(16)
    val kmask1 = 0x3f3f3f3f
    val kmask2 = 0x0f0f0f0f
    val kmask3 = 0x03030303

    val m4b = vdupqNU8(0xF)

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = y.d * x.dmin.toFloat()

        val q8sums = vpaddqS16(
            vld1qS16(y.bsums, 0),
            vld1qS16(y.bsums, 8),
        )

        readU32Into(x.scales, utmp, 3)

        val mins8 = vld1U8Words(
            utmp[1] and kmask1,
            ((utmp[2] ushr 4) and kmask2) or (((utmp[1] ushr 6) and kmask3) shl 4),
        )
        utmp[1] = (utmp[2] and kmask2) or (((utmp[0] ushr 6) and kmask3) shl 4)
        utmp[0] = utmp[0] and kmask1

        val mins = vreinterpretqS16U16(vmovlU8(mins8))
        val prod = vaddqS32(
            vmullS16(vgetLowS16(q8sums), vgetLowS16(mins)),
            vmullS16(vgetHighS16(q8sums), vgetHighS16(mins)),
        )
        sumf -= dmin * vaddvqS32(prod).toFloat()

        writeU32Into(utmp, scales)

        var q4 = 0
        var q8 = 0

        var sumi1 = 0
        var sumi2 = 0

        for (j in 0 until QKK / 64) {
            val q4bits = vld1qU8X2(x.qs, q4)
            q4 += 32
            var q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            var q4bytes = arrayOf(
                vreinterpretqS8U8(vandqU8(q4bits[0], m4b)),
                vreinterpretqS8U8(vandqU8(q4bits[1], m4b)),
            )
            val p0 = vdotqS32(q4bytes[0], q8bytes[0])
            val p1 = vdotqS32(q4bytes[1], q8bytes[1])
            sumi1 += vaddvqS32(vaddqS32(p0, p1)) * scales[2 * j].toInt()

            q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            q4bytes = arrayOf(
                vreinterpretqS8U8(vshrqNU8(q4bits[0], 4)),
                vreinterpretqS8U8(vshrqNU8(q4bits[1], 4)),
            )
            val p2 = vdotqS32(q4bytes[0], q8bytes[0])
            val p3 = vdotqS32(q4bytes[1], q8bytes[1])
            sumi2 += vaddvqS32(vaddqS32(p2, p3)) * scales[2 * j + 1].toInt()
        }
        sumf += d * (sumi1 + sumi2).toFloat()
    }
    return sumf
}

internal fun vecDotQ3kQ8k(n: Int, xs: List<BlockQ3K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ3kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val utmp = IntArray(4)
    val aux = IntArray(3)
    val kmask1 = 0x03030303
    val kmask2 = 0x0f0f0f0f

    val m3b = vdupqNU8(0x3)
    val m0 = vdupqNU8(1)
    val m1 = vshlqNU8(m0, 1)
    val m2 = vshlqNU8(m0, 2)
    val m3 = vshlqNU8(m0, 3)

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        var q3 = 0
        var q8 = 0

        val qhbits = vld1qU8X2(x.hmask, 0)

        var isum = 0

        // Set up scales
        readU32Into(x.scales, aux, 3)

        utmp[3] = ((aux[1] ushr 4) and kmask2) or (((aux[2] ushr 6) and kmask1) shl 4)
        utmp[2] = ((aux[0] ushr 4) and kmask2) or (((aux[2] ushr 4) and kmask1) shl 4)
        utmp[1] = (aux[1] and kmask2) or (((aux[2] ushr 2) and kmask1) shl 4)
        utmp[0] = (aux[0] and kmask2) or ((aux[2] and kmask1) shl 4)

        val scale = utmpToI8(utmp)
        for (j in 0 until 16) {
            scale[j] = (scale[j].toInt() - 32).toByte()
        }
        var scaleOff = 0

        for (j in 0 until QKK / 128) {
            val q3bits = vld1qU8X2(x.qs, q3)
            q3 += 32
            val q8bytes1 = vld1qS8X4(y.qs, q8)
            q8 += 64
            val q8bytes2 = vld1qS8X4(y.qs, q8)
            q8 += 64

            var q3h0 = vshlqNU8(vbicqU8(m0, qhbits[0]), 2)
            var q3h1 = vshlqNU8(vbicqU8(m0, qhbits[1]), 2)
            var q3h2 = vshlqNU8(vbicqU8(m1, qhbits[0]), 1)
            var q3h3 = vshlqNU8(vbicqU8(m1, qhbits[1]), 1)

            var q3bytes0 = vsubqS8(
                vreinterpretqS8U8(vandqU8(q3bits[0], m3b)),
                vreinterpretqS8U8(q3h0),
            )
            var q3bytes1 = vsubqS8(
                vreinterpretqS8U8(vandqU8(q3bits[1], m3b)),
                vreinterpretqS8U8(q3h1),
            )
            var q3bytes2 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[0], 2), m3b)),
                vreinterpretqS8U8(q3h2),
            )
            var q3bytes3 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[1], 2), m3b)),
                vreinterpretqS8U8(q3h3),
            )

            var p0 = vdotqS32(q3bytes0, q8bytes1[0])
            var p1 = vdotqS32(q3bytes1, q8bytes1[1])
            var p2 = vdotqS32(q3bytes2, q8bytes1[2])
            var p3 = vdotqS32(q3bytes3, q8bytes1[3])
            isum += vaddvqS32(p0) * scale[scaleOff].toInt() +
                vaddvqS32(p1) * scale[scaleOff + 1].toInt() +
                vaddvqS32(p2) * scale[scaleOff + 2].toInt() +
                vaddvqS32(p3) * scale[scaleOff + 3].toInt()
            scaleOff += 4

            q3h0 = vbicqU8(m2, qhbits[0])
            q3h1 = vbicqU8(m2, qhbits[1])
            q3h2 = vshrqNU8(vbicqU8(m3, qhbits[0]), 1)
            q3h3 = vshrqNU8(vbicqU8(m3, qhbits[1]), 1)

            q3bytes0 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[0], 4), m3b)),
                vreinterpretqS8U8(q3h0),
            )
            q3bytes1 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[1], 4), m3b)),
                vreinterpretqS8U8(q3h1),
            )
            q3bytes2 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[0], 6), m3b)),
                vreinterpretqS8U8(q3h2),
            )
            q3bytes3 = vsubqS8(
                vreinterpretqS8U8(vandqU8(vshrqNU8(q3bits[1], 6), m3b)),
                vreinterpretqS8U8(q3h3),
            )

            p0 = vdotqS32(q3bytes0, q8bytes2[0])
            p1 = vdotqS32(q3bytes1, q8bytes2[1])
            p2 = vdotqS32(q3bytes2, q8bytes2[2])
            p3 = vdotqS32(q3bytes3, q8bytes2[3])
            isum += vaddvqS32(p0) * scale[scaleOff].toInt() +
                vaddvqS32(p1) * scale[scaleOff + 1].toInt() +
                vaddvqS32(p2) * scale[scaleOff + 2].toInt() +
                vaddvqS32(p3) * scale[scaleOff + 3].toInt()
            scaleOff += 4

            if (j == 0) {
                qhbits[0] = vshrqNU8(qhbits[0], 4)
                qhbits[1] = vshrqNU8(qhbits[1], 4)
            }
        }
        sumf += d * isum.toFloat()
    }
    return sumf
}

internal fun vecDotQ2kQ8k(n: Int, xs: List<BlockQ2K>, ys: List<BlockQ8K>): Float {
    require(n % QKK == 0) {
        "vecDotQ2kQ8k: $n is not divisible by $QKK"
    }
    var sumf = 0.0f
    val aux = UByteArray(16)

    val m3 = vdupqNU8(0x3)
    val m4 = vdupqNU8(0xF)

    for ((x, y) in xs.zip(ys)) {
        val d = y.d * x.d.toFloat()
        val dmin = -y.d * x.dmin.toFloat()

        var q2 = 0
        var q8 = 0

        val minsAndScales = vld1qU8(x.scales, 0)
        val scales = vandqU8(minsAndScales, m4)
        vst1qU8(aux, 0, scales)

        val mins = vshrqNU8(minsAndScales, 4)
        val q8sums = vld1qS16X2(y.bsums, 0)
        val mins16 = arrayOf(
            vreinterpretqS16U16(vmovlU8(vgetLowU8(mins))),
            vreinterpretqS16U16(vmovlU8(vgetHighU8(mins))),
        )
        val s0 = vaddqS32(
            vmullS16(vgetLowS16(mins16[0]), vgetLowS16(q8sums[0])),
            vmullS16(vgetHighS16(mins16[0]), vgetHighS16(q8sums[0])),
        )
        val s1 = vaddqS32(
            vmullS16(vgetLowS16(mins16[1]), vgetLowS16(q8sums[1])),
            vmullS16(vgetHighS16(mins16[1]), vgetHighS16(q8sums[1])),
        )
        sumf += dmin * vaddvqS32(vaddqS32(s0, s1)).toFloat()

        var isum = 0
        var isIdx = 0

        for (j in 0 until QKK / 128) {
            val q2bits = vld1qU8X2(x.qs, q2)
            q2 += 32

            var q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            val q2bytes = arrayOf(
                vreinterpretqS8U8(vandqU8(q2bits[0], m3)),
                vreinterpretqS8U8(vandqU8(q2bits[1], m3)),
            )
            isum += multiplyAccumWithScale(aux, isIdx, 0, q2bytes, q8bytes)

            q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            q2bytes[0] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[0], 2), m3))
            q2bytes[1] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[1], 2), m3))
            isum += multiplyAccumWithScale(aux, isIdx, 2, q2bytes, q8bytes)

            q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            q2bytes[0] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[0], 4), m3))
            q2bytes[1] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[1], 4), m3))
            isum += multiplyAccumWithScale(aux, isIdx, 4, q2bytes, q8bytes)

            q8bytes = vld1qS8X2(y.qs, q8)
            q8 += 32
            q2bytes[0] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[0], 6), m3))
            q2bytes[1] = vreinterpretqS8U8(vandqU8(vshrqNU8(q2bits[1], 6), m3))
            isum += multiplyAccumWithScale(aux, isIdx, 6, q2bytes, q8bytes)

            isIdx += 8
        }
        sumf += d * isum.toFloat()
    }
    return sumf
}

private fun multiplyAccumWithScale(
    aux: UByteArray,
    isIdx: Int,
    index: Int,
    q2bytes: Array<ByteArray>,
    q8bytes: Array<ByteArray>,
): Int {
    val p1 = vdotqS32(q2bytes[0], q8bytes[0])
    val p2 = vdotqS32(q2bytes[1], q8bytes[1])
    return vaddvqS32(p1) * aux[isIdx + index].toInt() + vaddvqS32(p2) * aux[isIdx + 1 + index].toInt()
}
