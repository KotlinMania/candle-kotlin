// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.libc.CInt
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh

internal actual val vdupqNF32: (Float) -> FloatArray = { x -> FloatArray(4) { x } }
internal actual val vld1qF32: (FloatArray, Int) -> FloatArray = { mem, off -> FloatArray(4) { mem[off + it] } }
internal actual val vaddqF32: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] + b[it] } }
internal actual val vfmaqF32: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    FloatArray(4) { a[it] + b[it] * c[it] }
}
internal actual val vst1qF32: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    for (i in 0 until 4) mem[off + i] = a[i]
}
internal actual val vaddvqF32: (FloatArray) -> Float = { x -> x[0] + x[1] + x[2] + x[3] }
internal actual val vgetqLaneF32: (FloatArray, Int) -> Float = { x, lane -> x[lane] }
internal actual val fcvtl4s4h: (ShortArray) -> FloatArray = { bits ->
    FloatArray(4) { Float16Utils.f16BitsToFloat(bits[it]) }
}
internal actual val fcvtn4h4s: (FloatArray) -> ShortArray = { a ->
    ShortArray(4) { Float16Utils.floatToF16Bits(a[it]) }
}
internal actual val asmMovi8hZero: () -> ShortArray = { ShortArray(8) }
internal actual val asmFadd8h: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    ShortArray(8) {
        Float16Utils.floatToF16Bits(
            Float16Utils.f16BitsToFloat(a[it]) + Float16Utils.f16BitsToFloat(b[it]),
        )
    }
}
internal actual val asmFmla8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    ShortArray(8) {
        Float16Utils.floatToF16Bits(
            Float16Utils.f16BitsToFloat(a[it]) +
                Float16Utils.f16BitsToFloat(b[it]) * Float16Utils.f16BitsToFloat(c[it]),
        )
    }
}
internal actual val asmDup8hFromF32: (Float) -> ShortArray = { v ->
    val bits = Float16Utils.floatToF16Bits(v)
    ShortArray(8) { bits }
}
internal actual val asmReduceF16x8: (ShortArray) -> Float = { x ->
    var sum = 0.0f
    for (i in 0 until 8) sum += Float16Utils.f16BitsToFloat(x[i])
    sum
}
internal actual val asmFadd4sOnU16x8: (ShortArray, ShortArray) -> ShortArray = { a, b ->
    val out = ShortArray(8)
    for (lane in 0 until 4) {
        val fa = Float.fromBits((a[2 * lane].toInt() and 0xFFFF) or ((a[2 * lane + 1].toInt() and 0xFFFF) shl 16))
        val fb = Float.fromBits((b[2 * lane].toInt() and 0xFFFF) or ((b[2 * lane + 1].toInt() and 0xFFFF) shl 16))
        val sumBits = (fa + fb).toRawBits()
        out[2 * lane] = (sumBits and 0xFFFF).toShort()
        out[2 * lane + 1] = (sumBits ushr 16).toShort()
    }
    out
}
internal actual val asmBfdot4s8h: (ShortArray, ShortArray, ShortArray) -> ShortArray = { a, b, c ->
    val out = ShortArray(8)
    for (lane in 0 until 4) {
        val acc = Float.fromBits((a[2 * lane].toInt() and 0xFFFF) or ((a[2 * lane + 1].toInt() and 0xFFFF) shl 16))
        val b0 = Float16Utils.bf16BitsToFloat(b[2 * lane])
        val b1 = Float16Utils.bf16BitsToFloat(b[2 * lane + 1])
        val c0 = Float16Utils.bf16BitsToFloat(c[2 * lane])
        val c1 = Float16Utils.bf16BitsToFloat(c[2 * lane + 1])
        val nextBits = (acc + b0 * c0 + b1 * c1).toRawBits()
        out[2 * lane] = (nextBits and 0xFFFF).toShort()
        out[2 * lane + 1] = (nextBits ushr 16).toShort()
    }
    out
}
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(8) { a[it] + b[it] } }
internal actual val mm256CvtphPs: (ShortArray) -> FloatArray = { a ->
    FloatArray(8) { Float16Utils.f16BitsToFloat(a[it]) }
}
internal actual val accelVvexpf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = exp(src[i])
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = sqrt(src[i])
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = tanh(src[i])
}
