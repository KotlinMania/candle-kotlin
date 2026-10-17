// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
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

internal actual val mm256SetzeroPs: () -> FloatArray = { FloatArray(8) }
internal actual val mm256Set1Ps: (Float) -> FloatArray = { v -> FloatArray(8) { v } }
internal actual val mm256LoaduPs: (FloatArray, Int) -> FloatArray = { mem, off -> FloatArray(8) { mem[off + it] } }
internal actual val mm256AddPs: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(8) { a[it] + b[it] } }
internal actual val mm256FmaPs: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    FloatArray(8) { a[it] + b[it] * c[it] }
}
internal actual val mm256StoreuPs: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    for (i in 0 until 8) mem[off + i] = a[i]
}
internal actual val mm256ReduceOnePs: (FloatArray) -> Float = { x ->
    var sum = 0.0f
    for (i in 0 until 8) sum += x[i]
    sum
}
internal actual val mm256CvtphPs: (ShortArray, Int) -> FloatArray = { a, off ->
    FloatArray(8) { Float16Utils.f16BitsToFloat(a[off + it]) }
}
internal actual val mm256CvtpsPh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    for (i in 0 until 8) mem[off + i] = Float16Utils.floatToF16Bits(a[i])
}
internal actual val mm256CvtpbhPs: (ShortArray, Int) -> FloatArray = { a, off ->
    FloatArray(8) { Float16Utils.bf16BitsToFloat(a[off + it]) }
}
internal actual val mm256CvtnepsPbh: (FloatArray, ShortArray, Int) -> Unit = { a, mem, off ->
    for (i in 0 until 8) mem[off + i] = Float16Utils.floatToBF16Bits(a[i])
}

internal actual val f32x4Splat: (Float) -> FloatArray = { v -> FloatArray(4) { v } }
internal actual val v128Load: (FloatArray, Int) -> FloatArray = { mem, off -> FloatArray(4) { mem[off + it] } }
internal actual val f32x4Add: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] + b[it] } }
internal actual val f32x4Mul: (FloatArray, FloatArray) -> FloatArray = { a, b -> FloatArray(4) { a[it] * b[it] } }
internal actual val f32x4Fma: (FloatArray, FloatArray, FloatArray) -> FloatArray = { a, b, c ->
    FloatArray(4) { a[it] + b[it] * c[it] }
}
internal actual val v128Store: (FloatArray, Int, FloatArray) -> Unit = { mem, off, a ->
    for (i in 0 until 4) mem[off + i] = a[i]
}
internal actual val f32x4ReduceOne: (FloatArray) -> Float = { x -> x[0] + x[1] + x[2] + x[3] }

internal actual val accelSgemm: (
    Byte,
    Byte,
    Int,
    Int,
    Int,
    Float,
    FloatArray,
    Int,
    FloatArray,
    Int,
    Float,
    FloatArray,
    Int,
) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
    val ta = transa == 'T'.code.toByte() || transa == 't'.code.toByte()
    val tb = transb == 'T'.code.toByte() || transb == 't'.code.toByte()
    for (j in 0 until n) {
        for (i in 0 until m) {
            var sum = 0.0f
            for (p in 0 until k) {
                val aVal = if (ta) a[p + i * lda] else a[i + p * lda]
                val bVal = if (tb) b[j + p * ldb] else b[p + j * ldb]
                sum += aVal * bVal
            }
            val idx = j * ldc + i
            c[idx] = alpha * sum + beta * c[idx]
        }
    }
}
internal actual val accelDgemm: (
    Byte,
    Byte,
    Int,
    Int,
    Int,
    Double,
    DoubleArray,
    Int,
    DoubleArray,
    Int,
    Double,
    DoubleArray,
    Int,
) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
    val ta = transa == 'T'.code.toByte() || transa == 't'.code.toByte()
    val tb = transb == 'T'.code.toByte() || transb == 't'.code.toByte()
    for (j in 0 until n) {
        for (i in 0 until m) {
            var sum = 0.0
            for (p in 0 until k) {
                val aVal = if (ta) a[p + i * lda] else a[i + p * lda]
                val bVal = if (tb) b[j + p * ldb] else b[p + j * ldb]
                sum += aVal * bVal
            }
            val idx = j * ldc + i
            c[idx] = alpha * sum + beta * c[idx]
        }
    }
}
internal actual val accelVvexpf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = exp(src[i])
}
internal actual val accelVvexp: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = exp(src[i])
}
internal actual val accelVvsqrtf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = sqrt(src[i])
}
internal actual val accelVvsqrt: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = sqrt(src[i])
}
internal actual val accelVvsinf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = sin(src[i])
}
internal actual val accelVvsin: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = sin(src[i])
}
internal actual val accelVvcosf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = cos(src[i])
}
internal actual val accelVvcos: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = cos(src[i])
}
internal actual val accelVvlogf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = ln(src[i])
}
internal actual val accelVvlog: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = ln(src[i])
}
internal actual val accelVvtanhf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = tanh(src[i])
}
internal actual val accelVvtanh: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
    for (i in 0 until len) dst[i] = tanh(src[i])
}
internal actual val accelVdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
}
internal actual val accelVdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
}
internal actual val accelVdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
}
internal actual val accelVdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
}
internal actual val accelVdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
}
internal actual val accelVdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
}
internal actual val accelVdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
}
internal actual val accelVdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
}
internal actual val accelVdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
}
internal actual val accelVdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
}
internal actual val accelVdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = maxOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
}
internal actual val accelVdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit = { a, ia, b, ib, c, ic, n ->
    for (i in 0 until n.toInt()) c[(i * ic).toInt()] = maxOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
}
