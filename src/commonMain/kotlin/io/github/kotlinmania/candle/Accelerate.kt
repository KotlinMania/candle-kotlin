// port-lint: source accelerate.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CULong
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

public object Accelerate {
    public object Ffi {
        public fun sgemmFfi(
            transa: CChar,
            transb: CChar,
            m: CInt,
            n: CInt,
            k: CInt,
            alpha: CFloat,
            a: FloatArray,
            lda: CInt,
            b: FloatArray,
            ldb: CInt,
            beta: CFloat,
            c: FloatArray,
            ldc: CInt,
        ) {
            val ta = transa == 'T'.code.toByte() || transa == 't'.code.toByte()
            val tb = transb == 'T'.code.toByte() || transb == 't'.code.toByte()
            for (j in 0 until n) {
                for (i in 0 until m) {
                    var sum = 0.0f
                    for (p in 0 until k) {
                        val aVal = if (ta) a[i * lda + p] else a[p * lda + i]
                        val bVal = if (tb) b[p * ldb + j] else b[j * ldb + p]
                        sum += aVal * bVal
                    }
                    val idx = j * ldc + i
                    c[idx] = alpha * sum + beta * c[idx]
                }
            }
        }

        public fun dgemmFfi(
            transa: CChar,
            transb: CChar,
            m: CInt,
            n: CInt,
            k: CInt,
            alpha: CDouble,
            a: DoubleArray,
            lda: CInt,
            b: DoubleArray,
            ldb: CInt,
            beta: CDouble,
            c: DoubleArray,
            ldc: CInt,
        ) {
            val ta = transa == 'T'.code.toByte() || transa == 't'.code.toByte()
            val tb = transb == 'T'.code.toByte() || transb == 't'.code.toByte()
            for (j in 0 until n) {
                for (i in 0 until m) {
                    var sum = 0.0
                    for (p in 0 until k) {
                        val aVal = if (ta) a[i * lda + p] else a[p * lda + i]
                        val bVal = if (tb) b[p * ldb + j] else b[j * ldb + p]
                        sum += aVal * bVal
                    }
                    val idx = j * ldc + i
                    c[idx] = alpha * sum + beta * c[idx]
                }
            }
        }

        public fun vvexpf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = exp(src[i])
        }

        public fun vvexp(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = exp(src[i])
        }

        public fun vvsqrtf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = sqrt(src[i])
        }

        public fun vvsqrt(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = sqrt(src[i])
        }

        public fun vvsinf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = sin(src[i])
        }

        public fun vvsin(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = sin(src[i])
        }

        public fun vvcosf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = cos(src[i])
        }

        public fun vvcos(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = cos(src[i])
        }

        public fun vvlogf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = ln(src[i])
        }

        public fun vvlog(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = ln(src[i])
        }

        public fun vvtanhf(dst: FloatArray, src: FloatArray, len: CInt) {
            for (i in 0 until len) dst[i] = tanh(src[i])
        }

        public fun vvtanh(dst: DoubleArray, src: DoubleArray, len: CInt) {
            for (i in 0 until len) dst[i] = tanh(src[i])
        }

        public fun vdspVaddD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
            }
        }

        public fun vdspVadd(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
            }
        }

        public fun vdspVsubD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
            }
        }

        public fun vdspVsub(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
            }
        }

        public fun vdspVmulD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
            }
        }

        public fun vdspVmul(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
            }
        }

        public fun vdspVdivD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
            }
        }

        public fun vdspVdiv(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
            }
        }

        public fun vdspVminD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
            }
        }

        public fun vdspVmin(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
            }
        }

        public fun vdspVmaxD(
            a: DoubleArray,
            ia: CLong,
            b: DoubleArray,
            ib: CLong,
            c: DoubleArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = maxOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
            }
        }

        public fun vdspVmax(
            a: FloatArray,
            ia: CLong,
            b: FloatArray,
            ib: CLong,
            c: FloatArray,
            ic: CLong,
            n: CULong,
        ) {
            val count = n.toInt()
            for (i in 0 until count) {
                c[(i * ic).toInt()] = maxOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
            }
        }
    }

    public fun sgemm(
        transa: UByte,
        transb: UByte,
        m: Int,
        n: Int,
        k: Int,
        alpha: Float,
        a: FloatArray,
        lda: Int,
        b: FloatArray,
        ldb: Int,
        beta: Float,
        c: FloatArray,
        ldc: Int,
    ) {
        Ffi.sgemmFfi(transa.toByte(), transb.toByte(), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    public fun dgemm(
        transa: UByte,
        transb: UByte,
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    ) {
        Ffi.dgemmFfi(transa.toByte(), transb.toByte(), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    public fun vsExp(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvexpf(y, a, aLen)
    }

    public fun vdExp(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvexp(y, a, aLen)
    }

    public fun vsSqrt(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvsqrtf(y, a, aLen)
    }

    public fun vdSqrt(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvsqrt(y, a, aLen)
    }

    public fun vsSin(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvsinf(y, a, aLen)
    }

    public fun vdSin(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvsin(y, a, aLen)
    }

    public fun vsCos(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvcosf(y, a, aLen)
    }

    public fun vdCos(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvcos(y, a, aLen)
    }

    public fun vsTanh(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvtanhf(y, a, aLen)
    }

    public fun vdTanh(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvtanh(y, a, aLen)
    }

    public fun vsLn(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvlogf(y, a, aLen)
    }

    public fun vdLn(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vvlog(y, a, aLen)
    }

    public fun vsSqr(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        for (i in 0 until aLen) y[i] = a[i] * a[i]
    }

    public fun vdSqr(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        for (i in 0 until aLen) y[i] = a[i] * a[i]
    }

    public fun vsTanhInplace(y: FloatArray) {
        Ffi.vvtanhf(y, y, y.size)
    }

    public fun vdTanhInplace(y: DoubleArray) {
        Ffi.vvtanh(y, y, y.size)
    }

    public fun vsExpInplace(y: FloatArray) {
        Ffi.vvexpf(y, y, y.size)
    }

    public fun vdExpInplace(y: DoubleArray) {
        Ffi.vvexp(y, y, y.size)
    }

    public fun vsGelu(vs: FloatArray, ys: FloatArray) {
        val c = sqrt(2.0f / PI.toFloat())
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = c * v * (1.0f + 0.044715f * v * v)
        }
        vsTanhInplace(ys)
        for (i in vs.indices) {
            ys[i] = 0.5f * vs[i] * (1.0f + ys[i])
        }
    }

    public fun vdGelu(vs: DoubleArray, ys: DoubleArray) {
        val c = sqrt(2.0 / PI)
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = c * v * (1.0 + 0.044715 * v * v)
        }
        vdTanhInplace(ys)
        for (i in vs.indices) {
            ys[i] = 0.5 * vs[i] * (1.0 + ys[i])
        }
    }

    public fun vsSilu(vs: FloatArray, ys: FloatArray) {
        for (i in vs.indices) ys[i] = -vs[i]
        vsExpInplace(ys)
        for (i in vs.indices) ys[i] = vs[i] / (1.0f + ys[i])
    }

    public fun vdSilu(vs: DoubleArray, ys: DoubleArray) {
        for (i in vs.indices) ys[i] = -vs[i]
        vdExpInplace(ys)
        for (i in vs.indices) ys[i] = vs[i] / (1.0 + ys[i])
    }

    public fun vsAdd(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsAdd a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVadd(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdAdd(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdAdd a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVaddD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vsSub(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsSub a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVsub(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdSub(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdSub a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVsubD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vsMul(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMul a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVmul(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdMul(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMul a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVmulD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vsDiv(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsDiv a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVdiv(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdDiv(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdDiv a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVdivD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vsMax(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMax a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVmax(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdMax(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMax a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVmaxD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vsMin(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMin a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVmin(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }

    public fun vdMin(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMin a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdspVminD(b, 1L, a, 1L, y, 1L, a.size.toULong())
    }
}
