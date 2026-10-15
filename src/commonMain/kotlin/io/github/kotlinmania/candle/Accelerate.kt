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

internal expect val accelVvexpf: (FloatArray, FloatArray, CInt) -> Unit
internal expect val accelVvsqrtf: (FloatArray, FloatArray, CInt) -> Unit
internal expect val accelVvtanhf: (FloatArray, FloatArray, CInt) -> Unit

public object Accelerate {
    public object Ffi {
        public val sgemmFfi: (
            CChar,
            CChar,
            CInt,
            CInt,
            CInt,
            CFloat,
            FloatArray,
            CInt,
            FloatArray,
            CInt,
            CFloat,
            FloatArray,
            CInt,
        ) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
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

        public val dgemmFfi: (
            CChar,
            CChar,
            CInt,
            CInt,
            CInt,
            CDouble,
            DoubleArray,
            CInt,
            DoubleArray,
            CInt,
            CDouble,
            DoubleArray,
            CInt,
        ) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
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

        public val vvexpf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            accelVvexpf(dst, src, len)
        }

        public val vvexp: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = exp(src[i])
        }

        public val vvsqrtf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            accelVvsqrtf(dst, src, len)
        }

        public val vvsqrt: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = sqrt(src[i])
        }

        public val vvsinf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = sin(src[i])
        }

        public val vvsin: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = sin(src[i])
        }

        public val vvcosf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = cos(src[i])
        }

        public val vvcos: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = cos(src[i])
        }

        public val vvlogf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = ln(src[i])
        }

        public val vvlog: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = ln(src[i])
        }

        public val vvtanhf: (FloatArray, FloatArray, CInt) -> Unit = { dst, src, len ->
            accelVvtanhf(dst, src, len)
        }

        public val vvtanh: (DoubleArray, DoubleArray, CInt) -> Unit = { dst, src, len ->
            for (i in 0 until len) dst[i] = tanh(src[i])
        }

        public val vdspVaddD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
                }
            }

        public val vdspVadd: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] + a[(i * ia).toInt()]
                }
            }

        public val vdspVsubD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
                }
            }

        public val vdspVsub: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] - a[(i * ia).toInt()]
                }
            }

        public val vdspVmulD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
                }
            }

        public val vdspVmul: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] * a[(i * ia).toInt()]
                }
            }

        public val vdspVdivD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
                }
            }

        public val vdspVdiv: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = b[(i * ib).toInt()] / a[(i * ia).toInt()]
                }
            }

        public val vdspVminD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
                }
            }

        public val vdspVmin: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = minOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
                }
            }

        public val vdspVmaxD: (DoubleArray, CLong, DoubleArray, CLong, DoubleArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                val count = n.toInt()
                for (i in 0 until count) {
                    c[(i * ic).toInt()] = maxOf(b[(i * ib).toInt()], a[(i * ia).toInt()])
                }
            }

        public val vdspVmax: (FloatArray, CLong, FloatArray, CLong, FloatArray, CLong, CULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
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
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvexpf(y, a, aLen)
    }

    public fun vdExp(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvexp(y, a, aLen)
    }

    public fun vsSqrt(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvsqrtf(y, a, aLen)
    }

    public fun vdSqrt(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvsqrt(y, a, aLen)
    }

    public fun vsSin(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvsinf(y, a, aLen)
    }

    public fun vdSin(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvsin(y, a, aLen)
    }

    public fun vsCos(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvcosf(y, a, aLen)
    }

    public fun vdCos(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvcos(y, a, aLen)
    }

    public fun vsTanh(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvtanhf(y, a, aLen)
    }

    public fun vdTanh(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvtanh(y, a, aLen)
    }

    public fun vsLn(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvlogf(y, a, aLen)
    }

    public fun vdLn(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vvlog(y, a, aLen)
    }

    public fun vsSqr(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        for (i in 0 until aLen) {
            y[i] = a[i] * a[i]
        }
    }

    public fun vdSqr(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        for (i in 0 until aLen) {
            y[i] = a[i] * a[i]
        }
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
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = sqrt(2.0f / PI.toFloat()) * v * (1.0f + 0.044715f * v * v)
        }
        vsTanhInplace(ys)
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = 0.5f * v * (1.0f + ys[i])
        }
    }

    public fun vdGelu(vs: DoubleArray, ys: DoubleArray) {
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = sqrt(2.0 / PI) * v * (1.0 + 0.044715 * v * v)
        }
        vdTanhInplace(ys)
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = 0.5 * v * (1.0 + ys[i])
        }
    }

    public fun vsSilu(vs: FloatArray, ys: FloatArray) {
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = -v
        }
        vsExpInplace(ys)
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = v / (1.0f + ys[i])
        }
    }

    public fun vdSilu(vs: DoubleArray, ys: DoubleArray) {
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = -v
        }
        vdExpInplace(ys)
        for (i in vs.indices) {
            val v = vs[i]
            ys[i] = v / (1.0 + ys[i])
        }
    }

    public val vsAdd: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsAdd a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVadd(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdAdd: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdAdd a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVaddD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vsSub: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsSub a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVsub(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdSub: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdSub a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVsubD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vsMul: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMul a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVmul(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdMul: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMul a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVmulD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vsDiv: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsDiv a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVdiv(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdDiv: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdDiv a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVdivD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vsMax: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMax a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVmax(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdMax: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMax a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVmaxD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vsMin: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMin a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVmin(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }

    public val vdMin: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMin a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdspVminD(b, 1L, a, 1L, y, 1L, aLen.toULong())
    }
}
