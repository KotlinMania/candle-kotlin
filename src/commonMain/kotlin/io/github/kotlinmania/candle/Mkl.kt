// port-lint: source mkl.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

public object Mkl {
    public object Ffi {
        public fun vsTanh(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = tanh(a[i])
        }

        public fun vdTanh(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = tanh(a[i])
        }

        public fun vsExp(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = exp(a[i])
        }

        public fun vdExp(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = exp(a[i])
        }

        public fun vsLn(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = ln(a[i])
        }

        public fun vdLn(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = ln(a[i])
        }

        public fun vsSin(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = sin(a[i])
        }

        public fun vdSin(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = sin(a[i])
        }

        public fun vsCos(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = cos(a[i])
        }

        public fun vdCos(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = cos(a[i])
        }

        public fun vsSqrt(n: CInt, a: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = sqrt(a[i])
        }

        public fun vdSqrt(n: CInt, a: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = sqrt(a[i])
        }

        public fun vsAdd(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = a[i] + b[i]
        }

        public fun vdAdd(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = a[i] + b[i]
        }

        public fun vsSub(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = a[i] - b[i]
        }

        public fun vdSub(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = a[i] - b[i]
        }

        public fun vsMul(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = a[i] * b[i]
        }

        public fun vdMul(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = a[i] * b[i]
        }

        public fun vsDiv(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = a[i] / b[i]
        }

        public fun vdDiv(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = a[i] / b[i]
        }

        public fun vsFmax(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = maxOf(a[i], b[i])
        }

        public fun vdFmax(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = maxOf(a[i], b[i])
        }

        public fun vsFmin(n: CInt, a: FloatArray, b: FloatArray, y: FloatArray) {
            for (i in 0 until n) y[i] = minOf(a[i], b[i])
        }

        public fun vdFmin(n: CInt, a: DoubleArray, b: DoubleArray, y: DoubleArray) {
            for (i in 0 until n) y[i] = minOf(a[i], b[i])
        }

        public fun sgemm(
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

        public fun dgemm(
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

        public fun hgemm(
            transa: CChar,
            transb: CChar,
            m: CInt,
            n: CInt,
            k: CInt,
            alpha: Short,
            a: ShortArray,
            lda: CInt,
            b: ShortArray,
            ldb: CInt,
            beta: Short,
            c: ShortArray,
            ldc: CInt,
        ) {
            val ta = transa == 'T'.code.toByte() || transa == 't'.code.toByte()
            val tb = transb == 'T'.code.toByte() || transb == 't'.code.toByte()
            val alphaF = Float16Utils.f16BitsToFloat(alpha)
            val betaF = Float16Utils.f16BitsToFloat(beta)
            for (j in 0 until n) {
                for (i in 0 until m) {
                    var sum = 0.0f
                    for (p in 0 until k) {
                        val aVal = Float16Utils.f16BitsToFloat(if (ta) a[i * lda + p] else a[p * lda + i])
                        val bVal = Float16Utils.f16BitsToFloat(if (tb) b[p * ldb + j] else b[j * ldb + p])
                        sum += aVal * bVal
                    }
                    val idx = j * ldc + i
                    val cur = Float16Utils.f16BitsToFloat(c[idx])
                    c[idx] = Float16Utils.floatToF16Bits(alphaF * sum + betaF * cur)
                }
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
        Ffi.sgemm(transa.toByte(), transb.toByte(), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
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
        Ffi.dgemm(transa.toByte(), transb.toByte(), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    public fun hgemm(
        transa: UByte,
        transb: UByte,
        m: Int,
        n: Int,
        k: Int,
        alpha: Short,
        a: ShortArray,
        lda: Int,
        b: ShortArray,
        ldb: Int,
        beta: Short,
        c: ShortArray,
        ldc: Int,
    ) {
        Ffi.hgemm(transa.toByte(), transb.toByte(), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    public fun vsExp(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsExp(aLen, a, y)
    }

    public fun vdExp(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdExp(aLen, a, y)
    }

    public fun vsLn(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsLn(aLen, a, y)
    }

    public fun vdLn(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdLn(aLen, a, y)
    }

    public fun vsSin(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsSin(aLen, a, y)
    }

    public fun vdSin(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdSin(aLen, a, y)
    }

    public fun vsCos(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsCos(aLen, a, y)
    }

    public fun vdCos(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdCos(aLen, a, y)
    }

    public fun vsSqrt(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsSqrt(aLen, a, y)
    }

    public fun vdSqrt(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdSqrt(aLen, a, y)
    }

    public fun vsSqr(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsMul(aLen, a, a, y)
    }

    public fun vdSqr(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdMul(aLen, a, a, y)
    }

    public fun vsTanh(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vsTanh(aLen, a, y)
    }

    public fun vdTanh(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        require(aLen == yLen) { "a and y have different lengths $aLen <> $yLen" }
        Ffi.vdTanh(aLen, a, y)
    }

    public fun vsTanhInplace(y: FloatArray) {
        Ffi.vsTanh(y.size, y, y)
    }

    public fun vdTanhInplace(y: DoubleArray) {
        Ffi.vdTanh(y.size, y, y)
    }

    public fun vsExpInplace(y: FloatArray) {
        Ffi.vsExp(y.size, y, y)
    }

    public fun vdExpInplace(y: DoubleArray) {
        Ffi.vdExp(y.size, y, y)
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
        Ffi.vsAdd(a.size, a, b, y)
    }

    public fun vdAdd(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdAdd a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdAdd(a.size, a, b, y)
    }

    public fun vsSub(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsSub a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vsSub(a.size, a, b, y)
    }

    public fun vdSub(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdSub a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdSub(a.size, a, b, y)
    }

    public fun vsMul(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMul a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vsMul(a.size, a, b, y)
    }

    public fun vdMul(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMul a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdMul(a.size, a, b, y)
    }

    public fun vsDiv(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsDiv a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vsDiv(a.size, a, b, y)
    }

    public fun vdDiv(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdDiv a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdDiv(a.size, a, b, y)
    }

    public fun vsMax(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMax a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vsFmax(a.size, a, b, y)
    }

    public fun vdMax(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMax a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdFmax(a.size, a, b, y)
    }

    public fun vsMin(a: FloatArray, b: FloatArray, y: FloatArray) {
        require(a.size == y.size && b.size == y.size) { "vsMin a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vsFmin(a.size, a, b, y)
    }

    public fun vdMin(a: DoubleArray, b: DoubleArray, y: DoubleArray) {
        require(a.size == y.size && b.size == y.size) { "vdMin a,b,y len mismatch ${a.size} ${b.size} ${y.size}" }
        Ffi.vdFmin(a.size, a, b, y)
    }
}
