// port-lint: source mkl.rs
package io.github.kotlinmania.candle

import kotlin.math.PI
import kotlin.math.sqrt

public object Mkl {
    public object Ffi {
        public val vsTanh: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvtanhf(y, a, n)
        }

        public val vdTanh: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvtanh(y, a, n)
        }

        public val vsExp: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvexpf(y, a, n)
        }

        public val vdExp: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvexp(y, a, n)
        }

        public val vsLn: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvlogf(y, a, n)
        }

        public val vdLn: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvlog(y, a, n)
        }

        public val vsSin: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvsinf(y, a, n)
        }

        public val vdSin: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvsin(y, a, n)
        }

        public val vsCos: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvcosf(y, a, n)
        }

        public val vdCos: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvcos(y, a, n)
        }

        public val vsSqrt: (Int, FloatArray, FloatArray) -> Unit = { n, a, y ->
            accelVvsqrtf(y, a, n)
        }

        public val vdSqrt: (Int, DoubleArray, DoubleArray) -> Unit = { n, a, y ->
            accelVvsqrt(y, a, n)
        }

        public val vsAdd: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVadd(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdAdd: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVaddD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vsSub: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVsub(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdSub: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVsubD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vsMul: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVmul(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdMul: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVmulD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vsDiv: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVdiv(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdDiv: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVdivD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vsFmax: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVmax(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdFmax: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVmaxD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vsFmin: (Int, FloatArray, FloatArray, FloatArray) -> Unit = { n, a, b, y ->
            accelVdspVmin(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val vdFmin: (Int, DoubleArray, DoubleArray, DoubleArray) -> Unit = { n, a, b, y ->
            accelVdspVminD(b, 1L, a, 1L, y, 1L, n.toULong())
        }

        public val sgemm: (
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
            accelSgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }

        public val dgemm: (
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
            accelDgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        }

        public val hgemm: (
            Byte,
            Byte,
            Int,
            Int,
            Int,
            Short,
            ShortArray,
            Int,
            ShortArray,
            Int,
            Short,
            ShortArray,
            Int,
        ) -> Unit = { transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc ->
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
        Ffi.sgemm(
            transa.toByte() as Byte,
            transb.toByte() as Byte,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
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
        Ffi.dgemm(
            transa.toByte() as Byte,
            transb.toByte() as Byte,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
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
        Ffi.hgemm(
            transa.toByte() as Byte,
            transb.toByte() as Byte,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
    }

    public fun vsExp(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsExp(aLen as Int, a, y)
    }

    public fun vdExp(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdExp(aLen as Int, a, y)
    }

    public fun vsLn(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsLn(aLen as Int, a, y)
    }

    public fun vdLn(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdLn(aLen as Int, a, y)
    }

    public fun vsSin(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsSin(aLen as Int, a, y)
    }

    public fun vdSin(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdSin(aLen as Int, a, y)
    }

    public fun vsCos(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsCos(aLen as Int, a, y)
    }

    public fun vdCos(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdCos(aLen as Int, a, y)
    }

    public fun vsSqrt(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsSqrt(aLen as Int, a, y)
    }

    public fun vdSqrt(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdSqrt(aLen as Int, a, y)
    }

    public fun vsSqr(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsMul(aLen as Int, a, a, y)
    }

    public fun vdSqr(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdMul(aLen as Int, a, a, y)
    }

    public fun vsTanh(a: FloatArray, y: FloatArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vsTanh(aLen as Int, a, y)
    }

    public fun vdTanh(a: DoubleArray, y: DoubleArray) {
        val aLen = a.size
        val yLen = y.size
        if (aLen != yLen) {
            error("a and y have different lengths $aLen <> $yLen")
        }
        Ffi.vdTanh(aLen as Int, a, y)
    }

    // The vector functions from mkl can be performed in place by using the same array for input and
    // output.
    public fun vsTanhInplace(y: FloatArray) {
        Ffi.vsTanh(y.size as Int, y, y)
    }

    public fun vdTanhInplace(y: DoubleArray) {
        Ffi.vdTanh(y.size as Int, y, y)
    }

    public fun vsExpInplace(y: FloatArray) {
        Ffi.vsExp(y.size as Int, y, y)
    }

    public fun vdExpInplace(y: DoubleArray) {
        Ffi.vdExp(y.size as Int, y, y)
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
        Ffi.vsAdd(aLen as Int, a, b, y)
    }

    public val vdAdd: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdAdd a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdAdd(aLen as Int, a, b, y)
    }

    public val vsSub: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsSub a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vsSub(aLen as Int, a, b, y)
    }

    public val vdSub: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdSub a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdSub(aLen as Int, a, b, y)
    }

    public val vsMul: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMul a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vsMul(aLen as Int, a, b, y)
    }

    public val vdMul: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMul a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdMul(aLen as Int, a, b, y)
    }

    public val vsDiv: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsDiv a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vsDiv(aLen as Int, a, b, y)
    }

    public val vdDiv: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdDiv a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdDiv(aLen as Int, a, b, y)
    }

    public val vsMax: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMax a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vsFmax(aLen as Int, a, b, y)
    }

    public val vdMax: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMax a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdFmax(aLen as Int, a, b, y)
    }

    public val vsMin: (FloatArray, FloatArray, FloatArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vsMin a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vsFmin(aLen as Int, a, b, y)
    }

    public val vdMin: (DoubleArray, DoubleArray, DoubleArray) -> Unit = { a, b, y ->
        val aLen = a.size
        val bLen = b.size
        val yLen = y.size
        if (aLen != yLen || bLen != yLen) {
            error("vdMin a,b,y len mismatch $aLen $bLen $yLen")
        }
        Ffi.vdFmin(aLen as Int, a, b, y)
    }
}
