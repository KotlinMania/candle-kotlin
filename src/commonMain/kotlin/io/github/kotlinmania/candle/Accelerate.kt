// port-lint: source accelerate.rs
package io.github.kotlinmania.candle

import kotlin.math.PI
import kotlin.math.sqrt

internal expect val accelSgemm: (
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
) -> Unit
internal expect val accelDgemm: (
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
) -> Unit
internal expect val accelVvexpf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvexp: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVvsqrtf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvsqrt: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVvsinf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvsin: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVvcosf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvcos: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVvlogf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvlog: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVvtanhf: (FloatArray, FloatArray, Int) -> Unit
internal expect val accelVvtanh: (DoubleArray, DoubleArray, Int) -> Unit
internal expect val accelVdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit
internal expect val accelVdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit
internal expect val accelVdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit
internal expect val accelVdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit
internal expect val accelVdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit
internal expect val accelVdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit
internal expect val accelVdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit

public object Accelerate {
    public object Ffi {
        public val sgemmFfi: (
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

        public val dgemmFfi: (
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

        public val vvexpf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvexpf(dst, src, len)
        }

        public val vvexp: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvexp(dst, src, len)
        }

        public val vvsqrtf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvsqrtf(dst, src, len)
        }

        public val vvsqrt: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvsqrt(dst, src, len)
        }

        public val vvsinf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvsinf(dst, src, len)
        }

        public val vvsin: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvsin(dst, src, len)
        }

        public val vvcosf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvcosf(dst, src, len)
        }

        public val vvcos: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvcos(dst, src, len)
        }

        public val vvlogf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvlogf(dst, src, len)
        }

        public val vvlog: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvlog(dst, src, len)
        }

        public val vvtanhf: (FloatArray, FloatArray, Int) -> Unit = { dst, src, len ->
            accelVvtanhf(dst, src, len)
        }

        public val vvtanh: (DoubleArray, DoubleArray, Int) -> Unit = { dst, src, len ->
            accelVvtanh(dst, src, len)
        }

        public val vdspVaddD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVaddD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVadd: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVadd(a, ia, b, ib, c, ic, n)
            }

        public val vdspVsubD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVsubD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVsub: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVsub(a, ia, b, ib, c, ic, n)
            }

        public val vdspVmulD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVmulD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVmul: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVmul(a, ia, b, ib, c, ic, n)
            }

        public val vdspVdivD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVdivD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVdiv: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVdiv(a, ia, b, ib, c, ic, n)
            }

        public val vdspVminD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVminD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVmin: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVmin(a, ia, b, ib, c, ic, n)
            }

        public val vdspVmaxD: (DoubleArray, Long, DoubleArray, Long, DoubleArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVmaxD(a, ia, b, ib, c, ic, n)
            }

        public val vdspVmax: (FloatArray, Long, FloatArray, Long, FloatArray, Long, ULong) -> Unit =
            { a, ia, b, ib, c, ic, n ->
                accelVdspVmax(a, ia, b, ib, c, ic, n)
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
        Ffi.sgemmFfi(
            transa.toByte(),
            transb.toByte(),
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
        Ffi.dgemmFfi(
            transa.toByte(),
            transb.toByte(),
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
