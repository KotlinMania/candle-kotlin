// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle.internal

internal object CandleJni {
    init {
        System.loadLibrary("candle_jni")
    }

    @JvmStatic external fun neonReduceOneF32(x: FloatArray): Float

    @JvmStatic external fun neonReduceOneArmF32(x: FloatArray, offset: Int): Float

    @JvmStatic external fun neonVdupqNF32(v: Float, out: FloatArray)

    @JvmStatic external fun neonVld1qF32(memAddr: FloatArray, offset: Int, out: FloatArray)

    @JvmStatic external fun neonVaddqF32(a: FloatArray, b: FloatArray, out: FloatArray)

    @JvmStatic external fun neonVfmaqF32(a: FloatArray, b: FloatArray, c: FloatArray, out: FloatArray)

    @JvmStatic external fun neonVst1qF32(memAddr: FloatArray, offset: Int, a: FloatArray)

    @JvmStatic external fun neonF16LoadFcvtl(memAddr: ShortArray, offset: Int, out: FloatArray)

    @JvmStatic external fun neonF16StoreFcvtn(memAddr: ShortArray, offset: Int, a: FloatArray)

    @JvmStatic external fun neonF16ReduceOne(x: ShortArray): Float

    @JvmStatic external fun neonF16Zero(out: ShortArray)

    @JvmStatic external fun neonF16VecAdd(a: ShortArray, b: ShortArray, out: ShortArray)

    @JvmStatic external fun neonF16VecFma(a: ShortArray, b: ShortArray, c: ShortArray, out: ShortArray)

    @JvmStatic external fun neonF16FromF32(v: Float, out: ShortArray)

    @JvmStatic external fun neonF16VecStore(memAddr: ShortArray, offset: Int, a: ShortArray)

    @JvmStatic external fun neonBf16LoadShll(memAddr: ShortArray, offset: Int, out: FloatArray)

    @JvmStatic external fun neonBf16StoreShrn(memAddr: ShortArray, offset: Int, a: FloatArray)

    @JvmStatic external fun neonBf16VecAdd(a: ShortArray, b: ShortArray, out: ShortArray)

    @JvmStatic external fun neonBf16VecFma(a: ShortArray, b: ShortArray, c: ShortArray, out: ShortArray)

    @JvmStatic external fun avxZero(out: FloatArray)

    @JvmStatic external fun avxSet1Ps(v: Float, out: FloatArray)

    @JvmStatic external fun avxLoaduPs(memAddr: FloatArray, offset: Int, out: FloatArray)

    @JvmStatic external fun avxStoreuPs(memAddr: FloatArray, offset: Int, a: FloatArray)

    @JvmStatic external fun avxReduceOnePs(x: FloatArray): Float

    @JvmStatic external fun avxAddPs(a: FloatArray, b: FloatArray, out: FloatArray)

    @JvmStatic external fun avxFmaPs(a: FloatArray, b: FloatArray, c: FloatArray, out: FloatArray)

    @JvmStatic external fun avxCvtphPs(memAddr: ShortArray, offset: Int, out: FloatArray)

    @JvmStatic external fun avxCvtpsPh(a: FloatArray, memAddr: ShortArray, offset: Int)

    @JvmStatic external fun avxCvtpbhPs(memAddr: ShortArray, offset: Int, out: FloatArray)

    @JvmStatic external fun avxCvtnepsPbh(a: FloatArray, memAddr: ShortArray, offset: Int)

    @JvmStatic external fun simd128SplatF32(v: Float, out: FloatArray)

    @JvmStatic external fun simd128LoadF32(memAddr: FloatArray, offset: Int, out: FloatArray)

    @JvmStatic external fun simd128AddF32(a: FloatArray, b: FloatArray, out: FloatArray)

    @JvmStatic external fun simd128MulF32(a: FloatArray, b: FloatArray, out: FloatArray)

    @JvmStatic external fun simd128FmaF32(a: FloatArray, b: FloatArray, c: FloatArray, out: FloatArray)

    @JvmStatic external fun simd128StoreF32(memAddr: FloatArray, offset: Int, a: FloatArray)

    @JvmStatic external fun simd128ReduceOneF32(x: FloatArray): Float

    @JvmStatic external fun accelerateSgemm(
        transa: Byte,
        transb: Byte,
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
    )

    @JvmStatic external fun accelerateDgemm(
        transa: Byte,
        transb: Byte,
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
    )

    @JvmStatic external fun accelerateVvexpf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvexp(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVvsqrtf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvsqrt(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVvsinf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvsin(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVvcosf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvcos(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVvlogf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvlog(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVvtanhf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvtanh(dst: DoubleArray, src: DoubleArray, len: Int)

    @JvmStatic external fun accelerateVdspVadd(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVaddD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVsub(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVsubD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVmul(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVmulD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVdiv(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVdivD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVmin(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVminD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVmax(a: FloatArray, ia: Long, b: FloatArray, ib: Long, c: FloatArray, ic: Long, n: Long)

    @JvmStatic external fun accelerateVdspVmaxD(a: DoubleArray, ia: Long, b: DoubleArray, ib: Long, c: DoubleArray, ic: Long, n: Long)

    @JvmStatic external fun qneonVdotqS32(a: ByteArray, b: ByteArray, out: IntArray)

    @JvmStatic external fun qneonVmullS8(a: ByteArray, b: ByteArray, out: ShortArray)

    @JvmStatic external fun qneonVmullS16(a: ShortArray, b: ShortArray, out: IntArray)

    @JvmStatic external fun qneonVpaddlqS16(a: ShortArray, out: IntArray)

    @JvmStatic external fun qneonVpaddqS16(a: ShortArray, b: ShortArray, out: ShortArray)

    @JvmStatic external fun qneonVmlaqNF32(acc: FloatArray, v: FloatArray, s: Float, out: FloatArray)

    @JvmStatic external fun qneonVaddvqS32(a: IntArray): Int

    @JvmStatic external fun qavxMaddEpi16(a: ShortArray, b: ShortArray, out: IntArray)

    @JvmStatic external fun qavxMmMaddEpi16(a: ShortArray, b: ShortArray, out: IntArray)

    @JvmStatic external fun qavxMaddubsEpi16(ax: ByteArray, sy: ByteArray, out: ShortArray)

    @JvmStatic external fun qavxSignEpi8(a: ByteArray, b: ByteArray, out: ByteArray)

    @JvmStatic external fun qavxHaddEpi16(a: ShortArray, b: ShortArray, out: ShortArray)

    @JvmStatic external fun qavxHaddEpi32(a: IntArray, b: IntArray, out: IntArray)

    @JvmStatic external fun qsimd128DotI16x8(a: ShortArray, b: ShortArray, out: IntArray)
}
