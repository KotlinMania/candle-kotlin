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

    @JvmStatic external fun avxReduceOnePs(x: FloatArray): Float

    @JvmStatic external fun avxAddPs(a: FloatArray, b: FloatArray, out: FloatArray)

    @JvmStatic external fun avxFmaPs(a: FloatArray, b: FloatArray, c: FloatArray, out: FloatArray)

    @JvmStatic external fun avxCvtphPs(memAddr: ShortArray, offset: Int, out: FloatArray)

    @JvmStatic external fun avxCvtpsPh(a: FloatArray, memAddr: ShortArray, offset: Int)

    @JvmStatic external fun accelerateVvexpf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvsqrtf(dst: FloatArray, src: FloatArray, len: Int)

    @JvmStatic external fun accelerateVvtanhf(dst: FloatArray, src: FloatArray, len: Int)
}
