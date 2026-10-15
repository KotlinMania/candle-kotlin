// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle.internal

public external interface CandleNativeBindings {
    public fun neonReduceOneF32(x: FloatArray): Float

    public fun neonReduceOneArmF32(x: FloatArray, offset: Int): Float

    public fun neonVdupqNF32(v: Float): FloatArray

    public fun neonVld1qF32(memAddr: FloatArray, offset: Int): FloatArray

    public fun neonVaddqF32(a: FloatArray, b: FloatArray): FloatArray

    public fun neonVfmaqF32(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray

    public fun neonVst1qF32(memAddr: FloatArray, offset: Int, a: FloatArray)

    public fun neonF16LoadFcvtl(memAddr: ShortArray, offset: Int): FloatArray

    public fun neonF16StoreFcvtn(memAddr: ShortArray, offset: Int, a: FloatArray)

    public fun neonF16ReduceOne(x: ShortArray): Float

    public fun neonF16Zero(): ShortArray

    public fun neonF16VecAdd(a: ShortArray, b: ShortArray): ShortArray

    public fun neonF16VecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray

    public fun neonF16FromF32(v: Float): ShortArray

    public fun neonF16VecStore(memAddr: ShortArray, offset: Int, a: ShortArray)

    public fun neonBf16LoadShll(memAddr: ShortArray, offset: Int): FloatArray

    public fun neonBf16StoreShrn(memAddr: ShortArray, offset: Int, a: FloatArray)

    public fun neonBf16VecAdd(a: ShortArray, b: ShortArray): ShortArray

    public fun neonBf16VecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray

    public fun avxReduceOnePs(x: FloatArray): Float

    public fun avxAddPs(a: FloatArray, b: FloatArray): FloatArray

    public fun avxFmaPs(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray

    public fun avxCvtphPs(memAddr: ShortArray, offset: Int): FloatArray

    public fun avxCvtpsPh(a: FloatArray, memAddr: ShortArray, offset: Int)

    public fun accelerateVvexpf(dst: FloatArray, src: FloatArray, len: Int)

    public fun accelerateVvsqrtf(dst: FloatArray, src: FloatArray, len: Int)

    public fun accelerateVvtanhf(dst: FloatArray, src: FloatArray, len: Int)

    public fun qneonVdotqS32(a: ByteArray, b: ByteArray): IntArray

    public fun qneonVmullS8(a: ByteArray, b: ByteArray): ShortArray

    public fun qneonVmullS16(a: ShortArray, b: ShortArray): IntArray

    public fun qneonVpaddlqS16(a: ShortArray): IntArray

    public fun qneonVpaddqS16(a: ShortArray, b: ShortArray): ShortArray

    public fun qneonVmlaqNF32(acc: FloatArray, v: FloatArray, s: Float): FloatArray

    public fun qneonVaddvqS32(a: IntArray): Int

    public fun qavxMaddEpi16(a: ShortArray, b: ShortArray): IntArray

    public fun qavxMmMaddEpi16(a: ShortArray, b: ShortArray): IntArray

    public fun qavxMaddubsEpi16(ax: ByteArray, sy: ByteArray): ShortArray

    public fun qavxSignEpi8(a: ByteArray, b: ByteArray): ByteArray

    public fun qavxHaddEpi16(a: ShortArray, b: ShortArray): ShortArray

    public fun qavxHaddEpi32(a: IntArray, b: IntArray): IntArray

    public fun qsimd128DotI16x8(a: ShortArray, b: ShortArray): IntArray
}

@JsModule("@kotlinmania/candle-native-bindings")
@JsNonModule
public external val CandleNative: CandleNativeBindings
