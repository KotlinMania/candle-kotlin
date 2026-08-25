// port-lint: source candle-core/src/dtype.rs
package io.github.kotlinmania.candle

public enum class DType(
    public val typeName: String,
    public val sizeInBytes: Int,
) {
    U8("u8", 1),
    U32("u32", 4),
    I16("i16", 2),
    I32("i32", 4),
    I64("i64", 8),
    BF16("bf16", 2),
    F16("f16", 2),
    F32("f32", 4),
    F64("f64", 8),
    F8E4M3("f8e4m3", 1),
    F6E2M3("f6e2m3", 0),
    F6E3M2("f6e3m2", 0),
    F4("f4", 0),
    F8E8M0("f8e8m0", 1),
    ;

    public fun asStr(): String = typeName

    public fun isInt(): Boolean =
        when (this) {
            U8, U32, I16, I32, I64 -> true
            BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0 -> false
        }

    public fun isFloat(): Boolean =
        when (this) {
            U8, U32, I16, I32, I64 -> false
            BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0 -> true
        }

    public companion object {
        public fun fromString(s: String): DType =
            when (s.lowercase()) {
                "u8" -> U8
                "u32" -> U32
                "i16" -> I16
                "i32" -> I32
                "i64" -> I64
                "bf16" -> BF16
                "f16" -> F16
                "f32" -> F32
                "f64" -> F64
                "f8e4m3" -> F8E4M3
                "f6e2m3" -> F6E2M3
                "f6e3m2" -> F6E3M2
                "f4" -> F4
                "f8e8m0" -> F8E8M0
                else -> throw CandleException.Msg("cannot parse '$s' as a dtype")
            }
    }
}

public object Float16Utils {
    public fun floatToF16Bits(value: Float): Short {
        val fbits = value.toRawBits()
        val sign = (fbits ushr 16) and 0x8000
        val exp = ((fbits ushr 23) and 0xFF) - (127 - 15)
        val mant = fbits and 0x007FFFFF

        if (exp <= 0) {
            if (exp < -10) {
                return sign.toShort()
            }
            val m = (mant or 0x00800000) ushr (1 - exp)
            return (sign or (m ushr 13)).toShort()
        } else if (exp == 0xFF - (127 - 15)) {
            if (mant == 0) {
                return (sign or 0x7C00).toShort()
            }
            return (sign or 0x7C00 or (mant ushr 13).coerceAtLeast(1)).toShort()
        } else {
            if (exp > 30) {
                return (sign or 0x7C00).toShort()
            }
            return (sign or (exp shl 10) or (mant ushr 13)).toShort()
        }
    }

    public fun f16BitsToFloat(bits: Short): Float {
        val ibits = bits.toInt() and 0xFFFF
        val sign = (ibits and 0x8000) shl 16
        val exp = (ibits and 0x7C00) ushr 10
        val mant = ibits and 0x03FF

        if (exp == 0) {
            if (mant == 0) {
                return Float.fromBits(sign)
            }
            var m = mant
            var shift = 0
            while ((m and 0x0400) == 0) {
                m = m shl 1
                shift++
            }
            m = m and 0x03FF
            val e = (127 - 15 - shift + 1) shl 23
            return Float.fromBits(sign or e or (m shl 13))
        } else if (exp == 0x1F) {
            val e = 0xFF shl 23
            val m = if (mant != 0) mant shl 13 else 0
            return Float.fromBits(sign or e or m)
        } else {
            val e = (exp + (127 - 15)) shl 23
            val m = mant shl 13
            return Float.fromBits(sign or e or m)
        }
    }

    public fun floatToBF16Bits(value: Float): Short =
        (value.toRawBits() ushr 16).toShort()

    public fun bf16BitsToFloat(bits: Short): Float =
        Float.fromBits((bits.toInt() and 0xFFFF) shl 16)
}
