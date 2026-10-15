// port-lint: source dtype.rs
package io.github.kotlinmania.candle

/**
 * Types for elements that can be stored and manipulated using tensors.
 */

public class DTypeParseError(public val raw: String) : Exception("cannot parse '$raw' as a dtype")

/**
 * The different types of elements allowed in tensors.
 */
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

    /**
     * String representation for dtypes.
     */
    public fun asStr(): String = typeName

    public fun fmt(): String = asStr()

    /**
     * The size used by each element in bytes, i.e. 1 for U8, 4 for F32.
     */
    public fun sizeInBytes(): Int = sizeInBytes

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
        public interface Err

        public fun fromStr(s: String): DType =
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
                else -> throw DTypeParseError(s)
            }

        public fun fromString(s: String): DType = fromStr(s)
    }
}

public interface WithDType<T> {
    public val dtype: DType
    public fun fromF64(v: Double): T
    public fun toF64(v: T): Double
    public fun toScalar(v: T): Scalar
    public fun toCpuStorage(data: List<T>): CpuStorage
}

public interface IntDType<T> : WithDType<T> {
    public fun isTrue(v: T): Boolean
    public fun asUsize(v: T): Int
}

public object IntDTypeI64 : IntDType<Long> {
    override val dtype: DType = DType.I64
    override fun fromF64(v: Double): Long = v.toLong()
    override fun toF64(v: Long): Double = v.toDouble()
    override fun toScalar(v: Long): Scalar = Scalar.I64(v)
    override fun toCpuStorage(data: List<Long>): CpuStorage = CpuStorage.I64(data.toLongArray())
    override fun isTrue(v: Long): Boolean = v != 0L
    override fun asUsize(v: Long): Int = v.toInt()
}

public object IntDTypeU32 : IntDType<UInt> {
    override val dtype: DType = DType.U32
    override fun fromF64(v: Double): UInt = v.toLong().toUInt()
    override fun toF64(v: UInt): Double = v.toDouble()
    override fun toScalar(v: UInt): Scalar = Scalar.U32(v)
    override fun toCpuStorage(data: List<UInt>): CpuStorage = CpuStorage.U32(data.toUIntArray())
    override fun isTrue(v: UInt): Boolean = v != 0u
    override fun asUsize(v: UInt): Int = v.toInt()
}

public object IntDTypeU8 : IntDType<UByte> {
    override val dtype: DType = DType.U8
    override fun fromF64(v: Double): UByte = v.toInt().toUByte()
    override fun toF64(v: UByte): Double = v.toDouble()
    override fun toScalar(v: UByte): Scalar = Scalar.U8(v)
    override fun toCpuStorage(data: List<UByte>): CpuStorage = CpuStorage.U8(data.toUByteArray())
    override fun isTrue(v: UByte): Boolean = v != 0u.toUByte()
    override fun asUsize(v: UByte): Int = v.toInt()
}

public object IntDTypeI16 : IntDType<Short> {
    override val dtype: DType = DType.I16
    override fun fromF64(v: Double): Short = v.toInt().toShort()
    override fun toF64(v: Short): Double = v.toDouble()
    override fun toScalar(v: Short): Scalar = Scalar.I16(v)
    override fun toCpuStorage(data: List<Short>): CpuStorage = CpuStorage.I16(data.toShortArray())
    override fun isTrue(v: Short): Boolean = v != 0.toShort()
    override fun asUsize(v: Short): Int = v.toInt()
}

public object IntDTypeI32 : IntDType<Int> {
    override val dtype: DType = DType.I32
    override fun fromF64(v: Double): Int = v.toInt()
    override fun toF64(v: Int): Double = v.toDouble()
    override fun toScalar(v: Int): Scalar = Scalar.I32(v)
    override fun toCpuStorage(data: List<Int>): CpuStorage = CpuStorage.I32(data.toIntArray())
    override fun isTrue(v: Int): Boolean = v != 0
    override fun asUsize(v: Int): Int = v
}

public interface FloatDType<T> : WithDType<T>

public object FloatDTypeF32 : FloatDType<Float> {
    override val dtype: DType = DType.F32
    override fun fromF64(v: Double): Float = v.toFloat()
    override fun toF64(v: Float): Double = v.toDouble()
    override fun toScalar(v: Float): Scalar = Scalar.F32(v)
    override fun toCpuStorage(data: List<Float>): CpuStorage = CpuStorage.F32(data.toFloatArray())
}

public object FloatDTypeF64 : FloatDType<Double> {
    override val dtype: DType = DType.F64
    override fun fromF64(v: Double): Double = v
    override fun toF64(v: Double): Double = v
    override fun toScalar(v: Double): Scalar = Scalar.F64(v)
    override fun toCpuStorage(data: List<Double>): CpuStorage = CpuStorage.F64(data.toDoubleArray())
}

public object FloatDTypeF16 : FloatDType<Short> {
    override val dtype: DType = DType.F16
    override fun fromF64(v: Double): Short = Float16Utils.floatToF16Bits(v.toFloat())
    override fun toF64(v: Short): Double = Float16Utils.f16BitsToFloat(v).toDouble()
    override fun toScalar(v: Short): Scalar = Scalar.F16(v)
    override fun toCpuStorage(data: List<Short>): CpuStorage = CpuStorage.F16(data.toShortArray())
}

public object FloatDTypeBF16 : FloatDType<Short> {
    override val dtype: DType = DType.BF16
    override fun fromF64(v: Double): Short = Float16Utils.floatToBF16Bits(v.toFloat())
    override fun toF64(v: Short): Double = Float16Utils.bf16BitsToFloat(v).toDouble()
    override fun toScalar(v: Short): Scalar = Scalar.BF16(v)
    override fun toCpuStorage(data: List<Short>): CpuStorage = CpuStorage.BF16(data.toShortArray())
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
            val exponent = (127 - 14 - shift) shl 23
            val mantissa = (m and 0x03FF) shl 13
            return Float.fromBits(sign or exponent or mantissa)
        } else if (exp == 0x1F) {
            val exponent = 0xFF shl 23
            val mantissa = mant shl 13
            return Float.fromBits(sign or exponent or mantissa)
        } else {
            val exponent = (exp + (127 - 15)) shl 23
            val mantissa = mant shl 13
            return Float.fromBits(sign or exponent or mantissa)
        }
    }

    public fun floatToBF16Bits(value: Float): Short {
        val fbits = value.toRawBits()
        return (fbits ushr 16).toShort()
    }

    public fun bf16BitsToFloat(bits: Short): Float {
        val fbits = (bits.toInt() and 0xFFFF) shl 16
        return Float.fromBits(fbits)
    }
}
