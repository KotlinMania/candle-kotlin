// port-lint: source dtype.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result

/**
 * Types for elements that can be stored and manipulated using tensors.
 */

public class DTypeParseError(public val raw: String) : Exception("cannot parse '$raw' as a dtype") {
    public fun fmt(f: StringBuilder): Result<Unit> {
        f.append("cannot parse '$raw' as a dtype")
        return Ok(Unit)
    }
}

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
    public fun asStr(): String =
        when (this) {
            U8 -> "u8"
            U32 -> "u32"
            I16 -> "i16"
            I32 -> "i32"
            I64 -> "i64"
            BF16 -> "bf16"
            F16 -> "f16"
            F32 -> "f32"
            F64 -> "f64"
            F8E4M3 -> "f8e4m3"
            F6E2M3 -> "f6e2m3"
            F6E3M2 -> "f6e3m2"
            F4 -> "f4"
            F8E8M0 -> "f8e8m0"
        }

    /**
     * The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
     */
    public fun sizeInBytes(): Int =
        when (this) {
            U8 -> 1
            U32 -> 4
            I16 -> 2
            I32 -> 4
            I64 -> 8
            BF16 -> 2
            F16 -> 2
            F32 -> 4
            F64 -> 8
            F8E4M3 -> 1
            F6E2M3 -> 0
            F6E3M2 -> 0
            F4 -> 0
            F8E8M0 -> 1
        }

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

        public fun fromStr(s: String): Result<DType> =
            when (s) {
                "u8" -> Ok(U8)
                "u32" -> Ok(U32)
                "i16" -> Ok(I16)
                "i32" -> Ok(I32)
                "i64" -> Ok(I64)
                "bf16" -> Ok(BF16)
                "f16" -> Ok(F16)
                "f32" -> Ok(F32)
                "f64" -> Ok(F64)
                "f8e4m3" -> Ok(F8E4M3)
                "f6e2m3" -> Ok(F6E2M3)
                "f6e3m2" -> Ok(F6E3M2)
                "f4" -> Ok(F4)
                "f8e8m0" -> Ok(F8E8M0)
                else -> Result.failure(DTypeParseError(s.toString()))
            }

        public val fromString: (String) -> DType = { s ->
            fromStr(s.lowercase()).getOrThrow()
        }
    }
}

public interface WithDType<T> {
    public val dtype: DType
    public fun fromF64(v: Double): T
    public fun toF64(v: T): Double
    public fun toScalar(v: T): Scalar
    public fun cpuStorageRef(data: List<T>): CpuStorageRef
    public fun toCpuStorageOwned(data: List<T>): CpuStorage

    public fun toCpuStorage(data: List<T>): CpuStorage =
        toCpuStorageOwned(data.toList())

    public fun cpuStorageAsSlice(s: CpuStorage): Result<List<T>>
    public fun cpuStorageData(s: CpuStorage): Result<List<T>>
}

public interface IntDType<T> {
    public fun isTrue(v: T): Boolean
    public fun asUsize(v: T): Int
}

public object IntDTypeI64 : IntDType<Long> {
    override fun isTrue(v: Long): Boolean = v != 0L
    override fun asUsize(v: Long): Int = v.toInt()
}

public object IntDTypeU32 : IntDType<UInt> {
    override fun isTrue(v: UInt): Boolean = v != 0u
    override fun asUsize(v: UInt): Int = v.toInt()
}

public object IntDTypeU8 : IntDType<UByte> {
    override fun isTrue(v: UByte): Boolean = v != 0u.toUByte()
    override fun asUsize(v: UByte): Int = v.toInt()
}

public object IntDTypeI16 : IntDType<Short> {
    override fun isTrue(v: Short): Boolean = v != 0.toShort()
    override fun asUsize(v: Short): Int = v.toInt()
}

public object IntDTypeI32 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean = v != 0
    override fun asUsize(v: Int): Int = v
}

public interface FloatDType<T> : WithDType<T>

public object Float16Utils {
    public val floatToF16Bits: (Float) -> Short = { value ->
        val fbits = value.toRawBits()
        val sign = (fbits ushr 16) and 0x8000
        val exp = ((fbits ushr 23) and 0xFF) - (127 - 15)
        val mant = fbits and 0x007FFFFF

        if (exp <= 0) {
            if (exp < -10) {
                sign.toShort()
            } else {
                val m = (mant or 0x00800000) ushr (1 - exp)
                (sign or (m ushr 13)).toShort()
            }
        } else if (exp == 0xFF - (127 - 15)) {
            if (mant == 0) {
                (sign or 0x7C00).toShort()
            } else {
                (sign or 0x7C00 or (mant ushr 13).coerceAtLeast(1)).toShort()
            }
        } else {
            if (exp > 30) {
                (sign or 0x7C00).toShort()
            } else {
                (sign or (exp shl 10) or (mant ushr 13)).toShort()
            }
        }
    }

    public val f16BitsToFloat: (Short) -> Float = { bits ->
        val ibits = bits.toInt() and 0xFFFF
        val sign = (ibits and 0x8000) shl 16
        val exp = (ibits and 0x7C00) ushr 10
        val mant = ibits and 0x03FF

        if (exp == 0) {
            if (mant == 0) {
                Float.fromBits(sign)
            } else {
                var m = mant
                var shift = 0
                while ((m and 0x0400) == 0) {
                    m = m shl 1
                    shift++
                }
                val exponent = (127 - 14 - shift) shl 23
                val mantissa = (m and 0x03FF) shl 13
                Float.fromBits(sign or exponent or mantissa)
            }
        } else if (exp == 0x1F) {
            val exponent = 0xFF shl 23
            val mantissa = mant shl 13
            Float.fromBits(sign or exponent or mantissa)
        } else {
            val exponent = (exp + (127 - 15)) shl 23
            val mantissa = mant shl 13
            Float.fromBits(sign or exponent or mantissa)
        }
    }

    public val floatToBF16Bits: (Float) -> Short = { value ->
        val fbits = value.toRawBits()
        (fbits ushr 16).toShort()
    }

    public val bf16BitsToFloat: (Short) -> Float = { bits ->
        val fbits = (bits.toInt() and 0xFFFF) shl 16
        Float.fromBits(fbits)
    }
}
