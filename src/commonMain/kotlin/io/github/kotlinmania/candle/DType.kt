// port-lint: source dtype.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleResult.Err
import io.github.kotlinmania.candle.CandleResult.Ok

/**
 * Types for elements that can be stored and manipulated using tensors.
 */

/**
 * The different types of elements allowed in tensors.
 */
public enum class DType(
    public val typeName: String,
    public val sizeInBytes: Int,
) {
    // Unsigned 8 bits integer.
    U8("u8", 1),
    // Unsigned 32 bits integer.
    U32("u32", 4),
    // Signed 16 bits integer.
    I16("i16", 2),
    // Signed 32 bits integer.
    I32("i32", 4),
    // Signed 64 bits integer.
    I64("i64", 8),
    // Brain floating-point using half precision (16 bits).
    BF16("bf16", 2),
    // Floating-point using half precision (16 bits).
    F16("f16", 2),
    // Floating-point using single precision (32 bits).
    F32("f32", 4),
    // Floating-point using double precision (64 bits).
    F64("f64", 8),
    // 8-bit floating point with 4-bit exponent and 3-bit mantissa.
    F8E4M3("f8e4m3", 1),
    /** 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format) */
    F6E2M3("f6e2m3", 0),
    /** 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format) */
    F6E3M2("f6e3m2", 0),
    /** 4-bit float (MX4 format) */
    F4("f4", 0),
    /** 8-bit float with 8 exponent bits and 0 mantissa bits */
    F8E8M0("f8e8m0", 1),
    ;

    private val self: DType get() = this

    public companion object {
        public val Self: Companion get() = this
        public interface Err

        public fun fromStr(s: String): CandleResult<DType> {
            return when (s) {
                "u8" -> Ok(Self.U8)
                "u32" -> Ok(Self.U32)
                "i16" -> Ok(Self.I16)
                "i32" -> Ok(Self.I32)
                "i64" -> Ok(Self.I64)
                "bf16" -> Ok(Self.BF16)
                "f16" -> Ok(Self.F16)
                "f32" -> Ok(Self.F32)
                "f64" -> Ok(Self.F64)
                "f8e4m3" -> Ok(Self.F8E4M3)
                "f6e2m3" -> Ok(Self.F6E2M3)
                "f6e3m2" -> Ok(Self.F6E3M2)
                "f4" -> Ok(Self.F4)
                "f8e8m0" -> Ok(Self.F8E8M0)
                else -> Err(DTypeParseError(s))
            }
        }

        public val fromString: (String) -> DType = { s ->
            fromStr(s.lowercase()).getOrThrow()
        }
    }

    /**
     * String representation for dtypes.
     */
    public fun asStr(): String {
        return when (self) {
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
    }

    /**
     * The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
     */
    public fun sizeInBytes(): Int {
        return when (self) {
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
            F6E2M3 -> 0 // 6 bits
            F6E3M2 -> 0 // 6 bits
            F4 -> 0     // 4 bits
            F8E8M0 -> 1
        }
    }

    public fun isInt(): Boolean =
        when (self) {
            U8, U32, I16, I32, I64 -> true
            BF16,
            F16,
            F32,
            F64,
            F8E4M3,
            F6E2M3,
            F6E3M2,
            F4,
            F8E8M0
            -> false
        }

    public fun isFloat(): Boolean =
        when (self) {
            U8, U32, I16, I32, I64 -> false
            BF16,
            F16,
            F32,
            F64,
            F8E4M3,
            F6E2M3,
            F6E3M2,
            F4,
            F8E8M0
            -> true
        }
}

public class DTypeParseError(public val raw: String) : Exception("cannot parse '$raw' as a dtype") {
    private val self: String get() = raw

    public fun fmt(f: Formatter): CandleResult<Unit> {
        return write(f, "cannot parse '{}' as a dtype", self)
    }
}

public interface WithDType<T> {
    public val dtype: DType
    public fun fromF64(v: Double): T
    public fun toF64(v: T): Double
    public fun toScalar(v: T): Scalar
    public fun cpuStorageRef(data: List<T>): CpuStorageRef
    public fun toCpuStorageOwned(data: List<T>): CpuStorage

    public fun toCpuStorage(data: List<T>): CpuStorage {
        return self.toCpuStorageOwned(data.toVec())
    }

    public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<T>>
    public fun cpuStorageData(s: CpuStorage): CandleResult<List<T>>
}

public interface IntDType<T> {
    public fun isTrue(v: T): Boolean
    public fun asUsize(v: T): Int
}

public object IntDTypeI64 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean {
        return v != 0
    }
    override fun asUsize(v: Int): Int {
        return v
    }
}

public object IntDTypeU32 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean {
        return v != 0
    }
    override fun asUsize(v: Int): Int {
        return v
    }
}

public object IntDTypeU8 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean {
        return v != 0
    }
    override fun asUsize(v: Int): Int {
        return v
    }
}

public object IntDTypeI16 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean {
        return v != 0
    }
    override fun asUsize(v: Int): Int {
        return v
    }
}

public object IntDTypeI32 : IntDType<Int> {
    override fun isTrue(v: Int): Boolean {
        return v != 0
    }
    override fun asUsize(v: Int): Int {
        return v
    }
}

public interface FloatDType<T> : WithDType<T>

public typealias Float16 = io.github.kotlinmania.half.F16
public typealias BFloat16 = io.github.kotlinmania.half.BF16
public typealias F16 = io.github.kotlinmania.half.F16
public typealias BF16 = io.github.kotlinmania.half.BF16
public typealias F8E4M3 = io.github.kotlinmania.float8.F8E4M3
public typealias F8E5M2 = io.github.kotlinmania.float8.F8E5M2

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

