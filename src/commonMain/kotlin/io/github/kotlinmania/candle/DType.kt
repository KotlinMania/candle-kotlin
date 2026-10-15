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
                else -> Err(DTypeParseError(s.toString()))
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
            Self.U8 -> "u8"
            Self.U32 -> "u32"
            Self.I16 -> "i16"
            Self.I32 -> "i32"
            Self.I64 -> "i64"
            Self.BF16 -> "bf16"
            Self.F16 -> "f16"
            Self.F32 -> "f32"
            Self.F64 -> "f64"
            Self.F8E4M3 -> "f8e4m3"
            Self.F6E2M3 -> "f6e2m3"
            Self.F6E3M2 -> "f6e3m2"
            Self.F4 -> "f4"
            Self.F8E8M0 -> "f8e8m0"
        }
    }

    /**
     * The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
     */
    public fun sizeInBytes(): Int {
        return when (self) {
            Self.U8 -> 1
            Self.U32 -> 4
            Self.I16 -> 2
            Self.I32 -> 4
            Self.I64 -> 8
            Self.BF16 -> 2
            Self.F16 -> 2
            Self.F32 -> 4
            Self.F64 -> 8
            Self.F8E4M3 -> 1
            Self.F6E2M3 -> 0 // 6 bits
            Self.F6E3M2 -> 0 // 6 bits
            Self.F4 -> 0     // 4 bits
            Self.F8E8M0 -> 1
        }
    }

    public fun isInt(): Boolean =
        when (self) {
            Self.U8, Self.U32, Self.I16, Self.I32, Self.I64 -> true
            Self.BF16,
            Self.F16,
            Self.F32,
            Self.F64,
            Self.F8E4M3,
            Self.F6E2M3,
            Self.F6E3M2,
            Self.F4,
            Self.F8E8M0
            -> false
        }

    public fun isFloat(): Boolean =
        when (self) {
            Self.U8, Self.U32, Self.I16, Self.I32, Self.I64 -> false
            Self.BF16,
            Self.F16,
            Self.F32,
            Self.F64,
            Self.F8E4M3,
            Self.F6E2M3,
            Self.F6E3M2,
            Self.F4,
            Self.F8E8M0
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
    public fun toF64(self: T): Double
    public fun toScalar(self: T): Scalar
    public fun cpuStorageRef(data: List<T>): CpuStorageRef
    public fun toCpuStorageOwned(data: List<T>): CpuStorage

    public fun toCpuStorage(data: List<T>): CpuStorage {
        return self.toCpuStorageOwned(data.toVec())
    }

    public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<T>>
    public fun cpuStorageData(s: CpuStorage): CandleResult<List<T>>
}

public interface IntDType<T> {
    public fun isTrue(self: T): Boolean
    public fun asUsize(self: T): Int
}

public object IntDTypeI64 : IntDType<Int> {
    override fun isTrue(self: Int): Boolean {
        return self != 0
    }
    override fun asUsize(self: Int): Int {
        return self
    }
}

public object IntDTypeU32 : IntDType<Int> {
    override fun isTrue(self: Int): Boolean {
        return self != 0
    }
    override fun asUsize(self: Int): Int {
        return self
    }
}

public object IntDTypeU8 : IntDType<Int> {
    override fun isTrue(self: Int): Boolean {
        return self != 0
    }
    override fun asUsize(self: Int): Int {
        return self
    }
}

public object IntDTypeI16 : IntDType<Int> {
    override fun isTrue(self: Int): Boolean {
        return self != 0
    }
    override fun asUsize(self: Int): Int {
        return self
    }
}

public object IntDTypeI32 : IntDType<Int> {
    override fun isTrue(self: Int): Boolean {
        return self != 0
    }
    override fun asUsize(self: Int): Int {
        return self
    }
}

public interface FloatDType<T> : WithDType<T>
