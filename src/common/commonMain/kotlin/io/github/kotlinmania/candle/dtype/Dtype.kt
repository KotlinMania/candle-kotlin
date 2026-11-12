// port-lint: source dtype.rs
package io.github.kotlinmania.candle.dtype

import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.F64
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Scalar
import io.github.kotlinmania.candle.cpu.kernels.VecOps
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.CpuStorageRef
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.numtraits.Bounded
import io.github.kotlinmania.numtraits.NumAssign

// Types for elements that can be stored and manipulated using tensors.

public typealias Err = DTypeParseError

/**
 * The different types of elements allowed in tensors.
 */
public sealed class DType : FromStr {
    // Unsigned 8 bits integer.
    public data object U8 : DType()
    // Unsigned 32 bits integer.
    public data object U32 : DType()
    // Signed 16 bits integer.
    public data object I16 : DType()
    // Signed 32 bits integer.
    public data object I32 : DType()
    // Signed 64 bits integer.
    public data object I64 : DType()
    // Brain floating-point using half precision (16 bits).
    public data object Bf16 : DType()
    // Floating-point using half precision (16 bits).
    public data object F16 : DType()
    // Floating-point using single precision (32 bits).
    public data object F32 : DType()
    // Floating-point using double precision (64 bits).
    public data object F64 : DType()
    // 8-bit floating point with 4-bit exponent and 3-bit mantissa.
    public data object F8e4m3 : DType()
    /**
     * 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format)
     */
    public data object F6e2m3 : DType()
    /**
     * 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format)
     */
    public data object F6e3m2 : DType()
    /**
     * 4-bit float (MX4 format)
     */
    public data object F4 : DType()
    /**
     * 8-bit float with 8 exponent bits and 0 mantissa bits
     */
    public data object F8e8m0 : DType()

    public fun __rustEq(other: DType): Boolean = this == other

    /**
     * String representation for dtypes.
     */
    public open fun asStr(): String =
        when (this) {
            DType.u8 -> "u8"
            DType.u32 -> "u32"
            DType.i16 -> "i16"
            DType.i32 -> "i32"
            DType.i64 -> "i64"
            DType.bf16 -> "bf16"
            DType.f16 -> "f16"
            DType.f32 -> "f32"
            DType.f64 -> "f64"
            DType.f8e4m3 -> "f8e4m3"
            DType.f6e2m3 -> "f6e2m3"
            DType.f6e3m2 -> "f6e3m2"
            DType.f4 -> "f4"
            DType.f8e8m0 -> "f8e8m0"
        }

    /**
     * The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
     */
    public open fun sizeInBytes(): Int =
        when (this) {
            DType.u8 -> 1
            DType.u32 -> 4
            DType.i16 -> 2
            DType.i32 -> 4
            DType.i64 -> 8
            DType.bf16 -> 2
            DType.f16 -> 2
            DType.f32 -> 4
            DType.f64 -> 8
            DType.f8e4m3 -> 1
            DType.f6e2m3 -> 0
            // 6 bits
            DType.f6e3m2 -> 0
            // 6 bits
            DType.f4 -> 0
            // 4 bits
            DType.f8e8m0 -> 1
        }

    public open fun isInt(): Boolean =
        when (this) {
            DType.u8, DType.u32, DType.i16, DType.i32, DType.i64 -> true
            DType.bf16, DType.f16, DType.f32, DType.f64, DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> false
        }

    public open fun isFloat(): Boolean =
        when (this) {
            DType.u8, DType.u32, DType.i16, DType.i32, DType.i64 -> false
            DType.bf16, DType.f16, DType.f32, DType.f64, DType.f8e4m3, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> true
        }

    public companion object {
        public fun fromStr(s: String): Result<DType> =
            when (s) {
                "u8" -> Result.success(DType.U8)
                "u32" -> Result.success(DType.U32)
                "i16" -> Result.success(DType.I16)
                "i32" -> Result.success(DType.I32)
                "i64" -> Result.success(DType.I64)
                "bf16" -> Result.success(DType.BF16)
                "f16" -> Result.success(DType.F16)
                "f32" -> Result.success(DType.F32)
                "f64" -> Result.success(DType.F64)
                "f8e4m3" -> Result.success(DType.F8E4M3)
                "f6e2m3" -> Result.success(DType.F6E2M3)
                "f6e3m2" -> Result.success(DType.F6E3M2)
                "f4" -> Result.success(DType.F4)
                "f8e8m0" -> Result.success(DType.F8E8M0)
                else -> Err(DTypeParseError(s.toString()))
            }
    }
}

public open class DTypeParseError(
    public var value: String,
) : Error {
    public var field0: String
        get() = value
        set(v) {
            value = v
        }

    public open fun fmt(f: StringBuilder) {
        f.append("cannot parse '${this.value}' as a dtype")
    }
}

public interface WithDType<T> : NumAssign, Any, VecOps {
    public val DTYPE: DType

    public fun fromF64(v: Double): T

    public fun toF64(): Double

    public fun toScalar(): Scalar

    public fun cpuStorageRef(data: List<T>): CpuStorageRef

    public fun toCpuStorageOwned(data: List<T>): CpuStorage

    public fun toCpuStorage(data: List<T>): CpuStorage = data.toVec().toCpuStorageOwned()

    public fun cpuStorageAsSlice(s: CpuStorage): Result<List<T>>

    public fun cpuStorageData(s: CpuStorage): Result<List<T>>
}

public val DTYPE: DType = DType.F16

public fun withDTypeF16(v: Double): F16 = fromF64(v)

public fun F16.toF64(): Double = toF64(this)

public fun F16.toScalar(): Scalar = F16(this)

public fun withDTypeF16(data: List<F16>): CpuStorageRef = CpuStorageRef.f16(data)

public fun withDTypeF16(s: CpuStorage): Result<List<F16>> =
    when (s) {
        is CpuStorage.F16 -> {
            val (data) = s
            Result.success(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F16, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public val DTYPE1: DType = DType.BF16

public fun withDTypeBf16(v: Double): Bf16 = fromF64(v)

public fun Bf16.toF64(): Double = toF64(this)

public fun Bf16.toScalar(): Scalar = BF16(this)

public fun withDTypeBf16(data: List<Bf16>): CpuStorageRef = CpuStorageRef.bf16(data)

public fun withDTypeBf16(s: CpuStorage): Result<List<Bf16>> =
    when (s) {
        is CpuStorage.Bf16 -> {
            val (data) = s
            Result.success(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.BF16, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public val DTYPE2: DType = DType.F64

public fun withDTypeDouble(v: Double): Double = { v1: Double -> v1(v1) }

public fun Double.toF64(): Double = { v: Double -> v(this) }

public fun Double.toScalar(): Scalar = F64(this)

public fun withDTypeDouble(data: List<Double>): CpuStorageRef = CpuStorageRef.f64(data)

public fun withDTypeDouble(s: CpuStorage): Result<List<Double>> =
    when (s) {
        is CpuStorage.F64 -> {
            val (data) = s
            Result.success(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F64, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public val DTYPE3: DType = DType.F8E4M3

public fun withDTypeF8e4m3(v: Double): F8e4m3 = fromF64(v)

public fun F8e4m3.toF64(): Double = { v: F8e4m3 -> v.toF64()(this) }

public fun F8e4m3.toScalar(): Scalar = F8E4M3(this)

public fun withDTypeF8e4m3(data: List<F8e4m3>): CpuStorageRef = CpuStorageRef.f8e4m3(data)

public fun withDTypeF8e4m3(s: CpuStorage): Result<List<F8e4m3>> =
    when (s) {
        is CpuStorage.F8e4m3 -> {
            val (data) = s
            Result.success(data)
        }
        else -> Err(Error.UnexpectedDType(expected = DType.F8E4M3, got = s.dtype(), msg = "unexpected dtype").bt())
    }

public interface IntDType : WithDType, Bounded {
    public fun isTrue(): Boolean

    public fun asUsize(): Int
}

public fun Long.isTrue(): Boolean = this != 0

public fun Long.asUsize(): Int = this.toInt()

public fun UInt.isTrue(): Boolean = this != 0

public fun UInt.asUsize(): Int = this.toInt()

public fun UByte.isTrue(): Boolean = this != 0

public fun UByte.asUsize(): Int = this.toInt()

public fun Short.isTrue(): Boolean = this != 0

public fun Short.asUsize(): Int = this.toInt()

public fun Int.isTrue(): Boolean = this != 0

public fun Int.asUsize(): Int = this.toInt()

public interface FloatDType : WithDType
