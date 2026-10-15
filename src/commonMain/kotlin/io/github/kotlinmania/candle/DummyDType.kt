// port-lint: source dummy_dtype.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleResult.Err
import io.github.kotlinmania.candle.CandleResult.Ok

/**
 * Dummy data types for experimental/future float formats
 *
 * These are placeholder types for experimental floating-point formats
 * that are defined in the safetensors spec but not yet fully implemented.
 */

public class Formatter(public val out: StringBuilder = StringBuilder()) {
    public val sb: StringBuilder get() = out
    override fun toString(): String = out.toString()
}

internal fun write(f: Formatter, text: String): CandleResult<Unit> {
    f.out.append(text)
    return Ok(Unit)
}

/**
 * 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E2M3 {
    public fun toF64(): Double =
        throw IllegalStateException("F6E2M3 is a dummy type and cannot be converted")

    public fun toScalar(): Scalar =
        throw IllegalStateException("F6E2M3 is a dummy type and cannot be converted to scalar")

    public fun addAssign(other: F6E2M3): Unit =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun subAssign(other: F6E2M3): Unit =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun mulAssign(other: F6E2M3): Unit =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun divAssign(other: F6E2M3): Unit =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun remAssign(other: F6E2M3): Unit =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public operator fun plus(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public operator fun minus(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public operator fun times(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public operator fun div(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public operator fun rem(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun isZero(): Boolean =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations")

    public fun min(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun max(other: F6E2M3): F6E2M3 =
        throw IllegalStateException("F6E2M3 is a dummy type and does not support operations referencing $other")

    public fun fmt(f: Formatter) {
        write(f, "F6E2M3")
    }

    override fun toString(): String = "F6E2M3"

    public companion object {
        public val DTYPE: DType = DType.F8E4M3

        public fun fromF64(v: Double): F6E2M3 =
            throw IllegalStateException("F6E2M3 is a dummy type and cannot be constructed from $v")

        public fun cpuStorageRef(data: List<F6E2M3>): CpuStorageRef =
            throw IllegalStateException("F6E2M3 is a dummy type and does not support storage (${data.size})")

        public fun toCpuStorageOwned(data: List<F6E2M3>): CpuStorage =
            throw IllegalStateException("F6E2M3 is a dummy type and does not support storage (${data.size})")

        public fun cpuStorageData(s: CpuStorage): CandleResult<List<F6E2M3>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_data"))

        public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<F6E2M3>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_as_slice"))

        public fun zero(): F6E2M3 =
            throw IllegalStateException("F6E2M3 is a dummy type and does not support operations")

        public fun one(): F6E2M3 =
            throw IllegalStateException("F6E2M3 is a dummy type and does not support operations")

        public fun fromStrRadix(str: String, radix: UInt): CandleResult<F6E2M3> =
            throw IllegalStateException("F6E2M3 is a dummy type and does not support parsing ($str, $radix)")
    }
}

/**
 * 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E3M2 {
    public fun toF64(): Double =
        throw IllegalStateException("F6E3M2 is a dummy type and cannot be converted")

    public fun toScalar(): Scalar =
        throw IllegalStateException("F6E3M2 is a dummy type and cannot be converted to scalar")

    public fun addAssign(other: F6E3M2): Unit =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun subAssign(other: F6E3M2): Unit =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun mulAssign(other: F6E3M2): Unit =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun divAssign(other: F6E3M2): Unit =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun remAssign(other: F6E3M2): Unit =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public operator fun plus(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public operator fun minus(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public operator fun times(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public operator fun div(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public operator fun rem(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun isZero(): Boolean =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations")

    public fun min(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun max(other: F6E3M2): F6E3M2 =
        throw IllegalStateException("F6E3M2 is a dummy type and does not support operations referencing $other")

    public fun fmt(f: Formatter) {
        write(f, "F6E3M2")
    }

    override fun toString(): String = "F6E3M2"

    public companion object {
        public val DTYPE: DType = DType.F8E4M3

        public fun fromF64(v: Double): F6E3M2 =
            throw IllegalStateException("F6E3M2 is a dummy type and cannot be constructed from $v")

        public fun cpuStorageRef(data: List<F6E3M2>): CpuStorageRef =
            throw IllegalStateException("F6E3M2 is a dummy type and does not support storage (${data.size})")

        public fun toCpuStorageOwned(data: List<F6E3M2>): CpuStorage =
            throw IllegalStateException("F6E3M2 is a dummy type and does not support storage (${data.size})")

        public fun cpuStorageData(s: CpuStorage): CandleResult<List<F6E3M2>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_data"))

        public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<F6E3M2>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_as_slice"))

        public fun zero(): F6E3M2 =
            throw IllegalStateException("F6E3M2 is a dummy type and does not support operations")

        public fun one(): F6E3M2 =
            throw IllegalStateException("F6E3M2 is a dummy type and does not support operations")

        public fun fromStrRadix(str: String, radix: UInt): CandleResult<F6E3M2> =
            throw IllegalStateException("F6E3M2 is a dummy type and does not support parsing ($str, $radix)")
    }
}

/**
 * 4-bit float (MX4 format)
 * This is a dummy type.
 */
public class F4 {
    public fun toF64(): Double =
        throw IllegalStateException("F4 is a dummy type and cannot be converted")

    public fun toScalar(): Scalar =
        throw IllegalStateException("F4 is a dummy type and cannot be converted to scalar")

    public fun addAssign(other: F4): Unit =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun subAssign(other: F4): Unit =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun mulAssign(other: F4): Unit =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun divAssign(other: F4): Unit =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun remAssign(other: F4): Unit =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public operator fun plus(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public operator fun minus(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public operator fun times(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public operator fun div(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public operator fun rem(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun isZero(): Boolean =
        throw IllegalStateException("F4 is a dummy type and does not support operations")

    public fun min(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun max(other: F4): F4 =
        throw IllegalStateException("F4 is a dummy type and does not support operations referencing $other")

    public fun fmt(f: Formatter) {
        write(f, "F4")
    }

    override fun toString(): String = "F4"

    public companion object {
        public val DTYPE: DType = DType.F8E4M3

        public fun fromF64(v: Double): F4 =
            throw IllegalStateException("F4 is a dummy type and cannot be constructed from $v")

        public fun cpuStorageRef(data: List<F4>): CpuStorageRef =
            throw IllegalStateException("F4 is a dummy type and does not support storage (${data.size})")

        public fun toCpuStorageOwned(data: List<F4>): CpuStorage =
            throw IllegalStateException("F4 is a dummy type and does not support storage (${data.size})")

        public fun cpuStorageData(s: CpuStorage): CandleResult<List<F4>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_data"))

        public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<F4>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_as_slice"))

        public fun zero(): F4 =
            throw IllegalStateException("F4 is a dummy type and does not support operations")

        public fun one(): F4 =
            throw IllegalStateException("F4 is a dummy type and does not support operations")

        public fun fromStrRadix(str: String, radix: UInt): CandleResult<F4> =
            throw IllegalStateException("F4 is a dummy type and does not support parsing ($str, $radix)")
    }
}

/**
 * 8-bit float with 8 exponent bits and 0 mantissa bits
 * This is a dummy type.
 */
public class F8E8M0 {
    public fun toF64(): Double =
        throw IllegalStateException("F8E8M0 is a dummy type and cannot be converted")

    public fun toScalar(): Scalar =
        throw IllegalStateException("F8E8M0 is a dummy type and cannot be converted to scalar")

    public fun addAssign(other: F8E8M0): Unit =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun subAssign(other: F8E8M0): Unit =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun mulAssign(other: F8E8M0): Unit =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun divAssign(other: F8E8M0): Unit =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun remAssign(other: F8E8M0): Unit =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public operator fun plus(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public operator fun minus(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public operator fun times(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public operator fun div(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public operator fun rem(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun isZero(): Boolean =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations")

    public fun min(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun max(other: F8E8M0): F8E8M0 =
        throw IllegalStateException("F8E8M0 is a dummy type and does not support operations referencing $other")

    public fun fmt(f: Formatter) {
        write(f, "F8E8M0")
    }

    override fun toString(): String = "F8E8M0"

    public companion object {
        public val DTYPE: DType = DType.F8E4M3

        public fun fromF64(v: Double): F8E8M0 =
            throw IllegalStateException("F8E8M0 is a dummy type and cannot be constructed from $v")

        public fun cpuStorageRef(data: List<F8E8M0>): CpuStorageRef =
            throw IllegalStateException("F8E8M0 is a dummy type and does not support storage (${data.size})")

        public fun toCpuStorageOwned(data: List<F8E8M0>): CpuStorage =
            throw IllegalStateException("F8E8M0 is a dummy type and does not support storage (${data.size})")

        public fun cpuStorageData(s: CpuStorage): CandleResult<List<F8E8M0>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_data"))

        public fun cpuStorageAsSlice(s: CpuStorage): CandleResult<List<F8E8M0>> =
            Err(CandleException.UnsupportedDTypeForOp(s.dtype(), "cpu_storage_as_slice"))

        public fun zero(): F8E8M0 =
            throw IllegalStateException("F8E8M0 is a dummy type and does not support operations")

        public fun one(): F8E8M0 =
            throw IllegalStateException("F8E8M0 is a dummy type and does not support operations")

        public fun fromStrRadix(str: String, radix: UInt): CandleResult<F8E8M0> =
            throw IllegalStateException("F8E8M0 is a dummy type and does not support parsing ($str, $radix)")
    }
}

