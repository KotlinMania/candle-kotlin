// port-lint: source dummy_dtype.rs
package io.github.kotlinmania.candle.dummydtype

import io.github.kotlinmania.candle.cpu.kernels.VecOps
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cpubackend.CpuStorageRef
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.numtraits.Num
import io.github.kotlinmania.numtraits.One
import io.github.kotlinmania.numtraits.Zero

// Dummy data types for experimental/future float formats
//
// These are Note types for experimental floating-point formats
// that are defined in the safetensors spec but not yet fully implemented.

/**
 * 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public open class F6e2m3 : WithDType, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, Add, Sub, Mul, Div, Rem, Zero, One, Num, VecOps {
    public open fun toF64(): Double {
        error(("F6E2M3").toString() + " is a dummy type and cannot be converted")
    }

    public open fun toScalar(): Scalar {
        error(("F6E2M3").toString() + " is a dummy type and cannot be converted to scalar")
    }

    public open fun addAssign(other: F6e2m3) {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun subAssign(other: F6e2m3) {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun mulAssign(other: F6e2m3) {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun divAssign(other: F6e2m3) {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun remAssign(other: F6e2m3) {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun add(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun sub(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun mul(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun div(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun rem(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun isZero(): Boolean {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun min(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    public open fun max(other: F6e2m3): F6e2m3 {
        error(("F6E2M3").toString() + " is a dummy type and does not support operations")
    }

    // Display implementations
    public open fun fmt(f: StringBuilder) {
        f.append("F6E2M3")
    }

    public companion object {
        public val DTYPE: DType = DType.F6E2M3

        public fun fromF64(v: Double): F6e2m3 {
            error(("F6E2M3").toString() + " is a dummy type and cannot be constructed")
        }

        public fun cpuStorageRef(data: List<F6e2m3>): CpuStorageRef {
            error(("F6E2M3").toString() + " is a dummy type and does not support storage")
        }

        public fun toCpuStorageOwned(data: List<F6e2m3>): CpuStorage {
            error(("F6E2M3").toString() + " is a dummy type and does not support storage")
        }

        public fun cpuStorageData(s: CpuStorage): Result<List<F6e2m3>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F6E2M3, "cpu_storage_data").bt())

        public fun cpuStorageAsSlice(s: CpuStorage): Result<List<F6e2m3>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F6E2M3, "cpu_storage_as_slice").bt())

        public fun zero(): F6e2m3 {
            error(("F6E2M3").toString() + " is a dummy type and does not support operations")
        }

        public fun one(): F6e2m3 {
            error(("F6E2M3").toString() + " is a dummy type and does not support operations")
        }

        public fun fromStrRadix(str: String, radix: UInt): Result<F6e2m3> {
            error(("F6E2M3").toString() + " is a dummy type and does not support parsing")
        }
    }
}

/**
 * 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public open class F6e3m2 : WithDType, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, Add, Sub, Mul, Div, Rem, Zero, One, Num, VecOps {
    public open fun toF64(): Double {
        error(("F6E3M2").toString() + " is a dummy type and cannot be converted")
    }

    public open fun toScalar(): Scalar {
        error(("F6E3M2").toString() + " is a dummy type and cannot be converted to scalar")
    }

    public open fun addAssign(other: F6e3m2) {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun subAssign(other: F6e3m2) {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun mulAssign(other: F6e3m2) {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun divAssign(other: F6e3m2) {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun remAssign(other: F6e3m2) {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun add(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun sub(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun mul(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun div(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun rem(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun isZero(): Boolean {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun min(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun max(other: F6e3m2): F6e3m2 {
        error(("F6E3M2").toString() + " is a dummy type and does not support operations")
    }

    public open fun fmt(f: StringBuilder) {
        f.append("F6E3M2")
    }

    public companion object {
        public val DTYPE: DType = DType.F6E3M2

        public fun fromF64(v: Double): F6e3m2 {
            error(("F6E3M2").toString() + " is a dummy type and cannot be constructed")
        }

        public fun cpuStorageRef(data: List<F6e3m2>): CpuStorageRef {
            error(("F6E3M2").toString() + " is a dummy type and does not support storage")
        }

        public fun toCpuStorageOwned(data: List<F6e3m2>): CpuStorage {
            error(("F6E3M2").toString() + " is a dummy type and does not support storage")
        }

        public fun cpuStorageData(s: CpuStorage): Result<List<F6e3m2>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F6E3M2, "cpu_storage_data").bt())

        public fun cpuStorageAsSlice(s: CpuStorage): Result<List<F6e3m2>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F6E3M2, "cpu_storage_as_slice").bt())

        public fun zero(): F6e3m2 {
            error(("F6E3M2").toString() + " is a dummy type and does not support operations")
        }

        public fun one(): F6e3m2 {
            error(("F6E3M2").toString() + " is a dummy type and does not support operations")
        }

        public fun fromStrRadix(str: String, radix: UInt): Result<F6e3m2> {
            error(("F6E3M2").toString() + " is a dummy type and does not support parsing")
        }
    }
}

/**
 * 4-bit float (MX4 format)
 * This is a dummy type.
 */
public open class F4 : WithDType, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, Add, Sub, Mul, Div, Rem, Zero, One, Num, VecOps {
    public open fun toF64(): Double {
        error(("F4").toString() + " is a dummy type and cannot be converted")
    }

    public open fun toScalar(): Scalar {
        error(("F4").toString() + " is a dummy type and cannot be converted to scalar")
    }

    public open fun addAssign(other: F4) {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun subAssign(other: F4) {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun mulAssign(other: F4) {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun divAssign(other: F4) {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun remAssign(other: F4) {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun add(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun sub(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun mul(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun div(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun rem(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun isZero(): Boolean {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun min(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun max(other: F4): F4 {
        error(("F4").toString() + " is a dummy type and does not support operations")
    }

    public open fun fmt(f: StringBuilder) {
        f.append("F4")
    }

    public companion object {
        public val DTYPE: DType = DType.F4

        public fun fromF64(v: Double): F4 {
            error(("F4").toString() + " is a dummy type and cannot be constructed")
        }

        public fun cpuStorageRef(data: List<F4>): CpuStorageRef {
            error(("F4").toString() + " is a dummy type and does not support storage")
        }

        public fun toCpuStorageOwned(data: List<F4>): CpuStorage {
            error(("F4").toString() + " is a dummy type and does not support storage")
        }

        public fun cpuStorageData(s: CpuStorage): Result<List<F4>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F4, "cpu_storage_data").bt())

        public fun cpuStorageAsSlice(s: CpuStorage): Result<List<F4>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F4, "cpu_storage_as_slice").bt())

        public fun zero(): F4 {
            error(("F4").toString() + " is a dummy type and does not support operations")
        }

        public fun one(): F4 {
            error(("F4").toString() + " is a dummy type and does not support operations")
        }

        public fun fromStrRadix(str: String, radix: UInt): Result<F4> {
            error(("F4").toString() + " is a dummy type and does not support parsing")
        }
    }
}

/**
 * 8-bit float with 8 exponent bits and 0 mantissa bits
 * This is a dummy type.
 */
public open class F8e8m0 : WithDType, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign, Add, Sub, Mul, Div, Rem, Zero, One, Num, VecOps {
    public open fun toF64(): Double {
        error(("F8E8M0").toString() + " is a dummy type and cannot be converted")
    }

    public open fun toScalar(): Scalar {
        error(("F8E8M0").toString() + " is a dummy type and cannot be converted to scalar")
    }

    public open fun addAssign(other: F8e8m0) {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun subAssign(other: F8e8m0) {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun mulAssign(other: F8e8m0) {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun divAssign(other: F8e8m0) {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun remAssign(other: F8e8m0) {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun add(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun sub(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun mul(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun div(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun rem(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun isZero(): Boolean {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun min(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun max(other: F8e8m0): F8e8m0 {
        error(("F8E8M0").toString() + " is a dummy type and does not support operations")
    }

    public open fun fmt(f: StringBuilder) {
        f.append("F8E8M0")
    }

    public companion object {
        public val DTYPE: DType = DType.F8E8M0

        public fun fromF64(v: Double): F8e8m0 {
            error(("F8E8M0").toString() + " is a dummy type and cannot be constructed")
        }

        public fun cpuStorageRef(data: List<F8e8m0>): CpuStorageRef {
            error(("F8E8M0").toString() + " is a dummy type and does not support storage")
        }

        public fun toCpuStorageOwned(data: List<F8e8m0>): CpuStorage {
            error(("F8E8M0").toString() + " is a dummy type and does not support storage")
        }

        public fun cpuStorageData(s: CpuStorage): Result<List<F8e8m0>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F8E8M0, "cpu_storage_data").bt())

        public fun cpuStorageAsSlice(s: CpuStorage): Result<List<F8e8m0>> = Result.failure(Error.UnsupportedDTypeForOp(DType.F8E8M0, "cpu_storage_as_slice").bt())

        public fun zero(): F8e8m0 {
            error(("F8E8M0").toString() + " is a dummy type and does not support operations")
        }

        public fun one(): F8e8m0 {
            error(("F8E8M0").toString() + " is a dummy type and does not support operations")
        }

        public fun fromStrRadix(str: String, radix: UInt): Result<F8e8m0> {
            error(("F8E8M0").toString() + " is a dummy type and does not support parsing")
        }
    }
}
