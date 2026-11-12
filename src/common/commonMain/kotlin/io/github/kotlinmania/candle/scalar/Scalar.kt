// port-lint: source scalar.rs
package io.github.kotlinmania.candle.scalar

import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

// TensorScalar Enum and Trait

public sealed class Scalar : From<T> {
    public data class U8(public var value: UByte) : Scalar()
    public data class U32(public var value: UInt) : Scalar()
    public data class I16(public var value: Short) : Scalar()
    public data class I32(public var value: Int) : Scalar()
    public data class I64(public var value: Long) : Scalar()
    public data class Bf16(public var value: Bf16) : Scalar()
    public data class F16(public var value: F16) : Scalar()
    public data class F32(public var value: Float) : Scalar()
    public data class F64(public var value: Double) : Scalar()
    public data class F8e4m3(public var value: F8e4m3) : Scalar()

    public fun __rustEq(other: Scalar): Boolean = this == other

    public open fun dtype(): DType =
        when (this) {
            is Scalar.U8 -> DType.U8
            is Scalar.U32 -> DType.U32
            is Scalar.I16 -> DType.I16
            is Scalar.I32 -> DType.I32
            is Scalar.I64 -> DType.I64
            is Scalar.Bf16 -> DType.BF16
            is Scalar.F16 -> DType.F16
            is Scalar.F32 -> DType.F32
            is Scalar.F64 -> DType.F64
            is Scalar.F8e4m3 -> DType.F8E4M3
        }

    public open fun toF64(): Double =
        when (this) {
            is Scalar.U8 -> {
                val (v) = this
                v.toDouble()
            }
            is Scalar.U32 -> {
                val (v1) = this
                v1.toDouble()
            }
            is Scalar.I16 -> {
                val (v2) = this
                v2.toDouble()
            }
            is Scalar.I32 -> {
                val (v3) = this
                v3.toDouble()
            }
            is Scalar.I64 -> {
                val (v4) = this
                v4.toDouble()
            }
            is Scalar.Bf16 -> {
                val (v5) = this
                v5.toF64()
            }
            is Scalar.F16 -> {
                val (v6) = this
                v6.toF64()
            }
            is Scalar.F32 -> {
                val (v7) = this
                v7.toDouble()
            }
            is Scalar.F64 -> {
                val (v8) = this
                v8
            }
            is Scalar.F8e4m3 -> {
                val (v9) = this
                v9.toF64()
            }
        }

    public companion object {
        public fun <T : WithDType> from(value: T): Scalar = value.toScalar()

        public fun zero(dtype: DType): Scalar =
            when (dtype) {
                DType.u8 -> Scalar.u8(0)
                DType.u32 -> Scalar.u32(0)
                DType.i16 -> Scalar.i16(0)
                DType.i32 -> Scalar.i32(0)
                DType.i64 -> Scalar.i64(0)
                DType.bf16 -> Scalar.bf16(ZERO)
                DType.f16 -> Scalar.f16(ZERO)
                DType.f32 -> Scalar.f32(0.0)
                DType.f64 -> Scalar.f64(0.0)
                DType.f8e4m3 -> Scalar.f8e4m3(ZERO)
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    error("Cannot create zero scalar for dummy type $dtype")
                }
            }

        public fun one(dtype: DType): Scalar =
            when (dtype) {
                DType.u8 -> Scalar.u8(1)
                DType.u32 -> Scalar.u32(1)
                DType.i16 -> Scalar.i16(1)
                DType.i32 -> Scalar.i32(1)
                DType.i64 -> Scalar.i64(1)
                DType.bf16 -> Scalar.bf16(ONE)
                DType.f16 -> Scalar.f16(ONE)
                DType.f32 -> Scalar.f32(1.0)
                DType.f64 -> Scalar.f64(1.0)
                DType.f8e4m3 -> Scalar.f8e4m3(ONE)
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    error("Cannot create one scalar for dummy type $dtype")
                }
            }
    }
}

public sealed class TensorScalar {
    public data class Tensor(public var value: Tensor) : TensorScalar()
    public data class Scalar(public var value: Tensor) : TensorScalar()
}

public interface TensorOrScalar {
    public fun toTensorScalar(): Result<TensorScalar>
}

public fun Tensor.toTensorScalar(): Result<TensorScalar> = Result.success(TensorScalar.Tensor(this.clone()))

public fun <T : WithDType> T.toTensorScalar(): Result<TensorScalar> {
    val scalar = Tensor.new(this, Device.Cpu).getOrElse { return Result.failure(it) }
    return Result.success(TensorScalar.Scalar(scalar))
}
