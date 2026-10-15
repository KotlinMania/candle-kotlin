// port-lint: source scalar.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleResult.Ok

/** TensorScalar Enum and Trait */
public sealed class Scalar {
    public data class U8(val v: UByte) : Scalar() { val value: UByte get() = v }
    public data class U32(val v: UInt) : Scalar() { val value: UInt get() = v }
    public data class I16(val v: Short) : Scalar() { val value: Short get() = v }
    public data class I32(val v: Int) : Scalar() { val value: Int get() = v }
    public data class I64(val v: Long) : Scalar() { val value: Long get() = v }
    public data class BF16(val v: Short) : Scalar() { val value: Short get() = v }
    public data class F16(val v: Short) : Scalar() { val value: Short get() = v }
    public data class F32(val v: Float) : Scalar() { val value: Float get() = v }
    public data class F64(val v: Double) : Scalar() { val value: Double get() = v }
    public data class F8E4M3(val v: Byte) : Scalar() { val value: Byte get() = v }
    private val self: Scalar get() = this

    public companion object {
        public fun <T : Any> from(value: T): Scalar {
            return value.toScalar()
        }

        public fun zero(dtype: DType): Scalar {
            return when (dtype) {
                DType.U8 -> Scalar.U8(0u)
                DType.U32 -> Scalar.U32(0u)
                DType.I16 -> Scalar.I16(0)
                DType.I32 -> Scalar.I32(0)
                DType.I64 -> Scalar.I64(0L)
                DType.BF16 -> Scalar.BF16(bf16.ZERO)
                DType.F16 -> Scalar.F16(f16.ZERO)
                DType.F32 -> Scalar.F32(0.0f)
                DType.F64 -> Scalar.F64(0.0)
                DType.F8E4M3 -> Scalar.F8E4M3(f8e4m3.ZERO)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    panic("Cannot create zero scalar for dummy type {dtype:?}")
                }
            }
        }

        public fun one(dtype: DType): Scalar {
            return when (dtype) {
                DType.U8 -> Scalar.U8(1u)
                DType.U32 -> Scalar.U32(1u)
                DType.I16 -> Scalar.I16(1)
                DType.I32 -> Scalar.I32(1)
                DType.I64 -> Scalar.I64(1L)
                DType.BF16 -> Scalar.BF16(bf16.ONE)
                DType.F16 -> Scalar.F16(f16.ONE)
                DType.F32 -> Scalar.F32(1.0f)
                DType.F64 -> Scalar.F64(1.0)
                DType.F8E4M3 -> Scalar.F8E4M3(f8e4m3.ONE)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    panic("Cannot create one scalar for dummy type {dtype:?}")
                }
            }
        }
    }

    public fun dtype(): DType {
        return when (self) {
            is Scalar.U8 -> DType.U8
            is Scalar.U32 -> DType.U32
            is Scalar.I16 -> DType.I16
            is Scalar.I32 -> DType.I32
            is Scalar.I64 -> DType.I64
            is Scalar.BF16 -> DType.BF16
            is Scalar.F16 -> DType.F16
            is Scalar.F32 -> DType.F32
            is Scalar.F64 -> DType.F64
            is Scalar.F8E4M3 -> DType.F8E4M3
        }
    }

    public fun toF64(): Double {
        return when (val v = self) {
            is Scalar.U8 -> v.toDouble()
            is Scalar.U32 -> v.toDouble()
            is Scalar.I16 -> v.toDouble()
            is Scalar.I32 -> v.toDouble()
            is Scalar.I64 -> v.toDouble()
            is Scalar.BF16 -> v.toF64()
            is Scalar.F16 -> v.toF64()
            is Scalar.F32 -> v.toDouble()
            is Scalar.F64 -> v.v
            is Scalar.F8E4M3 -> v.toF64()
        }
    }
}

public sealed class TensorScalar {
    public data class Tensor(val tensor: io.github.kotlinmania.candle.Tensor) : TensorScalar()
    public data class Scalar(val tensor: io.github.kotlinmania.candle.Tensor) : TensorScalar()
}

public interface TensorOrScalar {
    public fun toTensorScalar(): CandleResult<TensorScalar>
}

public class TensorAsTensorOrScalar(public val self: Tensor) : TensorOrScalar {
    override fun toTensorScalar(): CandleResult<TensorScalar> {
        return Ok(TensorScalar.Tensor(self.clone()))
    }
}

public class ScalarAsTensorOrScalar<T : Any>(public val self: T) : TensorOrScalar {
    override fun toTensorScalar(): CandleResult<TensorScalar> {
        val scalar = Tensor.new(self, Device.Cpu).getOrThrow()
        return Ok(TensorScalar.Scalar(scalar))
    }
}
