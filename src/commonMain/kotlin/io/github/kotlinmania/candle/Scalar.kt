// port-lint: source scalar.rs
package io.github.kotlinmania.candle

/**
 * TensorScalar Enum and Trait
 */
public sealed class Scalar {
    private val self: Scalar get() = this

    public data class U8(
        val v: UByte,
    ) : Scalar() {
        val value: UByte get() = v
    }

    public data class U32(
        val v: UInt,
    ) : Scalar() {
        val value: UInt get() = v
    }

    public data class I16(
        val v: Short,
    ) : Scalar() {
        val value: Short get() = v
    }

    public data class I32(
        val v: Int,
    ) : Scalar() {
        val value: Int get() = v
    }

    public data class I64(
        val v: Long,
    ) : Scalar() {
        val value: Long get() = v
    }

    public data class BF16(
        val v: Short,
    ) : Scalar() {
        val value: Short get() = v
    }

    public data class F16(
        val v: Short,
    ) : Scalar() {
        val value: Short get() = v
    }

    public data class F32(
        val v: Float,
    ) : Scalar() {
        val value: Float get() = v
    }

    public data class F64(
        val v: Double,
    ) : Scalar() {
        val value: Double get() = v
    }

    public data class F8E4M3(
        val v: Byte,
    ) : Scalar() {
        val value: Byte get() = v
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
        return when (self) {
            is Scalar.U8 -> self.v.toDouble()
            is Scalar.U32 -> self.v.toDouble()
            is Scalar.I16 -> self.v.toDouble()
            is Scalar.I32 -> self.v.toDouble()
            is Scalar.I64 -> self.v.toDouble()
            is Scalar.BF16 -> Float16Utils.bf16BitsToFloat(self.v).toDouble()
            is Scalar.F16 -> Float16Utils.f16BitsToFloat(self.v).toDouble()
            is Scalar.F32 -> self.v.toDouble()
            is Scalar.F64 -> self.v
            is Scalar.F8E4M3 -> self.v.toDouble()
        }
    }

    public companion object {
        private val toScalar: (Any) -> Scalar = { value ->
            when (value) {
                is UByte -> Scalar.U8(value)
                is UInt -> Scalar.U32(value)
                is Short -> Scalar.I16(value)
                is Int -> Scalar.I32(value)
                is Long -> Scalar.I64(value)
                is Float -> Scalar.F32(value)
                is Double -> Scalar.F64(value)
                is Byte -> Scalar.F8E4M3(value)
                is Scalar -> value
                else -> Scalar.F64(0.0)
            }
        }

        public fun <T : Any> from(value: T): Scalar {
            return toScalar(value)
        }

        public fun zero(dtype: DType): Scalar {
            return when (dtype) {
                DType.U8 -> Scalar.U8(0u)
                DType.U32 -> Scalar.U32(0u)
                DType.I16 -> Scalar.I16(0)
                DType.I32 -> Scalar.I32(0)
                DType.I64 -> Scalar.I64(0L)
                DType.BF16 -> Scalar.BF16(0)
                DType.F16 -> Scalar.F16(0)
                DType.F32 -> Scalar.F32(0.0f)
                DType.F64 -> Scalar.F64(0.0)
                DType.F8E4M3 -> Scalar.F8E4M3(0)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    bail("Cannot create zero scalar for dummy type $dtype")
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
                DType.BF16 -> Scalar.BF16(Float16Utils.floatToBF16Bits(1.0f))
                DType.F16 -> Scalar.F16(Float16Utils.floatToF16Bits(1.0f))
                DType.F32 -> Scalar.F32(1.0f)
                DType.F64 -> Scalar.F64(1.0)
                DType.F8E4M3 -> Scalar.F8E4M3(1)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
                    bail("Cannot create one scalar for dummy type $dtype")
                }
            }
        }
    }
}

public sealed class TensorScalar {
    public data class Tensor(
        val tensor: io.github.kotlinmania.candle.Tensor,
    ) : TensorScalar()

    public data class Scalar(
        val tensor: io.github.kotlinmania.candle.Tensor,
    ) : TensorScalar()
}

public interface TensorOrScalar {
    public fun toTensorScalar(): TensorScalar
}

public class TensorAsTensorOrScalar(
    public val self: Tensor,
) : TensorOrScalar {
    public val tensor: Tensor get() = self
    override fun toTensorScalar(): TensorScalar {
        return TensorScalar.Tensor(self.clone())
    }
}

private val tensorNewScalar: (Any, Device) -> Tensor = { v, dev ->
    when (v) {
        is Tensor -> v
        is Double -> Tensor.new(v, dev)
        is Float -> Tensor.new(v, dev)
        is Int -> Tensor.new(v, dev)
        is Long -> Tensor.new(v, dev)
        else -> Tensor.new((v as Number).toDouble(), dev)
    }
}

public class ScalarAsTensorOrScalar<T : Any>(
    public val self: T,
) : TensorOrScalar {
    public val tensor: T get() = self
    override fun toTensorScalar(): TensorScalar {
        val scalar = tensorNewScalar(self, Device.Cpu)
        return TensorScalar.Scalar(scalar)
    }
}

