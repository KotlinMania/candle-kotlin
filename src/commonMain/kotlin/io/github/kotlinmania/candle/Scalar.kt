// port-lint: source candle-core/src/scalar.rs
package io.github.kotlinmania.candle

public sealed class Scalar {
    public data class U8(
        val value: UByte,
    ) : Scalar()

    public data class U32(
        val value: UInt,
    ) : Scalar()

    public data class I16(
        val value: Short,
    ) : Scalar()

    public data class I32(
        val value: Int,
    ) : Scalar()

    public data class I64(
        val value: Long,
    ) : Scalar()

    public data class BF16(
        val value: Short,
    ) : Scalar()

    public data class F16(
        val value: Short,
    ) : Scalar()

    public data class F32(
        val value: Float,
    ) : Scalar()

    public data class F64(
        val value: Double,
    ) : Scalar()

    public data class F8E4M3(
        val value: Byte,
    ) : Scalar()

    public fun dtype(): DType =
        when (this) {
            is U8 -> DType.U8
            is U32 -> DType.U32
            is I16 -> DType.I16
            is I32 -> DType.I32
            is I64 -> DType.I64
            is BF16 -> DType.BF16
            is F16 -> DType.F16
            is F32 -> DType.F32
            is F64 -> DType.F64
            is F8E4M3 -> DType.F8E4M3
        }

    public fun toF64(): Double =
        when (this) {
            is U8 -> value.toDouble()
            is U32 -> value.toDouble()
            is I16 -> value.toDouble()
            is I32 -> value.toDouble()
            is I64 -> value.toDouble()
            is BF16 -> Float16Utils.bf16BitsToFloat(value).toDouble()
            is F16 -> Float16Utils.f16BitsToFloat(value).toDouble()
            is F32 -> value.toDouble()
            is F64 -> value
            is F8E4M3 -> value.toDouble()
        }

    public companion object {
        public fun zero(dtype: DType): Scalar =
            when (dtype) {
                DType.U8 -> U8(0u)
                DType.U32 -> U32(0u)
                DType.I16 -> I16(0)
                DType.I32 -> I32(0)
                DType.I64 -> I64(0L)
                DType.BF16 -> BF16(0)
                DType.F16 -> F16(0)
                DType.F32 -> F32(0.0f)
                DType.F64 -> F64(0.0)
                DType.F8E4M3 -> F8E4M3(0)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
                    bail("Cannot create zero scalar for dummy type $dtype")
            }

        public fun one(dtype: DType): Scalar =
            when (dtype) {
                DType.U8 -> U8(1u)
                DType.U32 -> U32(1u)
                DType.I16 -> I16(1)
                DType.I32 -> I32(1)
                DType.I64 -> I64(1L)
                DType.BF16 -> BF16(Float16Utils.floatToBF16Bits(1.0f))
                DType.F16 -> F16(Float16Utils.floatToF16Bits(1.0f))
                DType.F32 -> F32(1.0f)
                DType.F64 -> F64(1.0)
                DType.F8E4M3 -> F8E4M3(1)
                DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
                    bail("Cannot create one scalar for dummy type $dtype")
            }

        public fun from(v: UByte): Scalar = U8(v)

        public fun from(v: UInt): Scalar = U32(v)

        public fun from(v: Short): Scalar = I16(v)

        public fun from(v: Int): Scalar = I32(v)

        public fun from(v: Long): Scalar = I64(v)

        public fun from(v: Float): Scalar = F32(v)

        public fun from(v: Double): Scalar = F64(v)
    }
}
