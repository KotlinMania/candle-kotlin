// port-lint: source op.rs
package io.github.kotlinmania.candle.op

import io.github.kotlinmania.candle.cpu.erf.erfF32
import io.github.kotlinmania.candle.cpu.erf.erfF64
import io.github.kotlinmania.candle.customop.CustomOp1
import io.github.kotlinmania.candle.customop.CustomOp2
import io.github.kotlinmania.candle.customop.CustomOp3
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.vdGelu
import io.github.kotlinmania.candle.vdSilu
import io.github.kotlinmania.candle.vsGelu
import io.github.kotlinmania.candle.vsSilu
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.numtraits.float.Float

// Tensor Operation Enums and Traits

public typealias Target = Op?

public sealed class CmpOp {
    public data object Eq : CmpOp()
    public data object Ne : CmpOp()
    public data object Le : CmpOp()
    public data object Ge : CmpOp()
    public data object Lt : CmpOp()
    public data object Gt : CmpOp()

    public fun __rustEq(other: CmpOp): Boolean = this == other
}

public sealed class ReduceOp {
    public data object Sum : ReduceOp()
    public data object Min : ReduceOp()
    public data object Max : ReduceOp()
    public data object ArgMin : ReduceOp()
    public data object ArgMax : ReduceOp()

    public fun __rustEq(other: ReduceOp): Boolean = this == other

    internal open fun name(): String =
        when (this) {
            ReduceOp.ArgMax -> "argmax"
            ReduceOp.ArgMin -> "argmin"
            ReduceOp.Min -> "min"
            ReduceOp.Max -> "max"
            ReduceOp.Sum -> "sum"
        }
}

// These ops return the same type as their input type.
public sealed class BinaryOp {
    public data object Add : BinaryOp()
    public data object Mul : BinaryOp()
    public data object Sub : BinaryOp()
    public data object Div : BinaryOp()
    public data object Maximum : BinaryOp()
    public data object Minimum : BinaryOp()

    public fun __rustEq(other: BinaryOp): Boolean = this == other
}

// Unary ops with no argument
public sealed class UnaryOp {
    public data object Exp : UnaryOp()
    public data object Log : UnaryOp()
    public data object Sin : UnaryOp()
    public data object Cos : UnaryOp()
    public data object Abs : UnaryOp()
    public data object Neg : UnaryOp()
    public data object Recip : UnaryOp()
    public data object Sqr : UnaryOp()
    public data object Sqrt : UnaryOp()
    public data object Gelu : UnaryOp()
    public data object GeluErf : UnaryOp()
    public data object Erf : UnaryOp()
    public data object Relu : UnaryOp()
    public data object Silu : UnaryOp()
    public data object Tanh : UnaryOp()
    public data object Floor : UnaryOp()
    public data object Ceil : UnaryOp()
    public data object Round : UnaryOp()
    public data object Sign : UnaryOp()

    public fun __rustEq(other: UnaryOp): Boolean = this == other
}

public sealed class Op {
    public data class Binary(public var field0: Tensor, public var field1: Tensor, public var field2: BinaryOp) : Op()
    public data class Unary(public var field0: Tensor, public var field1: UnaryOp) : Op()
    public data class Cmp(public var field0: Tensor, public var field1: CmpOp) : Op()
    // The third argument is the reduced shape with `keepdim = true`.
    public data class Reduce(public var field0: Tensor, public var field1: ReduceOp, public var field2: List<Int>) : Op()
    public data class Matmul(public var field0: Tensor, public var field1: Tensor) : Op()
    public data class Gather(public var field0: Tensor, public var field1: Tensor, public var field2: Int) : Op()
    public data class Scatter(public var field0: Tensor, public var field1: Tensor, public var field2: Tensor, public var field3: Int) : Op()
    public data class ScatterAdd(public var field0: Tensor, public var field1: Tensor, public var field2: Tensor, public var field3: Int) : Op()
    public data class IndexSelect(public var field0: Tensor, public var field1: Tensor, public var field2: Int) : Op()
    public data class IndexAdd(public var field0: Tensor, public var field1: Tensor, public var field2: Tensor, public var field3: Int) : Op()
    public data class WhereCond(public var field0: Tensor, public var field1: Tensor, public var field2: Tensor) : Op()
    public data class Conv1D(public var arg: Tensor, public var kernel: Tensor, public var padding: Int, public var stride: Int, public var dilation: Int) : Op()
    public data class ConvTranspose1D(public var arg: Tensor, public var kernel: Tensor, public var padding: Int, public var outputPadding: Int, public var stride: Int, public var dilation: Int) : Op()
    public data class Conv2D(public var arg: Tensor, public var kernel: Tensor, public var padding: Int, public var stride: Int, public var dilation: Int) : Op()
    public data class ConvTranspose2D(public var arg: Tensor, public var kernel: Tensor, public var padding: Int, public var outputPadding: Int, public var stride: Int, public var dilation: Int) : Op()
    public data class AvgPool2D(public var arg: Tensor, public var kernelSize: Pair<Int, Int>, public var stride: Pair<Int, Int>) : Op()
    public data class MaxPool2D(public var arg: Tensor, public var kernelSize: Pair<Int, Int>, public var stride: Pair<Int, Int>) : Op()
    public data class UpsampleNearest1D(public var arg: Tensor, public var targetSize: Int) : Op()
    public data class UpsampleNearest2D(public var arg: Tensor, public var targetH: Int, public var targetW: Int) : Op()
    public data class UpsampleBilinear2D(public var arg: Tensor, public var targetH: Int, public var targetW: Int, public var alignCorners: Boolean) : Op()
    public data class Cat(public var field0: List<Tensor>, public var field1: Int) : Op()
    // add is currently unused.
    public data class Affine(public var arg: Tensor, public var mul: Double, public var add: Double) : Op()
    public data class ToDType(public var value: Tensor) : Op()
    public data class Copy(public var value: Tensor) : Op()
    public data class Broadcast(public var value: Tensor) : Op()
    public data class Narrow(public var field0: Tensor, public var field1: Int, public var field2: Int, public var field3: Int) : Op()
    public data class SliceScatter0(public var field0: Tensor, public var field1: Tensor, public var field2: Int) : Op()
    public data class Reshape(public var value: Tensor) : Op()
    public data class ToDevice(public var value: Tensor) : Op()
    public data class Transpose(public var field0: Tensor, public var field1: Int, public var field2: Int) : Op()
    public data class Permute(public var field0: Tensor, public var field1: List<Int>) : Op()
    public data class Elu(public var field0: Tensor, public var field1: Double) : Op()
    public data class Powf(public var field0: Tensor, public var field1: Double) : Op()
    public data class CustomOp1(public var field0: Tensor, public var field1: CustomOp1) : Op()
    public data class CustomOp2(public var field0: Tensor, public var field1: Tensor, public var field2: CustomOp2) : Op()
    public data class CustomOp3(public var field0: Tensor, public var field1: Tensor, public var field2: Tensor, public var field3: CustomOp3) : Op()
}

public interface UnaryOpT<T> {
    public val NAME: String

    public val KERNEL: String

    public val V: T

    public fun bf16(v1: Bf16): Bf16

    public fun f16(v1: F16): F16

    public fun f32(v1: Float): Float

    public fun f64(v1: Double): Double

    public fun u8(v1: UByte): UByte

    public fun u32(v1: UInt): UInt

    public fun i16(v1: Short): Short

    public fun i32(v1: Int): Int

    public fun i64(v1: Long): Long

    public fun f8e4m3(v1: F8e4m3): F8e4m3

    public fun bf16Vec(xs: List<Bf16>, ys: List<Bf16>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.bf16() })
    }

    public fun f16Vec(xs: List<F16>, ys: List<F16>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f16() })
    }

    public fun f32Vec(xs: List<Float>, ys: List<Float>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f32() })
    }

    public fun f64Vec(xs: List<Double>, ys: List<Double>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f64() })
    }
}

public interface BinaryOpT<T> {
    public val NAME: String

    public val KERNEL: String

    public val V: T

    public fun bf16(v1: Bf16, v2: Bf16): Bf16

    public fun f16(v1: F16, v2: F16): F16

    public fun f32(v1: Float, v2: Float): Float

    public fun f64(v1: Double, v2: Double): Double

    public fun u8(v1: UByte, v2: UByte): UByte

    public fun u32(v1: UInt, v2: UInt): UInt

    public fun i16(v1: Short, v2: Short): Short

    public fun i32(v1: Int, v2: Int): Int

    public fun i64(v1: Long, v2: Long): Long

    public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3

    public fun bf16Vec(xs1: List<Bf16>, xs2: List<Bf16>, ys: List<Bf16>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.bf16(b) })
    }

    public fun f16Vec(xs1: List<F16>, xs2: List<F16>, ys: List<F16>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.f16(b) })
    }

    public fun f32Vec(xs1: List<Float>, xs2: List<Float>, ys: List<Float>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.f32(b) })
    }

    public fun f64Vec(xs1: List<Double>, xs2: List<Double>, ys: List<Double>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.f64(b) })
    }

    public fun u8Vec(xs1: List<UByte>, xs2: List<UByte>, ys: List<UByte>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.u8(b) })
    }

    public fun u32Vec(xs1: List<UInt>, xs2: List<UInt>, ys: List<UInt>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.u32(b) })
    }

    public fun i16Vec(xs1: List<Short>, xs2: List<Short>, ys: List<Short>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.i16(b) })
    }

    public fun i32Vec(xs1: List<Int>, xs2: List<Int>, ys: List<Int>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.i32(b) })
    }

    public fun i64Vec(xs1: List<Long>, xs2: List<Long>, ys: List<Long>) {
        ((xs1.zip(xs2)).zip(ys)).forEach({ ((a, b), y) -> y = a.i64(b) })
    }

    public fun bf16ScalarVec(scalar: Bf16, xs: List<Bf16>, ys: List<Bf16>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.bf16(scalar) })
    }

    public fun f16ScalarVec(scalar: F16, xs: List<F16>, ys: List<F16>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f16(scalar) })
    }

    public fun f32ScalarVec(scalar: Float, xs: List<Float>, ys: List<Float>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f32(scalar) })
    }

    public fun f64ScalarVec(scalar: Double, xs: List<Double>, ys: List<Double>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.f64(scalar) })
    }

    public fun u8ScalarVec(scalar: UByte, xs: List<UByte>, ys: List<UByte>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.u8(scalar) })
    }

    public fun u32ScalarVec(scalar: UInt, xs: List<UInt>, ys: List<UInt>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.u32(scalar) })
    }

    public fun i16ScalarVec(scalar: Short, xs: List<Short>, ys: List<Short>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.i16(scalar) })
    }

    public fun i32ScalarVec(scalar: Int, xs: List<Int>, ys: List<Int>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.i32(scalar) })
    }

    public fun i64ScalarVec(scalar: Long, xs: List<Long>, ys: List<Long>) {
        (xs.zip(ys)).forEach({ (x, y) -> y = x.i64(scalar) })
    }
}

public open class Add

public open class Div

public open class Mul

public open class Sub

public open class Maximum

public open class Minimum

public open class Exp

public open class Log

public open class Sin

public open class Cos

public open class Abs : UnaryOpT<Abs> {
    public companion object {
        public val NAME: String = "abs"

        public val KERNEL: String = "uabs"

        public val V: Abs = Abs()

        public fun bf16(v: Bf16): Bf16 = v.abs()

        public fun f16(v: F16): F16 = v.abs()

        public fun f32(v: Float): Float = v.abs()

        public fun f64(v: Double): Double = v.abs()

        public fun u8(v: UByte): UByte = v

        public fun u32(v: UInt): UInt = v

        public fun i16(v: Short): Short = v.abs()

        public fun i32(v: Int): Int = v.abs()

        public fun i64(v: Long): Long = v.abs()

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.abs()
    }
}

public open class Neg : UnaryOpT<Neg> {
    public companion object {
        public val NAME: String = "neg"

        public val KERNEL: String = "uneg"

        public val V: Neg = Neg()

        public fun bf16(v: Bf16): Bf16 = -v

        public fun f16(v: F16): F16 = -v

        public fun f32(v: Float): Float = -v

        public fun f64(v: Double): Double = -v

        public fun u8(arg0: UByte): UByte {
            error("no unary function for u8")
        }

        public fun u32(arg0: UInt): UInt {
            error("no unary function for u32")
        }

        public fun i16(arg0: Short): Short {
            error("no unary function for i16")
        }

        public fun i32(arg0: Int): Int {
            error("no unary function for i32")
        }

        public fun i64(arg0: Long): Long {
            error("no unary function for i64")
        }

        public fun f8e4m3(v: F8e4m3): F8e4m3 = -v
    }
}

public open class Recip : UnaryOpT<Recip> {
    public companion object {
        public val NAME: String = "recip"

        public val KERNEL: String = "urecip"

        public val V: Recip = Recip()

        public fun bf16(v: Bf16): Bf16 = v.recip()

        public fun f16(v: F16): F16 = v.recip()

        public fun f32(v: Float): Float = v.recip()

        public fun f64(v: Double): Double = v.recip()

        public fun u8(arg0: UByte): UByte {
            error("no unary function for u8")
        }

        public fun u32(arg0: UInt): UInt {
            error("no unary function for u32")
        }

        public fun i16(arg0: Short): Short {
            error("no unary function for i16")
        }

        public fun i32(arg0: Int): Int {
            error("no unary function for i32")
        }

        public fun i64(arg0: Long): Long {
            error("no unary function for i64")
        }

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.recip()
    }
}

public open class Sqr

public open class Sqrt

public open class Gelu : UnaryOpT<Gelu> {
    public companion object {
        /**
         * Tanh based approximation of the `gelu` operation
         * GeluErf is the more precise one.
         * <https://en.wikipedia.org/wiki/ActivationFunction#ComparisonOfActivationFunctions>
         */
        public val NAME: String = "gelu"

        public val V: Gelu = Gelu()

        public fun bf16(v: Bf16): Bf16 = fromF32Const(0.5) * v * (ONE + Tanh(fromF32Const(SQRT_TWO_OVER_PI_F32) * v * (ONE + fromF32Const(0.044715) * v * v)))

        public fun f16(v: F16): F16 = fromF32Const(0.5) * v * (ONE + Tanh(fromF32Const(SQRT_TWO_OVER_PI_F32) * v * (ONE + fromF32Const(0.044715) * v * v)))

        public fun f32(v: Float): Float = 0.5f * v * (1.0f + Float.Tanh(SQRT_TWO_OVER_PI_F32 * v * (1.0f + 0.044715f * v * v)))

        public fun f64(v: Double): Double = 0.5 * v * (1.0 + Double.Tanh(SQRT_TWO_OVER_PI_F64 * v * (1.0 + 0.044715 * v * v)))

        public fun u8(arg0: UByte): UByte = 0u

        public fun u32(arg0: UInt): UInt = 0u

        public fun i16(arg0: Short): Short = 0

        public fun i32(arg0: Int): Int = 0

        public fun i64(arg0: Long): Long = 0L

        public fun f8e4m3(v: F8e4m3): F8e4m3 = fromF32(0.5) * v * (ONE + Tanh(fromF32(SQRT_TWO_OVER_PI_F32) * v * (ONE + fromF32(0.044715) * v * v)))

        public val KERNEL: String = "ugelu"

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsGelu(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdGelu(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            vsGelu(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            vdGelu(xs, ys)
        }
    }
}

public open class GeluErf : UnaryOpT<GeluErf> {
    public companion object {
        public val NAME: String = "gelu_erf"

        public val KERNEL: String = "ugelu_erf"

        public val V: GeluErf = GeluErf()

        public fun bf16(v: Bf16): Bf16 = fromF64(GeluErf.f64(v.toF64()))

        public fun f16(v: F16): F16 = fromF64(GeluErf.f64(v.toF64()))

        public fun f32(v: Float): Float = (erfF32(v * Float.FRAC_1_SQRT_2) + 1.0f) * 0.5f * v

        public fun f64(v: Double): Double = (erfF64(v * Double.FRAC_1_SQRT_2) + 1.0) * 0.5 * v

        public fun u8(arg0: UByte): UByte = 0u

        public fun u32(arg0: UInt): UInt = 0u

        public fun i16(arg0: Short): Short = 0

        public fun i32(arg0: Int): Int = 0

        public fun i64(arg0: Long): Long = 0L

        public fun f8e4m3(v: F8e4m3): F8e4m3 = fromF32(GeluErf.f32(v.toF32()))
    }
}

public open class Erf : UnaryOpT<Erf> {
    public companion object {
        /**
         * `erf` operation
         * <https://en.wikipedia.org/wiki/ErrorFunction>
         */
        public val NAME: String = "erf"

        public val KERNEL: String = "uerf"

        public val V: Erf = Erf()

        public fun bf16(v: Bf16): Bf16 = fromF64(Erf.f64(v.toF64()))

        public fun f16(v: F16): F16 = fromF64(Erf.f64(v.toF64()))

        public fun f32(v: Float): Float = erfF32(v)

        public fun f64(v: Double): Double = erfF64(v)

        public fun u8(arg0: UByte): UByte = 0u

        public fun u32(arg0: UInt): UInt = 0u

        public fun i16(arg0: Short): Short = 0

        public fun i32(arg0: Int): Int = 0

        public fun i64(arg0: Long): Long = 0L

        public fun f8e4m3(v: F8e4m3): F8e4m3 = fromF64(Erf.f64(v.toF64()))
    }
}

public open class Relu : UnaryOpT<Relu> {
    public companion object {
        public val NAME: String = "relu"

        public val KERNEL: String = "urelu"

        public val V: Relu = Relu()

        public fun bf16(v: Bf16): Bf16 = v.max(ZERO)

        public fun f16(v: F16): F16 = v.max(ZERO)

        public fun f32(v: Float): Float = v.max(0.0f)

        public fun f64(v: Double): Double = v.max(0.0)

        public fun u8(v: UByte): UByte = v

        public fun u32(v: UInt): UInt = v

        public fun i16(v: Short): Short = v.max(0)

        public fun i32(v: Int): Int = v.max(0)

        public fun i64(v: Long): Long = v.max(0)

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.max(ZERO)
    }
}

public open class Silu : UnaryOpT<Silu> {
    public companion object {
        /**
         * Silu operation
         */
        public val NAME: String = "silu"

        public val V: Silu = Silu()

        public fun bf16(v: Bf16): Bf16 = v / (ONE + (-v).exp())

        public fun f16(v: F16): F16 = v / (ONE + (-v).exp())

        public fun f32(v: Float): Float = v / (1.0f + (-v).exp())

        public fun f64(v: Double): Double = v / (1.0 + (-v).exp())

        public fun u8(arg0: UByte): UByte = 0u

        public fun u32(arg0: UInt): UInt = 0u

        public fun i16(arg0: Short): Short = 0

        public fun i32(arg0: Int): Int = 0

        public fun i64(arg0: Long): Long = 0L

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v / (ONE + (-v).exp())

        public val KERNEL: String = "usilu"

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsSilu(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdSilu(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            vsSilu(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            vdSilu(xs, ys)
        }
    }
}

public open class Tanh

public open class Floor : UnaryOpT<Floor> {
    public companion object {
        public val NAME: String = "floor"

        public val KERNEL: String = "ufloor"

        public val V: Floor = Floor()

        public fun bf16(v: Bf16): Bf16 = v.floor()

        public fun f16(v: F16): F16 = v.floor()

        public fun f32(v: Float): Float = v.floor()

        public fun f64(v: Double): Double = v.floor()

        public fun u8(v: UByte): UByte = v

        public fun u32(v: UInt): UInt = v

        public fun i16(v: Short): Short = v

        public fun i32(v: Int): Int = v

        public fun i64(v: Long): Long = v

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.floor()
    }
}

public open class Ceil : UnaryOpT<Ceil> {
    public companion object {
        public val NAME: String = "ceil"

        public val KERNEL: String = "uceil"

        public val V: Ceil = Ceil()

        public fun bf16(v: Bf16): Bf16 = v.ceil()

        public fun f16(v: F16): F16 = v.ceil()

        public fun f32(v: Float): Float = v.ceil()

        public fun f64(v: Double): Double = v.ceil()

        public fun u8(v: UByte): UByte = v

        public fun u32(v: UInt): UInt = v

        public fun i16(v: Short): Short = v

        public fun i32(v: Int): Int = v

        public fun i64(v: Long): Long = v

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.ceil()
    }
}

public open class Round : UnaryOpT<Round> {
    public companion object {
        public val NAME: String = "round"

        public val KERNEL: String = "uround"

        public val V: Round = Round()

        public fun bf16(v: Bf16): Bf16 = v.round()

        public fun f16(v: F16): F16 = v.round()

        public fun f32(v: Float): Float = v.round()

        public fun f64(v: Double): Double = v.round()

        public fun u8(v: UByte): UByte = v

        public fun u32(v: UInt): UInt = v

        public fun i16(v: Short): Short = v

        public fun i32(v: Int): Int = v

        public fun i64(v: Long): Long = v

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.round()
    }
}

public open class Sign : UnaryOpT<Sign> {
    public companion object {
        public val NAME: String = "sign"

        public val KERNEL: String = "usign"

        public val V: Sign = Sign()

        public fun bf16(v: Bf16): Bf16 = from((if (v > ZERO) 1 else 0).toByte()) - from((if (v < ZERO) 1 else 0).toByte())

        public fun f16(v: F16): F16 = from((if (v > ZERO) 1 else 0).toByte()) - from((if (v < ZERO) 1 else 0).toByte())

        public fun f32(v: Float): Float = Float.from(v > 0.0f) - Float.from(v < 0.0f)

        public fun f64(v: Double): Double = Double.from(v > 0.0) - Double.from(v < 0.0)

        public fun u8(v: UByte): UByte = UByte.min(1, v)

        public fun u32(v: UInt): UInt = UInt.min(1, v)

        public fun i16(v: Short): Short = (if (v > 0.toShort()) 1 else 0).toShort() - (if (v < 0.toShort()) 1 else 0).toShort()

        public fun i32(v: Int): Int = (if (v > 0) 1 else 0) - (if (v < 0) 1 else 0)

        public fun i64(v: Long): Long = (if (v > 0L) 1L else 0L) - (if (v < 0L) 1L else 0L)

        public fun f8e4m3(v: F8e4m3): F8e4m3 =
            if (v > ZERO) {
                ONE
            } else if (v < ZERO) {
                -ONE
            } else {
                ZERO
            }
    }
}

// Hardcode the value for sqrt(2/pi)
// https://github.com/huggingface/candle/issues/1982
public val SQRT_TWO_OVER_PI_F32: Float = 0.79788456080286535587989211986876373f

public val SQRT_TWO_OVER_PI_F64: Double = 0.79788456080286535587989211986876373

/**
 * `BackpropOp` is a wrapper around `Option<Op>`. The main goal is to ensure that dependencies are
 * properly checked when creating a new value
 */
public open class BackpropOp(
    public var value: Op?,
) {
    public var field0: Op?
        get() = value
        set(v) {
            value = v
        }

    internal open fun isNone(): Boolean = this.value == null

    public open fun deref(): Op? = this.value

    public companion object {
        public fun none(): BackpropOp = BackpropOp(null)

        internal fun new1(arg: Tensor, f: (Tensor) -> Op): BackpropOp {
            val op =
                if (arg.trackOp()) {
                    f(arg.clone())
                } else {
                    null
                }
            return BackpropOp(op)
        }

        internal fun new2(arg1: Tensor, arg2: Tensor, f: (Tensor, Tensor) -> Op): BackpropOp {
            val op =
                if (arg1.trackOp() || arg2.trackOp()) {
                    f(arg1.clone(), arg2.clone())
                } else {
                    null
                }
            return BackpropOp(op)
        }

        internal fun new3(arg1: Tensor, arg2: Tensor, arg3: Tensor, f: (Tensor, Tensor, Tensor) -> Op): BackpropOp {
            val op =
                if (arg1.trackOp() || arg2.trackOp() || arg3.trackOp()) {
                    f(arg1.clone(), arg2.clone(), arg3.clone())
                } else {
                    null
                }
            return BackpropOp(op)
        }

        internal fun <A : AsRef<Tensor>> new(args: List<A>, f: (List<Tensor>) -> Op): BackpropOp {
            val op =
                if (args.any({ arg -> arg.asRef().trackOp() })) {
                    val args1: List<Tensor> = args.map({ arg -> arg.asRef().clone() }).collect()
                    f(args1)
                } else {
                    null
                }
            return BackpropOp(op)
        }
    }
}
