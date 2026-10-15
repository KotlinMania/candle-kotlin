// port-lint: source op.rs
package io.github.kotlinmania.candle

/**
 * Tensor Operation Enums and Traits
 */

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh

public enum class CmpOp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}

public enum class ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
    ;

    public fun name(): String =
        when (this) {
            ArgMax -> "argmax"
            ArgMin -> "argmin"
            Min -> "min"
            Max -> "max"
            Sum -> "sum"
        }

    public fun opName(): String = name()
}

// These ops return the same type as their input type.
public enum class BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

// Unary ops with no argument
public enum class UnaryOp {
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Neg,
    Recip,
    Sqr,
    Sqrt,
    Gelu,
    GeluErf,
    Erf,
    Relu,
    Silu,
    Tanh,
    Floor,
    Ceil,
    Round,
    Sign,
}

public sealed class Op {
    public data class Binary(
        val lhs: Tensor,
        val rhs: Tensor,
        val op: BinaryOp,
    ) : Op()

    public data class Unary(
        val arg: Tensor,
        val op: UnaryOp,
    ) : Op()

    public data class Cmp(
        val arg: Tensor,
        val op: CmpOp,
    ) : Op()

    // The third argument is the reduced shape with `keepdim=true`.
    public data class Reduce(
        val arg: Tensor,
        val op: ReduceOp,
        val dims: List<Int>,
    ) : Op()

    public data class Matmul(
        val lhs: Tensor,
        val rhs: Tensor,
    ) : Op()

    public data class Gather(
        val arg: Tensor,
        val index: Tensor,
        val dim: Int,
    ) : Op()

    public data class Scatter(
        val arg: Tensor,
        val index: Tensor,
        val src: Tensor,
        val dim: Int,
    ) : Op()

    public data class ScatterAdd(
        val arg: Tensor,
        val index: Tensor,
        val src: Tensor,
        val dim: Int,
    ) : Op()

    public data class IndexSelect(
        val arg: Tensor,
        val index: Tensor,
        val dim: Int,
    ) : Op()

    public data class IndexAdd(
        val arg: Tensor,
        val index: Tensor,
        val src: Tensor,
        val dim: Int,
    ) : Op()

    public data class WhereCond(
        val cond: Tensor,
        val onTrue: Tensor,
        val onFalse: Tensor,
    ) : Op()

    public data class Conv1D(
        val arg: Tensor,
        val kernel: Tensor,
        val padding: Int,
        val stride: Int,
        val dilation: Int,
    ) : Op()

    public data class ConvTranspose1D(
        val arg: Tensor,
        val kernel: Tensor,
        val padding: Int,
        val outputPadding: Int,
        val stride: Int,
        val dilation: Int,
    ) : Op()

    public data class Conv2D(
        val arg: Tensor,
        val kernel: Tensor,
        val padding: Int,
        val stride: Int,
        val dilation: Int,
    ) : Op()

    public data class ConvTranspose2D(
        val arg: Tensor,
        val kernel: Tensor,
        val padding: Int,
        val outputPadding: Int,
        val stride: Int,
        val dilation: Int,
    ) : Op()

    public data class AvgPool2D(
        val arg: Tensor,
        val kernelSize: Pair<Int, Int>,
        val stride: Pair<Int, Int>,
    ) : Op()

    public data class MaxPool2D(
        val arg: Tensor,
        val kernelSize: Pair<Int, Int>,
        val stride: Pair<Int, Int>,
    ) : Op()

    public data class UpsampleNearest1D(
        val arg: Tensor,
        val targetSize: Int,
    ) : Op()

    public data class UpsampleNearest2D(
        val arg: Tensor,
        val targetH: Int,
        val targetW: Int,
    ) : Op()

    public data class UpsampleBilinear2D(
        val arg: Tensor,
        val targetH: Int,
        val targetW: Int,
        val alignCorners: Boolean,
    ) : Op()

    public data class Cat(
        val args: List<Tensor>,
        val dim: Int,
    ) : Op()

    public data class Affine(
        val arg: Tensor,
        val mul: Double,
        val add: Double,
    ) : Op()

    public data class ToDType(
        val arg: Tensor,
    ) : Op()

    public data class Copy(
        val arg: Tensor,
    ) : Op()

    public data class Broadcast(
        val arg: Tensor,
    ) : Op()

    public data class Narrow(
        val arg: Tensor,
        val dim: Int,
        val start: Int,
        val len: Int,
    ) : Op()

    public data class SliceScatter0(
        val lhs: Tensor,
        val rhs: Tensor,
        val start: Int,
    ) : Op()

    public data class Reshape(
        val arg: Tensor,
    ) : Op()

    public data class ToDevice(
        val arg: Tensor,
    ) : Op()

    public data class Transpose(
        val arg: Tensor,
        val dim1: Int,
        val dim2: Int,
    ) : Op()

    public data class Permute(
        val arg: Tensor,
        val dims: List<Int>,
    ) : Op()

    public data class Elu(
        val arg: Tensor,
        val alpha: Double,
    ) : Op()

    public data class Powf(
        val arg: Tensor,
        val e: Double,
    ) : Op()

    public data class CustomOp1(
        val arg: Tensor,
        val op: io.github.kotlinmania.candle.CustomOp1,
    ) : Op()

    public data class CustomOp2(
        val lhs: Tensor,
        val rhs: Tensor,
        val op: io.github.kotlinmania.candle.CustomOp2,
    ) : Op()

    public data class CustomOp3(
        val t1: Tensor,
        val t2: Tensor,
        val t3: Tensor,
        val op: io.github.kotlinmania.candle.CustomOp3,
    ) : Op()
}

/**
 * `BackpropOp` is a wrapper around `Op?`. The main goal is to ensure that dependencies are
 * properly checked when creating a new value
 */
public sealed class BackpropOp {
    public object None : BackpropOp()

    public data class Op1(
        val arg: Tensor,
        val buildOp: (Tensor) -> Op,
    ) : BackpropOp()

    public data class Op2(
        val lhs: Tensor,
        val rhs: Tensor,
        val buildOp: (Tensor, Tensor) -> Op,
    ) : BackpropOp()

    public data class Op3(
        val t1: Tensor,
        val t2: Tensor,
        val t3: Tensor,
        val buildOp: (Tensor, Tensor, Tensor) -> Op,
    ) : BackpropOp()

    public data class OpN(
        val args: List<Tensor>,
        val buildOp: (List<Tensor>) -> Op,
    ) : BackpropOp()

    public fun toOp(): Op? =
        when (this) {
            is None -> null
            is Op1 -> buildOp(arg)
            is Op2 -> buildOp(lhs, rhs)
            is Op3 -> buildOp(t1, t2, t3)
            is OpN -> buildOp(args)
        }

    public fun isNone(): Boolean = this is None

    public fun deref(): Op? = toOp()

    public companion object {
        public fun none(): BackpropOp = None

        public fun new1(arg: Tensor, buildOp: (Tensor) -> Op): BackpropOp =
            if (arg.trackOp()) Op1(arg, buildOp) else None

        public fun new2(lhs: Tensor, rhs: Tensor, buildOp: (Tensor, Tensor) -> Op): BackpropOp =
            if (lhs.trackOp() || rhs.trackOp()) Op2(lhs, rhs, buildOp) else None

        public fun new3(
            t1: Tensor,
            t2: Tensor,
            t3: Tensor,
            buildOp: (Tensor, Tensor, Tensor) -> Op,
        ): BackpropOp =
            if (t1.trackOp() || t2.trackOp() || t3.trackOp()) Op3(t1, t2, t3, buildOp) else None

        public fun new(args: List<Tensor>, buildOp: (List<Tensor>) -> Op): BackpropOp =
            if (args.any { it.trackOp() }) OpN(args, buildOp) else None
    }
}

// Hardcode the value for sqrt(2/pi)
// https://github.com/huggingface/candle/issues/1982
private const val SQRT_TWO_OVER_PI_F32: Float = 0.7978845608028654f
private const val SQRT_TWO_OVER_PI_F64: Double = 0.7978845608028654
private const val FRAC_1_SQRT_2_F32: Float = 0.7071067811865475f
private const val FRAC_1_SQRT_2_F64: Double = 0.7071067811865476

public object MathOps {
    // Standard numerical erf implementation (Abramowitz and Stegun 7.1.26 formula)
    public fun erf(x: Double): Double {
        val sign = if (x < 0) -1.0 else 1.0
        val ax = abs(x)
        val p = 0.3275911
        val a1 = 0.254829592
        val a2 = -0.284496736
        val a3 = 1.421413741
        val a4 = -1.453152027
        val a5 = 1.061405429

        val t = 1.0 / (1.0 + p * ax)
        val y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
        return sign * y
    }

    public fun erf(x: Float): Float = erf(x.toDouble()).toFloat()

    public fun geluErf(x: Double): Double =
        (erf(x * FRAC_1_SQRT_2_F64) + 1.0) * 0.5 * x

    public fun geluErf(x: Float): Float =
        (erf(x * FRAC_1_SQRT_2_F32) + 1.0f) * 0.5f * x

    public fun gelu(x: Double): Double =
        0.5 * x * (1.0 + tanh(SQRT_TWO_OVER_PI_F64 * x * (1.0 + 0.044715 * x * x)))

    public fun gelu(x: Float): Float =
        0.5f * x * (1.0f + tanh(SQRT_TWO_OVER_PI_F32 * x * (1.0f + 0.044715f * x * x)))

    public fun silu(x: Double): Double = x / (1.0 + exp(-x))

    public fun silu(x: Float): Float = x / (1.0f + exp(-x))

    public fun relu(x: Double): Double = if (x > 0.0) x else 0.0

    public fun relu(x: Float): Float = if (x > 0.0f) x else 0.0f
}

public interface UnaryOpT {
    public val name: String
    public val kernel: String
    public fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(f32(v1.toFloat()))
    public fun f16(v1: Float16): Float16 = Float16.fromFloat(f32(v1.toFloat()))
    public fun f32(v1: Float): Float
    public fun f64(v1: Double): Double
    public fun u8(v1: UByte): UByte
    public fun u32(v1: UInt): UInt
    public fun i16(v1: Short): Short
    public fun i32(v1: Int): Int
    public fun i64(v1: Long): Long
    public fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(f32(v1.toFloat()))

    public fun bf16Vec(xs: ShortArray, ys: ShortArray) {
        for (i in xs.indices) {
            ys[i] = bf16(BFloat16(xs[i])).bits
        }
    }

    public fun f16Vec(xs: ShortArray, ys: ShortArray) {
        for (i in xs.indices) {
            ys[i] = f16(Float16(xs[i])).bits
        }
    }

    public fun f32Vec(xs: FloatArray, ys: FloatArray) {
        for (i in xs.indices) {
            ys[i] = f32(xs[i])
        }
    }

    public fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        for (i in xs.indices) {
            ys[i] = f64(xs[i])
        }
    }
}

public interface BinaryOpT {
    public val name: String
    public val kernel: String
    public fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 =
        BFloat16.fromFloat(f32(v1.toFloat(), v2.toFloat()))
    public fun f16(v1: Float16, v2: Float16): Float16 =
        Float16.fromFloat(f32(v1.toFloat(), v2.toFloat()))
    public fun f32(v1: Float, v2: Float): Float
    public fun f64(v1: Double, v2: Double): Double
    public fun u8(v1: UByte, v2: UByte): UByte
    public fun u32(v1: UInt, v2: UInt): UInt
    public fun i16(v1: Short, v2: Short): Short
    public fun i32(v1: Int, v2: Int): Int
    public fun i64(v1: Long, v2: Long): Long
    public fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 =
        F8E4M3.fromFloat(f32(v1.toFloat(), v2.toFloat()))

    public fun bf16Vec(xs1: ShortArray, xs2: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = bf16(BFloat16(xs1[i]), BFloat16(xs2[i])).bits
        }
    }

    public fun f16Vec(xs1: ShortArray, xs2: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = f16(Float16(xs1[i]), Float16(xs2[i])).bits
        }
    }

    public fun f32Vec(xs1: FloatArray, xs2: FloatArray, ys: FloatArray) {
        for (i in ys.indices) {
            ys[i] = f32(xs1[i], xs2[i])
        }
    }

    public fun f64Vec(xs1: DoubleArray, xs2: DoubleArray, ys: DoubleArray) {
        for (i in ys.indices) {
            ys[i] = f64(xs1[i], xs2[i])
        }
    }

    public fun u8Vec(xs1: UByteArray, xs2: UByteArray, ys: UByteArray) {
        for (i in ys.indices) {
            ys[i] = u8(xs1[i], xs2[i])
        }
    }

    public fun u32Vec(xs1: UIntArray, xs2: UIntArray, ys: UIntArray) {
        for (i in ys.indices) {
            ys[i] = u32(xs1[i], xs2[i])
        }
    }

    public fun i16Vec(xs1: ShortArray, xs2: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = i16(xs1[i], xs2[i])
        }
    }

    public fun i32Vec(xs1: IntArray, xs2: IntArray, ys: IntArray) {
        for (i in ys.indices) {
            ys[i] = i32(xs1[i], xs2[i])
        }
    }

    public fun i64Vec(xs1: LongArray, xs2: LongArray, ys: LongArray) {
        for (i in ys.indices) {
            ys[i] = i64(xs1[i], xs2[i])
        }
    }

    // Scalar-broadcast variants: ys[i] = f(xs[i], scalar).
    // Used by binaryMapVec for the (1,0) inner-stride branch where one tensor
    // is contiguous and the other broadcasts a single value.
    public fun bf16ScalarVec(scalar: BFloat16, xs: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = bf16(BFloat16(xs[i]), scalar).bits
        }
    }

    public fun f16ScalarVec(scalar: Float16, xs: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = f16(Float16(xs[i]), scalar).bits
        }
    }

    public fun f32ScalarVec(scalar: Float, xs: FloatArray, ys: FloatArray) {
        for (i in ys.indices) {
            ys[i] = f32(xs[i], scalar)
        }
    }

    public fun f64ScalarVec(scalar: Double, xs: DoubleArray, ys: DoubleArray) {
        for (i in ys.indices) {
            ys[i] = f64(xs[i], scalar)
        }
    }

    public fun u8ScalarVec(scalar: UByte, xs: UByteArray, ys: UByteArray) {
        for (i in ys.indices) {
            ys[i] = u8(xs[i], scalar)
        }
    }

    public fun u32ScalarVec(scalar: UInt, xs: UIntArray, ys: UIntArray) {
        for (i in ys.indices) {
            ys[i] = u32(xs[i], scalar)
        }
    }

    public fun i16ScalarVec(scalar: Short, xs: ShortArray, ys: ShortArray) {
        for (i in ys.indices) {
            ys[i] = i16(xs[i], scalar)
        }
    }

    public fun i32ScalarVec(scalar: Int, xs: IntArray, ys: IntArray) {
        for (i in ys.indices) {
            ys[i] = i32(xs[i], scalar)
        }
    }

    public fun i64ScalarVec(scalar: Long, xs: LongArray, ys: LongArray) {
        for (i in ys.indices) {
            ys[i] = i64(xs[i], scalar)
        }
    }
}

public object Add : BinaryOpT {
    override val name: String = "add"
    override val kernel: String = "badd"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = BFloat16.fromFloat(v1.toFloat() + v2.toFloat())
    override fun f16(v1: Float16, v2: Float16): Float16 = Float16.fromFloat(v1.toFloat() + v2.toFloat())
    override fun f32(v1: Float, v2: Float): Float = v1 + v2
    override fun f64(v1: Double, v2: Double): Double = v1 + v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 + v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 + v2
    override fun i16(v1: Short, v2: Short): Short = (v1 + v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 + v2
    override fun i64(v1: Long, v2: Long): Long = v1 + v2
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = F8E4M3.fromFloat(v1.toFloat() + v2.toFloat())
}

public object Div : BinaryOpT {
    override val name: String = "div"
    override val kernel: String = "bdiv"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = BFloat16.fromFloat(v1.toFloat() / v2.toFloat())
    override fun f16(v1: Float16, v2: Float16): Float16 = Float16.fromFloat(v1.toFloat() / v2.toFloat())
    override fun f32(v1: Float, v2: Float): Float = v1 / v2
    override fun f64(v1: Double, v2: Double): Double = v1 / v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 / v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 / v2
    override fun i16(v1: Short, v2: Short): Short = (v1 / v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 / v2
    override fun i64(v1: Long, v2: Long): Long = v1 / v2
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = F8E4M3.fromFloat(v1.toFloat() / v2.toFloat())
}

public object Mul : BinaryOpT {
    override val name: String = "mul"
    override val kernel: String = "bmul"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = BFloat16.fromFloat(v1.toFloat() * v2.toFloat())
    override fun f16(v1: Float16, v2: Float16): Float16 = Float16.fromFloat(v1.toFloat() * v2.toFloat())
    override fun f32(v1: Float, v2: Float): Float = v1 * v2
    override fun f64(v1: Double, v2: Double): Double = v1 * v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 * v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 * v2
    override fun i16(v1: Short, v2: Short): Short = (v1 * v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 * v2
    override fun i64(v1: Long, v2: Long): Long = v1 * v2
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = F8E4M3.fromFloat(v1.toFloat() * v2.toFloat())
}

public object Sub : BinaryOpT {
    override val name: String = "sub"
    override val kernel: String = "bsub"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = BFloat16.fromFloat(v1.toFloat() - v2.toFloat())
    override fun f16(v1: Float16, v2: Float16): Float16 = Float16.fromFloat(v1.toFloat() - v2.toFloat())
    override fun f32(v1: Float, v2: Float): Float = v1 - v2
    override fun f64(v1: Double, v2: Double): Double = v1 - v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 - v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 - v2
    override fun i16(v1: Short, v2: Short): Short = (v1 - v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 - v2
    override fun i64(v1: Long, v2: Long): Long = v1 - v2
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = F8E4M3.fromFloat(v1.toFloat() - v2.toFloat())
}

public object Maximum : BinaryOpT {
    override val name: String = "maximum"
    override val kernel: String = "bmaximum"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = if (v1.toFloat() < v2.toFloat()) v2 else v1
    override fun f16(v1: Float16, v2: Float16): Float16 = if (v1.toFloat() < v2.toFloat()) v2 else v1
    override fun f32(v1: Float, v2: Float): Float = if (v1 < v2) v2 else v1
    override fun f64(v1: Double, v2: Double): Double = if (v1 < v2) v2 else v1
    override fun u8(v1: UByte, v2: UByte): UByte = if (v1 < v2) v2 else v1
    override fun u32(v1: UInt, v2: UInt): UInt = if (v1 < v2) v2 else v1
    override fun i16(v1: Short, v2: Short): Short = if (v1 < v2) v2 else v1
    override fun i32(v1: Int, v2: Int): Int = if (v1 < v2) v2 else v1
    override fun i64(v1: Long, v2: Long): Long = if (v1 < v2) v2 else v1
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = if (v1.toFloat() < v2.toFloat()) v2 else v1
}

public object Minimum : BinaryOpT {
    override val name: String = "minimum"
    override val kernel: String = "bminimum"
    override fun bf16(v1: BFloat16, v2: BFloat16): BFloat16 = if (v1.toFloat() > v2.toFloat()) v2 else v1
    override fun f16(v1: Float16, v2: Float16): Float16 = if (v1.toFloat() > v2.toFloat()) v2 else v1
    override fun f32(v1: Float, v2: Float): Float = if (v1 > v2) v2 else v1
    override fun f64(v1: Double, v2: Double): Double = if (v1 > v2) v2 else v1
    override fun u8(v1: UByte, v2: UByte): UByte = if (v1 > v2) v2 else v1
    override fun u32(v1: UInt, v2: UInt): UInt = if (v1 > v2) v2 else v1
    override fun i16(v1: Short, v2: Short): Short = if (v1 > v2) v2 else v1
    override fun i32(v1: Int, v2: Int): Int = if (v1 > v2) v2 else v1
    override fun i64(v1: Long, v2: Long): Long = if (v1 > v2) v2 else v1
    override fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3 = if (v1.toFloat() > v2.toFloat()) v2 else v1
}

public object Exp : UnaryOpT {
    override val name: String = "exp"
    override val kernel: String = "uexp"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.exp(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.exp(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.exp(v1)
    override fun f64(v1: Double): Double = kotlin.math.exp(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.exp(v1.toFloat()))
}

public object Log : UnaryOpT {
    override val name: String = "log"
    override val kernel: String = "ulog"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.ln(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.ln(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.ln(v1)
    override fun f64(v1: Double): Double = kotlin.math.ln(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.ln(v1.toFloat()))
}

public object Sin : UnaryOpT {
    override val name: String = "sin"
    override val kernel: String = "usin"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.sin(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.sin(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.sin(v1)
    override fun f64(v1: Double): Double = kotlin.math.sin(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.sin(v1.toFloat()))
}

public object Cos : UnaryOpT {
    override val name: String = "cos"
    override val kernel: String = "ucos"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.cos(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.cos(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.cos(v1)
    override fun f64(v1: Double): Double = kotlin.math.cos(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.cos(v1.toFloat()))
}

public object Abs : UnaryOpT {
    override val name: String = "abs"
    override val kernel: String = "uabs"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.abs(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.abs(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.abs(v1)
    override fun f64(v1: Double): Double = kotlin.math.abs(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = kotlin.math.abs(v1.toInt()).toShort()
    override fun i32(v1: Int): Int = kotlin.math.abs(v1)
    override fun i64(v1: Long): Long = kotlin.math.abs(v1)
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.abs(v1.toFloat()))
}

public object Neg : UnaryOpT {
    override val name: String = "neg"
    override val kernel: String = "uneg"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(-v1.toFloat())
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(-v1.toFloat())
    override fun f32(v1: Float): Float = -v1
    override fun f64(v1: Double): Double = -v1
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = (-v1).toShort()
    override fun i32(v1: Int): Int = -v1
    override fun i64(v1: Long): Long = -v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(-v1.toFloat())
}

public object Recip : UnaryOpT {
    override val name: String = "recip"
    override val kernel: String = "urecip"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(1.0f / v1.toFloat())
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(1.0f / v1.toFloat())
    override fun f32(v1: Float): Float = 1.0f / v1
    override fun f64(v1: Double): Double = 1.0 / v1
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(1.0f / v1.toFloat())
}

public object Sqr : UnaryOpT {
    override val name: String = "sqr"
    override val kernel: String = "usqr"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(v1.toFloat() * v1.toFloat())
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(v1.toFloat() * v1.toFloat())
    override fun f32(v1: Float): Float = v1 * v1
    override fun f64(v1: Double): Double = v1 * v1
    override fun u8(v1: UByte): UByte = (v1 * v1).toUByte()
    override fun u32(v1: UInt): UInt = v1 * v1
    override fun i16(v1: Short): Short = (v1 * v1).toShort()
    override fun i32(v1: Int): Int = v1 * v1
    override fun i64(v1: Long): Long = v1 * v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(v1.toFloat() * v1.toFloat())
}

public object Sqrt : UnaryOpT {
    override val name: String = "sqrt"
    override val kernel: String = "usqrt"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.sqrt(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.sqrt(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.sqrt(v1)
    override fun f64(v1: Double): Double = kotlin.math.sqrt(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.sqrt(v1.toFloat()))
}

/**
 * Tanh based approximation of the `gelu` operation
 * GeluErf is the more precise one.
 * Activation function comparison of activation functions.
 */
public object Gelu : UnaryOpT {
    override val name: String = "gelu"
    override val kernel: String = "ugelu"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(f32(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(f32(v1.toFloat()))
    override fun f32(v1: Float): Float =
        0.5f * v1 * (1.0f + kotlin.math.tanh(SQRT_TWO_OVER_PI_F32 * v1 * (1.0f + 0.044715f * v1 * v1)))
    override fun f64(v1: Double): Double =
        0.5 * v1 * (1.0 + kotlin.math.tanh(SQRT_TWO_OVER_PI_F64 * v1 * (1.0 + 0.044715 * v1 * v1)))
    override fun u8(v1: UByte): UByte = 0u
    override fun u32(v1: UInt): UInt = 0u
    override fun i16(v1: Short): Short = 0
    override fun i32(v1: Int): Int = 0
    override fun i64(v1: Long): Long = 0L
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(f32(v1.toFloat()))
    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        for (i in xs.indices) {
            ys[i] = f32(xs[i])
        }
    }
    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        for (i in xs.indices) {
            ys[i] = f64(xs[i])
        }
    }
}

public object GeluErf : UnaryOpT {
    override val name: String = "gelu_erf"
    override val kernel: String = "ugelu_erf"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromDouble(f64(v1.toDouble()))
    override fun f16(v1: Float16): Float16 = Float16.fromDouble(f64(v1.toDouble()))
    override fun f32(v1: Float): Float = (MathOps.erf(v1 * FRAC_1_SQRT_2_F32) + 1.0f) * 0.5f * v1
    override fun f64(v1: Double): Double = (MathOps.erf(v1 * FRAC_1_SQRT_2_F64) + 1.0) * 0.5 * v1
    override fun u8(v1: UByte): UByte = 0u
    override fun u32(v1: UInt): UInt = 0u
    override fun i16(v1: Short): Short = 0
    override fun i32(v1: Int): Int = 0
    override fun i64(v1: Long): Long = 0L
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(f32(v1.toFloat()))
    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        for (i in xs.indices) {
            ys[i] = f32(xs[i])
        }
    }
    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        for (i in xs.indices) {
            ys[i] = f64(xs[i])
        }
    }
}

/**
 * `erf` operation
 * Error function.
 */
public object Erf : UnaryOpT {
    override val name: String = "erf"
    override val kernel: String = "uerf"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromDouble(f64(v1.toDouble()))
    override fun f16(v1: Float16): Float16 = Float16.fromDouble(f64(v1.toDouble()))
    override fun f32(v1: Float): Float = MathOps.erf(v1)
    override fun f64(v1: Double): Double = MathOps.erf(v1)
    override fun u8(v1: UByte): UByte = 0u
    override fun u32(v1: UInt): UInt = 0u
    override fun i16(v1: Short): Short = 0
    override fun i32(v1: Int): Int = 0
    override fun i64(v1: Long): Long = 0L
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromDouble(f64(v1.toDouble()))
    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        for (i in xs.indices) {
            ys[i] = f32(xs[i])
        }
    }
    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        for (i in xs.indices) {
            ys[i] = f64(xs[i])
        }
    }
}

public object Relu : UnaryOpT {
    override val name: String = "relu"
    override val kernel: String = "urelu"
    override fun bf16(v1: BFloat16): BFloat16 = if (v1.toFloat() > 0.0f) v1 else BFloat16.ZERO
    override fun f16(v1: Float16): Float16 = if (v1.toFloat() > 0.0f) v1 else Float16.ZERO
    override fun f32(v1: Float): Float = maxOf(0.0f, v1)
    override fun f64(v1: Double): Double = maxOf(0.0, v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = maxOf(0.toShort(), v1)
    override fun i32(v1: Int): Int = maxOf(0, v1)
    override fun i64(v1: Long): Long = maxOf(0L, v1)
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = if (v1.toFloat() > 0.0f) v1 else F8E4M3.ZERO
}

/**
 * Silu operation
 */
public object Silu : UnaryOpT {
    override val name: String = "silu"
    override val kernel: String = "usilu"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(f32(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(f32(v1.toFloat()))
    override fun f32(v1: Float): Float = v1 / (1.0f + kotlin.math.exp(-v1))
    override fun f64(v1: Double): Double = v1 / (1.0 + kotlin.math.exp(-v1))
    override fun u8(v1: UByte): UByte = 0u
    override fun u32(v1: UInt): UInt = 0u
    override fun i16(v1: Short): Short = 0
    override fun i32(v1: Int): Int = 0
    override fun i64(v1: Long): Long = 0L
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(f32(v1.toFloat()))
    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        for (i in xs.indices) {
            ys[i] = f32(xs[i])
        }
    }
    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        for (i in xs.indices) {
            ys[i] = f64(xs[i])
        }
    }
}

public object Tanh : UnaryOpT {
    override val name: String = "tanh"
    override val kernel: String = "utanh"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.tanh(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.tanh(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.tanh(v1)
    override fun f64(v1: Double): Double = kotlin.math.tanh(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.tanh(v1.toFloat()))
}

public object Floor : UnaryOpT {
    override val name: String = "floor"
    override val kernel: String = "ufloor"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.floor(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.floor(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.floor(v1)
    override fun f64(v1: Double): Double = kotlin.math.floor(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.floor(v1.toFloat()))
}

public object Ceil : UnaryOpT {
    override val name: String = "ceil"
    override val kernel: String = "uceil"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.ceil(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.ceil(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.ceil(v1)
    override fun f64(v1: Double): Double = kotlin.math.ceil(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.ceil(v1.toFloat()))
}

public object Round : UnaryOpT {
    override val name: String = "round"
    override val kernel: String = "uround"
    override fun bf16(v1: BFloat16): BFloat16 = BFloat16.fromFloat(kotlin.math.round(v1.toFloat()))
    override fun f16(v1: Float16): Float16 = Float16.fromFloat(kotlin.math.round(v1.toFloat()))
    override fun f32(v1: Float): Float = kotlin.math.round(v1)
    override fun f64(v1: Double): Double = kotlin.math.round(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
    override fun f8e4m3(v1: F8E4M3): F8E4M3 = F8E4M3.fromFloat(kotlin.math.round(v1.toFloat()))
}

public object Sign : UnaryOpT {
    override val name: String = "sign"
    override val kernel: String = "usign"
    override fun bf16(v1: BFloat16): BFloat16 {
        val f = v1.toFloat()
        return when {
            f > 0.0f -> BFloat16.ONE
            f < 0.0f -> BFloat16.fromFloat(-1.0f)
            else -> BFloat16.ZERO
        }
    }
    override fun f16(v1: Float16): Float16 {
        val f = v1.toFloat()
        return when {
            f > 0.0f -> Float16.ONE
            f < 0.0f -> Float16.fromFloat(-1.0f)
            else -> Float16.ZERO
        }
    }
    override fun f32(v1: Float): Float = (if (v1 > 0.0f) 1.0f else 0.0f) - (if (v1 < 0.0f) 1.0f else 0.0f)
    override fun f64(v1: Double): Double = (if (v1 > 0.0) 1.0 else 0.0) - (if (v1 < 0.0) 1.0 else 0.0)
    override fun u8(v1: UByte): UByte = minOf(1u, v1)
    override fun u32(v1: UInt): UInt = minOf(1u, v1)
    override fun i16(v1: Short): Short = ((if (v1 > 0) 1 else 0) - (if (v1 < 0) 1 else 0)).toShort()
    override fun i32(v1: Int): Int = (if (v1 > 0) 1 else 0) - (if (v1 < 0) 1 else 0)
    override fun i64(v1: Long): Long = (if (v1 > 0L) 1L else 0L) - (if (v1 < 0L) 1L else 0L)
    override fun f8e4m3(v1: F8E4M3): F8E4M3 {
        val f = v1.toFloat()
        return when {
            f > 0.0f -> F8E4M3.ONE
            f < 0.0f -> F8E4M3.fromFloat(-1.0f)
            else -> F8E4M3.ZERO
        }
    }
}

