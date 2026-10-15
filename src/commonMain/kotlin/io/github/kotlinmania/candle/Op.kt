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

    public fun opName(): String =
        when (this) {
            ArgMax -> "argmax"
            ArgMin -> "argmin"
            Min -> "min"
            Max -> "max"
            Sum -> "sum"
        }
}

public enum class BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}

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

    public data class Cat(
        val args: List<Tensor>,
        val dim: Int,
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
}

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

    public companion object {
        public fun none(): BackpropOp = None

        public fun new1(arg: Tensor, buildOp: (Tensor) -> Op): BackpropOp = Op1(arg, buildOp)

        public fun new2(lhs: Tensor, rhs: Tensor, buildOp: (Tensor, Tensor) -> Op): BackpropOp =
            Op2(lhs, rhs, buildOp)

        public fun new3(
            t1: Tensor,
            t2: Tensor,
            t3: Tensor,
            buildOp: (Tensor, Tensor, Tensor) -> Op,
        ): BackpropOp = Op3(t1, t2, t3, buildOp)

        public fun new(args: List<Tensor>, buildOp: (List<Tensor>) -> Op): BackpropOp =
            OpN(args, buildOp)
    }
}

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
        0.5 * x * (1.0 + erf(x / sqrt(2.0)))

    public fun geluErf(x: Float): Float = geluErf(x.toDouble()).toFloat()

    public fun gelu(x: Double): Double {
        val c = sqrt(2.0 / PI)
        return 0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x)))
    }

    public fun gelu(x: Float): Float = gelu(x.toDouble()).toFloat()

    public fun silu(x: Double): Double = x / (1.0 + exp(-x))

    public fun silu(x: Float): Float = (x.toDouble() / (1.0 + exp(-x.toDouble()))).toFloat()

    public fun relu(x: Double): Double = if (x > 0.0) x else 0.0

    public fun relu(x: Float): Float = if (x > 0.0f) x else 0.0f
}

public interface UnaryOpT {
    public val name: String
    public val kernel: String
    public fun f32(v1: Float): Float
    public fun f64(v1: Double): Double
    public fun u8(v1: UByte): UByte
    public fun u32(v1: UInt): UInt
    public fun i16(v1: Short): Short
    public fun i32(v1: Int): Int
    public fun i64(v1: Long): Long
}

public interface BinaryOpT {
    public val name: String
    public val kernel: String
    public fun f32(v1: Float, v2: Float): Float
    public fun f64(v1: Double, v2: Double): Double
    public fun u8(v1: UByte, v2: UByte): UByte
    public fun u32(v1: UInt, v2: UInt): UInt
    public fun i16(v1: Short, v2: Short): Short
    public fun i32(v1: Int, v2: Int): Int
    public fun i64(v1: Long, v2: Long): Long
}

public object Add : BinaryOpT {
    override val name: String = "add"
    override val kernel: String = "badd"
    override fun f32(v1: Float, v2: Float): Float = v1 + v2
    override fun f64(v1: Double, v2: Double): Double = v1 + v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 + v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 + v2
    override fun i16(v1: Short, v2: Short): Short = (v1 + v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 + v2
    override fun i64(v1: Long, v2: Long): Long = v1 + v2
}

public object Div : BinaryOpT {
    override val name: String = "div"
    override val kernel: String = "bdiv"
    override fun f32(v1: Float, v2: Float): Float = v1 / v2
    override fun f64(v1: Double, v2: Double): Double = v1 / v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 / v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 / v2
    override fun i16(v1: Short, v2: Short): Short = (v1 / v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 / v2
    override fun i64(v1: Long, v2: Long): Long = v1 / v2
}

public object Mul : BinaryOpT {
    override val name: String = "mul"
    override val kernel: String = "bmul"
    override fun f32(v1: Float, v2: Float): Float = v1 * v2
    override fun f64(v1: Double, v2: Double): Double = v1 * v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 * v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 * v2
    override fun i16(v1: Short, v2: Short): Short = (v1 * v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 * v2
    override fun i64(v1: Long, v2: Long): Long = v1 * v2
}

public object Sub : BinaryOpT {
    override val name: String = "sub"
    override val kernel: String = "bsub"
    override fun f32(v1: Float, v2: Float): Float = v1 - v2
    override fun f64(v1: Double, v2: Double): Double = v1 - v2
    override fun u8(v1: UByte, v2: UByte): UByte = (v1 - v2).toUByte()
    override fun u32(v1: UInt, v2: UInt): UInt = v1 - v2
    override fun i16(v1: Short, v2: Short): Short = (v1 - v2).toShort()
    override fun i32(v1: Int, v2: Int): Int = v1 - v2
    override fun i64(v1: Long, v2: Long): Long = v1 - v2
}

public object Maximum : BinaryOpT {
    override val name: String = "maximum"
    override val kernel: String = "bmaximum"
    override fun f32(v1: Float, v2: Float): Float = maxOf(v1, v2)
    override fun f64(v1: Double, v2: Double): Double = maxOf(v1, v2)
    override fun u8(v1: UByte, v2: UByte): UByte = maxOf(v1, v2)
    override fun u32(v1: UInt, v2: UInt): UInt = maxOf(v1, v2)
    override fun i16(v1: Short, v2: Short): Short = maxOf(v1, v2)
    override fun i32(v1: Int, v2: Int): Int = maxOf(v1, v2)
    override fun i64(v1: Long, v2: Long): Long = maxOf(v1, v2)
}

public object Minimum : BinaryOpT {
    override val name: String = "minimum"
    override val kernel: String = "bminimum"
    override fun f32(v1: Float, v2: Float): Float = minOf(v1, v2)
    override fun f64(v1: Double, v2: Double): Double = minOf(v1, v2)
    override fun u8(v1: UByte, v2: UByte): UByte = minOf(v1, v2)
    override fun u32(v1: UInt, v2: UInt): UInt = minOf(v1, v2)
    override fun i16(v1: Short, v2: Short): Short = minOf(v1, v2)
    override fun i32(v1: Int, v2: Int): Int = minOf(v1, v2)
    override fun i64(v1: Long, v2: Long): Long = minOf(v1, v2)
}

public object Exp : UnaryOpT {
    override val name: String = "exp"
    override val kernel: String = "uexp"
    override fun f32(v1: Float): Float = kotlin.math.exp(v1)
    override fun f64(v1: Double): Double = kotlin.math.exp(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Log : UnaryOpT {
    override val name: String = "log"
    override val kernel: String = "ulog"
    override fun f32(v1: Float): Float = kotlin.math.ln(v1)
    override fun f64(v1: Double): Double = kotlin.math.ln(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Sin : UnaryOpT {
    override val name: String = "sin"
    override val kernel: String = "usin"
    override fun f32(v1: Float): Float = kotlin.math.sin(v1)
    override fun f64(v1: Double): Double = kotlin.math.sin(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Cos : UnaryOpT {
    override val name: String = "cos"
    override val kernel: String = "ucos"
    override fun f32(v1: Float): Float = kotlin.math.cos(v1)
    override fun f64(v1: Double): Double = kotlin.math.cos(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Abs : UnaryOpT {
    override val name: String = "abs"
    override val kernel: String = "uabs"
    override fun f32(v1: Float): Float = kotlin.math.abs(v1)
    override fun f64(v1: Double): Double = kotlin.math.abs(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = kotlin.math.abs(v1.toInt()).toShort()
    override fun i32(v1: Int): Int = kotlin.math.abs(v1)
    override fun i64(v1: Long): Long = kotlin.math.abs(v1)
}

public object Neg : UnaryOpT {
    override val name: String = "neg"
    override val kernel: String = "uneg"
    override fun f32(v1: Float): Float = -v1
    override fun f64(v1: Double): Double = -v1
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = (-v1).toShort()
    override fun i32(v1: Int): Int = -v1
    override fun i64(v1: Long): Long = -v1
}

public object Recip : UnaryOpT {
    override val name: String = "recip"
    override val kernel: String = "urecip"
    override fun f32(v1: Float): Float = 1.0f / v1
    override fun f64(v1: Double): Double = 1.0 / v1
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Sqr : UnaryOpT {
    override val name: String = "sqr"
    override val kernel: String = "usqr"
    override fun f32(v1: Float): Float = v1 * v1
    override fun f64(v1: Double): Double = v1 * v1
    override fun u8(v1: UByte): UByte = (v1 * v1).toUByte()
    override fun u32(v1: UInt): UInt = v1 * v1
    override fun i16(v1: Short): Short = (v1 * v1).toShort()
    override fun i32(v1: Int): Int = v1 * v1
    override fun i64(v1: Long): Long = v1 * v1
}

public object Sqrt : UnaryOpT {
    override val name: String = "sqrt"
    override val kernel: String = "usqrt"
    override fun f32(v1: Float): Float = kotlin.math.sqrt(v1)
    override fun f64(v1: Double): Double = kotlin.math.sqrt(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Gelu : UnaryOpT {
    override val name: String = "gelu"
    override val kernel: String = "ugelu"
    override fun f32(v1: Float): Float = (0.5f * v1 * (1.0f + kotlin.math.tanh(kotlin.math.sqrt(2.0f / kotlin.math.PI.toFloat()) * (v1 + 0.044715f * v1 * v1 * v1))))
    override fun f64(v1: Double): Double = (0.5 * v1 * (1.0 + kotlin.math.tanh(kotlin.math.sqrt(2.0 / kotlin.math.PI) * (v1 + 0.044715 * v1 * v1 * v1))))
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object GeluErf : UnaryOpT {
    override val name: String = "gelu_erf"
    override val kernel: String = "ugelu_erf"
    override fun f32(v1: Float): Float = Gelu.f32(v1)
    override fun f64(v1: Double): Double = Gelu.f64(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Erf : UnaryOpT {
    override val name: String = "erf"
    override val kernel: String = "uerf"
    override fun f32(v1: Float): Float = kotlin.math.tanh(v1)
    override fun f64(v1: Double): Double = kotlin.math.tanh(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Relu : UnaryOpT {
    override val name: String = "relu"
    override val kernel: String = "urelu"
    override fun f32(v1: Float): Float = maxOf(0.0f, v1)
    override fun f64(v1: Double): Double = maxOf(0.0, v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = maxOf(0.toShort(), v1)
    override fun i32(v1: Int): Int = maxOf(0, v1)
    override fun i64(v1: Long): Long = maxOf(0L, v1)
}

public object Silu : UnaryOpT {
    override val name: String = "silu"
    override val kernel: String = "usilu"
    override fun f32(v1: Float): Float = v1 / (1.0f + kotlin.math.exp(-v1))
    override fun f64(v1: Double): Double = v1 / (1.0 + kotlin.math.exp(-v1))
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Tanh : UnaryOpT {
    override val name: String = "tanh"
    override val kernel: String = "utanh"
    override fun f32(v1: Float): Float = kotlin.math.tanh(v1)
    override fun f64(v1: Double): Double = kotlin.math.tanh(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Floor : UnaryOpT {
    override val name: String = "floor"
    override val kernel: String = "ufloor"
    override fun f32(v1: Float): Float = kotlin.math.floor(v1)
    override fun f64(v1: Double): Double = kotlin.math.floor(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Ceil : UnaryOpT {
    override val name: String = "ceil"
    override val kernel: String = "uceil"
    override fun f32(v1: Float): Float = kotlin.math.ceil(v1)
    override fun f64(v1: Double): Double = kotlin.math.ceil(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Round : UnaryOpT {
    override val name: String = "round"
    override val kernel: String = "uround"
    override fun f32(v1: Float): Float = kotlin.math.round(v1)
    override fun f64(v1: Double): Double = kotlin.math.round(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = v1
    override fun i32(v1: Int): Int = v1
    override fun i64(v1: Long): Long = v1
}

public object Sign : UnaryOpT {
    override val name: String = "sign"
    override val kernel: String = "usign"
    override fun f32(v1: Float): Float = kotlin.math.sign(v1)
    override fun f64(v1: Double): Double = kotlin.math.sign(v1)
    override fun u8(v1: UByte): UByte = v1
    override fun u32(v1: UInt): UInt = v1
    override fun i16(v1: Short): Short = kotlin.math.sign(v1.toFloat()).toInt().toShort()
    override fun i32(v1: Int): Int = kotlin.math.sign(v1.toFloat()).toInt()
    override fun i64(v1: Long): Long = kotlin.math.sign(v1.toDouble()).toLong()
}
