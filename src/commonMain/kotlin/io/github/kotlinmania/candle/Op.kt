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
    ;

    public interface Target
}

public enum class ReduceOp {
    Sum,
    Min,
    Max,
    ArgMin,
    ArgMax,
    ;

    public fun name(self: ReduceOp = this): String =
        when (self) {
            Self.ArgMax -> "argmax"
            Self.ArgMin -> "argmin"
            Self.Min -> "min"
            Self.Max -> "max"
            Self.Sum -> "sum"
        }

    public val opName: () -> String = { name() }

    public companion object Self
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

// Hardcode the value for sqrt(2/pi)
// https://github.com/huggingface/candle/issues/1982
private const val SQRT_TWO_OVER_PI_F32: Float = 0.7978845608028654f
private const val SQRT_TWO_OVER_PI_F64: Double = 0.7978845608028654
private const val FRAC_1_SQRT_2_F32: Float = 0.7071067811865475f
private const val FRAC_1_SQRT_2_F64: Double = 0.7071067811865476

public object MathOps {
    public val erf: (Double) -> Double = { x -> CpuErf.erfF64(x) }
    public val erfF32: (Float) -> Float = { x -> CpuErf.erfF32(x) }
    public val geluErf: (Double) -> Double = { x -> (CpuErf.erfF64(x * FRAC_1_SQRT_2_F64) + 1.0) * 0.5 * x }
    public val geluErfF32: (Float) -> Float = { x -> (CpuErf.erfF32(x * FRAC_1_SQRT_2_F32) + 1.0f) * 0.5f * x }
    public val gelu: (Double) -> Double = { x -> 0.5 * x * (1.0 + tanh(SQRT_TWO_OVER_PI_F64 * x * (1.0 + 0.044715 * x * x))) }
    public val geluF32: (Float) -> Float = { x -> 0.5f * x * (1.0f + tanh(SQRT_TWO_OVER_PI_F32 * x * (1.0f + 0.044715f * x * x))) }
    public val silu: (Double) -> Double = { x -> x / (1.0 + exp(-x)) }
    public val siluF32: (Float) -> Float = { x -> x / (1.0f + exp(-x)) }
    public val relu: (Double) -> Double = { x -> if (x > 0.0) x else 0.0 }
    public val reluF32: (Float) -> Float = { x -> if (x > 0.0f) x else 0.0f }
}

public interface UnaryOpT {
    public val name: String
    public val kernel: String
    public val Self: UnaryOpT get() = this

    public fun bf16(v1: BF16): BF16
    public fun f16(v1: F16): F16
    public fun f32(v1: Float): Float
    public fun f64(v1: Double): Double
    public fun u8(v1: UByte): UByte
    public fun u32(v1: UInt): UInt
    public fun i16(v1: Short): Short
    public fun i32(v1: Int): Int
    public fun i64(v1: Long): Long
    public fun f8e4m3(v1: F8E4M3): F8E4M3

    public fun bf16Vec(xs: Array<BF16>, ys: Array<BF16>) {
        xs.iter().zip(ys).forEach { (x, y) -> ys[y] = Self.bf16(x) }
    }

    public fun f16Vec(xs: Array<F16>, ys: Array<F16>) {
        xs.iter().zip(ys).forEach { (x, y) -> ys[y] = Self.f16(x) }
    }

    public fun f32Vec(xs: FloatArray, ys: FloatArray) {
        xs.iter().zip(ys).forEach { (x, y) -> ys[y] = Self.f32(x) }
    }

    public fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        xs.iter().zip(ys).forEach { (x, y) -> ys[y] = Self.f64(x) }
    }
}

public interface BinaryOpT {
    public val name: String
    public val kernel: String
    public val Self: BinaryOpT get() = this

    public fun bf16(v1: BF16, v2: BF16): BF16
    public fun f16(v1: F16, v2: F16): F16
    public fun f32(v1: Float, v2: Float): Float
    public fun f64(v1: Double, v2: Double): Double
    public fun u8(v1: UByte, v2: UByte): UByte
    public fun u32(v1: UInt, v2: UInt): UInt
    public fun i16(v1: Short, v2: Short): Short
    public fun i32(v1: Int, v2: Int): Int
    public fun i64(v1: Long, v2: Long): Long
    public fun f8e4m3(v1: F8E4M3, v2: F8E4M3): F8E4M3

    public fun bf16Vec(xs1: Array<BF16>, xs2: Array<BF16>, ys: Array<BF16>) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.bf16(a, b) }
    }

    public fun f16Vec(xs1: Array<F16>, xs2: Array<F16>, ys: Array<F16>) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.f16(a, b) }
    }

    public fun f32Vec(xs1: FloatArray, xs2: FloatArray, ys: FloatArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.f32(a, b) }
    }

    public fun f64Vec(xs1: DoubleArray, xs2: DoubleArray, ys: DoubleArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.f64(a, b) }
    }

    public fun u8Vec(xs1: UByteArray, xs2: UByteArray, ys: UByteArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.u8(a, b) }
    }

    public fun u32Vec(xs1: UIntArray, xs2: UIntArray, ys: UIntArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.u32(a, b) }
    }

    public fun i16Vec(xs1: ShortArray, xs2: ShortArray, ys: ShortArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.i16(a, b) }
    }

    public fun i32Vec(xs1: IntArray, xs2: IntArray, ys: IntArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.i32(a, b) }
    }

    public fun i64Vec(xs1: LongArray, xs2: LongArray, ys: LongArray) {
        xs1.iter()
            .zip(xs2)
            .zip(ys)
            .forEach { (a, b, y) -> ys[y] = Self.i64(a, b) }
    }

    // Scalar-broadcast variants: ys[i] = f(xs[i], scalar).
    // Used by binaryMapVec for the (1,0) inner-stride branch where one tensor
    // is contiguous and the other broadcasts a single value.
    public fun bf16ScalarVec(scalar: BF16, xs: Array<BF16>, ys: Array<BF16>) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.bf16(x, scalar) }
    }

    public fun f16ScalarVec(scalar: F16, xs: Array<F16>, ys: Array<F16>) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.f16(x, scalar) }
    }

    public fun f32ScalarVec(scalar: Float, xs: FloatArray, ys: FloatArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.f32(x, scalar) }
    }

    public fun f64ScalarVec(scalar: Double, xs: DoubleArray, ys: DoubleArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.f64(x, scalar) }
    }

    public fun u8ScalarVec(scalar: UByte, xs: UByteArray, ys: UByteArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.u8(x, scalar) }
    }

    public fun u32ScalarVec(scalar: UInt, xs: UIntArray, ys: UIntArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.u32(x, scalar) }
    }

    public fun i16ScalarVec(scalar: Short, xs: ShortArray, ys: ShortArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.i16(x, scalar) }
    }

    public fun i32ScalarVec(scalar: Int, xs: IntArray, ys: IntArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.i32(x, scalar) }
    }

    public fun i64ScalarVec(scalar: Long, xs: LongArray, ys: LongArray) {
        xs.iter()
            .zip(ys)
            .forEach { (x, y) -> ys[y] = Self.i64(x, scalar) }
    }
}

private class MacroBinaryOp(
    override val name: String,
    override val kernel: String,
    private val f32Fn: (Float, Float) -> Float,
    private val f64Fn: (Double, Double) -> Double,
    private val i64Fn: (Long, Long) -> Long,
) : BinaryOpT {
    override val bf16: (BF16, BF16) -> BF16 = { v1, v2 -> BF16.fromFloat(f32Fn(v1.toFloat(), v2.toFloat())) }
    override val f16: (F16, F16) -> F16 = { v1, v2 -> F16.fromFloat(f32Fn(v1.toFloat(), v2.toFloat())) }
    override val f32: (Float, Float) -> Float = f32Fn
    override val f64: (Double, Double) -> Double = f64Fn
    override val u8: (UByte, UByte) -> UByte = { v1, v2 -> i64Fn(v1.toLong(), v2.toLong()).toUByte() }
    override val u32: (UInt, UInt) -> UInt = { v1, v2 -> i64Fn(v1.toLong(), v2.toLong()).toUInt() }
    override val i16: (Short, Short) -> Short = { v1, v2 -> i64Fn(v1.toLong(), v2.toLong()).toShort() }
    override val i32: (Int, Int) -> Int = { v1, v2 -> i64Fn(v1.toLong(), v2.toLong()).toInt() }
    override val i64: (Long, Long) -> Long = i64Fn
    override val f8e4m3: (F8E4M3, F8E4M3) -> F8E4M3 = { v1, v2 -> F8E4M3.fromFloat(f32Fn(v1.toFloat(), v2.toFloat())) }
}

private class MacroUnaryOp(
    override val name: String,
    override val kernel: String,
    private val f32Fn: (Float) -> Float,
    private val f64Fn: (Double) -> Double,
) : UnaryOpT {
    override val bf16: (BF16) -> BF16 = { v -> BF16.fromFloat(f32Fn(v.toFloat())) }
    override val f16: (F16) -> F16 = { v -> F16.fromFloat(f32Fn(v.toFloat())) }
    override val f32: (Float) -> Float = f32Fn
    override val f64: (Double) -> Double = f64Fn
    override val u8: (UByte) -> UByte = { v -> v }
    override val u32: (UInt) -> UInt = { v -> v }
    override val i16: (Short) -> Short = { v -> v }
    override val i32: (Int) -> Int = { v -> v }
    override val i64: (Long) -> Long = { v -> v }
    override val f8e4m3: (F8E4M3) -> F8E4M3 = { v -> F8E4M3.fromFloat(f32Fn(v.toFloat())) }
}

public object Add : BinaryOpT by MacroBinaryOp("add", "badd", { a, b -> a + b }, { a, b -> a + b }, { a, b -> a + b })
public object Div : BinaryOpT by MacroBinaryOp("div", "bdiv", { a, b -> a / b }, { a, b -> a / b }, { a, b -> a / b })
public object Mul : BinaryOpT by MacroBinaryOp("mul", "bmul", { a, b -> a * b }, { a, b -> a * b }, { a, b -> a * b })
public object Sub : BinaryOpT by MacroBinaryOp("sub", "bsub", { a, b -> a - b }, { a, b -> a - b }, { a, b -> a - b })
public object Maximum : BinaryOpT by MacroBinaryOp("maximum", "bmaximum", { a, b -> if (a < b) b else a }, { a, b -> if (a < b) b else a }, { a, b -> if (a < b) b else a })
public object Minimum : BinaryOpT by MacroBinaryOp("minimum", "bminimum", { a, b -> if (a > b) b else a }, { a, b -> if (a > b) b else a }, { a, b -> if (a > b) b else a })
public object Exp : UnaryOpT by MacroUnaryOp("exp", "uexp", { v -> exp(v) }, { v -> exp(v) })
public object Log : UnaryOpT by MacroUnaryOp("log", "ulog", { v -> kotlin.math.ln(v) }, { v -> kotlin.math.ln(v) })
public object Sin : UnaryOpT by MacroUnaryOp("sin", "usin", { v -> kotlin.math.sin(v) }, { v -> kotlin.math.sin(v) })
public object Cos : UnaryOpT by MacroUnaryOp("cos", "ucos", { v -> kotlin.math.cos(v) }, { v -> kotlin.math.cos(v) })
public object Neg : UnaryOpT by MacroUnaryOp("neg", "uneg", { v -> -v }, { v -> -v })
public object Recip : UnaryOpT by MacroUnaryOp("recip", "urecip", { v -> 1.0f / v }, { v -> 1.0 / v })
public object Sqr : UnaryOpT by MacroUnaryOp("sqr", "usqr", { v -> v * v }, { v -> v * v })
public object Sqrt : UnaryOpT by MacroUnaryOp("sqrt", "usqrt", { v -> sqrt(v) }, { v -> sqrt(v) })
public object Tanh : UnaryOpT by MacroUnaryOp("tanh", "utanh", { v -> tanh(v) }, { v -> tanh(v) })

/**
 * Tanh based approximation of the `gelu` operation
 * GeluErf is the more precise one.
 * See Wikipedia Activation function Comparison of activation functions
 */
public object Gelu : UnaryOpT {
    override val name: String = "gelu"
    override val kernel: String = "ugelu"
    override fun bf16(v: BF16): BF16 =
        BF16.fromF32Const(0.5f) *
            v *
            (
                BF16.ONE +
                    BF16.tanh(
                        BF16.fromF32Const(SQRT_TWO_OVER_PI_F32) *
                            v *
                            (BF16.ONE + BF16.fromF32Const(0.044715f) * v * v),
                    )
            )
    override fun f16(v: F16): F16 =
        F16.fromF32Const(0.5f) *
            v *
            (
                F16.ONE +
                    F16.tanh(
                        F16.fromF32Const(SQRT_TWO_OVER_PI_F32) *
                            v *
                            (F16.ONE + F16.fromF32Const(0.044715f) * v * v),
                    )
            )
    override fun f32(v: Float): Float =
        0.5f * v * (1.0f + Float.tanh(SQRT_TWO_OVER_PI_F32 * v * (1.0f + 0.044715f * v * v)))
    override fun f64(v: Double): Double =
        0.5 * v * (1.0 + Double.tanh(SQRT_TWO_OVER_PI_F64 * v * (1.0 + 0.044715 * v * v)))
    override fun u8(v: UByte): UByte = 0u
    override fun u32(v: UInt): UInt = 0u
    override fun i16(v: Short): Short = 0
    override fun i32(v: Int): Int = 0
    override fun i64(v: Long): Long = 0L
    override fun f8e4m3(v: F8E4M3): F8E4M3 =
        F8E4M3.fromF32(0.5f) *
            v *
            (
                F8E4M3.ONE +
                    F8E4M3.tanh(
                        F8E4M3.fromF32(SQRT_TWO_OVER_PI_F32) *
                            v *
                            (F8E4M3.ONE + F8E4M3.fromF32(0.044715f) * v * v),
                    )
            )

    public object MklImpl {
        public fun f32Vec(xs: FloatArray, ys: FloatArray) {
            Mkl.vsGelu(xs, ys)
        }

        public fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
            Mkl.vdGelu(xs, ys)
        }
    }

    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        Accelerate.vsGelu(xs, ys)
    }

    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        Accelerate.vdGelu(xs, ys)
    }
}

/**
 * `erf` operation
 * See Wikipedia Error function
 */
public object Erf : UnaryOpT {
    override val name: String = "erf"
    override val kernel: String = "uerf"
    override fun bf16(v: BF16): BF16 = BF16.fromF64(Self.f64(v.toF64()))
    override fun f16(v: F16): F16 = F16.fromF64(Self.f64(v.toF64()))
    override fun f32(v: Float): Float = CpuErf.erfF32(v)
    override fun f64(v: Double): Double = CpuErf.erfF64(v)
    override fun u8(v: UByte): UByte = 0u
    override fun u32(v: UInt): UInt = 0u
    override fun i16(v: Short): Short = 0
    override fun i32(v: Int): Int = 0
    override fun i64(v: Long): Long = 0L
    override fun f8e4m3(v: F8E4M3): F8E4M3 = F8E4M3.fromF64(Self.f64(v.toF64()))
}

/**
 * Silu operation
 */
public object Silu : UnaryOpT {
    override val name: String = "silu"
    override val kernel: String = "usilu"
    override fun bf16(v: BF16): BF16 = v / (BF16.ONE + (-v).exp())
    override fun f16(v: F16): F16 = v / (F16.ONE + (-v).exp())
    override fun f32(v: Float): Float = v / (1.0f + (-v).exp())
    override fun f64(v: Double): Double = v / (1.0 + (-v).exp())
    override fun u8(v: UByte): UByte = 0u
    override fun u32(v: UInt): UInt = 0u
    override fun i16(v: Short): Short = 0
    override fun i32(v: Int): Int = 0
    override fun i64(v: Long): Long = 0L
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v / (F8E4M3.ONE + (-v).exp())

    public object MklImpl {
        public fun f32Vec(xs: FloatArray, ys: FloatArray) {
            Mkl.vsSilu(xs, ys)
        }

        public fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
            Mkl.vdSilu(xs, ys)
        }
    }

    override fun f32Vec(xs: FloatArray, ys: FloatArray) {
        Accelerate.vsSilu(xs, ys)
    }

    override fun f64Vec(xs: DoubleArray, ys: DoubleArray) {
        Accelerate.vdSilu(xs, ys)
    }
}

public object Abs : UnaryOpT {
    override val name: String = "abs"
    override val kernel: String = "uabs"
    override fun bf16(v: BF16): BF16 = v.abs()
    override fun f16(v: F16): F16 = v.abs()
    override fun f32(v: Float): Float = v.abs()
    override fun f64(v: Double): Double = v.abs()
    override fun u8(v: UByte): UByte = v
    override fun u32(v: UInt): UInt = v
    override fun i16(v: Short): Short = v.abs()
    override fun i32(v: Int): Int = v.abs()
    override fun i64(v: Long): Long = v.abs()
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v.abs()
}

public object Ceil : UnaryOpT {
    override val name: String = "ceil"
    override val kernel: String = "uceil"
    override fun bf16(v: BF16): BF16 = v.ceil()
    override fun f16(v: F16): F16 = v.ceil()
    override fun f32(v: Float): Float = v.ceil()
    override fun f64(v: Double): Double = v.ceil()
    override fun u8(v: UByte): UByte = v
    override fun u32(v: UInt): UInt = v
    override fun i16(v: Short): Short = v
    override fun i32(v: Int): Int = v
    override fun i64(v: Long): Long = v
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v.ceil()
}

public object Floor : UnaryOpT {
    override val name: String = "floor"
    override val kernel: String = "ufloor"
    override fun bf16(v: BF16): BF16 = v.floor()
    override fun f16(v: F16): F16 = v.floor()
    override fun f32(v: Float): Float = v.floor()
    override fun f64(v: Double): Double = v.floor()
    override fun u8(v: UByte): UByte = v
    override fun u32(v: UInt): UInt = v
    override fun i16(v: Short): Short = v
    override fun i32(v: Int): Int = v
    override fun i64(v: Long): Long = v
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v.floor()
}

public object Round : UnaryOpT {
    override val name: String = "round"
    override val kernel: String = "uround"
    override fun bf16(v: BF16): BF16 = v.round()
    override fun f16(v: F16): F16 = v.round()
    override fun f32(v: Float): Float = v.round()
    override fun f64(v: Double): Double = v.round()
    override fun u8(v: UByte): UByte = v
    override fun u32(v: UInt): UInt = v
    override fun i16(v: Short): Short = v
    override fun i32(v: Int): Int = v
    override fun i64(v: Long): Long = v
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v.round()
}

public object GeluErf : UnaryOpT {
    override val name: String = "gelu_erf"
    override val kernel: String = "ugelu_erf"
    override fun bf16(v: BF16): BF16 = BF16.fromF64(Self.f64(v.toF64()))
    override fun f16(v: F16): F16 = F16.fromF64(Self.f64(v.toF64()))
    override fun f32(v: Float): Float =
        (CpuErf.erfF32(v * FRAC_1_SQRT_2_F32) + 1.0f) * 0.5f * v
    override fun f64(v: Double): Double =
        (CpuErf.erfF64(v * FRAC_1_SQRT_2_F64) + 1.0) * 0.5 * v
    override fun u8(v: UByte): UByte = 0u
    override fun u32(v: UInt): UInt = 0u
    override fun i16(v: Short): Short = 0
    override fun i32(v: Int): Int = 0
    override fun i64(v: Long): Long = 0L
    override fun f8e4m3(v: F8E4M3): F8E4M3 = F8E4M3.fromF32(Self.f32(v.toF32()))
}

public object Relu : UnaryOpT {
    override val name: String = "relu"
    override val kernel: String = "urelu"
    override fun bf16(v: BF16): BF16 = v.max(BF16.ZERO)
    override fun f16(v: F16): F16 = v.max(F16.ZERO)
    override fun f32(v: Float): Float = v.max(0f)
    override fun f64(v: Double): Double = v.max(0.0)
    override fun u8(v: UByte): UByte = v
    override fun u32(v: UInt): UInt = v
    override fun i16(v: Short): Short = v.max(0)
    override fun i32(v: Int): Int = v.max(0)
    override fun i64(v: Long): Long = v.max(0)
    override fun f8e4m3(v: F8E4M3): F8E4M3 = v.max(F8E4M3.ZERO)
}

/**
 * `BackpropOp` is a wrapper around `Option<Op>`. The main goal is to ensure that dependencies are
 * properly checked when creating a new value
 */
public open class BackpropOp(
    public val inner: Op?,
) {
    public object None : BackpropOp(null)

    public val toOp: () -> Op? = { inner }

    public fun isNone(self: BackpropOp = this): Boolean = self.inner.isNone()

    public fun deref(self: BackpropOp = this): Op? = self.inner

    public companion object {
        public interface Target

        public val Self: (Op?) -> BackpropOp = { op -> if (op == null) None else BackpropOp(op) }

        public fun none(): BackpropOp = Self(null)

        public fun new1(arg: Tensor, f: (Tensor) -> Op): BackpropOp {
            val op = if (arg.trackOp()) {
                Some(f(arg.clone()))
            } else {
                null
            }
            return Self(op)
        }

        public fun new2(arg1: Tensor, arg2: Tensor, f: (Tensor, Tensor) -> Op): BackpropOp {
            val op = if (arg1.trackOp() || arg2.trackOp()) {
                Some(f(arg1.clone(), arg2.clone()))
            } else {
                null
            }
            return Self(op)
        }

        public fun new3(
            arg1: Tensor,
            arg2: Tensor,
            arg3: Tensor,
            f: (Tensor, Tensor, Tensor) -> Op,
        ): BackpropOp {
            val op = if (arg1.trackOp() || arg2.trackOp() || arg3.trackOp()) {
                Some(f(arg1.clone(), arg2.clone(), arg3.clone()))
            } else {
                null
            }
            return Self(op)
        }

        public fun new(args: List<Tensor>, f: (List<Tensor>) -> Op): BackpropOp {
            val op = if (args.iter().any { arg -> arg.asRef().trackOp() }) {
                val args: List<Tensor> = args.iter().map { arg -> arg.asRef().clone() }.collect()
                Some(f(args))
            } else {
                null
            }
            return Self(op)
        }
    }
}

public object Sign : UnaryOpT {
    override val name: String = "sign"
    override val kernel: String = "usign"
    override fun bf16(v: BF16): BF16 =
        BF16.from(v > BF16.ZERO) - BF16.from(v < BF16.ZERO)
    override fun f16(v: F16): F16 =
        F16.from(v > F16.ZERO) - F16.from(v < F16.ZERO)
    override fun f32(v: Float): Float =
        Float.from(v > 0f) - Float.from(v < 0f)
    override fun f64(v: Double): Double =
        Double.from(v > 0.0) - Double.from(v < 0.0)
    override fun u8(v: UByte): UByte = UByte.min(1u, v)
    override fun u32(v: UInt): UInt = UInt.min(1u, v)
    override fun i16(v: Short): Short = (v > 0).toI16() - (v < 0).toI16()
    override fun i32(v: Int): Int = (v > 0).toI32() - (v < 0).toI32()
    override fun i64(v: Long): Long = (v > 0L).toI64() - (v < 0L).toI64()
    override fun f8e4m3(v: F8E4M3): F8E4M3 =
        if (v > F8E4M3.ZERO) {
            F8E4M3.ONE
        } else if (v < F8E4M3.ZERO) {
            -F8E4M3.ONE
        } else {
            F8E4M3.ZERO
        }
}

