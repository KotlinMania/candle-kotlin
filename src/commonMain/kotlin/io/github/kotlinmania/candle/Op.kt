// port-lint: source candle-core/src/op.rs
package io.github.kotlinmania.candle

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

    public companion object {
        public fun new1(arg: Tensor, buildOp: (Tensor) -> Op): BackpropOp = Op1(arg, buildOp)

        public fun new2(lhs: Tensor, rhs: Tensor, buildOp: (Tensor, Tensor) -> Op): BackpropOp =
            Op2(lhs, rhs, buildOp)

        public fun new3(
            t1: Tensor,
            t2: Tensor,
            t3: Tensor,
            buildOp: (Tensor, Tensor, Tensor) -> Op,
        ): BackpropOp = Op3(t1, t2, t3, buildOp)
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
