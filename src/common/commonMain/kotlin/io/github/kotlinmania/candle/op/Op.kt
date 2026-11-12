// port-lint: source op.rs
package io.github.kotlinmania.candle.op

import io.github.kotlinmania.candle.cpu.erf.erfF32
import io.github.kotlinmania.candle.cpu.erf.erfF64
import io.github.kotlinmania.candle.customop.CustomOp1
import io.github.kotlinmania.candle.customop.CustomOp2
import io.github.kotlinmania.candle.customop.CustomOp3
import io.github.kotlinmania.candle.mkl.vdCos
import io.github.kotlinmania.candle.mkl.vdExp
import io.github.kotlinmania.candle.mkl.vdGelu
import io.github.kotlinmania.candle.mkl.vdLn
import io.github.kotlinmania.candle.mkl.vdSilu
import io.github.kotlinmania.candle.mkl.vdSin
import io.github.kotlinmania.candle.mkl.vdSqr
import io.github.kotlinmania.candle.mkl.vdSqrt
import io.github.kotlinmania.candle.mkl.vdTanh
import io.github.kotlinmania.candle.mkl.vsCos
import io.github.kotlinmania.candle.mkl.vsExp
import io.github.kotlinmania.candle.mkl.vsGelu
import io.github.kotlinmania.candle.mkl.vsLn
import io.github.kotlinmania.candle.mkl.vsSilu
import io.github.kotlinmania.candle.mkl.vsSin
import io.github.kotlinmania.candle.mkl.vsSqr
import io.github.kotlinmania.candle.mkl.vsSqrt
import io.github.kotlinmania.candle.mkl.vsTanh
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.numtraits.float.Float
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

// Tensor Operation Enums and Traits

public typealias Target = Op?

public enum class CmpOp(
    public val value: UInt,
) {
    Eq(0u),
    Ne(1u),
    Le(2u),
    Ge(3u),
    Lt(4u),
    Gt(5u),
}

public enum class ReduceOp(
    public val value: UInt,
) {
    Sum(0u),
    Min(1u),
    Max(2u),
    ArgMin(3u),
    ArgMax(4u),
    ;

    internal fun name(): String =
        when (this) {
            ReduceOp.ArgMax -> "argmax"
            ReduceOp.ArgMin -> "argmin"
            ReduceOp.Min -> "min"
            ReduceOp.Max -> "max"
            ReduceOp.Sum -> "sum"
        }
}

// These ops return the same type as their input type.
public enum class BinaryOp(
    public val value: UInt,
) {
    Add(0u),
    Mul(1u),
    Sub(2u),
    Div(3u),
    Maximum(4u),
    Minimum(5u),
}

// Unary ops with no argument
public enum class UnaryOp(
    public val value: UInt,
) {
    Exp(0u),
    Log(1u),
    Sin(2u),
    Cos(3u),
    Abs(4u),
    Neg(5u),
    Recip(6u),
    Sqr(7u),
    Sqrt(8u),
    Gelu(9u),
    GeluErf(10u),
    Erf(11u),
    Relu(12u),
    Silu(13u),
    Tanh(14u),
    Floor(15u),
    Ceil(16u),
    Round(17u),
    Sign(18u),
}

public sealed class Op {
    public data class Binary(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: BinaryOp,
    ) : Op()
    public data class Unary(
        public var field0: Tensor,
        public var field1: UnaryOp,
    ) : Op()
    public data class Cmp(
        public var field0: Tensor,
        public var field1: CmpOp,
    ) : Op()

    // The third argument is the reduced shape with `keepdim = true`.
    public data class Reduce(
        public var field0: Tensor,
        public var field1: ReduceOp,
        public var field2: List<Int>,
    ) : Op()
    public data class Matmul(
        public var field0: Tensor,
        public var field1: Tensor,
    ) : Op()
    public data class Gather(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Int,
    ) : Op()
    public data class Scatter(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Tensor,
        public var field3: Int,
    ) : Op()
    public data class ScatterAdd(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Tensor,
        public var field3: Int,
    ) : Op()
    public data class IndexSelect(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Int,
    ) : Op()
    public data class IndexAdd(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Tensor,
        public var field3: Int,
    ) : Op()
    public data class WhereCond(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Tensor,
    ) : Op()
    public data class Conv1D(
        public var arg: Tensor,
        public var kernel: Tensor,
        public var padding: Int,
        public var stride: Int,
        public var dilation: Int,
    ) : Op()
    public data class ConvTranspose1D(
        public var arg: Tensor,
        public var kernel: Tensor,
        public var padding: Int,
        public var outputPadding: Int,
        public var stride: Int,
        public var dilation: Int,
    ) : Op()
    public data class Conv2D(
        public var arg: Tensor,
        public var kernel: Tensor,
        public var padding: Int,
        public var stride: Int,
        public var dilation: Int,
    ) : Op()
    public data class ConvTranspose2D(
        public var arg: Tensor,
        public var kernel: Tensor,
        public var padding: Int,
        public var outputPadding: Int,
        public var stride: Int,
        public var dilation: Int,
    ) : Op()
    public data class AvgPool2D(
        public var arg: Tensor,
        public var kernelSize: Pair<Int, Int>,
        public var stride: Pair<Int, Int>,
    ) : Op()
    public data class MaxPool2D(
        public var arg: Tensor,
        public var kernelSize: Pair<Int, Int>,
        public var stride: Pair<Int, Int>,
    ) : Op()
    public data class UpsampleNearest1D(
        public var arg: Tensor,
        public var targetSize: Int,
    ) : Op()
    public data class UpsampleNearest2D(
        public var arg: Tensor,
        public var targetH: Int,
        public var targetW: Int,
    ) : Op()
    public data class UpsampleBilinear2D(
        public var arg: Tensor,
        public var targetH: Int,
        public var targetW: Int,
        public var alignCorners: Boolean,
    ) : Op()
    public data class Cat(
        public var field0: List<Tensor>,
        public var field1: Int,
    ) : Op()

    // add is currently unused.
    public data class Affine(
        public var arg: Tensor,
        public var mul: Double,
        public var add: Double,
    ) : Op()
    public data class ToDType(
        public var value: Tensor,
    ) : Op()
    public data class Copy(
        public var value: Tensor,
    ) : Op()
    public data class Broadcast(
        public var value: Tensor,
    ) : Op()
    public data class Narrow(
        public var field0: Tensor,
        public var field1: Int,
        public var field2: Int,
        public var field3: Int,
    ) : Op()
    public data class SliceScatter0(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Int,
    ) : Op()
    public data class Reshape(
        public var value: Tensor,
    ) : Op()
    public data class ToDevice(
        public var value: Tensor,
    ) : Op()
    public data class Transpose(
        public var field0: Tensor,
        public var field1: Int,
        public var field2: Int,
    ) : Op()
    public data class Permute(
        public var field0: Tensor,
        public var field1: List<Int>,
    ) : Op()
    public data class Elu(
        public var field0: Tensor,
        public var field1: Double,
    ) : Op()
    public data class Powf(
        public var field0: Tensor,
        public var field1: Double,
    ) : Op()
    public data class CustomOp1(
        public var field0: Tensor,
        public var field1: CustomOp1,
    ) : Op()
    public data class CustomOp2(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: CustomOp2,
    ) : Op()
    public data class CustomOp3(
        public var field0: Tensor,
        public var field1: Tensor,
        public var field2: Tensor,
        public var field3: CustomOp3,
    ) : Op()
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

    public fun bf16Vec(xs: List<Bf16>, ys: MutableList<Bf16>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = bf16(xs[i])
        }
    }

    public fun f16Vec(xs: List<F16>, ys: MutableList<F16>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f16(xs[i])
        }
    }

    public fun f32Vec(xs: List<Float>, ys: MutableList<Float>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f32(xs[i])
        }
    }

    public fun f64Vec(xs: List<Double>, ys: MutableList<Double>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f64(xs[i])
        }
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

    public fun bf16Vec(xs1: List<Bf16>, xs2: List<Bf16>, ys: MutableList<Bf16>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = bf16(xs1[i], xs2[i])
        }
    }

    public fun f16Vec(xs1: List<F16>, xs2: List<F16>, ys: MutableList<F16>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = f16(xs1[i], xs2[i])
        }
    }

    public fun f32Vec(xs1: List<Float>, xs2: List<Float>, ys: MutableList<Float>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = f32(xs1[i], xs2[i])
        }
    }

    public fun f64Vec(xs1: List<Double>, xs2: List<Double>, ys: MutableList<Double>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = f64(xs1[i], xs2[i])
        }
    }

    public fun u8Vec(xs1: List<UByte>, xs2: List<UByte>, ys: MutableList<UByte>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = u8(xs1[i], xs2[i])
        }
    }

    public fun u32Vec(xs1: List<UInt>, xs2: List<UInt>, ys: MutableList<UInt>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = u32(xs1[i], xs2[i])
        }
    }

    public fun i16Vec(xs1: List<Short>, xs2: List<Short>, ys: MutableList<Short>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = i16(xs1[i], xs2[i])
        }
    }

    public fun i32Vec(xs1: List<Int>, xs2: List<Int>, ys: MutableList<Int>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = i32(xs1[i], xs2[i])
        }
    }

    public fun i64Vec(xs1: List<Long>, xs2: List<Long>, ys: MutableList<Long>) {
        for (i in 0 until minOf(xs1.size, xs2.size, ys.size)) {
            ys[i] = i64(xs1[i], xs2[i])
        }
    }

    public fun bf16ScalarVec(scalar: Bf16, xs: List<Bf16>, ys: MutableList<Bf16>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = bf16(xs[i], scalar)
        }
    }

    public fun f16ScalarVec(scalar: F16, xs: List<F16>, ys: MutableList<F16>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f16(xs[i], scalar)
        }
    }

    public fun f32ScalarVec(scalar: Float, xs: List<Float>, ys: MutableList<Float>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f32(xs[i], scalar)
        }
    }

    public fun f64ScalarVec(scalar: Double, xs: List<Double>, ys: MutableList<Double>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = f64(xs[i], scalar)
        }
    }

    public fun u8ScalarVec(scalar: UByte, xs: List<UByte>, ys: MutableList<UByte>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = u8(xs[i], scalar)
        }
    }

    public fun u32ScalarVec(scalar: UInt, xs: List<UInt>, ys: MutableList<UInt>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = u32(xs[i], scalar)
        }
    }

    public fun i16ScalarVec(scalar: Short, xs: List<Short>, ys: MutableList<Short>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = i16(xs[i], scalar)
        }
    }

    public fun i32ScalarVec(scalar: Int, xs: List<Int>, ys: MutableList<Int>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = i32(xs[i], scalar)
        }
    }

    public fun i64ScalarVec(scalar: Long, xs: List<Long>, ys: MutableList<Long>) {
        for (i in 0 until minOf(xs.size, ys.size)) {
            ys[i] = i64(xs[i], scalar)
        }
    }
}

public class Add : BinaryOpT<Add> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Add> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Add> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Add =
                    Add()
                override fun writeNative(addr: Long, value: Add) {
                }
            }

        public val NAME: String = "add"

        public val KERNEL: String = "badd"

        public val V: Add = Add()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 = { v11, v21 -> v11 + v21(v11, v21) }

        public fun f16(v1: F16, v2: F16): F16 = { v11, v21 -> v11 + v21(v11, v21) }

        public fun f32(v1: Float, v2: Float): Float = { v11, v21 -> v11 + v21(v11, v21) }

        public fun f64(v1: Double, v2: Double): Double = { v11, v21 -> v11 + v21(v11, v21) }

        public fun u8(v1: UByte, v2: UByte): UByte = { v11, v21 -> v11 + v21(v11, v21) }

        public fun u32(v1: UInt, v2: UInt): UInt = { v11, v21 -> v11 + v21(v11, v21) }

        public fun i16(v1: Short, v2: Short): Short = { v11, v21 -> v11 + v21(v11, v21) }

        public fun i32(v1: Int, v2: Int): Int = { v11, v21 -> v11 + v21(v11, v21) }

        public fun i64(v1: Long, v2: Long): Long = { v11, v21 -> v11 + v21(v11, v21) }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 = { v11, v21 -> v11 + v21(v11, v21) }

        public fun f32Vec(lhs: List<Float>, rhs: List<Float>, res: List<Float>) {
            Float.vecAdd(lhs, rhs, res)
        }

        public fun f64Vec(lhs: List<Double>, rhs: List<Double>, res: List<Double>) {
            Double.vecAdd(lhs, rhs, res)
        }

        public fun bf16Vec(lhs: List<Bf16>, rhs: List<Bf16>, res: List<Bf16>) {
            Bf16.vecAdd(lhs, rhs, res)
        }

        public fun f16Vec(lhs: List<F16>, rhs: List<F16>, res: List<F16>) {
            F16.vecAdd(lhs, rhs, res)
        }

        public fun bf16ScalarVec(scalar: Bf16, xs: List<Bf16>, ys: List<Bf16>) {
            Bf16.scalarAdd(scalar, xs, ys)
        }

        public fun f16ScalarVec(scalar: F16, xs: List<F16>, ys: List<F16>) {
            F16.scalarAdd(scalar, xs, ys)
        }

        public fun f32ScalarVec(scalar: Float, xs: List<Float>, ys: List<Float>) {
            Float.scalarAdd(scalar, xs, ys)
        }

        public fun f64ScalarVec(scalar: Double, xs: List<Double>, ys: List<Double>) {
            Double.scalarAdd(scalar, xs, ys)
        }
    }
}

public class Div : BinaryOpT<Div> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Div> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Div> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Div =
                    Div()
                override fun writeNative(addr: Long, value: Div) {
                }
            }

        public val NAME: String = "div"

        public val KERNEL: String = "bdiv"

        public val V: Div = Div()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 = { v11, v21 -> v11 / v21(v11, v21) }

        public fun f16(v1: F16, v2: F16): F16 = { v11, v21 -> v11 / v21(v11, v21) }

        public fun f32(v1: Float, v2: Float): Float = { v11, v21 -> v11 / v21(v11, v21) }

        public fun f64(v1: Double, v2: Double): Double = { v11, v21 -> v11 / v21(v11, v21) }

        public fun u8(v1: UByte, v2: UByte): UByte = { v11, v21 -> v11 / v21(v11, v21) }

        public fun u32(v1: UInt, v2: UInt): UInt = { v11, v21 -> v11 / v21(v11, v21) }

        public fun i16(v1: Short, v2: Short): Short = { v11, v21 -> v11 / v21(v11, v21) }

        public fun i32(v1: Int, v2: Int): Int = { v11, v21 -> v11 / v21(v11, v21) }

        public fun i64(v1: Long, v2: Long): Long = { v11, v21 -> v11 / v21(v11, v21) }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 = { v11, v21 -> v11 / v21(v11, v21) }
    }
}

public class Mul : BinaryOpT<Mul> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Mul> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Mul> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Mul =
                    Mul()
                override fun writeNative(addr: Long, value: Mul) {
                }
            }

        public val NAME: String = "mul"

        public val KERNEL: String = "bmul"

        public val V: Mul = Mul()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 = { v11, v21 -> v11 * v21(v11, v21) }

        public fun f16(v1: F16, v2: F16): F16 = { v11, v21 -> v11 * v21(v11, v21) }

        public fun f32(v1: Float, v2: Float): Float = { v11, v21 -> v11 * v21(v11, v21) }

        public fun f64(v1: Double, v2: Double): Double = { v11, v21 -> v11 * v21(v11, v21) }

        public fun u8(v1: UByte, v2: UByte): UByte = { v11, v21 -> v11 * v21(v11, v21) }

        public fun u32(v1: UInt, v2: UInt): UInt = { v11, v21 -> v11 * v21(v11, v21) }

        public fun i16(v1: Short, v2: Short): Short = { v11, v21 -> v11 * v21(v11, v21) }

        public fun i32(v1: Int, v2: Int): Int = { v11, v21 -> v11 * v21(v11, v21) }

        public fun i64(v1: Long, v2: Long): Long = { v11, v21 -> v11 * v21(v11, v21) }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 = { v11, v21 -> v11 * v21(v11, v21) }

        public fun f32Vec(lhs: List<Float>, rhs: List<Float>, res: List<Float>) {
            Float.vecMul(lhs, rhs, res)
        }

        public fun f64Vec(lhs: List<Double>, rhs: List<Double>, res: List<Double>) {
            Double.vecMul(lhs, rhs, res)
        }

        public fun bf16Vec(lhs: List<Bf16>, rhs: List<Bf16>, res: List<Bf16>) {
            Bf16.vecMul(lhs, rhs, res)
        }

        public fun f16Vec(lhs: List<F16>, rhs: List<F16>, res: List<F16>) {
            F16.vecMul(lhs, rhs, res)
        }

        public fun bf16ScalarVec(scalar: Bf16, xs: List<Bf16>, ys: List<Bf16>) {
            Bf16.scalarMul(scalar, xs, ys)
        }

        public fun f16ScalarVec(scalar: F16, xs: List<F16>, ys: List<F16>) {
            F16.scalarMul(scalar, xs, ys)
        }

        public fun f32ScalarVec(scalar: Float, xs: List<Float>, ys: List<Float>) {
            Float.scalarMul(scalar, xs, ys)
        }

        public fun f64ScalarVec(scalar: Double, xs: List<Double>, ys: List<Double>) {
            Double.scalarMul(scalar, xs, ys)
        }
    }
}

public class Sub : BinaryOpT<Sub> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sub> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sub> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Sub =
                    Sub()
                override fun writeNative(addr: Long, value: Sub) {
                }
            }

        public val NAME: String = "sub"

        public val KERNEL: String = "bsub"

        public val V: Sub = Sub()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 = { v11, v21 -> v11 - v21(v11, v21) }

        public fun f16(v1: F16, v2: F16): F16 = { v11, v21 -> v11 - v21(v11, v21) }

        public fun f32(v1: Float, v2: Float): Float = { v11, v21 -> v11 - v21(v11, v21) }

        public fun f64(v1: Double, v2: Double): Double = { v11, v21 -> v11 - v21(v11, v21) }

        public fun u8(v1: UByte, v2: UByte): UByte = { v11, v21 -> v11 - v21(v11, v21) }

        public fun u32(v1: UInt, v2: UInt): UInt = { v11, v21 -> v11 - v21(v11, v21) }

        public fun i16(v1: Short, v2: Short): Short = { v11, v21 -> v11 - v21(v11, v21) }

        public fun i32(v1: Int, v2: Int): Int = { v11, v21 -> v11 - v21(v11, v21) }

        public fun i64(v1: Long, v2: Long): Long = { v11, v21 -> v11 - v21(v11, v21) }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 = { v11, v21 -> v11 - v21(v11, v21) }
    }
}

public class Maximum : BinaryOpT<Maximum> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Maximum> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Maximum> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Maximum =
                    Maximum()
                override fun writeNative(addr: Long, value: Maximum) {
                }
            }

        public val NAME: String = "maximum"

        public val KERNEL: String = "bmaximum"

        public val V: Maximum = Maximum()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f16(v1: F16, v2: F16): F16 =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f32(v1: Float, v2: Float): Float =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f64(v1: Double, v2: Double): Double =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun u8(v1: UByte, v2: UByte): UByte =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun u32(v1: UInt, v2: UInt): UInt =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i16(v1: Short, v2: Short): Short =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i32(v1: Int, v2: Int): Int =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i64(v1: Long, v2: Long): Long =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 =
            { v11, v21 ->
                if (v11 < v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }
    }
}

public class Minimum : BinaryOpT<Minimum> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Minimum> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Minimum> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Minimum =
                    Minimum()
                override fun writeNative(addr: Long, value: Minimum) {
                }
            }

        public val NAME: String = "minimum"

        public val KERNEL: String = "bminimum"

        public val V: Minimum = Minimum()

        public fun bf16(v1: Bf16, v2: Bf16): Bf16 =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f16(v1: F16, v2: F16): F16 =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f32(v1: Float, v2: Float): Float =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f64(v1: Double, v2: Double): Double =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun u8(v1: UByte, v2: UByte): UByte =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun u32(v1: UInt, v2: UInt): UInt =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i16(v1: Short, v2: Short): Short =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i32(v1: Int, v2: Int): Int =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun i64(v1: Long, v2: Long): Long =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }

        public fun f8e4m3(v1: F8e4m3, v2: F8e4m3): F8e4m3 =
            { v11, v21 ->
                if (v11 > v21) {
                    v21
                } else {
                    v11
                }(v11, v21)
            }
    }
}

public class Exp : UnaryOpT<Exp> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Exp> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Exp> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Exp =
                    Exp()
                override fun writeNative(addr: Long, value: Exp) {
                }
            }

        public val NAME: String = "exp"

        public val KERNEL: String = "uexp"

        public val V: Exp = Exp()

        public fun bf16(v: Bf16): Bf16 = v.exp()

        public fun f16(v: F16): F16 = v.exp()

        public fun f32(v: Float): Float = v.exp()

        public fun f64(v: Double): Double = v.exp()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.exp()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsExp(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdExp(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsExp(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdExp(xs, ys)
        }
    }
}

public class Log : UnaryOpT<Log> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Log> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Log> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Log =
                    Log()
                override fun writeNative(addr: Long, value: Log) {
                }
            }

        public val NAME: String = "log"

        public val KERNEL: String = "ulog"

        public val V: Log = Log()

        public fun bf16(v: Bf16): Bf16 = v.ln()

        public fun f16(v: F16): F16 = v.ln()

        public fun f32(v: Float): Float = v.ln()

        public fun f64(v: Double): Double = v.ln()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.ln()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsLn(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdLn(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsLn(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdLn(xs, ys)
        }
    }
}

public class Sin : UnaryOpT<Sin> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sin> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sin> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Sin =
                    Sin()
                override fun writeNative(addr: Long, value: Sin) {
                }
            }

        public val NAME: String = "sin"

        public val KERNEL: String = "usin"

        public val V: Sin = Sin()

        public fun bf16(v: Bf16): Bf16 = v.sin()

        public fun f16(v: F16): F16 = v.sin()

        public fun f32(v: Float): Float = v.sin()

        public fun f64(v: Double): Double = v.sin()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.sin()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsSin(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdSin(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsSin(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdSin(xs, ys)
        }
    }
}

public class Cos : UnaryOpT<Cos> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Cos> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Cos> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Cos =
                    Cos()
                override fun writeNative(addr: Long, value: Cos) {
                }
            }

        public val NAME: String = "cos"

        public val KERNEL: String = "ucos"

        public val V: Cos = Cos()

        public fun bf16(v: Bf16): Bf16 = v.cos()

        public fun f16(v: F16): F16 = v.cos()

        public fun f32(v: Float): Float = v.cos()

        public fun f64(v: Double): Double = v.cos()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.cos()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsCos(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdCos(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsCos(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdCos(xs, ys)
        }
    }
}

public class Abs : UnaryOpT<Abs> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Abs> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Abs> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Abs =
                    Abs()
                override fun writeNative(addr: Long, value: Abs) {
                }
            }

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

public class Neg : UnaryOpT<Neg> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Neg> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Neg> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Neg =
                    Neg()
                override fun writeNative(addr: Long, value: Neg) {
                }
            }

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

public class Recip : UnaryOpT<Recip> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Recip> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Recip> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Recip =
                    Recip()
                override fun writeNative(addr: Long, value: Recip) {
                }
            }

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

public class Sqr : UnaryOpT<Sqr> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sqr> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sqr> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Sqr =
                    Sqr()
                override fun writeNative(addr: Long, value: Sqr) {
                }
            }

        public val NAME: String = "sqr"

        public val KERNEL: String = "usqr"

        public val V: Sqr = Sqr()

        public fun bf16(v: Bf16): Bf16 = v * v

        public fun f16(v: F16): F16 = v * v

        public fun f32(v: Float): Float = v * v

        public fun f64(v: Double): Double = v * v

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v * v

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsSqr(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdSqr(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsSqr(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdSqr(xs, ys)
        }
    }
}

public class Sqrt : UnaryOpT<Sqrt> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sqrt> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sqrt> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Sqrt =
                    Sqrt()
                override fun writeNative(addr: Long, value: Sqrt) {
                }
            }

        public val NAME: String = "sqrt"

        public val KERNEL: String = "usqrt"

        public val V: Sqrt = Sqrt()

        public fun bf16(v: Bf16): Bf16 = v.sqrt()

        public fun f16(v: F16): F16 = v.sqrt()

        public fun f32(v: Float): Float = v.sqrt()

        public fun f64(v: Double): Double = v.sqrt()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.sqrt()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsSqrt(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdSqrt(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsSqrt(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdSqrt(xs, ys)
        }
    }
}

public class Gelu : UnaryOpT<Gelu> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Gelu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Gelu> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Gelu =
                    Gelu()
                override fun writeNative(addr: Long, value: Gelu) {
                }
            }

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
            io.github.kotlinmania.candle.accelerate
                .vsGelu(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdGelu(xs, ys)
        }
    }
}

public class GeluErf : UnaryOpT<GeluErf> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<GeluErf> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<GeluErf> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): GeluErf =
                    GeluErf()
                override fun writeNative(addr: Long, value: GeluErf) {
                }
            }

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

public class Erf : UnaryOpT<Erf> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Erf> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Erf> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Erf =
                    Erf()
                override fun writeNative(addr: Long, value: Erf) {
                }
            }

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

public class Relu : UnaryOpT<Relu> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Relu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Relu> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Relu =
                    Relu()
                override fun writeNative(addr: Long, value: Relu) {
                }
            }

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

public class Silu : UnaryOpT<Silu> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Silu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Silu> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Silu =
                    Silu()
                override fun writeNative(addr: Long, value: Silu) {
                }
            }

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
            io.github.kotlinmania.candle.accelerate
                .vsSilu(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdSilu(xs, ys)
        }
    }
}

public class Tanh : UnaryOpT<Tanh> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Tanh> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Tanh> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Tanh =
                    Tanh()
                override fun writeNative(addr: Long, value: Tanh) {
                }
            }

        public val NAME: String = "tanh"

        public val KERNEL: String = "utanh"

        public val V: Tanh = Tanh()

        public fun bf16(v: Bf16): Bf16 = v.tanh()

        public fun f16(v: F16): F16 = v.tanh()

        public fun f32(v: Float): Float = v.tanh()

        public fun f64(v: Double): Double = v.tanh()

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

        public fun f8e4m3(v: F8e4m3): F8e4m3 = v.tanh()

        public fun f32Vec(xs: List<Float>, ys: List<Float>) {
            vsTanh(xs, ys)
        }

        public fun f64Vec(xs: List<Double>, ys: List<Double>) {
            vdTanh(xs, ys)
        }

        public fun f32VecUnaryOpT(xs: List<Float>, ys: List<Float>) {
            io.github.kotlinmania.candle.accelerate
                .vsTanh(xs, ys)
        }

        public fun f64VecUnaryOpT(xs: List<Double>, ys: List<Double>) {
            io.github.kotlinmania.candle.accelerate
                .vdTanh(xs, ys)
        }
    }
}

public class Floor : UnaryOpT<Floor> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Floor> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Floor> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Floor =
                    Floor()
                override fun writeNative(addr: Long, value: Floor) {
                }
            }

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

public class Ceil : UnaryOpT<Ceil> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Ceil> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Ceil> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Ceil =
                    Ceil()
                override fun writeNative(addr: Long, value: Ceil) {
                }
            }

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

public class Round : UnaryOpT<Round> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Round> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Round> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Round =
                    Round()
                override fun writeNative(addr: Long, value: Round) {
                }
            }

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

public class Sign : UnaryOpT<Sign> {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sign> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Sign> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Sign =
                    Sign()
                override fun writeNative(addr: Long, value: Sign) {
                }
            }

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
public data class BackpropOp(
    public var value: Op? = null,
) {
    public var field0: Op?
        get() = value
        set(v) {
            value = v
        }

    internal fun isNone(): Boolean = this.value == null

    public fun deref(): Op? = this.value

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
