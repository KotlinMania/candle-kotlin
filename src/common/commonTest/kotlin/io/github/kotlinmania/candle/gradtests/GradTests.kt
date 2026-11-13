package io.github.kotlinmania.candle.gradtests

import io.github.kotlinmania.anyhow.Result
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Shape
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.Var
import io.github.kotlinmania.candlecore.backprop.GradStore
import io.github.kotlinmania.candlecore.testutils.assertTensorEq

public fun simpleGrad(device: Device): Result<Unit> {
    val x =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, 4.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x1 = x.asTensor()
    val y =
        when (
            val tryResult = (
                when (
                    val tryResult = (
                        when (val tryResult = (x1 * x1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        } + x1 * 5.0
                    )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } + 4.0
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = y.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = x1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 1.0, 4.0),
    )
    // y = x^2 + 5.x + 4
    check(
        when (val tryResult = y.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(28.0, 10.0, 40.0),
    )
    // dy/dx = 2.x + 5
    check(
        when (val tryResult = gradX.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(11.0, 7.0, 13.0),
    )
    return Result.Ok(Unit)
}

public fun sumGrad(device: Device): Result<Unit> {
    val x =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, 4.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x1 = x.asTensor()
    val y =
        when (
            val tryResult = (
                when (
                    val tryResult =
                        when (val tryResult = x1.sqr()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * 2.0
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = y.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(52.0),
    )
    // y = 2.x^2 so dy/dx = 4.x
    check(
        when (val tryResult = gradX.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(12.0, 4.0, 16.0),
    )
    // Same test as before but squeezing on the last dimension.
    val y1 =
        when (
            val tryResult =
                when (
                    val tryResult = (
                        when (
                            val tryResult =
                                when (val tryResult = x1.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sumKeepdim(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        } * 2.0
                    )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.squeeze(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads1 =
        when (val tryResult = y1.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX1 =
        when (val tryResult = grads1.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y1.toScalar<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 52.0,
    )
    // y = 2.x^2 so dy/dx = 4.x
    check(
        when (val tryResult = gradX1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(12.0, 4.0, 16.0),
    )
    return Result.Ok(Unit)
}

public fun expandGrad(device: Device): Result<Unit> {
    // Computing the gradient sums over a middle dim of a 5D tensor, which
    // takes the strided-index fallback path in the Metal reduce kernels,
    // see issue #3847.
    val data: List<Float> = (0 until 24).map({ v -> v.toFloat() }).toMutableList()
    val x =
        when (val tryResult = Var.fromVec(data, listOf(1, 1, 4, 6), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val e =
        when (
            val tryResult =
                when (val tryResult = x.asTensor().unsqueeze(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.expand(listOf(1, 1, 3, 4, 6))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val loss =
        when (
            val tryResult = (
                when (
                    val tryResult =
                        when (val tryResult = e.sqr()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * 0.5
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = loss.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // loss = 0.5 * sum over the 3 broadcast copies of x^2, so dloss/dx = 3.x
    check(
        when (
            val tryResult =
                when (val tryResult = gradX.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == (0 until 24).map({ v -> 3.0f * v.toFloat() }).toMutableList(),
    )
    return Result.Ok(Unit)
}

public fun matmulGrad(device: Device): Result<Unit> {
    val data: Vec = (0 until 12).map({ i -> i.toFloat() }).collect()
    val x =
        when (val tryResult = Var.fromSlice(data, Triple(2, 2, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data1: Vec = (0 until 12).map({ i -> i.toFloat() }).collect()
    val y =
        when (val tryResult = Var.fromSlice(data1, Triple(2, 3, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val c =
        when (val tryResult = x.matmul(y)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = c.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradY =
        when (val tryResult = grads.get(y).context("no grad for y")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(gradX.shape() == Shape.from(Triple(2, 2, 3)))
    check(gradY.shape() == Shape.from(Triple(2, 3, 2)))
    check(
        when (val tryResult = gradX.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1.0, 5.0, 9.0), listOf(1.0, 5.0, 9.0)), listOf(listOf(13.0, 17.0, 21.0), listOf(13.0, 17.0, 21.0))),
    )
    check(
        when (val tryResult = gradY.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(3.0, 3.0), listOf(5.0, 5.0), listOf(7.0, 7.0)), listOf(listOf(15.0, 15.0), listOf(17.0, 17.0), listOf(19.0, 19.0))),
    )
    return Result.Ok(Unit)
}

public fun assertZeroGrad(grads: GradStore, variable: Var, shape: List<Int>): Result<Unit> {
    val grad =
        when (val tryResult = grads.get(variable).context("no gradient for variable")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(grad.dims() == shape)
    check(
        when (
            val tryResult =
                when (val tryResult = grad.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == mutableListOf(0.0, grad.elemCount()),
    )
    return Result.Ok(Unit)
}

public fun assertZeroMatmulGrads(device: Device, lhsShape: List<Int>, rhsShape: List<Int>, outputShape: List<Int>): Result<Unit> {
    val lhs =
        when (val tryResult = Var.zeros(lhsShape, DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = Var.zeros(rhsShape, DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = lhs.matmul(rhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(output.dims() == outputShape)
    check(
        when (
            val tryResult =
                when (val tryResult = output.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == mutableListOf(0.0, output.elemCount()),
    )
    val grads =
        when (
            val tryResult =
                when (val tryResult = output.sumAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.backward()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertZeroGrad(grads, lhs, lhsShape)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return assertZeroGrad(grads, rhs, rhsShape)
}

public fun zeroMatmulGrad(device: Device): Result<Unit> {
    val cases: List<Triple<List<Int>, List<Int>, List<Int>>> = listOf(Triple(listOf(2, 0), listOf(0, 3), listOf(2, 3)), Triple(listOf(0, 2), listOf(2, 3), listOf(0, 3)), Triple(listOf(2, 3), listOf(3, 0), listOf(2, 0)), Triple(listOf(0, 2, 3), listOf(0, 3, 4), listOf(0, 2, 4)), Triple(listOf(2, 3, 0), listOf(2, 0, 4), listOf(2, 3, 4)))
    for ((lhsShape, rhsShape, outputShape) in cases) {
        when (val tryResult = assertZeroMatmulGrads(device, lhsShape, rhsShape, outputShape)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(Unit)
}

// The simplest gradient descent, using scalar variable.
public fun gradDescent(device: Device): Result<Unit> {
    val x =
        when (val tryResult = Var.new(0.0f, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val learningRate = 0.1
    for (step in 0 until 100) {
        val xt = x.asTensor()
        val c =
            when (
                val tryResult = (
                    when (val tryResult = (xt - 4.2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    } *
                        when (val tryResult = (xt - 4.2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val grads =
            when (val tryResult = c.backward()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val xGrad =
            when (val tryResult = grads.get(x).context("no grad for x")) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        when (
            val tryResult =
                x.set(
                    when (val tryResult = (xt - xGrad * learningRate)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    check(
        when (val tryResult = x.toScalar<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 4.199999,
    )
    return Result.Ok(Unit)
}

public fun unaryGrad(device: Device): Result<Unit> {
    val x =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, 4.0, 0.15), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x1 = x.asTensor()
    val y =
        when (
            val tryResult = (
                when (val tryResult = x1.log()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } + 1.0
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = y.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2.0986, 1.0, 2.3863, -0.8971),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.3333, 1.0, 0.25, 6.6667),
    )
    val y1 =
        when (val tryResult = x1.exp()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads1 =
        when (val tryResult = y1.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX1 =
        when (val tryResult = grads1.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y1, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(20.0855, 2.7183, 54.5982, 1.1618),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX1, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(20.0855, 2.7183, 54.5982, 1.1618),
    )
    val y2 =
        when (
            val tryResult =
                when (val tryResult = x1.exp()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.sqr()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads2 =
        when (val tryResult = y2.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX2 =
        when (val tryResult = grads2.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y2, 3)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(403.429, 7.389, 2980.958, 1.35),
    )
    // exp(x)^2 = exp(2*x)
    check(
        when (val tryResult = testutils.toVec1Round(gradX2, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(806.86, 14.78, 5961.92, 2.7),
    )
    val y3 =
        when (val tryResult = x1.sin()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads3 =
        when (val tryResult = y3.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX3 =
        when (val tryResult = grads3.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y3, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.1411, 0.8415, -0.7568, 0.1494),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX3, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.99, 0.5403, -0.6536, 0.9888),
    )
    val y4 =
        when (val tryResult = x1.cos()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads4 =
        when (val tryResult = y4.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX4 =
        when (val tryResult = grads4.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y4, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.99, 0.5403, -0.6536, 0.9888),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX4, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.1411, -0.8415, 0.7568, -0.1494),
    )
    val y5 =
        when (val tryResult = x1.sqr()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads5 =
        when (val tryResult = y5.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX5 =
        when (val tryResult = grads5.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y5.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(9.0, 1.0, 16.0, 0.0225),
    )
    check(
        when (val tryResult = gradX5.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(6.0, 2.0, 8.0, 0.3),
    )
    val y6 =
        when (
            val tryResult =
                when (val tryResult = x1.sqr()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.sqrt()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads6 =
        when (val tryResult = y6.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX6 =
        when (val tryResult = grads6.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y6.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 1.0, 4.0, 0.15),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX6, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 1.0, 1.0, 1.0),
    )
    val y7 =
        when (val tryResult = x1.neg()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads7 =
        when (val tryResult = y7.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX7 =
        when (val tryResult = grads7.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y7.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-3.0, -1.0, -4.0, -0.15),
    )
    check(
        when (val tryResult = gradX7.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.0, -1.0, -1.0, -1.0),
    )
    val y8 =
        when (val tryResult = x1.affine(0.2, 1.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads8 =
        when (val tryResult = y8.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX8 =
        when (val tryResult = grads8.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y8.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.6, 1.2, 1.8, 1.03),
    )
    check(
        when (val tryResult = gradX8.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.2, 0.2, 0.2, 0.2),
    )
    val y9 =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(1.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastDiv(x1)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads9 =
        when (val tryResult = y9.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX9 =
        when (val tryResult = grads9.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y9, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.3333, 1.0, 0.25, 6.6667),
    )
    check(
        when (val tryResult = gradX9.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.11111111, -1.0, -0.0625, -44.444443),
    )
    val y10 =
        when (
            val tryResult =
                x1.broadcastDiv(
                    when (val tryResult = Tensor.new(0.5f, device)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads10 =
        when (val tryResult = y10.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX10 =
        when (val tryResult = grads10.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y10.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(6.0, 2.0, 8.0, 0.3),
    )
    check(
        when (val tryResult = gradX10.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2.0, 2.0, 2.0, 2.0),
    )
    val x2 =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, 4.0, 0.15), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y11 =
        when (val tryResult = x2.powf(2.5)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads11 =
        when (val tryResult = y11.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX11 =
        when (val tryResult = grads11.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y11, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(15.59, 1.0, 32.0, 0.01),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX11, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(12.99, 2.5, 20.0, 0.15),
    )
    val y12 =
        when (val tryResult = x2.tanh()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads12 =
        when (val tryResult = y12.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX12 =
        when (val tryResult = grads12.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y12, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 0.76, 1.0, 0.15),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX12, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.01, 0.42, 0.0, 0.98),
    )
    // testing compared to pytorch nn.GELU(approximate = 'tanh')
    val y13 =
        when (val tryResult = x2.gelu()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads13 =
        when (val tryResult = y13.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX13 =
        when (val tryResult = grads13.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y13, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2.9964, 0.8412, 3.9999, 0.0839),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX13, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0116, 1.0830, 1.0003, 0.6188),
    )
    // Testing compared to pytorch torch.erf
    //
    // import torch
    // x = torch.tensor([3.0, 1.0, 4.0, 0.15], requiresGrad=True)
    // y = x.erf()
    // print(y)
    // loss = y.sum()
    // loss.backward()
    // print(x.grad)
    val y14 =
        when (val tryResult = x2.erf()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads14 =
        when (val tryResult = y14.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX14 =
        when (val tryResult = grads14.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y14, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 0.8427, 1.0, 0.168),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX14, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.0001, 0.4151, 0.0, 1.1033),
    )
    // Testing compared to pytorch nn.GELU(approximate = 'none')
    //
    // import torch
    // import torch.nn.functional as F
    // x = torch.tensor([3.0, 1.0, 4.0, 0.15], requiresGrad=True)
    // y = F.gelu(x, approximate='none')
    // print(y)
    // loss = y.sum()
    // loss.backward()
    // print(x.grad)
    val y15 =
        when (val tryResult = x2.geluErf()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads15 =
        when (val tryResult = y15.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX15 =
        when (val tryResult = grads15.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y15, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2.9960, 0.8413, 3.9999, 0.0839),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX15, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0119, 1.0833, 1.0005, 0.6188),
    )
    // Testing compared to pytorch elu
    //
    // import torch
    // import torch.nn.functional as F
    // x = torch.tensor([-1.0, 0.0, -2.0, 3.0], requiresGrad=True)
    // y = F.elu(x, alpha=2.0)
    // print(y)
    // loss = y.min
    // loss = y.sum()
    // loss.backward()
    // print(x.grad)
    val eluX =
        when (val tryResult = Var.new(listOf(-1.0f, 0.0, -2.0, 3.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y16 =
        when (val tryResult = eluX.elu(2.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads16 =
        when (val tryResult = y16.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX16 =
        when (val tryResult = grads16.get(eluX).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y16, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.2642, 0.0000, -1.7293, 3.0000),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX16, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.7358, 2.0000, 0.2707, 1.0000),
    )
    // testing compared to pytorch nn.Silu()
    val y17 =
        when (val tryResult = x2.silu()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads17 =
        when (val tryResult = y17.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX17 =
        when (val tryResult = grads17.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y17, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2.8577, 0.7311, 3.9281, 0.0806),
    )
    check(
        when (val tryResult = testutils.toVec1Round(gradX17, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0881, 0.9277, 1.0527, 0.5747),
    )
    if (device.isCpu()) {
        val x3 =
            when (val tryResult = Var.new(listOf(listOf(listOf(1.0f, 2.0, 3.0), listOf(4.0, 5.0, 6.0), listOf(7.0, 8.0, 9.0))), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val y18 =
            when (
                val tryResult =
                    when (val tryResult = x3.interpolate1d(12)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.reshape(36)
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val z =
            when (val tryResult = Tensor.new(listOf(1.0f, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val loss =
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = y18.unsqueeze(1)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.transpose(0, 1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.matmul(
                        when (val tryResult = z.unsqueeze(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val grads18 =
            when (val tryResult = loss.backward()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val gradX18 =
            when (val tryResult = grads18.get(x3).context("no grad for x")) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = testutils.toVec3Round(gradX18, 4)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(10.0f, 26.0, 42.0), listOf(58.0, 74.0, 90.0), listOf(106.0, 122.0, 138.0))),
        )
    }
    // manually checked: see comments
    val x4 =
        when (val tryResult = Var.new(listOf(listOf(listOf(listOf(1.0f, 2.0, 3.0), listOf(4.0, 5.0, 6.0), listOf(7.0, 8.0, 9.0)))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y19 =
        when (
            val tryResult =
                when (val tryResult = x4.interpolate2d(6, 6)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(36)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val z1 =
        when (val tryResult = Tensor.new(listOf(1.0f, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // gradient should be
    // row 1
    // 1+2+7+8 = 18
    // 3+4+9+10 = 26
    // 5+6+11+12 = 34
    // row 2
    // 13+14+19+20 = 66
    // 15+16+21+22 = 74
    // 17+18+23+24 = 82
    // row 3
    // 25+26+31+32 = 114
    // 27+28+33+34 = 122
    // 29+30+35+36 = 130
    val loss1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = y19.unsqueeze(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(0, 1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.matmul(
                    when (val tryResult = z1.unsqueeze(1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads19 =
        when (val tryResult = loss1.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX19 =
        when (val tryResult = grads19.get(x4).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = gradX19.flatten(0, 2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(18.0f, 26.0, 34.0), listOf(66.0, 74.0, 82.0), listOf(114.0, 122.0, 130.0)),
    )
    // manually checked: see comments
    val x5 =
        when (val tryResult = Var.new(listOf(listOf(listOf(listOf(1.0f, 2.0), listOf(4.0, 5.0)))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y20 =
        when (
            val tryResult =
                when (val tryResult = x5.interpolate2d(6, 6)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(36)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val z2 =
        when (val tryResult = Tensor.new(listOf(1.0f, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // gradient should be
    // row 1
    // 1+2+3+7+8+9+13+14+15 = 72
    // 4+5+6+10+11+12+16+17+18 = 99
    // row 2
    // 19+20+21+25+26+27+31+32+33 = 234
    // 22+23+24+28+29+30+34+35+36 = 243
    val loss2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = y20.unsqueeze(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(0, 1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.matmul(
                    when (val tryResult = z2.unsqueeze(1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads20 =
        when (val tryResult = loss2.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX20 =
        when (val tryResult = grads20.get(x5).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = gradX20.flatten(0, 2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(72.0f, 99.0), listOf(234.0, 261.0)),
    )
    // manually checked: see comments
    val x6 =
        when (val tryResult = Var.new(listOf(listOf(listOf(listOf(1.0f, 2.0), listOf(4.0, 5.0)), listOf(listOf(6.0f, 7.0), listOf(8.0, 9.0)))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y21 =
        when (
            val tryResult =
                when (val tryResult = x6.interpolate2d(4, 4)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val z3 =
        when (val tryResult = Tensor.new(listOf(1.0f, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // gradient should be
    // m1r1
    // 1+2+5+6=14
    // 3+4+7+8=22
    // m1r2
    // 9+10+13+14=46
    // 11+12+15+16=54
    // m2r1
    // 17+18+21+22=78
    // 19+20+23+24=86
    // m2r2
    // 25+26+29+30=110
    // 27+28+31+32=118
    val loss3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = y21.unsqueeze(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(0, 1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.matmul(
                    when (val tryResult = z3.unsqueeze(1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads21 =
        when (val tryResult = loss3.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX21 =
        when (val tryResult = grads21.get(x6).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec3Round(
                    when (val tryResult = gradX21.flatten(0, 1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(14.0f, 22.0), listOf(46.0, 54.0)), listOf(listOf(78.0, 86.0), listOf(110.0, 118.0))),
    )
    // manually checked: see comments
    val x7 =
        when (val tryResult = Var.new(listOf(listOf(listOf(listOf(1.0f, 2.0), listOf(4.0, 5.0))), listOf(listOf(listOf(6.0f, 7.0), listOf(8.0, 9.0)))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y22 =
        when (
            val tryResult =
                when (val tryResult = x7.interpolate2d(4, 4)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val z4 =
        when (val tryResult = Tensor.new(listOf(1.0f, 02.0, 03.0, 04.0, 05.0, 06.0, 07.0, 08.0, 09.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // gradient should be
    // m1r1
    // 1+2+5+6=14
    // 3+4+7+8=22
    // m1r2
    // 9+10+13+14=46
    // 11+12+15+16=54
    // m2r1
    // 17+18+21+22=78
    // 19+20+23+24=86
    // m2r2
    // 25+26+29+30=110
    // 27+28+31+32=118
    val loss4 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = y22.unsqueeze(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(0, 1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.matmul(
                    when (val tryResult = z4.unsqueeze(1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads22 =
        when (val tryResult = loss4.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX22 =
        when (val tryResult = grads22.get(x7).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec3Round(
                    when (val tryResult = gradX22.flatten(0, 1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(14.0f, 22.0), listOf(46.0, 54.0)), listOf(listOf(78.0, 86.0), listOf(110.0, 118.0))),
    )
    return Result.Ok(Unit)
}

public fun binaryGrad(device: Device): Result<Unit> {
    val x =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, -4.0, -1.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x1 = x.asTensor()
    // leaky relu
    val y =
        when (
            val tryResult =
                x1.maximum(
                    when (val tryResult = (x1 * 0.1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (val tryResult = y.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX =
        when (val tryResult = grads.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = x1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 1.0, -4.0, -1.0),
    )
    check(
        when (val tryResult = y.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 1.0, -0.4, -0.1),
    )
    check(
        when (val tryResult = gradX.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 1.0, 0.1, 0.1),
    )
    val y1 =
        when (
            val tryResult =
                x1.minimum(
                    when (val tryResult = (x1 * 0.1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads1 =
        when (val tryResult = y1.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX1 =
        when (val tryResult = grads1.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.3, 0.1, -4.0, -1.0),
    )
    check(
        when (val tryResult = gradX1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0.1, 0.1, 1.0, 1.0),
    )
    // This one is easy to mess up, we want the gradient to be one as it is the identity function.
    val y2 =
        when (val tryResult = x1.minimum(x1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads2 =
        when (val tryResult = y2.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX2 =
        when (val tryResult = grads2.get(x1).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = y2.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 1.0, -4.0, -1.0),
    )
    check(
        when (val tryResult = gradX2.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 1.0, 1.0, 1.0),
    )
    val xVar =
        when (val tryResult = Var.new(listOf(3.0f, 1.0, -4.0, -1.0, 5.0, 9.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x2 = xVar.asTensor()
    val yVar =
        when (val tryResult = Var.new(listOf(2.0f, 7.0, 1.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y3 = yVar.asTensor()
    val ss =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = x2.reshape(Pair(2, 3))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sliceScatter0(
                            when (val tryResult = y3.reshape(Pair(1, 3))) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                            1,
                        )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.sqr()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads3 =
        when (val tryResult = ss.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX3 =
        when (val tryResult = grads3.get(x2).context("no grad for x")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradY =
        when (val tryResult = grads3.get(y3).context("no grad for y")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = ss.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(9.0, 1.0, 16.0), listOf(4.0, 49.0, 1.0)),
    )
    check(
        when (val tryResult = gradX3.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(6.0, 2.0, -8.0, 0.0, 0.0, 0.0),
    )
    check(
        when (val tryResult = gradY.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(4.0, 14.0, 2.0),
    )
    return Result.Ok(Unit)
}

public fun indexSelectStridedIdsGrad(device: Device): Result<Unit> {
    val w =
        when (val tryResult = Var.new(listOf(listOf(1.0f, 2.0, 3.0), listOf(4.0, 5.0, 6.0), listOf(7.0, 8.0, 9.0)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val allIds =
        when (val tryResult = Tensor.new(listOf(listOf(0u, 1), listOf(2, 1)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids =
        when (val tryResult = allIds.i(Pair(0 until 0, 0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(!ids.isContiguous())
    val y =
        when (val tryResult = w.indexSelect(ids, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grads =
        when (
            val tryResult =
                when (val tryResult = y.sumAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.backward()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradW =
        when (val tryResult = grads.get(w).context("no grad for w")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = gradW.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.0, 1.0), listOf(0.0, 0.0, 0.0), listOf(1.0, 1.0, 1.0)),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun testFlipBackprop(): Result<Unit> {
    val device = Device.Cpu
    // Create a tensor (leaf node) that requires gradients
    val x =
        when (val tryResult = Var.ones(Pair(2, 2), DType.F64, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val weights =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(1.0, 5.0, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y =
        when (val tryResult = x.matmul(weights)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedY =
        when (val tryResult = Tensor.fromVec(mutableListOf(4.0, 6.0, 4.0, 6.0), Pair(2, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(y, expectedY)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val z =
        when (val tryResult = y.flip(listOf(1))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedZ =
        when (val tryResult = Tensor.fromVec(mutableListOf(6.0, 4.0, 6.0, 4.0), Pair(2, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(z, expectedZ)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val loss =
        when (val tryResult = z.sumAll()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradStore =
        when (val tryResult = loss.backward()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val gradX = gradStore.getId(x.id()).unwrap()
    val flippedWeights =
        when (val tryResult = weights.flip(listOf(1))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dlossDy =
        when (val tryResult = Tensor.ones(Pair(2, 2), DType.F64, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // dloss/dx = dloss/dy @ dy/dx = ones @ weight.flip.T
    val expectedGrad =
        when (
            val tryResult =
                dlossDy.matmul(
                    when (val tryResult = flippedWeights.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(gradX, expectedGrad)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}
