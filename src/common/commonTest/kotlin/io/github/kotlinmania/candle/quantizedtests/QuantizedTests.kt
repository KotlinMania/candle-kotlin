package io.github.kotlinmania.candle.quantizedtests

import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.candle.quantized.kquants.matmul
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.Var
import io.github.kotlinmania.candlecore.bail
import io.github.kotlinmania.candlecore.quantized
import io.github.kotlinmania.candlecore.quantized.GgmlDType
import io.github.kotlinmania.candlecore.testutils.toVec2Round
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.kquants.BlockQ2K
import io.github.kotlinmania.kquants.BlockQ3K
import io.github.kotlinmania.kquants.BlockQ4K
import io.github.kotlinmania.kquants.BlockQ5K
import io.github.kotlinmania.kquants.BlockQ6K
import io.github.kotlinmania.kquants.BlockQ8K
import io.github.kotlinmania.quantized.GgmlType
import io.github.kotlinmania.rand.prelude.*

public val GGML_TEST_SIZE: Int = 32 * 128

public val GGML_MAX_QUANTIZATION_TOTAL_ERROR: Float = 0.002f

public val GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: Float = 0.0075f

public val GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: Float = 0.0040f

public val GGML_MAX_DOT_PRODUCT_ERROR: Float = 0.02f

public fun testMatmul(device: Device, tupleArg1: List<Any?>, dtype: GgmlDType): Result<Unit> {
    if ((device.isCuda() || device.isMetal()) && (dtype == GgmlDType.Q8_1 || dtype == GgmlDType.Q8K)) {
        return Result.Ok(Unit)
    }
    val lhs = (0 until (m * k)).map({ v -> v.toFloat() / (m * k).toFloat() }).collect<Vec>()
    val rhs = (0 until (k * n)).map({ v -> v.toFloat() / (n * k).toFloat() }).collect<Vec>()
    val lhs1 =
        when (val tryResult = Tensor.fromSlice(lhs, Pair(m, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (val tryResult = Tensor.fromSlice(rhs, Pair(k, n), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm =
        when (val tryResult = lhs1.matmul(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val qtensor =
        when (
            val tryResult =
                quantized.QTensor.quantize(
                    when (val tryResult = rhs1.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    dtype,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val matmul =
        when (val tryResult = quantized.QMatMul.fromQtensor(qtensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res =
        when (val tryResult = matmul.forward(lhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val error: Float =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult = (
                                when (
                                    val tryResult =
                                        when (val tryResult = (mm - res)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.abs()
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                } /
                                    when (val tryResult = mm.abs()) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }
                            )
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val error1 = error / (b * m * n).toFloat()
    check(error1 <= 0.02f) { "Error $error is too big. \nExpected:\n $mm \nFound:\n $res\n for $dtype" }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun testMatmulMm(): Result<Unit> {
    val dtype = GgmlDType.Q8_0
    val device =
        when (val tryResult = Device.newMetal(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val m = 32
    val n = 32
    val k = 32
    val lhs = (0 until (m * k)).map({ v -> v.toFloat() / (m * k).toFloat() }).collect<Vec>()
    val rhs = (0 until (k * n)).map({ v -> v.toFloat() / (n * k).toFloat() }).collect<Vec>()
    val lhs1 =
        when (val tryResult = Tensor.fromSlice(lhs, Pair(m, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromSlice(rhs, listOf(1, 1, k, n), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.repeat(listOf(5, 20, 1, 1))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm =
        when (val tryResult = lhs1.broadcastMatmul(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val qtensor =
        when (
            val tryResult =
                quantized.QTensor.quantize(
                    when (val tryResult = lhs1.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    dtype,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val matmul =
        when (val tryResult = quantized.QMatMul.fromQtensor(qtensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res =
        when (val tryResult = matmul.forward(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val error: Float =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult = (
                                when (
                                    val tryResult =
                                        when (val tryResult = (mm - res)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.abs()
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                } /
                                    when (val tryResult = mm.abs()) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }
                            )
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val error1 = error / res.elemCount().toFloat()
    check(error1 <= 0.001f) { "Error $error is too big. \nExpected:\n $mm \nFound:\n $res\n for $dtype" }
    return Result.Ok(Unit)
}

public fun quantizedMatmul(device: Device): Result<Unit> {
    val (m, k, n) = Triple(3, 64, 4)
    val lhsS = (0 until (m * k)).map({ v -> v.toFloat() }).collect<Vec>()
    val lhs =
        when (val tryResult = Tensor.fromSlice(lhsS, Pair(m, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst = mutableListOf(42.0, 3 * 4)
    val rhsT = mutableListOf(kquants.BlockQ40.zeros(), 8)
    val rhs = (0 until (k * n)).map({ v -> v.toFloat() }).collect<Vec>()
    kquants.BlockQ40.fromFloat(rhs, rhsT)
    when (val tryResult = kquants.matmul(Triple(m, k, n), lhsS, rhsT, dst)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(ArrayDeque((ArrayDeque(dst)).map({ x -> x.round() })).collect<Vec>() == listOf(85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0, 341876.0, 994283.0, 1655709.0, 2301518.0))
    val tensorRhs =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromSlice(rhs, Pair(n, k), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm =
        when (val tryResult = lhs.matmul(tensorRhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = mm.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(85344.0, 214368.0, 343392.0, 472416.0), listOf(214368.0, 605536.0, 996704.0, 1387872.0), listOf(343392.0, 996704.0, 1650016.0, 2303328.0)),
    )
    val qtensor =
        when (
            val tryResult =
                quantized.QTensor.quantize(
                    when (val tryResult = tensorRhs.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    GgmlDType.Q4_0,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val matmul =
        when (val tryResult = quantized.QMatMul.fromQtensor(qtensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res =
        when (val tryResult = matmul.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (device) {
        is Device.Metal ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(84946.0, 214126.0, 344757.0, 473798.0), listOf(213458.0, 604350.0, 1000469.0, 1387990.0), listOf(341970.0, 994574.0, 1656181.0, 2302182.0)),
            )
        is Device.Cuda ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(84866.0, 214045.0, 344676.0, 473707.0), listOf(213425.0, 604313.0, 1000431.0, 1387960.0), listOf(342030.0, 994630.0, 1656248.0, 2302250.0)),
            )
        Device.Cpu ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(85120.0, 214562.0, 345455.0, 474748.0), listOf(213475.0, 604465.0, 1000686.0, 1388317.0), listOf(341876.0, 994283.0, 1655709.0, 2301518.0)),
            )
    }
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q4_0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulNeg(device: Device): Result<Unit> {
    val (m, k, n) = Triple(3, 64, 4)
    val lhsS = (0 until (m * k)).map({ v -> v.toFloat() - (m * k).toFloat() / 2.0f }).collect<Vec>()
    val lhs =
        when (val tryResult = Tensor.fromSlice(lhsS, Pair(m, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst = mutableListOf(42.0, 3 * 4)
    val rhsT = mutableListOf(kquants.BlockQ40.zeros(), 8)
    val rhs = (0 until k * n).map({ v -> v.toFloat() - (k * n).toFloat() / 3.0f }).collect<Vec>()
    val tensorRhs =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromSlice(rhs, Pair(n, k), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    kquants.BlockQ40.fromFloat(rhs, rhsT)
    when (val tryResult = kquants.matmul(Triple(m, k, n), lhsS, rhsT, dst)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(ArrayDeque((ArrayDeque(dst)).map({ x -> x.round() })).collect<Vec>() == listOf(243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0, -196472.0, 63012.0, 324585.0, 587902.0))
    val mm =
        when (val tryResult = lhs.matmul(tensorRhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = toVec2Round(mm, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(244064.0, -20128.0, -284320.0, -548512.0), listOf(23563.0, 21515.0, 19467.0, 17419.0), listOf(-196939.0, 63157.0, 323253.0, 583349.0)),
    )
    val qtensor =
        when (
            val tryResult =
                quantized.QTensor.quantize(
                    when (val tryResult = tensorRhs.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    GgmlDType.Q4_0,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val matmul =
        when (val tryResult = quantized.QMatMul.fromQtensor(qtensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res =
        when (val tryResult = matmul.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (device) {
        is Device.Metal ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(243659.0, -19716.0, -285444.0, -550439.0), listOf(23779.0, 21653.0, 19404.0, 18349.0), listOf(-196101.0, 63021.0, 324252.0, 587137.0)),
            )
        is Device.Cuda ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(243740.0, -19762.0, -285476.0, -550498.0), listOf(23774.0, 21645.0, 19395.0, 18364.0), listOf(-196045.0, 63030.0, 324120.0, 587079.0)),
            )
        Device.Cpu ->
            check(
                when (val tryResult = toVec2Round(res, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(243524.0, -19596.0, -285051.0, -549815.0), listOf(23777.0, 21651.0, 19398.0, 18367.0), listOf(-196472.0, 63012.0, 324585.0, 587902.0)),
            )
    }
    val lhs2 =
        when (val tryResult = Tensor.stack(listOf(lhs, lhs), 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res2 =
        when (val tryResult = matmul.forward(lhs2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res21 =
        when (val tryResult = res2.i(1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (res - res21)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } / res.elemCount().toFloat()
    if (device.isCuda()) {
        check(diff < 0.1f)
    } else {
        check(diff < 0.96f)
    }
    return Result.Ok(Unit)
}

public fun qmmBatch(dev: Device): Result<Unit> {
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(2, 256, 6, dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q2K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.shape().dims() == listOf(2, 6))
    val lhs2 =
        when (val tryResult = Tensor.cat(listOf(lhs, lhs), 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm2 =
        when (val tryResult = rhs2.forward(lhs2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm2.shape().dims() == listOf(4, 6))
    val diff2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = mm2.i(2 until Int.MAX_VALUE)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } - mm
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff2 == 0.0)
    val lhs3 =
        when (val tryResult = Tensor.cat(listOf(lhs2, lhs), 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm3 =
        when (val tryResult = rhs2.forward(lhs3)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm3.shape().dims() == listOf(6, 6))
    val diff3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = mm3.i(2 until 4)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } - mm
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff3 == 0.0)
    val diff31 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = mm3.i(4 until Int.MAX_VALUE)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } - mm
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff31 == 0.0)
    val lhs4 =
        when (val tryResult = Tensor.cat(listOf(lhs3, lhs3), 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm4 =
        when (val tryResult = rhs2.forward(lhs4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm4.shape().dims() == listOf(12, 6))
    val diff4 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = mm4.i(0 until 6)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } - mm3
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (dev.isCuda()) {
        // We use different fused kernels (MMVQ for batch<=8, MMQ for batch>8) on CUDA which accumulate differently than dequantize-then-matmul.
        // This can lead to small numerical differences especially for low-bit quants.
        check(0.0 < diff4 && diff4 < 0.5)
    } else {
        check(diff4 == 0.0)
    }
    val diff41 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = mm4.i(6 until Int.MAX_VALUE)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } -
                                            when (val tryResult = mm4.i(0 until 6)) {
                                                is Result.Ok<*> -> tryResult.value
                                                is Result.Err<*> -> return Result.Err(tryResult.error)
                                            }
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff41 == 0.0)
    return Result.Ok(Unit)
}

public fun embeddingWeight(device: Device): Result<Tensor> {
    val values =
        (0 until (8 * 256))
            .map({ i ->
                val x = i.toFloat()
                (x * 0.003f).sin() * 0.5 + (x * 0.007f).cos() * 0.25
            })
            .collect<Vec>()
    return Tensor.fromVec(values, Pair(8, 256), device)
}

public fun assertEmbeddingClose(dtype: GgmlDType, a: Tensor, b: Tensor, tol: Float): Result<Unit> {
    val a1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = a.toDevice(Device.Cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val b1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = b.toDevice(Device.Cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for ((idx, a2, b2) in (a1.iter().zip(b1.iter())).mapIndexed { index, item -> Pair(index, item) }) {
        check((a2 - b2).abs() <= tol) { "$dtype embedding mismatch at $idx: $a != $b" }
    }
    return Result.Ok(Unit)
}

public fun runQuantizedEmbedding(device: Device, dtype: GgmlDType, tol: Float): Result<Unit> {
    val w =
        when (val tryResult = embeddingWeight(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids =
        when (val tryResult = Tensor.fromVec(mutableListOf(3u, 1, 3, 7), Pair(2, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val q =
        when (val tryResult = quantized.QTensor.quantize(w, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val got =
        when (val tryResult = q.embedding(ids)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expected =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = q.dequantize(device)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.indexSelect(
                            when (val tryResult = ids.flattenAll()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                            0,
                        )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 2, 256))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    return assertEmbeddingClose(dtype, got, expected, tol)
}

@kotlin.test.Test
public fun quantizedEmbeddingCpu(): Result<Unit> {
    val device = Device.Cpu
    for (dtype in listOf(GgmlDType.F32, GgmlDType.F16, GgmlDType.BF16, GgmlDType.Q4_0, GgmlDType.Q4_1, GgmlDType.Q5_0, GgmlDType.Q5_1, GgmlDType.Q8_0, GgmlDType.Q8_1, GgmlDType.Q2K, GgmlDType.Q3K, GgmlDType.Q4K, GgmlDType.Q5K, GgmlDType.Q6K, GgmlDType.Q8K)) {
        when (val tryResult = runQuantizedEmbedding(device, dtype, 1e-6f)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedEmbeddingMetal(): Result<Unit> {
    val device =
        when (val tryResult = Device.newMetal(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (dtype in listOf(GgmlDType.Q4_0, GgmlDType.Q4_1, GgmlDType.Q5_0, GgmlDType.Q5_1, GgmlDType.Q8_0, GgmlDType.Q2K, GgmlDType.Q3K, GgmlDType.Q4K, GgmlDType.Q5K, GgmlDType.Q6K)) {
        when (val tryResult = runQuantizedEmbedding(device, dtype, 1e-3f)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(Unit)
}

public fun quantizeQ40(device: Device): Result<Unit> {
    val src = (0 until 32 * 4).map({ v -> v.toFloat() }).collect<Vec>()
    val src1 =
        when (val tryResult = Tensor.fromSlice(src, 32 * 4, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src1, GgmlDType.Q4_0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    check(
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625, 11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25, 23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0),
    )
    when (val tryResult = ggmlQuantizationErrorTest(GgmlDType.Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ41(device: Device): Result<Unit> {
    val src = (0 until 32 * 4).map({ v -> v.toFloat() }).collect<Vec>()
    val src1 =
        when (val tryResult = Tensor.fromSlice(src, 32 * 4, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src1, GgmlDType.Q4_1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    check(
        roundVector(
            when (val tryResult = dst.toVec1<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        ) == listOf(0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332, 12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73, 22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0, 34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398, 44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73, 56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066, 66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398, 78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797, 88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066, 100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398, 108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664, 118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996),
    )
    when (val tryResult = ggmlQuantizationErrorTest(GgmlDType.Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ50(device: Device): Result<Unit> {
    val src = (0 until 32 * 4).map({ v -> v.toFloat() }).collect<Vec>()
    val src1 =
        when (val tryResult = Tensor.fromSlice(src, 32 * 4, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src1, GgmlDType.Q5_0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    check(
        roundVector(
            when (val tryResult = dst.toVec1<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        ) == listOf(-0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625, 11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313, 23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438, 35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313, 47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125, 55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313, 65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188, 77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063, 89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063, 119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0),
    )
    when (val tryResult = ggmlQuantizationErrorTest(GgmlDType.Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ51(device: Device): Result<Unit> {
    val src = (0 until 32 * 4).map({ v -> v.toFloat() }).collect<Vec>()
    val src1 =
        when (val tryResult = Tensor.fromSlice(src, 32 * 4, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src1, GgmlDType.Q5_1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    check(
        roundVector(
            when (val tryResult = dst.toVec1<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        ) == listOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0),
    )
    when (val tryResult = ggmlQuantizationErrorTest(GgmlDType.Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun getTestVector2(bound: Float, size: Int, device: Device): Result<Tensor> {
    check(size.isMultipleOf(QK_K)) { "size must be a multiple of $QK_K" }
    val src = (0 until size).map({ v -> (v.toFloat() - size.toFloat() / 2.0f) * bound / (size.toFloat() / 2.0f) }).collect<Vec>()
    check(listOf(src[0], src[size / 2]) == listOf(-bound, 0.0))
    return Tensor.fromVec(src, size, device)
}

/**
 * Round a vector
 */
public fun roundVector(values: List<Float>): List<Float> = values.map({ x -> (1000.0 * x).round() / 1000.0 }).collect<Vec>()

public fun compareWithError(values: List<Float>, expected: List<Float>, tolerance: Float) {
    for ((i, value, expectedValue) in (values.zip(expected)).mapIndexed { index, item -> Pair(index, item) }) {
        val difference = (value - expectedValue).abs()
        check(difference < tolerance) { "Error at index $i: value = $value, expected = $expectedValue. Difference = $difference exceeds tolerance = $tolerance." }
    }
}

/**
 * Creates a vector similar to the ones used in GGML unit tests:
 * https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
 */
public fun createGgmlLikeVector(offset: Float): List<Float> = (0 until GGML_TEST_SIZE).map({ i -> 0.1 + 2.0 * (i.toFloat() + offset).cos() }).toMutableList()

/**
 * Calculates the root mean square error between two vectors
 */
public fun calculateRmse(a: List<Float>, b: List<Float>): Float {
    check(a.size == b.size)
    val sum = (a.zip(b)).map({ (a1, b1) -> (a1 - b1).powi(2) }).sum<Float>().sqrt()
    return sum / a.size.toFloat()
}

/**
 * Similar to the GGML quantization unit test:
 * https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
 */
public fun ggmlQuantizationErrorTest(dtype: GgmlDType, device: Device, maxError: Float): Result<Unit> {
    val src = createGgmlLikeVector(0.0f)
    val src1 =
        when (val tryResult = Tensor.fromSlice(src, GGML_TEST_SIZE, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src1, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val error =
        calculateRmse(
            when (val tryResult = src1.toVec1<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
            when (val tryResult = dst.toVec1<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
        )
    if (error > maxError) {
        bail("Quantization error {} exceeds max error {}", error, maxError)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun imatrixQuantizeQ6k(): Result<Unit> {
    val cpu = Device.Cpu
    var rowCounts = 0.0
    var ncall = 0.0
    var values =
        when (val tryResult = Tensor.zeros(768, DType.F32, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val lhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 512), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(512, 768), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val res =
            when (val tryResult = lhs.matmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values =
            when (
                val tryResult = (
                    values +
                        when (
                            val tryResult =
                                when (val tryResult = res.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sum(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        rowCounts +=
            (
                when (val tryResult = res.dim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toDouble()
        ncall += 1.0
    }
    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    val out =
        when (
            val tryResult = (
                when (val tryResult = (values / rowCounts)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * ncall
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val imatrix =
        when (val tryResult = out.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val xs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 768), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant1 =
        when (val tryResult = quantized.QTensor.quantize(xs, GgmlDType.Q6K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant2 =
        when (val tryResult = quantized.QTensor.quantizeImatrix(xs, imatrix, GgmlDType.Q6K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant1 =
        when (val tryResult = quant1.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant2 =
        when (val tryResult = quant2.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant1 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant2 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(err2 < err1) { "err2 $err2 > err1 $err1" }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun imatrixQuantizeQ5k(): Result<Unit> {
    val cpu = Device.Cpu
    var rowCounts = 0.0
    var ncall = 0.0
    var values =
        when (val tryResult = Tensor.zeros(768, DType.F32, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val lhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 512), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(512, 768), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val res =
            when (val tryResult = lhs.matmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values =
            when (
                val tryResult = (
                    values +
                        when (
                            val tryResult =
                                when (val tryResult = res.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sum(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        rowCounts +=
            (
                when (val tryResult = res.dim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toDouble()
        ncall += 1.0
    }
    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    val out =
        when (
            val tryResult = (
                when (val tryResult = (values / rowCounts)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * ncall
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val imatrix =
        when (val tryResult = out.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val xs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 768), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant1 =
        when (val tryResult = quantized.QTensor.quantize(xs, GgmlDType.Q5K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant2 =
        when (val tryResult = quantized.QTensor.quantizeImatrix(xs, imatrix, GgmlDType.Q5K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant1 =
        when (val tryResult = quant1.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant2 =
        when (val tryResult = quant2.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant1 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant2 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(err2 < err1) { "err2 $err2 > err1 $err1" }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun imatrixQuantizeQ4k(): Result<Unit> {
    // let data =
    // quantized.imatrixFile.loadImatrix("../Llama-3.2-3B-Instruct.imatrix").unwrap();
    // for (name, weights) in &data {
    // println("{name}, {} elems", weights.len());
    // }
    // dbg(&data["blk.0.attnQ.weight"].len());
    val cpu = Device.Cpu
    var rowCounts = 0.0
    var ncall = 0.0
    var values =
        when (val tryResult = Tensor.zeros(768, DType.F32, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val lhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 512), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(512, 768), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val res =
            when (val tryResult = lhs.matmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values =
            when (
                val tryResult = (
                    values +
                        when (
                            val tryResult =
                                when (val tryResult = res.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sum(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        rowCounts +=
            (
                when (val tryResult = res.dim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toDouble()
        ncall += 1.0
    }
    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    val out =
        when (
            val tryResult = (
                when (val tryResult = (values / rowCounts)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * ncall
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val imatrix =
        when (val tryResult = out.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val xs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 768), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant1 =
        when (val tryResult = quantized.QTensor.quantize(xs, GgmlDType.Q4K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant2 =
        when (val tryResult = quantized.QTensor.quantizeImatrix(xs, imatrix, GgmlDType.Q4K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant1 =
        when (val tryResult = quant1.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant2 =
        when (val tryResult = quant2.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant1 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant2 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(err2 < err1) { "err2 $err2 > err1 $err1" }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun imatrixQuantizeQ3k(): Result<Unit> {
    val cpu = Device.Cpu
    var rowCounts = 0.0
    var ncall = 0.0
    var values =
        when (val tryResult = Tensor.zeros(768, DType.F32, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val lhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 512), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(512, 768), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val res =
            when (val tryResult = lhs.matmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values =
            when (
                val tryResult = (
                    values +
                        when (
                            val tryResult =
                                when (val tryResult = res.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sum(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        rowCounts +=
            (
                when (val tryResult = res.dim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toDouble()
        ncall += 1.0
    }
    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    val out =
        when (
            val tryResult = (
                when (val tryResult = (values / rowCounts)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * ncall
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val imatrix =
        when (val tryResult = out.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val xs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 768), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant1 =
        when (val tryResult = quantized.QTensor.quantize(xs, GgmlDType.Q3K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant2 =
        when (val tryResult = quantized.QTensor.quantizeImatrix(xs, imatrix, GgmlDType.Q3K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant1 =
        when (val tryResult = quant1.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant2 =
        when (val tryResult = quant2.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant1 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant2 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(err2 < err1) { "err2 $err2 > err1 $err1" }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun imatrixQuantizeQ2k(): Result<Unit> {
    val cpu = Device.Cpu
    var rowCounts = 0.0
    var ncall = 0.0
    var values =
        when (val tryResult = Tensor.zeros(768, DType.F32, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val lhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 512), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (
                val tryResult =
                    Var.fromTensor(
                        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(512, 768), cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val res =
            when (val tryResult = lhs.matmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values =
            when (
                val tryResult = (
                    values +
                        when (
                            val tryResult =
                                when (val tryResult = res.sqr()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sum(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        rowCounts +=
            (
                when (val tryResult = res.dim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toDouble()
        ncall += 1.0
    }
    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    val out =
        when (
            val tryResult = (
                when (val tryResult = (values / rowCounts)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * ncall
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val imatrix =
        when (val tryResult = out.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val xs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(1024, 768), cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant1 =
        when (val tryResult = quantized.QTensor.quantize(xs, GgmlDType.Q2K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant2 =
        when (val tryResult = quantized.QTensor.quantizeImatrix(xs, imatrix, GgmlDType.Q2K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant1 =
        when (val tryResult = quant1.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dequant2 =
        when (val tryResult = quant2.dequantize(cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant1 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (dequant2 - xs)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.meanAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(err2 < err1) { "err2 $err2 > err1 $err1" }
    return Result.Ok(Unit)
}

public fun quantizeQ2k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q2K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.1f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.499, -0.366, -0.249, 0.0, 0.295, 0.492))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 6.0f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ3k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q3K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.03f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.493, -0.37, -0.243, -0.0, 0.292, 0.492))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 3.5f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ4k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q4K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.017f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.5, -0.373, -0.25, 0.0, 0.288, 0.498))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 4.5f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ5k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q5K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.009f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.5, -0.373, -0.25, 0.0, 0.279, 0.499))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 2.5f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ6k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q6K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.008f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.497, -0.372, -0.25, -0.0, 0.284, 0.5))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 2.0f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizeQ8k(device: Device): Result<Unit> {
    val dtype = GgmlDType.Q8K
    val src =
        when (val tryResult = getTestVector2(0.5f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quant =
        when (val tryResult = quantized.QTensor.quantize(src, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst =
        when (val tryResult = quant.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstF16 =
        when (val tryResult = quant.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dst.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    val src1 =
        when (val tryResult = src.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 =
        when (val tryResult = dst.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dst1.asSlice(), src1.asSlice(), 0.008f)
    // Test some specific values
    check(listOf(src1[0], src1[128], src1[256], src1[512], src1[800], src1[1023]) == listOf(-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344))
    val dst2 = roundVector(dst1)
    check(listOf(dst2[0], dst2[128], dst2[256], dst2[512], dst2[800], dst2[1023]) == listOf(-0.5, -0.374, -0.25, -0.0, 0.283, 0.499))
    val srcBig =
        when (val tryResult = getTestVector2(128.0f, 1024, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val quantBig =
        when (val tryResult = quantized.QTensor.quantize(srcBig, dtype)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig =
        when (val tryResult = quantBig.dequantize(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBigF16 =
        when (val tryResult = quantBig.dequantizeF16(device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult = (
                                                when (val tryResult = dstBig.toDtype(DType.F16)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                } - dstBigF16
                                            )
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.toDtype(DType.F32)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff1 == 0.0)
    val srcBig1 =
        when (val tryResult = srcBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dstBig1 =
        when (val tryResult = dstBig.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    compareWithError(dstBig1.asSlice(), srcBig1.asSlice(), 0.6f)
    when (val tryResult = ggmlQuantizationErrorTest(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

/**
 * Very simple dot product implementation
 */
public fun vecDotReference(a: List<Float>, b: List<Float>): Float = (a.zip(b)).map({ (a1, b1) -> a1 * b1 }).sum()

/**
 * Returns the error achieved by the GGML matmul unit test.
 */
public fun ggmlReferenceMatmulError(dtype: GgmlDType): Result<Float> {
    val err =
        when (dtype) {
            GgmlDType.f32 -> 0.000000
            GgmlDType.f16 -> 0.000010
            GgmlDType.bf16 -> 0.000200
            GgmlDType.q2k -> 0.004086
            GgmlDType.q3k -> 0.017
            GgmlDType.q4k -> 0.002425
            GgmlDType.q5k -> 0.000740
            GgmlDType.q6k -> 0.000952
            GgmlDType.q40 -> 0.001143
            GgmlDType.q41 -> 0.008
            GgmlDType.q50 -> 0.001353
            GgmlDType.q51 -> 0.00149
            GgmlDType.q80 -> 0.000092
            GgmlDType.q81 -> 0.000092
            // Not from the ggml repo.
            GgmlDType.q8k -> 0.00065
        }
    return Result.Ok(err)
}

/**
 * Similar to the GGML matmul unit test:
 * https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
 */
public fun <VecDotType, T : GgmlType> ggmlMatmulErrorTest(): Result<Unit> {
    val a = createGgmlLikeVector(0.0f)
    val b = createGgmlLikeVector(1.0f)
    when (val tryResult = ggmlMatmulErrorTest<T>(a.asSlice(), b.asSlice(), 1.0f)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    // Another example that is more likely to trigger the overflow reported in #1526
    val a1 = (0 until GGML_TEST_SIZE).map({ i -> i.toFloat() / GGML_TEST_SIZE.toFloat() }).collect<Vec>()
    val b1 = (0 until GGML_TEST_SIZE).map({ i -> i.toFloat() / GGML_TEST_SIZE.toFloat() }).collect<Vec>()
    when (val tryResult = ggmlMatmulErrorTest<T>(a1.asSlice(), b1.asSlice(), 2.0f)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun <VecDotType, T : GgmlType> ggmlMatmulErrorTest(a: List<Float>, b: List<Float>, errM: Float, fromFloatTFn: (Any?) -> Any?, vecDotTFn: (Any?) -> Any?, vecDotUnoptTFn: (Any?) -> Any?): Result<Unit> {
    val length = a.size
    val aQuant = mutableListOf(T.zeros(), length / T.BLCK_SIZE)
    val bQuant = mutableListOf(T.VecDotType.zeros(), length / T.VecDotType.BLCK_SIZE)
    fromFloatTFn(a, aQuant)
    T.VecDotType.fromFloat(b, bQuant)
    val result = vecDotTFn(length, aQuant, bQuant)
    val resultUnopt = vecDotUnoptTFn(length, aQuant, bQuant)
    if ((result - resultUnopt).abs() / length.toFloat() > 1e-6f) {
        bail("the opt and unopt vec-dot returned different values, opt: {result} vs unopt: {result_unopt}")
    }
    val dst = mutableListOf(0.0f, 1)
    when (val tryResult = matmul(Triple(1, length, 1), b, aQuant, dst)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val resultMatmul = dst[0]
    if ((resultMatmul - result).abs() / length.toFloat() > 1e-6f) {
        bail("calling matmul vs calling vec-dot directly returned different values, matmul: {result_matmul} vs vec-dot: {result}")
    }
    val referenceResult = vecDotReference(a, b)
    val verifyResult =
        closure@{ result1: Float, source: String ->
            val error = (result1 - referenceResult).abs() / length.toFloat()
            val ggmlError =
                when (val tryResult = ggmlReferenceMatmulError(T.DTYPE)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return@closure Result.Err(tryResult.error)
                } * errM
            if (!error.isFinite() || error > GGML_MAX_DOT_PRODUCT_ERROR) {
                bail("Dot product with dtype {:?} error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}. Source: {source}", T.DTYPE)
            }
            // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
            // => we use a slightly higher error threshold
            val ERROR_LENIENCY: Float = 0.00001f
            if (error - ERROR_LENIENCY > ggmlError) {
                bail("Dot product with dtype {:?} error {error} exceeds ggml reference error {ggml_error}. Source: {source}", T.DTYPE)
            }
            Result.Ok(Unit)
        }
    when (val tryResult = verifyResult(result, "vec-dot")) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = verifyResult(resultMatmul, "matmul")) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMm(): Result<Unit> {
    when (val tryResult = ggmlMatmulErrorTest<Float>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<F16>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<Bf16>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ40>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ41>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ50>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ51>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ80>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = ggmlMatmulErrorTest<kquants.BlockQ81>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

/**
 * generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
 */
public fun getRandomTensors(m: Int, k: Int, n: Int, device: Device): Result<Triple<Tensor, Tensor, Tensor>> {
    val rng = StdRng.seedFromU64(314159265358979)
    val lhs = (0 until m * k).map({ rng.random<Float>() - 0.5 }).collect<Vec>()
    val rhs = (0 until n * k).map({ rng.random<Float>() - 0.5 }).collect<Vec>()
    val lhs1 =
        when (val tryResult = Tensor.fromVec(lhs, Pair(m, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (val tryResult = Tensor.fromVec(rhs, Pair(n, k), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm =
        when (
            val tryResult =
                lhs1.matmul(
                    when (val tryResult = rhs1.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    return Result.Ok(Triple(lhs1, rhs1, mm))
}

// Not implemented on metal
public fun quantizedMatmulQ40Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q4_0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ41Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q4_1)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ50Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q5_0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ51Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q5_1)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ80Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q8_0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ40RepackedCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 4, 4, 256), GgmlDType.Q4_0)

@kotlin.test.Test
public fun quantizedMatmulQ80RepackedCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 4, 4, 256), GgmlDType.Q8_0)

@kotlin.test.Test
public fun quantizedMatmulQ6kRepackedCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 4, 8, 256), GgmlDType.Q6K)

@kotlin.test.Test
public fun quantizedMatmulQ5kRepackedCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 4, 8, 256), GgmlDType.Q5K)

@kotlin.test.Test
public fun quantizedMatmulQ4kRepackedCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 4, 8, 256), GgmlDType.Q4K)

@kotlin.test.Test
public fun quantizedMatmulQ4kRepackedGemvCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 1, 8, 256), GgmlDType.Q4K)

@kotlin.test.Test
public fun quantizedMatmulQ40RepackedGemvCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 1, 4, 256), GgmlDType.Q4_0)

@kotlin.test.Test
public fun quantizedMatmulQ5kRepackedGemvCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 1, 8, 256), GgmlDType.Q5K)

@kotlin.test.Test
public fun quantizedMatmulQ6kRepackedGemvCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 1, 8, 256), GgmlDType.Q6K)

@kotlin.test.Test
public fun quantizedMatmulQ80RepackedGemvCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 1, 4, 256), GgmlDType.Q8_0)

@kotlin.test.Test
public fun quantizedMatmulQ4kRepackFallbackCpu(): Result<Unit> = testMatmul(Device.Cpu, listOf(1, 3, 8, 256), GgmlDType.Q4K)

public fun quantizedMatmulQ81Bis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q8_1)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ2kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q2K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ3kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q3K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ4kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q4K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ5kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q5K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ6kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q6K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun quantizedMatmulQ8kBis(device: Device): Result<Unit> {
    when (val tryResult = testMatmul(device, listOf(1, 3, 4, 256), GgmlDType.Q8K)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ2k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q2K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(0.887, 0.428, 0.219, 1.669))
    when (val tryResult = ggmlMatmulErrorTest<BlockQ2K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ3k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q3K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(1.001, 1.425, -0.311, 1.493))
    when (val tryResult = ggmlMatmulErrorTest<BlockQ3K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ4k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q4K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(1.094, 1.442, -0.196, 1.587))
    when (val tryResult = ggmlMatmulErrorTest<BlockQ4K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ5k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q5K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(1.161, 1.498, -0.175, 1.739))
    // Expected: 0.000740408897
    when (val tryResult = ggmlMatmulErrorTest<BlockQ5K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ6k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q6K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(1.293, 1.497, -0.159, 1.737))
    when (val tryResult = ggmlMatmulErrorTest<BlockQ6K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun quantizedMatmulQ8k(): Result<Unit> {
    val cpu = Device.Cpu
    val (m, k, n) = Triple(11, 512, 21)
    val (lhs, rhs, mm) =
        when (val tryResult = getRandomTensors(m, k, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm.dims() == listOf(m, n))
    val dst =
        when (
            val tryResult =
                when (val tryResult = mm.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst1 = roundVector(listOf(dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]))
    check(dst1 == listOf(1.262, 1.513, -0.208, 1.702))
    val rhs1 =
        when (val tryResult = quantized.QTensor.quantize(rhs, GgmlDType.Q8K)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs2 =
        when (val tryResult = quantized.QMatMul.fromQtensor(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = rhs2.forward(lhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(mm1.dims() == listOf(m, n))
    val dst2 =
        when (
            val tryResult =
                when (val tryResult = mm1.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dst3 = roundVector(listOf(dst2[0], dst2[m * n / 3], dst2[m * n * 2 / 3], dst2[m * n - 1]))
    check(dst3 == listOf(1.241, 1.52, -0.2, 1.7))
    when (val tryResult = ggmlMatmulErrorTest<BlockQ8K>()) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun fromDataDequantMatchesCanonicalWhenCallerPassesCowOwned(device: Device): Result<Unit> {
    val cpu = Device.Cpu
    val n = 1024
    val srcData: List<Float> = (0 until n).map({ i -> ((i.toFloat()) * 0.013f).sin() }).toMutableList()
    val src =
        when (val tryResult = Tensor.fromVec(srcData, n, cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val qtCanonical =
        when (val tryResult = quantized.QTensor.quantize(src, GgmlDType.Q4_0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val canonicalDequant =
        when (
            val tryResult =
                when (val tryResult = qtCanonical.dequantize(cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val bytesOwned: List<UByte> =
        when (val tryResult = qtCanonical.data()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }.toVec()
    val storage =
        when (val tryResult = quantized.QStorage.fromData(bytesOwned, device, GgmlDType.Q4_0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val qtViaFromData =
        when (val tryResult = quantized.QTensor.new(storage, n)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val observedDequant =
        when (
            val tryResult =
                when (val tryResult = qtViaFromData.dequantize(device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff = (canonicalDequant.iter().zip(observedDequant.iter())).map({ (a, b) -> (a - b).abs() }).fold(0.0f, Float::max)
    check(maxDiff < 1e-5) { "QStorage::from_data dequant mismatch on $device (max |Δ| = $maxDiff)" }
    return Result.Ok(Unit)
}

// Repacked aarch64 kernels must agree with the dequantized reference across the m tiers
// that select different paths: 1 (gemv), 4/512 (tiled), 3 (per-row gemv), 23 (tiled + gemv tail).
@kotlin.test.Test
public fun qmatmulRepackMatchesReferenceAcrossM(): Result<Unit> {
    val dev = Device.Cpu
    val (k, n) = Pair(512, 256)
    for (dtype in listOf(GgmlDType.Q4K, GgmlDType.Q5K, GgmlDType.Q6K, GgmlDType.Q8_0)) {
        val weight =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(n, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val qtensor =
            when (val tryResult = quantized.QTensor.quantize(weight, dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val wref =
            when (val tryResult = qtensor.dequantize(dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val matmul =
            when (val tryResult = quantized.QMatMul.fromQtensor(qtensor)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (m in listOf(1, 3, 4, 8, 23, 512)) {
            val lhs =
                when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(m, k), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val got =
                when (val tryResult = matmul.forward(lhs)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val want =
                when (
                    val tryResult =
                        lhs.matmul(
                            when (val tryResult = wref.t()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val diff =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (val tryResult = (got - want)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.abs()
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.maxAll()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toScalar<Float>()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val scale =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = want.abs()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.maxAll()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toScalar<Float>()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(1.0)
            check(diff / scale < 5e-2) { "$dtype m=$m: rel diff ${diff / scale} too large" }
        }
    }
    return Result.Ok(Unit)
}

// Indexed (MoE) gemv must match dequantize + per-pair matmul for stacked expert weights.
// aarch64 only: x86 has no indexed path yet and falls back to dequantize-and-gather.
@kotlin.test.Test
public fun indexedGemvMatchesReference(): Result<Unit> {
    val dev = Device.Cpu
    val (nExperts, nOut, k) = Triple(8, 64, 512)
    for (dtype in listOf(GgmlDType.Q4K, GgmlDType.Q6K, GgmlDType.Q8_0)) {
        val w =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Triple(nExperts, nOut, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val qt =
            when (val tryResult = quantized.QTensor.quantize(w, dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val wref =
            when (val tryResult = qt.dequantize(dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        // 24x4 = 96 pairs exercises the bucketed per-expert matmul path; skewed ids
        // land some experts with dozens of rows and others with none
        for ((batch, topk, xT) in listOf(Triple(3, 4, 1), Triple(2, 2, 2), Triple(24, 4, 1))) {
            val x =
                when (val tryResult = Tensor.rand(-1.0f, 1.0f, Triple(batch, xT, k), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val idsV: List<UInt> = (0 until batch * topk).map({ i -> ((i * i * 3 + 1) % nExperts).toUInt() }).toMutableList()
            val ids =
                when (val tryResult = Tensor.fromVec(idsV.toMutableList(), Pair(batch, topk), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val got =
                when (val tryResult = qt.indexedGemv(x, ids)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.expect("indexed path supported")
            for (b in 0 until batch) {
                for (t in 0 until topk) {
                    val e = idsV[b * topk + t].toInt()
                    val xr =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (val tryResult = x.narrow(0, b, 1)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.narrow(1, t.min(xT - 1), 1)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.squeeze(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val we =
                        when (
                            val tryResult =
                                when (val tryResult = wref.narrow(0, e, 1)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.squeeze(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val want =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        xr.matmul(
                                            when (val tryResult = we.t()) {
                                                is Result.Ok<*> -> tryResult.value
                                                is Result.Err<*> -> return Result.Err(tryResult.error)
                                            },
                                        )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.squeeze(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val g =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult =
                                                when (val tryResult = got.narrow(0, b, 1)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                }.narrow(1, t, 1)
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.squeeze(0)
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.squeeze(0)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val diff =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (
                                            val tryResult =
                                                when (val tryResult = (g - want)) {
                                                    is Result.Ok<*> -> tryResult.value
                                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                                }.abs()
                                        ) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.maxAll()
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.toScalar<Float>()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val scale =
                        when (
                            val tryResult =
                                when (
                                    val tryResult =
                                        when (val tryResult = want.abs()) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        }.maxAll()
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.toScalar<Float>()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.max(1.0)
                    check(diff / scale < 5e-2) { "$dtype b=$b t=$t: rel diff ${diff / scale}" }
                }
            }
        }
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun bf16LhsMatmulMatchesF32(): Result<Unit> {
    val dev = Device.Cpu
    val (n, k) = Pair(64, 512)
    for (m in listOf(1, 4, 9)) {
        val w =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(n, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val qt =
            when (val tryResult = quantized.QTensor.quantize(w, GgmlDType.Q4K)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val mm =
            when (val tryResult = quantized.QMatMul.fromQtensor(qt)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val x =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Triple(1, m, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val y32 =
            when (val tryResult = mm.forward(x)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val ybf =
            when (
                val tryResult =
                    when (
                        val tryResult =
                            mm.forward(
                                when (val tryResult = x.toDtype(DType.BF16)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toDtype(DType.F32)
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val diff =
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = (y32 - ybf)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.abs()
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.maxAll()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toScalar<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val scale =
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = y32.abs()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.maxAll()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toScalar<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.max(1.0)
        check(diff / scale < 3e-2) { "m=$m: rel diff ${diff / scale}" }
    }
    return Result.Ok(Unit)
}

// The batched CPU matmul must produce exactly the same values as running the input one row at a
// time: both paths call the same `vecDot` kernels per output element, only the loop order differs.
// `n` is kept away from the repacked-kernel shapes (multiples of 4/8/16) so that this exercises the
// generic `kQuants.matmul` on every architecture, and `CustomOp1` is applied directly so that the
// float weight types are not dequantized by `QMatMul`.
@kotlin.test.Test
public fun qmatmulBatchedRowsMatchSingleRowCpu(): Result<Unit> {
    val dev = Device.Cpu
    val (n, k) = Pair(23, 512)
    for (dtype in listOf(GgmlDType.F32, GgmlDType.F16, GgmlDType.BF16, GgmlDType.Q4_0, GgmlDType.Q4_1, GgmlDType.Q5_0, GgmlDType.Q5_1, GgmlDType.Q8_0, GgmlDType.Q8_1, GgmlDType.Q2K, GgmlDType.Q3K, GgmlDType.Q4K, GgmlDType.Q5K, GgmlDType.Q6K, GgmlDType.Q8K)) {
        val w =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(n, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val qtensor =
            when (val tryResult = quantized.QTensor.quantize(w, dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (m in listOf(2, 3, 4, 7, 8, 9, 16, 17, 33, 64)) {
            val lhs =
                when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(m, k), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val batched =
                when (
                    val tryResult =
                        when (val tryResult = lhs.applyOp1NoBwd(qtensor)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toVec2<Float>()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            for ((rowIdx, batchedRow) in batched.iter().mapIndexed { index, item -> Pair(index, item) }) {
                val row =
                    when (val tryResult = lhs.narrow(0, rowIdx, 1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val single =
                    when (
                        val tryResult =
                            when (val tryResult = row.applyOp1NoBwd(qtensor)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.toVec2<Float>()
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                check(batchedRow == single[0])
            }
        }
    }
    return Result.Ok(Unit)
}

// `kQuants.matmul` is a safe fn that writes `dst` through a raw pointer from the pool threads, so
// an undersized `dst` has to be rejected up front rather than written out of bounds.
@kotlin.test.Test
public fun qmatmulCpuRejectsUndersizedDst() {
    val panic =
        kotlin.test.assertFails(block = fun() {
            val (m, k, n) = Triple(2, 32, 4)
            val lhs = mutableListOf(1.0f, m * k)
            val rhs = mutableListOf(1.0f, k * n)
            val rhsT = mutableListOf(kquants.BlockQ40.zeros(), n * k / 32)
            kquants.BlockQ40.fromFloat(rhs, rhsT)
            val dst = mutableListOf(0.0f, n)
            kquants.matmul(Triple(m, k, n), lhs, rhsT, dst)
        })
    kotlin.test.assertTrue(panic.message?.contains("dst too small") == true)
}

// Timing harness for the CPU quantized matmul prefill path, not a correctness test. Run with:
// cargo test -p candle-core --release --test quantizedTests -- --ignored --nocapture benchQmatmulPrefillCpu
@kotlin.test.Test
public fun benchQmatmulPrefillCpu(): Result<Unit> {
    val dev = Device.Cpu
    // (dtype, n, k, batch sizes). n = 2040 and n = 8184 are not multiples of 16, which keeps Q4K
    // and Q8_0 on the generic path on x86 even when the repacked kernels are available; the
    // 8184 x 8192 weights (38 MB / 71 MB) do not fit in a typical LLC.
    val cases: List<List<Any?>> = listOf(listOf(GgmlDType.Q4K, 2048, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q8_0, 2048, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q4K, 2040, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q4_0, 2048, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q5K, 2048, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q2K, 2048, 2048, listOf(1, 32, 128, 512)), listOf(GgmlDType.Q4K, 8184, 8192, listOf(1, 32, 128)), listOf(GgmlDType.Q8_0, 8184, 8192, listOf(1, 32, 128)))
    for ((dtype, n, k, ms) in cases) {
        val w =
            when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(n, k), dev)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val matmul =
            when (
                val tryResult =
                    quantized.QMatMul.fromQtensor(
                        when (val tryResult = quantized.QTensor.quantize(w, dtype)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (m in ms) {
            val lhs =
                when (val tryResult = Tensor.rand(-1.0f, 1.0f, Pair(m, k), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            for (unused in 0 until 3) {
                when (val tryResult = matmul.forward(lhs)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            val iters = (4096 / m).clamp(4, 200)
            val start = Instant.now()
            for (unused1 in 0 until iters) {
                when (val tryResult = matmul.forward(lhs)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            val secs = start.elapsed().asSecsF64() / iters.toDouble()
            println("$dtype n=$n k=$k m=$m: ${secs * 1e3} ms/call ${m.toDouble() / secs} rows/s")
        }
    }
    return Result.Ok(Unit)
}

private const val QUANTIZED_MATMUL_TEMPLATE: String = "fun \$fnName(device:& Device)-> Result<()> {testMatmul(device, (1, 3, 4, 256), \$dtype)?; Ok(())} testDevice(\$fnName, \$fnNameCpu, \$fnNameCuda, \$fnNameMetal);"

public fun quantizedMatmulMacro(fnName: String = "", fnNameCpu: String = "", fnNameCuda: String = "", fnNameMetal: String = "", dtype: String = ""): String =
    arrayOf("\$fnName" to fnName, "\$fnNameCpu" to fnNameCpu, "\$fnNameCuda" to fnNameCuda, "\$fnNameMetal" to fnNameMetal, "\$dtype" to dtype).fold(QUANTIZED_MATMUL_TEMPLATE) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
