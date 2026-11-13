package io.github.kotlinmania.candle.tensortests

import io.github.kotlinmania.candlecore.D
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Storage
import io.github.kotlinmania.candlecore.StorageRef
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.bail
import io.github.kotlinmania.candlecore.cuda.CudaError
import io.github.kotlinmania.candlecore.testutils.assertTensorEq
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import io.github.kotlinmania.float8.F8E4M3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.half.f16.fromF32
import io.github.kotlinmania.rand.rng
import io.github.kotlinmania.rustlang.core.result.andThen

public fun zeros(device: Device): Result<Unit> {
    val tensor =
        when (val tryResult = Tensor.zeros(Pair(5, 2), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (dim1, dim2) =
        when (val tryResult = tensor.dims2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(dim1 == 5)
    check(dim2 == 2)
    return Result.Ok(Unit)
}

public fun ones(device: Device): Result<Unit> {
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.U8, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 1, 1), listOf(1, 1, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.U32, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 1, 1), listOf(1, 1, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.I64, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 1, 1), listOf(1, 1, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.F32, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, 1.0)),
    )
    if (!device.isMetal()) {
        check(
            when (
                val tryResult =
                    when (val tryResult = Tensor.ones(Pair(2, 3), DType.F64, device)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Double>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, 1.0)),
        )
    }
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.F16, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<F16>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(fromF32(1.0), fromF32(1.0), fromF32(1.0)), listOf(fromF32(1.0), fromF32(1.0), fromF32(1.0))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.ones(Pair(2, 3), DType.BF16, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Bf16>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            listOf(
                listOf(
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                ),
                listOf(
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                    io.github.kotlinmania.half.bf16
                        .fromF32(1.0),
                ),
            ),
    )
    if (!device.isMetal()) {
        check(
            when (
                val tryResult =
                    when (val tryResult = Tensor.ones(Pair(2, 3), DType.F8E4M3, device)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<F8e4m3>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(F8e4m3.fromF32(1.0), F8e4m3.fromF32(1.0), F8e4m3.fromF32(1.0)), listOf(F8e4m3.fromF32(1.0), F8e4m3.fromF32(1.0), F8e4m3.fromF32(1.0))),
        )
    }
    return Result.Ok(Unit)
}

public fun full(device: Device): Result<Unit> {
    val tensor =
        when (val tryResult = Tensor.zeros(Pair(3, 4), DType.U32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = tensor.constSet(42u.into())) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = tensor.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(42, 42, 42, 42), listOf(42, 42, 42, 42), listOf(42, 42, 42, 42)),
    )
    when (
        val tryResult =
            when (val tryResult = tensor.i(Pair(0 until 0, 2))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.constSet(1337u.into())
    ) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = tensor.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(42, 42, 1337, 42), listOf(42, 42, 1337, 42), listOf(42, 42, 1337, 42)),
    )
    when (
        val tryResult =
            when (val tryResult = tensor.i(Pair(2, 0 until 0))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.constSet(1u.into())
    ) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = tensor.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(42, 42, 1337, 42), listOf(42, 42, 1337, 42), listOf(1, 1, 1, 1)),
    )
    return Result.Ok(Unit)
}

public fun constSet(device: Device): Result<Unit> {
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.full(42u, Pair(2, 3), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(42, 42, 42), listOf(42, 42, 42)),
    )
    return Result.Ok(Unit)
}

public fun arange(device: Device): Result<Unit> {
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u.toUByte(), 5u.toUByte(), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 1, 2, 3, 4),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.arangeStep(0u.toUByte(), 5u.toUByte(), 2, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 2, 4),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.arangeStep(0u.toUByte(), 5u.toUByte(), 3, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 3),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.arangeStep(5L, 0L, -1, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(5, 4, 3, 2, 1),
    )
    if (!device.isMetal()) {
        check(
            when (
                val tryResult =
                    when (val tryResult = Tensor.arangeStep(F8e4m3.fromF32(0.0), F8e4m3.fromF32(5.0), F8e4m3.fromF32(2.0), device)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec1<F8e4m3>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(F8e4m3.fromF32(0.0), F8e4m3.fromF32(2.0), F8e4m3.fromF32(4.0)),
        )
    }
    return Result.Ok(Unit)
}

public fun addMul(device: Device): Result<Unit> {
    val tensor =
        when (val tryResult = Tensor.new(listOf(3.0f, 1.0, 4.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dim1 =
        when (val tryResult = tensor.dims1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(dim1 == 3)
    val content: List<Float> =
        when (val tryResult = tensor.toVec1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content == listOf(3.0, 1.0, 4.0))
    val tensor1 =
        when (val tryResult = Tensor.add(tensor, tensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val content1: List<Float> =
        when (val tryResult = tensor1.toVec1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content1 == listOf(6.0, 2.0, 8.0))
    val tensor2 =
        when (val tryResult = Tensor.mul(tensor1, tensor1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val content2: List<Float> =
        when (val tryResult = tensor2.toVec1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content2 == listOf(36.0, 4.0, 64.0))
    return Result.Ok(Unit)
}

public fun tensor2d(device: Device): Result<Unit> {
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dims =
        when (val tryResult = tensor.dims2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(dims == Pair(2, 5))
    val content: List<List<Float>> =
        when (val tryResult = tensor.toVec2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content == data)
    return Result.Ok(Unit)
}

public fun clamp(device: Device): Result<Unit> {
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor1 =
        when (val tryResult = tensor.clamp(1.5, 6.2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tensor1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 1.5, 4.0, 1.5, 5.0), listOf(2.0, 1.5, 6.2, 6.2, 2.0)),
    )
    return Result.Ok(Unit)
}

public fun asort(device: Device): Result<Unit> {
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.1, 5.0), listOf(2.1, 1.0, 7.0, 8.0, 2.0))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val indexes =
        when (val tryResult = tensor.argSortLastDim(true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 3, 0, 2, 4), listOf(1, 4, 0, 2, 3)),
    )
    val indexes1 =
        when (val tryResult = tensor.argSortLastDim(false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes1.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(4, 2, 0, 3, 1), listOf(3, 2, 0, 4, 1)),
    )
    val (sorted, indexes2) =
        when (val tryResult = tensor.sortLastDim(true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes2.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 3, 0, 2, 4), listOf(1, 4, 0, 2, 3)),
    )
    check(
        when (val tryResult = sorted.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.1, 3.0, 4.0, 5.0), listOf(1.0, 2.0, 2.1, 7.0, 8.0)),
    )
    val (sorted1, indexes3) =
        when (val tryResult = tensor.sortLastDim(false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes3.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(4, 2, 0, 3, 1), listOf(3, 2, 0, 4, 1)),
    )
    check(
        when (val tryResult = sorted1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(5.0, 4.0, 3.0, 1.1, 1.0), listOf(8.0, 7.0, 2.1, 2.0, 1.0)),
    )
    val offsetView =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(listOf(1.0f, 2.0, 3.0), listOf(30.0, 10.0, 20.0)), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(0, 1, 1)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (sorted2, indexes4) =
        when (val tryResult = offsetView.sortLastDim(true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes4.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 2, 0)),
    )
    check(
        when (val tryResult = sorted2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(10.0, 20.0, 30.0)),
    )
    val (sorted3, indexes5) =
        when (val tryResult = offsetView.sortLastDim(false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = indexes5.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 2, 1)),
    )
    check(
        when (val tryResult = sorted3.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(30.0, 20.0, 10.0)),
    )
    return Result.Ok(Unit)
}

/**
 * Test sorting a large tensor that exceeds 1024 elements.
 */
public fun asortBig(device: Device): Result<Unit> {
    val SIZE: Int = 2000
    // Skip on metal for now
    if (device.isMetal()) {
        return Result.Ok(Unit)
    }
    val data: List<Float> = (0 until SIZE).map({ x -> (SIZE - x).toFloat() }).toMutableList()
    val tensor =
        when (val tryResult = Tensor.new(data.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val indexes =
        when (val tryResult = tensor.argSortLastDim(true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedIndexes: List<UInt> = (0 until SIZE).rev().map({ x -> x.toUInt() }).toMutableList()
    check(
        when (val tryResult = indexes.toVec1<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expectedIndexes,
    )
    val indexes1 =
        when (val tryResult = tensor.argSortLastDim(false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedIndexes1: List<UInt> = (0 until SIZE).map({ x -> x.toUInt() }).toMutableList()
    check(
        when (val tryResult = indexes1.toVec1<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expectedIndexes1,
    )
    return Result.Ok(Unit)
}

public fun unaryOp(device: Device): Result<Unit> {
    val data = listOf(listOf(-3.0f, 1.0, 4.0, -0.1, 0.5), listOf(2.7, -1.8, -0.28, 1.8, 2.8))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.gelu()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-0.0036, 0.8412, 3.9999, -0.046, 0.3457), listOf(2.6911, -0.0647, -0.1091, 1.7353, 2.7933)),
    )
    val tF16 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.toDtype(DType.F16)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.gelu()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.F32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult = (
                                when (val tryResult = tensor.gelu()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                } - tF16
                            )
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = maxDiff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } < 5e-3,
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.geluErf()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-0.004, 0.8413, 3.9999, -0.046, 0.3457), listOf(2.6906, -0.0647, -0.1091, 1.7353, 2.7928)),
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.erf()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-1.0, 0.8427, 1.0, -0.1125, 0.5205), listOf(0.9999, -0.9891, -0.3079, 0.9891, 0.9999)),
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.silu()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-0.1423, 0.7311, 3.9281, -0.0475, 0.3112), listOf(2.53, -0.2553, -0.1205, 1.5447, 2.6395)),
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.ceil()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-3.0, 1.0, 4.0, -0.0, 1.0), listOf(3.0, -1.0, -0.0, 2.0, 3.0)),
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.floor()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-3.0, 1.0, 4.0, -1.0, 0.0), listOf(2.0, -2.0, -1.0, 1.0, 2.0)),
    )
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.round()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(-3.0, 1.0, 4.0, -0.0, 1.0), listOf(3.0, -2.0, -0.0, 2.0, 3.0)),
    )
    val tensor1 =
        when (val tryResult = Tensor.new(listOf(2997.9246, 314.15926f), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec1Round(
                    when (val tryResult = tensor1.roundTo(2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2997.92, 314.16),
    )
    check(
        when (
            val tryResult =
                testutils.toVec1Round(
                    when (val tryResult = tensor1.roundTo(-2)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3000.0, 300.0),
    )
    val tensor2 =
        when (val tryResult = Tensor.new(listOf(-1.01f, -0.9, -0.1, 0.0, -0.0, 0.1, 0.9, 1.0, 1.1), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.sign()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    )
    val tensor3 =
        when (val tryResult = Tensor.new(listOf(-1.0f, 0.0, -2.0, 3.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val y =
        when (val tryResult = tensor3.elu(2.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.2642, 0.0000, -1.7293, 3.0000),
    )
    // This test failed on metal prior to the following PR:
    // https://github.com/huggingface/candle/pull/2490
    val y1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = tensor3.reshape(Pair(2, 2))) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.t()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.elu(2.0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.flattenAll()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec1Round(y1, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(-1.2642, -1.7293, 0.0000, 3.0000),
    )
    return Result.Ok(Unit)
}

public fun binaryOp(device: Device): Result<Unit> {
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val tensor1 =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data2 = listOf(listOf(5.0f, 5.0, 5.0, 5.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val tensor2 =
        when (val tryResult = Tensor.new(data2, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor =
        when (
            val tryResult = (
                tensor1 + when (val tryResult = (tensor1 * tensor1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } / (tensor1 + tensor2)
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dims =
        when (val tryResult = tensor.dims2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(dims == Pair(2, 5))
    val content: List<List<Float>> =
        when (val tryResult = tensor.toVec2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content[0] == listOf(4.125, 1.1666666, 5.7777777, 1.1666666, 7.5))
    check(content[1] == listOf(3.0, 1.5, 10.5, 12.0, 3.0))
    val tensor1 =
        when (val tryResult = (tensor - tensor)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val content1: List<List<Float>> =
        when (val tryResult = tensor1.toVec2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(content1[0] == listOf(0.0, 0.0, 0.0, 0.0, 0.0))
    val min =
        when (
            val tryResult =
                tensor1.minimum(
                    when (val tryResult = (tensor2 * 0.5)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val max =
        when (
            val tryResult =
                tensor1.maximum(
                    when (val tryResult = (tensor2 * 0.5)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = min.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(2.5, 1.0, 2.5, 1.0, 2.5), listOf(1.0, 0.5, 3.5, 4.0, 1.0)),
    )
    check(
        when (val tryResult = max.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 2.5, 4.0, 2.5, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0)),
    )
    val (b, s, h, pad) = listOf(3, 4, 5, 2)
    val base =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(1.0f, (b * (s + pad) * h + 1).toFloat(), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(b, s + pad, h))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ub =
        when (val tryResult = base.narrow(1, 0, s)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // uniform blocks: strides [h * (s + pad), h, 1]
    val cont =
        when (val tryResult = ub.contiguous()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val scale =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(1.0f, (b * s * h + 1).toFloat(), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(b, s, h))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = (ub + scale)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (val tryResult = (cont + scale)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    check(
        when (
            val tryResult =
                when (val tryResult = (ub * scale)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (val tryResult = (cont * scale)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    // contiguous lhs, uniform blocks rhs.
    check(
        when (
            val tryResult =
                when (val tryResult = (scale + ub)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (val tryResult = (scale + cont)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    return Result.Ok(Unit)
}

public fun ternaryOp(device: Device): Result<Unit> {
    val data = listOf(listOf(0u.toUByte(), 1, 0, 1, 0), listOf(1, 1, 1, 0, 0))
    val ids =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data1 = listOf(listOf(0.0f, 1.0, 2.0, 3.0, 4.0), listOf(5.0, 6.0, 7.0, 8.0, 9.0))
    val a =
        when (val tryResult = Tensor.new(data1, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data2 = listOf(listOf(10.0f, 11.0, 12.0, 13.0, 14.0), listOf(15.0, 16.0, 17.0, 18.0, 19.0))
    val b =
        when (val tryResult = Tensor.new(data2, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor =
        when (val tryResult = ids.whereCond(a, b)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dims = tensor.dims()
    check(dims == listOf(2, 5))
    val result: List<Float> =
        when (
            val tryResult =
                when (val tryResult = tensor.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result == listOf(10.0, 1.0, 12.0, 3.0, 14.0, 5.0, 6.0, 7.0, 18.0, 19.0))
    val fill =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(7.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAs(Pair(2, 5))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // strides [0, 0]
    check(
        when (
            val tryResult =
                when (val tryResult = ids.whereCond(fill, b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(10.0, 7.0, 12.0, 7.0, 14.0), listOf(7.0, 7.0, 7.0, 18.0, 19.0)),
    )
    val mask =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(listOf(0u.toUByte(), 1, 1, 0, 1)), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAs(Pair(2, 5))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // strides [0, 1]
    check(
        when (
            val tryResult =
                when (val tryResult = mask.whereCond(a, b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = mask.contiguous()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.whereCond(a, b)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val aT =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = Tensor.arange(0.0f, 10.0, device)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.reshape(Pair(5, 2))
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val idsT =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(listOf(0u.toUByte(), 1), listOf(1, 1), listOf(0, 0), listOf(1, 0), listOf(1, 1)), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = idsT.whereCond(aT, b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = idsT.contiguous()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.whereCond(
                                when (val tryResult = aT.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                                b,
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val narrowed =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = Tensor.arange(0.0f, 14.0, device)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.reshape(Pair(2, 7))
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(1, 1, 5)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // strides [7, 1], offset 1
    check(
        when (
            val tryResult =
                when (val tryResult = ids.whereCond(fill, narrowed)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            ids.whereCond(
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                                when (val tryResult = narrowed.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val fill2 =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(9.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAs(Pair(2, 5))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = ids.whereCond(a, fill)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            ids.whereCond(
                                a,
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    check(
        when (
            val tryResult =
                when (val tryResult = ids.whereCond(fill, fill2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            ids.whereCond(
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                                when (val tryResult = fill2.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val rowMask =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(listOf(1u.toUByte()), listOf(0u.toUByte())), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAs(Pair(2, 5))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // strides [1, 0]
    val rowRef =
        when (val tryResult = rowMask.contiguous()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = rowMask.whereCond(a, b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (val tryResult = rowRef.whereCond(a, b)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    check(
        when (
            val tryResult =
                when (val tryResult = rowMask.whereCond(a, fill)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            rowRef.whereCond(
                                a,
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    check(
        when (
            val tryResult =
                when (val tryResult = rowMask.whereCond(fill, b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            rowRef.whereCond(
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                                b,
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    check(
        when (
            val tryResult =
                when (val tryResult = rowMask.whereCond(fill, fill2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (
                val tryResult =
                    when (
                        val tryResult =
                            rowRef.whereCond(
                                when (val tryResult = fill.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                                when (val tryResult = fill2.contiguous()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec2<Float>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    return Result.Ok(Unit)
}

public fun transpose(device: Device): Result<Unit> {
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(data, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dims =
        when (val tryResult = tensor.dims2()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(dims == Pair(5, 2))
    check(
        when (val tryResult = tensor.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0f, 2.0), listOf(1.0, 1.0), listOf(4.0, 7.0), listOf(1.0, 8.0), listOf(5.0, 2.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == data,
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.contiguous()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.t()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == data,
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult = (
                        when (
                            val tryResult =
                                when (val tryResult = (tensor + 1.0)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.t()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        } - 1.0
                    )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == data,
    )
    return Result.Ok(Unit)
}

public fun variable(device: Device): Result<Unit> {
    // Values taken from https://pytorch.org/docs/stable/generated/torch.var.html
    val data = listOf(listOf(0.2035f, 1.2959, 1.8101, -0.4644), listOf(1.5027, -0.3270, 0.5905, 0.6538), listOf(-1.5745, 1.3330, -0.5596, -0.6548), listOf(0.1264, -0.5080, 1.6420, 0.1992))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                testutils.toVec2Round(
                    when (val tryResult = tensor.varKeepdim(1)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    4,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0631), listOf(0.559), listOf(1.4893), listOf(0.8258)),
    )
    return Result.Ok(Unit)
}

public fun sum(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3u, 1, 4), listOf(1, 5, 9)), listOf(listOf(2, 1, 7), listOf(8, 2, 8)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.sumKeepdim(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(8), listOf(15)), listOf(listOf(10), listOf(18))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.sumKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(5, 2, 11), listOf(9, 7, 17))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.sumKeepdim(Triple(0, 2, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(51))),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = tensor.t()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sumKeepdim(1)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.t()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(8), listOf(15)), listOf(listOf(10), listOf(18))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.sumKeepdim(Pair(2, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(8 + 15)), listOf(listOf(10 + 18))),
    )
    val data1: List<UInt> = (0 until 4000u).toMutableList()
    val tensor1 =
        when (val tryResult = Tensor.new(data1.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor1.sumKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(7998000),
    )
    val tensor2 =
        when (val tryResult = tensor1.reshape(Pair(2000, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.sumKeepdim(Pair(0, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.sumKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.sumKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.sumKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3998000, 4000000)),
    )
    // Make the tensor non contiguous.
    val tensor3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.sumKeepdim(Pair(0, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.sumKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.sumKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7998000)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.sumKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3998000, 4000000)),
    )
    val t1 =
        when (val tryResult = tensor3.reshape(Triple(200, 5, 4))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(0, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (tensor4 in listOf(t1, t2)) {
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.sumKeepdim(Triple(0, 1, 2))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(7998000))),
        )
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = tensor4.sumKeepdim(0)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.sumKeepdim(2)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.sumKeepdim(1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(7998000))),
        )
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = tensor4.sumKeepdim(0)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.sumKeepdim(Pair(1, 2))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(7998000))),
        )
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (val tryResult = tensor4.sumKeepdim(1)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.sumKeepdim(Pair(0, 2))
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(7998000))),
        )
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.sumKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(398000, 398200, 398400, 398600), listOf(398800, 399000, 399200, 399400), listOf(399600, 399800, 400000, 400200), listOf(400400, 400600, 400800, 401000), listOf(401200, 401400, 401600, 401800))),
        )
    }
    val data2 = listOf(listOf(1u.toUByte(), 2, 3), listOf(4, 5, 6))
    val tensor5 =
        when (val tryResult = Tensor.new(data2, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor5.sumKeepdim(1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(6), listOf(15)),
    )
    val data3 = listOf(listOf(1L, 2, 3), listOf(4, 5, 6))
    val tensor6 =
        when (val tryResult = Tensor.new(data3, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor6.sumKeepdim(1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(6), listOf(15)),
    )
    val data4 = mutableListOf(16777217u)
    data4.extend(List(32) { 1u })
    val tensor7 =
        when (val tryResult = Tensor.new(data4.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor7.sumKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(16777249),
    )
    if (!device.isMetal()) {
        val data5 = mutableListOf(16777217.0)
        data5.extend(List(32) { 1.0 })
        val tensor8 =
            when (val tryResult = Tensor.new(data5.asSlice(), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor8.sumKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec1<Double>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(16777249.0),
        )
    }
    val empty =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(List(4) { listOf(1.0f, 2.0) }, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(1, 0, 0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = empty.sum(1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == List(4) { 0.0 },
    )
    return Result.Ok(Unit)
}

public fun min(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3u, 1, 4), listOf(1, 5, 9)), listOf(listOf(2, 1, 7), listOf(8, 2, 8)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.minKeepdim(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1), listOf(1)), listOf(listOf(1), listOf(2))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.minKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(2, 1, 4), listOf(1, 2, 8))),
    )
    val data1: List<UInt> = (200 until 4000u).toMutableList()
    val tensor1 =
        when (val tryResult = Tensor.new(data1.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor1.minKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(200),
    )
    val tensor2 =
        when (val tryResult = tensor1.reshape(Pair(1900, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.minKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.minKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.minKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.minKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.minKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200, 201)),
    )
    // Make the tensor non contiguous.
    val tensor3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.minKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.minKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.minKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.minKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.minKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(200, 201)),
    )
    val t1 =
        when (val tryResult = tensor3.reshape(Triple(190, 5, 4))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(0, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (tensor4 in listOf(t1, t2)) {
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = tensor4.minKeepdim(0)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.minKeepdim(2)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.minKeepdim(1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(200))),
        )
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.minKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(200, 201, 202, 203), listOf(204, 205, 206, 207), listOf(208, 209, 210, 211), listOf(212, 213, 214, 215), listOf(216, 217, 218, 219))),
        )
    }
    return Result.Ok(Unit)
}

public fun max(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3u, 1, 4), listOf(1, 5, 9)), listOf(listOf(2, 1, 7), listOf(8, 2, 8)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.maxKeepdim(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(4), listOf(9)), listOf(listOf(7), listOf(8))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.maxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(3, 1, 7), listOf(8, 5, 9))),
    )
    val data1: List<UInt> = (200 until 4000u).toMutableList()
    val tensor1 =
        when (val tryResult = Tensor.new(data1.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor1.maxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3999),
    )
    val tensor2 =
        when (val tryResult = tensor1.reshape(Pair(1900, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.maxKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.maxKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3999)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.maxKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.maxKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3999)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.maxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3998, 3999)),
    )
    // Make the tensor non contiguous.
    val tensor3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.maxKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.maxKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3999)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.maxKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.maxKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3999)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.maxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3998, 3999)),
    )
    val t1 =
        when (val tryResult = tensor3.reshape(Triple(190, 5, 4))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(0, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (tensor4 in listOf(t1, t2)) {
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = tensor4.maxKeepdim(0)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.maxKeepdim(2)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.maxKeepdim(1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(3999))),
        )
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.maxKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(3980, 3981, 3982, 3983), listOf(3984, 3985, 3986, 3987), listOf(3988, 3989, 3990, 3991), listOf(3992, 3993, 3994, 3995), listOf(3996, 3997, 3998, 3999))),
        )
    }
    return Result.Ok(Unit)
}

public fun argmin(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3u, 1, 4), listOf(1, 5, 9)), listOf(listOf(2, 1, 7), listOf(8, 2, 8)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.argminKeepdim(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1), listOf(0)), listOf(listOf(1), listOf(1))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.argminKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1, 0, 0), listOf(0, 1, 1))),
    )
    val data1: List<UInt> = (200 until 4000u).toMutableList()
    val tensor1 =
        when (val tryResult = Tensor.new(data1.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor1.argminKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0),
    )
    val tensor2 =
        when (val tryResult = tensor1.reshape(Pair(1900, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.argminKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argminKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.argminKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argminKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.argminKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 0)),
    )
    // Make the tensor non contiguous.
    val tensor3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.argminKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argminKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.argminKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argminKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.argminKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 0)),
    )
    val t1 =
        when (val tryResult = tensor3.reshape(Triple(190, 5, 4))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(0, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (tensor4 in listOf(t1, t2)) {
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = tensor4.argminKeepdim(0)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.argminKeepdim(2)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.argminKeepdim(1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(0))),
        )
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.argminKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(0, 0, 0, 0), listOf(0, 0, 0, 0), listOf(0, 0, 0, 0), listOf(0, 0, 0, 0), listOf(0, 0, 0, 0))),
        )
    }
    return Result.Ok(Unit)
}

public fun argmax(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3u, 1, 4), listOf(1, 5, 9)), listOf(listOf(2, 1, 7), listOf(8, 2, 8)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.argmaxKeepdim(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(2), listOf(2)), listOf(listOf(2), listOf(0))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.argmaxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0, 0, 1), listOf(1, 0, 0))),
    )
    val data1: List<UInt> = (200 until 4000u).toMutableList()
    val tensor1 =
        when (val tryResult = Tensor.new(data1.asSlice(), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor1.argmaxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3799),
    )
    val tensor2 =
        when (val tryResult = tensor1.reshape(Pair(1900, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.argmaxKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argmaxKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.argmaxKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argmaxKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor2.argmaxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1899, 1899)),
    )
    // Make the tensor non contiguous.
    val tensor3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor2.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.argmaxKeepdim(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argmaxKeepdim(1)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor3.argmaxKeepdim(1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.argmaxKeepdim(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor3.argmaxKeepdim(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1899, 1899)),
    )
    val t1 =
        when (val tryResult = tensor3.reshape(Triple(190, 5, 4))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(0, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (tensor4 in listOf(t1, t2)) {
        check(
            when (
                val tryResult =
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = tensor4.argmaxKeepdim(0)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                    }.argmaxKeepdim(2)
                            ) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }.argmaxKeepdim(1)
                    ) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(0))),
        )
        check(
            when (
                val tryResult =
                    when (val tryResult = tensor4.argmaxKeepdim(0)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.toVec3<UInt>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(listOf(189, 189, 189, 189), listOf(189, 189, 189, 189), listOf(189, 189, 189, 189), listOf(189, 189, 189, 189), listOf(189, 189, 189, 189))),
        )
    }
    return Result.Ok(Unit)
}

public fun narrow(device: Device): Result<Unit> {
    val data = listOf(listOf(listOf(3.0f, 1.0, 4.0), listOf(1.0, 5.0, 9.0)), listOf(listOf(2.0, 1.0, 7.0), listOf(8.0, 2.0, 8.0)))
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.narrow(2, 1, 2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1.0, 4.0), listOf(5.0, 9.0)), listOf(listOf(1.0, 7.0), listOf(2.0, 8.0))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.narrow(1, 1, 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1.0, 5.0, 9.0)), listOf(listOf(8.0, 2.0, 8.0))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.narrow(0, 0, 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(3.0, 1.0, 4.0), listOf(1.0, 5.0, 9.0))),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.narrow(0, 1, 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(2.0, 1.0, 7.0), listOf(8.0, 2.0, 8.0))),
    )
    // The following has been checked against PyTorch via:
    // import torch
    // t = torch.tensor([[[3., 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]])
    // t.transpose(-1, -2).narrow(1, 1, 2)
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.narrow(1, 1, 2)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1.0, 5.0), listOf(4.0, 9.0)), listOf(listOf(1.0, 2.0), listOf(7.0, 8.0))),
    )
    return Result.Ok(Unit)
}

public fun broadcast(device: Device): Result<Unit> {
    val data = listOf(3.0f, 1.0, 4.0)
    val tensor =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.broadcastLeft(Pair(3, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(3.0, 1.0, 4.0)), listOf(listOf(3.0, 1.0, 4.0)), listOf(listOf(3.0, 1.0, 4.0))),
    )
    return Result.Ok(Unit)
}

public fun sliceSet(device: Device): Result<Unit> {
    val (b, h, maxT, d) = listOf(2, 4, 7, 3)
    val cache =
        when (val tryResult = Tensor.zeros(listOf(b, h, maxT, d), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, listOf(b, h, 4, d), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = cache.sliceSet(tensor, 2, 0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val cacheT =
        when (val tryResult = cache.narrow(2, 0, 4)) {
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
                                when (val tryResult = (cacheT - tensor)) {
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
    when (val tryResult = cache.sliceSet(tensor, 2, 1)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val cacheT1 =
        when (val tryResult = cache.narrow(2, 1, 4)) {
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
                                when (val tryResult = (cacheT1 - tensor)) {
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
    val ones =
        when (val tryResult = Tensor.ones(listOf(b, h, 1, d), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = cache.sliceSet(ones, 2, 6)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val diff2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = cache.narrow(2, 5, 1)) {
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
    val diff3 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = cache.narrow(2, 6, 1)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } - 1.0
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
    // This used to create a deadlock rather than returning an actual error.
    check(cache.sliceSet(cache, 0, 0).isErr())
    return Result.Ok(Unit)
}

public fun cat(device: Device): Result<Unit> {
    // 1D
    val t1 =
        when (val tryResult = Tensor.new(listOf(3.0f, 1.0, 4.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (val tryResult = Tensor.new(listOf(1.0f, 5.0, 9.0, 2.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t3 =
        when (val tryResult = Tensor.new(listOf(6.0f, 5.0, 3.0, 5.0, 8.0, 9.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.cat(listOf(t1), 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0f, 1.0, 4.0),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.cat(listOf(t1, t2), 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0f, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.cat(listOf(t1, t2, t3), 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0f, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0),
    )
    // 2D
    val data = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0))
    val t11 =
        when (val tryResult = Tensor.new(data, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data2 = listOf(listOf(5.0f, 5.0, 5.0, 5.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0))
    val t21 =
        when (val tryResult = Tensor.new(data2, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.cat(listOf(t11, t21), 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0), listOf(5.0, 5.0, 5.0, 5.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0)),
    )
    // PyTorch equivalent:
    // import torch
    // t1 = torch.tensor([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2]])
    // t2 = torch.tensor([[5]*5, [2, 7, 1, 8, 2]])
    // torch.cat([t1.t(), t2.t()], dim=1).t()
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                Tensor.cat(
                                    listOf(
                                        when (val tryResult = t11.t()) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        },
                                        when (val tryResult = t21.t()) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        },
                                    ),
                                    1,
                                )
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.t()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0), listOf(5.0, 5.0, 5.0, 5.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = Tensor.cat(listOf(t11, t21), 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 1.0, 4.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0), listOf(2.0, 7.0, 1.0, 8.0, 2.0, 2.0, 7.0, 1.0, 8.0, 2.0)),
    )
    // 3D
    val t12 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0, 48L, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 6, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t22 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(100, 124L, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t31 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(10000, 10032L, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tCat =
        when (val tryResult = Tensor.cat(listOf(t12, t22, t31), 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t13 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t12.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t23 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t22.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t32 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t31.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tCat2 =
        when (val tryResult = Tensor.cat(listOf(t13, t23, t32), 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tCat.eq(tCat2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toDtype(DType.F32)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.sumAll()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 104.0,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 0, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 0,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 4, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 16,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 5, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 20,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(1, 5, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 44,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 6, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 100,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(1, 6, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 112,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 6, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 101,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 7, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 105,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(0, 12, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 10013,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tCat.i(Triple(1, 12, 3))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Long>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 10031,
    )
    // compare contiguous to uniform blocks
    val (b, s, h, pad) = listOf(3, 4, 5, 2)
    val base =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, (b * (s + pad) * h).toFloat(), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(b, s + pad, h))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tp1 =
        when (val tryResult = base.narrow(1, 0, s)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tp2 =
        when (val tryResult = base.narrow(1, pad, s)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val catPadded =
        when (val tryResult = Tensor.cat(listOf(tp1, tp2), 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val catCont =
        when (
            val tryResult =
                Tensor.cat(
                    listOf(
                        when (val tryResult = tp1.contiguous()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                        when (val tryResult = tp2.contiguous()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    ),
                    1,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = catPadded.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (val tryResult = catCont.toVec3<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    return Result.Ok(Unit)
}

public fun embeddings(device: Device): Result<Unit> {
    val ids =
        when (val tryResult = Tensor.new(listOf(0u, 2u, 1u), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t =
        when (val tryResult = Tensor.new(listOf(listOf(0.0f, 1.0f), listOf(2.0f, 3.0f), listOf(4.0f, 5.0f)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs =
        when (val tryResult = t.embedding(ids)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0), listOf(4.0, 5.0), listOf(2.0, 3.0)),
    )
    val hs1 =
        when (val tryResult = t.indexSelect(ids, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0), listOf(4.0, 5.0), listOf(2.0, 3.0)),
    )
    val hs2 =
        when (
            val tryResult =
                t.indexSelect(
                    when (val tryResult = ids.toDtype(DType.I64)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    0,
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0), listOf(4.0, 5.0), listOf(2.0, 3.0)),
    )
    val ids1 =
        when (val tryResult = Tensor.new(listOf(UInt.MAX_VALUE, 2u, UInt.MAX_VALUE), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs3 =
        when (val tryResult = t.indexSelect(ids1, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs3.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 0.0), listOf(4.0, 5.0), listOf(0.0, 0.0)),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun indexSelectFail(): Result<Unit> {
    // Check that an error is properly reported on out of bounds.
    val ids =
        when (val tryResult = Tensor.new(listOf(4u, 2u, 1u), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t =
        when (val tryResult = Tensor.new(listOf(listOf(0.0f, 1.0f), listOf(2.0f, 3.0f), listOf(4.0f, 5.0f)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs = t.indexSelect(ids, 0)
    check(hs.isErr())
    return Result.Ok(Unit)
}

// The test below triggers an unwinding panic as there is a panic within the
// @cfg(feature = "cuda")
// @Test
// @shouldPanic
// fn indexSelectFailGpu() {
// // Check that a panic happens for out of bounds in cuda
// if let Ok(device) = Device.newCuda(0) {
// if let Ok(ids) = Tensor.new(&[4u32, 2u32, 1u32], &device) {
// if let Ok(t) = Tensor.new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &device) {
// let _ = t.indexSelect(&ids, 0);
// }
// }
// }
// }
public fun cmp(device: Device): Result<Unit> {
    val t1 =
        when (val tryResult = Tensor.new(listOf(listOf(0.0f, 1.0f), listOf(2.0f, 3.0f), listOf(4.0f, 5.0f)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (val tryResult = Tensor.new(listOf(listOf(1.0f, 0.0f), listOf(3.0f, 3.0f), listOf(4.0f, 7.0f)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t1.eq(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 0), listOf(0, 1), listOf(1, 0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t1.ne(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 1), listOf(1, 0), listOf(0, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t1.le(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 0), listOf(1, 1), listOf(1, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t1.lt(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1, 0), listOf(1, 0), listOf(0, 1)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t1.gt(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1), listOf(0, 0), listOf(0, 0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t1.ge(t2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UByte>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1), listOf(0, 1), listOf(1, 0)),
    )
    return Result.Ok(Unit)
}

public fun indexSelect(device: Device): Result<Unit> {
    val ids =
        when (val tryResult = Tensor.new(listOf(0u, 2u, 1u), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 12.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    for (dtype in listOf(DType.U8, DType.U32, DType.I64)) {
        val ids1 =
            when (val tryResult = ids.toDtype(dtype)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val hs =
            when (val tryResult = t.indexSelect(ids1, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = hs.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(0.0, 2.0, 1.0), listOf(3.0, 5.0, 4.0), listOf(6.0, 8.0, 7.0), listOf(9.0, 11.0, 10.0)),
        )
        val hs1 =
            when (val tryResult = t.indexSelect(ids1, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = hs1.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(0.0, 1.0, 2.0), listOf(6.0, 7.0, 8.0), listOf(3.0, 4.0, 5.0)),
        )
        // Prior to https://github.com/huggingface/candle/pull/1022
        // There would be a bug where the last values in the result tensor would be set to 0.
        val ids2 =
            when (val tryResult = Tensor.new(listOf(0u, 2u, 1u, 0u, 2u, 1u), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val hs2 =
            when (val tryResult = t.indexSelect(ids2, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = hs2.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(0.0, 1.0, 2.0), listOf(6.0, 7.0, 8.0), listOf(3.0, 4.0, 5.0), listOf(0.0, 1.0, 2.0), listOf(6.0, 7.0, 8.0), listOf(3.0, 4.0, 5.0)),
        )
        // Test when selecting dim > 0 with ids size different from elem count of
        // target dim in source/input.
        val ids3 =
            when (val tryResult = Tensor.new(listOf(1u, 0u, 1u), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val t1 =
            when (
                val tryResult =
                    when (val tryResult = Tensor.arange(1.0f, 5.0f, device)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.reshape(Pair(2, 2))
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = t1.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(1.0, 2.0), listOf(3.0, 4.0)),
        )
        val hs3 =
            when (val tryResult = t1.indexSelect(ids3, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(
            when (val tryResult = hs3.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            } == listOf(listOf(2.0, 1.0, 2.0), listOf(4.0, 3.0, 4.0)),
        )
    }
    return Result.Ok(Unit)
}

public fun indexAdd(device: Device): Result<Unit> {
    val ids =
        when (val tryResult = Tensor.new(listOf(0u, 1u, 1u), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 12.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    val init =
        when (val tryResult = Tensor.ones(Pair(4, 2), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs =
        when (val tryResult = init.indexAdd(ids, t, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 4.0), listOf(4.0, 10.0), listOf(7.0, 16.0), listOf(10.0, 22.0)),
    )
    val init1 =
        when (val tryResult = Tensor.zeros(Pair(4, 2), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids1 =
        when (val tryResult = Tensor.new(listOf(1u, 0u, 0u), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs1 =
        when (val tryResult = init1.indexAdd(ids1, t, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 0.0), listOf(9.0, 3.0), listOf(15.0, 6.0), listOf(21.0, 9.0)),
    )
    val init2 =
        when (val tryResult = Tensor.zeros(Pair(6, 3), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids2 =
        when (val tryResult = Tensor.new(listOf(5u, 0u, 1u, 0u), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs2 =
        when (val tryResult = init2.indexAdd(ids2, t, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(12.0, 14.0, 16.0), listOf(6.0, 7.0, 8.0), listOf(0.0, 0.0, 0.0), listOf(0.0, 0.0, 0.0), listOf(0.0, 0.0, 0.0), listOf(0.0, 1.0, 2.0)),
    )
    return Result.Ok(Unit)
}

public fun sliceScatter(device: Device): Result<Unit> {
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 12.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    val src =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(100.0f, 106.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t.sliceScatter0(src, 0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(100.0, 101.0, 102.0), listOf(103.0, 104.0, 105.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t.sliceScatter0(src, 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(100.0, 101.0, 102.0), listOf(103.0, 104.0, 105.0), listOf(9.0, 10.0, 11.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t.sliceScatter0(src, 2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(100.0, 101.0, 102.0), listOf(103.0, 104.0, 105.0)),
    )
    // The source is a view whose storage continues past it, which is what
    // `narrow` yields. Only the two rows the view describes may be written:
    // sizing the copy from the storage instead lets it run to the end of the
    // source's buffer and overwrite the rows after the scatter point.
    val big =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(100.0f, 112.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val srcView =
        when (val tryResult = big.narrow(0, 0, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t.sliceScatter0(srcView, 1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(100.0, 101.0, 102.0), listOf(103.0, 104.0, 105.0), listOf(9.0, 10.0, 11.0)),
    )
    // Same, with the view starting partway into its storage.
    val srcView1 =
        when (val tryResult = big.narrow(0, 2, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t.sliceScatter0(srcView1, 2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(106.0, 107.0, 108.0), listOf(9.0, 10.0, 11.0)),
    )
    return Result.Ok(Unit)
}

public fun scatter(device: Device): Result<Unit> {
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 12.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    val ids =
        when (val tryResult = Tensor.new(listOf(listOf(0u, 1, 2), listOf(3, 4, 0), listOf(3, 3, 1), listOf(2, 0, 4)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val init =
        when (val tryResult = Tensor.ones(Pair(4, 5), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs =
        when (val tryResult = init.scatterAdd(ids, t, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 2.0, 3.0, 1.0, 1.0), listOf(6.0, 1.0, 1.0, 4.0, 5.0), listOf(1.0, 9.0, 1.0, 14.0, 1.0), listOf(11.0, 1.0, 10.0, 1.0, 12.0)),
    )
    val hs1 =
        when (val tryResult = init.scatter(ids, t, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0, 1.0, 1.0), listOf(5.0, 1.0, 1.0, 3.0, 4.0), listOf(1.0, 8.0, 1.0, 7.0, 1.0), listOf(10.0, 1.0, 9.0, 1.0, 11.0)),
    )
    val init1 =
        when (val tryResult = Tensor.ones(Pair(6, 3), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs2 =
        when (val tryResult = init1.scatterAdd(ids, t, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 11.0, 6.0), listOf(1.0, 2.0, 9.0), listOf(10.0, 1.0, 3.0), listOf(10.0, 8.0, 1.0), listOf(1.0, 5.0, 12.0), listOf(1.0, 1.0, 1.0)),
    )
    val hs3 =
        when (val tryResult = init1.scatter(ids, t, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs3.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 10.0, 5.0), listOf(1.0, 1.0, 8.0), listOf(9.0, 1.0, 2.0), listOf(6.0, 7.0, 1.0), listOf(1.0, 4.0, 11.0), listOf(1.0, 1.0, 1.0)),
    )
    val hs4 =
        run {
            val ids1 =
                when (val tryResult = Tensor.new(listOf(listOf(0u, UInt.MAX_VALUE, 2), listOf(3, 4, UInt.MAX_VALUE), listOf(3, 3, 1), listOf(UInt.MAX_VALUE, UInt.MAX_VALUE, 4)), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = init1.scatter(ids1, t, 0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
    check(
        when (val tryResult = hs4.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 1.0), listOf(1.0, 1.0, 8.0), listOf(1.0, 1.0, 2.0), listOf(6.0, 7.0, 1.0), listOf(1.0, 4.0, 11.0), listOf(1.0, 1.0, 1.0)),
    )
    when (val tryResult = init1.scatterSet(ids, t, 0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (val tryResult = init1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 10.0, 5.0), listOf(1.0, 1.0, 8.0), listOf(9.0, 1.0, 2.0), listOf(6.0, 7.0, 1.0), listOf(1.0, 4.0, 11.0), listOf(1.0, 1.0, 1.0)),
    )
    return Result.Ok(Unit)
}

public fun gather(device: Device): Result<Unit> {
    val ids =
        when (val tryResult = Tensor.new(listOf(listOf(0u), listOf(2u), listOf(1u), listOf(0u)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 12.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0), listOf(9.0, 10.0, 11.0)),
    )
    val hs =
        when (val tryResult = t.gather(ids, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0), listOf(5.0), listOf(7.0), listOf(9.0)),
    )
    val ids1 =
        when (val tryResult = Tensor.new(listOf(listOf(0u, 0u), listOf(2u, 0u), listOf(1u, 1u), listOf(0u, 2u)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs1 =
        when (val tryResult = t.gather(ids1, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 0.0), listOf(5.0, 3.0), listOf(7.0, 7.0), listOf(9.0, 11.0)),
    )
    val ids2 =
        when (val tryResult = Tensor.new(listOf(listOf(0u, 2u, 0u)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs2 =
        when (val tryResult = t.gather(ids2, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 7.0, 2.0)),
    )
    val ids3 =
        when (val tryResult = Tensor.new(listOf(listOf(0u, 2u, 0u), listOf(0u, 1u, 1u)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs3 =
        when (val tryResult = t.gather(ids3, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs3.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 7.0, 2.0), listOf(0.0, 4.0, 5.0)),
    )
    val hs4 =
        run {
            val ids4 =
                when (val tryResult = Tensor.new(listOf(listOf(0u, 0u), listOf(2u, UInt.MAX_VALUE), listOf(UInt.MAX_VALUE, 1u), listOf(0u, 2u)), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = t.gather(ids4, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
    check(
        when (val tryResult = hs4.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 0.0), listOf(5.0, 0.0), listOf(0.0, 7.0), listOf(9.0, 11.0)),
    )
    // Random data
    // Dim: 0
    val t1 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(108.0f, -47.0, 16.0, -56.0, -83.0, -130.0, 210.0), listOf(253.0, 95.0, 151.0, 228.0, -210.0, -123.0, -127.0), listOf(-9.0, -217.0, 2.0, -78.0, 163.0, 245.0, -204.0), listOf(-246.0, 79.0, -238.0, 88.0, -226.0, -184.0, 171.0), listOf(8.0, -48.0, -153.0, 234.0, -34.0, 166.0, -153.0), listOf(124.0, 0.0, -10.0, -61.0, -242.0, -15.0, -238.0)), listOf(listOf(12.0, -64.0, -199.0, 244.0, -240.0, 156.0, -128.0), listOf(173.0, -57.0, 4.0, -198.0, 233.0, -110.0, 238.0), listOf(95.0, 82.0, 0.0, 240.0, 53.0, -211.0, 209.0), listOf(-122.0, 167.0, -212.0, 227.0, -144.0, 61.0, 118.0), listOf(-63.0, -146.0, 200.0, 244.0, 168.0, -167.0, 116.0), listOf(-125.0, -147.0, 110.0, -253.0, -178.0, -250.0, -18.0)), listOf(listOf(57.0, 86.0, -50.0, 56.0, 92.0, 205.0, -78.0), listOf(-137.0, -156.0, -18.0, 248.0, -61.0, -239.0, 14.0), listOf(-248.0, -30.0, -50.0, -70.0, -251.0, 250.0, -83.0), listOf(-221.0, 67.0, 72.0, 59.0, -24.0, -154.0, 232.0), listOf(-144.0, -23.0, -74.0, 5.0, 93.0, 171.0, 205.0), listOf(46.0, -77.0, -38.0, -226.0, 246.0, 161.0, -17.0)), listOf(listOf(-153.0, -231.0, -236.0, 161.0, 126.0, 2.0, -22.0), listOf(-229.0, -41.0, 209.0, 164.0, 234.0, 160.0, 57.0), listOf(223.0, 254.0, -186.0, -162.0, -46.0, -160.0, -102.0), listOf(65.0, 30.0, 213.0, -253.0, 59.0, 224.0, -154.0), listOf(-82.0, -203.0, -177.0, 17.0, 31.0, -256.0, -246.0), listOf(176.0, -135.0, -65.0, 54.0, -56.0, 210.0, 76.0)), listOf(listOf(-10.0, -245.0, 168.0, 124.0, -14.0, -33.0, -178.0), listOf(25.0, -43.0, -39.0, 132.0, -89.0, 169.0, 179.0), listOf(187.0, -215.0, 32.0, -133.0, 87.0, -7.0, -168.0), listOf(-224.0, -215.0, -5.0, -230.0, -58.0, -162.0, 128.0), listOf(158.0, -137.0, -122.0, -100.0, -202.0, -83.0, 136.0), listOf(30.0, -185.0, -144.0, 250.0, 209.0, -40.0, 127.0)), listOf(listOf(-196.0, 108.0, -245.0, 122.0, 146.0, -228.0, 62.0), listOf(-1.0, -66.0, 160.0, 137.0, 13.0, -172.0, -21.0), listOf(244.0, 199.0, -164.0, 28.0, 119.0, -175.0, 198.0), listOf(-62.0, 253.0, -162.0, 195.0, -95.0, -230.0, -211.0), listOf(123.0, -72.0, -26.0, -107.0, -139.0, 64.0, 245.0), listOf(11.0, -126.0, -182.0, 108.0, -12.0, 184.0, -127.0)), listOf(listOf(-159.0, 126.0, 176.0, 161.0, 73.0, -111.0, -138.0), listOf(-187.0, 214.0, -217.0, -33.0, -223.0, -201.0, -212.0), listOf(-61.0, -120.0, -166.0, -172.0, -95.0, 53.0, 196.0), listOf(-33.0, 86.0, 134.0, -152.0, 154.0, -53.0, 74.0), listOf(186.0, -28.0, -154.0, -174.0, 141.0, -109.0, 217.0), listOf(82.0, 35.0, 252.0, 145.0, 181.0, 74.0, -87.0))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids5 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(6u, 6, 4, 3, 4, 4, 6), listOf(3, 3, 2, 4, 4, 4, 6), listOf(3, 3, 0, 2, 4, 6, 4), listOf(2, 5, 1, 2, 6, 6, 1), listOf(2, 1, 6, 5, 3, 2, 3), listOf(6, 1, 0, 1, 0, 2, 6)), listOf(listOf(4, 6, 4, 3, 3, 3, 2), listOf(4, 3, 2, 4, 4, 4, 6), listOf(2, 3, 0, 2, 4, 6, 4), listOf(6, 5, 1, 2, 6, 6, 1), listOf(4, 1, 6, 5, 3, 2, 3), listOf(1, 1, 0, 1, 0, 2, 6)), listOf(listOf(3, 6, 4, 3, 3, 3, 2), listOf(2, 3, 2, 4, 4, 4, 6), listOf(4, 3, 0, 2, 4, 6, 4), listOf(0, 5, 1, 2, 6, 6, 1), listOf(6, 1, 6, 5, 3, 2, 3), listOf(4, 1, 0, 1, 0, 2, 6)), listOf(listOf(0, 6, 4, 3, 3, 3, 2), listOf(5, 3, 2, 4, 4, 4, 6), listOf(0, 3, 0, 2, 4, 6, 4), listOf(3, 5, 1, 2, 6, 6, 1), listOf(0, 1, 6, 5, 3, 2, 3), listOf(3, 1, 0, 1, 0, 2, 6))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs5 =
        when (val tryResult = t1.gather(ids5, 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs5.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-159.0f, 126.0, 168.0, 161.0, -14.0, -33.0, -138.0), listOf(-229.0, -41.0, -18.0, 132.0, -89.0, 169.0, -212.0), listOf(223.0, 254.0, 2.0, -70.0, 87.0, 53.0, -168.0), listOf(-221.0, 253.0, -212.0, 59.0, 154.0, -53.0, 118.0), listOf(-144.0, -146.0, -154.0, -107.0, 31.0, 171.0, -246.0), listOf(82.0, -147.0, -10.0, -253.0, -242.0, 161.0, -87.0)), listOf(listOf(-10.0, 126.0, 168.0, 161.0, 126.0, 2.0, -78.0), listOf(25.0, -41.0, -18.0, 132.0, -89.0, 169.0, -212.0), listOf(-248.0, 254.0, 2.0, -70.0, 87.0, 53.0, -168.0), listOf(-33.0, 253.0, -212.0, 59.0, 154.0, -53.0, 118.0), listOf(158.0, -146.0, -154.0, -107.0, 31.0, 171.0, -246.0), listOf(-125.0, -147.0, -10.0, -253.0, -242.0, 161.0, -87.0)), listOf(listOf(-153.0, 126.0, 168.0, 161.0, 126.0, 2.0, -78.0), listOf(-137.0, -41.0, -18.0, 132.0, -89.0, 169.0, -212.0), listOf(187.0, 254.0, 2.0, -70.0, 87.0, 53.0, -168.0), listOf(-246.0, 253.0, -212.0, 59.0, 154.0, -53.0, 118.0), listOf(186.0, -146.0, -154.0, -107.0, 31.0, 171.0, -246.0), listOf(30.0, -147.0, -10.0, -253.0, -242.0, 161.0, -87.0)), listOf(listOf(108.0, 126.0, 168.0, 161.0, 126.0, 2.0, -78.0), listOf(-1.0, -41.0, -18.0, 132.0, -89.0, 169.0, -212.0), listOf(-9.0, 254.0, 2.0, -70.0, 87.0, 53.0, -168.0), listOf(65.0, 253.0, -212.0, 59.0, 154.0, -53.0, 118.0), listOf(8.0, -146.0, -154.0, -107.0, 31.0, 171.0, -246.0), listOf(176.0, -147.0, -10.0, -253.0, -242.0, 161.0, -87.0))),
    )
    // Dim: 1
    val t2 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(-117.0f, -175.0, 69.0, -163.0), listOf(200.0, 242.0, -21.0, -67.0), listOf(179.0, 150.0, -126.0, -75.0), listOf(-118.0, 38.0, -138.0, -13.0), listOf(-221.0, 136.0, -185.0, 180.0), listOf(58.0, 182.0, -204.0, -149.0)), listOf(listOf(3.0, -148.0, -58.0, -154.0), listOf(-43.0, 45.0, -108.0, 4.0), listOf(-69.0, -249.0, -71.0, -21.0), listOf(80.0, 110.0, -152.0, -235.0), listOf(-88.0, 7.0, 92.0, -250.0), listOf(-186.0, 207.0, -242.0, 98.0)), listOf(listOf(238.0, 19.0, 64.0, -242.0), listOf(-150.0, -97.0, 218.0, 58.0), listOf(111.0, -233.0, 204.0, -212.0), listOf(-242.0, -232.0, 83.0, 42.0), listOf(153.0, 62.0, -251.0, 219.0), listOf(-117.0, 36.0, -119.0, 10.0)), listOf(listOf(215.0, 159.0, -169.0, -27.0), listOf(-83.0, 101.0, -88.0, 169.0), listOf(-205.0, 93.0, 225.0, -64.0), listOf(-162.0, 240.0, 214.0, 23.0), listOf(-112.0, 6.0, 21.0, 245.0), listOf(-38.0, 113.0, 93.0, 215.0)), listOf(listOf(91.0, -188.0, -148.0, 101.0), listOf(74.0, 203.0, -35.0, 55.0), listOf(-116.0, -130.0, -153.0, -96.0), listOf(58.0, 22.0, -45.0, -194.0), listOf(-221.0, -134.0, 73.0, 159.0), listOf(-203.0, -254.0, 31.0, 235.0)), listOf(listOf(105.0, -53.0, 61.0, 186.0), listOf(-195.0, 234.0, 75.0, -1.0), listOf(51.0, 139.0, 160.0, -108.0), listOf(-173.0, -167.0, 161.0, 19.0), listOf(83.0, -246.0, 156.0, -222.0), listOf(109.0, 39.0, -149.0, 137.0))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids6 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(4u, 4, 4, 2)), listOf(listOf(0, 4, 4, 3)), listOf(listOf(1, 5, 3, 4)), listOf(listOf(0, 3, 3, 2)), listOf(listOf(1, 1, 5, 2)), listOf(listOf(1, 4, 5, 4))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs6 =
        when (val tryResult = t2.gather(ids6, 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs6.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-221.0, 136.0, -185.0, -75.0)), listOf(listOf(3.0, 7.0, 92.0, -235.0)), listOf(listOf(-150.0, 36.0, 83.0, 219.0)), listOf(listOf(215.0, 240.0, 214.0, -64.0)), listOf(listOf(74.0, 203.0, 31.0, -96.0)), listOf(listOf(-195.0, -246.0, -149.0, -222.0))),
    )
    // Dim: 2
    val t3 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(-162.0f, 202.0), listOf(-126.0, -39.0), listOf(35.0, -65.0), listOf(1.0, 80.0)), listOf(listOf(37.0, 248.0), listOf(-191.0, 89.0), listOf(117.0, -40.0), listOf(-217.0, 220.0))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids7 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(1u), listOf(0), listOf(1), listOf(1)), listOf(listOf(0), listOf(1), listOf(0), listOf(1))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs7 =
        when (val tryResult = t3.gather(ids7, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs7.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(202.0), listOf(-126.0), listOf(-65.0), listOf(80.0)), listOf(listOf(37.0), listOf(89.0), listOf(117.0), listOf(220.0))),
    )
    val t4 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(-21.0f, -197.0), listOf(194.0, 122.0)), listOf(listOf(255.0, -106.0), listOf(-191.0, 250.0)), listOf(listOf(33.0, -117.0), listOf(43.0, 10.0)), listOf(listOf(-130.0, 238.0), listOf(-217.0, -92.0))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val ids8 =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(0u, 1), listOf(1, 0)), listOf(listOf(1, 0), listOf(0, 1)), listOf(listOf(0, 1), listOf(0, 1)), listOf(listOf(1, 0), listOf(1, 0))), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val hs8 =
        when (val tryResult = t4.gather(ids8, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = hs8.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-21.0, -197.0), listOf(122.0, 194.0)), listOf(listOf(-106.0, 255.0), listOf(-191.0, 250.0)), listOf(listOf(33.0, -117.0), listOf(43.0, 10.0)), listOf(listOf(238.0, -130.0), listOf(-92.0, -217.0))),
    )
    return Result.Ok(Unit)
}

public fun broadcasting(device: Device): Result<Unit> {
    val t1 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 24.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(4, 2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (val tryResult = Tensor.new(listOf(100.0f, 200.0f), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s =
        when (
            val tryResult =
                t1.broadcastAdd(
                    when (val tryResult = t2.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(100.0, 101.0, 102.0), listOf(203.0, 204.0, 205.0)), listOf(listOf(106.0, 107.0, 108.0), listOf(209.0, 210.0, 211.0)), listOf(listOf(112.0, 113.0, 114.0), listOf(215.0, 216.0, 217.0)), listOf(listOf(118.0, 119.0, 120.0), listOf(221.0, 222.0, 223.0))),
    )
    val s1 =
        when (
            val tryResult =
                when (val tryResult = t1.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAdd(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s1.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(100.0, 203.0), listOf(101.0, 204.0), listOf(102.0, 205.0)), listOf(listOf(106.0, 209.0), listOf(107.0, 210.0), listOf(108.0, 211.0)), listOf(listOf(112.0, 215.0), listOf(113.0, 216.0), listOf(114.0, 217.0)), listOf(listOf(118.0, 221.0), listOf(119.0, 222.0), listOf(120.0, 223.0))),
    )
    val s2 =
        when (
            val tryResult =
                t1.broadcastSub(
                    when (val tryResult = t2.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s2.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-100.0, -99.0, -98.0), listOf(-197.0, -196.0, -195.0)), listOf(listOf(-94.0, -93.0, -92.0), listOf(-191.0, -190.0, -189.0)), listOf(listOf(-88.0, -87.0, -86.0), listOf(-185.0, -184.0, -183.0)), listOf(listOf(-82.0, -81.0, -80.0), listOf(-179.0, -178.0, -177.0))),
    )
    val s3 =
        when (
            val tryResult =
                when (val tryResult = t1.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastSub(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s3.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-100.0, -197.0), listOf(-99.0, -196.0), listOf(-98.0, -195.0)), listOf(listOf(-94.0, -191.0), listOf(-93.0, -190.0), listOf(-92.0, -189.0)), listOf(listOf(-88.0, -185.0), listOf(-87.0, -184.0), listOf(-86.0, -183.0)), listOf(listOf(-82.0, -179.0), listOf(-81.0, -178.0), listOf(-80.0, -177.0))),
    )
    // Test a narrowed version as this uses a layout startOffset.
    val t11 =
        when (val tryResult = t1.i(2 until Int.MAX_VALUE)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s4 =
        when (
            val tryResult =
                t11.broadcastAdd(
                    when (val tryResult = t2.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s4.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(112.0, 113.0, 114.0), listOf(215.0, 216.0, 217.0)), listOf(listOf(118.0, 119.0, 120.0), listOf(221.0, 222.0, 223.0))),
    )
    val s5 =
        when (
            val tryResult =
                when (val tryResult = t11.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastAdd(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s5.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(112.0, 215.0), listOf(113.0, 216.0), listOf(114.0, 217.0)), listOf(listOf(118.0, 221.0), listOf(119.0, 222.0), listOf(120.0, 223.0))),
    )
    val s6 =
        when (
            val tryResult =
                t11.broadcastSub(
                    when (val tryResult = t2.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s6.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-88.0, -87.0, -86.0), listOf(-185.0, -184.0, -183.0)), listOf(listOf(-82.0, -81.0, -80.0), listOf(-179.0, -178.0, -177.0))),
    )
    val s7 =
        when (
            val tryResult =
                when (val tryResult = t11.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastSub(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s7.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-88.0, -185.0), listOf(-87.0, -184.0), listOf(-86.0, -183.0)), listOf(listOf(-82.0, -179.0), listOf(-81.0, -178.0), listOf(-80.0, -177.0))),
    )
    val t3 =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(1.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastDiv(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s8 =
        when (
            val tryResult =
                t11.broadcastMul(
                    when (val tryResult = t2.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val sDiv =
        when (
            val tryResult =
                t11.broadcastDiv(
                    when (val tryResult = t3.reshape(Pair(2, 1))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s8.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1200.0, 1300.0, 1400.0), listOf(3000.0, 3200.0, 3400.0)), listOf(listOf(1800.0, 1900.0, 2000.0), listOf(4200.0, 4400.0, 4600.0))),
    )
    check(
        when (val tryResult = s8.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (val tryResult = sDiv.toVec3<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val s9 =
        when (
            val tryResult =
                when (val tryResult = t11.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastMul(t2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val sDiv1 =
        when (
            val tryResult =
                when (val tryResult = t11.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.broadcastDiv(t3)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = s9.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1200.0, 3000.0), listOf(1300.0, 3200.0), listOf(1400.0, 3400.0)), listOf(listOf(1800.0, 4200.0), listOf(1900.0, 4400.0), listOf(2000.0, 4600.0))),
    )
    check(
        when (val tryResult = s9.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (val tryResult = sDiv1.toVec3<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    return Result.Ok(Unit)
}

public fun randn(device: Device): Result<Unit> {
    // We do not expect deterministic elements at any index.
    // There once was a bug that had a deterministic zero element in evenly sized tensors.
    val N: Int = 2
    val tensor =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(5, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(tensor.dims() == listOf(5, 3))
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    val tensor2 =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(5, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tensor.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } !=
            when (val tryResult = tensor2.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val tensor1 =
        when (val tryResult = Tensor.rand(0.0f, 1.0f, Pair(5, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(tensor1.dims() == listOf(5, 3))
    // Check that the seed gets updated by checking that
    // a new series of numbers is generated each time
    val tensor21 =
        when (val tryResult = Tensor.rand(0.0f, 1.0f, Pair(5, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tensor1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } !=
            when (val tryResult = tensor21.toVec2<Float>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    val v =
        when (val tryResult = (0 until 100).map({ Tensor.randn(0.0f, 1.0f, N, device).andThen({ t -> t.toVec1<Float>() }) }).collect<Result<Vec>>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check((0 until N).all({ i -> v.windows(2).any({ pair -> pair[0][i] != pair[1][i] }) })) { "There are deterministic values in the randn tensors" }
    val v1 =
        when (val tryResult = (0 until 100).map({ Tensor.rand(0.0f, 1.0f, N, device).andThen({ t -> t.toVec1<Float>() }) }).collect<Result<Vec>>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check((0 until N).all({ i -> v1.windows(2).any({ pair -> pair[0][i] != pair[1][i] }) })) { "There are deterministic values in the rand tensors" }
    return Result.Ok(Unit)
}

public fun repeatWithZeroFactor(device: Device): Result<Unit> {
    val tensor =
        when (val tryResult = Tensor.new(listOf(listOf(1u, 2), listOf(3, 4)), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val repeatedRows =
        when (val tryResult = tensor.repeat(Pair(0, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(repeatedRows.dims() == listOf(0, 4))
    check(repeatedRows.elemCount() == 0)
    val repeatedColumns =
        when (val tryResult = tensor.repeat(Pair(2, 0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(repeatedColumns.dims() == listOf(4, 0))
    check(repeatedColumns.elemCount() == 0)
    return Result.Ok(Unit)
}

public fun meshgridWithEmptyAxis(device: Device): Result<Unit> {
    val emptyAxis =
        when (val tryResult = Tensor.fromVec(mutableListOf(), 0, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val coordinates =
        when (val tryResult = Tensor.new(listOf(10.0f, 20.0, 30.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val grids =
        when (val tryResult = Tensor.meshgrid(listOf(emptyAxis, coordinates), false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(grids.len() == 2)
    for (grid in grids) {
        check(grid.dims() == listOf(0, 3))
        check(grid.elemCount() == 0)
    }
    return Result.Ok(Unit)
}

public fun zeroDim(device: Device): Result<Unit> {
    val t =
        when (val tryResult = Tensor.zeros(Triple(4, 0, 1), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 0, 1),
    )
    val t2 =
        when (val tryResult = Tensor.zeros(Triple(4, 3, 1), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tCat =
        when (val tryResult = Tensor.cat(listOf(t, t2), 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tCat.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 3, 1),
    )
    val tCat1 =
        when (val tryResult = Tensor.cat(listOf(t, t), 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tCat1.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 0, 1),
    )
    val tUnary =
        when (val tryResult = t.sqrt()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tUnary.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 0, 1),
    )
    val tPlus =
        when (val tryResult = (t + 1.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tPlus.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 0, 1),
    )
    val tMm =
        when (
            val tryResult =
                t2.matmul(
                    when (val tryResult = t.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tMm.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 3, 0),
    )
    val tMm1 =
        when (
            val tryResult =
                t.matmul(
                    when (val tryResult = t2.t()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tMm1.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 0, 3),
    )
    val tMm2 =
        when (
            val tryResult =
                when (val tryResult = t.t()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.matmul(t)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tMm2.dims3()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == Triple(4, 1, 1),
    )
    return Result.Ok(Unit)
}

public fun tensorSendSync(device: Device): Result<Unit> {
    val tensor =
        when (val tryResult = Tensor.new(mutableListOf(1.0f, 2.0, 3.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (unused in 0 until 10) {
        val tensor1 = tensor.clone()
        spawn({
            val new = tensor1.add(tensor1).unwrap()
            val result: List<Float> = new.toVec1().unwrap()
            check(result == mutableListOf(2.0f, 4.0, 6.0))
        })
    }
    val result: List<Float> = tensor.toVec1().unwrap()
    check(result == mutableListOf(1.0f, 2.0, 3.0))
    val tensor2 =
        when (val tryResult = Tensor.new(mutableListOf(1.0f, 2.0, 3.0), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    tensor2.device().synchronize().unwrap()
    val new =
        spawn({
            val new = tensor2.add(tensor2).unwrap()
            new.device().synchronize().unwrap()
            new
        }).join().unwrap()
    val result1: List<Float> = new.toVec1().unwrap()
    check(result1 == mutableListOf(2.0f, 4.0, 6.0))
    return Result.Ok(Unit)
}

// There was originally a bug on the CPU implementation for randn
// https://github.com/huggingface/candle/issues/381
@kotlin.test.Test
public fun randnHasneg(): Result<Unit> {
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.randn(0.0f, 1.0f, 200, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (t.iter().all({ v -> v >= 0.0 })) {
        bail("all values in tensors are non-negative")
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun padWithSame(): Result<Unit> {
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(1.0f, 5.0f, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t0 =
        when (val tryResult = t.padWithSame(0, 1, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t0.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 2.0), listOf(1.0, 2.0), listOf(3.0, 4.0), listOf(3.0, 4.0), listOf(3.0, 4.0)),
    )
    val t1 =
        when (val tryResult = t.padWithSame(1, 1, 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.0, 2.0, 2.0, 2.0), listOf(3.0, 3.0, 4.0, 4.0, 4.0)),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun i64Abs(): Result<Unit> {
    val t =
        when (val tryResult = Tensor.new(listOf(-42L, 1337), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t1 =
        when (val tryResult = t.abs()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t1.toVec1<Long>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(42, 1337),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun trilTriuEye(): Result<Unit> {
    val t =
        when (val tryResult = Tensor.tril2(4, DType.F32, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 0.0, 0.0, 0.0), listOf(1.0, 1.0, 0.0, 0.0), listOf(1.0, 1.0, 1.0, 0.0), listOf(1.0, 1.0, 1.0, 1.0)),
    )
    val t1 =
        when (val tryResult = Tensor.triu2(4, DType.F32, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.0, 1.0, 1.0), listOf(0.0, 1.0, 1.0, 1.0), listOf(0.0, 0.0, 1.0, 1.0), listOf(0.0, 0.0, 0.0, 1.0)),
    )
    val t2 =
        when (val tryResult = Tensor.eye(4, DType.F32, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 0.0, 0.0, 0.0), listOf(0.0, 1.0, 0.0, 0.0), listOf(0.0, 0.0, 1.0, 0.0), listOf(0.0, 0.0, 0.0, 1.0)),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun cumsum(): Result<Unit> {
    val t = listOf(3.0f, 1.0, 4.0, 1.0, 5.0)
    val t1 =
        when (val tryResult = Tensor.new(t, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t1.cumsum(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3.0, 4.0, 8.0, 9.0, 14.0),
    )
    val t2 =
        when (val tryResult = t1.unsqueeze(1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t2.cumsum(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0), listOf(4.0), listOf(8.0), listOf(9.0), listOf(14.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t2.cumsum(1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0), listOf(1.0), listOf(4.0), listOf(1.0), listOf(5.0)),
    )
    val t3 = listOf(listOf(3.0f, 1.0, 4.0, 1.0, 5.0), listOf(2.0, 1.0, 7.0, 8.0, 2.0))
    val t4 =
        when (val tryResult = Tensor.new(t3, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = t4.cumsum(1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 4.0, 8.0, 9.0, 14.0), listOf(2.0, 3.0, 10.0, 18.0, 20.0)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = t4.cumsum(0)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 1.0, 4.0, 1.0, 5.0), listOf(5.0, 2.0, 11.0, 9.0, 7.0)),
    )
    return Result.Ok(Unit)
}

/**
 * A helper function for floating point comparison. Both a and b must be 1D Tensor and contains the same amount of data.
 * Assertion passes if the difference of all pairs of a and b is smaller than epsilon.
 */
public fun assertClose(a: Tensor, b: Tensor, epsilon: Double): Result<Unit> {
    val aVec: List<Double> =
        when (val tryResult = a.toVec1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val bVec: List<Double> =
        when (val tryResult = b.toVec1()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(aVec.size == bVec.size)
    for ((a1, b1) in (aVec.zip(bVec))) {
        check((a1 - b1).abs() < epsilon)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun logSumExp(): Result<Unit> {
    val input =
        when (val tryResult = Tensor.new(listOf(listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), listOf(listOf(-1000.0, -999.0, -1001.0), listOf(1000.0, 999.0, 1001.0))), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.logSumExp(D.Minus1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // The expectations obtained from pytorch.
    val expected =
        when (val tryResult = Tensor.new(listOf(listOf(3.4076, 6.4076), listOf(-998.5924, 1001.4076)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(output.dims() == expected.dims())
    when (
        val tryResult =
            assertClose(
                when (val tryResult = output.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
                when (val tryResult = expected.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
                0.00001,
            )
    ) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    check(
        when (
            val tryResult =
                when (val tryResult = input.logSumExp(Pair(0, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Double>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1000.0, 999.0, 1001.0),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = input.logSumExp()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Double>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } ==
            when (val tryResult = input.toVec3<Double>()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            },
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun pow(): Result<Unit> {
    val lhs =
        when (val tryResult = Tensor.new(listOf(listOf(1.0f, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = (lhs - 2.0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val res =
        when (val tryResult = lhs.pow(rhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec2Round(res, 3)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 1.0, 3.0), listOf(16.0, 125.0, 1296.0)),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun testFlip1d(): Result<Unit> {
    // 1D: [0, 1, 2, 3, 4]
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0, 5.0, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(5)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val flipped =
        when (val tryResult = t.flip(listOf(0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // Expected: [4, 3, 2, 1, 0]
    val expected =
        when (val tryResult = Tensor.fromVec(mutableListOf(4.0, 3.0, 2.0, 1.0, 0.0), 5, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(flipped, expected)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun testFlip2d(): Result<Unit> {
    // 2D:
    // [[0, 1, 2],
    // [3, 4, 5]]
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0, 6.0, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val flipped =
        when (val tryResult = t.flip(listOf(0, 1))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // Expected:
    // [[5, 4, 3],
    // [2, 1, 0]]
    val expected =
        when (val tryResult = Tensor.fromVec(mutableListOf(5.0, 4.0, 3.0, 2.0, 1.0, 0.0), Pair(2, 3), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(flipped, expected)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun testFlip3dChannels(): Result<Unit> {
    // 3D:
    // [[[0,1,2],
    // [3,4,5]],
    //
    // [[6,7,8],
    // [9,10,11]]]
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0, 12.0, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val flipped =
        when (val tryResult = t.flip(listOf(2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // Expected:
    // [[[2,1,0],
    // [5,4,3]],
    //
    // [[8,7,6],
    // [11,10,9]]]
    val expected =
        when (val tryResult = Tensor.fromVec(mutableListOf(2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 8.0, 7.0, 6.0, 11.0, 10.0, 9.0), Triple(2, 2, 3), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(flipped, expected)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun tensorFromDataValidatesConcreteShape(): Result<Unit> {
    val values = listOf(1.0f, 2.0, 3.0, 4.0)
    check(Tensor.fromSlice(values, Triple(2, 2, 2), Device.Cpu).isErr())
    check(Tensor.fromSlice(values, Pair(2, 1), Device.Cpu).isErr())
    check(Tensor.fromVec(values.toVec(), Triple(2, 2, 2), Device.Cpu).isErr())
    check(Tensor.fromVec(values.toVec(), Pair(2, 1), Device.Cpu).isErr())
    val tensor =
        when (val tryResult = Tensor.fromSlice(values, Pair(2, 2), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(tensor.dims() == listOf(2, 2))
    val tensor1 =
        when (val tryResult = Tensor.fromVec(values.toVec(), Pair(Unit, 2), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(tensor1.dims() == listOf(2, 2))
    val empty: List<Float> = listOf()
    val tensor2 =
        when (val tryResult = Tensor.fromSlice(empty, Pair(0, 3), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(tensor2.dims() == listOf(0, 3))
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun tensorNew(): Result<Unit> {
    val t1 =
        when (val tryResult = Tensor.new(mutableListOf(1.0f, 2.0, 3.0), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t1.toVec1<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(1.0, 2.0, 3.0),
    )
    val t2 =
        when (val tryResult = Tensor.new(mutableListOf(mutableListOf(1.0f, 2.0, 3.0), mutableListOf(4.0, 5.0, 6.0)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)),
    )
    val t3 =
        when (val tryResult = Tensor.new(mutableListOf(mutableListOf(mutableListOf(1.0f, 2.0, 3.0), mutableListOf(4.0, 5.0, 6.0)), mutableListOf(mutableListOf(3.0f, 1.0, 4.0), mutableListOf(1.0, 5.0, 9.0))), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t3.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), listOf(listOf(3.0, 1.0, 4.0), listOf(1.0, 5.0, 9.0))),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun tensorNorm(): Result<Unit> {
    val t =
        when (val tryResult = Tensor.new(listOf(listOf(3.0, 4.0), listOf(0.0, 0.0)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val norm =
        when (val tryResult = t.norm()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = norm.toScalar<Double>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 5.0,
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun transfersCudaToDevice(): Result<Unit> {
    val devices =
        when (val tryResult = CudaContext.deviceCount().mapErr(CudaError::from)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (devices < 2) {
        return Result.Ok(Unit)
    }
    val first =
        when (val tryResult = Device.newCuda(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data: List<UInt> = (0 until 262144).toMutableList()
    val rng = rng()
    data.shuffle(rng)
    val t1 =
        when (val tryResult = Tensor.fromVec(data, Pair(512, 512), first)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val second =
        when (val tryResult = Device.newCuda(1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (val tryResult = t1.toDevice(second)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = t1.device().asCudaDevice()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }.id() !=
            when (val tryResult = t2.device().asCudaDevice()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.id(),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun allocatesTwiceWhenTransferringToSameDevice(): Result<Unit> {
    val first =
        when (val tryResult = Device.newCuda(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val second =
        when (val tryResult = Device.newCuda(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data: List<UInt> = (0 until 262144).toMutableList()
    val rng = rng()
    data.shuffle(rng)
    val t1 =
        when (val tryResult = Tensor.fromVec(data, Pair(512, 512), first)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val t2 =
        when (val tryResult = t1.toDevice(second)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (storage1, _) = t1.storageAndLayout()
    val (storage2, _) = t2.storageAndLayout()
    val extract =
        { s: StorageRef ->
            when (val matchVal = s.deref()) {
                is Storage.Cuda -> {
                    val (c) = matchVal
                    val slice = c.asCudaSlice<UInt>().unwrap()
                    val ptr = slice.devicePtr(slice.stream()).first
                    ptr
                }
                else -> error("unsupported")
            }
        }
    val id1 = extract(storage1)
    val id2 = extract(storage2)
    check(id1 != id2)
    return Result.Ok(Unit)
}
