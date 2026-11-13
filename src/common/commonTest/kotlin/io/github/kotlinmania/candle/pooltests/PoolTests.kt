package io.github.kotlinmania.candle.pooltests

import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor

// https://github.com/huggingface/candle/issues/364
public fun avgPool2d(dev: Device): Result<Unit> {
    val data: List<Float> = mutableListOf(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    val t =
        when (val tryResult = Tensor.fromVec(data, listOf(1, 1, 4, 4), dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t.avgPool2d(2)) {
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
    check(
        when (val tryResult = pool.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.5f, 1.0), listOf(1.0, 1.0)),
    )
    val data1: List<Float> = mutableListOf(1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0)
    val t1 =
        when (val tryResult = Tensor.fromVec(data1, listOf(1, 1, 2, 8), dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.avgPool2d(2)) {
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
    check(
        when (val tryResult = pool1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(5.0 / 4.0, 6.0 / 4.0, 6.0 / 4.0, 1.0)),
    )
    return Result.Ok(Unit)
}

public fun maxPool2d(dev: Device): Result<Unit> {
    val data: List<Float> = mutableListOf(1.0, 2.0, 1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0)
    val t =
        when (val tryResult = Tensor.fromVec(data, listOf(1, 1, 4, 4), dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t.maxPool2d(2)) {
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
    check(
        when (val tryResult = pool.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(2.0f, 3.0), listOf(5.0, 1.0)),
    )
    val t1 =
        when (val tryResult = t.reshape(listOf(1, 1, 2, 8))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.maxPool2d(2)) {
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
    check(
        when (val tryResult = pool1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(2.0, 3.0, 5.0, 1.0)),
    )
    return Result.Ok(Unit)
}

// This test corresponds to the following PyTorch script.
// import torch
// torch.manualSeed(4242)
//
// t = torch.randn((1, 2, 4, 4))
// print(t.flatten())
// res = torch.nn.functional.avgPool2d(t, 2)
// print(res)
//
public fun avgPool2dPytorch(dev: Device): Result<Unit> {
    if (dev.isMetal()) {
        return Result.Ok(Unit)
    }
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(0.4056f, -0.8689, -0.0773, -1.5630, -2.8012, -1.5059, 0.3972, 1.0852, 0.4997, 3.0616, 1.6541, 0.0964, -0.8338, -1.6523, -0.8323, -0.1699, 0.0823, 0.3526, 0.6843, 0.2395, 1.2279, -0.9287, -1.7030, 0.1370, 0.6047, 0.3770, -0.6266, 0.3529, 2.2013, -0.6836, 0.2477, 1.3127), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 2, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool =
        when (
            val tryResult =
                when (val tryResult = t.avgPool2d(2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.squeeze(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec3Round(pool, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(-1.1926, -0.0395), listOf(0.2688, 0.1871)), listOf(listOf(0.1835, -0.1606), listOf(0.6249, 0.3217))),
    )
    val pool1 =
        when (
            val tryResult =
                when (val tryResult = t.avgPool2d(3)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.squeeze(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = testutils.toVec3Round(pool1, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0.085)), listOf(listOf(0.0078))),
    )
    val t1 =
        when (val tryResult = t.reshape(listOf(1, 1, 4, 8))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val pool2 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t1.avgPool2d(2)) {
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
    check(
        when (val tryResult = testutils.toVec2Round(pool2, 4)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.7745, 0.0276, -1.6983, 0.12), listOf(0.3542, 0.1625, 0.4542, -0.0014)),
    )
    return Result.Ok(Unit)
}

public fun upsampleNearest2d(dev: Device): Result<Unit> {
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 6.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val upsampled =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t.upsampleNearest2d(4, 6)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.i(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.i(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = t.i(0)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.i(0)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0)),
    )
    check(
        when (val tryResult = upsampled.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 0.0, 1.0, 1.0, 2.0, 2.0), listOf(0.0, 0.0, 1.0, 1.0, 2.0, 2.0), listOf(3.0, 3.0, 4.0, 4.0, 5.0, 5.0), listOf(3.0, 3.0, 4.0, 4.0, 5.0, 5.0)),
    )
    return Result.Ok(Unit)
}
