@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.displaytests

import io.github.kotlinmania.anyhow.Result
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device.Cpu
import io.github.kotlinmania.candlecore.Tensor

@kotlin.test.Test
public fun displayScalar(): Result<Unit> {
    val t =
        when (val tryResult = Tensor.new(1234u, Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s = "$t"
    check(s == "[1234]\nTensor[[], u32]")
    val t1 =
        when (
            val tryResult =
                when (val tryResult = t.toDtype(DType.F32)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.neg()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s1 = "${when (val tryResult = (t1 / 10.0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }}"
    check(s1 == "[-123.4000]\nTensor[[], f32]")
    val s2 = "${when (val tryResult = (t1 / 1e8)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }}"
    check(s2 == "[-1.2340e-5]\nTensor[[], f32]")
    val s3 = "${when (val tryResult = (t1 * 1e8)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }}"
    check(s3 == "[-1.2340e11]\nTensor[[], f32]")
    val s4 = "${when (val tryResult = (t1 * 0.0)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }}"
    check(s4 == "[0.]\nTensor[[], f32]")
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun displayVector(): Result<Unit> {
    val t =
        when (val tryResult = Tensor.new<UIntArray>(listOf(), Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s = "$t"
    check(s == "[]\nTensor[[0], u32]")
    val s1 = "$t"
    check(s1 == "[ 0.1235,  1.0000, -1.2000,  4.1000,     NaN]\nTensor[[5], f64]")
    val s2 = "\n$t"
    val expected = "\n[42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,\n 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,\n 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42., 42.,\n 42., 42.]\nTensor[[50], f32]"
    check(s2 == expected)
    val s3 = "$t"
    check(s3 == "[42., 42., 42., ..., 42., 42., 42.]\nTensor[[11000], f32]")
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun displayMultiDim(): Result<Unit> {
    val t =
        when (
            val tryResult = (
                when (val tryResult = Tensor.ones(Pair(200, 100), DType.F32, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } * 42.0
            )
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val s = "\n$t"
    val expected = "\n[[42., 42., 42., ..., 42., 42., 42.],\n [42., 42., 42., ..., 42., 42., 42.],\n [42., 42., 42., ..., 42., 42., 42.],\n ...\n [42., 42., 42., ..., 42., 42., 42.],\n [42., 42., 42., ..., 42., 42., 42.],\n [42., 42., 42., ..., 42., 42., 42.]]\nTensor[[200, 100], f32]"
    check(s == expected)
    val t2 = "\n$t"
    val expected1 = "\n[[[[[42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    ...\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.]]]],\n [[[[42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    ...\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.],\n    [42., 42., 42., ..., 42., 42., 42.]]]]]\nTensor[[2, 1, 1, 100, 100], f32]"
    check(t2 == expected1)
    return Result.Ok(Unit)
}
