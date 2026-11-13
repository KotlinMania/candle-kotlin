@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.indexingtests

import io.github.kotlinmania.anyhow.Result
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Tensor

@kotlin.test.Test
public fun integerIndex(): Result<Unit> {
    val dev = Device.Cpu
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 2 * 3, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val result =
        when (val tryResult = tensor.i(1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result.dims() == listOf(3))
    check(
        when (val tryResult = result.toVec1<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(3, 4, 5),
    )
    val result1 =
        when (val tryResult = tensor.i(Pair(0 until 0, 2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result1.dims() == listOf(2))
    check(
        when (val tryResult = result1.toVec1<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(2, 5),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun rangeIndex(): Result<Unit> {
    val dev = Device.Cpu
    // RangeFull
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 2 * 3, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(2, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val result =
        when (val tryResult = tensor.i(0 until 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result.dims() == listOf(2, 3))
    check(
        when (val tryResult = result.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2), listOf(3, 4, 5)),
    )
    // Range
    val tensor1 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 4 * 3, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val result1 =
        when (val tryResult = tensor1.i(1 until 3)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result1.dims() == listOf(2, 3))
    check(
        when (val tryResult = result1.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3, 4, 5), listOf(6, 7, 8)),
    )
    // RangeFrom
    val result2 =
        when (val tryResult = tensor1.i(2 until Int.MAX_VALUE)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result2.dims() == listOf(2, 3))
    check(
        when (val tryResult = result2.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(6, 7, 8), listOf(9, 10, 11)),
    )
    // RangeTo
    val result3 =
        when (val tryResult = tensor1.i(0 until 2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result3.dims() == listOf(2, 3))
    check(
        when (val tryResult = result3.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2), listOf(3, 4, 5)),
    )
    // RangeInclusive
    val result4 =
        when (val tryResult = tensor1.i(1..2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result4.dims() == listOf(2, 3))
    check(
        when (val tryResult = result4.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3, 4, 5), listOf(6, 7, 8)),
    )
    // RangeTo
    val result5 =
        when (val tryResult = tensor1.i(0 until 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result5.dims() == listOf(1, 3))
    check(
        when (val tryResult = result5.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2)),
    )
    // RangeToInclusive
    val result6 =
        when (val tryResult = tensor1.i(0..1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result6.dims() == listOf(2, 3))
    check(
        when (val tryResult = result6.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2), listOf(3, 4, 5)),
    )
    // Empty range
    val result7 =
        when (val tryResult = tensor1.i(1 until 1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result7.dims() == listOf(0, 3))
    val empty: List<UIntArray> = listOf()
    check(
        when (val tryResult = result7.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == empty,
    )
    // Similar to PyTorch, allow empty ranges when the computed length is negative.
    val result8 =
        when (val tryResult = tensor1.i(1 until 0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(result8.dims() == listOf(0, 3))
    val empty1: List<UIntArray> = listOf()
    check(
        when (val tryResult = result8.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == empty1,
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun index3d(): Result<Unit> {
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromIter(0 until 24u, Device.Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0, 0, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 0,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(1, 0, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 12,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0, 1, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 4,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0, 1, 3))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toScalar<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == 7,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0 until 2, 0, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 12),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0 until 2, 0 until 0, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 4, 8), listOf(12, 16, 20)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(0 until 2, 0 until 0, 3))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec2<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3, 7, 11), listOf(15, 19, 23)),
    )
    check(
        when (
            val tryResult =
                when (val tryResult = tensor.i(Triple(1, 0 until 0, 3))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(15, 19, 23),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun sliceAssign(): Result<Unit> {
    val dev = Device.Cpu
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 4 * 5, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(4, 5))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val src =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 2 * 3, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Pair(3, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out =
        when (val tryResult = tensor.sliceAssign(listOf(1 until 4, 3 until 5), src)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = out.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2, 3, 4), listOf(5, 6, 7, 0, 1), listOf(10, 11, 12, 2, 3), listOf(15, 16, 17, 4, 5)),
    )
    val out1 =
        when (val tryResult = tensor.sliceAssign(listOf(0 until 3, 0 until 2), src)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = out1.toVec2<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0, 1, 2, 3, 4), listOf(2, 3, 7, 8, 9), listOf(4, 5, 12, 13, 14), listOf(15, 16, 17, 18, 19)),
    )
    return Result.Ok(Unit)
}
