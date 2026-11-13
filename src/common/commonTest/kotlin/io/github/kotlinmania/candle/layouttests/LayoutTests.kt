package io.github.kotlinmania.candle.layouttests

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Device.Cpu
import io.github.kotlinmania.candle.Result
import io.github.kotlinmania.candle.Tensor

public fun contiguous(device: Device): Result<Unit> {
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = tensor.toVec3<UInt>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0, 1, 2, 3), listOf(4, 5, 6, 7), listOf(8, 9, 10, 11)), listOf(listOf(12, 13, 14, 15), listOf(16, 17, 18, 19), listOf(20, 21, 22, 23))),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0, 4, 8), listOf(1, 5, 9), listOf(2, 6, 10), listOf(3, 7, 11)), listOf(listOf(12, 16, 20), listOf(13, 17, 21), listOf(14, 18, 22), listOf(15, 19, 23))),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.transpose(0, 1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0, 1, 2, 3), listOf(12, 13, 14, 15)), listOf(listOf(4, 5, 6, 7), listOf(16, 17, 18, 19)), listOf(listOf(8, 9, 10, 11), listOf(20, 21, 22, 23))),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.transpose(0, 1)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = tensor.i(1 until Int.MAX_VALUE)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.transpose(0, 1)
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(12, 13, 14, 15)), listOf(listOf(16, 17, 18, 19)), listOf(listOf(20, 21, 22, 23))),
    )
    check(
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = tensor.transpose(0, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<UInt>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(listOf(0, 12), listOf(4, 16), listOf(8, 20)), listOf(listOf(1, 13), listOf(5, 17), listOf(9, 21)), listOf(listOf(2, 14), listOf(6, 18), listOf(10, 22)), listOf(listOf(3, 15), listOf(7, 19), listOf(11, 23))),
    )
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun stridedBlocks(): Result<Unit> {
    val tensor =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val matchVal = tensor.stridedBlocks()) {
        is candle.StridedBlocks.SingleBlock -> {
            val startOffset = matchVal.startOffset
            val len = matchVal.len
            check(startOffset == 0)
            check(len == 24)
        }
        else -> {
            val other = matchVal
            error("unexpected block structure $other")
        }
    }
    val tensor1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = Tensor.arange(0u, 26u, Cpu)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.i(2 until Int.MAX_VALUE)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val matchVal1 = tensor1.stridedBlocks()) {
        is candle.StridedBlocks.SingleBlock -> {
            val startOffset1 = matchVal1.startOffset
            val len1 = matchVal1.len
            check(startOffset1 == 2)
            check(len1 == 24)
        }
        else -> {
            val other1 = matchVal1
            error("unexpected block structure $other")
        }
    }
    val tensor2 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor3 =
        when (val tryResult = tensor2.i(1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val matchVal2 = tensor3.stridedBlocks()) {
        is candle.StridedBlocks.SingleBlock -> {
            val startOffset2 = matchVal2.startOffset
            val len2 = matchVal2.len
            check(startOffset2 == 12)
            check(len2 == 12)
        }
        else -> {
            val other2 = matchVal2
            error("unexpected block structure $other")
        }
    }
    val tensor4 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor5 =
        when (
            val tryResult =
                when (val tryResult = tensor4.i(Pair(0 until 0, 1))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.contiguous()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val matchVal3 = tensor5.stridedBlocks()) {
        is candle.StridedBlocks.SingleBlock -> {
            val startOffset3 = matchVal3.startOffset
            val len3 = matchVal3.len
            check(startOffset3 == 0)
            check(len3 == 8)
            check(
                when (val tryResult = tensor5.toVec2<UInt>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } == listOf(listOf(4, 5, 6, 7), listOf(16, 17, 18, 19)),
            )
        }
        else -> {
            val other3 = matchVal3
            error("unexpected block structure $other")
        }
    }
    val tensor6 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensor7 =
        when (val tryResult = tensor6.i(Pair(0 until 0, 1))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val matchVal4 = tensor7.stridedBlocks()) {
        is candle.StridedBlocks.UniformBlocks -> {
            val startOffset4 = matchVal4.startOffset
            val blockLen = matchVal4.blockLen
            val count = matchVal4.count
            val srcStride = matchVal4.srcStride
            check(startOffset4 == 4)
            check(blockLen == 4)
            check(count == 2)
            check(srcStride == 12)
        }
        else -> {
            val other4 = matchVal4
            error("unexpected block structure $other")
        }
    }
    val tensor8 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (
        val matchVal5 =
            when (val tryResult = tensor8.t()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.stridedBlocks()
    ) {
        is candle.StridedBlocks.MultipleBlocks -> {
            val blockStartIndex = matchVal5.blockStartIndex
            val blockLen1 = matchVal5.blockLen
            check(blockLen1 == 1)
            check(blockStartIndex.collect<Vec>() == listOf(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23))
        }
        else -> {
            val other5 = matchVal5
            error("unexpected block structure $other")
        }
    }
    val tensor9 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0u, 24u, Cpu)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(Triple(2, 3, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (
        val matchVal6 =
            when (val tryResult = tensor9.transpose(0, 1)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.stridedBlocks()
    ) {
        is candle.StridedBlocks.MultipleBlocks -> {
            val blockStartIndex1 = matchVal6.blockStartIndex
            val blockLen2 = matchVal6.blockLen
            check(blockLen2 == 4)
            check(blockStartIndex1.collect<Vec>() == listOf(0, 12, 4, 16, 8, 20))
        }
        else -> {
            val other6 = matchVal6
            error("unexpected block structure $other")
        }
    }
    return Result.Ok(Unit)
}
