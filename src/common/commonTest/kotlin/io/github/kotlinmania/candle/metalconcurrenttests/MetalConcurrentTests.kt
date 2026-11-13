package io.github.kotlinmania.candle.metalconcurrenttests

import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.quantized.GgmlDType
import io.github.kotlinmania.candlecore.quantized.QTensor

// readbacks must wait on the command buffer holding their own blit, not shared device state
@kotlin.test.Test
public fun concurrentReadback(): Result<Unit> {
    val device =
        when (val tryResult = Device.newMetal(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    scope({ scope ->
        for (thread in 0 until 8) {
            val device1 = device.clone()
            scope.spawn({
                for (iter in 0 until 100) {
                    val value = (thread * 1000 + iter).toDouble()
                    val a = Tensor.full(value.toFloat(), Pair(64, 64), device1).unwrap()
                    val b = a.affine(2.0, 1.0).unwrap()
                    val values =
                        b
                            .flattenAll()
                            .unwrap()
                            .toVec1<Float>()
                            .unwrap()
                    val expected = (2.0 * value + 1.0).toFloat()
                    check(values.iter().all({ x -> x == expected })) { "thread $thread iter $iter: expected $expected, got ${((values).slice(0 until 4))}" }
                }
            })
        }
    })
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun concurrentQuantizedDataRoundtrip(): Result<Unit> {
    val device =
        when (val tryResult = Device.newMetal(0)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    scope({ scope ->
        for (thread in 0 until 8) {
            val device1 = device.clone()
            scope.spawn({
                for (iter in 0 until 25) {
                    val src = Tensor.rand(-1.0f, 1.0f, Pair(256, 256), device1).unwrap()
                    val q = QTensor.quantize(src, GgmlDType.Q8_0).unwrap()
                    val bytes = q.data().unwrap()
                    val q2 = QTensor.quantize(src, GgmlDType.Q8_0).unwrap()
                    val bytes2 = q2.data().unwrap()
                    check(bytes == bytes2)
                }
            })
        }
    })
    return Result.Ok(Unit)
}
