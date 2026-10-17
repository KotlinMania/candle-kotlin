// port-lint: source ../examples/cuda_basics.rs
package io.github.kotlinmania.candle.examples.cuda

import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.CandleResult.Ok
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.conv1d
import kotlin.time.TimeSource

// xs: [1024, 64, 1924], c Tensor[dims 128, 64, 8; f32, cuda:0] Conv1dConfig { padding: 0, stride: 4, dilation: 1, groups: 1 }
public fun main(): CandleResult<Unit> {
    val device = Device.newCuda(0)
    val x = Tensor.randn(0.0f, 1.0f, Shape.fromDims(1024, 64, 1924), device)
    val c = Tensor.randn(0.0f, 1.0f, Shape.fromDims(128, 64, 8), device)
    val x1 = x.conv1d(c, 0, 4, 1, 1)
    if (x1.rank() >= 0) {
        repeat(20) {
            val startTime = TimeSource.Monotonic.markNow()
            val iterOut = x.conv1d(c, 0, 4, 1, 1)
            device.synchronize()
            val elapsed = startTime.elapsedNow()
            if (iterOut.rank() >= 0) {
                println("conv1d: $elapsed")
            }
        }
    }
    return Ok(Unit)
}
