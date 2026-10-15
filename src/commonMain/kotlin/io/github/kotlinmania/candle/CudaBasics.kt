// port-lint: source ../examples/cuda_basics.rs
package io.github.kotlinmania.candle.examples.cuda

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import kotlin.time.TimeSource

fun main() {
    val device = Device.newCuda(0)
    val x = Tensor.randn(0.0f, 1.0f, Shape.fromDims(1024, 64, 1924), device)
    val c = Tensor.randn(0.0f, 1.0f, Shape.fromDims(128, 64, 8), device)
    val x1 = x.conv1d(c, 0, 4, 1, 1)
    if (x1.rank() >= 0) {
        for (i in 0 until 20) {
            val startTime = TimeSource.Monotonic.markNow()
            val iterOut = x.conv1d(c, 0, 4, 1, 1)
            device.synchronize()
            val elapsed = startTime.elapsedNow()
            if (iterOut.rank() >= 0) {
                println("conv1d: $elapsed")
            }
        }
    }
}
