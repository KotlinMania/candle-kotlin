// port-lint: source ../examples/metal_basics.rs
package io.github.kotlinmania.candle.examples.metal

import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.CandleResult.Ok
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

public fun main(): CandleResult<Unit> {
    // This requires the code to be run with MTL_CAPTURE_ENABLED=1
    val device = Device.newMetal(0)
    val metalDevice = when (device) {
        is Device.Metal -> device.device
        else -> bail("unexpected device")
    }
    metalDevice.capture("candle.gputrace")
    // This first synchronize ensures that a new command buffer gets created after setting up the
    // capture scope.
    device.synchronize()
    val x = Tensor.randn(0.0f, 1.0f, Shape.fromDims(listOf(128, 128)), device)
    val x1 = x.add(x)
    println(x1)
    // This second synchronize ensures that the command buffer gets committed before the end of the
    // capture scope.
    device.synchronize()
    return Ok(Unit)
}
