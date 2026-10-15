// port-lint: source ../examples/metal_basics.rs
package io.github.kotlinmania.candle.examples.metal

import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor
import io.github.kotlinmania.candle.bail

fun main() {
    val device = Device.newMetal(0)
    val metalDevice = when (device) {
        is Device.Metal -> device.device
        else -> bail("unexpected device")
    }
    metalDevice.capture("candle.gputrace")
    device.synchronize()
    val x = Tensor.randn(0.0f, 1.0f, Shape.fromDims(128, 128), device)
    val x1 = x.add(x)
    println(x1)
    device.synchronize()
}
