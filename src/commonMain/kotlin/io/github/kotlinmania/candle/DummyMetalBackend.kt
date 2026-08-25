// port-lint: source candle-core/src/dummy_metal_backend.rs
package io.github.kotlinmania.candle

public data class MetalDevice(
    val gpuId: Int = 0,
) {
    public fun id(): Int = gpuId
}

public class MetalStorage {
    public fun transferToDevice(dst: MetalDevice): MetalStorage =
        throw CandleException.NotCompiledWithMetalSupport
}
