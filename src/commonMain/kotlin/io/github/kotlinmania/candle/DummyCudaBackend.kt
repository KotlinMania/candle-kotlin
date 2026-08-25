// port-lint: source candle-core/src/dummy_cuda_backend.rs
package io.github.kotlinmania.candle

public data class CudaDevice(
    val gpuId: Int = 0,
) {
    public fun id(): Int = gpuId
}

public class CudaStorage {
    public fun transferToDevice(dst: CudaDevice): CudaStorage =
        throw CandleException.NotCompiledWithCudaSupport
}
