// port-lint: source quantized/dummy_metal.rs
package io.github.kotlinmania.candle.quantized

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.dummymetalbackend.MetalDevice
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.Shape

public data class QMetalStorage(
    public var dtype: GgmlDType = GgmlDType.entries.first(),
    public var device: MetalDevice = MetalDevice(),
) {
    public fun dequantize(elemCount: ULong): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantize(src: MetalStorage): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantizeImatrix(src: MetalStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantizeOnto(src: CpuStorage): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int = 0

    public fun embedding(rows: ULong, hidden: ULong, ids: MetalStorage, idsL: Layout): Result<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun fwd(selfShape: Shape, storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> = Err(Error.NotCompiledWithMetalSupport)

    public fun data(): Result<List<UByte>> = Err(Error.NotCompiledWithMetalSupport)

    public fun indexedMoeForward(arg0: Shape, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout): Result<Pair<MetalStorage, Shape>> = Err(Error.NotCompiledWithMetalSupport)

    public companion object {
        public fun zeros(arg0: MetalDevice, arg1: ULong, arg2: GgmlDType): Result<QMetalStorage> = Err(Error.NotCompiledWithMetalSupport)
    }
}

public fun <T : GgmlType> loadQuantized(device: MetalDevice, data: List<T>): Result<QStorage> = Err(Error.NotCompiledWithMetalSupport)
