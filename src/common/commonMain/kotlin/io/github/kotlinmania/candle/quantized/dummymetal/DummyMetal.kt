// port-lint: source quantized/dummy_metal.rs
package io.github.kotlinmania.candle.quantized.dummymetal

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.dummymetalbackend.MetalDevice
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QStorage
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.Shape

public open class QMetalStorage(
    public var dtype: GgmlDType,
    public var device: MetalDevice,
) {
    public open fun dtype(): GgmlDType = this.dtype

    public open fun device(): MetalDevice = this.device

    public open fun dequantize(elemCount: ULong): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun quantize(src: MetalStorage): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun quantizeImatrix(src: MetalStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun quantizeOnto(src: CpuStorage): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun storageSizeInBytes(): Int = 0

    public open fun embedding(rows: ULong, hidden: ULong, ids: MetalStorage, idsL: Layout): Result<MetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun fwd(selfShape: Shape, storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun data(): Result<List<UByte>> = Result.failure(Error.NotCompiledWithMetalSupport)

    public open fun indexedMoeForward(arg0: Shape, arg1: MetalStorage, arg2: Layout, arg3: MetalStorage, arg4: Layout): Result<Pair<MetalStorage, Shape>> = Result.failure(Error.NotCompiledWithMetalSupport)

    public companion object {
        public fun zeros(arg0: MetalDevice, arg1: ULong, arg2: GgmlDType): Result<QMetalStorage> = Result.failure(Error.NotCompiledWithMetalSupport)
    }
}

public fun <T : GgmlType> loadQuantized(device: MetalDevice, data: List<T>): Result<QStorage> = Result.failure(Error.NotCompiledWithMetalSupport)
