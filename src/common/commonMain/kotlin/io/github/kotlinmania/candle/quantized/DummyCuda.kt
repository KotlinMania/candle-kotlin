// port-lint: source quantized/dummy_cuda.rs
package io.github.kotlinmania.candle.quantized

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr

public data class QCudaStorage(
    public var dtype: GgmlDType = GgmlDType.entries.first(),
    public var device: CudaDevice = CudaDevice(),
) {
    public fun dequantize(elemCount: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun dequantizeF16(elemCount: ULong): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantize(src: CudaStorage): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantizeImatrix(src: CudaStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantizeOnto(src: CpuStorage): Result<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun devicePtr(): Result<ConstPtr> = Err(Error.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int = 0

    public fun embedding(rows: ULong, hidden: ULong, ids: CudaStorage, idsL: Layout): Result<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun fwd(selfShape: Shape, storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> = Err(Error.NotCompiledWithCudaSupport)

    public fun data(): Result<List<UByte>> = Err(Error.NotCompiledWithCudaSupport)

    public fun indexedMoeForward(arg0: Shape, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout): Result<Pair<CudaStorage, Shape>> = Err(Error.NotCompiledWithCudaSupport)

    public companion object {
        public fun zeros(arg0: CudaDevice, arg1: ULong, arg2: GgmlDType): Result<QCudaStorage> = Err(Error.NotCompiledWithCudaSupport)
    }
}

public fun <T : GgmlType> loadQuantized(device: CudaDevice, data: List<T>): Result<QStorage> = Err(Error.NotCompiledWithCudaSupport)
