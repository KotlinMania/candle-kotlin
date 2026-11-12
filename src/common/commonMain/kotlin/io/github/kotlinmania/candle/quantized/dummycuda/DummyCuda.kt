// port-lint: source quantized/dummy_cuda.rs
package io.github.kotlinmania.candle.quantized.dummycuda

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.dummycudabackend.CudaDevice
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QStorage
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.Shape

public open class QCudaStorage(
    public var dtype: GgmlDType,
    public var device: CudaDevice,
) {
    public open fun dtype(): GgmlDType = this.dtype

    public open fun device(): CudaDevice = this.device

    public open fun dequantize(elemCount: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun dequantizeF16(elemCount: ULong): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun quantize(src: CudaStorage): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun quantizeImatrix(src: CudaStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: ULong): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun quantizeOnto(src: CpuStorage): Result<Unit> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun devicePtr(): Result<COpaquePointer?> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun storageSizeInBytes(): Int = 0

    public open fun embedding(rows: ULong, hidden: ULong, ids: CudaStorage, idsL: Layout): Result<CudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun fwd(selfShape: Shape, storage: CudaStorage, layout: Layout): Result<Pair<CudaStorage, Shape>> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun data(): Result<List<UByte>> = Result.failure(Error.NotCompiledWithCudaSupport)

    public open fun indexedMoeForward(arg0: Shape, arg1: CudaStorage, arg2: Layout, arg3: CudaStorage, arg4: Layout): Result<Pair<CudaStorage, Shape>> = Result.failure(Error.NotCompiledWithCudaSupport)

    public companion object {
        public fun zeros(arg0: CudaDevice, arg1: ULong, arg2: GgmlDType): Result<QCudaStorage> = Result.failure(Error.NotCompiledWithCudaSupport)
    }
}

public fun <T : GgmlType> loadQuantized(device: CudaDevice, data: List<T>): Result<QStorage> = Result.failure(Error.NotCompiledWithCudaSupport)
