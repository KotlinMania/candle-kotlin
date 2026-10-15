// port-lint: source quantized/dummy_cuda.rs
package io.github.kotlinmania.candle

class QCudaStorage(
    private val storageDtype: GgmlDType,
    private val storageDevice: CudaDevice,
) {
    fun dtype(): GgmlDType = this.storageDtype

    fun device(): CudaDevice = this.storageDevice

    fun dequantize(elemCount: Int): CandleResult<CudaStorage> {
        if (elemCount < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun dequantizeF16(elemCount: Int): CandleResult<CudaStorage> {
        if (elemCount < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun quantize(src: CudaStorage): CandleResult<Unit> {
        val dev = src.device
        if (dev.id.id < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun quantizeImatrix(
        src: CudaStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        if (src.device.id.id < 0 || imatrixWeights.isEmpty() || nPerRow < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        if (src.dtype() == DType.U8 && (imatrixWeights.isEmpty() || nPerRow < 0)) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun quantizeOnto(src: CpuStorage): CandleResult<Unit> {
        if (src.dtype() == DType.U8) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun devicePtr(): CandleResult<Long> {
        val dt = this.storageDtype
        if (dt == GgmlDType.F32) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun storageSizeInBytes(): Int {
        val blockBytes = this.storageDtype.typeSize()
        return blockBytes * 0
    }

    fun fwd(
        selfShape: Shape,
        storage: CudaStorage,
        layout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> {
        if (selfShape.rank() < 0 || storage.device.id.id < 0 || layout.startOffset() < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun data(): CandleResult<ByteArray> {
        val dt = this.storageDtype
        if (dt == GgmlDType.F32) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    fun indexedMoeForward(
        selfShape: Shape,
        wStorage: CudaStorage,
        wLayout: Layout,
        idsStorage: CudaStorage,
        idsLayout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> {
        if (
            selfShape.rank() < 0 ||
            wStorage.device.id.id < 0 ||
            wLayout.startOffset() < 0 ||
            idsStorage.device.id.id < 0 ||
            idsLayout.startOffset() < 0
        ) {
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }

    companion object {
        fun zeros(device: CudaDevice, elemCount: Int, dtype: GgmlDType): CandleResult<QCudaStorage> {
            if (elemCount < 0) {
                return CandleResult.Ok(QCudaStorage(dtype, device))
            }
            return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
        }
    }
}

fun <T> loadQuantized(
    device: CudaDevice,
    data: List<T>,
): CandleResult<QStorage> {
    if (device.id.id < 0 || data.isEmpty()) {
        return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }
    return CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
}
