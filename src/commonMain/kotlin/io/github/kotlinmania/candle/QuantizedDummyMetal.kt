// port-lint: source quantized/dummy_metal.rs
package io.github.kotlinmania.candle

class QMetalStorage(
    private val storageDtype: GgmlDType,
    private val storageDevice: MetalDevice,
) {
    fun dtype(): GgmlDType = this.storageDtype

    fun device(): MetalDevice = this.storageDevice

    fun dequantize(elemCount: Int): CandleResult<MetalStorage> {
        if (elemCount < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun quantize(src: MetalStorage): CandleResult<Unit> {
        if (src.dtype == DType.U8) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun quantizeImatrix(
        src: MetalStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        if (src.dtype == DType.U8 && (imatrixWeights.isEmpty() || nPerRow < 0)) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        if (src.dtype() == DType.U8 && (imatrixWeights.isEmpty() || nPerRow < 0)) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun quantizeOnto(src: CpuStorage): CandleResult<Unit> {
        if (src.dtype() == DType.U8) {
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
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        if (selfShape.rank() < 0 || storage.dtype == DType.U8 || layout.startOffset() < 0) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun data(): CandleResult<ByteArray> {
        val dt = this.storageDtype
        if (dt == GgmlDType.F32) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    fun indexedMoeForward(
        selfShape: Shape,
        wStorage: MetalStorage,
        wLayout: Layout,
        idsStorage: MetalStorage,
        idsLayout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        if (
            selfShape.rank() < 0 ||
            wStorage.dtype == DType.U8 ||
            wLayout.startOffset() < 0 ||
            idsStorage.dtype == DType.U8 ||
            idsLayout.startOffset() < 0
        ) {
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }

    companion object {
        fun zeros(device: MetalDevice, elemCount: Int, dtype: GgmlDType): CandleResult<QMetalStorage> {
            if (elemCount < 0) {
                return CandleResult.Ok(QMetalStorage(dtype, device))
            }
            return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
        }
    }
}

fun <T> loadQuantized(
    device: MetalDevice,
    data: List<T>,
): CandleResult<QStorage> {
    if (device.id.id < 0 || data.isEmpty()) {
        return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }
    return CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
}
