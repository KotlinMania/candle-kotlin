// port-lint: source quantized/dummy_cuda.rs
package io.github.kotlinmania.candle

public class QCudaStorage(
    private val dtype: GgmlDType,
    private val device: CudaDevice,
) {
    public fun dtype(): GgmlDType = this.dtype

    public fun device(): CudaDevice = this.device

    public fun dequantize(_elemCount: Int): CandleResult<CudaStorage> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun dequantizeF16(_elemCount: Int): CandleResult<CudaStorage> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun quantize(_src: CudaStorage): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun quantizeImatrix(
        _src: CudaStorage,
        _imatrixWeights: FloatArray,
        _nPerRow: Int,
    ): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun quantizeImatrixOnto(
        _src: CpuStorage,
        _imatrixWeights: FloatArray,
        _nPerRow: Int,
    ): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun quantizeOnto(_src: CpuStorage): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun devicePtr(): CandleResult<Long> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int = 0

    public fun fwd(
        _selfShape: Shape,
        _storage: CudaStorage,
        _layout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun data(): CandleResult<ByteArray> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun indexedMoeForward(
        _selfShape: Shape,
        _wStorage: CudaStorage,
        _wLayout: Layout,
        _idsStorage: CudaStorage,
        _idsLayout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public companion object {
        public fun zeros(
            _device: CudaDevice,
            _elemCount: Int,
            _dtype: GgmlDType,
        ): CandleResult<QCudaStorage> =
            CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
    }
}

public fun <T> loadQuantized(
    _device: CudaDevice,
    _data: List<T>,
): CandleResult<QStorage> =
    CandleResult.Err(CandleError.NotCompiledWithCudaSupport)
