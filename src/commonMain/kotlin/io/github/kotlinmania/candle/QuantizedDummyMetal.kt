// port-lint: source quantized/dummy_metal.rs
package io.github.kotlinmania.candle

public class QMetalStorage(
    private val dtype: GgmlDType,
    private val device: MetalDevice,
) {
    public fun dtype(): GgmlDType = this.dtype

    public fun device(): MetalDevice = this.device

    public fun dequantize(_elemCount: Int): CandleResult<MetalStorage> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun quantize(_src: MetalStorage): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun quantizeImatrix(
        _src: MetalStorage,
        _imatrixWeights: FloatArray,
        _nPerRow: Int,
    ): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun quantizeImatrixOnto(
        _src: CpuStorage,
        _imatrixWeights: FloatArray,
        _nPerRow: Int,
    ): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun quantizeOnto(_src: CpuStorage): CandleResult<Unit> =
        CandleResult.Err(CandleError.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int = 0

    public fun fwd(
        _selfShape: Shape,
        _storage: MetalStorage,
        _layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun data(): CandleResult<ByteArray> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public fun indexedMoeForward(
        _selfShape: Shape,
        _wStorage: MetalStorage,
        _wLayout: Layout,
        _idsStorage: MetalStorage,
        _idsLayout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> =
        CandleResult.Err(CandleError.NotCompiledWithMetalSupport)

    public companion object {
        public fun zeros(
            _device: MetalDevice,
            _elemCount: Int,
            _dtype: GgmlDType,
        ): CandleResult<QMetalStorage> =
            CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
    }
}

public fun <T> loadQuantized(
    _device: MetalDevice,
    _data: List<T>,
): CandleResult<QStorage> =
    CandleResult.Err(CandleError.NotCompiledWithMetalSupport)
