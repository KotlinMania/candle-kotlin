package io.github.kotlinmania.candle

/**
 * Implementation of quantized CUDA storage when CUDA support has not been compiled in.
 *
 * Provides stub QCudaStorage methods and quantized loading routines that return
 * NotCompiledWithCudaSupport when invoked without CUDA compilation enabled.
 */
// port-lint: source quantized/dummy_cuda.rs

private val Err: (CandleException) -> CandleResult<Nothing> = { CandleResult.Err(it) }

public class QCudaStorage(
    private val dtype: GgmlDType,
    private val device: CudaDevice,
) {
    private val self: QCudaStorage get() = this

    public fun dtype(): GgmlDType = self.dtype

    public fun device(): CudaDevice = self.device

    public fun dequantize(elemCount: Int): CandleResult<CudaStorage> = Err(CandleException.NotCompiledWithCudaSupport)

    public fun dequantizeF16(elemCount: Int): CandleResult<CudaStorage> = Err(CandleException.NotCompiledWithCudaSupport)

    public fun quantize(src: CudaStorage): CandleResult<Unit> = Err(CandleException.NotCompiledWithCudaSupport)

    public fun quantizeImatrix(
        src: CudaStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    public fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    public fun quantizeOnto(src: CpuStorage): CandleResult<Unit> = Err(CandleException.NotCompiledWithCudaSupport)

    public fun devicePtr(): CandleResult<Long> {
        self
        return Err(CandleException.NotCompiledWithCudaSupport)
    }

    public fun storageSizeInBytes(): Int {
        self
        return 0
    }

    public fun fwd(
        selfShape: Shape,
        storage: CudaStorage,
        layout: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    public fun data(): CandleResult<ByteArray> {
        self
        return Err(CandleException.NotCompiledWithCudaSupport)
    }

    public fun indexedMoeForward(
        s: Shape,
        w: CudaStorage,
        wl: Layout,
        i: CudaStorage,
        il: Layout,
    ): CandleResult<Pair<CudaStorage, Shape>> =
        Err(
            CandleException.NotCompiledWithCudaSupport,
        )

    public companion object {
        public fun zeros(
            d: CudaDevice,
            e: Int,
            dt: GgmlDType,
        ): CandleResult<QCudaStorage> = Err(CandleException.NotCompiledWithCudaSupport)
    }
}

public fun <T> loadQuantized(
    device: CudaDevice,
    data: List<T>,
): CandleResult<QStorage> =
    Err(
        CandleException.NotCompiledWithCudaSupport,
    )
