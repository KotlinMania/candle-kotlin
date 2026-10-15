package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error

/**
 * Implementation of quantized CUDA storage when CUDA support has not been compiled in.
 *
 * Provides stub QCudaStorage methods and quantized loading routines that return
 * NotCompiledWithCudaSupport when invoked without CUDA compilation enabled.
 */
// port-lint: source quantized/dummy_cuda.rs

private val Err: (CandleError) -> CandleResult<Nothing> = { CandleResult.Err(it) }

public class QCudaStorage(
    private val dtype: GgmlDType,
    private val device: CudaDevice
) {
    private val self: QCudaStorage get() = this

    public fun dtype(): GgmlDType {
        return self.dtype
    }

    public fun device(): CudaDevice {
        return self.device
    }

    public fun dequantize(
        elemCount: Int
    ): CandleResult<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun dequantizeF16(
        elemCount: Int
    ): CandleResult<CudaStorage> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantize(
        src: CudaStorage
    ): CandleResult<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun quantizeImatrix(
        src: CudaStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int
    ): CandleResult<Unit> =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    public fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int
    ): CandleResult<Unit> =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    public fun quantizeOnto(
        src: CpuStorage
    ): CandleResult<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun devicePtr(): CandleResult<Long> {
        self; return Err(Error.NotCompiledWithCudaSupport)
    }

    public fun storageSizeInBytes(): Int {
        self; return 0
    }

    public fun fwd(
        selfShape: Shape,
        storage: CudaStorage,
        layout: Layout
    ): CandleResult<Pair<CudaStorage, Shape>> =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    public fun data(): CandleResult<ByteArray> {
        self; return Err(Error.NotCompiledWithCudaSupport)
    }

    public fun indexedMoeForward(
        s: Shape,
        w: CudaStorage,
        wl: Layout,
        i: CudaStorage,
        il: Layout
    ): CandleResult<Pair<CudaStorage, Shape>> =
        Err(
            Error.NotCompiledWithCudaSupport
        )

    public companion object {
        public fun zeros(
            d: CudaDevice, e: Int, dt: GgmlDType
        ): CandleResult<QCudaStorage> = Err(Error.NotCompiledWithCudaSupport)
    }
}

public fun <T> loadQuantized(
    device: CudaDevice,
    data: List<T>
): CandleResult<QStorage> = Err(
    Error.NotCompiledWithCudaSupport
)
