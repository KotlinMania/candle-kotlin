package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException

/**
 * Implementation of quantized Metal storage when Metal support has not been compiled in.
 *
 * Provides stub QMetalStorage methods and quantized loading routines that return
 * NotCompiledWithMetalSupport when invoked without Metal compilation enabled.
 */
// port-lint: source quantized/dummy_metal.rs

private val Err: (CandleException) -> CandleResult<Nothing> = { CandleResult.Err(it) }

public class QMetalStorage(
    private val dtype: GgmlDType,
    private val device: MetalDevice,
    public val buffer: Any? = null,
) {
    private val self: QMetalStorage get() = this

    public fun dtype(): GgmlDType = self.dtype

    public fun device(): MetalDevice = self.device

    public fun dequantize(elemCount: Int): CandleResult<MetalStorage> = Err(CandleException.NotCompiledWithMetalSupport)

    public fun quantize(src: MetalStorage): CandleResult<Unit> = Err(CandleException.NotCompiledWithMetalSupport)

    public fun quantizeImatrix(
        src: MetalStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> =
        Err(
            CandleException.NotCompiledWithMetalSupport,
        )

    public fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> =
        Err(
            CandleException.NotCompiledWithMetalSupport,
        )

    public fun quantizeOnto(src: CpuStorage): CandleResult<Unit> = Err(CandleException.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int {
        self
        return 0
    }

    public fun fwd(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> =
        Err(
            CandleException.NotCompiledWithMetalSupport,
        )

    public fun data(): CandleResult<ByteArray> {
        self
        return Err(CandleException.NotCompiledWithMetalSupport)
    }

    public fun indexedMoeForward(
        s: Shape,
        w: MetalStorage,
        wl: Layout,
        i: MetalStorage,
        il: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> =
        Err(
            CandleException.NotCompiledWithMetalSupport,
        )

    public companion object {
        public fun zeros(
            d: MetalDevice,
            e: Int,
            dt: GgmlDType,
        ): CandleResult<QMetalStorage> = Err(CandleException.NotCompiledWithMetalSupport)
    }
}

public fun <T> loadQuantized(
    device: MetalDevice,
    data: List<T>,
): CandleResult<QStorage> =
    Err(
        CandleException.NotCompiledWithMetalSupport,
    )
