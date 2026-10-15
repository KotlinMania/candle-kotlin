package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error

/**
 * Implementation of quantized Metal storage when Metal support has not been compiled in.
 *
 * Provides stub QMetalStorage methods and quantized loading routines that return
 * NotCompiledWithMetalSupport when invoked without Metal compilation enabled.
 */
// port-lint: source quantized/dummy_metal.rs

private val Err: (CandleError) -> CandleResult<Nothing> = { CandleResult.Err(it) }

public class QMetalStorage(
    private val dtype: GgmlDType,
    private val device: MetalDevice
) {
    private val self: QMetalStorage get() = this

    public fun dtype(): GgmlDType {
        return self.dtype
    }

    public fun device(): MetalDevice {
        return self.device
    }

    public fun dequantize(
        elemCount: Int
    ): CandleResult<MetalStorage> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantize(
        src: MetalStorage
    ): CandleResult<Unit> = Err(Error.NotCompiledWithMetalSupport)

    public fun quantizeImatrix(
        src: MetalStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int
    ): CandleResult<Unit> =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    public fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int
    ): CandleResult<Unit> =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    public fun quantizeOnto(
        src: CpuStorage
    ): CandleResult<Unit> = Err(Error.NotCompiledWithCudaSupport)

    public fun storageSizeInBytes(): Int {
        self; return 0
    }

    public fun fwd(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout
    ): CandleResult<Pair<MetalStorage, Shape>> =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    public fun data(): CandleResult<ByteArray> {
        self; return Err(Error.NotCompiledWithMetalSupport)
    }

    public fun indexedMoeForward(
        s: Shape,
        w: MetalStorage,
        wl: Layout,
        i: MetalStorage,
        il: Layout
    ): CandleResult<Pair<MetalStorage, Shape>> =
        Err(
            Error.NotCompiledWithMetalSupport
        )

    public companion object {
        public fun zeros(
            d: MetalDevice, e: Int, dt: GgmlDType
        ): CandleResult<QMetalStorage> = Err(Error.NotCompiledWithMetalSupport)
    }
}

public fun <T> loadQuantized(
    device: MetalDevice,
    data: List<T>
): CandleResult<QStorage> = Err(
    Error.NotCompiledWithMetalSupport
)
