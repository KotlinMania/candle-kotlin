// port-lint: source cpu/features.rs
package io.github.kotlinmania.candle.cpu.features

internal open class CpuFeatures(
    internal var avx: Boolean,
    internal var avx2: Boolean,
    internal var fma: Boolean,
    internal var f16c: Boolean,
    internal var avx512f: Boolean,
    internal var avx512bw: Boolean,
    internal var avx512vl: Boolean,
    internal var avx512bf16: Boolean,
    internal var avx512vnni: Boolean,
    internal var neon: Boolean,
    internal var dotprod: Boolean,
    internal var i8mm: Boolean,
    internal var fp16: Boolean,
    internal var bf16: Boolean,
)

internal fun get(): CpuFeatures {
    val FEATURES: OnceLock<CpuFeatures>  = OnceLock.new()
    return FEATURES.getOrInit(::detect)
}

public fun detect(): CpuFeatures = CpuFeatures.default().copy(avx = isX86FeatureDetected("avx"), avx2 = isX86FeatureDetected("avx2"), fma = isX86FeatureDetected("fma"), f16c = isX86FeatureDetected("f16c"), avx512f = isX86FeatureDetected("avx512f"), avx512bw = isX86FeatureDetected("avx512bw"), avx512vl = isX86FeatureDetected("avx512vl"), avx512bf16 = isX86FeatureDetected("avx512bf16"), avx512vnni = isX86FeatureDetected("avx512vnni"))
