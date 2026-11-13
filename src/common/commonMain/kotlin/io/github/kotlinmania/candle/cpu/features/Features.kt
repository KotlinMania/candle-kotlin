// port-lint: source cpu/features.rs
package io.github.kotlinmania.candle.cpu.features

import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

internal data class CpuFeatures(
    internal var avx: Boolean = false,
    internal var avx2: Boolean = false,
    internal var fma: Boolean = false,
    internal var f16c: Boolean = false,
    internal var avx512f: Boolean = false,
    internal var avx512bw: Boolean = false,
    internal var avx512vl: Boolean = false,
    internal var avx512bf16: Boolean = false,
    internal var avx512vnni: Boolean = false,
    internal var neon: Boolean = false,
    internal var dotprod: Boolean = false,
    internal var i8mm: Boolean = false,
    internal var fp16: Boolean = false,
    internal var bf16: Boolean = false,
) {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CpuFeatures> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CpuFeatures> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 14L
                override fun readNative(addr: Long): CpuFeatures =
                    CpuFeatures(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 1L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 2L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 3L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 4L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 5L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 6L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 7L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 9L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 10L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 11L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 12L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 13L),
                    )
                override fun writeNative(addr: Long, value: CpuFeatures) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr, value.avx)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 1L, value.avx2)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 2L, value.fma)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 3L, value.f16c)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 4L, value.avx512f)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 5L, value.avx512bw)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 6L, value.avx512vl)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 7L, value.avx512bf16)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 8L, value.avx512vnni)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 9L, value.neon)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 10L, value.dotprod)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 11L, value.i8mm)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 12L, value.fp16)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 13L, value.bf16)
                }
            }
    }
}

internal fun get(): CpuFeatures {
    val FEATURES: OnceLock<CpuFeatures> = OnceLock.new()
    return FEATURES.getOrInit(::detect)
}

public fun detect(): CpuFeatures = CpuFeatures.default().copy(avx = isX86FeatureDetected("avx"), avx2 = isX86FeatureDetected("avx2"), fma = isX86FeatureDetected("fma"), f16c = isX86FeatureDetected("f16c"), avx512f = isX86FeatureDetected("avx512f"), avx512bw = isX86FeatureDetected("avx512bw"), avx512vl = isX86FeatureDetected("avx512vl"), avx512bf16 = isX86FeatureDetected("avx512bf16"), avx512vnni = isX86FeatureDetected("avx512vnni"))
