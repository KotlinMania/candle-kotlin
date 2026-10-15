// port-lint: source utils.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.envie.getenv
import io.github.kotlinmania.libc.new.apple.libpthread.pthread.pthreadSetQosClassSelfNp
import io.github.kotlinmania.libc.new.apple.libpthread.sys.QosClassT
import io.github.kotlinmania.libc.unix.bsd.apple.sysctlbyname
import io.github.kotlinmania.oncecell.sync.OnceCell

/**
 * Useful functions for checking features.
 */

public class CandlePool(
    public val numThreads: Int,
) {
    public fun setThreadAffinity() {
        // On non-macOS platforms we currently leave thread affinity untouched.
    }
}

private val cfg: (String) -> Boolean = { false }
private val getPhysical: () -> Int = { 4 }

private val POOL: OnceCell<CandlePool> = OnceCell.new()

public fun getNumThreads(): Int {
    // Respond to the same environment variable as rayon.
    return when (val x = getenv("RAYON_NUM_THREADS")?.let { s -> s.toIntOrNull() }) {
        null -> defaultNumThreads()
        else -> if (x > 0) x else defaultNumThreads()
    }
}

public fun defaultNumThreads(): Int {
    val physicalRaw = perfCoreCount() ?: getPhysical()
    val physical = maxOf(physicalRaw, 1) // safeguard against bad number
    return if (physical <= 4) {
        physical
    } else {
        // NOTE: When the CPU backend is optimized further so as not to trash shared
        // caches etc the amount of threads can be increased.
        minOf(physical, physical / 2)
    }
}

/**
 * On Apple Silicon: P-core count via `hw.perflevel0.logicalcpu`.
 * Returns None on Intel Macs (key absent) or on error.
 */
public fun perfCoreCount(): Int? {
    val count = 0u
    val size = 4uL
    val ret = sysctlbyname(
        "hw.perflevel0.logicalcpu",
        null,
        size,
        null,
        0uL,
    )
    return if (ret == 0 && count > 0u) count.toInt() else null
}

public fun candlePool(): CandlePool {
    return POOL.getOrInit {
        setThreadAffinity()
        CandlePool(getNumThreads())
    }
}

/**
 * Elevate the thread QoS so macOS prefers running on Performance (P) cores.
 */
public fun setThreadAffinity() {
    pthreadSetQosClassSelfNp(QosClassT.QOS_CLASS_USER_INTERACTIVE, 0)
}

public fun <R> withThreadpool(f: () -> R): R {
    candlePool().setThreadAffinity()
    return f()
}

public fun hasAccelerate(): Boolean {
    return cfg("accelerate")
}

public fun hasMkl(): Boolean {
    return cfg("mkl")
}

public fun cudaIsAvailable(): Boolean {
    return cfg("cuda")
}

public fun metalIsAvailable(): Boolean {
    return cfg("metal")
}

public fun withAvx(): Boolean {
    return cfg("avx2")
}

public fun withNeon(): Boolean {
    return cfg("neon")
}

public fun withSimd128(): Boolean {
    return cfg("simd128")
}

public fun withF16c(): Boolean {
    return cfg("f16c")
}
