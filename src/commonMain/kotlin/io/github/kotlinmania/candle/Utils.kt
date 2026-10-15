// port-lint: source utils.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.libc.new.apple.libpthread.pthread.pthreadSetQosClassSelfNp
import io.github.kotlinmania.libc.new.apple.libpthread.sys.QosClassT
import io.github.kotlinmania.libc.unix.bsd.apple.sysctlbyname

/**
 * Useful functions for checking features.
 */

public class CandlePool(
    public val numThreads: Int,
) {
    public fun <R> install(f: () -> R): R = f()

    public fun setThreadAffinity() {
        io.github.kotlinmania.candle.setThreadAffinity()
    }
}

private var pool: CandlePool? = null

public fun getNumThreads(): Int {
    // Respond to the same environment variable as rayon.
    return defaultNumThreads()
}

public fun defaultNumThreads(): Int {
    val physical = maxOf(1, perfCoreCount() ?: 4) // safeguard against bad number
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
    val ret = sysctlbyname("hw.perflevel0.logicalcpu", null, size, null, 0uL)
    return if (ret == 0 && count > 0u) count.toInt() else null
}

public fun candlePool(): CandlePool {
    val current = pool
    if (current != null) return current
    setThreadAffinity()
    val created = CandlePool(getNumThreads())
    pool = created
    return created
}

/**
 * Elevate the thread QoS so macOS prefers running on Performance (P) cores.
 */
public fun setThreadAffinity() {
    pthreadSetQosClassSelfNp(QosClassT.QOS_CLASS_USER_INTERACTIVE, 0)
}

public fun <R> withThreadpool(f: () -> R): R = candlePool().install(f)

public fun hasAccelerate(): Boolean = false

public fun hasMkl(): Boolean = false

public fun cudaIsAvailable(): Boolean = false

public fun metalIsAvailable(): Boolean = false

public fun withAvx(): Boolean = false

public fun withNeon(): Boolean = false

public fun withSimd128(): Boolean = false

public fun withF16c(): Boolean = false
