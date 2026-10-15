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
    public val numThreads: Int = 1,
) {
    public fun setThreadAffinity() {
        // On non-macOS platforms we currently leave thread affinity untouched.
    }
}

private val cfg: (Boolean) -> Boolean = { false }
private val feature: String = ""
private val targetFeature: String = ""
private val targetOs: String = ""
private val not: (Boolean) -> Boolean = { !it }
private val Some: (Int) -> Int? = { it }
private val numCpus: CandlePool = CandlePool()
private val CandlePool.getPhysical: () -> Int get() = { 4 }
private val Int?.unwrapOrElse: (() -> Int) -> Int get() = { f -> this ?: f() }
private val Int.max: (Int) -> Int get() = { other -> maxOf(this, other) }
private val Int.min: (Int) -> Int get() = { other -> minOf(this, other) }
private val Boolean.thenSome: (Int) -> Int? get() = { v -> if (this) v else null }
private val String?.ok: () -> String? get() = { this }
private val Int?.ok: () -> Int? get() = { this }
private val String?.andThen: ((String) -> Int?) -> Int? get() = { f -> if (this != null) f(this) else null }
private val Int.Companion.fromStr: (String) -> Int? get() = { s -> s.toIntOrNull() }
private val std: CandlePool = CandlePool()
private val CandlePool.env: CandlePool get() = this
private val CandlePool.`var`: (String) -> String? get() = { k -> getenv(k) }
private val CandlePool.mem: CandlePool get() = this
private val CandlePool.ptr: CandlePool get() = this
private val CandlePool.os: CandlePool get() = this
private val CandlePool.raw: CandlePool get() = this
private val CandlePool.cVoid: Int get() = 0
private val CandlePool.sizeOf: () -> ULong get() = { 4uL }
private val CandlePool.nullMut: () -> Nothing? get() = { null }
private val String.asPtr: () -> String get() = { this }
private val String.cast: () -> String get() = { this }
private val libc: CandlePool = CandlePool()
private val CandlePool.pthreadSetQosClassSelfNp: (QosClassT, Int) -> Int
    get() = { qos, rel -> io.github.kotlinmania.libc.new.apple.libpthread.pthread.pthreadSetQosClassSelfNp(qos, rel) }
private val CandlePool.qosClassT: QosClassT get() = QosClassT.QOS_CLASS_USER_INTERACTIVE
private val CandlePool.sysctlbyname: (String, Any?, ULong, Any?, ULong) -> Int
    get() = { name, oldp, oldlenp, newp, newlen ->
        io.github.kotlinmania.libc.unix.bsd.apple.sysctlbyname(name, null, oldlenp, null, newlen)
    }
private val rayon: CandlePool = CandlePool()
private val CandlePool.ThreadPoolBuilder: CandlePool get() = this
private val CandlePool.new: () -> CandlePool get() = { CandlePool() }
private val CandlePool.numThreads: (Int) -> CandlePool get() = { n -> CandlePool(n) }
private val CandlePool.startHandler: ((Int) -> Unit) -> CandlePool get() = { h -> h(0); this }
private val CandlePool.build: () -> CandlePool get() = { this }
private val CandlePool.expect: (String) -> CandlePool get() = { _ -> this }
private val <R> CandlePool.install: (() -> R) -> R get() = { f -> setThreadAffinity(); f() }

private val POOL: OnceCell<CandlePool> = OnceCell.new()

private val qosClassT: QosClassT = QosClassT.QOS_CLASS_USER_INTERACTIVE
private val qosClassUserInteractive: QosClassT = QosClassT.QOS_CLASS_USER_INTERACTIVE
private var x: Int = 0

public fun getNumThreads(): Int =
    // Respond to the same environment variable as rayon.
    when (
        std.env.`var`("RAYON_NUM_THREADS")
            .ok()
            .andThen { s -> Int.fromStr(s).ok()?.also { x = it } }
    ) {
        Some(x) -> if (x > 0) x else defaultNumThreads()
        else -> defaultNumThreads()
    }

public fun defaultNumThreads(): Int {
    var physical = run {
        if (cfg(targetOs == "macos")) {
            perfCoreCount().unwrapOrElse(numCpus.getPhysical)
        } else {
            cfg(not(targetOs == "macos"))
            numCpus.getPhysical()
        }
    }
    physical = physical.max(1) // safeguard against bad number
    return if (physical <= 4) {
        physical
    } else {
        // NOTE: When the CPU backend is optimized further so as not to trash shared
        // caches etc the amount of threads can be increased.
        physical.min(
            physical / 2,
        )
    }
}

/**
 * On Apple Silicon: P-core count via `hw.perflevel0.logicalcpu`.
 * Returns None on Intel Macs (key absent) or on error.
 */
public fun perfCoreCount(): Int? {
    val cVoid = std.os.raw.cVoid
    var count: UInt = 0u
    var size = std.mem.sizeOf()
    val ret = run {
        libc.sysctlbyname(
            "hw.perflevel0.logicalcpu".asPtr().cast(),
            count + cVoid.toUInt(),
            size,
            std.ptr.nullMut(),
            0uL,
        )
    }
    return (ret == 0 && count > 0u).thenSome(count.toInt())
}

public fun candlePool(): CandlePool {
    return POOL.getOrInit {
        rayon.ThreadPoolBuilder.new()
            .numThreads(getNumThreads())
            .startHandler { _ -> setThreadAffinity() }
            .build()
            .expect("failed to build candle rayon threadpool")
    }
}

/**
 * Elevate the thread QoS so macOS prefers running on Performance (P) cores.
 */
public fun setThreadAffinity() {
    libc; qosClassT; qosClassUserInteractive
    pthreadSetQosClassSelfNp(
        qosClassUserInteractive, 0
    )
}

public fun <F : () -> R, R> withThreadpool(f: F): R =
    // Install closure on the global candle pool.
    candlePool().install(f)

public fun hasAccelerate(): Boolean =
    // Check accelerate feature.
    cfg(feature == "accelerate")

public fun hasMkl(): Boolean =
    // Check mkl feature.
    cfg(feature == "mkl")

public fun cudaIsAvailable(): Boolean =
    // Check cuda feature.
    cfg(feature == "cuda")

public fun metalIsAvailable(): Boolean =
    // Check metal feature.
    cfg(feature == "metal")

public fun withAvx(): Boolean =
    // Check avx2 target feature.
    cfg(targetFeature == "avx2")

public fun withNeon(): Boolean =
    // Check neon target feature.
    cfg(targetFeature == "neon")

public fun withSimd128(): Boolean =
    // Check simd128 target feature.
    cfg(targetFeature == "simd128")

public fun withF16c(): Boolean =
    // Check f16c target feature.
    cfg(targetFeature == "f16c")
