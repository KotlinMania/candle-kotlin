// port-lint: source utils.rs
package io.github.kotlinmania.candle.utils

import io.github.kotlinmania.candle.RustAny
import io.github.kotlinmania.candle.cpu.features.get
import io.github.kotlinmania.libc.pthreadSetQosClassSelfNp
import io.github.kotlinmania.libc.qosclasst.QOS_CLASS_USER_INTERACTIVE
import io.github.kotlinmania.libc.sysctlbyname
import io.github.kotlinmania.numcpus.getPhysical
import io.github.kotlinmania.rayon.ThreadPool
import io.github.kotlinmania.rustlang.core.ffi.CVoid
import io.github.kotlinmania.rustlang.core.mem.AddressableLocal
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.std.cell.UnsafeCell
import io.github.kotlinmania.rustlang.std.hint.spinLoop
import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.OnceLock
import io.github.kotlinmania.rustlang.std.sync.atomic.AtomicBool
import io.github.kotlinmania.rustlang.std.sync.atomic.AtomicUsize
import io.github.kotlinmania.rustlang.std.sync.atomic.Ordering
import io.github.kotlinmania.rustlang.std.thread.Thread

// Useful functions for checking features.

public fun interface WorkDescTrampolineCallback {
    public operator fun invoke(arg0: ConstPtr<Unit>, arg1: Int)
}

public val DEFAULT_BARRIER_POOL_SPIN_LIMIT: UInt = 25000u

/**
 * Per-worker control on its own 64-byte cache line to prevent false sharing.
 */
public data class Slot(
    /**
     * `Release`-stored by parent to signal work. `Acquire-loaded by worker.
     */
    public var go: AtomicUsize = AtomicUsize(),
    /**
     * `Release`-stored by worker (and its subtree) when done. `Acquire`-loaded by parent.
     */
    public var done: AtomicUsize = AtomicUsize(),
)

/**
 * Closure pointer written by main before the `Release` on each root slot's `go`.
 */
public class WorkDesc internal constructor(
    public var trampoline: WorkDescTrampolineCallback,
    dataCtorArg: ConstPtr<Unit>,
) {
    public var data: ConstPtr<Unit> = dataCtorArg
        internal set

    public constructor() : this(
        trampoline = WorkDescTrampolineCallback { _, _ -> },
        dataCtorArg = ConstPtr(0L),
    )

    public operator fun component1(): WorkDescTrampolineCallback = this.trampoline

    public operator fun component2(): ConstPtr<Unit> = this.data

    public fun copy(trampoline: WorkDescTrampolineCallback = this.trampoline, data: ConstPtr<Unit> = this.data): WorkDesc = WorkDesc(trampoline = trampoline, dataCtorArg = data)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is WorkDesc) return false
        if (this.trampoline != other.trampoline) return false
        if (this.data != other.data) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.trampoline.hashCode()
        result = 31 * result + this.data.hashCode()
        return result
    }

    public companion object {
        public operator fun invoke(
            trampoline: WorkDescTrampolineCallback = WorkDescTrampolineCallback { _, _ -> },
            data: ConstPtr<Unit> = ConstPtr(0L),
        ): WorkDesc =
            WorkDesc(
                trampoline = trampoline,
                dataCtorArg = data,
            )
    }
}

public class BarrierPoolInner internal constructor(
    public var slots: List<Slot>,
    handlesCtorArg: OnceLock<List<Thread>>,
    workCtorArg: UnsafeCell<WorkDesc>,
    public var stop: AtomicBool,
    panicPayloadCtorArg: Mutex<RustAny?>,
) {
    /**
     * Thread handles for unpark.
     */
    public var handles: OnceLock<List<Thread>> = handlesCtorArg
        internal set

    public var work: UnsafeCell<WorkDesc> = workCtorArg
        internal set

    public var panicPayload: Mutex<RustAny?> = panicPayloadCtorArg
        internal set

    public constructor() : this(
        slots = emptyList(),
        handlesCtorArg = OnceLock<List<Thread>>(),
        workCtorArg = UnsafeCell<WorkDesc>(),
        stop = AtomicBool(),
        panicPayloadCtorArg = Mutex<RustAny?>(),
    )

    public operator fun component1(): List<Slot> = this.slots

    public operator fun component2(): OnceLock<List<Thread>> = this.handles

    public operator fun component3(): UnsafeCell<WorkDesc> = this.work

    public operator fun component4(): AtomicBool = this.stop

    public operator fun component5(): Mutex<RustAny?> = this.panicPayload

    public fun copy(slots: List<Slot> = this.slots, handles: OnceLock<List<Thread>> = this.handles, work: UnsafeCell<WorkDesc> = this.work, stop: AtomicBool = this.stop, panicPayload: Mutex<RustAny?> = this.panicPayload): BarrierPoolInner = BarrierPoolInner(slots = slots, handlesCtorArg = handles, workCtorArg = work, stop = stop, panicPayloadCtorArg = panicPayload)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BarrierPoolInner) return false
        if (this.slots != other.slots) return false
        if (this.handles != other.handles) return false
        if (this.work != other.work) return false
        if (this.stop != other.stop) return false
        if (this.panicPayload != other.panicPayload) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.slots.hashCode()
        result = 31 * result + this.handles.hashCode()
        result = 31 * result + this.work.hashCode()
        result = 31 * result + this.stop.hashCode()
        result = 31 * result + this.panicPayload.hashCode()
        return result
    }

    public companion object {
        public operator fun invoke(
            slots: List<Slot> = emptyList(),
            handles: OnceLock<List<Thread>> = OnceLock<List<Thread>>(),
            work: UnsafeCell<WorkDesc> = UnsafeCell<WorkDesc>(),
            stop: AtomicBool = AtomicBool(),
            panicPayload: Mutex<RustAny?> = Mutex<RustAny?>(),
        ): BarrierPoolInner =
            BarrierPoolInner(
                slots = slots,
                handlesCtorArg = handles,
                workCtorArg = work,
                stop = stop,
                panicPayloadCtorArg = panicPayload,
            )
    }
}

/**
 * Two-level tree barrier pool.
 *
 * Main signals one leader per CPU cluster (O(nClusters) stores, not O(nWorkers)).
 * Leaders fan work out to intra-cluster followers while main executes its own
 * slice. On-completion reduction mirrors the fan-out: followers -> leader -> main.
 * Main drains only nCluster slots instead of nWorker slots on completion.
 */
public class BarrierPool internal constructor(
    public var inner: BarrierPoolInner,
    threadsCtorArg: MutableList<JoinHandle<Unit>>,
    /**
     * Workers that main signals directly
     */
    public var rootWorkers: List<Int>,
    callLockCtorArg: Mutex<Int>,
) {
    public var threads: MutableList<JoinHandle<Unit>> = threadsCtorArg
        internal set

    /**
     * Serializes concurrent callers. Stores the current generation.
     */
    public var callLock: Mutex<Int> = callLockCtorArg
        internal set

    public constructor() : this(
        inner = BarrierPoolInner(),
        threadsCtorArg = mutableListOf(),
        rootWorkers = emptyList(),
        callLockCtorArg = Mutex<Int>(),
    )

    public operator fun component1(): BarrierPoolInner = this.inner

    public operator fun component2(): MutableList<JoinHandle<Unit>> = this.threads

    public operator fun component3(): List<Int> = this.rootWorkers

    public operator fun component4(): Mutex<Int> = this.callLock

    public fun copy(inner: BarrierPoolInner = this.inner, threads: MutableList<JoinHandle<Unit>> = this.threads, rootWorkers: List<Int> = this.rootWorkers, callLock: Mutex<Int> = this.callLock): BarrierPool = BarrierPool(inner = inner, threadsCtorArg = threads, rootWorkers = rootWorkers, callLockCtorArg = callLock)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BarrierPool) return false
        if (this.inner != other.inner) return false
        if (this.threads != other.threads) return false
        if (this.rootWorkers != other.rootWorkers) return false
        if (this.callLock != other.callLock) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.inner.hashCode()
        result = 31 * result + this.threads.hashCode()
        result = 31 * result + this.rootWorkers.hashCode()
        result = 31 * result + this.callLock.hashCode()
        return result
    }

    public fun nWorkers(): Int = this.threads.size

    public fun execute(f: (Int) -> Unit) {
        val n = this.threads.size
        if (n == 0) {
            f(0)
            return
        }
        var guard = this.callLock.lock().unwrap()
        val newGen = guard.wrappingAdd(1)
        guard = newGen
        run {
            val work = this.inner.work.get()
            work.trampoline = WorkDescTrampolineCallback { data, tid -> callTrampoline<(Int) -> Unit>(data, tid) }
            work.data = ConstPtr<Unit>((constPtrFromRustCast<(Int) -> Unit>(f)).rawAddress, (constPtrFromRustCast<(Int) -> Unit>(f)).rawMeta)
        }
        val handles = this.inner.handles.get()!!
        for (root in this.rootWorkers) {
            this.inner.slots[root]
                .go
                .store(newGen, Ordering.Release)
            handles[root].unpark()
        }
        // Guard waits for workers and propagates panics on drop.
        // Main thread participates as worker n (workers are 0..n-1).
        f(n)
        // _guard drops here, waiting for root workers.
    }

    // Uniform, short work units can avoid shared-cursor contention by assigning
    // one contiguous range per thread.
    internal fun executeStatic(nItems: Int, f: (Range<Int>) -> Unit) {
        if (nItems == 0) {
            return
        }
        if (IN_BARRIER_POOL.with({ f1 -> f1.get() })) {
            f(0 until nItems)
            return
        }
        val chunk = nItems.divCeil(this.threads.size + 1)
        this.execute({ tid ->
            val start = tid * chunk
            if (start < nItems) {
                f(start until nItems.min(start + chunk))
            }
        })
    }

    // Static per-tid slices make every op wait for the slowest thread; a shared cursor
    // lets fast threads absorb a straggler's work instead of spinning at the barrier.
    public fun executeChunked(nItems: Int, f: (Range<Int>) -> Unit) {
        if (nItems == 0) {
            return
        }
        // nested use from a worker would deadlock on the call lock; run inline instead
        if (IN_BARRIER_POOL.with({ f1 -> f1.get() })) {
            f(0 until nItems)
            return
        }
        val nThreads = this.threads.size + 1
        // ~8 grabs per thread amortizes cursor contention while keeping tail imbalance small
        val chunk = nItems.divCeil(nThreads * 8).max(1)
        val cursor = AtomicUsize.new(0)
        this.execute({
            while (true) {
                val start = cursor.fetchAdd(chunk, Ordering.Relaxed)
                if (start >= nItems) {
                    break
                }
                f(start until nItems.min(start + chunk))
            }
        })
    }

    public fun drop() {
        if (this.threads.isEmpty()) {
            return
        }
        this.inner.stop.store(true, Ordering.Relaxed)
        val newGen =
            this.callLock
                .lock()
                .unwrap()
                .wrappingAdd(1)
        val handles = this.inner.handles.get()
        // Signal ALL slots and unpark them so parked workers wake immediately.
        for ((i, slot) in this.inner.slots.enumerate()) {
            slot.go.store(newGen, Ordering.Release)
            if (handles != null) {
                val hs = handles
                hs[i].unpark()
            }
        }
        for (t in this.threads.drain(0 until 0)) {
            t.join()
        }
    }

    public companion object {
        public operator fun invoke(
            inner: BarrierPoolInner = BarrierPoolInner(),
            threads: MutableList<JoinHandle<Unit>> = mutableListOf(),
            rootWorkers: List<Int> = emptyList(),
            callLock: Mutex<Int> = Mutex<Int>(),
        ): BarrierPool =
            BarrierPool(
                inner = inner,
                threadsCtorArg = threads,
                rootWorkers = rootWorkers,
                callLockCtorArg = callLock,
            )

        public fun new(nWorkers: Int): BarrierPool {
            val inner = BarrierPoolInner(slots = (0 until nWorkers).map({ Slot(go = AtomicUsize.new(0), done = AtomicUsize.new(0)) }).collect<Vec>().intoBoxedSlice(), handles = OnceLock.new(), work = UnsafeCell.new(WorkDesc(trampoline = {}, data = ConstPtr(0L))), stop = AtomicBool.new(false), panicPayload = Mutex.new(null))
            val innerPtr = (constPtrFromRustCast<BarrierPoolInner>(inner)).rawAddress.toInt()
            val clusterSize = getClusterSize()
            val (children, rootWorkers) = computeTree(nWorkers, clusterSize)
            val spinLimit = barrierPoolSpinLimit()
            val threads: Vec =
                (0 until nWorkers)
                    .map({ tid ->
                        val children1 = children[tid].clone()
                        Builder
                            .new()
                            .name("candle-bp-$tid")
                            .spawn(
                                closure@{
                                    IN_BARRIER_POOL.with({ f -> f.set(true) })
                                    setThreadAffinity()
                                    val inner1 = (ObjectSlotLayout<BarrierPoolInner>()).readNative(((ConstPtr<BarrierPoolInner>(innerPtr.toLong()))).rawAddress)
                                    val slot = inner1.slots[tid]
                                    var gen = 0
                                    while (true) {
                                        // Spin briefly then park, waiting for next work item.
                                        var spins = 0u
                                        while (true) {
                                            val g = slot.go.load(Ordering.Acquire)
                                            if (g != gen) {
                                                gen = g
                                                break
                                            }
                                            if (inner1.stop.load(Ordering.Relaxed)) {
                                                return@closure
                                            }
                                            spins += 1u
                                            if (spins < spinLimit) {
                                                spinLoop()
                                            } else {
                                                // Park deposits a token; unpark() before park() returns immediately — no race.
                                                park()
                                                spins = 0u
                                            }
                                        }
                                        if (inner1.stop.load(Ordering.Relaxed)) {
                                            break
                                        }
                                        // Fan work out to children.
                                        val handles = inner1.handles.get()!!
                                        for (child in children1) {
                                            inner1.slots[child].go.store(gen, Ordering.Release)
                                            handles[child].unpark()
                                        }
                                        // Execute own work, catching panics so done is always signalled.
                                        when (
                                            val matchVal =
                                                catchUnwind(
                                                    AssertUnwindSafe({
                                                        val work = inner1.work.get()
                                                        (work.trampoline)(work.data, tid)
                                                    }),
                                                )
                                        ) {
                                            is Err -> {
                                                val (payload) = matchVal
                                                var p = inner1.panicPayload.lock().unwrap()
                                                if (p == null) {
                                                    p = payload
                                                }
                                            }
                                            else -> {}
                                        }
                                        // Reduce: wait for all children before signalling parent.
                                        for (child1 in children1) {
                                            while (inner1.slots[child1].done.load(Ordering.Acquire) != gen) {
                                                spinLoop()
                                            }
                                        }
                                        // Signal done unconditionally so main/parent never deadlocks.
                                        slot.done.store(gen, Ordering.Release)
                                    }
                                },
                            ).expect("failed to spawn candle barrier pool worker")
                    })
                    .collect()
            inner.handles
                .set(
                    threads
                        .iter()
                        .map({ jh -> jh.thread().clone() })
                        .collect<Vec>()
                        .intoBoxedSlice(),
                ).ok()
            return BarrierPool(inner = inner, threads = threads, rootWorkers = rootWorkers, callLock = Mutex.new(0))
        }
    }
}

public fun <F : (Int) -> Unit> callTrampoline(data: ConstPtr<Unit>, tid: Int) {
    (ObjectSlotLayout<(Int) -> Unit>()).readNative(((ConstPtr<(Int) -> Unit>(data.rawAddress, data.rawMeta))).rawAddress)(tid)
}

/**
 * Build the two-level fan-out tree. First worker in each cluster is leader.
 */
public fun computeTree(nWorkers: Int, clusterSize: Int): Pair<List<List<Int>>, List<Int>> {
    val children = mutableListOf(mutableListOf(), nWorkers)
    if (nWorkers == 0 || clusterSize <= 1 || clusterSize == Int.MAX_VALUE) {
        // Flat: every worker is a root.
        return Pair(children, (0 until nWorkers).collect())
    }
    val roots = mutableListOf()
    // First cluster: main + (clusterSize - 1) workers.
    val c0 = (clusterSize - 1).min(nWorkers)
    roots.add(0)
    for (f in 1 until c0) {
        children[0].push(f)
    }
    // Remaining clusters: clusterSize workers each.
    var next = c0
    while (next < nWorkers) {
        val leader = next
        roots.add(leader)
        val end = nWorkers.min(next + clusterSize)
        for (f1 in (leader + 1) until end) {
            children[leader].push(f1)
        }
        next = end
    }
    return Pair(children, roots)
}

public fun getClusterSize(): Int {
    // Later we should use `sysctlbyname("hw.perflevel0.cpusperl2")` and similar for other systems to get precise cluster size.
    // For now we use these hardcoded numbers.
    // Using `Int.MAX` simply means that the tree is flat. Every core is it's own leader.
    return 6
}

public fun barrierPoolSpinLimit(): UInt = variable("CANDLE_BARRIER_POOL_SPIN_LIMIT").ok().andThen({ s -> UInt.fromStr(s).ok() }).unwrapOr(DEFAULT_BARRIER_POOL_SPIN_LIMIT)

/**
 * RAII guard that blocks until all root workers have finished.
 */
public data class BarrierGuard(
    public var inner: BarrierPoolInner = BarrierPoolInner(),
    public var roots: List<Int> = emptyList(),
    public var gen: Int = 0,
) {
    public fun drop() {
        for (root in this.roots) {
            while (this.inner.slots[root]
                    .done
                    .load(Ordering.Acquire) != this.gen
            ) {
                spinLoop()
            }
        }
        // Always clear any stored panic. Propagate unless main is not already unwinding.
        val payload =
            this.inner.panicPayload
                .lock()
                .unwrap()
                .take()
        if (!panicking()) {
            if (payload != null) {
                val p = payload
                resumeUnwind(p)
            }
        }
    }
}

public var BARRIER_POOL: OnceLock<BarrierPool> = OnceLock.new()

/**
 * Persistent barrier pool
 */
public fun barrierPool(): BarrierPool = BARRIER_POOL.getOrInit({ BarrierPool.new(candleNumThreads().saturatingSub(1)) })

public fun <R> withThreadpool(f: () -> R): R = candlePool().install(f)

public fun initGlobalThreadpool() {
    setThreadAffinity()
    ThreadPoolBuilder
        .new()
        .numThreads(getNumThreads())
        .startHandler({ setThreadAffinity() })
        .buildGlobal()
}

public fun defaultNumThreads(): Int {
    val physical =
        run {
            perfCoreCount()
                ?: (linuxEnvCpuMask() ?: linuxPreferredCpus())?.size
                ?: getPhysical()
        }
    return physical.max(1)
    // safeguard against bad number
}

public fun rayonNumThreads(): Int =
    variable("RAYON_NUM_THREADS")
        .ok()
        .andThen({ s -> Int.fromStr(s).ok() })
        .filter({ nt -> nt > 0 })
        .unwrapOrElse(::defaultNumThreads)

public fun candleNumThreads(): Int =
    variable("CANDLE_NUM_THREADS")
        .ok()
        .andThen({ s -> Int.fromStr(s).ok() })
        .filter({ nt -> nt > 0 })
        .unwrapOrElse(::rayonNumThreads)

public fun getNumThreads(): Int {
    // Respond to the same environment variable as rayon.
    return rayonNumThreads()
}

/**
 * On Apple Silicon: P-core count via `hw.perflevel0.logicalcpu`.
 * Returns None on Intel Macs (key absent) or on error.
 */
public fun perfCoreCount(): Int? {
    val localStorageL429C4 = AddressableLocal(0u, PrimitiveLayouts.U32)
    var count by localStorageL429C4
    val size = sizeOf<UInt>(typeLayoutT = PrimitiveLayouts.U32)
    val ret = sysctlbyname("hw.perflevel0.logicalcpu".asPtr().cast(), MutPtr<CVoid>((MutPtr<UInt>(localStorageL429C4.asMutPtr().rawAddress)).rawAddress, (MutPtr<UInt>(localStorageL429C4.asMutPtr().rawAddress)).rawMeta), size, MutPtr(0L), 0)
    return (ret == 0 && count > 0u).thenSome(count.toInt())
}

public fun linuxPreferredCpus(): List<Int>? {
    val CPUS: OnceLock<List<Int>?> = OnceLock.new()
    return CPUS.getOrInit(::linuxDetectPreferredCpus).asDeref()
}

public fun linuxEnvCpuMask(): List<Int>? {
    val CPUS: OnceLock<List<Int>?> = OnceLock.new()
    return CPUS
        .getOrInit({
            variable("CANDLE_CPU_MASK").ok().andThen({ mask -> linuxParseCpuList(mask) })
        })
        .asDeref()
}

public fun linuxAffinityEnabled(): Boolean = variable("CANDLE_CPU_AFFINITY").map({ v -> v != "0" && !v.eqIgnoreAsciiCase("false") }).unwrapOr(false)

public fun linuxAffinityCpus(): List<Int>? = (linuxEnvCpuMask() ?: linuxAffinityEnabled().then(::linuxPreferredCpus).flatten())

public fun linuxParseCpuList(mask: String): List<Int>? {
    val cpus = mutableListOf()
    for (part in mask.split(',').map(String::trim).filter({ part -> !part.isEmpty() })) {
        val (start2, end2) =
            when (val matchVal = part.splitOnce('-')) {
                null -> {
                    val cpu = (part.parse<Int>().ok() ?: return null)
                    Pair(cpu, cpu)
                }
                else -> {
                    val start = matchVal.first
                    val end = matchVal.second
                    val start1 = (start.trim().parse<Int>().ok() ?: return null)
                    val end1 = (end.trim().parse<Int>().ok() ?: return null)
                    Pair(start1, end1)
                }
            }
        if (start2 > end2) {
            return null
        }
        cpus.extend(start2..end2)
    }
    cpus.sortUnstable()
    cpus.dedup()
    return (!cpus.isEmpty()).thenSome(cpus)
}

public fun linuxDetectPreferredCpus(): List<Int>? {
    data class Cpu(
        public var idx: Int = 0,
        public var capacity: Int = 0,
    )
    val cpus = mutableListOf()
    for (entry in (readDir("/sys/devices/system/cpu").ok() ?: return null).flatten()) {
        val cpuName = entry.fileName()
        val cpuName1 = (cpuName.toStr() ?: return null)
        val idx = (cpuName1.takeIf { it.startsWith("cpu") }?.removePrefix("cpu"))?.let({ idx -> idx.parse<Int>().ok() }) ?: continue
        val path = entry.path()
        val capacity = ((readToString(path.join("cpu_capacity")).ok() ?: return null).trim().parse<Int>().ok() ?: return null)
        cpus.add(Cpu(idx = idx, capacity = capacity))
    }
    if (cpus.size < 2) {
        return null
    }
    val min = (ArrayDeque((ArrayDeque(cpus)).map({ cpu -> cpu.capacity })).min() ?: return null)
    val max = (ArrayDeque((ArrayDeque(cpus)).map({ cpu -> cpu.capacity })).max() ?: return null)
    if (max == 0 || min * 10 >= max * 9) {
        return null
    }
    val totalCpus = cpus.size
    val preferred = ArrayDeque((ArrayDeque(cpus).filter({ cpu -> cpu.capacity * 10 >= max * 9 })).map({ cpu -> cpu.idx })).collect<Vec>()
    preferred.sortUnstable()
    return (preferred.size < totalCpus).thenSome(preferred)
}

public var POOL: OnceLock<ThreadPool> = OnceLock.new()

internal fun candlePool(): ThreadPool =
    POOL.getOrInit({
        ThreadPoolBuilder
            .new()
            .numThreads(getNumThreads())
            .startHandler({ setThreadAffinity() })
            .build()
            .expect("failed to build candle rayon threadpool")
    })

/**
 * Elevate the thread QoS so macOS prefers running on Performance (P) cores.
 */
public fun setThreadAffinity() {
    run {
        pthreadSetQosClassSelfNp(QOS_CLASS_USER_INTERACTIVE, 0)
    }
}

public fun hasAccelerate(): Boolean = false

public fun hasMkl(): Boolean = false

public fun cudaIsAvailable(): Boolean = false

public fun metalIsAvailable(): Boolean = false

public fun withAvx(): Boolean = get().avx2

public fun withNeon(): Boolean = get().neon

public fun withSimd128(): Boolean = false

public fun withF16c(): Boolean = get().f16c
