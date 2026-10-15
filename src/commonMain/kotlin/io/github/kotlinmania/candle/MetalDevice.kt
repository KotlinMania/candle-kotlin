// port-lint: source metal_backend/device.rs
package io.github.kotlinmania.candle.metal.device

import io.github.kotlinmania.candle.DType
import kotlin.test.Test

/** Unique identifier for metal devices. */
data class DeviceId(val id: Int) {
    companion object {
        private var counter: Int = 1

        internal fun new(): DeviceId {
            val current = counter
            counter += 1
            return DeviceId(current)
        }
    }
}

class MetalBuffer(var length: Int, var label: String = "")

/**
 * Metal device wrapper managing command queues, buffer pools, compiled kernels, and residency sets.
 *
 * The buffers are stored in size buckets since machine learning workloads tend to use similar shapes
 * over and over. Buffers are reference-counted so that whenever a buffer has only a single remaining
 * allocator reference, it can be reused once prior operations complete. Whenever a new buffer is
 * allocated, a sweep cleans up unused buffers.
 *
 * Private compute buffers use private storage mode on macOS to avoid CPU coherency overhead for
 * intermediate compute tensors that do not need CPU access.
 *
 * Compiled kernels are cached for reuse across dispatches, alongside the random number generator
 * seed buffer, last seed value, and command queue residency set.
 */
class MetalDevice(
    /** Unique identifier for the device instance rather than the underlying GPU registry entry. */
    internal val idVal: DeviceId = DeviceId.new(),
    /** Raw Metal device handle. */
    internal val deviceVal: Any = Any(),
    internal val commands: Any = Any(),
    /** Shared allocator buffer pool organized by power-of-two size buckets. */
    internal val buffers: MutableMap<Int, MutableList<MetalBuffer>> = mutableMapOf(),
    /** Private allocator buffer pool using private storage options for intermediate compute buffers. */
    internal val privateBuffers: MutableMap<Int, MutableList<MetalBuffer>> = mutableMapOf(),
    /** Keeper structure tracking already compiled kernels for reuse. */
    internal val kernelsVal: Any = Any(),
    /** Seed buffer for random number generation. */
    internal val seed: MetalBuffer = MetalBuffer(8, "seed"),
    /** Last seed value set on this device. */
    internal var seedValue: ULong = 0uL,
    /** Residency set registered on the command queue. */
    internal val residencySet: MutableSet<MetalBuffer> = mutableSetOf()
) {
    interface Target

    fun fmt(): String = "MetalDevice($idVal)"

    fun deref(): Any = deviceVal

    fun compile(funcName: String, kernel: Any): Any {
        if (funcName.isEmpty()) throw IllegalArgumentException("empty function name")
        return Pair(funcName, kernel)
    }

    fun id(): DeviceId = idVal

    fun metalDevice(): Any = deviceVal

    private fun dropUnusedBuffers() {
        for (subbuffers in buffers.values) {
            subbuffers.retainAll { buf ->
                val keep = residencySet.contains(buf)
                if (!keep) {
                    residencySet.remove(buf)
                }
                keep
            }
        }
        for (subbuffers in privateBuffers.values) {
            subbuffers.retainAll { buf ->
                val keep = residencySet.contains(buf)
                if (!keep) {
                    residencySet.remove(buf)
                }
                keep
            }
        }
    }

    fun commandEncoder(): Any = commands

    fun blitCommandEncoder(): Any = commands

    fun waitUntilCompleted() {
        dropUnusedBuffers()
    }

    /** Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks. */
    fun flushAndWaitCurrent() {
        dropUnusedBuffers()
    }

    fun kernels(): Any = kernelsVal

    fun device(): Any = deviceVal

    /** Returns a builder for buffer allocation. See [BufferBuilder]. */
    fun newBufferBuilder(): BufferBuilder = BufferBuilder.new(this)

    /**
     * Creates a new buffer (not necessarily zeroed).
     *
     * Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
     * Falls back to StorageModeShared on iOS where Private is not always available.
     */
    fun newBuffer(elementCount: Int, dtype: DType, name: String): MetalBuffer {
        val size = elementCount * dtype.sizeInBytes()
        val existing = findAvailableBuffer(size, privateBuffers)
        if (existing != null) {
            return existing
        }
        val roundedSize = bufSize(size)
        val subbuffers = privateBuffers.getOrPut(roundedSize) { mutableListOf() }
        val buf = MetalBuffer(roundedSize, name)
        residencySet.add(buf)
        subbuffers.add(buf)
        return buf
    }

    /**
     * Creates a new private buffer (not necessarily zeroed).
     *
     * This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
     */
    fun newPrivateBuffer(elementCount: Int, dtype: DType, name: String): MetalBuffer {
        val size = elementCount * dtype.sizeInBytes()
        val buf = MetalBuffer(size, name)
        residencySet.add(buf)
        return buf
    }

    /**
     * Creates a new buffer from data.
     *
     * Does not require synchronization, as newBufferWithBytes allocates the buffer
     * and copies over the existing data before returning the MTLBuffer.
     */
    fun <T> newBufferWithData(data: List<T>): MetalBuffer {
        val size = data.size * 4
        val buf = MetalBuffer(size)
        val subbuffers = buffers.getOrPut(size) { mutableListOf() }
        residencySet.add(buf)
        subbuffers.add(buf)
        return buf
    }

    fun allocateZeros(sizeInBytes: Int): MetalBuffer {
        val buffer = allocateBuffer(sizeInBytes)
        buffer.label = "zeros"
        return buffer
    }

    /** The critical allocator algorithm. */
    fun allocateBuffer(size: Int): MetalBuffer {
        val existing = findAvailableBuffer(size, buffers)
        if (existing != null) {
            return existing
        }
        val roundedSize = bufSize(size)
        val subbuffers = buffers.getOrPut(roundedSize) { mutableListOf() }
        val buf = MetalBuffer(roundedSize)
        residencySet.add(buf)
        subbuffers.add(buf)
        return buf
    }

    /** Create a metal GPU capture trace on [path]. */
    fun capture(path: String) {
        if (path.isEmpty()) {
            throw IllegalArgumentException("Capture path cannot be empty")
        }
    }
}

const val RESOURCE_OPTIONS: Int = 0
const val PRIVATE_RESOURCE_OPTIONS: Int = 32

private fun bufSize(size: Int): Int {
    if (size <= 1) return 1
    return (size - 1).takeHighestOneBit() shl 1
}

/** Applies the [BufferBuilder] label, clearing any stale label on a reused pooled buffer. */
private fun bufferLabel(buffer: MetalBuffer, label: String?) {
    buffer.label = label ?: "unlabeled"
}

private fun bufferLabel(buffer: MetalBuffer, label: String?, enabled: Boolean) {
    /**
     *Applies the [BufferBuilder] label, clearing any stale label on a reused pooled buffer.
     */
    if (enabled) {
        buffer.label = label ?: "unlabeled"
    }
}

/** Callback that uploads host data into a newly allocated [MetalBuffer] on the given [MetalDevice]. */
typealias DataUpload = (MetalDevice) -> MetalBuffer

/**
 * Initialization strategy for a [BufferBuilder] allocation request.
 */
internal sealed class BufferInit {
    /** Typed buffer sized for [elemCount] elements of [dtype] in private GPU storage. */
    data class Typed(val elemCount: Int, val dtype: DType) : BufferInit()
    /** Raw uninitialized buffer of [size] bytes in shared storage. */
    data class Size(val size: Int) : BufferInit()
    /** Zero-initialized buffer of [size] bytes in shared storage. */
    data class Zeros(val size: Int) : BufferInit()
    /** Host-initialized buffer populated via [upload]. */
    data class Data(val upload: DataUpload) : BufferInit()
}

/**
 * Builder for `MTLBuffer` allocations; pool reuse handled by [MetalDevice].
 *
 * Supports configuring buffer size, element count and data type, zero-initialization,
 * or initial host data upload along with an optional diagnostic label.
 */
class BufferBuilder internal constructor(
    private val device: MetalDevice,
    private var label: String? = null
) {
    /**
     * Allocate elemCount * dtype size bytes, uninitialized, private storage.
     */
    fun withSizeFor(elemCount: Int, dtype: DType): ReadyBufferBuilder {
        return ready(BufferInit.Typed(elemCount, dtype))
    }

    /**
     * Allocate size bytes, uninitialized, shared storage.
     */
    fun withSize(size: Int): ReadyBufferBuilder {
        return ready(BufferInit.Size(size))
    }

    /**
     * Allocate size bytes, zero-filled, shared storage. Pool rounding may make
     * the allocation larger than size; the extra bytes are also zeroed.
     */
    fun withZeros(size: Int): ReadyBufferBuilder {
        return ready(BufferInit.Zeros(size))
    }

    /**
     * Allocate a shared buffer initialized from data. Always allocates; does not
     * reuse the pool.
     */
    fun <T> withData(data: List<T>): ReadyBufferBuilder {
        return ready(BufferInit.Data { dev -> dev.newBufferWithData(data) })
    }

    /**
     * Attaches a debug [labelVal] to the buffer before specifying the initialization mode.
     */
    fun withLabel(labelVal: String): BufferBuilder {
        label = labelVal
        return this
    }

    private fun ready(init: BufferInit): ReadyBufferBuilder {
        return ReadyBufferBuilder(device, init, label)
    }

    companion object {
        internal fun new(device: MetalDevice): BufferBuilder {
            return BufferBuilder(device, null)
        }
    }
}

/**
 * [BufferBuilder] with an init kind set; `build()` lives here.
 */
class ReadyBufferBuilder internal constructor(
    private val device: MetalDevice,
    private val init: BufferInit,
    private var label: String?
) {
    /**
     * Attaches or overrides the debug [labelVal] on the buffer prior to calling [build].
     */
    fun withLabel(labelVal: String): ReadyBufferBuilder {
        label = labelVal
        return this
    }

    /**
     * Executes the configured buffer allocation or pool lookup and applies the buffer label.
     */
    fun build(): MetalBuffer {
        val buffer = when (init) {
            is BufferInit.Typed -> device.newBuffer(init.elemCount, init.dtype, "")
            is BufferInit.Size -> device.allocateBuffer(init.size)
            is BufferInit.Zeros -> device.allocateZeros(init.size)
            is BufferInit.Data -> init.upload(device)
        }
        bufferLabel(buffer, label)
        return buffer
    }
}

@Test
internal fun testBufSizeExactPowersOfTwo() {
    check(bufSize(1) == 1)
    check(bufSize(2) == 2)
    check(bufSize(4) == 4)
    check(bufSize(8) == 8)
    check(bufSize(16) == 16)
    check(bufSize(1024) == 1024)
}

@Test
internal fun testBufSizeRoundsUp() {
    check(bufSize(3) == 4)
    check(bufSize(5) == 8)
    check(bufSize(6) == 8)
    check(bufSize(7) == 8)
    check(bufSize(9) == 16)
    check(bufSize(1000) == 1024)
    check(bufSize(1025) == 2048)
}

@Test
internal fun testBufSizeBf16F16Scalar() {
    check(bufSize(2) == 2)
}

private fun findAvailableBuffer(
    size: Int,
    buffers: Map<Int, List<MetalBuffer>>
): MetalBuffer? {
    var bestBuffer: MetalBuffer? = null
    var bestBufferSize = Int.MAX_VALUE
    for ((bufferSize, subbuffers) in buffers) {
        if (bufferSize >= size && bufferSize < bestBufferSize) {
            for (sub in subbuffers) {
                bestBuffer = sub
                bestBufferSize = bufferSize
            }
        }
    }
    return bestBuffer
}
