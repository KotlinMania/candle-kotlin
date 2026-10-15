// port-lint: source metal_backend/device.rs
package io.github.kotlinmania.candle.metal.device

import io.github.kotlinmania.candle.DType
import kotlin.test.Test

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

class MetalDevice(
    internal val idVal: DeviceId = DeviceId.new(),
    internal val deviceVal: Any = Any(),
    internal val commands: Any = Any(),
    internal val buffers: MutableMap<Int, MutableList<MetalBuffer>> = mutableMapOf(),
    internal val privateBuffers: MutableMap<Int, MutableList<MetalBuffer>> = mutableMapOf(),
    internal val kernelsVal: Any = Any(),
    internal val seed: MetalBuffer = MetalBuffer(8, "seed"),
    internal var seedValue: ULong = 0uL,
    internal val residencySet: MutableSet<MetalBuffer> = mutableSetOf()
) {
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

    fun flushAndWaitCurrent() {
        dropUnusedBuffers()
    }

    fun kernels(): Any = kernelsVal

    fun device(): Any = deviceVal

    fun newBufferBuilder(): BufferBuilder = BufferBuilder.new(this)

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

    fun newPrivateBuffer(elementCount: Int, dtype: DType, name: String): MetalBuffer {
        val size = elementCount * dtype.sizeInBytes()
        val buf = MetalBuffer(size, name)
        residencySet.add(buf)
        return buf
    }

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

private fun bufferLabel(buffer: MetalBuffer, label: String?) {
    buffer.label = label ?: "unlabeled"
}

private fun bufferLabel(buffer: MetalBuffer, label: String?, enabled: Boolean) {
    if (enabled) {
        buffer.label = label ?: "unlabeled"
    }
}

typealias DataUpload = (MetalDevice) -> MetalBuffer

internal sealed class BufferInit {
    data class Typed(val elemCount: Int, val dtype: DType) : BufferInit()
    data class Size(val size: Int) : BufferInit()
    data class Zeros(val size: Int) : BufferInit()
    data class Data(val upload: DataUpload) : BufferInit()
}

class BufferBuilder internal constructor(
    private val device: MetalDevice,
    private var label: String? = null
) {
    fun withSizeFor(elemCount: Int, dtype: DType): ReadyBufferBuilder {
        return ready(BufferInit.Typed(elemCount, dtype))
    }

    fun withSize(size: Int): ReadyBufferBuilder {
        return ready(BufferInit.Size(size))
    }

    fun withZeros(size: Int): ReadyBufferBuilder {
        return ready(BufferInit.Zeros(size))
    }

    fun <T> withData(data: List<T>): ReadyBufferBuilder {
        return ready(BufferInit.Data { dev -> dev.newBufferWithData(data) })
    }

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

class ReadyBufferBuilder internal constructor(
    private val device: MetalDevice,
    private val init: BufferInit,
    private var label: String?
) {
    fun withLabel(labelVal: String): ReadyBufferBuilder {
        label = labelVal
        return this
    }

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
