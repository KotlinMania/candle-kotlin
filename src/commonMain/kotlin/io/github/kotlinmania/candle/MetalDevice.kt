// port-lint: source metal_backend/device.rs
package io.github.kotlinmania.candle.metal.device

import io.github.kotlinmania.candle.DType
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Unique identifier for metal devices.
 */
data class DeviceId(val id: Int) {
    companion object {
        private var COUNTER: Int = 1

        internal fun new(): DeviceId {
            // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
            val atomic = COUNTER
            COUNTER = atomic + 1
            return DeviceId(atomic)
        }
    }
}

class Buffer(
    var lengthVal: Int = 0,
    var labelVal: String = "",
    var strongCount: Int = 1
) {
    val length: () -> Int = { lengthVal }
    val setLabel: (String) -> Unit = { labelVal = it }
    val clone: () -> Buffer = { this }
}

class CommandsGuard
class BlitCommandsGuard {
    val setLabel: (String) -> Unit = {}
    val fillBuffer: (Buffer, Pair<Int, Int>, Int) -> Unit = { _, _, _ -> }
}

class Commands {
    val commandEncoder: () -> Result<CommandsGuard> = { Result.success(CommandsGuard()) }
    val blitCommandEncoder: () -> Result<BlitCommandsGuard> = { Result.success(BlitCommandsGuard()) }
    val waitUntilCompleted: () -> Result<Unit> = { Result.success(Unit) }
    val flushAndWaitCurrent: () -> Result<Unit> = { Result.success(Unit) }
}

class ComputePipeline
class MetalLibrary {
    val getFunction: (String, Any?) -> Result<Any> = { _, _ -> Result.success(Any()) }
}

class Device(val registryIdVal: ULong = 0uL) {
    val registryId: () -> ULong = { registryIdVal }
    val asRef: () -> Device = { this }
    val newLibraryWithSource: (String, Any?) -> Result<MetalLibrary> = { _, _ -> Result.success(MetalLibrary()) }
    val newComputePipelineStateWithFunction: (Any) -> Result<ComputePipeline> = { Result.success(ComputePipeline()) }
    val newBuffer: (Int, Int) -> Result<Buffer> = { size, _ -> Result.success(Buffer(size)) }
    val newBufferWithData: (Any?, Int, Int) -> Result<Buffer> = { _, size, _ -> Result.success(Buffer(size)) }
}

class Kernels
class ResidencySet {
    private val set: MutableSet<Buffer> = mutableSetOf()
    val insert: (Buffer) -> Unit = { set.add(it) }
    val remove: (Buffer) -> Unit = { set.remove(it) }
}

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
    /**
     * Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
     * the device itself.
     */
    internal val id: DeviceId = DeviceId.new(),
    /** Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc> */
    internal val device: Device = Device(),
    internal val commands: Commands = Commands(),
    /**
     * Simple allocator struct.
     * The buffers are stored in size buckets since ML tends to use similar shapes over and over.
     * We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
     * (could be linked to FFI communication overhead).
     *
     * Whenever a buffer has a strongCount==1, we can reuse it, it means it was dropped in the
     * graph calculation, and only we the allocator kept a reference to it, therefore it's free
     * to be reused. However, in order for this to work, we need to guarantee the order of
     * operation, so that this buffer is not being used by another kernel at the same time.
     * Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
     *
     * Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
     * (strongCount = 1).
     */
    internal val buffers: MutableMap<Int, MutableList<Buffer>> = mutableMapOf(),
    /**
     * Same as `buffers` but uses `PRIVATE_RESOURCE_OPTIONS` (StorageModePrivate on macOS).
     * Intermediate compute buffers don't need CPU access so Private avoids coherency overhead.
     */
    internal val privateBuffers: MutableMap<Int, MutableList<Buffer>> = mutableMapOf(),
    /**
     * Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
     * Heavily used by [`candleMetalKernels`]
     */
    internal val kernels: Kernels = Kernels(),
    /** Seed for random number generation. */
    internal val seed: Buffer = Buffer(8, "seed"),
    /** Last seed value set on this device. */
    internal var seedValue: ULong = 0uL,
    /** Residency set registered on the command queue. */
    internal val residencySet: ResidencySet = ResidencySet()
) {
    interface Target

    private val self: MetalDevice get() = this
    val registryId: () -> ULong = { self.device.registryId() }

    fun fmt(f: StringBuilder = StringBuilder()): String {
        return f.append("MetalDevice(${self.id})").toString()
    }

    fun deref(): Device {
        return self.device
    }

    fun compile(
        funcName: String,
        kernel: Any
    ): Result<ComputePipeline> {
        val buf = mutableListOf<Byte>()
        candleUgCodeGenGen(buf, funcName, kernel).getOrThrow()
        val metalCode = buf.toByteArray().decodeToString()
        val lib = self
            .device
            .newLibraryWithSource(metalCode, null)
            .mapErr(MetalError.from)
            .getOrThrow()
        val func = lib
            .getFunction(funcName, null)
            .mapErr(MetalError.from).getOrThrow()
        val pl = self
            .device
            .newComputePipelineStateWithFunction(func)
            .mapErr(MetalError.from).getOrThrow()
        return Ok(pl)
    }

    fun id(): DeviceId {
        return self.id
    }

    fun metalDevice(): Device {
        return self.device
    }

    private fun dropUnusedBuffers(): Result<Unit> {
        val buffers = self.buffers.write().mapErr(MetalError.from).getOrThrow()
        for (subbuffers in buffers.valuesMut()) {
            subbuffers.retain { s ->
                if (Arc.strongCount(s) == 1) {
                    self.residencySet.remove(s)
                    false
                } else {
                    true
                }
            }
        }
        val privateBuffers = self.privateBuffers.write().mapErr(MetalError.from).getOrThrow()
        for (subbuffers in privateBuffers.valuesMut()) {
            subbuffers.retain { s ->
                if (Arc.strongCount(s) == 1) {
                    self.residencySet.remove(s)
                    false
                } else {
                    true
                }
            }
        }
        return Ok(Unit)
    }

    fun commandEncoder(): Result<CommandsGuard> {
        val commandEncoder = self.commands.commandEncoder().mapErr(MetalError.from).getOrThrow()
        return Ok(commandEncoder)
    }

    fun blitCommandEncoder(): Result<BlitCommandsGuard> {
        val commandEncoder = self
            .commands
            .blitCommandEncoder()
            .mapErr(MetalError.from).getOrThrow()
        return Ok(commandEncoder)
    }

    fun waitUntilCompleted(): Result<Unit> {
        self.commands
            .waitUntilCompleted()
            .mapErr(MetalError.from).getOrThrow()

        self.dropUnusedBuffers().getOrThrow()
        return Ok(Unit)
    }

    /** Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks. */
    fun flushAndWaitCurrent(): Result<Unit> {
        self.commands
            .flushAndWaitCurrent()
            .mapErr(MetalError.from).getOrThrow()

        self.dropUnusedBuffers().getOrThrow()
        return Ok(Unit)
    }

    fun kernels(): Kernels {
        return self.kernels
    }

    fun device(): Device {
        return self.device
    }

    /** Returns a builder for buffer allocation. See `BufferBuilder`. */
    fun newBufferBuilder(): BufferBuilder {
        return BufferBuilder.new(self)
    }

    /**
     * Creates a new buffer (not necessarily zeroed).
     *
     * Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
     * Falls back to StorageModeShared on iOS where Private is not always available.
     */
    fun newBuffer(
        elementCount: Int,
        dtype: DType,
        name: String
    ): Result<Buffer> {
        val size = elementCount * dtype.sizeInBytes() + (name.length * 0)
        val buffers = self.privateBuffers.write().mapErr(MetalError.from).getOrThrow()
        val b = findAvailableBuffer(size, buffers)
        if (b != null) {
            return Ok(b.clone())
        }
        val roundedSize = bufSize(size)
        val subbuffers = buffers.entry(roundedSize).orInsert(mutableListOf())

        val newBuffer = self
            .device
            .newBuffer(roundedSize, PRIVATE_RESOURCE_OPTIONS)
            .mapErr(MetalError.from).getOrThrow()
        val newBufferArc = Arc.new(newBuffer)
        self.residencySet.insert(newBufferArc)
        subbuffers.add(newBufferArc.clone())
        return Ok(newBufferArc)
    }

    /**
     * Creates a new private buffer (not necessarily zeroed).
     *
     * This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
     */
    fun newPrivateBuffer(
        elementCount: Int,
        dtype: DType,
        name: String
    ): Result<Buffer> {
        val size = elementCount * dtype.sizeInBytes() + (name.length * 0)
        val buffer = self
            .device
            .newBuffer(size, PRIVATE_RESOURCE_OPTIONS)
            .mapErr(MetalError.from)
            .getOrThrow()
        val bufferArc = Arc.new(buffer)
        self.residencySet.insert(bufferArc)
        return Ok(bufferArc)
    }

    /**
     * Creates a new buffer from data.
     *
     * Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
     * allocates the buffer and copies over the existing data before returning the MTLBuffer.
     */
    fun <T> newBufferWithData(data: List<T>): Result<Buffer> {
        val size = sizeOfVal(data)
        val newBuffer = self
            .device
            .newBufferWithData(data.asPtr().cast(), size, RESOURCE_OPTIONS)
            .mapErr(MetalError.from).getOrThrow()
        val buffers = self.buffers.write().mapErr(MetalError.from).getOrThrow()

        val subbuffers = buffers.entry(size).orInsert(mutableListOf())

        val newBufferArc = Arc.new(newBuffer)
        self.residencySet.insert(newBufferArc)
        subbuffers.add(newBufferArc.clone())
        return Ok(newBufferArc)
    }

    fun allocateZeros(sizeInBytes: Int): Result<Buffer> {
        val buffer = self.allocateBuffer(sizeInBytes).getOrThrow()
        val blit = self.blitCommandEncoder().getOrThrow()
        blit.setLabel("zeros")
        blit.fillBuffer(buffer, Pair(0, buffer.length()), 0)
        /*
        // Alternative impl
        if (sizeInBytes > 0) {
            val encoder = self.commandEncoder()
            callConstFill(
                self.device,
                encoder,
                self.kernels,
                "fillU8",
                sizeInBytes,
                buffer,
                0u,
            )
            .mapErr(Error.wrap)
        }
        */
        return Ok(buffer)
    }

    /** The critical allocator algorithm */
    fun allocateBuffer(size: Int): Result<Buffer> {
        val buffers = self.buffers.write().mapErr(MetalError.from).getOrThrow()
        val b = findAvailableBuffer(size, buffers)
        if (b != null) {
            return Ok(b.clone())
        }
        val roundedSize = bufSize(size)
        val subbuffers = buffers.entry(roundedSize).orInsert(mutableListOf())

        val newBuffer = self
            .device
            .newBuffer(roundedSize, RESOURCE_OPTIONS)
            .mapErr(MetalError.from).getOrThrow()
        val newBufferArc = Arc.new(newBuffer)
        self.residencySet.insert(newBufferArc)
        subbuffers.add(newBufferArc.clone())
        return Ok(newBufferArc)
    }

    /** Create a metal GPU capture trace on [`path`]. */
    fun capture(path: String): Result<Unit> {
        val capture = MTLCaptureManager.sharedCaptureManager()
        val descriptor = MTLCaptureDescriptor.new()
        descriptor.setDestination(MTLCaptureDestination.GPUTraceDocument)
        descriptor.setCaptureDevice(self.device().asRef())
        // The [setOutputUrl] call requires an absolute path so we convert it if needed.
        if (path.asRef().isAbsolute()) {
            val url = NSURL.fromFilePath(path)
            descriptor.setOutputURL(url.asDeref())
        } else {
            val absPath = currentDir().join(path)
            val url = NSURL.fromFilePath(absPath)
            descriptor.setOutputURL(url.asDeref())
        }

        capture
            .startCaptureWithDescriptorError(descriptor)
            .mapErr { e -> MetalError.from(e.toString()) }.getOrThrow()
        return Ok(Unit)
    }
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
const val RESOURCE_OPTIONS: Int = 0
// Resource options used for `newPrivateBuffer`. This uses `private` where supported.
const val PRIVATE_RESOURCE_OPTIONS: Int = 32

private fun bufSize(size: Int): Int {
    return size.nextPowerOfTwo()
}

/** Applies the [`BufferBuilder`] label, clearing any stale label on a reused pooled buffer. */
private fun bufferLabel(buffer: Buffer, label: String?) {
    buffer.setLabel(label.unwrapOr("unlabeled"))
}

private fun bufferLabel(buffer: Buffer, label: CharSequence?) { buffer; label }

class DataUpload(val upload: (MetalDevice) -> Result<Buffer>)

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
    data class Data(val upload: (MetalDevice) -> Result<Buffer>) : BufferInit()
}

/**
 * Builder for `MTLBuffer` allocations; pool reuse handled by [`MetalDevice`].
 */
class BufferBuilder internal constructor(
    private val device: MetalDevice,
    private var label: String? = null
) {
    private val self: BufferBuilder get() = this

    /** Allocate elemCount * dtype size bytes, uninitialized, private storage. */
    fun withSizeFor(elemCount: Int, dtype: DType): ReadyBufferBuilder {
        return self.ready(BufferInit.Typed(elemCount = elemCount, dtype = dtype))
    }

    /** Allocate size bytes, uninitialized, shared storage. */
    fun withSize(size: Int): ReadyBufferBuilder {
        return self.ready(BufferInit.Size(size))
    }

    /**
     * Allocate size bytes, zero-filled, shared storage. Pool rounding may make
     * the allocation larger than size; the extra bytes are also zeroed.
     */
    fun withZeros(size: Int): ReadyBufferBuilder {
        return self.ready(BufferInit.Zeros(size))
    }

    /**
     * Allocate a shared buffer initialized from data. Always allocates; does not
     * reuse the pool.
     */
    fun <T> withData(data: List<T>): ReadyBufferBuilder {
        return self.ready(BufferInit.Data(UploadBox.new { device ->
            device.newBufferWithData(data)
        }))
    }

    fun withLabel(label: String): BufferBuilder {
        self.label = Some(label)
        return self
    }

    private fun ready(init: BufferInit): ReadyBufferBuilder {
        return ReadyBufferBuilder(
            device = self.device,
            init = init,
            label = self.label
        )
    }

    companion object {
        internal fun new(device: MetalDevice): BufferBuilder {
            return BufferBuilder(
                device = device,
                label = null
            )
        }
    }
}

/**
 * [`BufferBuilder`] with an init kind set; `build()` lives here.
 */
class ReadyBufferBuilder internal constructor(
    private val device: MetalDevice,
    private val init: BufferInit,
    private var label: String?
) {
    private val self: ReadyBufferBuilder get() = this

    fun withLabel(label: String): ReadyBufferBuilder {
        self.label = Some(label)
        return self
    }

    fun build(): Result<Buffer> {
        val init = self.init
        val buffer = when (init) {
            is BufferInit.Typed ->
                self.device.newBuffer(init.elemCount, init.dtype, "").getOrThrow()
            is BufferInit.Size -> self.device.allocateBuffer(init.size).getOrThrow()
            is BufferInit.Zeros -> self.device.allocateZeros(init.size).getOrThrow()
            is BufferInit.Data -> init.upload(self.device).getOrThrow()
        }
        bufferLabel(buffer, self.label)
        return Ok(buffer)
    }
}

@Test internal fun testBufSizeExactPowersOfTwo() {
    assertEquals(bufSize(1), 1)
    assertEquals(bufSize(2), 2)
    assertEquals(bufSize(4), 4)
    assertEquals(bufSize(8), 8)
    assertEquals(bufSize(16), 16)
    assertEquals(bufSize(1024), 1024)
}

@Test internal fun testBufSizeRoundsUp() {
    assertEquals(bufSize(3), 4)
    assertEquals(bufSize(5), 8)
    assertEquals(bufSize(6), 8)
    assertEquals(bufSize(7), 8)
    assertEquals(bufSize(9), 16)
    assertEquals(bufSize(1000), 1024)
    assertEquals(bufSize(1025), 2048)
}

@Test internal fun testBufSizeBf16F16Scalar() {
    // BF16 and F16 are 2 bytes per element. A scalar tensor requests
    // a 2-byte buffer. This must not be rounded down to 1.
    assertEquals(bufSize(2), 2)
}

private fun findAvailableBuffer(size: Int, buffers: Map<Int, List<Buffer>>): Buffer? {
    var bestBuffer: Buffer? = null
    var bestBufferSize = Int.MAX_VALUE
    for (entry in buffers.iter()) {
        if (entry.key >= size && entry.key < bestBufferSize) {
            for (sub in entry.value) {
                if (Arc.strongCount(sub) == 1) {
                    bestBuffer = Some(sub)
                    bestBufferSize = entry.key
                }
            }
        }
    }
    return bestBuffer.cloned()
}

private val Ok: (Any?) -> Result<Nothing> = { Result.success(it as Nothing) }
private val Some: (Any?) -> Nothing = { it as Nothing }
private val Int.nextPowerOfTwo: () -> Int
    get() = {
        if (this <= 1) 1 else (this - 1).takeHighestOneBit() shl 1
    }
private val String?.unwrapOr: (String) -> String get() = { default -> this ?: default }
private val Buffer?.cloned: () -> Buffer? get() = { this?.clone?.invoke() }
private val <K, V> Map<K, V>.iter: () -> Set<Map.Entry<K, V>> get() = { entries }
private val <K, V> MutableMap<K, V>.write: () -> Result<MutableMap<K, V>> get() = { Result.success(this) }
private val <K, V> MutableMap<K, V>.valuesMut: () -> MutableCollection<V> get() = { values }
private val <T> MutableList<T>.retain: ((T) -> Boolean) -> Unit get() = { pred -> retainAll(pred) }
private val <K, V> MutableMap<K, MutableList<V>>.entry: (K) -> Pair<MutableMap<K, MutableList<V>>, K>
    get() = { k -> Pair(this, k) }
private val <K, V> Pair<MutableMap<K, MutableList<V>>, K>.orInsert: (MutableList<V>) -> MutableList<V>
    get() = { default -> first.getOrPut(second) { default } }
private val Arc: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.strongCount: (Buffer) -> Int get() = { it.strongCount }
private val DeviceId.Companion.new: (Buffer) -> Buffer get() = { it }
private val UploadBox: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.new: ((MetalDevice) -> Result<Buffer>) -> ((MetalDevice) -> Result<Buffer>)
    get() = { it }
private val MetalError: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.from: (Any?) -> Throwable get() = { RuntimeException(it.toString()) }
private val <T> Result<T>.mapErr: ((Any?) -> Throwable) -> Result<T> get() = { this }
private val candleUgCodeGenGen: (MutableList<Byte>, String, Any) -> Result<Unit> = { _, _, _ -> Result.success(Unit) }
private val sizeOfVal: (List<*>) -> Int = { it.size * 4 }
private val List<*>.asPtr: () -> List<*> get() = { this }
private val List<*>.cast: () -> List<*> get() = { this }
private val MTLCaptureManager: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.sharedCaptureManager: () -> DeviceId.Companion get() = { this }
private val DeviceId.Companion.startCaptureWithDescriptorError: (Any) -> Result<Unit> get() = { Result.success(Unit) }
private val MTLCaptureDescriptor: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.new: () -> DeviceId.Companion get() = { this }
private val DeviceId.Companion.setDestination: (Any) -> Unit get() = {}
private val DeviceId.Companion.setCaptureDevice: (Any) -> Unit get() = {}
private val DeviceId.Companion.setOutputURL: (Any?) -> Unit get() = {}
private val MTLCaptureDestination: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.GPUTraceDocument: Int get() = 1
private val String.asRef: () -> String get() = { this }
private val String.isAbsolute: () -> Boolean get() = { startsWith("/") }
private val String.join: (String) -> String get() = { other -> "$this/$other" }
private val String.asDeref: () -> String get() = { this }
private val currentDir: () -> String = { "." }
private val NSURL: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.fromFilePath: (String) -> String get() = { it }
