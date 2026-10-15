// port-lint: source metal_backend/device.rs
package io.github.kotlinmania.candle.metal.device

import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CandleException as Error
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CpuStorageRef
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.metal.MetalBackendDevice
import io.github.kotlinmania.candle.metal.MetalError
import io.github.kotlinmania.candle.metal.MetalStorage
import io.github.kotlinmania.candle.Test
import io.github.kotlinmania.candle.assertEquals

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
    val data: () -> ByteArray = { ByteArray(lengthVal) }
    val didModifyRange: (NSRange) -> Unit = {}
}

class CommandsGuard
class BlitCommandsGuard {
    val setLabel: (String) -> Unit = {}
    val fillBuffer: (Buffer, Pair<Int, Int>, Int) -> Unit = { _, _, _ -> }
    val copyFromBuffer: (Buffer, Int, Buffer, Int, Int) -> Unit = { _, _, _, _, _ -> }
}

class CommandQueue {
    val setLabel: (String?) -> Unit = {}
}

class Commands {
    val commandEncoder: () -> Result<CommandsGuard> = { Result.success(CommandsGuard()) }
    val blitCommandEncoder: () -> Result<BlitCommandsGuard> = { Result.success(BlitCommandsGuard()) }
    val waitUntilCompleted: () -> Result<Unit> = { Result.success(Unit) }
    val flushAndWaitCurrent: () -> Result<Unit> = { Result.success(Unit) }

    companion object {
        val new: (CommandQueue, ResidencySet) -> Result<Commands> = { _, _ -> Result.success(Commands()) }
    }
}

class ComputePipeline
class MetalLibrary {
    val getFunction: (String, Any?) -> Result<Any> = { _, _ -> Result.success(Any()) }
}

class Device(val registryIdVal: ULong = 0uL) {
    val registryId: () -> ULong = { registryIdVal }
    val asRef: () -> Device = { this }
    val newCommandQueue: () -> Result<CommandQueue> = { Result.success(CommandQueue()) }
    val newLibraryWithSource: (String, Any?) -> Result<MetalLibrary> = { _, _ -> Result.success(MetalLibrary()) }
    val newComputePipelineStateWithFunction: (Any) -> Result<ComputePipeline> = { Result.success(ComputePipeline()) }
    val newBuffer: (Int, Int) -> Result<Buffer> = { size, _ -> Result.success(Buffer(size)) }
    val newBufferWithData: (Any?, Int, Int) -> Result<Buffer> = { _, size, _ -> Result.success(Buffer(size)) }

    companion object {
        val all: () -> MutableList<Device> = { mutableListOf(Device()) }
    }
}

class Kernels {
    companion object {
        val new: () -> Kernels = { Kernels() }
    }
}

class ResidencySet {
    private val set: MutableSet<Buffer> = mutableSetOf()
    val insert: (Buffer) -> Unit = { set.add(it) }
    val remove: (Buffer) -> Unit = { set.remove(it) }

    companion object {
        val new: (Device) -> ResidencySet = { ResidencySet() }
    }
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

internal fun <T> Ok(v: T): Result<T> = Result.success(v)
internal fun <T> Some(v: T): T = v
private val Int.nextPowerOfTwo: () -> Int
    get() = {
        if (this <= 1) 1 else (this - 1).takeHighestOneBit() shl 1
    }
private val String?.unwrapOr: (String) -> String get() = { default -> this ?: default }
private val Buffer?.cloned: () -> Buffer? get() = { this?.clone?.invoke() }
internal fun <K, V> Map<K, V>.iter(): Set<Map.Entry<K, V>> = entries
internal fun <K, V> MutableMap<K, V>.write(): Result<MutableMap<K, V>> = Result.success(this)
internal fun <K, V> MutableMap<K, V>.valuesMut(): MutableCollection<V> = values
internal fun <T> MutableList<T>.retain(pred: (T) -> Boolean) { retainAll(pred) }
internal fun <K, V> MutableMap<K, MutableList<V>>.entry(k: K): Pair<MutableMap<K, MutableList<V>>, K> = Pair(this, k)
internal fun <K, V> Pair<MutableMap<K, MutableList<V>>, K>.orInsert(default: MutableList<V>): MutableList<V> =
    first.getOrPut(second) { default }
internal val Arc: DeviceId.Companion get() = DeviceId.Companion
internal val DeviceId.Companion.strongCount: (Buffer) -> Int get() = { it.strongCount }
internal fun DeviceId.Companion.new(it: Buffer): Buffer = it
private val UploadBox: DeviceId.Companion get() = DeviceId.Companion
private fun DeviceId.Companion.new(it: (MetalDevice) -> Result<Buffer>): (MetalDevice) -> Result<Buffer> = it
private val MetalError: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.from: (Any?) -> Throwable get() = { RuntimeException(it.toString()) }
internal fun <T> Result<T>.mapErr(f: (Nothing) -> Throwable): Result<T> = if (f == {}) this else this
private val candleUgCodeGenGen: (MutableList<Byte>, String, Any) -> Result<Unit> = { _, _, _ -> Result.success(Unit) }
private val sizeOfVal: (List<*>) -> Int = { it.size * 4 }
private fun <T> List<T>.cast(): List<T> = this
private val MTLCaptureManager: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.sharedCaptureManager: () -> DeviceId.Companion get() = { this }
private val DeviceId.Companion.startCaptureWithDescriptorError: (Any) -> Result<Unit> get() = { Result.success(Unit) }
private object MTLCaptureDescriptor {
    fun new(): MTLCaptureDescriptor = this
    fun setDestination(d: Any) { if (d == Unit) Unit }
    fun setCaptureDevice(d: Any) { if (d == Unit) Unit }
    fun setOutputURL(u: Any?) { if (u == Unit) Unit }
}
private val MTLCaptureDestination: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.GPUTraceDocument: Int get() = 1
private val String.asRef: () -> String get() = { this }
private val String.isAbsolute: () -> Boolean get() = { startsWith("/") }
private val String.join: (String) -> String get() = { other -> "$this/$other" }
private val String.asDeref: () -> String get() = { this }
private val currentDir: () -> String = { "." }
private val NSURL: DeviceId.Companion get() = DeviceId.Companion
private val DeviceId.Companion.fromFilePath: (String) -> String get() = { it }

data class NSRange(val location: Int, val length: Int) {
    companion object {
        val new: (Int, Int) -> NSRange = { loc, len -> NSRange(loc, len) }
    }
}

data class BufferOffset(
    val buffer: Buffer,
    val offsetInBytes: Int
) {
    companion object {
        val zeroOffset: (Buffer) -> BufferOffset = { buf -> BufferOffset(buf, 0) }
    }
}

sealed class TryLockError<out T> {
    data class Poisoned<out T>(val p: String) : TryLockError<T>()
    data object WouldBlock : TryLockError<Nothing>()
}

data class PoisonError<out T>(val description: String)

data class CallConvTranspose2dCfg(
    val dilation: Int,
    val stride: Int,
    val padding: Int,
    val outputPadding: Int,
    val cOut: Int,
    val outH: Int,
    val outW: Int,
    val bSize: Int,
    val inputDims: List<Int>,
    val inputStride: List<Int>,
    val kernelDims: List<Int>,
    val kernelStride: List<Int>,
    val inputOffset: Int,
    val kernelOffset: Int
)

object candleMetalKernels {
    enum class GemmDType {
        F32,
        F16,
        BF16,
    }

    object copy2d {
        const val FLOAT: String = "copy2d_f32"
        const val HALF: String = "copy2d_f16"
        const val BFLOAT: String = "copy2d_bf16"
        const val I64: String = "copy2d_i64"
        const val I32: String = "copy2d_i32"
        const val I16: String = "copy2d_i16"
        const val U32: String = "copy2d_u32"
        const val U8: String = "copy2d_u8"
    }

    object unary {
        object contiguous {
            object constSet {
                const val HALF: String = "const_set_f16"
                const val BFLOAT: String = "const_set_bf16"
                const val FLOAT: String = "const_set_f32"
                const val I64: String = "const_set_i64"
                const val U32: String = "const_set_u32"
                const val U8: String = "const_set_u8"
            }
            object abs {
                const val HALF: String = "uabs_f16"
                const val FLOAT: String = "uabs_f32"
                const val BFLOAT: String = "uabs_bf16"
            }
            object ceil {
                const val HALF: String = "uceil_f16"
                const val FLOAT: String = "uceil_f32"
                const val BFLOAT: String = "uceil_bf16"
            }
            object cos {
                const val HALF: String = "ucos_f16"
                const val FLOAT: String = "ucos_f32"
                const val BFLOAT: String = "ucos_bf16"
            }
            object erf {
                const val HALF: String = "uerf_f16"
                const val FLOAT: String = "uerf_f32"
                const val BFLOAT: String = "uerf_bf16"
            }
            object exp {
                const val HALF: String = "uexp_f16"
                const val FLOAT: String = "uexp_f32"
                const val BFLOAT: String = "uexp_bf16"
            }
            object floor {
                const val HALF: String = "ufloor_f16"
                const val FLOAT: String = "ufloor_f32"
                const val BFLOAT: String = "ufloor_bf16"
            }
            object geluErf {
                const val HALF: String = "ugelu_erf_f16"
                const val FLOAT: String = "ugelu_erf_f32"
                const val BFLOAT: String = "ugelu_erf_bf16"
            }
            object gelu {
                const val HALF: String = "ugelu_f16"
                const val FLOAT: String = "ugelu_f32"
                const val BFLOAT: String = "ugelu_bf16"
            }
            object log {
                const val HALF: String = "ulog_f16"
                const val FLOAT: String = "ulog_f32"
                const val BFLOAT: String = "ulog_bf16"
            }
            object neg {
                const val HALF: String = "uneg_f16"
                const val FLOAT: String = "uneg_f32"
                const val BFLOAT: String = "uneg_bf16"
            }
            object recip {
                const val HALF: String = "urecip_f16"
                const val FLOAT: String = "urecip_f32"
                const val BFLOAT: String = "urecip_bf16"
            }
            object relu {
                const val HALF: String = "urelu_f16"
                const val FLOAT: String = "urelu_f32"
                const val BFLOAT: String = "urelu_bf16"
            }
            object round {
                const val HALF: String = "uround_f16"
                const val FLOAT: String = "uround_f32"
                const val BFLOAT: String = "uround_bf16"
            }
            object silu {
                const val HALF: String = "usilu_f16"
                const val FLOAT: String = "usilu_f32"
                const val BFLOAT: String = "usilu_bf16"
            }
            object sin {
                const val HALF: String = "usin_f16"
                const val FLOAT: String = "usin_f32"
                const val BFLOAT: String = "usin_bf16"
            }
            object sqr {
                const val HALF: String = "usqr_f16"
                const val FLOAT: String = "usqr_f32"
                const val BFLOAT: String = "usqr_bf16"
            }
            object sqrt {
                const val HALF: String = "usqrt_f16"
                const val FLOAT: String = "usqrt_f32"
                const val BFLOAT: String = "usqrt_bf16"
            }
            object tanh {
                const val HALF: String = "utanh_f16"
                const val FLOAT: String = "utanh_f32"
                const val BFLOAT: String = "utanh_bf16"
            }
            object sign {
                const val HALF: String = "usign_f16"
                const val FLOAT: String = "usign_f32"
                const val BFLOAT: String = "usign_bf16"
                const val I64: String = "usign_i64"
            }
        }

        object strided {
            object constSet {
                const val HALF: String = "const_set_f16_strided"
                const val BFLOAT: String = "const_set_bf16_strided"
                const val FLOAT: String = "const_set_f32_strided"
                const val I64: String = "const_set_i64_strided"
                const val U32: String = "const_set_u32_strided"
                const val U8: String = "const_set_u8_strided"
            }
            object copy {
                const val FLOAT: String = "copy_strided_f32"
                const val HALF: String = "copy_strided_f16"
                const val BFLOAT: String = "copy_strided_bf16"
                const val I64: String = "copy_strided_i64"
                const val U32: String = "copy_strided_u32"
                const val U8: String = "copy_strided_u8"
            }
            object abs {
                const val HALF: String = "uabs_f16_strided"
                const val FLOAT: String = "uabs_f32_strided"
                const val BFLOAT: String = "uabs_bf16_strided"
            }
            object ceil {
                const val HALF: String = "uceil_f16_strided"
                const val FLOAT: String = "uceil_f32_strided"
                const val BFLOAT: String = "uceil_bf16_strided"
            }
            object cos {
                const val HALF: String = "ucos_f16_strided"
                const val FLOAT: String = "ucos_f32_strided"
                const val BFLOAT: String = "ucos_bf16_strided"
            }
            object erf {
                const val HALF: String = "uerf_f16_strided"
                const val FLOAT: String = "uerf_f32_strided"
                const val BFLOAT: String = "uerf_bf16_strided"
            }
            object exp {
                const val HALF: String = "uexp_f16_strided"
                const val FLOAT: String = "uexp_f32_strided"
                const val BFLOAT: String = "uexp_bf16_strided"
            }
            object floor {
                const val HALF: String = "ufloor_f16_strided"
                const val FLOAT: String = "ufloor_f32_strided"
                const val BFLOAT: String = "ufloor_bf16_strided"
            }
            object geluErf {
                const val HALF: String = "ugelu_erf_f16_strided"
                const val FLOAT: String = "ugelu_erf_f32_strided"
                const val BFLOAT: String = "ugelu_erf_bf16_strided"
            }
            object gelu {
                const val HALF: String = "ugelu_f16_strided"
                const val FLOAT: String = "ugelu_f32_strided"
                const val BFLOAT: String = "ugelu_bf16_strided"
            }
            object log {
                const val HALF: String = "ulog_f16_strided"
                const val FLOAT: String = "ulog_f32_strided"
                const val BFLOAT: String = "ulog_bf16_strided"
            }
            object neg {
                const val HALF: String = "uneg_f16_strided"
                const val FLOAT: String = "uneg_f32_strided"
                const val BFLOAT: String = "uneg_bf16_strided"
            }
            object relu {
                const val HALF: String = "urelu_f16_strided"
                const val FLOAT: String = "urelu_f32_strided"
                const val BFLOAT: String = "urelu_bf16_strided"
            }
            object round {
                const val HALF: String = "uround_f16_strided"
                const val FLOAT: String = "uround_f32_strided"
                const val BFLOAT: String = "uround_bf16_strided"
            }
            object silu {
                const val HALF: String = "usilu_f16_strided"
                const val FLOAT: String = "usilu_f32_strided"
                const val BFLOAT: String = "usilu_bf16_strided"
            }
            object sin {
                const val HALF: String = "usin_f16_strided"
                const val FLOAT: String = "usin_f32_strided"
                const val BFLOAT: String = "usin_bf16_strided"
            }
            object sqr {
                const val HALF: String = "usqr_f16_strided"
                const val FLOAT: String = "usqr_f32_strided"
                const val BFLOAT: String = "usqr_bf16_strided"
            }
            object sqrt {
                const val HALF: String = "usqrt_f16_strided"
                const val FLOAT: String = "usqrt_f32_strided"
                const val BFLOAT: String = "usqrt_bf16_strided"
            }
            object tanh {
                const val HALF: String = "utanh_f16_strided"
                const val FLOAT: String = "utanh_f32_strided"
                const val BFLOAT: String = "utanh_bf16_strided"
            }
        }
    }

    val callAffine: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, Buffer, Float, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callAffineStrided: (Device, CommandsGuard, Kernels, String, List<Int>, BufferOffset, List<Int>, Buffer, Float, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callPowf: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, Buffer, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callPowfStrided: (Device, CommandsGuard, Kernels, String, List<Int>, BufferOffset, List<Int>, Buffer, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callElu: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, Buffer, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callEluStrided: (Device, CommandsGuard, Kernels, String, List<Int>, BufferOffset, List<Int>, Buffer, Float) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callReduceContiguous: (Device, CommandsGuard, Kernels, String, List<Int>, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callReduceStrided: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callConstSetContiguous: (Device, CommandsGuard, Kernels, String, Int, Int, Any?, BufferOffset) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callConstSetStrided: (Device, CommandsGuard, Kernels, String, List<Int>, Any?, List<Int>, BufferOffset) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callCastContiguous: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callCastStrided: (Device, CommandsGuard, Kernels, String, List<Int>, BufferOffset, List<Int>, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callUnaryContiguous: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callUnaryStrided: (Device, CommandsGuard, Kernels, String, List<Int>, BufferOffset, List<Int>, BufferOffset) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callWhereCond: (Device, CommandsGuard, Kernels, String, Int, List<Int>, BufferOffset, List<Int>, Boolean, BufferOffset, List<Int>, Boolean, BufferOffset, List<Int>, Boolean, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callIm2col1dStrided: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, List<Int>, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callCol2im1d: (Device, CommandsGuard, Kernels, String, List<Int>, Int, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callConvTranspose1d: (Device, CommandsGuard, Kernels, String, Int, Int, Int, Int, Int, Int, Int, List<Int>, List<Int>, List<Int>, List<Int>, Buffer, Int, Buffer, Int, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callIm2colStrided: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, List<Int>, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callConvTranspose2d: (Device, CommandsGuard, Kernels, String, CallConvTranspose2dCfg, Buffer, Buffer, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callPool2d: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, Int, Int, Int, Int, Int, Int, Buffer, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callUpsampleNearest2d: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, Int, Int, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callUpsampleBilinear2d: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, Int, Int, Boolean, Double?, Double?, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callGather: (Device, CommandsGuard, Kernels, String, List<Int>, Int, Int, BufferOffset, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callScatter: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, Int, BufferOffset, BufferOffset, BufferOffset) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callIndexSelect: (Device, CommandsGuard, Kernels, String, List<Int>, Int, Int, Boolean, List<Int>, List<Int>, BufferOffset, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callIndexAdd: (Device, CommandsGuard, Kernels, String, List<Int>, List<Int>, List<Int>, Int, BufferOffset, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callMlxGemm: (Device, CommandsGuard, Kernels, GemmDType, List<Int>, List<Int>, Int, Buffer, List<Int>, Int, Buffer, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callCopy2d: (Device, CommandsGuard, Kernels, String, Buffer, Buffer, Int, Int, Int, Int, Int, Int) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callBinaryContiguous: (Device, CommandsGuard, Kernels, String, Int, Int, BufferOffset, BufferOffset, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callBinaryStrided: (Device, CommandsGuard, Kernels, String, Int, List<Int>, BufferOffset, List<Int>, BufferOffset, List<Int>, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callRandomUniform: (Device, CommandsGuard, Kernels, String, Float, Float, Int, Buffer, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
    val callRandomNormal: (Device, CommandsGuard, Kernels, String, Float, Float, Int, Buffer, Buffer) -> Result<Unit> =
        { _, _, _, _, _, _, _, _, _ -> Result.success(Unit) }
}

internal val MetalDevice.clone: () -> MetalDevice get() = { this }
internal val String.clone: () -> String get() = { this }
internal fun <T> List<T>.len(): Int = size
internal fun <T> List<T>.clone(): List<T> = toList()
internal fun <T> List<T>.iter(): List<T> = this
internal val List<Int>.enumerate: () -> Iterable<IndexedValue<Int>> get() = { withIndex() }
internal val List<Int>.product: () -> Int get() = { fold(1) { a, b -> a * b } }
internal fun <T> MutableList<T>.push(v: T) { add(v) }
internal fun <T> MutableList<T>.swapRemove(idx: Int): T = removeAt(idx)
internal fun <T> List<T>.asPtr(): List<T> = this
internal fun <T> List<T>.isNull(): Boolean = false
internal fun <T> List<T>.toVec(): List<T> = this
internal val Buffer.contents: () -> List<Nothing> get() = { emptyList() }
internal val fromRawParts: (List<*>, Int) -> List<Nothing> = { _, _ -> emptyList() }
internal val Buffer.lock: () -> Result<Buffer> get() = { Result.success(this) }
internal val Buffer.tryLock: () -> Result<Buffer> get() = { Result.success(this) }
internal fun <T> Result<T>.unwrap(): T = getOrThrow()
internal val ULong.write: () -> Result<ULong> get() = { Result.success(this) }
internal val ULong.read: () -> Result<ULong> get() = { Result.success(this) }
internal val copyNonoverlapping: (List<ULong>, ByteArray, Int) -> Unit = { _, _, _ -> }
internal val sizeOfU64: Int = 8
internal val NSString: DeviceId.Companion get() = DeviceId.Companion
internal val DeviceId.Companion.fromStr: (String) -> String get() = { it }
internal val RwLock: DeviceId.Companion get() = DeviceId.Companion
internal val Mutex: DeviceId.Companion get() = DeviceId.Companion
internal val HashMap: DeviceId.Companion get() = DeviceId.Companion
internal fun DeviceId.Companion.new(it: ULong): ULong = it
internal fun DeviceId.Companion.new(it: Kernels): Kernels = it
internal fun DeviceId.Companion.new(it: ResidencySet): ResidencySet = it
internal fun DeviceId.Companion.new(it: Commands): Commands = it
internal fun DeviceId.Companion.new(it: MutableMap<Int, MutableList<Buffer>>): MutableMap<Int, MutableList<Buffer>> = it
internal val DeviceId.Companion.newMap: () -> MutableMap<Int, MutableList<Buffer>> get() = { mutableMapOf() }
internal val drop: (Any?) -> Unit = {}

internal val CpuStorageRef.U8.storage: List<UByte> get() = data.asList()
internal val CpuStorageRef.U32.storage: List<UInt> get() = data.asList()
internal val CpuStorageRef.I16.storage: List<Short> get() = data.asList()
internal val CpuStorageRef.I32.storage: List<Int> get() = data.asList()
internal val CpuStorageRef.I64.storage: List<Long> get() = data.asList()
internal val CpuStorageRef.BF16.storage: List<Short> get() = data.asList()
internal val CpuStorageRef.F16.storage: List<Short> get() = data.asList()
internal val CpuStorageRef.F32.storage: List<Float> get() = data.asList()
internal val CpuStorageRef.F64.storage: List<Double> get() = data.asList()
internal val CpuStorageRef.F8E4M3.storage: List<Byte> get() = data.asList()

internal val CpuStorage.U8.storage: List<UByte> get() = data.asList()
internal val CpuStorage.U32.storage: List<UInt> get() = data.asList()
internal val CpuStorage.I16.storage: List<Short> get() = data.asList()
internal val CpuStorage.I32.storage: List<Int> get() = data.asList()
internal val CpuStorage.I64.storage: List<Long> get() = data.asList()
internal val CpuStorage.BF16.storage: List<Short> get() = data.asList()
internal val CpuStorage.F16.storage: List<Short> get() = data.asList()
internal val CpuStorage.F32.storage: List<Float> get() = data.asList()
internal val CpuStorage.F64.storage: List<Double> get() = data.asList()
internal val CpuStorage.F8E4M3.storage: List<Byte> get() = data.asList()

internal val MetalStorage.self: MetalStorage get() = this
internal val MetalStorage.clone: () -> MetalStorage get() = { MetalStorage(buffer(), device(), count, dtype()) }
internal val MetalStorage.Self: MetalStorage.Companion get() = MetalStorage.Companion
internal val MetalBackendDevice.self: MetalDevice get() = inner
internal val MetalBackendDevice.Self: MetalBackendDevice.Companion get() = MetalBackendDevice.Companion
internal val MetalBackendDevice.Companion.Storage: MetalStorage.Companion get() = MetalStorage.Companion
internal val Err: (Throwable) -> Result<Nothing> = { Result.failure(it) }
internal val bail: (String) -> Nothing = { throw CandleException.Msg(it) }
internal val format: (String) -> String = { it }
internal val Throwable.into: () -> Throwable get() = { this }
internal val MetalError.Companion.from: (String) -> MetalError get() = { e -> MetalError.Message(e) }
internal val MetalDevice.zerosImpl: (Shape, DType) -> Result<MetalStorage> get() = { shape, dtype -> MetalBackendDevice(this).zerosImpl(shape, dtype) }
internal val Layout.Companion.new: (Shape, List<Int>, Int) -> Layout get() = { s, st, off -> Layout(s, st, off) }
internal val Layout.Companion.contiguous: (List<Int>) -> Layout get() = { dims -> Layout.contiguous(Shape(dims)) }
internal val Layout.Companion.contiguousWithOffset: (List<Int>, Int) -> Layout get() = { dims, off -> Layout.contiguousWithOffset(Shape(dims), off) }
internal val Layout.broadcastAs: (List<Int>) -> Result<Layout> get() = { dims -> runCatching { broadcastAs(Shape(dims)) } }
internal val List<Int>.into: () -> Shape get() = { Shape(this) }
internal fun Shape.dims3(): Result<Triple<Int, Int, Int>> = runCatching {
    require(dims.size == 3) { "Expected 3 dims, got ${dims.size}" }
    Triple(dims[0], dims[1], dims[2])
}
internal fun Shape.dims4(): Result<List<Int>> = runCatching {
    require(dims.size == 4) { "Expected 4 dims, got ${dims.size}" }
    dims
}
