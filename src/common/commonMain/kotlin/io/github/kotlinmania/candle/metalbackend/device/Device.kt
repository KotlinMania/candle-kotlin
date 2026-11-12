// port-lint: source metal_backend/device.rs
package io.github.kotlinmania.candle.metalbackend.device

import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.metalbackend.MetalError
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.metal.BlitCommandsGuard
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.BufferMap
import io.github.kotlinmania.candlemetalkernels.metal.Commands
import io.github.kotlinmania.candlemetalkernels.metal.CommandsGuard
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLResourceOptions
import io.github.kotlinmania.candlemetalkernels.metal.ResidencySet
import io.github.kotlinmania.objc2foundation.Nsurl
import io.github.kotlinmania.objc2metal.MTLCaptureDescriptor
import io.github.kotlinmania.objc2metal.MTLCaptureDestination
import io.github.kotlinmania.objc2metal.MTLCaptureManager

public typealias Target = Device

/**
 * Unique identifier for metal devices.
 */
public open class DeviceId(
    public var value: Int,
) {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }

    public companion object {
        internal fun new(): DeviceId {
            val COUNTER: atomic.AtomicUsize  = atomic.AtomicUsize.new(1)
            return DeviceId(COUNTER.fetchAdd(1, atomic.Ordering.Relaxed))
        }
    }
}

public open class MetalDevice(
    /**
     * Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
     * the device itself.
     */
    internal var id: DeviceId,
    /**
     * Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
     */
    internal var device: Device,
    internal var commands: Commands,
    /**
     * Simple allocator struct.
     * The buffers are stored in size buckets since ML tends to use similar shapes over and over.
     * We store the buffers in [Arc] because it's much faster than Obj-c internal ref counting
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
    internal var buffers: RwLock<BufferMap>,
    /**
     * Same as `buffers` but uses `PRIVATE_RESOURCE_OPTIONS` (StorageModePrivate on macOS).
     * Intermediate compute buffers don't need CPU access so Private avoids coherency overhead.
     */
    internal var privateBuffers: RwLock<BufferMap>,
    /**
     * Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
     * Heavily used by [candleMetalKernels]
     */
    internal var kernels: Kernels,
    /**
     * Seed for random number generation.
     */
    internal var seed: Mutex<Buffer>,
    /**
     * Last seed value set on this device.
     */
    internal var seedValue: RwLock<ULong>,
    /**
     * Residency set registered on the command queue.
     */
    internal var residencySet: ResidencySet,
) {
    public open fun fmt(f: StringBuilder) {
        f.append("MetalDevice(${this.id})")
    }

    public open fun deref(): Device = this.device

    public open fun id(): DeviceId = this.id

    public open fun metalDevice(): Device = this.device

    public open fun dropUnusedBuffers(): Result<Unit> {
        val buffers = this.buffers.write().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        for (subbuffers in buffers.valuesMut()) {
            subbuffers.retain({ s ->
                if (Arc.strongCount(s) == 1) {
                    this.residencySet.remove(s)
                    false
                } else {
                    true
                }
            })
        }
        val privateBuffers = this.privateBuffers.write().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        for (subbuffers1 in privateBuffers.valuesMut()) {
            subbuffers1.retain({ s ->
                if (Arc.strongCount(s) == 1) {
                    this.residencySet.remove(s)
                    false
                } else {
                    true
                }
            })
        }
        return Result.success(Unit)
    }

    public open fun commandEncoder(): Result<CommandsGuard> {
        val commandEncoder = this.commands.commandEncoder().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        return Result.success(commandEncoder)
    }

    public open fun blitCommandEncoder(): Result<BlitCommandsGuard> {
        val commandEncoder = this.commands.blitCommandEncoder().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        return Result.success(commandEncoder)
    }

    public open fun waitUntilCompleted(): Result<Unit> {
        this.commands.waitUntilCompleted().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        this.dropUnusedBuffers().getOrElse { return Result.failure(it) }
        return Result.success(Unit)
    }

    /**
     * Commit and wait on the buffer holding the caller's work; safe for concurrent CPU readbacks.
     */
    public open fun flushAndWaitCurrent(): Result<Unit> {
        this.commands.flushAndWaitCurrent().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        this.dropUnusedBuffers().getOrElse { return Result.failure(it) }
        return Result.success(Unit)
    }

    public open fun kernels(): Kernels = this.kernels

    public open fun device(): Device = this.device

    /**
     * Registers buffers in the device's residency set, keeping them
     * permanently GPU-resident instead of paying per-command-buffer residency
     * bookkeeping. Useful for buffers candle did not allocate, e.g.
     * `newBufferWithBytesNoCopy` views over an mmap'd weights file. No-op on
     * systems without residency-set support.
     */
    public open fun registerBuffers(bufs: Any) {
        this.residencySet.insertBatch(bufs)
    }

    /**
     * Unregisters buffers previously passed to `registerBuffers`, releasing
     * the set's retain so they can be deallocated. Only unregister buffers
     * you registered yourself, after GPU work referencing them has completed.
     */
    public open fun unregisterBuffers(bufs: Any) {
        this.residencySet.removeBatch(bufs)
    }

    /**
     * Returns a builder for buffer allocation. See `BufferBuilder`.
     */
    public open fun newBufferBuilder(): BufferBuilder = BufferBuilder.new(this)

    /**
     * Creates a new buffer (not necessarily zeroed).
     *
     * Uses StorageModePrivate on macOS for faster GPU access (no CPU coherency overhead).
     * Falls back to StorageModeShared on iOS where Private is not always available.
     */
    public open fun newBuffer(elementCount: Int, dtype: DType, name: String): Result<Buffer> {
        val size = elementCount * dtype.sizeInBytes()
        val buffers = this.privateBuffers.write().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        when (val matchVal = findAvailableBuffer(size, buffers)) {
            null -> {}
            else -> {
                val b = matchVal
                return Result.success(b.clone())
            }
        }
        val size1 = bufSize(size)
        val subbuffers = buffers.entry(size1).orInsert(mutableListOf())
        val newBuffer = this.device.newBuffer(size1, PRIVATE_RESOURCE_OPTIONS).mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        val newBuffer1 = newBuffer
        this.residencySet.insert(newBuffer1)
        subbuffers.push(newBuffer1.clone())
        return Result.success(newBuffer1)
    }

    /**
     * Creates a new private buffer (not necessarily zeroed).
     *
     * This is intentionally not in the Metal buffer pool to allow the efficient implementation of persistent buffers.
     */
    public open fun newPrivateBuffer(elementCount: Int, dtype: DType, name: String): Result<Buffer> {
        val size = elementCount * dtype.sizeInBytes()
        val buffer = this.device.newBuffer(size, PRIVATE_RESOURCE_OPTIONS).mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        val buffer1 = buffer
        this.residencySet.insert(buffer1)
        return Result.success(buffer1)
    }

    /**
     * Creates a new buffer from data.
     *
     * Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
     * allocates the buffer and copies over the existing data before returning the MTLBuffer.
     */
    public open fun <T> newBufferWithData(data: List<T>): Result<Buffer> {
        val size = sizeOfVal(data)
        val newBuffer = this.device.newBufferWithData(data.asPtr(), size, RESOURCE_OPTIONS).mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        val buffers = this.buffers.write().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        val subbuffers = buffers.entry(size).orInsert(mutableListOf())
        val newBuffer1 = newBuffer
        this.residencySet.insert(newBuffer1)
        subbuffers.push(newBuffer1.clone())
        return Result.success(newBuffer1)
    }

    public open fun allocateZeros(sizeInBytes: Int): Result<Buffer> {
        val buffer = this.allocateBuffer(sizeInBytes).getOrElse { return Result.failure(it) }
        val blit = this.blitCommandEncoder().getOrElse { return Result.failure(it) }
        blit.setLabel("zeros")
        blit.fillBuffer(buffer, Pair(0, buffer.length()), 0)
        //
        // // Alternative impl
        // if sizeInBytes > 0 {
        // let encoder = self.commandEncoder()?;
        // callConstFill(
        // &self.device,
        // &encoder,
        // &self.kernels,
        // "fillU8",
        // sizeInBytes,
        // &buffer,
        // 0u8,
        // )
        // .mapErr(Error.wrap)?;
        // }
        //
        return Result.success(buffer)
    }

    /**
     * The critical allocator algorithm
     */
    public open fun allocateBuffer(size: Int): Result<Buffer> {
        val buffers = this.buffers.write().mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        when (val matchVal = findAvailableBuffer(size, buffers)) {
            null -> {}
            else -> {
                val b = matchVal
                // Cloning also ensures we increment the strong count
                return Result.success(b.clone())
            }
        }
        val size1 = bufSize(size)
        val subbuffers = buffers.entry(size1).orInsert(mutableListOf())
        val newBuffer = this.device.newBuffer(size1, RESOURCE_OPTIONS).mapErr(MetalError::from).getOrElse { return Result.failure(it) }
        val newBuffer1 = newBuffer
        this.residencySet.insert(newBuffer1)
        subbuffers.push(newBuffer1.clone())
        return Result.success(newBuffer1)
    }

    /**
     * Create a metal GPU capture trace on [path].
     */
    public open fun <P : AsRef<Path>> capture(path: P): Result<Unit> {
        val capture = MTLCaptureManager.sharedCaptureManager()
        val descriptor = MTLCaptureDescriptor.new()
        descriptor.setDestination(MTLCaptureDestination.GPUTraceDocument)
        descriptor.setCaptureDevice(this.device().asRef())
        // The [setOutputUrl] call requires an absolute path so we convert it if needed.
        if (path.asRef().isAbsolute()) {
            val url = Nsurl.fromFilePath(path)
            descriptor.setOutputURL(url.asDeref())
        } else {
            val path1 = currentDir().getOrElse { return Result.failure(it) }.join(path)
            val url1 = Nsurl.fromFilePath(path1)
            descriptor.setOutputURL(url1.asDeref())
        }
        capture.startCaptureWithDescriptorError(descriptor).mapErr({ e -> MetalError.from(e.toString()) }).getOrElse { return Result.failure(it) }
        return Result.success(Unit)
    }
}

// Resource options used for creating buffers. Shared storage mode allows both CPU and GPU to access the buffer.
public val RESOURCE_OPTIONS: MTLResourceOptions = MTLResourceOptions(MTLResourceOptions.StorageModeShared.first or MTLResourceOptions.HazardTrackingModeUntracked.first)

// Resource options used for `newPrivateBuffer`. This uses `private` where supported.
public val PRIVATE_RESOURCE_OPTIONS: MTLResourceOptions = RESOURCE_OPTIONS

public val PRIVATE_RESOURCE_OPTIONS1: MTLResourceOptions = MTLResourceOptions(MTLResourceOptions.StorageModePrivate.first or MTLResourceOptions.HazardTrackingModeUntracked.first)

public fun bufSize(size: Int): Int = size.nextPowerOfTwo()

/**
 * Applies the [BufferBuilder] label, clearing any stale label on a reused pooled buffer.
 */
public fun bufferLabel(buffer: Buffer, label: String?) {
    buffer.setLabel((label ?: "unlabeled"))
}

public typealias DataUpload = (MetalDevice) -> Result<Buffer>

public sealed class BufferInit {
    public data class Typed(public var elemCount: Int, public var dtype: DType) : BufferInit()
    public data class Size(public var value: Int) : BufferInit()
    public data class Zeros(public var value: Int) : BufferInit()
    public data class Data(public var value: DataUpload) : BufferInit()
}

/**
 * Builder for `MTLBuffer` allocations; pool reuse handled by [MetalDevice].
 */
public open class BufferBuilder(
    public var device: MetalDevice,
    public var label: String?,
) {
    /**
     * Allocate elemCount * dtype size bytes, uninitialized, private storage.
     */
    public open fun withSizeFor(elemCount: Int, dtype: DType): ReadyBufferBuilder = this.ready(BufferInit.Typed(elemCount = elemCount, dtype = dtype))

    /**
     * Allocate size bytes, uninitialized, shared storage.
     */
    public open fun withSize(size: Int): ReadyBufferBuilder = this.ready(BufferInit.Size(size))

    /**
     * Allocate size bytes, zero-filled, shared storage. Pool rounding may make
     * the allocation larger than size; the extra bytes are also zeroed.
     */
    public open fun withZeros(size: Int): ReadyBufferBuilder = this.ready(BufferInit.Zeros(size))

    /**
     * Allocate a shared buffer initialized from data. Always allocates; does not
     * reuse the pool.
     */
    public open fun <T> withData(data: List<T>): ReadyBufferBuilder =
        this.ready(BufferInit.Data({ device ->
            device.newBufferWithData(data)
        }))

    public open fun withLabel(label: String): BufferBuilder {
        this.label = label
        return this
    }

    public open fun ready(init: BufferInit): ReadyBufferBuilder = ReadyBufferBuilder(device = this.device, init = init, label = this.label)

    public companion object {
        public fun new(device: MetalDevice): BufferBuilder = BufferBuilder(device = device, label = null)
    }
}

/**
 * [BufferBuilder] with an init kind set; `build()` lives here.
 */
public open class ReadyBufferBuilder(
    public var device: MetalDevice,
    public var init: BufferInit,
    public var label: String?,
) {
    public open fun withLabel(label: String): ReadyBufferBuilder {
        this.label = label
        return this
    }

    public open fun build(): Result<Buffer> {
        val buffer =
            when (val matchVal = this.init) {
                is BufferInit.Typed -> {
                    val elemCount = matchVal.elemCount
                    val dtype = matchVal.dtype
                    this.device.newBuffer(elemCount, dtype, "").getOrElse { return Result.failure(it) }
                }
                is BufferInit.Size -> {
                    val (size) = matchVal
                    this.device.allocateBuffer(size).getOrElse { return Result.failure(it) }
                }
                is BufferInit.Zeros -> {
                    val (size1) = matchVal
                    this.device.allocateZeros(size1).getOrElse { return Result.failure(it) }
                }
                is BufferInit.Data -> {
                    val (upload) = matchVal
                    upload(this.device).getOrElse { return Result.failure(it) }
                }
            }
        bufferLabel(buffer, this.label)
        return Result.success(buffer)
    }
}

public fun findAvailableBuffer(size: Int, buffers: BufferMap): Buffer? {
    var bestBuffer: Buffer? = null
    var bestBufferSize = Int.MAX_VALUE
    for ((bufferSize, subbuffers) in buffers.iter()) {
        if (bufferSize >= size && bufferSize < bestBufferSize) {
            for (sub in subbuffers) {
                if (Arc.strongCount(sub) == 1) {
                    bestBuffer = sub
                    bestBufferSize = bufferSize
                }
            }
        }
    }
    return bestBuffer.cloned()
}
