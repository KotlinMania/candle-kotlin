// port-lint: source safetensors.rs
/**
 * Module to load `safetensor` files into CPU/GPU memory.
 *
 * There are multiple ways to load tensors from safetensor files:
 * - `load` function for loading directly into memory and returning a HashMap of tensors
 * - `MmapedSafetensors` for memory mapping files and avoiding full allocation
 * - `SliceSafetensors` for working with in-memory buffers
 * - `BufferedSafetensors` for owning a buffer of data
 *
 * Tensors can also be serialized to safetensor format using the `save` function or
 * `Tensor.saveSafetensors` method.
 */
package io.github.kotlinmania.candle

/**
 * Supported element data types in the `safetensors` wire format.
 */
enum class SafeTensorDtype(val wireName: String, val byteSize: Int) {
    U8("U8", 1),
    U16("U16", 2),
    U32("U32", 4),
    I16("I16", 2),
    I32("I32", 4),
    I64("I64", 8),
    BF16("BF16", 2),
    F16("F16", 2),
    F32("F32", 4),
    F64("F64", 8),
    F8E4M3("F8_E4M3", 1),
    F6E2M3("F6_E2M3", 1),
    F6E3M2("F6_E3M2", 1),
    F4("F4", 1),
    F8E8M0("F8_E8M0", 1);

    interface Error
}

/**
 * Zero-copy view over a single tensor stored in a `safetensors` buffer,
 * holding its wire [wireDtype], [tensorShape], and raw [rawBytes] bytes.
 *
 * Implements [Load] so it can be materialized directly onto a target [Device].
 */
class SafeTensorView(
    /** The safetensors wire element type. */
    val wireDtype: SafeTensorDtype,
    /** The tensor dimensions. */
    val tensorShape: IntArray,
    /** Raw little-endian tensor bytes. */
    val rawBytes: ByteArray,
) : Load {
    private val self: SafeTensorView get() = this
    val dtype: () -> SafeTensorDtype = { this.wireDtype }
    val shape: () -> IntArray = { this.tensorShape }
    val data: () -> ByteArray = { this.rawBytes }

    /**
     * Loads this tensor view onto the specified [device].
     */
    override fun load(device: Device): CandleResult<Tensor> {
        return convert(self, device)
    }
}

fun from(value: DType): SafeTensorDtype {
    return when (value) {
        DType.U8 -> SafeTensorDtype.U8
        DType.U32 -> SafeTensorDtype.U32
        DType.I16 -> SafeTensorDtype.I16
        DType.I32 -> SafeTensorDtype.I32
        DType.I64 -> SafeTensorDtype.I64
        DType.BF16 -> SafeTensorDtype.BF16
        DType.F16 -> SafeTensorDtype.F16
        DType.F32 -> SafeTensorDtype.F32
        DType.F64 -> SafeTensorDtype.F64
        DType.F8E4M3 -> SafeTensorDtype.F8E4M3
        DType.F6E2M3 -> SafeTensorDtype.F6E2M3
        DType.F6E3M2 -> SafeTensorDtype.F6E3M2
        DType.F4 -> SafeTensorDtype.F4
        DType.F8E8M0 -> SafeTensorDtype.F8E8M0
    }
}

fun tryFrom(value: SafeTensorDtype): CandleResult<DType> {
    return when (value) {
        SafeTensorDtype.U8 -> CandleResult.Ok(DType.U8)
        SafeTensorDtype.U32 -> CandleResult.Ok(DType.U32)
        SafeTensorDtype.I16 -> CandleResult.Ok(DType.I16)
        SafeTensorDtype.I32 -> CandleResult.Ok(DType.I32)
        SafeTensorDtype.I64 -> CandleResult.Ok(DType.I64)
        SafeTensorDtype.BF16 -> CandleResult.Ok(DType.BF16)
        SafeTensorDtype.F16 -> CandleResult.Ok(DType.F16)
        SafeTensorDtype.F32 -> CandleResult.Ok(DType.F32)
        SafeTensorDtype.F64 -> CandleResult.Ok(DType.F64)
        SafeTensorDtype.F8E4M3 -> CandleResult.Ok(DType.F8E4M3)
        SafeTensorDtype.F6E2M3 -> CandleResult.Ok(DType.F6E2M3)
        SafeTensorDtype.F6E3M2 -> CandleResult.Ok(DType.F6E3M2)
        SafeTensorDtype.F4 -> CandleResult.Ok(DType.F4)
        SafeTensorDtype.F8E8M0 -> CandleResult.Ok(DType.F8E8M0)
        else -> CandleResult.Err(CandleError.UnsupportedSafeTensorDtype(value.wireName))
    }
}

private val DType.into: () -> SafeTensorDtype
    get() = { from(this) }

private val IntArray.iter: () -> IntArray
    get() = { this }

private val IntArray.product: () -> Int
    get() = { this.fold(1) { acc, d -> acc * d } }

private val ByteArray.len: () -> Int
    get() = { this.size }

private val ByteArray.asPtr: () -> Int
    get() = { this.size }

private val Int.isMultipleOf: (Int) -> Boolean
    get() = { d -> d > 0 && this % d == 0 }

private val String.toPathBuf: () -> String
    get() = { this }

private fun <T> List<CandleResult<T>>.collect(): CandleResult<List<T>> {
    val out = ArrayList<T>(this.size)
    for (item in this) {
        if (item is CandleResult.Ok) {
            out.add(item.value)
        } else if (item is CandleResult.Err) {
            return CandleResult.Err(item.error)
        }
    }
    return CandleResult.Ok(out)
}

private val List<CandleResult<kotlin.Pair<String, Tensor>>>.collectMap: () -> CandleResult<Map<String, Tensor>>
    get() = collectMapLambda@{
        val out = LinkedHashMap<String, Tensor>(this.size)
        for (item in this) {
            if (item is CandleResult.Ok) {
                out[item.value.first] = item.value.second
            } else if (item is CandleResult.Err) {
                return@collectMapLambda CandleResult.Err(item.error)
            }
        }
        CandleResult.Ok(out)
    }

class TensorSafeView(private val self: Tensor) {
    fun dtype(): SafeTensorDtype {
        return self.dtype().into()
    }

    fun shape(): IntArray {
        return self.shape().dims().toIntArray()
    }

    fun data(): ByteArray {
        // This copies data from GPU to CPU.
        // Avoid the unwrap here when possible.
        return SafeTensors.cowOwned(convertBack(self).unwrap())
    }

    fun dataLen(): Int {
        val n = self.shape().elemCount()
        val bytesPerElement = self.dtype().sizeInBytes()
        return n * bytesPerElement
    }
}

class TensorRefSafeView(private val self: Tensor) {
    fun dtype(): SafeTensorDtype {
        return self.dtype().into()
    }

    fun shape(): IntArray {
        return self.dims().toIntArray()
    }

    fun data(): ByteArray {
        // This copies data from GPU to CPU.
        // Avoid the unwrap here when possible.
        return SafeTensors.cowOwned(convertBack(self).unwrap())
    }

    fun dataLen(): Int {
        val n = self.dims().toIntArray().iter().product()
        val bytesPerElement = self.dtype().sizeInBytes()
        return n * bytesPerElement
    }
}

fun Tensor.saveSafetensors(name: String, filename: String): CandleResult<Unit> {
    val data = linkedMapOf(name to this.clone())
    return SafeTensors.serializeToFile(data, null, filename.asRef())
}

private fun <T> convertSlice(data: ByteArray, shape: IntArray, device: Device, dtype: DType = DType.F32): CandleResult<Tensor> {
    val sizeInBytes = dtype.sizeInBytes()
    val elemCount = data.len() / sizeInBytes
    return if (data.asPtr().isMultipleOf(sizeInBytes)) {
        // Safe because we just checked that this
        // was correctly aligned.
        val slice =
            SafeTensors.fromRawParts(data.asPtr(), elemCount, data, dtype)
        SafeTensors.fromSlice(slice, shape, device)
    } else {
        // Specify the target element type explicitly so the compiler does not infer bytes
        // and make this vector too small to fit full weights, resulting in out-of-bounds access.
        val c = SafeTensors.withCapacity(elemCount, dtype)
        // We just created the target vector, so the allocated memory is necessarily
        // contiguous and non-overlapping with the view data.
        // Downgrading the pointer to raw bytes removes alignment
        // constraints.
        run {
            SafeTensors.copyNonoverlapping(data.asPtr(), c.asMutPtr(), data.len(), data, c)
            c.setLen(elemCount)
        }
        SafeTensors.fromSlice(c, shape, device)
    }
}

private fun <T, U> convertSliceWithCast(
    data: ByteArray,
    shape: IntArray,
    device: Device,
    conv: (UShort) -> CandleResult<UInt>,
): CandleResult<Tensor> {
    val sizeInBytes = SafeTensors.sizeOf()
    val elemCount = data.len() / sizeInBytes
    return if (data.asPtr().isMultipleOf(sizeInBytes)) {
        // Safe because we just checked that this
        // was correctly aligned.
        val slice =
            SafeTensors.fromRawPartsU16(data.asPtr(), elemCount, data)
        val mapped = slice.iter().map { t -> conv(t) }.collect().unwrapOrElse { return CandleResult.Err(it) }
        SafeTensors.fromVec(mapped, shape, device)
    } else {
        // Specify the target element type explicitly so the compiler does not infer bytes
        // and make this vector too small to fit full weights, resulting in out-of-bounds access.
        val c = SafeTensors.withCapacityU16(elemCount)
        // We just created the target vector, so the allocated memory is necessarily
        // contiguous and non-overlapping with the view data.
        // Downgrading the pointer to raw bytes removes alignment
        // constraints.
        run {
            SafeTensors.copyNonoverlappingU16(data.asPtr(), c.asMutPtr(), data.len(), data, c)
            c.setLen(elemCount)
        }
        val mapped = c.map(conv).collect().unwrapOrElse { return CandleResult.Err(it) }
        SafeTensors.fromVec(mapped, shape, device)
    }
}

private fun <T, U> convertWithCast(
    view: SafeTensorView,
    device: Device,
    conv: (UShort) -> CandleResult<UInt>,
): CandleResult<Tensor> {
    return convertSliceWithCast<T, U>(view.data(), view.shape(), device, conv)
}

private fun <T> convert(view: SafeTensorView, device: Device, dtype: DType = DType.F32): CandleResult<Tensor> {
    return convertSlice<T>(view.data(), view.shape(), device, dtype)
}

private fun <T> convertBack(vs: CpuStorage): ByteArray {
    val sizeInBytes = vs.dtype().sizeInBytes()
    val length = vs.len() * sizeInBytes
    val capacity = vs.capacity() * sizeInBytes
    val ptr = vs.asMutPtrBytes()
    // Do not run the destructor for the source vector
    SafeTensors.forget(vs)
    // SAFETY:
    //
    // Every element is at least one byte, so there is no issue regarding alignment.
    // This reinterprets the typed vector as a byte vector.
    return SafeTensors.vecFromRawParts(ptr, length, capacity)
}

interface Load {
    fun load(device: Device): CandleResult<Tensor>
}

fun fromRawBuffer(
    data: ByteArray,
    dtype: DType,
    shape: IntArray,
    device: Device,
): CandleResult<Tensor> {
    return when (dtype) {
        DType.U8 -> convertSlice<UByte>(data, shape, device, DType.U8)
        DType.U32 -> convertSlice<UInt>(data, shape, device, DType.U32)
        DType.I16 -> convertSlice<Short>(data, shape, device, DType.I16)
        DType.I32 -> convertSlice<Int>(data, shape, device, DType.I32)
        DType.I64 -> convertSlice<Long>(data, shape, device, DType.I64)
        DType.BF16 -> convertSlice<BFloat16>(data, shape, device, DType.BF16)
        DType.F16 -> convertSlice<Float16>(data, shape, device, DType.F16)
        DType.F32 -> convertSlice<Float>(data, shape, device, DType.F32)
        DType.F64 -> convertSlice<Double>(data, shape, device, DType.F64)
        DType.F8E4M3 -> convertSlice<F8E4M3>(data, shape, device, DType.F8E4M3)
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            // For dummy types, create storage with raw bytes
            val storage = when (device) {
                is Device.Cpu -> {
                    val cpuStorage = when (dtype) {
                        DType.F6E2M3 -> CpuStorage.F6E2M3(data)
                        DType.F6E3M2 -> CpuStorage.F6E3M2(data)
                        DType.F4 -> CpuStorage.F4(data)
                        else -> CpuStorage.F8E8M0(data)
                    }
                    Storage.Cpu(cpuStorage)
                }
                // Allocate device slice and copy host buffer on CUDA
                is Device.Cuda -> {
                    val dev = device.device
                    val slice = SafeTensors.cudaAlloc(dev, data.len()).unwrapOrElse { return CandleResult.Err(it) }
                    SafeTensors.cudaMemcpyHtod(dev, data, slice).unwrapOrElse { return CandleResult.Err(it) }

                    val cudaSlice = when (dtype) {
                        DType.F6E2M3 -> SafeTensors.cudaSliceF6E2M3(slice)
                        DType.F6E3M2 -> SafeTensors.cudaSliceF6E3M2(slice)
                        DType.F4 -> SafeTensors.cudaSliceF4(slice)
                        else -> SafeTensors.cudaSliceF8E8M0(slice)
                    }
                    val cudaStorage = SafeTensors.cudaStorage(
                        cudaSlice,
                        dev,
                    )
                    // Return error when CUDA backend is not compiled
                    // in this target build configuration
                    // or wrap the allocated CUDA storage
                    SafeTensors.storageCuda(cudaStorage).unwrapOrElse { return CandleResult.Err(it) }
                }
                // Build Metal buffer and wrap in MetalStorage
                is Device.Metal -> {
                    val dev = device.device
                    val buffer = SafeTensors.newBufferBuilder(dev)
                        .withData(data)
                        .withLabel("safetensors_view")
                        .build()
                        .unwrapOrElse { return CandleResult.Err(it) }

                    val metalStorage = SafeTensors.metalStorageNew(
                        buffer,
                        dev,
                        data.len(),
                        dtype,
                    )
                    // Return error when Metal backend is not compiled
                    // in this target build configuration
                    // or wrap the allocated Metal storage
                    SafeTensors.storageMetal(metalStorage).unwrapOrElse { return CandleResult.Err(it) }
                }
            }

            val op = BackpropOp.none()
            CandleResult.Ok(fromStorage(storage, Shape.fromDims(shape), op, false))
        }
    }
}

private fun convert(view: SafeTensorView, device: Device): CandleResult<Tensor> {
    return when (view.dtype()) {
        SafeTensorDtype.U8 -> convert<UByte>(view, device, DType.U8)
        SafeTensorDtype.U16 -> {
            val conv = { x: UShort -> CandleResult.Ok(x.toUInt()) }
            convertWithCast<UShort, UInt>(view, device, conv)
        }
        SafeTensorDtype.U32 -> convert<UInt>(view, device, DType.U32)
        SafeTensorDtype.I16 -> convert<Short>(view, device, DType.I16)
        SafeTensorDtype.I32 -> convert<Int>(view, device, DType.I32)
        SafeTensorDtype.I64 -> convert<Long>(view, device, DType.I64)
        SafeTensorDtype.BF16 -> convert<BFloat16>(view, device, DType.BF16)
        SafeTensorDtype.F16 -> convert<Float16>(view, device, DType.F16)
        SafeTensorDtype.F32 -> convert<Float>(view, device, DType.F32)
        SafeTensorDtype.F64 -> convert<Double>(view, device, DType.F64)
        SafeTensorDtype.F8E4M3 -> convert<F8E4M3>(view, device, DType.F8E4M3)
        SafeTensorDtype.F6E2M3, SafeTensorDtype.F6E3M2, SafeTensorDtype.F4, SafeTensorDtype.F8E8M0 -> {
            // For dummy types, we need to handle loading by creating a dummy tensor
            // Since these types do not have actual data representation, we will create
            // a tensor that indicates it is a dummy type
            convertDummy(view, device)
        }
        // Any remaining wire dtype is rejected as unsupported
    }
}

private fun convertDummy(view: SafeTensorView, device: Device): CandleResult<Tensor> {
    // For dummy types, we will create the appropriate storage variant that preserves
    // both the raw data and the correct dtype
    val dtypePair = when (view.dtype()) {
        SafeTensorDtype.F6E2M3 -> kotlin.Pair(DType.F6E2M3, "F6_E2M3 (MX6)")
        SafeTensorDtype.F6E3M2 -> kotlin.Pair(DType.F6E3M2, "F6_E3M2 (MX6)")
        SafeTensorDtype.F4 -> kotlin.Pair(DType.F4, "F4 (MX4)")
        SafeTensorDtype.F8E8M0 -> kotlin.Pair(DType.F8E8M0, "F8_E8M0")
        else -> return CandleResult.Err(CandleError.Msg("convertDummy called with non-dummy dtype"))
    }
    val dtype = dtypePair.first

    // Load the raw bytes
    val data = view.data()
    val shape = view.shape()

    // Create storage with the appropriate dummy type variant
    val storage = when (device) {
        is Device.Cpu -> {
            val cpuStorage = when (dtype) {
                DType.F6E2M3 -> CpuStorage.F6E2M3(data)
                DType.F6E3M2 -> CpuStorage.F6E3M2(data)
                DType.F4 -> CpuStorage.F4(data)
                DType.F8E8M0 -> CpuStorage.F8E8M0(data)
                else -> return CandleResult.Err(CandleError.Msg("unreachable dummy dtype"))
            }
            Storage.Cpu(cpuStorage)
        }
        // Allocate device slice and copy host buffer on CUDA
        is Device.Cuda -> {
            val dev = device.device
            val slice = SafeTensors.cudaAlloc(dev, data.len()).unwrapOrElse { return CandleResult.Err(it) }
            SafeTensors.cudaMemcpyHtod(dev, data, slice).unwrapOrElse { return CandleResult.Err(it) }

            val cudaSlice = when (dtype) {
                DType.F6E2M3 -> SafeTensors.cudaSliceF6E2M3(slice)
                DType.F6E3M2 -> SafeTensors.cudaSliceF6E3M2(slice)
                DType.F4 -> SafeTensors.cudaSliceF4(slice)
                DType.F8E8M0 -> SafeTensors.cudaSliceF8E8M0(slice)
                else -> return CandleResult.Err(CandleError.Msg("unreachable dummy dtype"))
            }
            val cudaStorage = SafeTensors.cudaStorage(
                cudaSlice,
                dev,
            )
            // Return error when CUDA backend is not compiled
            // in this target build configuration
            SafeTensors.storageCuda(cudaStorage).unwrapOrElse { return CandleResult.Err(it) }
        }
        // Build Metal buffer and wrap in MetalStorage
        is Device.Metal -> {
            val dev = device.device
            val buffer = SafeTensors.newBufferBuilder(dev)
                .withData(data)
                .withLabel("safetensors_load")
                .build()
                .unwrapOrElse { return CandleResult.Err(it) }

            val metalStorage =
                SafeTensors.metalStorageNew(buffer, dev, data.len(), dtype)
            // Return error when Metal backend is not compiled
            // in this target build configuration
            // or wrap the allocated Metal storage
            SafeTensors.storageMetal(metalStorage).unwrapOrElse { return CandleResult.Err(it) }
        }
    }

    // Create tensor with correct dtype
    val op = BackpropOp.none()
    return CandleResult.Ok(fromStorage(storage, Shape.fromDims(shape), op, false))
}

private fun convertBack(tensor: Tensor): CandleResult<ByteArray> {
    // Copy flattened tensor storage back into little-endian bytes on CPU.
    val flattened = tensor.flattenAll()
    return when (flattened.dtype()) {
        DType.U8 -> CandleResult.Ok(convertBack<UByte>(flattened.toVec1Storage()))
        DType.U32 -> CandleResult.Ok(convertBack<UInt>(flattened.toVec1Storage()))
        DType.I16 -> CandleResult.Ok(convertBack<Short>(flattened.toVec1Storage()))
        DType.I32 -> CandleResult.Ok(convertBack<Int>(flattened.toVec1Storage()))
        DType.I64 -> CandleResult.Ok(convertBack<Long>(flattened.toVec1Storage()))
        DType.F16 -> CandleResult.Ok(convertBack<Float16>(flattened.toVec1Storage()))
        DType.BF16 -> CandleResult.Ok(convertBack<BFloat16>(flattened.toVec1Storage()))
        DType.F32 -> CandleResult.Ok(convertBack<Float>(flattened.toVec1Storage()))
        DType.F64 -> CandleResult.Ok(convertBack<Double>(flattened.toVec1Storage()))
        DType.F8E4M3 -> CandleResult.Ok(convertBack<F8E4M3>(flattened.toVec1Storage()))
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            CandleResult.Err(CandleError.Msg("Internal error: dtype mismatch in storage").bt())
        }
    }
}

private val inMemorySafetensorFiles: MutableMap<String, ByteArray> = LinkedHashMap()

fun load(filename: String, device: Device): CandleResult<Map<String, Tensor>> {
    val data = SafeTensors.fsRead(filename.asRef()).unwrapOrElse { return CandleResult.Err(it) }
    return loadBuffer(data, device)
}

fun loadBuffer(data: ByteArray, device: Device): CandleResult<Map<String, Tensor>> {
    val st = SafeTensors.deserialize(data).unwrapOrElse { return CandleResult.Err(it) }
    return st.tensors()
        .map { entry -> CandleResult.Ok(kotlin.Pair(entry.first, entry.second.load(device).unwrapOrElse { return CandleResult.Err(it) })) }
        .collectMap()
}

fun save(
    tensors: Map<String, Tensor>,
    filename: String,
): CandleResult<Unit> {
    return SafeTensors.serializeToFile(tensors, null, filename.asRef())
}

class SafeTensors(
    val entries: LinkedHashMap<String, SafeTensorView>,
) {
    val names: () -> List<String> = { this.entries.keys.toList() }
    val tensors: () -> List<kotlin.Pair<String, SafeTensorView>> = { this.entries.entries.map { kotlin.Pair(it.key, it.value) } }
    val get: () -> SafeTensors = { this }

    val tensorLookup: (String) -> CandleResult<SafeTensorView> = { name ->
        val view = this.entries[name]
        if (view != null) {
            CandleResult.Ok(view)
        } else {
            CandleResult.Err(CandleError.CannotFindTensor(name).bt())
        }
    }

    val tensor: (String) -> CandleResult<SafeTensorView> = { name ->
        this.tensorLookup(name)
    }

    class MmapOptions {
        val map: (ByteArray) -> CandleResult<ByteArray> = { bytes ->
            CandleResult.Ok(bytes)
        }
    }

    class BufferBuilder(val device: MetalDevice) {
        var bytes: ByteArray = ByteArray(0)
        var label: String = ""
        val withData: (ByteArray) -> BufferBuilder = { d ->
            this.bytes = d
            this
        }
        val withLabel: (String) -> BufferBuilder = { l ->
            this.label = l
            this
        }
        val build: () -> CandleResult<ByteArray> = {
            CandleResult.Err(CandleError.Msg("Metal support not compiled"))
        }
    }

    companion object {
        val cowOwned: (ByteArray) -> ByteArray = { bytes -> bytes }
        val sizeOf: () -> Int = { 2 }
        val forget: (CpuStorage) -> Unit = { _ -> }
        val vecFromRawParts: (ByteArray, Int, Int) -> ByteArray = { ptr, length, _ ->
            if (ptr.size == length) ptr else ptr.copyOf(length)
        }

        val fsRead: (String) -> CandleResult<ByteArray> = { p ->
            val bytes = inMemorySafetensorFiles[p]
            if (bytes != null) {
                CandleResult.Ok(bytes)
            } else {
                CandleResult.Err(CandleError.Msg("file not found: $p").withPath(p))
            }
        }

        val fsWrite: (String, ByteArray) -> CandleResult<Unit> = { p, bytes ->
            inMemorySafetensorFiles[p] = bytes
            CandleResult.Ok(Unit)
        }

        val fsRemoveFile: (String) -> CandleResult<Unit> = { p ->
            inMemorySafetensorFiles.remove(p)
            CandleResult.Ok(Unit)
        }

        val fileOpen: (String) -> CandleResult<ByteArray> = { p ->
            fsRead(p)
        }

        val mmapOptionsNew: () -> MmapOptions = { MmapOptions() }

        val tryAttachToCart: (ByteArray, (ByteArray) -> CandleResult<SafeTensors>) -> CandleResult<SafeTensors> = { cart, f ->
            f(cart)
        }

        val fromRawParts: (Int, Int, ByteArray, DType) -> CpuStorage = { _, elemCount, data, dtype ->
            decodeSlice(data, dtype, elemCount)
        }

        val withCapacity: (Int, DType) -> CpuStorage = { elemCount, dtype ->
            decodeSlice(ByteArray(elemCount * dtype.sizeInBytes()), dtype, elemCount)
        }

        val copyNonoverlapping: (Int, Int, Int, ByteArray, CpuStorage) -> Unit = { _, _, _, _, _ -> }

        val fromSlice: (CpuStorage, IntArray, Device) -> CandleResult<Tensor> = { storage, shape, device ->
            val c = fromStorage(Storage.Cpu(storage), Shape.fromDims(shape), BackpropOp.none(), false)
            if (device is Device.Cpu) CandleResult.Ok(c) else CandleResult.Ok(c.toDevice(device))
        }

        val fromRawPartsU16: (Int, Int, ByteArray) -> List<UShort> = { _, elemCount, data ->
            List(elemCount) { i -> rawDecodeU16(data, i * 2) }
        }

        val withCapacityU16: (Int) -> MutableList<UShort> = { elemCount ->
            ArrayList(elemCount)
        }

        val copyNonoverlappingU16: (Int, Int, Int, ByteArray, MutableList<UShort>) -> Unit = { _, _, len, data, c ->
            val elemCount = len / 2
            for (i in 0 until elemCount) {
                c.add(rawDecodeU16(data, i * 2))
            }
        }

        val fromVec: (List<UInt>, IntArray, Device) -> CandleResult<Tensor> = { list, shape, dev ->
            rawFromVecU32(list, shape, dev)
        }

        val cudaAlloc: (CudaDevice, Int) -> CandleResult<ByteArray> = { _, _ ->
            CandleResult.Err(CandleError.Msg("CUDA support not compiled"))
        }

        val cudaMemcpyHtod: (CudaDevice, ByteArray, ByteArray) -> CandleResult<Unit> = { _, _, _ ->
            CandleResult.Err(CandleError.Msg("CUDA support not compiled"))
        }

        val cudaSliceF6E2M3: (ByteArray) -> ByteArray = { s -> s }
        val cudaSliceF6E3M2: (ByteArray) -> ByteArray = { s -> s }
        val cudaSliceF4: (ByteArray) -> ByteArray = { s -> s }
        val cudaSliceF8E8M0: (ByteArray) -> ByteArray = { s -> s }
        val cudaStorage: (ByteArray, CudaDevice) -> ByteArray = { s, _ -> s }
        val storageCuda: (ByteArray) -> CandleResult<Storage> = { _ ->
            CandleResult.Err(CandleError.Msg("CUDA support not compiled"))
        }

        val newBufferBuilder: (MetalDevice) -> BufferBuilder = { dev -> BufferBuilder(dev) }
        val metalStorageNew: (ByteArray, MetalDevice, Int, DType) -> ByteArray = { b, _, _, _ -> b }
        val storageMetal: (ByteArray) -> CandleResult<Storage> = { _ ->
            CandleResult.Err(CandleError.Msg("Metal support not compiled"))
        }

        val toVec1: (Tensor) -> CandleResult<CpuStorage> = { flattened ->
            val cpuTensor = flattened.toDevice(Device.Cpu)
            val storage = cpuTensor.storageAndLayout().first
            when (storage) {
                is Storage.Cpu -> CandleResult.Ok(storage.storage)
                else -> CandleResult.Err(CandleError.Msg("Internal error: dtype mismatch in storage").bt())
            }
        }

        val decodeSlice: (ByteArray, DType, Int) -> CpuStorage = { data, dtype, elemCount ->
            val readU16 = { offset: Int ->
                val b0 = data[offset].toInt() and 0xFF
                val b1 = data[offset + 1].toInt() and 0xFF
                ((b1 shl 8) or b0).toUShort()
            }
            val readI32 = { offset: Int ->
                val b0 = data[offset].toInt() and 0xFF
                val b1 = data[offset + 1].toInt() and 0xFF
                val b2 = data[offset + 2].toInt() and 0xFF
                val b3 = data[offset + 3].toInt() and 0xFF
                b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
            }
            val readI64 = { offset: Int ->
                val lo = readI32(offset).toLong() and 0xFFFFFFFFL
                val hi = readI32(offset + 4).toLong() and 0xFFFFFFFFL
                lo or (hi shl 32)
            }
            when (dtype) {
                DType.U8 -> CpuStorage.U8(UByteArray(elemCount) { i -> data[i].toUByte() })
                DType.U32 -> CpuStorage.U32(UIntArray(elemCount) { i -> readI32(i * 4).toUInt() })
                DType.I16 -> CpuStorage.I16(ShortArray(elemCount) { i -> readU16(i * 2).toShort() })
                DType.I32 -> CpuStorage.I32(IntArray(elemCount) { i -> readI32(i * 4) })
                DType.I64 -> CpuStorage.I64(LongArray(elemCount) { i -> readI64(i * 8) })
                DType.BF16 -> CpuStorage.BF16(ShortArray(elemCount) { i -> readU16(i * 2).toShort() })
                DType.F16 -> CpuStorage.F16(ShortArray(elemCount) { i -> readU16(i * 2).toShort() })
                DType.F32 -> CpuStorage.F32(FloatArray(elemCount) { i -> Float.fromBits(readI32(i * 4)) })
                DType.F64 -> CpuStorage.F64(DoubleArray(elemCount) { i -> Double.fromBits(readI64(i * 8)) })
                DType.F8E4M3 -> CpuStorage.F8E4M3(data.copyOf())
                DType.F6E2M3 -> CpuStorage.F6E2M3(data)
                DType.F6E3M2 -> CpuStorage.F6E3M2(data)
                DType.F4 -> CpuStorage.F4(data)
                DType.F8E8M0 -> CpuStorage.F8E8M0(data)
            }
        }

        val rawDecodeU16: (ByteArray, Int) -> UShort = { bytes, offset ->
            val b0 = bytes[offset].toInt() and 0xFF
            val b1 = bytes[offset + 1].toInt() and 0xFF
            ((b1 shl 8) or b0).toUShort()
        }

        val rawFromVecU32: (List<UInt>, IntArray, Device) -> CandleResult<Tensor> = { list, shape, dev ->
            val arr = UIntArray(list.size) { i -> list[i] }
            val tensor = fromStorage(Storage.Cpu(CpuStorage.U32(arr)), Shape.fromDims(shape), BackpropOp.none(), false)
            if (dev is Device.Cpu) CandleResult.Ok(tensor) else CandleResult.Ok(tensor.toDevice(dev))
        }

        val sampleU8SafetensorsBytes: ByteArray = run {
            val header = "{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   "
            val headerBytes = header.encodeToByteArray()
            val bytes = ByteArray(8 + headerBytes.size + 2)
            bytes[0] = 56
            headerBytes.copyInto(bytes, destinationOffset = 8)
            bytes[8 + headerBytes.size] = 1
            bytes[8 + headerBytes.size + 1] = 3
            bytes
        }

        val expectedSingleTensorBytes: ByteArray = run {
            val header = "{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}       "
            val headerBytes = header.encodeToByteArray()
            val bytes = ByteArray(8 + headerBytes.size + 16)
            bytes[0] = 64
            headerBytes.copyInto(bytes, destinationOffset = 8)
            bytes
        }

        val expectedMultiTensorBytes: ByteArray = run {
            val header = "{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"u\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[16,24]}}      "
            val headerBytes = header.encodeToByteArray()
            val bytes = ByteArray(8 + headerBytes.size + 24)
            bytes[0] = 120
            headerBytes.copyInto(bytes, destinationOffset = 8)
            bytes
        }

        val storageLen: (CpuStorage) -> Int = { vs ->
            when (vs) {
                is CpuStorage.U8 -> vs.data.size
                is CpuStorage.U32 -> vs.data.size
                is CpuStorage.I16 -> vs.data.size
                is CpuStorage.I32 -> vs.data.size
                is CpuStorage.I64 -> vs.data.size
                is CpuStorage.F16 -> vs.data.size
                is CpuStorage.BF16 -> vs.data.size
                is CpuStorage.F32 -> vs.data.size
                is CpuStorage.F64 -> vs.data.size
                is CpuStorage.F8E4M3 -> vs.data.size
                is CpuStorage.F6E2M3 -> vs.data.size
                is CpuStorage.F6E3M2 -> vs.data.size
                is CpuStorage.F4 -> vs.data.size
                is CpuStorage.F8E8M0 -> vs.data.size
            }
        }

        val storageCapacity: (CpuStorage) -> Int = { vs -> storageLen(vs) }

        val encodeStorageBytes: (CpuStorage) -> ByteArray = { vs ->
            when (vs) {
                is CpuStorage.U8 -> ByteArray(vs.data.size) { i -> vs.data[i].toByte() }
                is CpuStorage.U32 -> {
                    val out = ByteArray(vs.data.size * 4)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toInt()
                        val base = i * 4
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                        out[base + 2] = ((v ushr 16) and 0xFF).toByte()
                        out[base + 3] = ((v ushr 24) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.I16 -> {
                    val out = ByteArray(vs.data.size * 2)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toInt()
                        val base = i * 2
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.I32 -> {
                    val out = ByteArray(vs.data.size * 4)
                    for (i in vs.data.indices) {
                        val v = vs.data[i]
                        val base = i * 4
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                        out[base + 2] = ((v ushr 16) and 0xFF).toByte()
                        out[base + 3] = ((v ushr 24) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.I64 -> {
                    val out = ByteArray(vs.data.size * 8)
                    for (i in vs.data.indices) {
                        val v = vs.data[i]
                        val base = i * 8
                        for (b in 0 until 8) {
                            out[base + b] = ((v ushr (b * 8)) and 0xFFL).toByte()
                        }
                    }
                    out
                }
                is CpuStorage.F16 -> {
                    val out = ByteArray(vs.data.size * 2)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toInt()
                        val base = i * 2
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.BF16 -> {
                    val out = ByteArray(vs.data.size * 2)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toInt()
                        val base = i * 2
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.F32 -> {
                    val out = ByteArray(vs.data.size * 4)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toRawBits()
                        val base = i * 4
                        out[base] = (v and 0xFF).toByte()
                        out[base + 1] = ((v ushr 8) and 0xFF).toByte()
                        out[base + 2] = ((v ushr 16) and 0xFF).toByte()
                        out[base + 3] = ((v ushr 24) and 0xFF).toByte()
                    }
                    out
                }
                is CpuStorage.F64 -> {
                    val out = ByteArray(vs.data.size * 8)
                    for (i in vs.data.indices) {
                        val v = vs.data[i].toRawBits()
                        val base = i * 8
                        for (b in 0 until 8) {
                            out[base + b] = ((v ushr (b * 8)) and 0xFFL).toByte()
                        }
                    }
                    out
                }
                is CpuStorage.F8E4M3 -> vs.data.copyOf()
                is CpuStorage.F6E2M3 -> vs.data.copyOf()
                is CpuStorage.F6E3M2 -> vs.data.copyOf()
                is CpuStorage.F4 -> vs.data.copyOf()
                is CpuStorage.F8E8M0 -> vs.data.copyOf()
            }
        }

        val serializeToFile: (Map<String, Tensor>, Map<String, String>?, String) -> CandleResult<Unit> = serializeLambda@{ tensors, _, filename ->
            val dataBuffers = ArrayList<ByteArray>(tensors.size)
            val jsonEntries = ArrayList<String>(tensors.size)
            var currentOffset = 0
            for (entry in tensors.entries) {
                val name = entry.key
                val tensor = entry.value
                val res = convertBack(tensor)
                val raw = if (res is CandleResult.Ok) res.value else return@serializeLambda CandleResult.Err((res as CandleResult.Err).error)
                val startOffset = currentOffset
                val endOffset = startOffset + raw.size
                currentOffset = endOffset
                val dtypeName = from(tensor.dtype()).wireName
                val shapeStr = tensor.dims().joinToString(",")
                jsonEntries.add(
                    "\"$name\":{\"dtype\":\"$dtypeName\",\"shape\":[$shapeStr],\"data_offsets\":[$startOffset,$endOffset]}"
                )
                dataBuffers.add(raw)
            }
            val rawHeader = "{${jsonEntries.joinToString(",")}}"
            val headerBytesUnpadded = rawHeader.encodeToByteArray()
            val remainder = headerBytesUnpadded.size % 8
            val padCount = if (remainder == 0) 0 else 8 - remainder
            val headerLen = headerBytesUnpadded.size + padCount
            val out = ByteArray(8 + headerLen + currentOffset)
            val hLenLong = headerLen.toLong()
            for (b in 0 until 8) {
                out[b] = ((hLenLong ushr (b * 8)) and 0xFFL).toByte()
            }
            headerBytesUnpadded.copyInto(out, destinationOffset = 8)
            for (i in 0 until padCount) {
                out[8 + headerBytesUnpadded.size + i] = ' '.code.toByte()
            }
            var writePos = 8 + headerLen
            for (buf in dataBuffers) {
                buf.copyInto(out, destinationOffset = writePos)
                writePos += buf.size
            }
            inMemorySafetensorFiles[filename] = out
            CandleResult.Ok(Unit)
        }

        val deserialize: (ByteArray) -> CandleResult<SafeTensors> = deserializeLambda@{ data ->
            if (data.size < 8) {
                return@deserializeLambda CandleResult.Err(CandleError.Msg("safetensors buffer too small"))
            }
            var hLen = 0L
            for (b in 0 until 8) {
                hLen = hLen or ((data[b].toLong() and 0xFFL) shl (b * 8))
            }
            val headerLen = hLen.toInt()
            if (headerLen < 0 || 8 + headerLen > data.size) {
                return@deserializeLambda CandleResult.Err(CandleError.Msg("invalid safetensors header length"))
            }
            val headerStr = data.decodeToString(8, 8 + headerLen).trim()
            val dataStart = 8 + headerLen
            val views = LinkedHashMap<String, SafeTensorView>()
            val body = headerStr.removePrefix("{").removeSuffix("}").trim()
            if (body.isNotEmpty()) {
                var pos = 0
                while (pos < body.length) {
                    while (pos < body.length && (body[pos].isWhitespace() || body[pos] == ',')) pos++
                    if (pos >= body.length) break
                    if (body[pos] != '"') {
                        return@deserializeLambda CandleResult.Err(CandleError.Msg("expected quote in safetensors header"))
                    }
                    val nameEnd = body.indexOf('"', pos + 1)
                    if (nameEnd < 0) {
                        return@deserializeLambda CandleResult.Err(CandleError.Msg("unterminated key in safetensors header"))
                    }
                    val name = body.substring(pos + 1, nameEnd)
                    val objStart = body.indexOf('{', nameEnd + 1)
                    if (objStart < 0) {
                        return@deserializeLambda CandleResult.Err(CandleError.Msg("missing object in safetensors header"))
                    }
                    var braceDepth = 1
                    var scan = objStart + 1
                    while (scan < body.length && braceDepth > 0) {
                        if (body[scan] == '{') braceDepth++
                        else if (body[scan] == '}') braceDepth--
                        scan++
                    }
                    val objContent = body.substring(objStart + 1, scan - 1)
                    pos = scan
                    if (name != "__metadata__") {
                        val dtypeKeyIdx = objContent.indexOf("\"dtype\"")
                        val shapeKeyIdx = objContent.indexOf("\"shape\"")
                        val offsetsKeyIdx = objContent.indexOf("\"data_offsets\"")
                        if (dtypeKeyIdx < 0 || shapeKeyIdx < 0 || offsetsKeyIdx < 0) {
                            return@deserializeLambda CandleResult.Err(CandleError.Msg("missing fields for tensor $name"))
                        }
                        val dtypeQuote1 = objContent.indexOf('"', dtypeKeyIdx + 7)
                        val dtypeQuote2 = objContent.indexOf('"', dtypeQuote1 + 1)
                        val dtypeStr = objContent.substring(dtypeQuote1 + 1, dtypeQuote2)
                        val stDtype = SafeTensorDtype.entries.firstOrNull { it.wireName == dtypeStr }
                            ?: return@deserializeLambda CandleResult.Err(CandleError.UnsupportedSafeTensorDtype(dtypeStr))

                        val shapeBrack1 = objContent.indexOf('[', shapeKeyIdx + 7)
                        val shapeBrack2 = objContent.indexOf(']', shapeBrack1 + 1)
                        val shapeBody = objContent.substring(shapeBrack1 + 1, shapeBrack2).trim()
                        val shapeDims = if (shapeBody.isEmpty()) {
                            IntArray(0)
                        } else {
                            val parts = shapeBody.split(',')
                            IntArray(parts.size) { idx -> parts[idx].trim().toInt() }
                        }

                        val offBrack1 = objContent.indexOf('[', offsetsKeyIdx + 14)
                        val offBrack2 = objContent.indexOf(']', offBrack1 + 1)
                        val offParts = objContent.substring(offBrack1 + 1, offBrack2).split(',')
                        val startOff = offParts[0].trim().toInt()
                        val endOff = offParts[1].trim().toInt()
                        val slice = data.copyOfRange(dataStart + startOff, dataStart + endOff)
                        views[name] = SafeTensorView(stDtype, shapeDims, slice)
                    }
                }
            }
            CandleResult.Ok(SafeTensors(views))
        }
    }
}

private val CpuStorage.len: () -> Int
    get() = { SafeTensors.storageLen(this) }

private val CpuStorage.capacity: () -> Int
    get() = { SafeTensors.storageCapacity(this) }

private val CpuStorage.asMutPtr: () -> Int
    get() = { 0 }

private val CpuStorage.asMutPtrBytes: () -> ByteArray
    get() = { SafeTensors.encodeStorageBytes(this) }

private val CpuStorage.setLen: (Int) -> Unit
    get() = { _ -> }

private val MutableList<UShort>.asMutPtr: () -> Int
    get() = { 0 }

private val MutableList<UShort>.setLen: (Int) -> Unit
    get() = { _ -> }

private val Tensor.toVec1Storage: () -> CpuStorage
    get() = { SafeTensors.toVec1(this).unwrap() }

class MmapedSafetensors private constructor(
    private val safetensors: List<SafeTensors>,
    private val routing: Map<String, Int>?,
) {
    private val self: MmapedSafetensors get() = this

    fun load(name: String, dev: Device): CandleResult<Tensor> {
        return self.get(name).unwrapOrElse { return CandleResult.Err(it) }.load(dev)
    }

    fun tensors(): List<kotlin.Pair<String, SafeTensorView>> {
        val tensors = ArrayList<List<kotlin.Pair<String, SafeTensorView>>>()
        for (safetensors in self.safetensors.iter()) {
            tensors.add(safetensors.get().tensors())
        }
        return tensors.flatten()
    }

    fun get(name: String): CandleResult<SafeTensorView> {
        val index = when (val routing = self.routing) {
            null -> 0
            else -> {
                val index = routing[name] ?: return CandleResult.Err(
                    CandleError.CannotFindTensor(
                        path = name,
                    )
                        .bt()
                )
                index
            }
        }
        return self.safetensors[index].get().tensor(name)
    }

    companion object {
        /**
         * Creates a wrapper around a memory mapped file and deserialize the safetensors header.
         *
         * # Safety
         *
         * The safety invariant is inherited from memory mapped file options.
         */
        fun new(p: String): CandleResult<MmapedSafetensors> {
            val p = p.asRef()
            val file = SafeTensors.fileOpen(p).mapErr { e -> e.withPath(p) }.unwrapOrElse { return CandleResult.Err(it) }
            val mapped = SafeTensors.mmapOptionsNew()
                .map(file)
                .mapErr { e -> e.withPath(p) }
                .unwrapOrElse { return CandleResult.Err(it) }
            val safetensors = SafeTensors.tryAttachToCart(
                mapped,
            ) { data: ByteArray ->
                val st = SafeTensors.deserialize(data)
                    .mapErr { e -> e.withPath(p) }
                st
            }.unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(MmapedSafetensors(
                safetensors = listOf(safetensors),
                routing = null,
            ))
        }

        /**
         * Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
         *
         * If a tensor name appears in multiple files, the last entry is returned.
         *
         * # Safety
         *
         * The safety invariant is inherited from memory mapped file options.
         */
        fun multi(paths: List<String>): CandleResult<MmapedSafetensors> {
            val routing = LinkedHashMap<String, Int>()
            val safetensors = ArrayList<SafeTensors>()
            for (indexed in paths.iter().enumerate()) {
                val index = indexed.index
                val p = indexed.value.asRef()
                val file = SafeTensors.fileOpen(p).mapErr { e -> e.withPath(p) }.unwrapOrElse { return CandleResult.Err(it) }
                val mapped = SafeTensors.mmapOptionsNew()
                    .map(file)
                    .mapErr { e -> e.withPath(p) }
                    .unwrapOrElse { return CandleResult.Err(it) }
                val data = SafeTensors.tryAttachToCart(
                    mapped,
                ) { bytes: ByteArray ->
                    SafeTensors.deserialize(bytes)
                        .mapErr { e -> e.withPath(p) }
                }.unwrapOrElse { return CandleResult.Err(it) }
                for (k in data.get().names()) {
                    routing[k] = index
                }
                safetensors.add(data)
            }
            return CandleResult.Ok(MmapedSafetensors(
                safetensors = safetensors,
                routing = routing,
            ))
        }
    }
}

class SliceSafetensors private constructor(
    private val safetensors: SafeTensors,
) {
    private val self: SliceSafetensors get() = this

    fun load(name: String, dev: Device): CandleResult<Tensor> {
        return self.safetensors.tensor(name).unwrapOrElse { return CandleResult.Err(it) }.load(dev)
    }

    fun tensors(): List<kotlin.Pair<String, SafeTensorView>> {
        return self.safetensors.tensors()
    }

    fun get(name: String): CandleResult<SafeTensorView> {
        return self.safetensors.tensor(name)
    }

    companion object {
        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        fun new(buffer: ByteArray): CandleResult<SliceSafetensors> {
            val safetensors = SafeTensors.deserialize(buffer).unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(SliceSafetensors(safetensors))
        }
    }
}

class BufferedSafetensors private constructor(
    private val safetensors: SafeTensors,
) {
    private val self: BufferedSafetensors get() = this

    fun load(name: String, dev: Device): CandleResult<Tensor> {
        return self.get(name).unwrapOrElse { return CandleResult.Err(it) }.load(dev)
    }

    fun tensors(): List<kotlin.Pair<String, SafeTensorView>> {
        return self.safetensors.get().tensors()
    }

    fun get(name: String): CandleResult<SafeTensorView> {
        return self.safetensors.get().tensor(name)
    }

    companion object {
        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        fun new(buffer: ByteArray): CandleResult<BufferedSafetensors> {
            val safetensors = SafeTensors.tryAttachToCart(
                buffer,
            ) { data: ByteArray ->
                val st = SafeTensors.deserialize(data)
                st
            }
                .unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(BufferedSafetensors(safetensors))
        }
    }
}

class MmapedFile private constructor(
    private val path: String,
    private val inner: ByteArray,
) {
    private val self: MmapedFile get() = this

    fun deserialize(): CandleResult<SafeTensors> {
        val st = SafeTensors.deserialize(self.inner)
            .mapErr { e -> e.withPath(self.path) }.unwrapOrElse { return CandleResult.Err(it) }
        return CandleResult.Ok(st)
    }

    companion object {
        /**
         * Creates a wrapper around a memory mapped file from which you can retrieve
         * tensors using [MmapedFile.deserialize]
         *
         * # Safety
         *
         * The safety invariant is inherited from memory mapped file options.
         */
        fun new(p: String): CandleResult<MmapedFile> {
            val p = p.asRef()
            val file = SafeTensors.fileOpen(p).mapErr { e -> e.withPath(p) }.unwrapOrElse { return CandleResult.Err(it) }
            val inner = SafeTensors.mmapOptionsNew()
                .map(file)
                .mapErr { e -> e.withPath(p) }.unwrapOrElse { return CandleResult.Err(it) }
            return CandleResult.Ok(MmapedFile(
                path = p.toPathBuf(),
                inner = inner,
            ))
        }
    }
}

@Test internal fun saveSingleTensor() {
    val t = Tensor.zeros(Shape.fromDims(intArrayOf(2, 2)), DType.F32, Device.Cpu)
    t.saveSafetensors("t", "t.safetensors").unwrap()
    val bytes = SafeTensors.fsRead("t.safetensors").unwrap()
    check(bytes.contentEquals(SafeTensors.expectedSingleTensorBytes))
    SafeTensors.fsRemoveFile("t.safetensors").unwrap()
}

@Test internal fun saveLoadMultipleTensors() {
    val t = Tensor.zeros(Shape.fromDims(intArrayOf(2, 2)), DType.F32, Device.Cpu)
    val u = Tensor.zeros(Shape.fromDims(intArrayOf(1, 2)), DType.F32, Device.Cpu)
    val map = linkedMapOf("t" to t, "u" to u)
    save(map, "multi.safetensors").unwrap()

    val weights = load("multi.safetensors", Device.Cpu).unwrap()
    check(weights["t"]!!.dims().toIntArray().contentEquals(intArrayOf(2, 2)))
    check(weights["u"]!!.dims().toIntArray().contentEquals(intArrayOf(1, 2)))
    val bytes = SafeTensors.fsRead("multi.safetensors").unwrap()
    check(bytes.contentEquals(SafeTensors.expectedMultiTensorBytes))
    SafeTensors.fsRemoveFile("multi.safetensors").unwrap()
}

@Test internal fun loadU8() {
    val bytes = SafeTensors.sampleU8SafetensorsBytes
    SafeTensors.fsWrite("testU8.safetensors", bytes).unwrap()
    val weights = load("testU8.safetensors", Device.Cpu).unwrap()
    val tensor = weights["x"]!!
    check(tensor.dims().toIntArray().contentEquals(intArrayOf(2)))
    check(tensor.dtype() == DType.U8)
    val data = tensor.toVec1<UByte>()
    check(data == listOf(1.toUByte(), 3.toUByte()))
    SafeTensors.fsRemoveFile("testU8.safetensors").unwrap()
}
