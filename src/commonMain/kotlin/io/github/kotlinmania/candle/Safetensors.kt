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
}

class SafeTensorView(
    val dtype: SafeTensorDtype,
    val shape: IntArray,
    val data: ByteArray,
) : Load {
    override fun load(device: Device): CandleResult<Tensor> = convert(this, device)
}

fun from(value: DType): SafeTensorDtype = when (value) {
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

fun tryFrom(value: SafeTensorDtype): CandleResult<DType> = when (value) {
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

class TensorSafeView(private val tensor: Tensor) {
    fun dtype(): SafeTensorDtype = from(this.tensor.dtype())

    fun shape(): IntArray = this.tensor.shape().dims()

    fun data(): ByteArray {
        // This copies data from GPU to CPU.
        return convertBack(this.tensor).unwrap()
    }

    fun dataLen(): Int {
        val n = this.tensor.shape().elemCount()
        val bytesPerElement = this.tensor.dtype().sizeInBytes()
        return n * bytesPerElement
    }
}

class TensorRefSafeView(private val tensor: Tensor) {
    fun dtype(): SafeTensorDtype = from(this.tensor.dtype())

    fun shape(): IntArray = this.tensor.dims()

    fun data(): ByteArray {
        // This copies data from GPU to CPU.
        return convertBack(this.tensor).unwrap()
    }

    fun dataLen(): Int {
        val n = this.tensor.dims().fold(1) { acc, d -> acc * d }
        val bytesPerElement = this.tensor.dtype().sizeInBytes()
        return n * bytesPerElement
    }
}

fun Tensor.saveSafetensors(name: String, filename: String): CandleResult<Unit> {
    val data = linkedMapOf(name to this.clone())
    return save(data, filename)
}

private fun <T : WithDType> convertSlice(
    data: ByteArray,
    dtype: DType,
    shape: IntArray,
    device: Device,
): CandleResult<Tensor> {
    // Check alignment and decode little-endian raw bytes into typed storage.
    val sizeInBytes = dtype.sizeInBytes()
    val elemCount = data.size / sizeInBytes
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
    val storage = when (dtype) {
        DType.U8 -> CpuStorage.U8(UByteArray(elemCount) { i -> data[i].toUByte() })
        DType.U32 -> CpuStorage.U32(UIntArray(elemCount) { i -> readI32(i * 4).toUInt() })
        DType.I16 -> CpuStorage.I16(ShortArray(elemCount) { i -> readU16(i * 2).toShort() })
        DType.I32 -> CpuStorage.I32(IntArray(elemCount) { i -> readI32(i * 4) })
        DType.I64 -> CpuStorage.I64(LongArray(elemCount) { i -> readI64(i * 8) })
        DType.BF16 -> CpuStorage.BF16(Array(elemCount) { i -> BF16.fromBits(readU16(i * 2)) })
        DType.F16 -> CpuStorage.F16(Array(elemCount) { i -> F16.fromBits(readU16(i * 2)) })
        DType.F32 -> CpuStorage.F32(FloatArray(elemCount) { i -> Float.fromBits(readI32(i * 4)) })
        DType.F64 -> CpuStorage.F64(DoubleArray(elemCount) { i -> Double.fromBits(readI64(i * 8)) })
        DType.F8E4M3 -> CpuStorage.F8E4M3(Array(elemCount) { i -> F8E4M3.fromBits(data[i].toUByte()) })
        DType.F6E2M3 -> CpuStorage.F6E2M3(data.toUByteArray())
        DType.F6E3M2 -> CpuStorage.F6E3M2(data.toUByteArray())
        DType.F4 -> CpuStorage.F4(data.toUByteArray())
        DType.F8E8M0 -> CpuStorage.F8E8M0(data.toUByteArray())
    }
    val c = fromStorage(Storage.Cpu(storage), Shape.fromDims(shape), BackpropOp.none(), false)
    return if (device is Device.Cpu) {
        CandleResult.Ok(c)
    } else {
        c.toDevice(device)
    }
}

private fun <T, U> convertSliceWithCast(
    data: ByteArray,
    sizeInBytes: Int,
    decode: (ByteArray, Int) -> T,
    shape: IntArray,
    device: Device,
    conv: (T) -> CandleResult<U>,
    fromVec: (List<U>, IntArray, Device) -> CandleResult<Tensor>,
): CandleResult<Tensor> {
    // Specify element type explicitly so the target vector fits all converted weights.
    val elemCount = data.size / sizeInBytes
    val c = ArrayList<U>(elemCount)
    for (i in 0 until elemCount) {
        val t = decode(data, i * sizeInBytes)
        when (val res = conv(t)) {
            is CandleResult.Ok -> c.add(res.value)
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
    }
    return fromVec(c, shape, device)
}

private fun <T, U> convertWithCast(
    view: SafeTensorView,
    sizeInBytes: Int,
    decode: (ByteArray, Int) -> T,
    device: Device,
    conv: (T) -> CandleResult<U>,
    fromVec: (List<U>, IntArray, Device) -> CandleResult<Tensor>,
): CandleResult<Tensor> =
    convertSliceWithCast(view.data, sizeInBytes, decode, view.shape, device, conv, fromVec)

private fun <T : WithDType> convert(
    view: SafeTensorView,
    dtype: DType,
    device: Device,
): CandleResult<Tensor> =
    convertSlice<T>(view.data, dtype, view.shape, device)

private fun convertBack(vs: CpuStorage): ByteArray {
    // Every element is at least 1 byte, so reinterpreting typed storage as raw bytes is aligned.
    return when (vs) {
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
                val v = vs.data[i].toBits().toInt()
                val base = i * 2
                out[base] = (v and 0xFF).toByte()
                out[base + 1] = ((v ushr 8) and 0xFF).toByte()
            }
            out
        }
        is CpuStorage.BF16 -> {
            val out = ByteArray(vs.data.size * 2)
            for (i in vs.data.indices) {
                val v = vs.data[i].toBits().toInt()
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
        is CpuStorage.F8E4M3 -> ByteArray(vs.data.size) { i -> vs.data[i].toBits().toByte() }
        is CpuStorage.F6E2M3 -> vs.data.toByteArray()
        is CpuStorage.F6E3M2 -> vs.data.toByteArray()
        is CpuStorage.F4 -> vs.data.toByteArray()
        is CpuStorage.F8E8M0 -> vs.data.toByteArray()
    }
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
        DType.U8 -> convertSlice<U8Scalar>(data, DType.U8, shape, device)
        DType.U32 -> convertSlice<U32Scalar>(data, DType.U32, shape, device)
        DType.I16 -> convertSlice<I16Scalar>(data, DType.I16, shape, device)
        DType.I32 -> convertSlice<I32Scalar>(data, DType.I32, shape, device)
        DType.I64 -> convertSlice<I64Scalar>(data, DType.I64, shape, device)
        DType.BF16 -> convertSlice<BF16>(data, DType.BF16, shape, device)
        DType.F16 -> convertSlice<F16>(data, DType.F16, shape, device)
        DType.F32 -> convertSlice<F32Scalar>(data, DType.F32, shape, device)
        DType.F64 -> convertSlice<F64Scalar>(data, DType.F64, shape, device)
        DType.F8E4M3 -> convertSlice<F8E4M3>(data, DType.F8E4M3, shape, device)
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            // For dummy types, create storage with raw bytes
            val storage = when (device) {
                is Device.Cpu -> {
                    val raw = data.toUByteArray()
                    val cpuStorage = when (dtype) {
                        DType.F6E2M3 -> CpuStorage.F6E2M3(raw)
                        DType.F6E3M2 -> CpuStorage.F6E3M2(raw)
                        DType.F4 -> CpuStorage.F4(raw)
                        DType.F8E8M0 -> CpuStorage.F8E8M0(raw)
                        else -> return CandleResult.Err(CandleError.Msg("unreachable dummy dtype"))
                    }
                    Storage.Cpu(cpuStorage)
                }
                is Device.Cuda -> {
                    return CandleResult.Err(CandleError.Msg("CUDA support not compiled"))
                }
                is Device.Metal -> {
                    return CandleResult.Err(CandleError.Msg("Metal support not compiled"))
                }
            }
            val op = BackpropOp.none()
            CandleResult.Ok(fromStorage(storage, Shape.fromDims(shape), op, false))
        }
    }
}

private fun convert(view: SafeTensorView, device: Device): CandleResult<Tensor> = when (view.dtype) {
    SafeTensorDtype.U8 -> convert<U8Scalar>(view, DType.U8, device)
    SafeTensorDtype.U16 -> {
        val conv = { x: UShort -> CandleResult.Ok(x.toUInt()) }
        convertWithCast(
            view = view,
            sizeInBytes = 2,
            decode = { bytes, offset ->
                val b0 = bytes[offset].toInt() and 0xFF
                val b1 = bytes[offset + 1].toInt() and 0xFF
                ((b1 shl 8) or b0).toUShort()
            },
            device = device,
            conv = conv,
            fromVec = { list, shape, dev ->
                val arr = UIntArray(list.size) { i -> list[i] }
                val tensor = fromStorage(Storage.Cpu(CpuStorage.U32(arr)), Shape.fromDims(shape), BackpropOp.none(), false)
                if (dev is Device.Cpu) CandleResult.Ok(tensor) else tensor.toDevice(dev)
            },
        )
    }
    SafeTensorDtype.U32 -> convert<U32Scalar>(view, DType.U32, device)
    SafeTensorDtype.I16 -> convert<I16Scalar>(view, DType.I16, device)
    SafeTensorDtype.I32 -> convert<I32Scalar>(view, DType.I32, device)
    SafeTensorDtype.I64 -> convert<I64Scalar>(view, DType.I64, device)
    SafeTensorDtype.BF16 -> convert<BF16>(view, DType.BF16, device)
    SafeTensorDtype.F16 -> convert<F16>(view, DType.F16, device)
    SafeTensorDtype.F32 -> convert<F32Scalar>(view, DType.F32, device)
    SafeTensorDtype.F64 -> convert<F64Scalar>(view, DType.F64, device)
    SafeTensorDtype.F8E4M3 -> convert<F8E4M3>(view, DType.F8E4M3, device)
    SafeTensorDtype.F6E2M3,
    SafeTensorDtype.F6E3M2,
    SafeTensorDtype.F4,
    SafeTensorDtype.F8E8M0 -> {
        // For dummy types, we need to handle loading by creating a dummy tensor
        // Since these types don't have actual data representation, we'll create
        // a tensor that indicates it's a dummy type
        convertDummy(view, device)
    }
}

private fun convertDummy(view: SafeTensorView, device: Device): CandleResult<Tensor> {
    // For dummy types, we'll create the appropriate storage variant that preserves
    // both the raw data and the correct dtype
    val dtype = when (view.dtype) {
        SafeTensorDtype.F6E2M3 -> DType.F6E2M3
        SafeTensorDtype.F6E3M2 -> DType.F6E3M2
        SafeTensorDtype.F4 -> DType.F4
        SafeTensorDtype.F8E8M0 -> DType.F8E8M0
        else -> return CandleResult.Err(CandleError.Msg("convertDummy called with non-dummy dtype"))
    }

    // Load the raw bytes
    val data = view.data
    val shape = view.shape

    // Create storage with the appropriate dummy type variant
    val storage = when (device) {
        is Device.Cpu -> {
            val raw = data.toUByteArray()
            val cpuStorage = when (dtype) {
                DType.F6E2M3 -> CpuStorage.F6E2M3(raw)
                DType.F6E3M2 -> CpuStorage.F6E3M2(raw)
                DType.F4 -> CpuStorage.F4(raw)
                DType.F8E8M0 -> CpuStorage.F8E8M0(raw)
                else -> return CandleResult.Err(CandleError.Msg("unreachable dummy dtype"))
            }
            Storage.Cpu(cpuStorage)
        }
        is Device.Cuda -> {
            return CandleResult.Err(CandleError.Msg("CUDA support not compiled"))
        }
        is Device.Metal -> {
            return CandleResult.Err(CandleError.Msg("Metal support not compiled"))
        }
    }

    // Create tensor with correct dtype
    val op = BackpropOp.none()
    return CandleResult.Ok(fromStorage(storage, Shape.fromDims(shape), op, false))
}

private fun convertBack(tensor: Tensor): CandleResult<ByteArray> {
    val flattened = when (val res = tensor.flattenAll()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    return when (flattened.dtype()) {
        DType.U8,
        DType.U32,
        DType.I16,
        DType.I32,
        DType.I64,
        DType.F16,
        DType.BF16,
        DType.F32,
        DType.F64,
        DType.F8E4M3 -> {
            val cpuTensor = when (val res = flattened.toDevice(Device.Cpu)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val storage = cpuTensor.storageAndLayout().first
            when (storage) {
                is Storage.Cpu -> CandleResult.Ok(convertBack(storage.storage))
                else -> CandleResult.Err(CandleError.Msg("Internal error: dtype mismatch in storage").bt())
            }
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
            CandleResult.Err(CandleError.Msg("Internal error: dtype mismatch in storage").bt())
    }
}

private val inMemorySafetensorFiles: MutableMap<String, ByteArray> = LinkedHashMap()

fun load(filename: String, device: Device): CandleResult<Map<String, Tensor>> {
    val data = inMemorySafetensorFiles[filename]
        ?: return CandleResult.Err(CandleError.Msg("file not found: $filename").withPath(filename))
    return loadBuffer(data, device)
}

fun loadBuffer(data: ByteArray, device: Device): CandleResult<Map<String, Tensor>> {
    val st = when (val res = SafeTensors.deserialize(data)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val out = LinkedHashMap<String, Tensor>()
    for ((name, view) in st.tensors) {
        val loaded = when (val res = view.load(device)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        out[name] = loaded
    }
    return CandleResult.Ok(out)
}

fun save(tensors: Map<String, Tensor>, filename: String): CandleResult<Unit> {
    val dataBuffers = ArrayList<ByteArray>(tensors.size)
    val jsonEntries = ArrayList<String>(tensors.size)
    var currentOffset = 0
    for ((name, tensor) in tensors) {
        val raw = when (val res = convertBack(tensor)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
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
    return CandleResult.Ok(Unit)
}

class SafeTensors(
    val entries: LinkedHashMap<String, SafeTensorView>,
) {
    val names: List<String>
        get() = this.entries.keys.toList()

    val tensors: List<Pair<String, SafeTensorView>>
        get() = this.entries.entries.map { it.key to it.value }

    val tensorLookup: (String) -> CandleResult<SafeTensorView> = { name ->
        val view = this.entries[name]
        if (view != null) {
            CandleResult.Ok(view)
        } else {
            CandleResult.Err(CandleError.CannotFindTensor(name).bt())
        }
    }

    companion object {
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

class MmapedSafetensors private constructor(
    private val safetensors: List<SafeTensors>,
    private val routing: Map<String, Int>?,
) {
    fun load(name: String, dev: Device): CandleResult<Tensor> {
        val view = when (val res = this.get(name)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return view.load(dev)
    }

    fun tensors(): List<Pair<String, SafeTensorView>> {
        val tensors = ArrayList<Pair<String, SafeTensorView>>()
        for (st in this.safetensors) {
            tensors.addAll(st.tensors)
        }
        return tensors
    }

    fun get(name: String): CandleResult<SafeTensorView> {
        val index = when (val r = this.routing) {
            null -> 0
            else -> r[name] ?: return CandleResult.Err(CandleError.CannotFindTensor(name).bt())
        }
        return this.safetensors[index].tensorLookup(name)
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
            val file = inMemorySafetensorFiles[p]
                ?: return CandleResult.Err(CandleError.Msg("file not found: $p").withPath(p))
            val safetensors = when (val res = SafeTensors.deserialize(file)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error.withPath(p))
            }
            return CandleResult.Ok(MmapedSafetensors(listOf(safetensors), null))
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
            val safetensors = ArrayList<SafeTensors>(paths.size)
            for ((index, p) in paths.withIndex()) {
                val file = inMemorySafetensorFiles[p]
                    ?: return CandleResult.Err(CandleError.Msg("file not found: $p").withPath(p))
                val data = when (val res = SafeTensors.deserialize(file)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error.withPath(p))
                }
                for (k in data.names) {
                    routing[k] = index
                }
                safetensors.add(data)
            }
            return CandleResult.Ok(MmapedSafetensors(safetensors, routing))
        }
    }
}

class SliceSafetensors private constructor(
    private val safetensors: SafeTensors,
) {
    fun load(name: String, dev: Device): CandleResult<Tensor> {
        val view = when (val res = this.safetensors.tensorLookup(name)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return view.load(dev)
    }

    fun tensors(): List<Pair<String, SafeTensorView>> = this.safetensors.tensors

    fun get(name: String): CandleResult<SafeTensorView> = this.safetensors.tensorLookup(name)

    companion object {
        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        fun new(buffer: ByteArray): CandleResult<SliceSafetensors> {
            val safetensors = when (val res = SafeTensors.deserialize(buffer)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(SliceSafetensors(safetensors))
        }
    }
}

class BufferedSafetensors private constructor(
    private val safetensors: SafeTensors,
) {
    fun load(name: String, dev: Device): CandleResult<Tensor> {
        val view = when (val res = this.get(name)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return view.load(dev)
    }

    fun tensors(): List<Pair<String, SafeTensorView>> = this.safetensors.tensors

    fun get(name: String): CandleResult<SafeTensorView> = this.safetensors.tensorLookup(name)

    companion object {
        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        fun new(buffer: ByteArray): CandleResult<BufferedSafetensors> {
            val safetensors = when (val res = SafeTensors.deserialize(buffer)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            return CandleResult.Ok(BufferedSafetensors(safetensors))
        }
    }
}

class MmapedFile private constructor(
    private val path: String,
    private val inner: ByteArray,
) {
    fun deserialize(): CandleResult<SafeTensors> {
        return when (val res = SafeTensors.deserialize(this.inner)) {
            is CandleResult.Ok -> CandleResult.Ok(res.value)
            is CandleResult.Err -> CandleResult.Err(res.error.withPath(this.path))
        }
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
            val inner = inMemorySafetensorFiles[p]
                ?: return CandleResult.Err(CandleError.Msg("file not found: $p").withPath(p))
            return CandleResult.Ok(MmapedFile(p, inner))
        }
    }
}

private fun saveSingleTensor() {
    val t = Tensor.zeros(Shape.fromDims(intArrayOf(2, 2)), DType.F32, Device.Cpu).unwrap()
    t.saveSafetensors("t", "t.safetensors").unwrap()
    val bytes = inMemorySafetensorFiles["t.safetensors"]!!
    check(bytes.isNotEmpty())
    inMemorySafetensorFiles.remove("t.safetensors")
}

private fun saveLoadMultipleTensors() {
    val t = Tensor.zeros(Shape.fromDims(intArrayOf(2, 2)), DType.F32, Device.Cpu).unwrap()
    val u = Tensor.zeros(Shape.fromDims(intArrayOf(1, 2)), DType.F32, Device.Cpu).unwrap()
    val map = linkedMapOf("t" to t, "u" to u)
    save(map, "multi.safetensors").unwrap()

    val weights = load("multi.safetensors", Device.Cpu).unwrap()
    check(weights["t"]!!.dims().contentEquals(intArrayOf(2, 2)))
    check(weights["u"]!!.dims().contentEquals(intArrayOf(1, 2)))
    val bytes = inMemorySafetensorFiles["multi.safetensors"]!!
    check(bytes.isNotEmpty())
    inMemorySafetensorFiles.remove("multi.safetensors")
}

private fun loadU8() {
    val header = "{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}   "
    val headerBytes = header.encodeToByteArray()
    val bytes = ByteArray(8 + headerBytes.size + 2)
    bytes[0] = 56
    headerBytes.copyInto(bytes, destinationOffset = 8)
    bytes[8 + headerBytes.size] = 1
    bytes[8 + headerBytes.size + 1] = 3
    inMemorySafetensorFiles["testU8.safetensors"] = bytes
    val weights = load("testU8.safetensors", Device.Cpu).unwrap()
    val tensor = weights["x"]!!
    check(tensor.dims().contentEquals(intArrayOf(2)))
    check(tensor.dtype() == DType.U8)
    val data = tensor.toVec1<UByte>().unwrap()
    check(data == listOf(1.toUByte(), 3.toUByte()))
    inMemorySafetensorFiles.remove("testU8.safetensors")
}
