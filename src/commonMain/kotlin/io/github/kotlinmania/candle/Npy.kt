// port-lint: source npy.rs
/**
 * Numpy support for tensors.
 *
 * The spec for the npy format can be found in
 * [npy-format](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html).
 * The functions from this module can be used to read tensors from npy/npz files
 * or write tensors to these files. A npy file contains a single tensor (unnamed)
 * whereas a npz file can contain multiple named tensors. npz files are also compressed.
 *
 * These two formats are easy to use in Python using the numpy library.
 *
 * ```python
 * import numpy as np
 * x = np.arange(10)
 *
 * # Write a npy file.
 * np.save("test.npy", x)
 *
 * # Read a value from the npy file.
 * x = np.load("test.npy")
 *
 * # Write multiple values to a npz file.
 * values = { "x": x, "xPlusOne": x + 1 }
 * np.savez("test.npz", **values)
 *
 * # Load multiple values from a npz file.
 * values = np.loadz("test.npz")
 * ```
 */
package io.github.kotlinmania.candle

import io.github.kotlinmania.zip.ByteArrayZipSink
import io.github.kotlinmania.zip.ByteArrayZipSource
import io.github.kotlinmania.zip.CompressionMethod
import io.github.kotlinmania.zip.FileOptions
import io.github.kotlinmania.zip.ZipArchive
import io.github.kotlinmania.zip.ZipError
import io.github.kotlinmania.zip.ZipWriter
import kotlin.test.Test

private val NPY_MAGIC_STRING: ByteArray = byteArrayOf(
    0x93.toByte(),
    'N'.code.toByte(),
    'U'.code.toByte(),
    'M'.code.toByte(),
    'P'.code.toByte(),
    'Y'.code.toByte(),
)
private const val NPY_SUFFIX: String = ".npy"

/**
 * Sequential byte reader for parsing NPY and NPZ streams.
 *
 * Tracks the current read [position] within the underlying [bytes] buffer
 * and provides exact-length reads returning [CandleResult].
 */
class NpyByteReader(
    private val bytes: ByteArray,
    var position: Int = 0,
) {
    /**
     * Reads exactly `dst.size` bytes into `dst`, advancing [position],
     * or returns a [CandleError.Npy] error on unexpected end of stream.
     */
    val readExact: (ByteArray) -> CandleResult<Unit> = { dst ->
        if (this.position + dst.size > this.bytes.size) {
            CandleResult.Err(CandleError.Npy("unexpected end of npy stream"))
        } else {
            this.bytes.copyInto(dst, destinationOffset = 0, startIndex = this.position, endIndex = this.position + dst.size)
            this.position += dst.size
            CandleResult.Ok(Unit)
        }
    }
}

/**
 * Growable byte writer for serializing tensors in NPY and NPZ formats.
 *
 * Automatically doubles internal capacity as bytes are appended and exposes
 * the serialized byte slice via [toByteArray].
 */
class NpyByteWriter {
    private var buffer: ByteArray = ByteArray(256)
    var size: Int = 0
        private set

    /**
     * Appends all bytes from `src` to the writer buffer.
     */
    val writeAll: (ByteArray) -> CandleResult<Unit> = { src ->
        if (this.size + src.size > this.buffer.size) {
            var newCap = this.buffer.size * 2
            while (newCap < this.size + src.size) newCap *= 2
            this.buffer = this.buffer.copyOf(newCap)
        }
        src.copyInto(this.buffer, destinationOffset = this.size)
        this.size += src.size
        CandleResult.Ok(Unit)
    }

    /**
     * Returns a copy of the written bytes of length [size].
     */
    val toByteArray: () -> ByteArray = {
        this.buffer.copyOf(this.size)
    }
}

private val inMemoryNpyFiles: MutableMap<String, ByteArray> = LinkedHashMap()
private val inMemoryNpzFiles: MutableMap<String, ByteArray> = LinkedHashMap()

private fun readHeader(reader: NpyByteReader): CandleResult<String> {
    val magicString = ByteArray(NPY_MAGIC_STRING.size)
    when (val res = reader.readExact(magicString)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    if (!magicString.contentEquals(NPY_MAGIC_STRING)) {
        return CandleResult.Err(CandleError.Npy("magic string mismatch"))
    }
    val version = ByteArray(2)
    when (val res = reader.readExact(version)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val headerLenLen = when (version[0].toInt() and 0xFF) {
        1 -> 2
        2 -> 4
        else -> {
            val otherwise = version[0].toInt() and 0xFF
            return CandleResult.Err(CandleError.Npy("unsupported version $otherwise"))
        }
    }
    val headerLenBytes = ByteArray(headerLenLen)
    when (val res = reader.readExact(headerLenBytes)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    var headerLen = 0
    for (idx in headerLenBytes.indices.reversed()) {
        val v = headerLenBytes[idx].toInt() and 0xFF
        headerLen = 256 * headerLen + v
    }
    val header = ByteArray(headerLen)
    when (val res = reader.readExact(header)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    return CandleResult.Ok(header.decodeToString())
}

internal data class Header(
    val descr: DType,
    val fortranOrder: Boolean,
    val shape: List<Int>,
) {
    fun shape(): Shape = Shape.fromDims(this.shape.toIntArray())

    fun toString(validate: Boolean = true): CandleResult<String> {
        val fortranOrderStr = if (this.fortranOrder) "True" else "False"
        var shapeStr = this.shape.joinToString(",") { it.toString() }
        val descrStr = when (this.descr) {
            DType.BF16 -> return CandleResult.Err(CandleError.Npy("bf16 is not supported"))
            DType.F16 -> "f2"
            DType.F32 -> "f4"
            DType.F64 -> "f8"
            DType.I16 -> "i2"
            DType.I32 -> "i4"
            DType.I64 -> "i8"
            DType.U32 -> "u4"
            DType.U8 -> "u1"
            DType.F8E4M3 -> return CandleResult.Err(CandleError.Npy("f8e4m3 is not supported"))
            DType.F6E2M3 -> return CandleResult.Err(CandleError.Npy("f6e2m3 is not supported"))
            DType.F6E3M2 -> return CandleResult.Err(CandleError.Npy("f6e3m2 is not supported"))
            DType.F4 -> return CandleResult.Err(CandleError.Npy("f4 is not supported"))
            DType.F8E8M0 -> return CandleResult.Err(CandleError.Npy("f8e8m0 is not supported"))
        }
        if (shapeStr.isNotEmpty() && validate) {
            shapeStr += ","
        }
        return CandleResult.Ok(
            "{'descr': '<$descrStr', 'fortran_order': $fortranOrderStr, 'shape': ($shapeStr), }"
        )
    }

    companion object {
        // Parser for the npy header dictionary representation.
        fun parse(header: String): CandleResult<Header> {
            val trimmed = header.trim { c -> c == '{' || c == '}' || c == ',' || c.isWhitespace() }

            val parts = ArrayList<String>()
            var startIndex = 0
            var cntParenthesis = 0L
            for (index in trimmed.indices) {
                when (trimmed[index]) {
                    '(' -> cntParenthesis += 1L
                    ')' -> cntParenthesis -= 1L
                    ',' -> if (cntParenthesis == 0L) {
                        parts.add(trimmed.substring(startIndex, index))
                        startIndex = index + 1
                    }
                }
            }
            parts.add(trimmed.substring(startIndex))
            val partMap = HashMap<String, String>()
            for (rawPart in parts) {
                val part = rawPart.trim()
                if (part.isNotEmpty()) {
                    val splitParts = part.split(':')
                    if (splitParts.size == 2) {
                        val key = splitParts[0].trim { c -> c == '\'' || c.isWhitespace() }
                        val value = splitParts[1].trim { c -> c == '\'' || c.isWhitespace() }
                        partMap[key] = value
                    } else {
                        return CandleResult.Err(CandleError.Npy("unable to parse header $trimmed"))
                    }
                }
            }
            val fortranOrder = when (val fo = partMap["fortran_order"]) {
                null -> false
                "False" -> false
                "True" -> true
                else -> return CandleResult.Err(CandleError.Npy("unknown fortran_order $fo"))
            }
            val descr = when (val d = partMap["descr"]) {
                null -> return CandleResult.Err(CandleError.Npy("no descr in header"))
                else -> {
                    if (d.isEmpty()) {
                        return CandleResult.Err(CandleError.Npy("empty descr"))
                    }
                    if (d.startsWith('>')) {
                        return CandleResult.Err(CandleError.Npy("little-endian descr $d"))
                    }
                    // The only supported types in tensor are:
                    //     float64, float32, float16,
                    //     complex64, complex128,
                    //     int64, int32, int16, int8,
                    //     uint8, and bool.
                    when (val norm = d.trim { c -> c == '=' || c == '<' || c == '|' }) {
                        "e", "f2" -> DType.F16
                        "f", "f4" -> DType.F32
                        "d", "f8" -> DType.F64
                        "i", "i4" -> DType.I32
                        "q", "i8" -> DType.I64
                        "h", "i2" -> DType.I16
                        "B", "u1" -> DType.U8
                        "I", "u4" -> DType.U32
                        "?", "b1" -> DType.U8
                        else -> return CandleResult.Err(CandleError.Npy("unrecognized descr $norm"))
                    }
                }
            }
            val shape = when (val s = partMap["shape"]) {
                null -> return CandleResult.Err(CandleError.Npy("no shape in header"))
                else -> {
                    val cleanShape = s.trim { c -> c == '(' || c == ')' || c == ',' }
                    if (cleanShape.isEmpty()) {
                        emptyList()
                    } else {
                        val dims = ArrayList<Int>()
                        for (v in cleanShape.split(',')) {
                            val parsed = v.trim().toIntOrNull()
                                ?: return CandleResult.Err(CandleError.Npy("invalid dimension in shape $s"))
                            dims.add(parsed)
                        }
                        dims
                    }
                }
            }
            return CandleResult.Ok(
                Header(
                    descr = descr,
                    fortranOrder = fortranOrder,
                    shape = shape,
                )
            )
        }
    }
}

internal fun fromReader(
    shape: Shape,
    dtype: DType,
    reader: NpyByteReader,
): CandleResult<Tensor> {
    val elemCount = shape.elemCount()
    val readBytes = { count: Int ->
        val buf = ByteArray(count)
        when (val res = reader.readExact(buf)) {
            is CandleResult.Ok -> CandleResult.Ok(buf)
            is CandleResult.Err -> CandleResult.Err(res.error)
        }
    }
    val decodeU16 = { buf: ByteArray, offset: Int ->
        val b0 = buf[offset].toInt() and 0xFF
        val b1 = buf[offset + 1].toInt() and 0xFF
        ((b1 shl 8) or b0).toUShort()
    }
    val decodeI32 = { buf: ByteArray, offset: Int ->
        val b0 = buf[offset].toInt() and 0xFF
        val b1 = buf[offset + 1].toInt() and 0xFF
        val b2 = buf[offset + 2].toInt() and 0xFF
        val b3 = buf[offset + 3].toInt() and 0xFF
        b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }
    val decodeI64 = { buf: ByteArray, offset: Int ->
        val lo = decodeI32(buf, offset).toLong() and 0xFFFFFFFFL
        val hi = decodeI32(buf, offset + 4).toLong() and 0xFFFFFFFFL
        lo or (hi shl 32)
    }
    return when (dtype) {
        DType.BF16 -> {
            val raw = when (val res = readBytes(elemCount * 2)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = Array(elemCount) { i -> BF16.fromBits(decodeU16(raw, i * 2)) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.BF16(dataT)), shape, BackpropOp.none(), false))
        }
        DType.F16 -> {
            val raw = when (val res = readBytes(elemCount * 2)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = Array(elemCount) { i -> F16.fromBits(decodeU16(raw, i * 2)) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.F16(dataT)), shape, BackpropOp.none(), false))
        }
        DType.F32 -> {
            val raw = when (val res = readBytes(elemCount * 4)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = FloatArray(elemCount) { i -> Float.fromBits(decodeI32(raw, i * 4)) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.F32(dataT)), shape, BackpropOp.none(), false))
        }
        DType.F64 -> {
            val raw = when (val res = readBytes(elemCount * 8)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = DoubleArray(elemCount) { i -> Double.fromBits(decodeI64(raw, i * 8)) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.F64(dataT)), shape, BackpropOp.none(), false))
        }
        DType.U8 -> {
            val raw = when (val res = readBytes(elemCount)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = UByteArray(elemCount) { i -> raw[i].toUByte() }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.U8(dataT)), shape, BackpropOp.none(), false))
        }
        DType.U32 -> {
            val raw = when (val res = readBytes(elemCount * 4)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = UIntArray(elemCount) { i -> decodeI32(raw, i * 4).toUInt() }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.U32(dataT)), shape, BackpropOp.none(), false))
        }
        DType.I16 -> {
            val raw = when (val res = readBytes(elemCount * 2)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = ShortArray(elemCount) { i -> decodeU16(raw, i * 2).toShort() }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.I16(dataT)), shape, BackpropOp.none(), false))
        }
        DType.I32 -> {
            val raw = when (val res = readBytes(elemCount * 4)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = IntArray(elemCount) { i -> decodeI32(raw, i * 4) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.I32(dataT)), shape, BackpropOp.none(), false))
        }
        DType.I64 -> {
            val raw = when (val res = readBytes(elemCount * 8)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataT = LongArray(elemCount) { i -> decodeI64(raw, i * 8) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.I64(dataT)), shape, BackpropOp.none(), false))
        }
        DType.F8E4M3 -> {
            val raw = when (val res = readBytes(elemCount)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val dataF8 = Array(elemCount) { i -> F8E4M3.fromBits(raw[i].toUByte()) }
            CandleResult.Ok(fromStorage(Storage.Cpu(CpuStorage.F8E4M3(dataF8)), shape, BackpropOp.none(), false))
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 ->
            CandleResult.Err(CandleError.UnsupportedDTypeForOp(dtype, "fromReader").bt())
    }
}

/**
 * Reads a npy file and return the stored multi-dimensional array as a tensor.
 */
fun readNpy(path: String): CandleResult<Tensor> {
    val bytes = inMemoryNpyFiles[path]
        ?: return CandleResult.Err(CandleError.Npy("file not found: $path"))
    val reader = NpyByteReader(bytes)
    val headerStr = when (val res = readHeader(reader)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val header = when (val res = Header.parse(headerStr)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    if (header.fortranOrder) {
        return CandleResult.Err(CandleError.Npy("fortran order not supported"))
    }
    return fromReader(header.shape(), header.descr, reader)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays together with their names.
 */
fun readNpz(path: String): CandleResult<List<Pair<String, Tensor>>> {
    val zipBytes = inMemoryNpzFiles[path]
        ?: return CandleResult.Err(CandleError.Npy("npz file not found: $path"))
    val zip = ZipArchive.new(ByteArrayZipSource(zipBytes)).getOrElse { err ->
        return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
    }
    val result = ArrayList<Pair<String, Tensor>>(zip.len())
    for (i in 0 until zip.len()) {
        val zipFile = zip.byIndex(i).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val name = zipFile.name().removeSuffix(NPY_SUFFIX)
        val reader = NpyByteReader(zipFile.readToEnd())
        val headerStr = when (val res = readHeader(reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val header = when (val res = Header.parse(headerStr)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (header.fortranOrder) {
            return CandleResult.Err(CandleError.Npy("fortran order not supported"))
        }
        val s = when (val res = fromReader(header.shape(), header.descr, reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        result.add(name to s)
    }
    return CandleResult.Ok(result)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays for some specified names.
 */
fun readNpzByName(path: String, names: Array<String>): CandleResult<List<Tensor>> {
    val zipBytes = inMemoryNpzFiles[path]
        ?: return CandleResult.Err(CandleError.Npy("npz file not found: $path"))
    val zip = ZipArchive.new(ByteArrayZipSource(zipBytes)).getOrElse { err ->
        return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
    }
    val result = ArrayList<Tensor>(names.size)
    for (name in names) {
        val zipFile = zip.byName("$name$NPY_SUFFIX").getOrElse {
            return CandleResult.Err(CandleError.Npy("no array for $name in $path"))
        }
        val reader = NpyByteReader(zipFile.readToEnd())
        val headerStr = when (val res = readHeader(reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val header = when (val res = Header.parse(headerStr)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (header.fortranOrder) {
            return CandleResult.Err(CandleError.Npy("fortran order not supported"))
        }
        val s = when (val res = fromReader(header.shape(), header.descr, reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        result.add(s)
    }
    return CandleResult.Ok(result)
}

private fun Tensor.write(f: NpyByteWriter): CandleResult<Unit> {
    f.writeAll(NPY_MAGIC_STRING)
    f.writeAll(byteArrayOf(1, 0))
    val header = Header(
        descr = this.dtype(),
        fortranOrder = false,
        shape = this.dims().toList(),
    )
    var headerStr = when (val res = header.toString(true)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val pad = 16 - (NPY_MAGIC_STRING.size + 5 + headerStr.length) % 16
    for (i in 0 until (pad % 16)) {
        headerStr += ' '
    }
    headerStr += '\n'
    f.writeAll(byteArrayOf((headerStr.length % 256).toByte(), (headerStr.length / 256).toByte()))
    f.writeAll(headerStr.encodeToByteArray())
    return f.writeAll(TensorSafeView(this).data())
}

/**
 * Writes a multi-dimensional array in the npy format.
 */
fun Tensor.writeNpy(path: String): CandleResult<Unit> {
    val f = NpyByteWriter()
    when (val res = this.write(f)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    inMemoryNpyFiles[path] = f.toByteArray()
    return CandleResult.Ok(Unit)
}

/**
 * Writes multiple multi-dimensional arrays using the npz format.
 */
fun writeNpz(ts: List<Pair<String, Tensor>>, path: String): CandleResult<Unit> {
    val sink = ByteArrayZipSink()
    val zip = ZipWriter(sink)
    val options = FileOptions.default().compressionMethod(CompressionMethod.Stored)
    for ((name, tensor) in ts) {
        zip.startFile("$name.npy", options).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val writer = NpyByteWriter()
        when (val res = tensor.write(writer)) {
            is CandleResult.Ok -> {}
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        zip.write(writer.toByteArray()).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
    }
    zip.finish().getOrElse { err ->
        return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
    }
    inMemoryNpzFiles[path] = sink.toByteArray()
    return CandleResult.Ok(Unit)
}

/**
 * Lazy tensor loader.
 */
class NpzTensors private constructor(
    private val indexPerName: Map<String, Int>,
    private val path: String,
    // We do not store a zip reader as it needs mutable access to extract data. Instead we
    // re-create a zip reader for each tensor.
) {
    fun names(): List<String> = this.indexPerName.keys.toList()

    /**
     * This only returns the shape and dtype for a named tensor. Compared to `get`, this avoids
     * reading the whole tensor data.
     */
    fun getShapeAndDtype(name: String): CandleResult<Pair<Shape, DType>> {
        val index = this.indexPerName[name]
            ?: return CandleResult.Err(CandleError.Msg("cannot find tensor $name").bt())
        val zipBytes = inMemoryNpzFiles[this.path]
            ?: return CandleResult.Err(CandleError.Npy("npz file not found: ${this.path}"))
        val zip = ZipArchive.new(ByteArrayZipSource(zipBytes)).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val zipFile = zip.byIndex(index).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val reader = NpyByteReader(zipFile.readToEnd())
        val headerStr = when (val res = readHeader(reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val header = when (val res = Header.parse(headerStr)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return CandleResult.Ok(header.shape() to header.descr)
    }

    fun get(name: String): CandleResult<Tensor?> {
        val index = this.indexPerName[name] ?: return CandleResult.Ok(null)
        // We hope that the file has not changed since first reading it.
        val zipBytes = inMemoryNpzFiles[this.path]
            ?: return CandleResult.Err(CandleError.Npy("npz file not found: ${this.path}"))
        val zip = ZipArchive.new(ByteArrayZipSource(zipBytes)).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val zipFile = zip.byIndex(index).getOrElse { err ->
            return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
        }
        val reader = NpyByteReader(zipFile.readToEnd())
        val headerStr = when (val res = readHeader(reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val header = when (val res = Header.parse(headerStr)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (header.fortranOrder) {
            return CandleResult.Err(CandleError.Npy("fortran order not supported"))
        }
        val tensor = when (val res = fromReader(header.shape(), header.descr, reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return CandleResult.Ok(tensor)
    }

    companion object {
        fun new(path: String): CandleResult<NpzTensors> {
            val zipBytes = inMemoryNpzFiles[path]
                ?: return CandleResult.Err(CandleError.Npy("npz file not found: $path"))
            val zip = ZipArchive.new(ByteArrayZipSource(zipBytes)).getOrElse { err ->
                return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
            }
            val indexPerName = LinkedHashMap<String, Int>()
            for (i in 0 until zip.len()) {
                val file = zip.byIndex(i).getOrElse { err ->
                    return CandleResult.Err(CandleError.Zip(err.message ?: err.toString(), err as? ZipError))
                }
                val name = file.name().removeSuffix(NPY_SUFFIX)
                indexPerName[name] = i
            }
            return CandleResult.Ok(NpzTensors(indexPerName, path))
        }
    }
}

@Test
internal fun parse() {
    val h1 = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }"
    check(
        Header.parse(h1).unwrap() ==
            Header(
                descr = DType.F64,
                fortranOrder = false,
                shape = listOf(128),
            )
    )
    val h2Str = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }"
    val h2 = Header.parse(h2Str).unwrap()
    check(
        h2 ==
            Header(
                descr = DType.F32,
                fortranOrder = true,
                shape = listOf(256, 1, 128),
            )
    )
    check(
        h2.toString(true).unwrap() ==
            "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }"
    )

    val h3 = Header(
        descr = DType.U32,
        fortranOrder = false,
        shape = emptyList(),
    )
    check(
        h3.toString(true).unwrap() ==
            "{'descr': '<u4', 'fortran_order': False, 'shape': (), }"
    )
}
