// port-lint: source npy.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Numpy support for tensors.
 *
 * The spec for the npy format can be found in npy-format.
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
private val NPY_MAGIC_STRING: ByteArray = byteArrayOf(
    0x93.toByte(),
    'N'.code.toByte(),
    'U'.code.toByte(),
    'M'.code.toByte(),
    'P'.code.toByte(),
    'Y'.code.toByte(),
)
private const val NPY_SUFFIX: String = ".npy"

public interface Read {
    public val readExact: (ByteArray) -> CandleResult<Unit>
    public val readU16Into: (UShortArray) -> CandleResult<Unit>
    public val readF32Into: (FloatArray) -> CandleResult<Unit>
    public val readF64Into: (DoubleArray) -> CandleResult<Unit>
    public val readU32Into: (UIntArray) -> CandleResult<Unit>
    public val readI16Into: (ShortArray) -> CandleResult<Unit>
    public val readI32Into: (IntArray) -> CandleResult<Unit>
    public val readI64Into: (LongArray) -> CandleResult<Unit>
}

public interface Write {
    public val writeAll: (ByteArray) -> CandleResult<Unit>
    public val writeU8: (UByte) -> CandleResult<Unit>
    public val writeU16: (UShort) -> CandleResult<Unit>
    public val writeI16: (Short) -> CandleResult<Unit>
    public val writeU32: (UInt) -> CandleResult<Unit>
    public val writeI32: (Int) -> CandleResult<Unit>
    public val writeI64: (Long) -> CandleResult<Unit>
    public val writeF32: (Float) -> CandleResult<Unit>
    public val writeF64: (Double) -> CandleResult<Unit>
}

public class NpyByteReader(
    private val data: ByteArray,
    private var pos: Int = 0,
) : Read {
    override val readExact: (ByteArray) -> CandleResult<Unit> = { buf ->
        if (pos + buf.size > data.size) {
            CandleResult.Err(CandleError.Io("unexpected EOF"))
        } else {
            data.copyInto(buf, 0, pos, pos + buf.size)
            pos += buf.size
            CandleResult.Ok(Unit)
        }
    }

    override val readU16Into: (UShortArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 2)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val b0 = bytes[i * 2].toInt() and 0xFF
                val b1 = bytes[i * 2 + 1].toInt() and 0xFF
                dst[i] = ((b1 shl 8) or b0).toUShort()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readF32Into: (FloatArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                val bits = (bytes[base].toInt() and 0xFF) or
                    ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[base + 3].toInt() and 0xFF) shl 24)
                dst[i] = Float.fromBits(bits)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readF64Into: (DoubleArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 8)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 8
                var bits = 0L
                for (b in 0 until 8) {
                    bits = bits or ((bytes[base + b].toLong() and 0xFFL) shl (b * 8))
                }
                dst[i] = Double.fromBits(bits)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readU32Into: (UIntArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                val bits = (bytes[base].toInt() and 0xFF) or
                    ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[base + 3].toInt() and 0xFF) shl 24)
                dst[i] = bits.toUInt()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI16Into: (ShortArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 2)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val b0 = bytes[i * 2].toInt() and 0xFF
                val b1 = bytes[i * 2 + 1].toInt() and 0xFF
                dst[i] = ((b1 shl 8) or b0).toShort()
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI32Into: (IntArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 4)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 4
                dst[i] = (bytes[base].toInt() and 0xFF) or
                    ((bytes[base + 1].toInt() and 0xFF) shl 8) or
                    ((bytes[base + 2].toInt() and 0xFF) shl 16) or
                    ((bytes[base + 3].toInt() and 0xFF) shl 24)
            }
            CandleResult.Ok(Unit)
        }
    }

    override val readI64Into: (LongArray) -> CandleResult<Unit> = { dst ->
        val bytes = ByteArray(dst.size * 8)
        val r = readExact(bytes)
        if (r is CandleResult.Err) {
            r
        } else {
            for (i in dst.indices) {
                val base = i * 8
                var bits = 0L
                for (b in 0 until 8) {
                    bits = bits or ((bytes[base + b].toLong() and 0xFFL) shl (b * 8))
                }
                dst[i] = bits
            }
            CandleResult.Ok(Unit)
        }
    }
}

public class NpyByteWriter : Write {
    private var buf: ByteArray = ByteArray(1024)
    private var size: Int = 0

    private val ensureCapacity: (Int) -> Unit = { additional ->
        if (size + additional > buf.size) {
            val newSize = maxOf(buf.size * 2, size + additional)
            buf = buf.copyOf(newSize)
        }
    }

    override val writeAll: (ByteArray) -> CandleResult<Unit> = { bytes ->
        ensureCapacity(bytes.size)
        bytes.copyInto(buf, size)
        size += bytes.size
        CandleResult.Ok(Unit)
    }

    override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
        ensureCapacity(1)
        buf[size++] = v.toByte()
        CandleResult.Ok(Unit)
    }

    override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
        val bits = v.toInt()
        ensureCapacity(2)
        buf[size++] = (bits and 0xFF).toByte()
        buf[size++] = ((bits ushr 8) and 0xFF).toByte()
        CandleResult.Ok(Unit)
    }

    override val writeI16: (Short) -> CandleResult<Unit> = { v ->
        writeU16(v.toUShort())
    }

    override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
        val bits = v.toInt()
        ensureCapacity(4)
        buf[size++] = (bits and 0xFF).toByte()
        buf[size++] = ((bits ushr 8) and 0xFF).toByte()
        buf[size++] = ((bits ushr 16) and 0xFF).toByte()
        buf[size++] = ((bits ushr 24) and 0xFF).toByte()
        CandleResult.Ok(Unit)
    }

    override val writeI32: (Int) -> CandleResult<Unit> = { v ->
        writeU32(v.toUInt())
    }

    override val writeI64: (Long) -> CandleResult<Unit> = { v ->
        ensureCapacity(8)
        for (i in 0 until 8) {
            buf[size++] = ((v ushr (i * 8)) and 0xFFL).toByte()
        }
        CandleResult.Ok(Unit)
    }

    override val writeF32: (Float) -> CandleResult<Unit> = { v ->
        writeI32(v.toRawBits())
    }

    override val writeF64: (Double) -> CandleResult<Unit> = { v ->
        writeI64(v.toRawBits())
    }

    public val toByteArray: () -> ByteArray = {
        buf.copyOf(size)
    }
}

private class File(
    val path: String,
    private val reader: NpyByteReader = NpyByteReader(ByteArray(0)),
    private val writer: NpyByteWriter = NpyByteWriter(),
    private val isWrite: Boolean = false,
) : Read by reader, Write {
    override val writeAll: (ByteArray) -> CandleResult<Unit> = { bytes ->
        val r = writer.writeAll(bytes)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
        val r = writer.writeU8(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
        val r = writer.writeU16(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI16: (Short) -> CandleResult<Unit> = { v ->
        val r = writer.writeI16(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
        val r = writer.writeU32(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI32: (Int) -> CandleResult<Unit> = { v ->
        val r = writer.writeI32(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeI64: (Long) -> CandleResult<Unit> = { v ->
        val r = writer.writeI64(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeF32: (Float) -> CandleResult<Unit> = { v ->
        val r = writer.writeF32(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }
    override val writeF64: (Double) -> CandleResult<Unit> = { v ->
        val r = writer.writeF64(v)
        if (isWrite) platformWriteFileBytes(path, writer.toByteArray())
        r
    }

    companion object {
        val open: (String) -> CandleResult<File> = { p ->
            CandleResult.Ok(File(p, reader = NpyByteReader(platformReadFileBytes(p))))
        }
        val create: (String) -> CandleResult<File> = { p ->
            CandleResult.Ok(File(p, isWrite = true))
        }
    }
}

private class BufReader(val file: File) {
    companion object {
        val new: (File) -> BufReader = { f -> BufReader(f) }
    }
}

private class ZipEntryReader(
    private val entryName: String,
    bytes: ByteArray,
) : Read by NpyByteReader(bytes) {
    val name: () -> String = { entryName }
}

private object zip {
    enum class CompressionMethod {
        Stored,
    }

    object write {
        class FileOptions<T> {
            val compressionMethod: (CompressionMethod) -> FileOptions<T> = { this }

            companion object {
                val default: () -> FileOptions<Unit> = { FileOptions() }
            }
        }
    }

    class ZipArchive(private val entries: List<Pair<String, ByteArray>>) {
        val len: () -> Int = { entries.size }
        val byIndex: (Int) -> CandleResult<ZipEntryReader> = { i ->
            val e = entries[i]
            CandleResult.Ok(ZipEntryReader(e.first, e.second))
        }
        val byName: (String) -> CandleResult<ZipEntryReader> = { target ->
            val found = entries.firstOrNull { it.first == target }
            if (found != null) {
                CandleResult.Ok(ZipEntryReader(found.first, found.second))
            } else {
                CandleResult.Err(CandleError.Zip("entry not found: $target"))
            }
        }

        companion object {
            val new: (BufReader) -> CandleResult<ZipArchive> = { br ->
                val map = platformReadZipEntries(br.file.path)
                CandleResult.Ok(ZipArchive(map.entries.map { Pair(it.key, it.value) }))
            }
        }
    }

    class ZipWriter(private val file: File) : Write {
        private val entries = linkedMapOf<String, ByteArray>()
        private var currentName: String = ""
        private var currentWriter: NpyByteWriter = NpyByteWriter()

        private val flushCurrent: () -> Unit = {
            if (currentName.isNotEmpty()) {
                entries[currentName] = currentWriter.toByteArray()
                platformWriteZipEntries(file.path, entries)
            }
        }

        val startFile: (String, write.FileOptions<Unit>) -> CandleResult<Unit> = { name, _ ->
            flushCurrent()
            currentName = name
            currentWriter = NpyByteWriter()
            CandleResult.Ok(Unit)
        }

        override val writeAll: (ByteArray) -> CandleResult<Unit> = { b ->
            val r = currentWriter.writeAll(b)
            flushCurrent()
            r
        }
        override val writeU8: (UByte) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU8(v)
            flushCurrent()
            r
        }
        override val writeU16: (UShort) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU16(v)
            flushCurrent()
            r
        }
        override val writeI16: (Short) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI16(v)
            flushCurrent()
            r
        }
        override val writeU32: (UInt) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeU32(v)
            flushCurrent()
            r
        }
        override val writeI32: (Int) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI32(v)
            flushCurrent()
            r
        }
        override val writeI64: (Long) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeI64(v)
            flushCurrent()
            r
        }
        override val writeF32: (Float) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeF32(v)
            flushCurrent()
            r
        }
        override val writeF64: (Double) -> CandleResult<Unit> = { v ->
            val r = currentWriter.writeF64(v)
            flushCurrent()
            r
        }

        companion object {
            val new: (File) -> ZipWriter = { f -> ZipWriter(f) }
        }
    }
}

private object Error {
    val Npy: (String) -> CandleException = { msg -> CandleError.Npy(msg) }
    val UnsupportedDTypeForOp: (DType, String) -> CandleException = { dtype, op ->
        CandleError.UnsupportedDTypeForOp(dtype, op)
    }
}

private val format: (String, Any?) -> String = { fmt, a ->
    fmt.replace("{otherwise}", a.toString())
        .replace("{header}", a.toString())
        .replace("{fortran_order}", a.toString())
        .replace("{descr}", a.toString())
        .replace("{name}", a.toString())
        .replace("{}", a.toString())
}

public fun <R : Read> readHeader(reader: R): CandleResult<String> {
    val magicString = ByteArray(NPY_MAGIC_STRING.size)
    reader.readExact(magicString).unwrap()
    if (!magicString.contentEquals(NPY_MAGIC_STRING)) {
        return CandleResult.Err(Error.Npy("magic string mismatch".toString()))
    }
    val version = ByteArray(2)
    reader.readExact(version).unwrap()
    val headerLenLen = when (version[0].toInt() and 0xFF) {
        1 -> 2
        2 -> 4
        else -> {
            val otherwise = version[0].toInt() and 0xFF
            return CandleResult.Err(Error.Npy(format("unsupported version {otherwise}", otherwise)))
        }
    }
    val headerLenBytes = ByteArray(headerLenLen)
    reader.readExact(headerLenBytes).unwrap()
    val headerLen = headerLenBytes
        .reversed()
        .fold(0) { acc, v -> 256 * acc + (v.toInt() and 0xFF) }
    val header = ByteArray(headerLen)
    reader.readExact(header).unwrap()
    return CandleResult.Ok(header.decodeToString())
}

public data class Header(
    val descr: DType,
    val fortranOrder: Boolean,
    val shape: List<Int>,
) {
    public fun shape(self: Header = this): Shape {
        return Shape.from(self.shape.toIntArray())
    }

    public fun toString(self: Header = this): CandleResult<String> {
        val fortranOrder = if (self.fortranOrder) "True" else "False"
        var shape = self
            .shape
            .map { x -> x.toString() }
            .joinToString(",")
        val descr = when (self.descr) {
            DType.BF16 -> return CandleResult.Err(Error.Npy("bf16 is not supported"))
            DType.F16 -> "f2"
            DType.F32 -> "f4"
            DType.F64 -> "f8"
            DType.I16 -> "i2"
            DType.I32 -> "i4"
            DType.I64 -> "i8"
            DType.U32 -> "u4"
            DType.U8 -> "u1"
            DType.F8E4M3 -> return CandleResult.Err(Error.Npy("f8e4m3 is not supported"))
            DType.F6E2M3 -> return CandleResult.Err(Error.Npy("f6e2m3 is not supported"))
            DType.F6E3M2 -> return CandleResult.Err(Error.Npy("f6e3m2 is not supported"))
            DType.F4 -> return CandleResult.Err(Error.Npy("f4 is not supported"))
            DType.F8E8M0 -> return CandleResult.Err(Error.Npy("f8e8m0 is not supported"))
        }
        if (shape.isNotEmpty()) {
            shape += ','
        }
        return CandleResult.Ok(
            "{'descr': '<$descr', 'fortran_order': $fortranOrder, 'shape': ($shape), }",
        )
    }

    public val toStringResult: () -> CandleResult<String> = { toString(this) }

    public companion object Self {
        // Hacky parser for the npy header, a typical example would be
        // descr f8, fortranOrder False, shape (128,)
        public fun parse(header: String): CandleResult<Header> {
            val header = header.trim { c -> c == '{' || c == '}' || c == ',' || c.isWhitespace() }

            val parts: MutableList<String> = mutableListOf()
            var startIndex = 0
            var cntParenthesis = 0L
            for (indexed in header.withIndex()) {
                val index = indexed.index
                val c = indexed.value
                when (c) {
                    '(' -> cntParenthesis += 1L
                    ')' -> cntParenthesis -= 1L
                    ',' -> {
                        if (cntParenthesis == 0L) {
                            parts.add(header.substring(startIndex, index))
                            startIndex = index + 1
                        }
                    }
                    else -> Unit
                }
            }
            parts.add(header.substring(startIndex))
            val partMap: MutableMap<String, String> = HashMap()
            for (part in parts) {
                val part = part.trim()
                if (part.isNotEmpty()) {
                    val split = part.split(':')
                    if (split.size == 2) {
                        val quote = 39.toChar()
                        val key = split[0].trim { c -> c == quote || c.isWhitespace() }
                        val value = split[1].trim { c -> c == quote || c.isWhitespace() }
                        partMap[key] = value
                    } else {
                        return CandleResult.Err(Error.Npy(format("unable to parse header {header}", header)))
                    }
                }
            }
            val fortranOrderVal = partMap["fortran_order"]
            val fortranOrder = when (fortranOrderVal) {
                null -> false
                "False" -> false
                "True" -> true
                else -> return CandleResult.Err(Error.Npy(format("unknown fortran_order {fortran_order}", fortranOrderVal)))
            }
            val descrVal = partMap["descr"]
            val descr = when (descrVal) {
                null -> return CandleResult.Err(Error.Npy("no descr in header".toString()))
                else -> {
                    if (descrVal.isEmpty()) {
                        return CandleResult.Err(Error.Npy("empty descr".toString()))
                    }
                    if (descrVal.startsWith('>')) {
                        return CandleResult.Err(Error.Npy(format("little-endian descr {descr}", descrVal)))
                    }
                    // the only supported types in tensor are:
                    //     float64, float32, float16,
                    //     complex64, complex128,
                    //     int64, int32, int16, int8,
                    //     uint8, and bool.
                    val trimmedDescr = descrVal.trim { c -> c == '=' || c == '<' || c == '|' }
                    when (trimmedDescr) {
                        "e", "f2" -> DType.F16
                        "f", "f4" -> DType.F32
                        "d", "f8" -> DType.F64
                        "i", "i4" -> DType.I32
                        "q", "i8" -> DType.I64
                        "h", "i2" -> DType.I16
                        "B", "u1" -> DType.U8
                        "I", "u4" -> DType.U32
                        "?", "b1" -> DType.U8
                        else -> return CandleResult.Err(Error.Npy(format("unrecognized descr {descr}", trimmedDescr)))
                    }
                }
            }
            val shapeVal = partMap["shape"]
            val shape = when (shapeVal) {
                null -> return CandleResult.Err(Error.Npy("no shape in header".toString()))
                else -> {
                    val shapeStr = shapeVal.trim { c -> c == '(' || c == ')' || c == ',' }
                    if (shapeStr.isEmpty()) {
                        emptyList()
                    } else {
                        val parsed = mutableListOf<Int>()
                        for (v in shapeStr.split(',')) {
                            val n = v.trim().toIntOrNull()
                                ?: return CandleResult.Err(CandleError.ParseInt("invalid int: $v"))
                            parsed.add(n)
                        }
                        parsed
                    }
                }
            }
            return CandleResult.Ok(
                Header(
                    descr = descr,
                    fortranOrder = fortranOrder,
                    shape = shape,
                ),
            )
        }
    }
}

public fun Tensor.Companion.fromReader(
    shape: Shape,
    dtype: DType,
    reader: Read,
): CandleResult<Tensor> {
    val elemCount = shape.elemCount()
    return when (dtype) {
        DType.BF16 -> {
            val dataU16 = UShortArray(elemCount)
            reader.readU16Into(dataU16).unwrap()
            val dataT = List(elemCount) { BF16.fromBits(dataU16[it]) }
            CandleResult.Ok(Tensor.fromVecBf16(dataT, shape, Device.Cpu))
        }
        DType.F16 -> {
            val dataU16 = UShortArray(elemCount)
            reader.readU16Into(dataU16).unwrap()
            val dataT = List(elemCount) { F16.fromBits(dataU16[it]) }
            CandleResult.Ok(Tensor.fromVecF16(dataT, shape, Device.Cpu))
        }
        DType.F32 -> {
            val dataT = FloatArray(elemCount)
            reader.readF32Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSlice(dataT, shape, Device.Cpu))
        }
        DType.F64 -> {
            val dataT = DoubleArray(elemCount)
            reader.readF64Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSliceF64(dataT, shape, Device.Cpu))
        }
        DType.U8 -> {
            val dataBytes = ByteArray(elemCount)
            reader.readExact(dataBytes).unwrap()
            val dataT = UByteArray(elemCount) { dataBytes[it].toUByte() }
            CandleResult.Ok(Tensor.fromSliceU8(dataT, shape, Device.Cpu))
        }
        DType.U32 -> {
            val dataT = UIntArray(elemCount)
            reader.readU32Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSliceU32(dataT, shape, Device.Cpu))
        }
        DType.I16 -> {
            val dataT = ShortArray(elemCount)
            reader.readI16Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSliceI16(dataT, shape, Device.Cpu))
        }
        DType.I32 -> {
            val dataT = IntArray(elemCount)
            reader.readI32Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSliceI32(dataT, shape, Device.Cpu))
        }
        DType.I64 -> {
            val dataT = LongArray(elemCount)
            reader.readI64Into(dataT).unwrap()
            CandleResult.Ok(Tensor.fromSliceI64(dataT, shape, Device.Cpu))
        }
        DType.F8E4M3 -> {
            val dataT = ByteArray(elemCount)
            reader.readExact(dataT).unwrap()
            val dataF8 = List(elemCount) { F8E4M3.fromBits(dataT[it].toUByte()) }
            CandleResult.Ok(Tensor.fromVecF8E4M3(dataF8, shape, Device.Cpu))
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            CandleResult.Err(Error.UnsupportedDTypeForOp(dtype, "from_reader").bt())
        }
    }
}

/** Reads a npy file and return the stored multi-dimensional array as a tensor. */
public fun readNpy(path: String): CandleResult<Tensor> {
    val reader = File.open(path).unwrap()
    val header = readHeader(reader).unwrap()
    val parsedHeader = Header.parse(header).unwrap()
    if (parsedHeader.fortranOrder) {
        return CandleResult.Err(Error.Npy("fortran order not supported".toString()))
    }
    return Tensor.fromReader(parsedHeader.shape(), parsedHeader.descr, reader)
}

/** Reads a npz file and returns the stored multi-dimensional arrays together with their names. */
public fun readNpz(path: String): CandleResult<List<Pair<String, Tensor>>> {
    val zipReader = BufReader.new(File.open(path).unwrap())
    val zip = zip.ZipArchive.new(zipReader).unwrap()
    val result = mutableListOf<Pair<String, Tensor>>()
    for (i in 0 until zip.len()) {
        val reader = zip.byIndex(i).unwrap()
        val rawName = reader.name()
        val name = rawName.removeSuffix(NPY_SUFFIX)
        val header = readHeader(reader).unwrap()
        val parsedHeader = Header.parse(header).unwrap()
        if (parsedHeader.fortranOrder) {
            return CandleResult.Err(Error.Npy("fortran order not supported".toString()))
        }
        val s = Tensor.fromReader(parsedHeader.shape(), parsedHeader.descr, reader).unwrap()
        result.add(Pair(name, s))
    }
    return CandleResult.Ok(result)
}

/** Reads a npz file and returns the stored multi-dimensional arrays for some specified names. */
public fun readNpzByName(path: String, names: Array<String>): CandleResult<List<Tensor>> {
    val zipReader = BufReader.new(File.open(path).unwrap())
    val zip = zip.ZipArchive.new(zipReader).unwrap()
    val result = mutableListOf<Tensor>()
    for (name in names.iterator()) {
        val res = zip.byName(format("{name}$NPY_SUFFIX", name))
        val reader = when (res) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(
                Error.Npy(format("no array for {name} in $path", name)),
            )
        }
        val header = readHeader(reader).unwrap()
        val parsedHeader = Header.parse(header).unwrap()
        if (parsedHeader.fortranOrder) {
            return CandleResult.Err(Error.Npy("fortran order not supported".toString()))
        }
        val s = Tensor.fromReader(parsedHeader.shape(), parsedHeader.descr, reader).unwrap()
        result.add(s)
    }
    return CandleResult.Ok(result)
}

public fun Tensor.write(f: Write, self: Tensor = this): CandleResult<Unit> {
    f.writeAll(NPY_MAGIC_STRING).unwrap()
    f.writeAll(byteArrayOf(1, 0)).unwrap()
    val headerObj = Header(
        descr = self.dtype(),
        fortranOrder = false,
        shape = self.dims().toList(),
    )
    var header = headerObj.toString().unwrap()
    val pad = 16 - (NPY_MAGIC_STRING.size + 5 + header.length) % 16
    for (i in 0 until (pad % 16)) {
        header += ' '
    }
    header += 10.toChar()
    f.writeAll(byteArrayOf((header.length % 256).toByte(), (header.length / 256).toByte())).unwrap()
    f.writeAll(header.encodeToByteArray()).unwrap()
    val writer = f as? NpyByteWriter ?: NpyByteWriter()
    self.writeBytes(writer).unwrap()
    if (f !== writer) {
        f.writeAll(writer.toByteArray()).unwrap()
    }
    return CandleResult.Ok(Unit)
}

/** Writes a multi-dimensional array in the npy format. */
public fun Tensor.writeNpy(path: String, self: Tensor = this): CandleResult<Unit> {
    val f = File.create(path).unwrap()
    return self.write(f)
}

/** Writes multiple multi-dimensional arrays using the npz format. */
public fun writeNpz(ts: List<Pair<String, Tensor>>, path: String): CandleResult<Unit> {
    val zipWriter = zip.ZipWriter.new(File.create(path).unwrap())
    val options: zip.write.FileOptions<Unit> =
        zip.write.FileOptions.default().compressionMethod(zip.CompressionMethod.Stored)

    for (entry in ts.iterator()) {
        val name = entry.first
        val tensor = entry.second
        zipWriter.startFile(format("{}.npy", name), options).unwrap()
        tensor.write(zipWriter).unwrap()
    }
    return CandleResult.Ok(Unit)
}

/** Lazy tensor loader. */
public class NpzTensors private constructor(
    private val indexPerName: Map<String, Int>,
    private val path: String,
) {
    public fun names(self: NpzTensors = this): List<String> {
        return self.indexPerName.keys.toList()
    }

    /**
     * This only returns the shape and dtype for a named tensor. Compared to `get`, this avoids
     * reading the whole tensor data.
     */
    public fun getShapeAndDtype(name: String, self: NpzTensors = this): CandleResult<Pair<Shape, DType>> {
        val idx = self.indexPerName[name]
        val index = when (idx) {
            null -> bail("cannot find tensor $name")
            else -> idx
        }
        val zipReader = BufReader.new(File.open(self.path).unwrap())
        val zip = zip.ZipArchive.new(zipReader).unwrap()
        val reader = zip.byIndex(index).unwrap()
        val header = readHeader(reader).unwrap()
        val parsedHeader = Header.parse(header).unwrap()
        return CandleResult.Ok(Pair(parsedHeader.shape(), parsedHeader.descr))
    }

    public fun get(name: String, self: NpzTensors = this): CandleResult<Tensor?> {
        val idx = self.indexPerName[name]
        val index = when (idx) {
            null -> return CandleResult.Ok(null)
            else -> idx
        }
        // We hope that the file has not changed since first reading it.
        val zipReader = BufReader.new(File.open(self.path).unwrap())
        val zip = zip.ZipArchive.new(zipReader).unwrap()
        val reader = zip.byIndex(index).unwrap()
        val header = readHeader(reader).unwrap()
        val parsedHeader = Header.parse(header).unwrap()
        if (parsedHeader.fortranOrder) {
            return CandleResult.Err(Error.Npy("fortran order not supported".toString()))
        }
        val tensor = Tensor.fromReader(parsedHeader.shape(), parsedHeader.descr, reader).unwrap()
        return CandleResult.Ok(tensor)
    }

    public companion object Self {
        public fun new(path: String): CandleResult<NpzTensors> {
            val pathOwned = path
            val zipReader = BufReader.new(File.open(pathOwned).unwrap())
            val zip = zip.ZipArchive.new(zipReader).unwrap()
            val indexPerName = HashMap<String, Int>()
            for (i in 0 until zip.len()) {
                val file = zip.byIndex(i).unwrap()
                val rawName = file.name()
                val name = rawName.removeSuffix(NPY_SUFFIX)
                indexPerName[name] = i
            }
            return CandleResult.Ok(
                NpzTensors(
                    indexPerName = indexPerName,
                    path = pathOwned,
                ),
            )
        }
    }
}

internal class NpyTests {
    @Test
    public fun parse() {
        val h = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }"
        assertEquals(
            Header.parse(h).unwrap(),
            Header(
                descr = DType.F64,
                fortranOrder = false,
                shape = listOf(128),
            ),
        )
        val h2Str = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }"
        val h2 = Header.parse(h2Str).unwrap()
        assertEquals(
            h2,
            Header(
                descr = DType.F32,
                fortranOrder = true,
                shape = listOf(256, 1, 128),
            ),
        )
        assertEquals(
            h2.toString().unwrap(),
            "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }",
        )

        val h3 = Header(
            descr = DType.U32,
            fortranOrder = false,
            shape = emptyList(),
        )
        assertEquals(
            h3.toString().unwrap(),
            "{'descr': '<u4', 'fortran_order': False, 'shape': (), }",
        )
    }
}
