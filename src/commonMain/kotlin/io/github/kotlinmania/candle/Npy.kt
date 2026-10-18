// port-lint: source npy.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.float8.F8E4M3
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16

private val NPY_MAGIC_STRING: ByteArray =
    byteArrayOf(
        0x93.toByte(),
        'N'.code.toByte(),
        'U'.code.toByte(),
        'M'.code.toByte(),
        'P'.code.toByte(),
        'Y'.code.toByte(),
    )
private const val NPY_SUFFIX: String = ".npy"

private fun Err(e: CandleException): Nothing = throw e

private val String.into: () -> String get() = { this }
private val String.asRef: () -> String get() = { this }
private val String.toOwned: () -> String get() = { this }
private val String.len: () -> Int get() = { length }
private val String.asBytes: () -> ByteArray get() = { encodeToByteArray() }
private val String.stripSuffix: (String) -> String? get() = { s -> if (endsWith(s)) removeSuffix(s) else null }
private val String?.unwrapOr: (String) -> String get() = { def -> this ?: def }
private val String.charIndices: () -> Iterable<IndexedValue<Char>> get() = { withIndex() }
private val String.trimMatches: ((Char) -> Boolean) -> String get() = { pred -> trim(pred) }
private val String.parse: () -> Int get() = {
    toIntOrNull() ?: throw CandleException
        .ParseInt("invalid int: $this")
}
private val ByteArray.len: () -> Int get() = { size }
private val ByteArray.iter: () -> List<Byte> get() = { toList() }
private val ByteArray.intoIter: () -> Sequence<UByte> get() = { asSequence().map { it.toUByte() } }
private val List<Int>.asSlice: () -> IntArray get() = { toIntArray() }
private val List<String>.join: (String) -> String get() = { sep -> joinToString(sep) }

private fun <T> Set<T>.collect(): List<T> = toList()

private fun <K, V> Map<K, V>.keys(): Set<K> = this.keys

private val IntArray.toVec: () -> List<Int> get() = { toList() }
private val Tensor.asRef: () -> Tensor get() = { this }
private val Tensor.self: Tensor get() = this
private val Header.self: Header get() = this
private val NpzTensors.self: NpzTensors get() = this
private val Self: Tensor.Companion get() = Tensor.Companion

private fun Tensor.Companion.fromVecBf16(
    data: List<BF16>,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromVecImpl(data, shape, device, false)

private fun Tensor.Companion.fromVecF16(
    data: List<F16>,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromVecImpl(data, shape, device, false)

private fun Tensor.Companion.fromSliceF64(
    data: DoubleArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromSlice(data, shape, device)

private fun Tensor.Companion.fromSliceU8(
    data: UByteArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromSlice(data, shape, device)

private fun Tensor.Companion.fromSliceU32(
    data: UIntArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromSlice(data, shape, device)

private fun Tensor.Companion.fromSliceI16(
    data: ShortArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromStorage(Storage.Cpu(CpuStorage.I16(data.copyOf())), shape, BackpropOp.none(), false)

private fun Tensor.Companion.fromSliceI32(
    data: IntArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromSlice(data, shape, device)

private fun Tensor.Companion.fromSliceI64(
    data: LongArray,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromSlice(data, shape, device)

private fun Tensor.Companion.fromVecF8E4M3(
    data: List<F8E4M3>,
    shape: Shape,
    device: Device,
): Tensor = Tensor.fromVecImpl(data, shape, device, false)

private val Tensor.writeBytesToWrite: (Write) -> Unit get() = { f ->
    val writer = f as? NpyByteWriter ?: NpyByteWriter()
    this.writeBytes(writer).unwrap()
    if (f !== writer) f.writeAll(writer.toByteArray()).unwrap()
}

public fun <R : Read> readHeader(reader: R): String {
    val magicString = ByteArray(NPY_MAGIC_STRING.len())
    reader.readExact(magicString).unwrap()
    if (!magicString.contentEquals(NPY_MAGIC_STRING)) {
        Err(CandleException.Npy("magic string mismatch"))
    }
    val version = ByteArray(2)
    reader.readExact(version).unwrap()
    val headerLenLen =
        when (val otherwise = version[0].toInt() and 0xFF) {
            1 -> 2
            2 -> 4
            else -> Err(CandleException.Npy("unsupported version $otherwise"))
        }
    val headerLenBytes = ByteArray(headerLenLen)
    reader.readExact(headerLenBytes).unwrap()
    val headerLen =
        headerLenBytes
            .iter()
            .rev()
            .fold(0) { acc, v -> 256 * acc + (v.toInt() and 0xFF) }
    val header = ByteArray(headerLen)
    reader.readExact(header).unwrap()
    return Ok(String.fromUtf8Lossy(header))
}

public class Header(
    public val descr: DType,
    public val fortranOrder: Boolean,
    internal val shape: List<Int>,
) {
    public fun shape(): Shape = Shape.from(self.shape.asSlice())

    public fun copy(
        descr: DType = this.descr,
        fortranOrder: Boolean = this.fortranOrder,
        shape: List<Int> = this.shape,
    ): Header = Header(descr, fortranOrder, shape)

    override fun equals(other: Any?): Boolean =
        other is Header && descr == other.descr && fortranOrder == other.fortranOrder && shape == other.shape

    override fun hashCode(): Int = 31 * (31 * descr.hashCode() + fortranOrder.hashCode()) + shape.hashCode()

    override fun toString(): String {
        val fortranOrder = if (self.fortranOrder) "True" else "False"
        var shape =
            self
                .shape
                .iter()
                .map { x -> x.toString() }
                .collect()
                .join(",")
        val descr =
            when (self.descr) {
                DType.BF16 -> Err(CandleException.Npy("bf16 is not supported".into()))
                DType.F16 -> "f2"
                DType.F32 -> "f4"
                DType.F64 -> "f8"
                DType.I16 -> "i2"
                DType.I32 -> "i4"
                DType.I64 -> "i8"
                DType.U32 -> "u4"
                DType.U8 -> "u1"
                DType.F8E4M3 -> Err(CandleException.Npy("f8e4m3 is not supported".into()))
                DType.F6E2M3 -> Err(CandleException.Npy("f6e2m3 is not supported".into()))
                DType.F6E3M2 -> Err(CandleException.Npy("f6e3m2 is not supported".into()))
                DType.F4 -> Err(CandleException.Npy("f4 is not supported".into()))
                DType.F8E8M0 -> Err(CandleException.Npy("f8e8m0 is not supported".into()))
            }
        if (shape.isNotEmpty()) {
            shape += ','
        }
        return Ok(
            "{'descr': '<$descr', 'fortran_order': $fortranOrder, 'shape': ($shape), }",
        )
    }

    public companion object {
        public val Self: Companion get() = this

        // Hacky parser for the npy header, a typical example would be
        // descr f8, fortranOrder False, shape (128,)
        public fun parse(header: String): Header {
            val trimmedHeader =
                header.trimMatches { c: Char -> c == '{' || c == '}' || c == ',' || c.isWhitespace() }

            val parts: MutableList<String> = mutableListOf()
            var startIndex = 0
            var cntParenthesis = 0L
            for ((index, c) in trimmedHeader.charIndices()) {
                when {
                    c == '(' -> {
                        cntParenthesis += 1
                    }

                    c == ')' -> {
                        cntParenthesis -= 1
                    }

                    c == ',' && cntParenthesis == 0L -> {
                        parts.push(trimmedHeader.substring(startIndex, index).toOwned())
                        startIndex = index + 1
                    }

                    else -> {}
                }
            }
            parts.push(trimmedHeader.substring(startIndex).toOwned())
            val partMap: HashMap<String, String> = HashMap()
            for (part in parts.iter()) {
                val trimmedPart = part.trim()
                if (trimmedPart.isNotEmpty()) {
                    val split = trimmedPart.split(':').collect()
                    if (split.size == 2) {
                        val keyTrim = split[0].trimMatches { c: Char -> c == '\'' || c.isWhitespace() }
                        val valTrim = split[1].trimMatches { c: Char -> c == '\'' || c.isWhitespace() }
                        partMap[keyTrim.toOwned()] = valTrim.toOwned()
                    } else {
                        Err(CandleException.Npy("unable to parse header $trimmedHeader"))
                    }
                }
            }
            val fortranOrder =
                when (val fortranOrderVal = partMap["fortran_order"]) {
                    null -> {
                        false
                    }

                    else -> {
                        when (fortranOrderVal.asRef()) {
                            "False" -> false
                            "True" -> true
                            else -> Err(CandleException.Npy("unknown fortran_order $fortranOrderVal"))
                        }
                    }
                }
            val descr =
                when (val descrVal = partMap["descr"]) {
                    null -> {
                        Err(CandleException.Npy("no descr in header"))
                    }

                    else -> {
                        if (descrVal.isEmpty()) {
                            Err(CandleException.Npy("empty descr"))
                        }
                        if (descrVal.startsWith('>')) {
                            Err(CandleException.Npy("little-endian descr $descrVal"))
                        }
                        // the only supported types in tensor are:
                        //     float64, float32, float16,
                        //     complex64, complex128,
                        //     int64, int32, int16, int8,
                        //     uint8, and bool.
                        when (val trimmedDescr = descrVal.trimMatches { c: Char -> c == '=' || c == '<' || c == '|' }) {
                            "e", "f2" -> DType.F16

                            "f", "f4" -> DType.F32

                            "d", "f8" -> DType.F64

                            "i", "i4" -> DType.I32

                            "q", "i8" -> DType.I64

                            "h", "i2" -> DType.I16

                            // "b" | "i1" -> DType.S8,
                            "B", "u1" -> DType.U8

                            "I", "u4" -> DType.U32

                            "?", "b1" -> DType.U8

                            // "F" | "F4" -> DType.C64,
                            // "D" | "F8" -> DType.C128,
                            else -> Err(CandleException.Npy("unrecognized descr $trimmedDescr"))
                        }
                    }
                }
            val shape =
                when (val shapeVal = partMap["shape"]) {
                    null -> {
                        Err(CandleException.Npy("no shape in header"))
                    }

                    else -> {
                        val trimmedShape = shapeVal.trimMatches { c: Char -> c == '(' || c == ')' || c == ',' }
                        if (trimmedShape.isEmpty()) {
                            emptyList()
                        } else {
                            trimmedShape
                                .split(',')
                                .map { v -> v.trim().parse() }
                                .collect()
                        }
                    }
                }
            return Ok(
                Header(
                    descr = descr,
                    fortranOrder = fortranOrder,
                    shape = shape,
                ),
            )
        }
    }
}

public fun <R : Read> Tensor.Companion.fromReader(
    shape: Shape,
    dtype: DType,
    reader: R,
): Tensor {
    val elemCount = shape.elemCount()
    return when (dtype) {
        DType.BF16 -> {
            val dataU16 = UShortArray(elemCount)
            reader.readU16Into(dataU16).unwrap()
            Tensor.fromVecBf16(List(elemCount) { BF16.fromBits(dataU16[it]) }, shape, Device.Cpu)
        }

        DType.F16 -> {
            val dataU16 = UShortArray(elemCount)
            reader.readU16Into(dataU16).unwrap()
            Tensor.fromVecF16(List(elemCount) { F16.fromBits(dataU16[it]) }, shape, Device.Cpu)
        }

        DType.F32 -> {
            val dataT = FloatArray(elemCount)
            reader.readF32Into(dataT).unwrap()
            Tensor.fromSlice(dataT, shape, Device.Cpu)
        }

        DType.F64 -> {
            val dataT = DoubleArray(elemCount)
            reader.readF64Into(dataT).unwrap()
            Tensor.fromSliceF64(dataT, shape, Device.Cpu)
        }

        DType.U8 -> {
            val dataBytes = ByteArray(elemCount)
            reader.readExact(dataBytes).unwrap()
            Tensor.fromSliceU8(UByteArray(elemCount) { dataBytes[it].toUByte() }, shape, Device.Cpu)
        }

        DType.U32 -> {
            val dataT = UIntArray(elemCount)
            reader.readU32Into(dataT).unwrap()
            Tensor.fromSliceU32(dataT, shape, Device.Cpu)
        }

        DType.I16 -> {
            val dataT = ShortArray(elemCount)
            reader.readI16Into(dataT).unwrap()
            Tensor.fromSliceI16(dataT, shape, Device.Cpu)
        }

        DType.I32 -> {
            val dataT = IntArray(elemCount)
            reader.readI32Into(dataT).unwrap()
            Tensor.fromSliceI32(dataT, shape, Device.Cpu)
        }

        DType.I64 -> {
            val dataT = LongArray(elemCount)
            reader.readI64Into(dataT).unwrap()
            Tensor.fromSliceI64(dataT, shape, Device.Cpu)
        }

        DType.F8E4M3 -> {
            val dataT = ByteArray(elemCount)
            reader.readExact(dataT).unwrap()
            val dataF8: List<F8E4M3> =
                dataT.intoIter().map { F8E4M3.fromBits(it) }.collect()
            Tensor.fromVecF8E4M3(dataF8, shape, Device.Cpu)
        }

        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            Err(CandleException.UnsupportedDTypeForOp(dtype, "from_reader").bt())
        }
    }
}

/** Reads a npy file and return the stored multi-dimensional array as a tensor. */
public fun readNpy(path: String): Tensor {
    val reader = File.open(path.asRef()).unwrap()
    val rawHeader = readHeader(reader)
    val header = Header.parse(rawHeader)
    if (header.fortranOrder) {
        Err(CandleException.Npy("fortran order not supported"))
    }
    return Self.fromReader(header.shape(), header.descr, reader)
}

/** Reads a npz file and returns the stored multi-dimensional arrays together with their names. */
public fun readNpz(path: String): List<Pair<String, Tensor>> {
    val zipReader = BufReader.new(File.open(path.asRef()).unwrap())
    val zip = Zip.ZipArchive.new(zipReader).unwrap()
    val result = mutableListOf<Pair<String, Tensor>>()
    for (i in 0 until zip.len()) {
        val reader = zip.byIndex(i).unwrap()
        val name =
            run {
                val rawName = reader.name()
                rawName.stripSuffix(NPY_SUFFIX).unwrapOr(rawName).toOwned()
            }
        val rawHeader = readHeader(reader)
        val header = Header.parse(rawHeader)
        if (header.fortranOrder) {
            Err(CandleException.Npy("fortran order not supported"))
        }
        val s = Self.fromReader(header.shape(), header.descr, reader)
        result.push(Pair(name, s))
    }
    return Ok(result)
}

/** Reads a npz file and returns the stored multi-dimensional arrays for some specified names. */
public fun readNpzByName(
    path: String,
    names: Array<String>,
): List<Tensor> {
    val zipReader = BufReader.new(File.open(path.asRef()).unwrap())
    val zip = Zip.ZipArchive.new(zipReader).unwrap()
    val result = mutableListOf<Tensor>()
    for (name in names.iterator()) {
        val reader =
            when (val res = zip.byName("$name$NPY_SUFFIX")) {
                is CandleResult.Ok -> {
                    res.value
                }

                is CandleResult.Err -> {
                    Err(
                        CandleException.Npy(
                            "no array for $name in " +
                                path.asRef(),
                        ),
                    )
                }
            }
        val rawHeader = readHeader(reader)
        val header = Header.parse(rawHeader)
        if (header.fortranOrder) {
            Err(CandleException.Npy("fortran order not supported"))
        }
        val s = Self.fromReader(header.shape(), header.descr, reader)
        result.push(s)
    }
    return Ok(result)
}

public fun <T : Write> Tensor.write(f: T) {
    f.writeAll(NPY_MAGIC_STRING).unwrap()
    f.writeAll(byteArrayOf(1, 0)).unwrap()
    val headerObj =
        Header(
            descr = self.dtype(),
            fortranOrder = false,
            shape = self.dims().toVec(),
        )
    var header = headerObj.toString()
    val pad = 16 - (NPY_MAGIC_STRING.len() + 5 + header.len()) % 16
    repeat(pad % 16) {
        header += ' '
    }
    header += '\n'
    f.writeAll(byteArrayOf((header.len() % 256).toByte(), (header.len() / 256).toByte())).unwrap()
    f.writeAll(header.asBytes()).unwrap()
    self.writeBytesToWrite(f)
}

/** Writes a multi-dimensional array in the npy format. */
public fun Tensor.writeNpy(path: String) {
    val f = File.create(path.asRef()).unwrap()
    self.write(f)
}

/** Writes multiple multi-dimensional arrays using the npz format. */
internal fun writeNpz(
    ts: List<Pair<String, Tensor>>,
    path: String,
) {
    val zip = Zip.ZipWriter.new(File.create(path.asRef()).unwrap())
    val options: io.github.kotlinmania.candle.Zip.write.FileOptions<Unit> =
        io.github.kotlinmania.candle.Zip.write.FileOptions
            .default()
            .compressionMethod(io.github.kotlinmania.candle.Zip.CompressionMethod.Stored)

    for ((name, tensor) in ts.iter()) {
        zip.startFile("${name.asRef()}.npy", options).unwrap()
        tensor.asRef().write(zip)
    }
    Ok(Unit)
}

/** Lazy tensor loader. */
public class NpzTensors private constructor(
    private val indexPerName: Map<String, Int>,
    private val path: String,
) {
    public fun names(): List<String> = self.indexPerName.keys().collect()

    /**
     * This only returns the shape and dtype for a named tensor. Compared to `get`, this avoids
     * reading the whole tensor data.
     */
    public fun getShapeAndDtype(name: String): Pair<Shape, DType> {
        val index =
            when (val idx = self.indexPerName[name]) {
                null -> bail("cannot find tensor $name")
                else -> idx
            }
        val zipReader = BufReader.new(File.open(self.path).unwrap())
        val zip = Zip.ZipArchive.new(zipReader).unwrap()
        val reader = zip.byIndex(index).unwrap()
        val rawHeader = readHeader(reader)
        val header = Header.parse(rawHeader)
        return Ok(Pair(header.shape(), header.descr))
    }

    public fun get(name: String): Tensor? {
        val index =
            when (val idx = self.indexPerName[name]) {
                null -> return Ok(null)
                else -> idx
            }
        // We hope that the file has not changed since first reading it.
        val zipReader = BufReader.new(File.open(self.path).unwrap())
        val zip = Zip.ZipArchive.new(zipReader).unwrap()
        val reader = zip.byIndex(index).unwrap()
        val rawHeader = readHeader(reader)
        val header = Header.parse(rawHeader)
        if (header.fortranOrder) {
            Err(CandleException.Npy("fortran order not supported"))
        }
        val tensor = Tensor.fromReader(header.shape(), header.descr, reader)
        return Ok(tensor)
    }

    public companion object {
        public val Self: Companion get() = this

        public fun new(path: String): NpzTensors {
            val ownedPath = path.asRef().toOwned()
            val zipReader = BufReader.new(File.open(ownedPath).unwrap())
            val zip = Zip.ZipArchive.new(zipReader).unwrap()
            val indexPerName = HashMap<String, Int>()
            for (i in 0 until zip.len()) {
                val file = zip.byIndex(i).unwrap()
                val name =
                    run {
                        val rawName = file.name()
                        rawName.stripSuffix(NPY_SUFFIX).unwrapOr(rawName).toOwned()
                    }
                indexPerName[name] = i
            }
            return Ok(
                NpzTensors(
                    indexPerName = indexPerName,
                    path = ownedPath,
                ),
            )
        }
    }
}

internal class NpyTests {
    @Test
    public fun parse() {
        val h1 = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }"
        assertEquals(
            Header.parse(h1),
            Header(
                descr = DType.F64,
                fortranOrder = false,
                shape = listOf(128),
            ),
        )
        val h2Raw = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }"
        val h2 = Header.parse(h2Raw)
        assertEquals(
            h2,
            Header(
                descr = DType.F32,
                fortranOrder = true,
                shape = listOf(256, 1, 128),
            ),
        )
        assertEquals(
            h2.toString(),
            "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }",
        )
        val h3 =
            Header(
                descr = DType.U32,
                fortranOrder = false,
                shape = emptyList(),
            )
        assertEquals(
            h3.toString(),
            "{'descr': '<u4', 'fortran_order': False, 'shape': (), }",
        )
    }
}
