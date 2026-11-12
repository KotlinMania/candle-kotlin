// port-lint: source npy.rs
package io.github.kotlinmania.candle.npy

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.bf16
import io.github.kotlinmania.half.f16

// Numpy support for tensors.
//
// The spec for the npy format can be found in
// [npy-format](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html).
// The functions from this module can be used to read tensors from npy/npz files
// or write tensors to these files. A npy file contains a single tensor (unnamed)
// whereas a npz file can contain multiple named tensors. npz files are also compressed.
//
// These two formats are easy to use in Python using the numpy library.
//
// ```kotlin
// import numpy as np x = np.arange(10) np.save("test.npy", x) x = np.load("test.npy") values = { "x": x, "xPlusOne": x + 1 } np.savez("test.npz", * * values) values = np.loadz("test.npz")
// ```

public val NPY_MAGIC_STRING: List<UByte> = "NUMPY".encodeToByteArray().map { it.toUByte() }

public val NPY_SUFFIX: String = ".npy"

public fun <R : Read> readHeader(reader: R): Result<String> {
    val magicString = mutableListOf(0u.toUByte(), NPY_MAGIC_STRING.size)
    reader.readExact(magicString).getOrElse { return Result.failure(it) }
    if (magicString != NPY_MAGIC_STRING) {
        return Result.failure(Error.Npy("magic string mismatch".toString()))
    }
    val version = List(2) { 0u.toUByte() }
    reader.readExact(version).getOrElse { return Result.failure(it) }
    val headerLenLen =
        when (val matchVal = version[0]) {
            1 -> 2
            2 -> 4
            else -> {
                val otherwise = matchVal
                return Result.failure(Error.Npy("unsupported version $otherwise"))
            }
        }
    val headerLen = mutableListOf(0u.toUByte(), headerLenLen)
    reader.readExact(headerLen).getOrElse { return Result.failure(it) }
    val headerLen1 = ArrayDeque(headerLen).rev().fold(0, { acc, v -> 256 * acc + v.toInt() })
    val header = mutableListOf(0u.toUByte(), headerLen1)
    reader.readExact(header).getOrElse { return Result.failure(it) }
    return Result.success(String.fromUtf8Lossy(header).toString())
}

public open class Header(
    public var descr: DType,
    public var fortranOrder: Boolean,
    public var shape: List<Int>,
) {
    public open fun shape(): Shape = Shape.from(this.shape.asSlice())

    public override fun toString(): Result<String> {
        val fortranOrder =
            if (this.fortranOrder) {
                "True"
            } else {
                "False"
            }
        val shape = this.shape.map({ x -> x.toString() }).collect<Vec>().join(",")
        val descr =
            when (this.descr) {
                DType.bf16 -> Result.failure(Error.Npy("bf16 is not supported".into())).getOrElse { return Result.failure(it) }
                DType.f16 -> "f2"
                DType.f32 -> "f4"
                DType.f64 -> "f8"
                DType.i16 -> "i2"
                DType.i32 -> "i4"
                DType.i64 -> "i8"
                DType.u32 -> "u4"
                DType.u8 -> "u1"
                DType.f8e4m3 -> Result.failure(Error.Npy("f8e4m3 is not supported".into())).getOrElse { return Result.failure(it) }
                DType.f6e2m3 -> Result.failure(Error.Npy("f6e2m3 is not supported".into())).getOrElse { return Result.failure(it) }
                DType.f6e3m2 -> Result.failure(Error.Npy("f6e3m2 is not supported".into())).getOrElse { return Result.failure(it) }
                DType.f4 -> Result.failure(Error.Npy("f4 is not supported".into())).getOrElse { return Result.failure(it) }
                DType.f8e8m0 -> Result.failure(Error.Npy("f8e8m0 is not supported".into())).getOrElse { return Result.failure(it) }
            }
        if (!shape.isEmpty()) {
            shape.push(',')
        }
        return Result.success("{'descr': '<$descr', 'fortran_order': $fortranOrder, 'shape': ($shape), }")
    }

    public companion object {
        // Hacky parser for the npy header, a typical example would be:
        // {'descr': '<f8', 'fortranOrder': False, 'shape': (128,), }
        public fun parse(header: String): Result<Header> {
            val header1 = header.trimMatches({ c: Char -> c == '{' || c == '}' || c == ',' || c.isWhitespace() })
            val parts: List<String>  = mutableListOf()
            var startIndex = 0
            var cntParenthesis = 0L
            for ((index, c) in header1.charIndices()) {
                when {
                    c == '(' -> cntParenthesis += 1L
                    c == ')' -> cntParenthesis -= 1L
                    c == ',' && cntParenthesis == 0L -> {
                        parts.add(header1[startIndex until index].toOwned())
                        startIndex = index + 1
                    }
                    else -> {}
                }
            }
            parts.add(header1[startIndex until header1.size].toOwned())
            val partMap: Map<String, String>  = HashMap.new()
            for (part in ArrayDeque(parts)) {
                val part1 = part.trim()
                if (!part1.isEmpty()) {
                    run {
                        val matchVal = part1.split(':').collect<Vec>().asSlice()
                        when {
                            matchVal.size == 2 -> {
                                val key = matchVal[0]
                                val value = matchVal[1]
                                val key1 = key.trimMatches({ c1: Char -> c1 == '\'' || c1.isWhitespace() })
                                val value1 = value.trimMatches({ c1: Char -> c1 == '\'' || c1.isWhitespace() })
                                partMap.insert(key1.toOwned(), value1.toOwned())
                            }
                            else -> { return Result.failure(Error.Npy("unable to parse header $header")) }
                        }
                    }
                }
            }
            val fortranOrder1 =
                when (val matchVal1 = partMap.get("fortran_order")) {
                    null -> false
                    else -> {
                        val fortranOrder = matchVal1
                        when (fortranOrder.asRef()) {
                            "False" -> false
                            "True" -> true
                            else -> { return Result.failure(Error.Npy("unknown fortran_order $fortranOrder")) }
                        }
                    }
                }
            val descr2 =
                when (val matchVal2 = partMap.get("descr")) {
                    null -> { return Result.failure(Error.Npy("no descr in header".toString())) }
                    else -> {
                        val descr = matchVal2
                        if (descr.isEmpty()) {
                            return Result.failure(Error.Npy("empty descr".toString()))
                        }
                        if (descr.startsWith('>')) {
                            return Result.failure(Error.Npy("little-endian descr $descr"))
                        }
                        // the only supported types in tensor are:
                        // float64, float32, float16,
                        // complex64, complex128,
                        // int64, int32, int16, int8,
                        // uint8, and bool.
                        when (val matchVal3 = descr.trimMatches({ c1: Char -> c1 == '=' || c1 == '<' || c1 == '|' })) {
                            "e", "f2" -> DType.F16
                            "f", "f4" -> DType.F32
                            "d", "f8" -> DType.F64
                            "i", "i4" -> DType.I32
                            "q", "i8" -> DType.I64
                            "h", "i2" -> DType.I16
                            // "b" | "i1" => DType.S8,
                            "B", "u1" -> DType.U8
                            "I", "u4" -> DType.U32
                            "?", "b1" -> DType.U8
                            // "F" | "F4" => DType.C64,
                            // "D" | "F8" => DType.C128,
                            else -> {
                                val descr1 = matchVal3
                                return Result.failure(Error.Npy("unrecognized descr $descr"))
                            }
                        }
                    }
                }
            val shape2 =
                when (val matchVal4 = partMap.get("shape")) {
                    null -> { return Result.failure(Error.Npy("no shape in header".toString())) }
                    else -> {
                        val shape = matchVal4
                        val shape1 = shape.trimMatches({ c1: Char -> c1 == '(' || c1 == ')' || c1 == ',' })
                        if (shape1.isEmpty()) {
                            mutableListOf()
                        } else {
                            shape1.split(',').map({ v -> v.trim().parse<Int>() }).collect<Result<Vec>>().getOrElse { return Result.failure(it) }
                        }
                    }
                }
            return Result.success(Header(descr = descr2, fortranOrder = fortranOrder1, shape = shape2))
        }
    }
}

// Note: Add the possibility to read directly to a device?
internal fun <R : Read> Tensor.Companion.fromReader(shape: Shape, dtype: DType, reader: R): Result<Tensor> {
    val elemCount = shape.elemCount()
    return when (dtype) {
        DType.bf16 -> {
            val dataT = mutableListOf(bf16.ZERO, elemCount)
            reader.readU16Into<LittleEndian>(dataT.reinterpretCastMut()).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT, shape, Device.Cpu)
        }
        DType.f16 -> {
            val dataT1 = mutableListOf(f16.ZERO, elemCount)
            reader.readU16Into<LittleEndian>(dataT1.reinterpretCastMut()).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT1, shape, Device.Cpu)
        }
        DType.f32 -> {
            val dataT2 = mutableListOf(0.0f, elemCount)
            reader.readF32Into<LittleEndian>(dataT2).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT2, shape, Device.Cpu)
        }
        DType.f64 -> {
            val dataT3 = mutableListOf(0.0, elemCount)
            reader.readF64Into<LittleEndian>(dataT3).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT3, shape, Device.Cpu)
        }
        DType.u8 -> {
            val dataT4 = mutableListOf(0u.toUByte(), elemCount)
            reader.readExact(dataT4).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT4, shape, Device.Cpu)
        }
        DType.u32 -> {
            val dataT5 = mutableListOf(0u, elemCount)
            reader.readU32Into<LittleEndian>(dataT5).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT5, shape, Device.Cpu)
        }
        DType.i16 -> {
            val dataT6 = mutableListOf(0, elemCount)
            reader.readI16Into<LittleEndian>(dataT6).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT6, shape, Device.Cpu)
        }
        DType.i32 -> {
            val dataT7 = mutableListOf(0, elemCount)
            reader.readI32Into<LittleEndian>(dataT7).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT7, shape, Device.Cpu)
        }
        DType.i64 -> {
            val dataT8 = mutableListOf(0L, elemCount)
            reader.readI64Into<LittleEndian>(dataT8).getOrElse { return Result.failure(it) }
            Tensor.fromVec(dataT8, shape, Device.Cpu)
        }
        DType.f8e4m3 -> {
            val dataT9 = mutableListOf(0u.toUByte(), elemCount)
            reader.readExact(dataT9).getOrElse { return Result.failure(it) }
            val dataF8: List<F8e4m3> = ArrayDeque((ArrayDeque(dataT9)).map(F8e4m3::fromBits)).collect()
            Tensor.fromVec(dataF8, shape, Device.Cpu)
        }
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            Result.failure(Error.UnsupportedDTypeForOp(dtype, "from_reader").bt())
        }
    }
}

/**
 * Reads a npy file and return the stored multi-dimensional array as a tensor.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpy(path: T): Result<Tensor> {
    val reader = File.open(path.asRef()).getOrElse { return Result.failure(it) }
    val header = readHeader(reader).getOrElse { return Result.failure(it) }
    val header1 = Header.parse(header).getOrElse { return Result.failure(it) }
    if (header1.fortranOrder) {
        return Result.failure(Error.Npy("fortran order not supported".toString()))
    }
    return Tensor.fromReader(header1.shape(), header1.descr, reader)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays together with their names.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpz(path: T): Result<List<Pair<String, Tensor>>> {
    val zipReader = BufReader.new(File.open(path.asRef()).getOrElse { return Result.failure(it) })
    val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
    val result = mutableListOf<Pair<String, Tensor>>()
    for (i in 0 until zip.len()) {
        val reader = zip.byIndex(i).getOrElse { return Result.failure(it) }
        val name1 = run {
            val name = reader.name()
            (name.takeIf { it.endsWith(NPY_SUFFIX) }?.removeSuffix(NPY_SUFFIX) ?: name).toOwned()
        }
        val header = readHeader(reader).getOrElse { return Result.failure(it) }
        val header1 = Header.parse(header).getOrElse { return Result.failure(it) }
        if (header1.fortranOrder) {
            return Result.failure(Error.Npy("fortran order not supported".toString()))
        }
        val s = Tensor.fromReader(header1.shape(), header1.descr, reader).getOrElse { return Result.failure(it) }
        result.add(Pair(name1, s))
    }
    return Result.success(result)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays for some specified names.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpzByName(path: T, names: List<String>): Result<List<Tensor>> {
    val zipReader = BufReader.new(File.open(path.asRef()).getOrElse { return Result.failure(it) })
    val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
    val result = mutableListOf<Tensor>()
    for (name in names.iter()) {
        var reader1 =
            run {
                val matchVal = zip.byName("$name$npySuffix")
                if (matchVal.isSuccess) {
                    val reader = matchVal.getOrThrow()
                    reader
                } else {
                    Result.failure(Error.Npy("no array for $name in ${path.asRef()}")).getOrElse { return Result.failure(it) }
                }
            }
        val header = readHeader(reader1).getOrElse { return Result.failure(it) }
        val header1 = Header.parse(header).getOrElse { return Result.failure(it) }
        if (header1.fortranOrder) {
            return Result.failure(Error.Npy("fortran order not supported".toString()))
        }
        val s = Tensor.fromReader(header1.shape(), header1.descr, reader1).getOrElse { return Result.failure(it) }
        result.add(s)
    }
    return Result.success(result)
}

public fun <T : Write> Tensor.write(f: T): Result<Unit> {
    f.writeAll(NPY_MAGIC_STRING).getOrElse { return Result.failure(it) }
    f.writeAll(listOf(1u.toUByte(), 0u.toUByte())).getOrElse { return Result.failure(it) }
    val header = Header(descr = this.dtype(), fortranOrder = false, shape = this.dims().toVec())
    val header1 = header.toString().getOrElse { return Result.failure(it) }
    val pad = 16 - (NPY_MAGIC_STRING.size + 5 + header1.len()) % 16
    for (unused in 0 until pad % 16) {
        header1.push(' ')
    }
    header1.push('\n')
    f.writeAll(listOf((header1.len() % 256).toUByte(), (header1.len() / 256).toUByte())).getOrElse { return Result.failure(it) }
    f.writeAll(header1.encodeToByteArray().map { it.toUByte() }).getOrElse { return Result.failure(it) }
    return this.writeBytes(f)
}

/**
 * Writes a multi-dimensional array in the npy format.
 */
public fun <T : AsRef<Path>> Tensor.writeNpy(path: T): Result<Unit> {
    val f = File.create(path.asRef()).getOrElse { return Result.failure(it) }
    return this.write(f)
}

/**
 * Writes multiple multi-dimensional arrays using the npz format.
 */
public fun <S : AsRef<String>, T : AsRef<Tensor>, P : AsRef<Path>> Tensor.Companion.writeNpz(ts: List<Pair<S, T>>, path: P): Result<Unit> {
    val zip = ZipWriter.new(File.create(path.asRef()).getOrElse { return Result.failure(it) })
    val options: FileOptions<Unit> = FileOptions.default().compressionMethod(CompressionMethod.Stored)
    for ((name, tensor) in ts.iter()) {
        zip.startFile("${name.asRef()}.npy", options).getOrElse { return Result.failure(it) }
        tensor.asRef().write(zip).getOrElse { return Result.failure(it) }
    }
    return Result.success(Unit)
}

/**
 * Lazy tensor loader.
 */
public open class NpzTensors(
    public var indexPerName: Map<String, Int>,
    public var path: PathBuf,
) {
    public open fun names(): List<String> = this.indexPerName.keys().collect()

    /**
     * This only returns the shape and dtype for a named tensor. Compared to `get`, this avoids
     * reading the whole tensor data.
     */
    public open fun getShapeAndDtype(name: String): Result<Pair<Shape, DType>> {
        val index1 =
            when (val matchVal = this.indexPerName.get(name)) {
                null -> bail("cannot find tensor {name}")
                else -> {
                    val index = matchVal
                    index
                }
            }
        val zipReader = BufReader.new(File.open(this.path).getOrElse { return Result.failure(it) })
        val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
        val reader = zip.byIndex(index1).getOrElse { return Result.failure(it) }
        val header = readHeader(reader).getOrElse { return Result.failure(it) }
        val header1 = Header.parse(header).getOrElse { return Result.failure(it) }
        return Result.success(Pair(header1.shape(), header1.descr))
    }

    public open fun get(name: String): Result<Tensor?> {
        val index1 =
            when (val matchVal = this.indexPerName.get(name)) {
                null -> { return Result.success(null) }
                else -> {
                    val index = matchVal
                    index
                }
            }
        // We hope that the file has not changed since first reading it.
        val zipReader = BufReader.new(File.open(this.path).getOrElse { return Result.failure(it) })
        val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
        val reader = zip.byIndex(index1).getOrElse { return Result.failure(it) }
        val header = readHeader(reader).getOrElse { return Result.failure(it) }
        val header1 = Header.parse(header).getOrElse { return Result.failure(it) }
        if (header1.fortranOrder) {
            return Result.failure(Error.Npy("fortran order not supported".toString()))
        }
        val tensor = Tensor.fromReader(header1.shape(), header1.descr, reader).getOrElse { return Result.failure(it) }
        return Result.success(tensor)
    }

    public companion object {
        public fun <T : AsRef<Path>> new(path: T): Result<NpzTensors> {
            val path1 = path.asRef().toOwned()
            val zipReader = BufReader.new(File.open(path1).getOrElse { return Result.failure(it) })
            val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
            var indexPerName = HashMap.new()
            for (i in 0 until zip.len()) {
                val file = zip.byIndex(i).getOrElse { return Result.failure(it) }
                val name1 = run {
                    val name = file.name()
                    (name.takeIf { it.endsWith(NPY_SUFFIX) }?.removeSuffix(NPY_SUFFIX) ?: name).toOwned()
                }
                indexPerName.insert(name1, i)
            }
            return Result.success(NpzTensors(indexPerName = indexPerName, path = path1))
        }
    }
}
