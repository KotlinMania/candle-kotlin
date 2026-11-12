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
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.fs.File
import io.github.kotlinmania.rustlang.std.io.BufReader
import io.github.kotlinmania.rustlang.std.io.Read
import io.github.kotlinmania.rustlang.std.io.write
import io.github.kotlinmania.rustlang.std.path.Path

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
    when (val tryResult = reader.readExact(magicString)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    if (magicString != NPY_MAGIC_STRING) {
        return Err(Error.Npy("magic string mismatch".toString()))
    }
    val version = List(2) { 0u.toUByte() }
    when (val tryResult = reader.readExact(version)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val headerLenLen =
        when (val matchVal = version[0]) {
            1 -> 2
            2 -> 4
            else -> {
                val otherwise = matchVal
                return Err(Error.Npy("unsupported version $otherwise"))
            }
        }
    val headerLen = mutableListOf(0u.toUByte(), headerLenLen)
    when (val tryResult = reader.readExact(headerLen)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val headerLen1 = ArrayDeque(headerLen).rev().fold(0, { acc, v -> 256 * acc + v.toInt() })
    val header = mutableListOf(0u.toUByte(), headerLen1)
    when (val tryResult = reader.readExact(header)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return Ok(String.fromUtf8Lossy(header).toString())
}

public data class Header(
    public var descr: DType = DType.entries.first(),
    public var fortranOrder: Boolean = false,
    public var shape: List<Int> = emptyList(),
) {
    public override fun toString(): Result<String> {
        val fortranOrder =
            if (this.fortranOrder) {
                "True"
            } else {
                "False"
            }
        val shape =
            this.shape
                .map({ x -> x.toString() })
                .collect<Vec>()
                .join(",")
        val descr =
            when (this.descr) {
                DType.bf16 ->
                    when (val tryResult = Err(Error.Npy("bf16 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f16 -> "f2"
                DType.f32 -> "f4"
                DType.f64 -> "f8"
                DType.i16 -> "i2"
                DType.i32 -> "i4"
                DType.i64 -> "i8"
                DType.u32 -> "u4"
                DType.u8 -> "u1"
                DType.f8e4m3 ->
                    when (val tryResult = Err(Error.Npy("f8e4m3 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f6e2m3 ->
                    when (val tryResult = Err(Error.Npy("f6e2m3 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f6e3m2 ->
                    when (val tryResult = Err(Error.Npy("f6e3m2 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f4 ->
                    when (val tryResult = Err(Error.Npy("f4 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                DType.f8e8m0 ->
                    when (val tryResult = Err(Error.Npy("f8e8m0 is not supported".into()))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            }
        if (!shape.isEmpty()) {
            shape.push(',')
        }
        return Ok("{'descr': '<$descr', 'fortran_order': $fortranOrder, 'shape': ($shape), }")
    }

    public companion object {
        // Hacky parser for the npy header, a typical example would be:
        // {'descr': '<f8', 'fortranOrder': False, 'shape': (128,), }
        public fun parse(header: String): Result<Header> {
            val header1 = header.trimMatches({ c: Char -> c == '{' || c == '}' || c == ',' || c.isWhitespace() })
            val parts: List<String> = mutableListOf()
            var startIndex = 0
            var cntParenthesis = 0L
            for ((index, c) in header1.charIndices()) {
                when {
                    c == '(' -> cntParenthesis += 1L
                    c == ')' -> cntParenthesis -= 1L
                    c == ',' && cntParenthesis == 0L -> {
                        parts.add(((header1).slice(startIndex until index)).toOwned())
                        startIndex = index + 1
                    }
                    else -> {}
                }
            }
            parts.add(((header1).slice(startIndex until header1.size)).toOwned())
            val partMap: Map<String, String> = HashMap.new()
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
                            else -> {
                                return Err(Error.Npy("unable to parse header $header"))
                            }
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
                            else -> {
                                return Err(Error.Npy("unknown fortran_order $fortranOrder"))
                            }
                        }
                    }
                }
            val descr2 =
                when (val matchVal2 = partMap.get("descr")) {
                    null -> {
                        return Err(Error.Npy("no descr in header".toString()))
                    }
                    else -> {
                        val descr = matchVal2
                        if (descr.isEmpty()) {
                            return Err(Error.Npy("empty descr".toString()))
                        }
                        if (descr.startsWith('>')) {
                            return Err(Error.Npy("little-endian descr $descr"))
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
                                return Err(Error.Npy("unrecognized descr $descr"))
                            }
                        }
                    }
                }
            val shape2 =
                when (val matchVal4 = partMap.get("shape")) {
                    null -> {
                        return Err(Error.Npy("no shape in header".toString()))
                    }
                    else -> {
                        val shape = matchVal4
                        val shape1 = shape.trimMatches({ c1: Char -> c1 == '(' || c1 == ')' || c1 == ',' })
                        if (shape1.isEmpty()) {
                            mutableListOf()
                        } else {
                            when (val tryResult = shape1.split(',').map({ v -> v.trim().parse<Int>() }).collect<Result<Vec>>()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        }
                    }
                }
            return Ok(Header(descr = descr2, fortranOrder = fortranOrder1, shape = shape2))
        }
    }
}

// Note: Add the possibility to read directly to a device?
internal fun <R : Read> Tensor.Companion.fromReader(shape: Shape, dtype: DType, reader: R): Result<Tensor> {
    val elemCount = shape.elemCount()
    return when (dtype) {
        DType.bf16 -> {
            val dataT = mutableListOf(bf16.ZERO, elemCount)
            when (val tryResult = reader.readU16Into<LittleEndian>(dataT.reinterpretCastMut())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT, shape, Device.Cpu)
        }
        DType.f16 -> {
            val dataT1 = mutableListOf(f16.ZERO, elemCount)
            when (val tryResult = reader.readU16Into<LittleEndian>(dataT1.reinterpretCastMut())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT1, shape, Device.Cpu)
        }
        DType.f32 -> {
            val dataT2 = mutableListOf(0.0f, elemCount)
            when (val tryResult = reader.readF32Into<LittleEndian>(dataT2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT2, shape, Device.Cpu)
        }
        DType.f64 -> {
            val dataT3 = mutableListOf(0.0, elemCount)
            when (val tryResult = reader.readF64Into<LittleEndian>(dataT3)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT3, shape, Device.Cpu)
        }
        DType.u8 -> {
            val dataT4 = mutableListOf(0u.toUByte(), elemCount)
            when (val tryResult = reader.readExact(dataT4)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT4, shape, Device.Cpu)
        }
        DType.u32 -> {
            val dataT5 = mutableListOf(0u, elemCount)
            when (val tryResult = reader.readU32Into<LittleEndian>(dataT5)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT5, shape, Device.Cpu)
        }
        DType.i16 -> {
            val dataT6 = mutableListOf(0, elemCount)
            when (val tryResult = reader.readI16Into<LittleEndian>(dataT6)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT6, shape, Device.Cpu)
        }
        DType.i32 -> {
            val dataT7 = mutableListOf(0, elemCount)
            when (val tryResult = reader.readI32Into<LittleEndian>(dataT7)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT7, shape, Device.Cpu)
        }
        DType.i64 -> {
            val dataT8 = mutableListOf(0L, elemCount)
            when (val tryResult = reader.readI64Into<LittleEndian>(dataT8)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Tensor.fromVec(dataT8, shape, Device.Cpu)
        }
        DType.f8e4m3 -> {
            val dataT9 = mutableListOf(0u.toUByte(), elemCount)
            when (val tryResult = reader.readExact(dataT9)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            val dataF8: List<F8e4m3> = ArrayDeque((ArrayDeque(dataT9)).map(F8e4m3::fromBits)).collect()
            Tensor.fromVec(dataF8, shape, Device.Cpu)
        }
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            Err(Error.UnsupportedDTypeForOp(dtype, "from_reader").bt())
        }
    }
}

/**
 * Reads a npy file and return the stored multi-dimensional array as a tensor.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpy(path: T): Result<Tensor> {
    val reader =
        when (val tryResult = File.open(path.asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val header =
        when (val tryResult = readHeader(reader)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val header1 =
        when (val tryResult = Header.parse(header)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    if (header1.fortranOrder) {
        return Err(Error.Npy("fortran order not supported".toString()))
    }
    return Tensor.fromReader(header1.shape(), header1.descr, reader)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays together with their names.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpz(path: T): Result<List<Pair<String, Tensor>>> {
    val zipReader =
        BufReader.new(
            when (val tryResult = File.open(path.asRef())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    val zip =
        when (val tryResult = ZipArchive.new(zipReader)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val result = mutableListOf<Pair<String, Tensor>>()
    for (i in 0 until zip.len()) {
        val reader =
            when (val tryResult = zip.byIndex(i)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val name1 =
            run {
                val name = reader.name()
                (name.takeIf { it.endsWith(NPY_SUFFIX) }?.removeSuffix(NPY_SUFFIX) ?: name).toOwned()
            }
        val header =
            when (val tryResult = readHeader(reader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header1 =
            when (val tryResult = Header.parse(header)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (header1.fortranOrder) {
            return Err(Error.Npy("fortran order not supported".toString()))
        }
        val s =
            when (val tryResult = Tensor.fromReader(header1.shape(), header1.descr, reader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        result.add(Pair(name1, s))
    }
    return Ok(result)
}

/**
 * Reads a npz file and returns the stored multi-dimensional arrays for some specified names.
 */
public fun <T : AsRef<Path>> Tensor.Companion.readNpzByName(path: T, names: List<String>): Result<List<Tensor>> {
    val zipReader =
        BufReader.new(
            when (val tryResult = File.open(path.asRef())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    val zip =
        when (val tryResult = ZipArchive.new(zipReader)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val result = mutableListOf<Tensor>()
    for (name in names.iter()) {
        var reader1 =
            run {
                val matchVal = zip.byName("$name$npySuffix")
                when (matchVal) {
                    is Result.Ok -> {
                        val reader = matchVal.value
                        reader
                    }
                    is Result.Err -> {
                        when (val tryResult = Err(Error.Npy("no array for $name in ${path.asRef()}"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
            }
        val header =
            when (val tryResult = readHeader(reader1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header1 =
            when (val tryResult = Header.parse(header)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (header1.fortranOrder) {
            return Err(Error.Npy("fortran order not supported".toString()))
        }
        val s =
            when (val tryResult = Tensor.fromReader(header1.shape(), header1.descr, reader1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        result.add(s)
    }
    return Ok(result)
}

public fun <T : Write> Tensor.write(f: T): Result<Unit> {
    when (val tryResult = f.writeAll(NPY_MAGIC_STRING)) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    when (val tryResult = f.writeAll(listOf(1u.toUByte(), 0u.toUByte()))) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val header = Header(descr = this.dtype(), fortranOrder = false, shape = this.dims().toVec())
    val header1 =
        when (val tryResult = header.toString()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val pad = 16 - (NPY_MAGIC_STRING.size + 5 + header1.len()) % 16
    for (unused in 0 until pad % 16) {
        header1.push(' ')
    }
    header1.push('\n')
    when (val tryResult = f.writeAll(listOf((header1.len() % 256).toUByte(), (header1.len() / 256).toUByte()))) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    when (val tryResult = f.writeAll(header1.encodeToByteArray().asUByteArray())) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    return this.writeBytes(f)
}

/**
 * Writes a multi-dimensional array in the npy format.
 */
public fun <T : AsRef<Path>> Tensor.writeNpy(path: T): Result<Unit> {
    val f =
        when (val tryResult = File.create(path.asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return this.write(f)
}

/**
 * Writes multiple multi-dimensional arrays using the npz format.
 */
public fun <S : AsRef<String>, T : AsRef<Tensor>, P : AsRef<Path>> Tensor.Companion.writeNpz(ts: List<Pair<S, T>>, path: P): Result<Unit> {
    val zip =
        ZipWriter.new(
            when (val tryResult = File.create(path.asRef())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    val options: FileOptions<Unit> = FileOptions.default().compressionMethod(CompressionMethod.Stored)
    for ((name, tensor) in ts.iter()) {
        when (val tryResult = zip.startFile("${name.asRef()}.npy", options)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        when (val tryResult = tensor.asRef().write(zip)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    }
    return Ok(Unit)
}

/**
 * Lazy tensor loader.
 */
public data class NpzTensors(
    public var indexPerName: Map<String, Int> = emptyMap(),
    public var path: PathBuf = PathBuf(),
) {
    public fun names(): List<String> = this.indexPerName.keys().collect()

    /**
     * This only returns the shape and dtype for a named tensor. Compared to `get`, this avoids
     * reading the whole tensor data.
     */
    public fun getShapeAndDtype(name: String): Result<Pair<Shape, DType>> {
        val index1 =
            when (val matchVal = this.indexPerName.get(name)) {
                null -> bail("cannot find tensor {name}")
                else -> {
                    val index = matchVal
                    index
                }
            }
        val zipReader =
            BufReader.new(
                when (val tryResult = File.open(this.path)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                },
            )
        val zip =
            when (val tryResult = ZipArchive.new(zipReader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val reader =
            when (val tryResult = zip.byIndex(index1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header =
            when (val tryResult = readHeader(reader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header1 =
            when (val tryResult = Header.parse(header)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(Pair(header1.shape(), header1.descr))
    }

    public fun get(name: String): Result<Tensor?> {
        val index1 =
            when (val matchVal = this.indexPerName.get(name)) {
                null -> {
                    return Ok(null)
                }
                else -> {
                    val index = matchVal
                    index
                }
            }
        // We hope that the file has not changed since first reading it.
        val zipReader =
            BufReader.new(
                when (val tryResult = File.open(this.path)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                },
            )
        val zip =
            when (val tryResult = ZipArchive.new(zipReader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val reader =
            when (val tryResult = zip.byIndex(index1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header =
            when (val tryResult = readHeader(reader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val header1 =
            when (val tryResult = Header.parse(header)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (header1.fortranOrder) {
            return Err(Error.Npy("fortran order not supported".toString()))
        }
        val tensor =
            when (val tryResult = Tensor.fromReader(header1.shape(), header1.descr, reader)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(tensor)
    }

    public companion object {
        public fun <T : AsRef<Path>> new(path: T): Result<NpzTensors> {
            val path1 = path.asRef().toOwned()
            val zipReader =
                BufReader.new(
                    when (val tryResult = File.open(path1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            val zip =
                when (val tryResult = ZipArchive.new(zipReader)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            var indexPerName = HashMap.new()
            for (i in 0 until zip.len()) {
                val file =
                    when (val tryResult = zip.byIndex(i)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val name1 =
                    run {
                        val name = file.name()
                        (name.takeIf { it.endsWith(NPY_SUFFIX) }?.removeSuffix(NPY_SUFFIX) ?: name).toOwned()
                    }
                indexPerName.insert(name1, i)
            }
            return Ok(NpzTensors(indexPerName = indexPerName, path = path1))
        }
    }
}
