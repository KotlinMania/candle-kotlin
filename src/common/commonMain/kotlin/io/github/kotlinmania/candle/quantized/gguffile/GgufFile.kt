// port-lint: source quantized/gguf_file.rs
package io.github.kotlinmania.candle.quantized.gguffile

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QTensor
import io.github.kotlinmania.candle.quantized.ggmlfile.qtensorFromGgml
import io.github.kotlinmania.candle.shape.Shape

// Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
//
// Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

public val DEFAULT_ALIGNMENT: ULong = 32uL

// Caps mirror ggml-org/llama.cpp#19856 (GGUF_MAX_STRING_LENGTH,
// GGUF_MAX_ARRAY_ELEMENTS) and GGML_MAX_DIMS from ggml.h. The depth cap
// bounds recursion through nested `Value.Array` so a crafted file cannot
// overflow the stack with a chain of arrays-of-arrays.
public val GGUF_MAX_STRING_LENGTH: ULong = 1uL shl 30

public val GGUF_MAX_ARRAY_ELEMENTS: ULong = 1uL shl 30

public val GGUF_MAX_TENSOR_DIMS: UInt = 4u

public val GGUF_MAX_VALUE_DEPTH: Int = 64

// `fileSize` is the byte length captured once up front, so this avoids
// seeking to the end and back on every length-prefixed read.
public fun <R : Seek> remainingBytes(reader: R, fileSize: ULong): Result<ULong> {
    val cur = reader.streamPosition().getOrElse { return Result.failure(it) }
    return Result.success(fileSize.saturatingSub(cur))
}

public fun <R : Read> readLength(reader: R, magic: VersionedMagic): Result<ULong> {
    return when (magic) {
        VersionedMagic.GgufV1 -> Result.success((reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }).toULong())
        VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> Result.success(reader.readU64<LittleEndian>().getOrElse { return Result.failure(it) })
    }
}

public sealed class Magic : TryFrom<UInt> {
    public data object Gguf : Magic()

    public fun __rustEq(other: Magic): Boolean = this == other

    public companion object {
        public fun tryFrom(value: UInt): Result<Magic> {
            val magic =
                when (value) {
                    0x46554747, 0x47475546 -> Magic.Gguf
                    else -> bail("unknown magic 0x{value:08x}")
                }
            return Result.success(magic)
        }
    }
}

public sealed class VersionedMagic {
    public data object GgufV1 : VersionedMagic()
    public data object GgufV2 : VersionedMagic()
    public data object GgufV3 : VersionedMagic()

    public fun __rustEq(other: VersionedMagic): Boolean = this == other

    public open fun lengthPrefixSize(): ULong =
        when (this) {
            VersionedMagic.GgufV1 -> 4uL
            VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> 8uL
        }

    public companion object {
        public fun <R : Read> read(reader: R): Result<VersionedMagic> {
            val magic = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
            val magic1 = Magic.tryFrom(magic).getOrElse { return Result.failure(it) }
            val version = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
            val versionedMagic =
                run {
                    val matchVal = Pair(magic1, version)
                    when {
                        matchVal.first == Magic.Gguf && matchVal.second == 1 -> VersionedMagic.GgufV1
                        matchVal.first == Magic.Gguf && matchVal.second == 2 -> VersionedMagic.GgufV2
                        matchVal.first == Magic.Gguf && matchVal.second == 3 -> VersionedMagic.GgufV3
                        else -> bail("gguf: unsupported magic/version {magic:?}/{version}")
                    }
                }
            return Result.success(versionedMagic)
        }
    }
}

public open class TensorInfo(
    public var ggmlDtype: GgmlDType,
    public var shape: Shape,
    public var offset: ULong,
) {
    public open fun <R> read(reader: R, tensorDataOffset: ULong, device: Device): Result<QTensor> where R : Seek, R : Read {
        val tensorElems = this.shape.elemCount()
        val blockSize = this.ggmlDtype.blockSize()
        if (!tensorElems.isMultipleOf(blockSize)) {
            bail("the number of elements {tensor_elems} is not divisible by the block size {block_size}")
        }
        val sizeInBytes = tensorElems / blockSize * this.ggmlDtype.typeSize()
        val tensorStart = tensorDataOffset.saturatingAdd(this.offset)
        val fileSize = reader.seek(SeekFrom.End(0)).getOrElse { return Result.failure(it) }
        val remaining = fileSize.saturatingSub(tensorStart)
        if (sizeInBytes.toULong() > remaining) {
            bail("tensor needs {size_in_bytes} bytes at offset {tensor_start}, only {remaining} remaining in file")
        }
        val rawData = mutableListOf(0u.toUByte(), sizeInBytes)
        reader.seek(SeekFrom.Start(tensorStart)).getOrElse { return Result.failure(it) }
        reader.readExact(rawData).getOrElse { return Result.failure(it) }
        return qtensorFromGgml(this.ggmlDtype, rawData, this.shape.dims().toVec(), device)
    }
}

public open class Content(
    public var magic: VersionedMagic,
    public var metadata: Map<String, Value>,
    public var tensorInfos: Map<String, TensorInfo>,
    public var tensorDataOffset: ULong,
) {
    public open fun <R> tensor(reader: R, name: String, device: Device): Result<QTensor> where R : Seek, R : Read {
        val tensorInfo1 =
            when (val matchVal = this.tensorInfos.get(name)) {
                null -> bail("cannot find tensor info for {name}")
                else -> {
                    val tensorInfo = matchVal
                    tensorInfo
                }
            }
        return tensorInfo1.read(reader, this.tensorDataOffset, device)
    }

    public companion object {
        public fun <R> read(reader: R): Result<Content> where R : Seek, R : Read {
            // Capture the file size once so the bounds checks below don't have to
            // seek to the end and back on every length-prefixed read.
            val start = reader.streamPosition().getOrElse { return Result.failure(it) }
            val fileSize = reader.seek(SeekFrom.End(0)).getOrElse { return Result.failure(it) }
            reader.seek(SeekFrom.Start(start)).getOrElse { return Result.failure(it) }
            val magic = VersionedMagic.read(reader).getOrElse { return Result.failure(it) }
            val tensorCount = readLength(reader, magic).getOrElse { return Result.failure(it) }
            val metadataKvCount = readLength(reader, magic).getOrElse { return Result.failure(it) }
            if (tensorCount > GGUF_MAX_ARRAY_ELEMENTS) {
                bail("gguf: tensor_count {tensor_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
            }
            if (metadataKvCount > GGUF_MAX_ARRAY_ELEMENTS) {
                bail("gguf: metadata_kv_count {metadata_kv_count} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
            }
            // Reject header-declared counts that can't fit in the file at minimum size.
            // Per-entry minima: a metadata kv is at least `keyLenPrefix + u32 valueType
            // + 1 byte value`; a tensor info is at least`nameLenPrefix + u32 nDims
            // + u32 dtype + u64 offset`.
            val prefix = magic.lengthPrefixSize()
            val minPerKv = prefix + 4 + 1
            val minPerTensor = prefix + 4 + 4 + 8
            val needed = metadataKvCount.saturatingMul(minPerKv).saturatingAdd(tensorCount.saturatingMul(minPerTensor))
            val remaining = remainingBytes(reader, fileSize).getOrElse { return Result.failure(it) }
            if (needed > remaining) {
                bail("gguf: header declares {tensor_count} tensors and {metadata_kv_count} metadata entries, needs at least {needed} bytes, only {remaining} remaining")
            }
            var metadata = HashMap.new()
            for (idx in 0 until metadataKvCount) {
                val key = readString(reader, magic, fileSize).getOrElse { return Result.failure(it) }
                val valueType = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                val valueType1 = ValueType.fromU32(valueType).getOrElse { return Result.failure(it) }
                val value = Value.read(reader, valueType1, magic, 0, fileSize).getOrElse { return Result.failure(it) }
                metadata.insert(key, value)
            }
            var tensorInfos = HashMap.new()
            for (idx1 in 0 until tensorCount) {
                val tensorName = readString(reader, magic, fileSize).getOrElse { return Result.failure(it) }
                val nDimensions = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                if (nDimensions > GGUF_MAX_TENSOR_DIMS) {
                    bail("gguf: tensor '{tensor_name}' has {n_dimensions} dimensions, max is {GGUF_MAX_TENSOR_DIMS}")
                }
                var dimensions2: List<Int> =
                    when (magic) {
                        VersionedMagic.GgufV1 -> {
                            val dimensions = mutableListOf(0, nDimensions.toInt())
                            reader.readU32Into<LittleEndian>(dimensions).getOrElse { return Result.failure(it) }
                            ArrayDeque((ArrayDeque(dimensions)).map({ c -> c.toInt() })).collect()
                        }
                        VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> {
                            val dimensions1 = mutableListOf(0, nDimensions.toInt())
                            reader.readU64Into<LittleEndian>(dimensions1).getOrElse { return Result.failure(it) }
                            ArrayDeque((ArrayDeque(dimensions1)).map({ c -> c.toInt() })).collect()
                        }
                    }
                dimensions2.reverse()
                val ggmlDtype = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                val ggmlDtype1 = GgmlDType.fromU32(ggmlDtype).getOrElse { return Result.failure(it) }
                val offset = reader.readU64<LittleEndian>().getOrElse { return Result.failure(it) }
                tensorInfos.insert(tensorName, TensorInfo(shape = Shape.from(dimensions2), offset = offset, ggmlDtype = ggmlDtype1))
            }
            val position = reader.streamPosition().getOrElse { return Result.failure(it) }
            val alignment =
                run {
                    val matchVal = metadata.get("general.alignment")
                    when {
                        matchVal != null && matchVal is Value.U8 -> {
                            val (v) = matchVal as Value.U8
                            v.toULong()
                        }
                        matchVal != null && matchVal is Value.U16 -> {
                            val (v1) = matchVal as Value.U16
                            v1.toULong()
                        }
                        matchVal != null && matchVal is Value.U32 -> {
                            val (v2) = matchVal as Value.U32
                            v2.toULong()
                        }
                        matchVal != null && matchVal is Value.I8 && (matchVal as Value.I8).value >= 0 -> {
                            val (v3) = matchVal as Value.I8
                            v3.toULong()
                        }
                        matchVal != null && matchVal is Value.I16 && (matchVal as Value.I16).value >= 0 -> {
                            val (v4) = matchVal as Value.I16
                            v4.toULong()
                        }
                        matchVal != null && matchVal is Value.I32 && (matchVal as Value.I32).value >= 0 -> {
                            val (v5) = matchVal as Value.I32
                            v5.toULong()
                        }
                        else -> DEFAULT_ALIGNMENT
                    }
                }
            val tensorDataOffset = position.divCeil(alignment) * alignment
            return Result.success(Content(magic = magic, metadata = metadata, tensorInfos = tensorInfos, tensorDataOffset = tensorDataOffset))
        }
    }
}

public fun <R> readString(reader: R, magic: VersionedMagic, fileSize: ULong): Result<String> where R : Read, R : Seek {
    val len = readLength(reader, magic).getOrElse { return Result.failure(it) }
    if (len > GGUF_MAX_STRING_LENGTH) {
        bail("gguf: string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}")
    }
    val remaining = remainingBytes(reader, fileSize).getOrElse { return Result.failure(it) }
    if (len > remaining) {
        bail("gguf: string length {len} exceeds remaining file bytes {remaining}")
    }
    val v = mutableListOf(0u.toUByte(), len.toInt())
    reader.readExact(v).getOrElse { return Result.failure(it) }
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while (true) {
        val matchVal = v.last()
        if (!(matchVal == 0)) break
        v.removeLastOrNull()
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    return Result.success(String.fromUtf8Lossy(v).intoOwned())
}

public sealed class ValueType {
    // The value is a 8-bit unsigned integer.
    public data object U8 : ValueType()
    // The value is a 8-bit signed integer.
    public data object I8 : ValueType()
    // The value is a 16-bit unsigned little-endian integer.
    public data object U16 : ValueType()
    // The value is a 16-bit signed little-endian integer.
    public data object I16 : ValueType()
    // The value is a 32-bit unsigned little-endian integer.
    public data object U32 : ValueType()
    // The value is a 32-bit signed little-endian integer.
    public data object I32 : ValueType()
    // The value is a 64-bit unsigned little-endian integer.
    public data object U64 : ValueType()
    // The value is a 64-bit signed little-endian integer.
    public data object I64 : ValueType()
    // The value is a 32-bit IEEE754 floating point number.
    public data object F32 : ValueType()
    // The value is a 64-bit IEEE754 floating point number.
    public data object F64 : ValueType()
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    public data object Bool : ValueType()
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    public data object String : ValueType()
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    public data object Array : ValueType()

    public fun __rustEq(other: ValueType): Boolean = this == other

    public open fun toU32(): UInt =
        when (this) {
            ValueType.u8 -> 0u
            ValueType.i8 -> 1u
            ValueType.u16 -> 2u
            ValueType.i16 -> 3u
            ValueType.u32 -> 4u
            ValueType.i32 -> 5u
            ValueType.f32 -> 6u
            ValueType.Bool -> 7u
            ValueType.String -> 8u
            ValueType.Array -> 9u
            ValueType.u64 -> 10u
            ValueType.i64 -> 11u
            ValueType.f64 -> 12u
        }

    /**
     * Minimum on-disk size of one value of this type, used to reject array
     * lengths that exceed the remaining file size before allocating.
     */
    public open fun minDiskSize(magic: VersionedMagic): ULong =
        when (this) {
            ValueType.u8, ValueType.i8, ValueType.Bool -> 1uL
            ValueType.u16, ValueType.i16 -> 2uL
            ValueType.u32, ValueType.i32, ValueType.f32 -> 4uL
            ValueType.u64, ValueType.i64, ValueType.f64 -> 8uL
            ValueType.String -> magic.lengthPrefixSize()
            ValueType.Array -> 4uL + magic.lengthPrefixSize()
        }

    public companion object {
        public fun fromU32(v: UInt): Result<ValueType> {
            val v2 =
                when (v) {
                    0 -> ValueType.U8
                    1 -> ValueType.I8
                    2 -> ValueType.U16
                    3 -> ValueType.I16
                    4 -> ValueType.U32
                    5 -> ValueType.I32
                    6 -> ValueType.F32
                    7 -> ValueType.Bool
                    8 -> ValueType.String
                    9 -> ValueType.Array
                    10 -> ValueType.U64
                    11 -> ValueType.I64
                    12 -> ValueType.F64
                    else -> {
                        val v1 = v
                        bail("unrecognized value-type {v:#08x}")
                    }
                }
            return Result.success(v2)
        }
    }
}

public sealed class Value {
    public data class U8(public var value: UByte) : Value()
    public data class I8(public var value: Byte) : Value()
    public data class U16(public var value: UShort) : Value()
    public data class I16(public var value: Short) : Value()
    public data class U32(public var value: UInt) : Value()
    public data class I32(public var value: Int) : Value()
    public data class U64(public var value: ULong) : Value()
    public data class I64(public var value: Long) : Value()
    public data class F32(public var value: Float) : Value()
    public data class F64(public var value: Double) : Value()
    public data class Bool(public var value: Boolean) : Value()
    public data class String(public var value: String) : Value()
    public data class Array(public var value: List<Value>) : Value()

    public open fun valueType(): ValueType =
        when (this) {
            is U8 -> ValueType.U8
            is I8 -> ValueType.I8
            is U16 -> ValueType.U16
            is I16 -> ValueType.I16
            is U32 -> ValueType.U32
            is I32 -> ValueType.I32
            is U64 -> ValueType.U64
            is I64 -> ValueType.I64
            is F32 -> ValueType.F32
            is F64 -> ValueType.F64
            is Bool -> ValueType.Bool
            is String -> ValueType.String
            is Array -> ValueType.Array
        }

    public open fun toU8(): Result<UByte> =
        when (this) {
            is U8 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a u8 {v:?}")
            }
        }

    public open fun toI8(): Result<Byte> =
        when (this) {
            is I8 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a i8 {v:?}")
            }
        }

    public open fun toU16(): Result<UShort> =
        when (this) {
            is U16 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a u16 {v:?}")
            }
        }

    public open fun toI16(): Result<Short> =
        when (this) {
            is I16 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a i16 {v:?}")
            }
        }

    public open fun toU32(): Result<UInt> =
        when (this) {
            is U32 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a u32 {v:?}")
            }
        }

    public open fun toI32(): Result<Int> =
        when (this) {
            is I32 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a i32 {v:?}")
            }
        }

    /**
     * This will also automatically upcast any integral types which will not truncate.
     */
    public open fun toU64(): Result<ULong> =
        when (this) {
            is U64 -> {
                val (v) = this
                Result.success(v)
            }
            // Autoupcast cases here
            is U8 -> {
                val (v1) = this
                Result.success(v1.toULong())
            }
            is U16 -> {
                val (v2) = this
                Result.success(v2.toULong())
            }
            is U32 -> {
                val (v3) = this
                Result.success(v3.toULong())
            }
            is Bool -> {
                val (v4) = this
                Result.success(v4.toULong())
            }
            else -> {
                val v5 = this
                bail("not a u64 or upcastable to u64 {v:?}")
            }
        }

    public open fun toI64(): Result<Long> =
        when (this) {
            is I64 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a i64 {v:?}")
            }
        }

    public open fun toF32(): Result<Float> =
        when (this) {
            is F32 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a f32 {v:?}")
            }
        }

    public open fun toF64(): Result<Double> =
        when (this) {
            is F64 -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a f64 {v:?}")
            }
        }

    public open fun toBool(): Result<Boolean> =
        when (this) {
            is Bool -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a bool {v:?}")
            }
        }

    public open fun toVec(): Result<List<Value>> =
        when (this) {
            is Array -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a vec {v:?}")
            }
        }

    public override fun toString(): Result<String> =
        when (this) {
            is String -> {
                val (v) = this
                Result.success(v)
            }
            else -> {
                val v1 = this
                bail("not a string {v:?}")
            }
        }

    public open fun <W : Write> write(w: W): Result<Unit> {
        when (this) {
            is U8 -> {
                val (v) = this
                w.writeU8(v).getOrElse { return Result.failure(it) }
            }
            is I8 -> {
                val (v1) = this
                w.writeI8(v1).getOrElse { return Result.failure(it) }
            }
            is U16 -> {
                val (v2) = this
                w.writeU16<LittleEndian>(v2).getOrElse { return Result.failure(it) }
            }
            is I16 -> {
                val (v3) = this
                w.writeI16<LittleEndian>(v3).getOrElse { return Result.failure(it) }
            }
            is U32 -> {
                val (v4) = this
                w.writeU32<LittleEndian>(v4).getOrElse { return Result.failure(it) }
            }
            is I32 -> {
                val (v5) = this
                w.writeI32<LittleEndian>(v5).getOrElse { return Result.failure(it) }
            }
            is U64 -> {
                val (v6) = this
                w.writeU64<LittleEndian>(v6).getOrElse { return Result.failure(it) }
            }
            is I64 -> {
                val (v7) = this
                w.writeI64<LittleEndian>(v7).getOrElse { return Result.failure(it) }
            }
            is F32 -> {
                val (v8) = this
                w.writeF32<LittleEndian>(v8).getOrElse { return Result.failure(it) }
            }
            is F64 -> {
                val (v9) = this
                w.writeF64<LittleEndian>(v9).getOrElse { return Result.failure(it) }
            }
            is Bool -> {
                val (v10) = this
                w.writeU8(UByte.from(v10)).getOrElse { return Result.failure(it) }
            }
            is String -> {
                val (v11) = this
                writeString(w, v11.asStr()).getOrElse { return Result.failure(it) }
            }
            is Array -> {
                val (v12) = this
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                val valueType1 =
                    if (v12.isEmpty()) {
                        // Doesn't matter, the array is empty.
                        ValueType.U32
                    } else {
                        val valueType: HashSet = v12.map({ elem -> elem.valueType() }).collect()
                        if (valueType.len() != 1) {
                            bail("multiple value-types in the same array {value_type:?}")
                        }
                        valueType.intoIter().next().context("empty value_type").getOrElse { return Result.failure(it) }
                    }
                w.writeU32<LittleEndian>(valueType1.toU32()).getOrElse { return Result.failure(it) }
                w.writeU64<LittleEndian>(v12.size.toULong()).getOrElse { return Result.failure(it) }
                for (elem in v12.iter()) {
                    elem.write(w).getOrElse { return Result.failure(it) }
                }
            }
        }
        return Result.success(Unit)
    }

    public companion object {
        public fun <R> read(reader: R, valueType: ValueType, magic: VersionedMagic, depth: Int, fileSize: ULong): Result<Value> where R : Read, R : Seek {
            if (depth > GGUF_MAX_VALUE_DEPTH) {
                bail("gguf: value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}")
            }
            val v =
                when (valueType) {
                    ValueType.u8 -> Value.u8(reader.readU8().getOrElse { return Result.failure(it) })
                    ValueType.i8 -> Value.i8(reader.readI8().getOrElse { return Result.failure(it) })
                    ValueType.u16 -> Value.u16(reader.readU16<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.i16 -> Value.i16(reader.readI16<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.u32 -> Value.u32(reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.i32 -> Value.i32(reader.readI32<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.u64 -> Value.u64(reader.readU64<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.i64 -> Value.i64(reader.readI64<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.f32 -> Value.f32(reader.readF32<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.f64 -> Value.f64(reader.readF64<LittleEndian>().getOrElse { return Result.failure(it) })
                    ValueType.Bool -> when (val matchVal = reader.readU8().getOrElse { return Result.failure(it) }) {
                        0 -> Value.Bool(false)
                        1 -> Value.Bool(true)
                        else -> {
                            val b = matchVal
                            bail("unexpected bool value {b}")
                        }
                    }
                    ValueType.String -> Value.String(readString(reader, magic, fileSize).getOrElse { return Result.failure(it) })
                    ValueType.Array -> {
                        val valueType1 = reader.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                        val valueType2 = ValueType.fromU32(valueType1).getOrElse { return Result.failure(it) }
                        val len = readLength(reader, magic).getOrElse { return Result.failure(it) }
                        if (len > GGUF_MAX_ARRAY_ELEMENTS) {
                            bail("gguf: array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
                        }
                        val needed = len.saturatingMul(valueType2.minDiskSize(magic))
                        val remaining = remainingBytes(reader, fileSize).getOrElse { return Result.failure(it) }
                        if (needed > remaining) {
                            bail("gguf: array of {len} elements needs at least {needed} bytes, only {remaining} remaining")
                        }
                        val vs = mutableListOf()
                        for (unused in 0 until len) {
                            vs.add(Value.read(reader, valueType2, magic, depth + 1, fileSize).getOrElse { return Result.failure(it) })
                        }
                        Value.Array(vs)
                    }
                }
            return Result.success(v)
        }
    }
}

public fun <W : Write> writeString(w: W, str: String): Result<Unit> {
    val bytes = str.encodeToByteArray().map { it.toUByte() }
    w.writeU64<LittleEndian>(bytes.size.toULong()).getOrElse { return Result.failure(it) }
    w.writeAll(bytes).getOrElse { return Result.failure(it) }
    return Result.success(Unit)
}

public fun <W> write(w: W, metadata: List<Pair<String, Value>>, tensors: List<Pair<String, QTensor>>): Result<Unit> where W : Seek, W : Write {
    w.writeU32<LittleEndian>(0x46554747).getOrElse { return Result.failure(it) }
    w.writeU32<LittleEndian>(2).getOrElse { return Result.failure(it) }
    // version 2.
    w.writeU64<LittleEndian>(tensors.size.toULong()).getOrElse { return Result.failure(it) }
    w.writeU64<LittleEndian>(metadata.size.toULong()).getOrElse { return Result.failure(it) }
    for ((name, value) in metadata.iter()) {
        writeString(w, name).getOrElse { return Result.failure(it) }
        w.writeU32<LittleEndian>(value.valueType().toU32()).getOrElse { return Result.failure(it) }
        value.write(w).getOrElse { return Result.failure(it) }
    }
    var offset = 0
    val offsets = ArrayList(tensors.size)
    for ((name1, tensor) in tensors.iter()) {
        writeString(w, name1).getOrElse { return Result.failure(it) }
        val dims = tensor.shape().dims()
        w.writeU32<LittleEndian>(dims.len().toUInt()).getOrElse { return Result.failure(it) }
        for (dim in dims.iter().rev()) {
            w.writeU64<LittleEndian>(dim.toULong()).getOrElse { return Result.failure(it) }
        }
        w.writeU32<LittleEndian>(tensor.dtype().toU32()).getOrElse { return Result.failure(it) }
        w.writeU64<LittleEndian>(offset.toULong()).getOrElse { return Result.failure(it) }
        offsets.add(offset)
        val sizeInBytes = tensor.storageSizeInBytes()
        val padding = 31 - (31 + sizeInBytes) % 32
        offset += sizeInBytes + padding
    }
    val pos = (w.streamPosition().getOrElse { return Result.failure(it) }).toInt()
    val padding1 = 31 - (31 + pos) % 32
    w.writeAll(mutableListOf(0u.toUByte(), padding1)).getOrElse { return Result.failure(it) }
    val tensorStartPos = (w.streamPosition().getOrElse { return Result.failure(it) }).toInt()
    for ((offset1, name, tensor1) in (ArrayDeque(offsets).zip(tensors))) {
        val pos1 = (w.streamPosition().getOrElse { return Result.failure(it) }).toInt()
        if (tensorStartPos + offset1 != pos1) {
            bail("internal error, unexpected current position {tensor_start_pos} {offset} {pos}")
        }
        val data = tensor1.data().getOrElse { return Result.failure(it) }
        val sizeInBytes1 = data.len()
        w.writeAll(data).getOrElse { return Result.failure(it) }
        val padding2 = 31 - (31 + sizeInBytes1) % 32
        w.writeAll(mutableListOf(0u.toUByte(), padding2)).getOrElse { return Result.failure(it) }
    }
    return Result.success(Unit)
}
