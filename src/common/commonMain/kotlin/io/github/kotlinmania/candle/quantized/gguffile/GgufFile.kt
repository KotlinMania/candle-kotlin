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
    val cur =
        when (val tryResult = reader.streamPosition()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    return Result.Ok(fileSize.saturatingSub(cur))
}

public fun <R : Read> readLength(reader: R, magic: VersionedMagic): Result<ULong> {
    return when (magic) {
        VersionedMagic.GgufV1 ->
            Result.Ok(
                (
                    when (val tryResult = reader.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                ).toULong(),
            )
        VersionedMagic.GgufV2, VersionedMagic.GgufV3 ->
            Result.Ok(
                when (val tryResult = reader.readU64<LittleEndian>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
            )
    }
}

public enum class Magic(
    public val value: UInt,
) {
    Gguf(0u),
    ;

    public companion object {
        public fun tryFrom(value: UInt): Result<Magic> {
            val magic =
                when (value) {
                    0x46554747, 0x47475546 -> Magic.Gguf
                    else -> bail("unknown magic 0x{value:08x}")
                }
            return Result.Ok(magic)
        }
    }
}

public enum class VersionedMagic(
    public val value: UInt,
) {
    GgufV1(0u),
    GgufV2(1u),
    GgufV3(2u),
    ;

    public fun lengthPrefixSize(): ULong =
        when (this) {
            VersionedMagic.GgufV1 -> 4uL
            VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> 8uL
        }

    public companion object {
        public fun <R : Read> read(reader: R): Result<VersionedMagic> {
            val magic =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val magic1 =
                when (val tryResult = Magic.tryFrom(magic)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val version =
                when (val tryResult = reader.readU32<LittleEndian>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
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
            return Result.Ok(versionedMagic)
        }
    }
}

public data class TensorInfo(
    public var ggmlDtype: GgmlDType = GgmlDType.entries.first(),
    public var shape: Shape = Shape(),
    public var offset: ULong = 0uL,
) {
    public fun <R> read(reader: R, tensorDataOffset: ULong, device: Device): Result<QTensor> where R : Seek, R : Read {
        val tensorElems = this.shape.elemCount()
        val blockSize = this.ggmlDtype.blockSize()
        if (!tensorElems.isMultipleOf(blockSize)) {
            bail("the number of elements {tensor_elems} is not divisible by the block size {block_size}")
        }
        val sizeInBytes = tensorElems / blockSize * this.ggmlDtype.typeSize()
        val tensorStart = tensorDataOffset.saturatingAdd(this.offset)
        val fileSize =
            when (val tryResult = reader.seek(SeekFrom.End(0))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val remaining = fileSize.saturatingSub(tensorStart)
        if (sizeInBytes.toULong() > remaining) {
            bail("tensor needs {size_in_bytes} bytes at offset {tensor_start}, only {remaining} remaining in file")
        }
        val rawData = mutableListOf(0u.toUByte(), sizeInBytes)
        when (val tryResult = reader.seek(SeekFrom.Start(tensorStart))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = reader.readExact(rawData)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return qtensorFromGgml(this.ggmlDtype, rawData, this.shape.dims().toVec(), device)
    }
}

public data class Content(
    public var magic: VersionedMagic = VersionedMagic.entries.first(),
    public var metadata: MutableMap<String, Value> = MutableMap<String, Value>(),
    public var tensorInfos: MutableMap<String, TensorInfo> = MutableMap<String, TensorInfo>(),
    public var tensorDataOffset: ULong = 0uL,
) {
    public fun <R> tensor(reader: R, name: String, device: Device): Result<QTensor> where R : Seek, R : Read {
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
            val start =
                when (val tryResult = reader.streamPosition()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val fileSize =
                when (val tryResult = reader.seek(SeekFrom.End(0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            when (val tryResult = reader.seek(SeekFrom.Start(start))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
            val magic =
                when (val tryResult = VersionedMagic.read(reader)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val tensorCount =
                when (val tryResult = readLength(reader, magic)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val metadataKvCount =
                when (val tryResult = readLength(reader, magic)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
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
            val remaining =
                when (val tryResult = remainingBytes(reader, fileSize)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            if (needed > remaining) {
                bail("gguf: header declares {tensor_count} tensors and {metadata_kv_count} metadata entries, needs at least {needed} bytes, only {remaining} remaining")
            }
            var metadata = mutableMapOf()
            for (idx in 0 until metadataKvCount) {
                val key =
                    when (val tryResult = readString(reader, magic, fileSize)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val valueType =
                    when (val tryResult = reader.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val valueType1 =
                    when (val tryResult = ValueType.fromU32(valueType)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val value =
                    when (val tryResult = Value.read(reader, valueType1, magic, 0, fileSize)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                metadata.put(key, value)
            }
            var tensorInfos = mutableMapOf()
            for (idx1 in 0 until tensorCount) {
                val tensorName =
                    when (val tryResult = readString(reader, magic, fileSize)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val nDimensions =
                    when (val tryResult = reader.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (nDimensions > GGUF_MAX_TENSOR_DIMS) {
                    bail("gguf: tensor '{tensor_name}' has {n_dimensions} dimensions, max is {GGUF_MAX_TENSOR_DIMS}")
                }
                var dimensions2: List<Int> =
                    when (magic) {
                        VersionedMagic.GgufV1 -> {
                            val dimensions = mutableListOf(0, nDimensions.toInt())
                            when (val tryResult = reader.readU32Into<LittleEndian>(dimensions)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                            ArrayDeque((ArrayDeque(dimensions)).map({ c -> c.toInt() })).toMutableList()
                        }
                        VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> {
                            val dimensions1 = mutableListOf(0, nDimensions.toInt())
                            when (val tryResult = reader.readU64Into<LittleEndian>(dimensions1)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                            ArrayDeque((ArrayDeque(dimensions1)).map({ c -> c.toInt() })).toMutableList()
                        }
                    }
                dimensions2.reverse()
                val ggmlDtype =
                    when (val tryResult = reader.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val ggmlDtype1 =
                    when (val tryResult = GgmlDType.fromU32(ggmlDtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val offset =
                    when (val tryResult = reader.readU64<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                tensorInfos.put(tensorName, TensorInfo(shape = Shape.from(dimensions2), offset = offset, ggmlDtype = ggmlDtype1))
            }
            val position =
                when (val tryResult = reader.streamPosition()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
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
            return Result.Ok(Content(magic = magic, metadata = metadata, tensorInfos = tensorInfos, tensorDataOffset = tensorDataOffset))
        }
    }
}

public fun <R> readString(reader: R, magic: VersionedMagic, fileSize: ULong): Result<String> where R : Read, R : Seek {
    val len =
        when (val tryResult = readLength(reader, magic)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (len > GGUF_MAX_STRING_LENGTH) {
        bail("gguf: string length {len} exceeds max {GGUF_MAX_STRING_LENGTH}")
    }
    val remaining =
        when (val tryResult = remainingBytes(reader, fileSize)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    if (len > remaining) {
        bail("gguf: string length {len} exceeds remaining file bytes {remaining}")
    }
    val v = mutableListOf(0u.toUByte(), len.toInt())
    when (val tryResult = reader.readExact(v)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while (true) {
        val matchVal = v.last()
        if (!(matchVal == 0)) break
        v.removeLastOrNull()
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    return Result.Ok(String.fromUtf8Lossy(v).intoOwned())
}

public enum class ValueType(
    public val value: UInt,
) {
    // The value is a 8-bit unsigned integer.
    U8(0u),

    // The value is a 8-bit signed integer.
    I8(1u),

    // The value is a 16-bit unsigned little-endian integer.
    U16(2u),

    // The value is a 16-bit signed little-endian integer.
    I16(3u),

    // The value is a 32-bit unsigned little-endian integer.
    U32(4u),

    // The value is a 32-bit signed little-endian integer.
    I32(5u),

    // The value is a 64-bit unsigned little-endian integer.
    U64(6u),

    // The value is a 64-bit signed little-endian integer.
    I64(7u),

    // The value is a 32-bit IEEE754 floating point number.
    F32(8u),

    // The value is a 64-bit IEEE754 floating point number.
    F64(9u),

    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool(10u),

    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String(11u),

    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array(12u),
    ;

    public fun toU32(): UInt =
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
    public fun minDiskSize(magic: VersionedMagic): ULong =
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
            return Result.Ok(v2)
        }
    }
}

public sealed class Value {
    public data class U8(
        public var value: UByte,
    ) : Value()
    public data class I8(
        public var value: Byte,
    ) : Value()
    public data class U16(
        public var value: UShort,
    ) : Value()
    public data class I16(
        public var value: Short,
    ) : Value()
    public data class U32(
        public var value: UInt,
    ) : Value()
    public data class I32(
        public var value: Int,
    ) : Value()
    public data class U64(
        public var value: ULong,
    ) : Value()
    public data class I64(
        public var value: Long,
    ) : Value()
    public data class F32(
        public var value: Float,
    ) : Value()
    public data class F64(
        public var value: Double,
    ) : Value()
    public data class Bool(
        public var value: Boolean,
    ) : Value()
    public data class String(
        public var value: String,
    ) : Value()
    public data class Array(
        public var value: List<Value>,
    ) : Value()

    public fun valueType(): ValueType =
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

    public fun toU8(): Result<UByte> =
        when (this) {
            is U8 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a u8 {v:?}")
            }
        }

    public fun toI8(): Result<Byte> =
        when (this) {
            is I8 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a i8 {v:?}")
            }
        }

    public fun toU16(): Result<UShort> =
        when (this) {
            is U16 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a u16 {v:?}")
            }
        }

    public fun toI16(): Result<Short> =
        when (this) {
            is I16 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a i16 {v:?}")
            }
        }

    public fun toU32(): Result<UInt> =
        when (this) {
            is U32 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a u32 {v:?}")
            }
        }

    public fun toI32(): Result<Int> =
        when (this) {
            is I32 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a i32 {v:?}")
            }
        }

    /**
     * This will also automatically upcast any integral types which will not truncate.
     */
    public fun toU64(): Result<ULong> =
        when (this) {
            is U64 -> {
                val (v) = this
                Result.Ok(v)
            }
            // Autoupcast cases here
            is U8 -> {
                val (v1) = this
                Result.Ok(v1.toULong())
            }
            is U16 -> {
                val (v2) = this
                Result.Ok(v2.toULong())
            }
            is U32 -> {
                val (v3) = this
                Result.Ok(v3.toULong())
            }
            is Bool -> {
                val (v4) = this
                Result.Ok(v4.toULong())
            }
            else -> {
                val v5 = this
                bail("not a u64 or upcastable to u64 {v:?}")
            }
        }

    public fun toI64(): Result<Long> =
        when (this) {
            is I64 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a i64 {v:?}")
            }
        }

    public fun toF32(): Result<Float> =
        when (this) {
            is F32 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a f32 {v:?}")
            }
        }

    public fun toF64(): Result<Double> =
        when (this) {
            is F64 -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a f64 {v:?}")
            }
        }

    public fun toBool(): Result<Boolean> =
        when (this) {
            is Bool -> {
                val (v) = this
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a bool {v:?}")
            }
        }

    public fun toVec(): Result<List<Value>> =
        when (this) {
            is Array -> {
                val (v) = this
                Result.Ok(v)
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
                Result.Ok(v)
            }
            else -> {
                val v1 = this
                bail("not a string {v:?}")
            }
        }

    public fun <W : Write> write(w: W): Result<Unit> {
        when (this) {
            is U8 -> {
                val (v) = this
                when (val tryResult = w.writeU8(v)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is I8 -> {
                val (v1) = this
                when (val tryResult = w.writeI8(v1)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is U16 -> {
                val (v2) = this
                when (val tryResult = w.writeU16<LittleEndian>(v2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is I16 -> {
                val (v3) = this
                when (val tryResult = w.writeI16<LittleEndian>(v3)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is U32 -> {
                val (v4) = this
                when (val tryResult = w.writeU32<LittleEndian>(v4)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is I32 -> {
                val (v5) = this
                when (val tryResult = w.writeI32<LittleEndian>(v5)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is U64 -> {
                val (v6) = this
                when (val tryResult = w.writeU64<LittleEndian>(v6)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is I64 -> {
                val (v7) = this
                when (val tryResult = w.writeI64<LittleEndian>(v7)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is F32 -> {
                val (v8) = this
                when (val tryResult = w.writeF32<LittleEndian>(v8)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is F64 -> {
                val (v9) = this
                when (val tryResult = w.writeF64<LittleEndian>(v9)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is Bool -> {
                val (v10) = this
                when (val tryResult = w.writeU8(UByte.from(v10))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            is String -> {
                val (v11) = this
                when (val tryResult = writeString(w, v11)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
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
                        when (val tryResult = valueType.intoIter().next().context("empty value_type")) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                when (val tryResult = w.writeU32<LittleEndian>(valueType1.toU32())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                when (val tryResult = w.writeU64<LittleEndian>(v12.size.toULong())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                for (elem in v12.iter()) {
                    when (val tryResult = elem.write(w)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
            }
        }
        return Result.Ok(Unit)
    }

    public companion object {
        public fun <R> read(reader: R, valueType: ValueType, magic: VersionedMagic, depth: Int, fileSize: ULong): Result<Value> where R : Read, R : Seek {
            if (depth > GGUF_MAX_VALUE_DEPTH) {
                bail("gguf: value nesting depth exceeds max {GGUF_MAX_VALUE_DEPTH}")
            }
            val v =
                when (valueType) {
                    ValueType.u8 ->
                        Value.u8(
                            when (val tryResult = reader.readU8()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.i8 ->
                        Value.i8(
                            when (val tryResult = reader.readI8()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.u16 ->
                        Value.u16(
                            when (val tryResult = reader.readU16<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.i16 ->
                        Value.i16(
                            when (val tryResult = reader.readI16<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.u32 ->
                        Value.u32(
                            when (val tryResult = reader.readU32<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.i32 ->
                        Value.i32(
                            when (val tryResult = reader.readI32<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.u64 ->
                        Value.u64(
                            when (val tryResult = reader.readU64<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.i64 ->
                        Value.i64(
                            when (val tryResult = reader.readI64<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.f32 ->
                        Value.f32(
                            when (val tryResult = reader.readF32<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.f64 ->
                        Value.f64(
                            when (val tryResult = reader.readF64<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.Bool ->
                        when (
                            val matchVal =
                                when (val tryResult = reader.readU8()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }
                        ) {
                            0 -> Value.Bool(false)
                            1 -> Value.Bool(true)
                            else -> {
                                val b = matchVal
                                bail("unexpected bool value {b}")
                            }
                        }
                    ValueType.String ->
                        Value.String(
                            when (val tryResult = readString(reader, magic, fileSize)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    ValueType.Array -> {
                        val valueType1 =
                            when (val tryResult = reader.readU32<LittleEndian>()) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        val valueType2 =
                            when (val tryResult = ValueType.fromU32(valueType1)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        val len =
                            when (val tryResult = readLength(reader, magic)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        if (len > GGUF_MAX_ARRAY_ELEMENTS) {
                            bail("gguf: array length {len} exceeds max {GGUF_MAX_ARRAY_ELEMENTS}")
                        }
                        val needed = len.saturatingMul(valueType2.minDiskSize(magic))
                        val remaining =
                            when (val tryResult = remainingBytes(reader, fileSize)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        if (needed > remaining) {
                            bail("gguf: array of {len} elements needs at least {needed} bytes, only {remaining} remaining")
                        }
                        val vs = mutableListOf()
                        for (unused in 0 until len) {
                            vs.add(
                                when (val tryResult = Value.read(reader, valueType2, magic, depth + 1, fileSize)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                },
                            )
                        }
                        Value.Array(vs)
                    }
                }
            return Result.Ok(v)
        }
    }
}

public fun <W : Write> writeString(w: W, str: String): Result<Unit> {
    val bytes = str.encodeToByteArray().asUByteArray()
    when (val tryResult = w.writeU64<LittleEndian>(bytes.size.toULong())) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = w.writeAll(bytes)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun <W> write(w: W, metadata: List<Pair<String, Value>>, tensors: List<Pair<String, QTensor>>): Result<Unit> where W : Seek, W : Write {
    when (val tryResult = w.writeU32<LittleEndian>(0x46554747)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = w.writeU32<LittleEndian>(2)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    // version 2.
    when (val tryResult = w.writeU64<LittleEndian>(tensors.size.toULong())) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = w.writeU64<LittleEndian>(metadata.size.toULong())) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    for ((name, value) in metadata.iter()) {
        when (val tryResult = writeString(w, name)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = w.writeU32<LittleEndian>(value.valueType().toU32())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = value.write(w)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    var offset = 0
    val offsets = ArrayList(tensors.size)
    for ((name1, tensor) in tensors.iter()) {
        when (val tryResult = writeString(w, name1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        val dims = tensor.shape().dims()
        when (val tryResult = w.writeU32<LittleEndian>(dims.len().toUInt())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        for (dim in dims.iter().rev()) {
            when (val tryResult = w.writeU64<LittleEndian>(dim.toULong())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        when (val tryResult = w.writeU32<LittleEndian>(tensor.dtype().toU32())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = w.writeU64<LittleEndian>(offset.toULong())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        offsets.add(offset)
        val sizeInBytes = tensor.storageSizeInBytes()
        val padding = 31 - (31 + sizeInBytes) % 32
        offset += sizeInBytes + padding
    }
    val pos =
        (
            when (val tryResult = w.streamPosition()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        ).toInt()
    val padding1 = 31 - (31 + pos) % 32
    when (val tryResult = w.writeAll(mutableListOf(0u.toUByte(), padding1))) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    val tensorStartPos =
        (
            when (val tryResult = w.streamPosition()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        ).toInt()
    for ((offset1, name, tensor1) in (offsets.zip(tensors))) {
        val pos1 =
            (
                when (val tryResult = w.streamPosition()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ).toInt()
        if (tensorStartPos + offset1 != pos1) {
            bail("internal error, unexpected current position {tensor_start_pos} {offset} {pos}")
        }
        val data =
            when (val tryResult = tensor1.data()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val sizeInBytes1 = data.len()
        when (val tryResult = w.writeAll(data)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        val padding2 = 31 - (31 + sizeInBytes1) % 32
        when (val tryResult = w.writeAll(mutableListOf(0u.toUByte(), padding2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    }
    return Result.Ok(Unit)
}
