// port-lint: source quantized/gguf_file.rs
/**
 * Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
 *
 * Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */
package io.github.kotlinmania.candle.gguf

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.NpyByteReader
import io.github.kotlinmania.candle.NpyByteWriter
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.ggml.qtensorFromGgml

public const val DEFAULT_ALIGNMENT: ULong = 32uL

// Caps mirror ggml-org/llama.cpp#19856 (GGUF_MAX_STRING_LENGTH,
// GGUF_MAX_ARRAY_ELEMENTS) and GGML_MAX_DIMS from ggml.h. The depth cap
// bounds recursion through nested `Value.Array` so a crafted file cannot
// overflow the stack with a chain of arrays-of-arrays.
private const val GGUF_MAX_STRING_LENGTH: ULong = 1uL shl 30
private const val GGUF_MAX_ARRAY_ELEMENTS: ULong = 1uL shl 30
private const val GGUF_MAX_TENSOR_DIMS: UInt = 4u
private const val GGUF_MAX_VALUE_DEPTH: Int = 64

// `fileSize` is the byte length captured once up front, so this avoids
// seeking to the end and back on every length-prefixed read.
private fun <R : NpyByteReader> remainingBytes(reader: R, fileSize: ULong): Result<ULong> {
    val cur = reader.streamPosition().unwrap()
    return Ok(fileSize.saturatingSub(cur))
}

private fun <R : NpyByteReader> readLength(reader: R, magic: VersionedMagic): Result<ULong> =
    when (magic) {
        VersionedMagic.GgufV1 -> Ok(reader.readU32<LittleEndian>().unwrap().toULong())
        VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> Ok(reader.readU64<LittleEndian>().unwrap())
    }

private enum class Magic {
    Gguf,
    ;

    interface Error

    companion object {
        fun tryFrom(value: UInt): Result<Magic> {
            val magic = when (value) {
                0x46554747u, 0x47475546u -> Gguf
                else -> bail("unknown magic 0x${value.toString(16)}")
            }
            return Ok(magic)
        }
    }
}

public enum class VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3,
    ;

    internal fun lengthPrefixSize(): ULong = when (this) {
        GgufV1 -> 4uL
        GgufV2, GgufV3 -> 8uL
    }

    public companion object {
        internal fun <R : NpyByteReader> read(reader: R): Result<VersionedMagic> {
            val rawMagic = reader.readU32<LittleEndian>().unwrap()
            val magic = Magic.tryFrom(rawMagic).unwrap()
            val version = reader.readU32<LittleEndian>().unwrap()
            val versionedMagic = when (Pair(magic, version)) {
                Pair(Magic.Gguf, 1u) -> GgufV1
                Pair(Magic.Gguf, 2u) -> GgufV2
                Pair(Magic.Gguf, 3u) -> GgufV3
                else -> bail("gguf: unsupported magic/version $magic/$version")
            }
            return Ok(versionedMagic)
        }
    }
}

public data class TensorInfo(
    public val ggmlDtype: GgmlDType,
    public val shape: Shape,
    public val offset: ULong,
) {
    public fun <R : NpyByteReader> read(
        reader: R,
        tensorDataOffset: ULong,
        device: Device,
    ): Result<QTensor> {
        val tensorElems = this.shape.elemCount()
        val blockSize = this.ggmlDtype.blockSize()
        if (!tensorElems.isMultipleOf(blockSize)) {
            bail(
                "the number of elements $tensorElems is not divisible by the block size $blockSize",
            )
        }
        val sizeInBytes = tensorElems / blockSize * this.ggmlDtype.typeSize()
        val tensorStart = tensorDataOffset.saturatingAdd(this.offset)
        val fileSize = reader.seek(SeekFrom.End(0)).unwrap()
        val remaining = fileSize.saturatingSub(tensorStart)
        if (sizeInBytes.toULong() > remaining) {
            bail(
                "tensor needs $sizeInBytes bytes at offset $tensorStart, only $remaining remaining in file",
            )
        }
        val rawData = ByteArray(sizeInBytes)
        reader.seek(SeekFrom.Start(tensorStart)).unwrap()
        reader.readExact(rawData).unwrap()
        return qtensorFromGgml(
            this.ggmlDtype,
            rawData,
            this.shape.dims().toVec(),
            device,
        )
    }
}

public class Content(
    public val magic: VersionedMagic,
    public val metadata: Map<String, Value>,
    public val tensorInfos: Map<String, TensorInfo>,
    public val tensorDataOffset: ULong,
) {
    public fun <R : NpyByteReader> tensor(
        reader: R,
        name: String,
        device: Device,
    ): Result<QTensor> {
        val info = this.tensorInfos.get(name)
        val tensorInfo = when (info) {
            null -> bail("cannot find tensor info for $name")
            else -> info
        }
        return tensorInfo.read(reader, this.tensorDataOffset, device)
    }

    public companion object {
        public fun <R : NpyByteReader> read(reader: R): Result<Content> {
            // Capture the file size once so the bounds checks below don't have to
            // seek to the end and back on every length-prefixed read.
            val start = reader.streamPosition().unwrap()
            val fileSize = reader.seek(SeekFrom.End(0)).unwrap()
            reader.seek(SeekFrom.Start(start)).unwrap()

            val magic = VersionedMagic.read(reader).unwrap()
            val tensorCount = readLength(reader, magic).unwrap()
            val metadataKvCount = readLength(reader, magic).unwrap()

            if (tensorCount > GGUF_MAX_ARRAY_ELEMENTS) {
                bail("gguf: tensorCount $tensorCount exceeds max $GGUF_MAX_ARRAY_ELEMENTS")
            }
            if (metadataKvCount > GGUF_MAX_ARRAY_ELEMENTS) {
                bail(
                    "gguf: metadataKvCount $metadataKvCount exceeds max $GGUF_MAX_ARRAY_ELEMENTS",
                )
            }

            // Reject header-declared counts that can't fit in the file at minimum size.
            // Per-entry minima: a metadata kv is at least `keyLenPrefix + u32 valueType
            // + 1 byte value`; a tensor info is at least `nameLenPrefix + u32 nDims
            // + u32 dtype + u64 offset`.
            val prefix = magic.lengthPrefixSize()
            val minPerKv = prefix + 4uL + 1uL
            val minPerTensor = prefix + 4uL + 4uL + 8uL
            val needed = metadataKvCount
                .saturatingMul(minPerKv)
                .saturatingAdd(tensorCount.saturatingMul(minPerTensor))
            val remaining = remainingBytes(reader, fileSize).unwrap()
            if (needed > remaining) {
                bail(
                    "gguf: header declares $tensorCount tensors and $metadataKvCount metadata entries, needs at least $needed bytes, only $remaining remaining",
                )
            }

            val metadata = HashMap.new<String, Value>()
            for (idx in 0uL until metadataKvCount) {
                val key = readString(reader, magic, fileSize).unwrap()
                val rawValueType = reader.readU32<LittleEndian>().unwrap()
                val valueType = ValueType.fromU32(rawValueType).unwrap()
                val value = Value.read(reader, valueType, magic, 0, fileSize).unwrap()
                metadata.insert(key, value)
            }
            val tensorInfos = HashMap.new<String, TensorInfo>()
            for (idx in 0uL until tensorCount) {
                val tensorName = readString(reader, magic, fileSize).unwrap()
                val nDimensions = reader.readU32<LittleEndian>().unwrap()
                if (nDimensions > GGUF_MAX_TENSOR_DIMS) {
                    bail(
                        "gguf: tensor '$tensorName' has $nDimensions dimensions, max is $GGUF_MAX_TENSOR_DIMS",
                    )
                }

                val dimensions: MutableList<Int> = when (magic) {
                    VersionedMagic.GgufV1 -> {
                        val rawDims = UIntArray(nDimensions.toInt())
                        reader.readU32Into<LittleEndian>(rawDims).unwrap()
                        rawDims.intoIter().map { c -> c.toInt() }.collect()
                    }
                    VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> {
                        val rawDims = ULongArray(nDimensions.toInt())
                        reader.readU64Into<LittleEndian>(rawDims).unwrap()
                        rawDims.intoIter().map { c -> c.toInt() }.collect()
                    }
                }

                dimensions.reverse()
                val rawGgmlDtype = reader.readU32<LittleEndian>().unwrap()
                val ggmlDtype = GgmlDType.fromU32(rawGgmlDtype).unwrap()
                val offset = reader.readU64<LittleEndian>().unwrap()
                tensorInfos.insert(
                    tensorName,
                    TensorInfo(
                        shape = Shape.from(dimensions),
                        offset = offset,
                        ggmlDtype = ggmlDtype,
                    ),
                )
            }
            val position = reader.streamPosition().unwrap()
            val alignVal = metadata.get("general.alignment")
            val alignment = when (alignVal) {
                is Value.U8 -> alignVal.v.toULong()
                is Value.U16 -> alignVal.v.toULong()
                is Value.U32 -> alignVal.v.toULong()
                is Value.I8 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                is Value.I16 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                is Value.I32 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                else -> DEFAULT_ALIGNMENT
            }
            val tensorDataOffset = position.divCeil(alignment) * alignment
            return Ok(
                Content(
                    magic = magic,
                    metadata = metadata,
                    tensorInfos = tensorInfos,
                    tensorDataOffset = tensorDataOffset,
                ),
            )
        }
    }
}

private fun <R : NpyByteReader> readString(
    reader: R,
    magic: VersionedMagic,
    fileSize: ULong,
): Result<String> {
    val len = readLength(reader, magic).unwrap()
    if (len > GGUF_MAX_STRING_LENGTH) {
        bail("gguf: string length $len exceeds max $GGUF_MAX_STRING_LENGTH")
    }
    val remaining = remainingBytes(reader, fileSize).unwrap()
    if (len > remaining) {
        bail("gguf: string length $len exceeds remaining file bytes $remaining")
    }
    val v = ByteArray(len.toInt())
    reader.readExact(v).unwrap()
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    while (v.last() == 0.toByte()) {
        v.pop()
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    return Ok(String.fromUtf8Lossy(v).intoOwned())
}

public enum class ValueType {
    // The value is a 8-bit unsigned integer.
    U8,
    // The value is a 8-bit signed integer.
    I8,
    // The value is a 16-bit unsigned little-endian integer.
    U16,
    // The value is a 16-bit signed little-endian integer.
    I16,
    // The value is a 32-bit unsigned little-endian integer.
    U32,
    // The value is a 32-bit signed little-endian integer.
    I32,
    // The value is a 64-bit unsigned little-endian integer.
    U64,
    // The value is a 64-bit signed little-endian integer.
    I64,
    // The value is a 32-bit IEEE754 floating point number.
    F32,
    // The value is a 64-bit IEEE754 floating point number.
    F64,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    Bool,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    String,
    // The value is an array of other values, with the length and type prepended.
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    Array,
    ;

    internal fun toU32(): UInt = when (this) {
        U8 -> 0u
        I8 -> 1u
        U16 -> 2u
        I16 -> 3u
        U32 -> 4u
        I32 -> 5u
        F32 -> 6u
        Bool -> 7u
        String -> 8u
        Array -> 9u
        U64 -> 10u
        I64 -> 11u
        F64 -> 12u
    }

    /**
     * Minimum on-disk size of one value of this type, used to reject array
     * lengths that exceed the remaining file size before allocating.
     */
    internal fun minDiskSize(magic: VersionedMagic): ULong = when (this) {
        U8, I8, Bool -> 1uL
        U16, I16 -> 2uL
        U32, I32, F32 -> 4uL
        U64, I64, F64 -> 8uL
        String -> magic.lengthPrefixSize()
        Array -> 4uL + magic.lengthPrefixSize()
    }

    public companion object {
        internal fun fromU32(v: UInt): Result<ValueType> {
            val vt = when (v) {
                0u -> U8
                1u -> I8
                2u -> U16
                3u -> I16
                4u -> U32
                5u -> I32
                6u -> F32
                7u -> Bool
                8u -> String
                9u -> Array
                10u -> U64
                11u -> I64
                12u -> F64
                else -> bail("unrecognized value-type 0x${v.toString(16)}")
            }
            return Ok(vt)
        }
    }
}

public sealed class Value {
    public data class U8(public val v: UByte) : Value()
    public data class I8(public val v: Byte) : Value()
    public data class U16(public val v: UShort) : Value()
    public data class I16(public val v: Short) : Value()
    public data class U32(public val v: UInt) : Value()
    public data class I32(public val v: Int) : Value()
    public data class U64(public val v: ULong) : Value()
    public data class I64(public val v: Long) : Value()
    public data class F32(public val v: Float) : Value()
    public data class F64(public val v: Double) : Value()
    public data class Bool(public val v: Boolean) : Value()
    public data class String(public val v: kotlin.String) : Value()
    public data class Array(public val v: List<Value>) : Value()

    public fun valueType(): ValueType = when (this) {
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

    public fun toU8(): Result<UByte> = when (this) {
        is U8 -> Ok(this.v)
        else -> bail("not a u8 $this")
    }

    public fun toI8(): Result<Byte> = when (this) {
        is I8 -> Ok(this.v)
        else -> bail("not a i8 $this")
    }

    public fun toU16(): Result<UShort> = when (this) {
        is U16 -> Ok(this.v)
        else -> bail("not a u16 $this")
    }

    public fun toI16(): Result<Short> = when (this) {
        is I16 -> Ok(this.v)
        else -> bail("not a i16 $this")
    }

    public fun toU32(): Result<UInt> = when (this) {
        is U32 -> Ok(this.v)
        else -> bail("not a u32 $this")
    }

    public fun toI32(): Result<Int> = when (this) {
        is I32 -> Ok(this.v)
        else -> bail("not a i32 $this")
    }

    /**
     * This will also automatically upcast any integral types which will not truncate.
     */
    public fun toU64(): Result<ULong> = when (this) {
        is U64 -> Ok(this.v)
        // Autoupcast cases here
        is U8 -> Ok(this.v.toULong())
        is U16 -> Ok(this.v.toULong())
        is U32 -> Ok(this.v.toULong())
        is Bool -> Ok(if (this.v) 1uL else 0uL)
        else -> bail("not a u64 or upcastable to u64 $this")
    }

    public fun toI64(): Result<Long> = when (this) {
        is I64 -> Ok(this.v)
        else -> bail("not a i64 $this")
    }

    public fun toF32(): Result<Float> = when (this) {
        is F32 -> Ok(this.v)
        else -> bail("not a f32 $this")
    }

    public fun toF64(): Result<Double> = when (this) {
        is F64 -> Ok(this.v)
        else -> bail("not a f64 $this")
    }

    public fun toBool(): Result<Boolean> = when (this) {
        is Bool -> Ok(this.v)
        else -> bail("not a bool $this")
    }

    public fun toVec(): Result<List<Value>> = when (this) {
        is Array -> Ok(this.v)
        else -> bail("not a vec $this")
    }

    public fun toString(validate: Boolean = true): Result<kotlin.String> = when (this) {
        is String -> if (validate || this.v.isNotEmpty()) Ok(this.v) else Ok(this.v)
        else -> bail("not a string $this")
    }

    internal fun <W : NpyByteWriter> write(w: W): Result<Unit> {
        when (this) {
            is U8 -> w.writeU8(this.v).unwrap()
            is I8 -> w.writeI8(this.v).unwrap()
            is U16 -> w.writeU16<LittleEndian>(this.v).unwrap()
            is I16 -> w.writeI16<LittleEndian>(this.v).unwrap()
            is U32 -> w.writeU32<LittleEndian>(this.v).unwrap()
            is I32 -> w.writeI32<LittleEndian>(this.v).unwrap()
            is U64 -> w.writeU64<LittleEndian>(this.v).unwrap()
            is I64 -> w.writeI64<LittleEndian>(this.v).unwrap()
            is F32 -> w.writeF32<LittleEndian>(this.v).unwrap()
            is F64 -> w.writeF64<LittleEndian>(this.v).unwrap()
            is Bool -> w.writeU8(UByte.from(this.v)).unwrap()
            is String -> writeString(w, this.v.asStr()).unwrap()
            is Array -> {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                val valueType = if (this.v.isEmpty()) {
                    // Doesn't matter, the array is empty.
                    ValueType.U32
                } else {
                    val valueTypes: Set<ValueType> =
                        this.v.iter().map { elem -> elem.valueType() }.collect()
                    if (valueTypes.len() != 1) {
                        bail("multiple value-types in the same array $valueTypes")
                    }
                    valueTypes.intoIter().next().context("empty valueType").unwrap()
                }
                w.writeU32<LittleEndian>(valueType.toU32()).unwrap()
                w.writeU64<LittleEndian>(this.v.len().toULong()).unwrap()
                for (elem in this.v.iter()) {
                    elem.write(w).unwrap()
                }
            }
        }
        return Ok(Unit)
    }

    public companion object {
        internal fun <R : NpyByteReader> read(
            reader: R,
            valueType: ValueType,
            magic: VersionedMagic,
            depth: Int,
            fileSize: ULong,
        ): Result<Value> {
            if (depth > GGUF_MAX_VALUE_DEPTH) {
                bail("gguf: value nesting depth exceeds max $GGUF_MAX_VALUE_DEPTH")
            }
            val v = when (valueType) {
                ValueType.U8 -> U8(reader.readU8().unwrap())
                ValueType.I8 -> I8(reader.readI8().unwrap())
                ValueType.U16 -> U16(reader.readU16<LittleEndian>().unwrap())
                ValueType.I16 -> I16(reader.readI16<LittleEndian>().unwrap())
                ValueType.U32 -> U32(reader.readU32<LittleEndian>().unwrap())
                ValueType.I32 -> I32(reader.readI32<LittleEndian>().unwrap())
                ValueType.U64 -> U64(reader.readU64<LittleEndian>().unwrap())
                ValueType.I64 -> I64(reader.readI64<LittleEndian>().unwrap())
                ValueType.F32 -> F32(reader.readF32<LittleEndian>().unwrap())
                ValueType.F64 -> F64(reader.readF64<LittleEndian>().unwrap())
                ValueType.Bool -> {
                    val b = reader.readU8().unwrap().toInt()
                    when (b) {
                        0 -> Bool(false)
                        1 -> Bool(true)
                        else -> bail("unexpected bool value $b")
                    }
                }
                ValueType.String -> String(readString(reader, magic, fileSize).unwrap())
                ValueType.Array -> {
                    val rawVt = reader.readU32<LittleEndian>().unwrap()
                    val elemVt = ValueType.fromU32(rawVt).unwrap()
                    val len = readLength(reader, magic).unwrap()
                    if (len > GGUF_MAX_ARRAY_ELEMENTS) {
                        bail("gguf: array length $len exceeds max $GGUF_MAX_ARRAY_ELEMENTS")
                    }
                    val needed = len.saturatingMul(elemVt.minDiskSize(magic))
                    val remaining = remainingBytes(reader, fileSize).unwrap()
                    if (needed > remaining) {
                        bail(
                            "gguf: array of $len elements needs at least $needed bytes, only $remaining remaining",
                        )
                    }
                    val vs = Vec.new<Value>()
                    for (i in 0uL until len) {
                        vs.push(
                            Value.read(
                                reader,
                                elemVt,
                                magic,
                                depth + 1,
                                fileSize,
                            ).unwrap(),
                        )
                    }
                    Array(vs)
                }
            }
            return Ok(v)
        }
    }
}

private fun <W : NpyByteWriter> writeString(w: W, str: String): Result<Unit> {
    val bytes = str.asBytes()
    w.writeU64<LittleEndian>(bytes.len().toULong()).unwrap()
    w.writeAll(bytes).unwrap()
    return Ok(Unit)
}

public fun <W : NpyByteWriter> write(
    w: W,
    metadata: List<Pair<String, Value>>,
    tensors: List<Pair<String, QTensor>>,
): Result<Unit> {
    w.writeU32<LittleEndian>(0x46554747u).unwrap()
    w.writeU32<LittleEndian>(2u).unwrap() // version 2.
    w.writeU64<LittleEndian>(tensors.len().toULong()).unwrap()
    w.writeU64<LittleEndian>(metadata.len().toULong()).unwrap()
    for ((name, value) in metadata.iter()) {
        writeString(w, name).unwrap()
        w.writeU32<LittleEndian>(value.valueType().toU32()).unwrap()
        value.write(w).unwrap()
    }
    var offset = 0
    val offsets = Vec.withCapacity<Int>(tensors.len())
    for ((name, tensor) in tensors.iter()) {
        writeString(w, name).unwrap()
        val dims = tensor.shape().dims()
        w.writeU32<LittleEndian>(dims.len().toUInt()).unwrap()
        for (dim in dims.iter().rev()) {
            w.writeU64<LittleEndian>(dim.toULong()).unwrap()
        }
        w.writeU32<LittleEndian>(tensor.dtype().toU32()).unwrap()
        w.writeU64<LittleEndian>(offset.toULong()).unwrap()
        offsets.push(offset)
        val sizeInBytes = tensor.storageSizeInBytes()
        val padding = 31 - (31 + sizeInBytes) % 32
        offset += sizeInBytes + padding
    }
    val pos = w.streamPosition().unwrap().toInt()
    val padding = 31 - (31 + pos) % 32
    w.writeAll(ByteArray(padding)).unwrap()
    val tensorStartPos = w.streamPosition().unwrap().toInt()
    for ((expectedOffset, pair) in offsets.iter().zip(tensors.iter())) {
        val tensor = pair.second
        val curPos = w.streamPosition().unwrap().toInt()
        if (tensorStartPos + expectedOffset != curPos) {
            bail(
                "internal error, unexpected current position $tensorStartPos $expectedOffset $curPos",
            )
        }
        val data = tensor.data().unwrap()
        val sizeInBytes = data.len()
        w.writeAll(data).unwrap()
        val tensorPad = 31 - (31 + sizeInBytes) % 32
        w.writeAll(ByteArray(tensorPad)).unwrap()
    }
    return Ok(Unit)
}

