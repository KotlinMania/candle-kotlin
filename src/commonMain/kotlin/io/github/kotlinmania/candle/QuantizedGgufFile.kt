// port-lint: source quantized/gguf_file.rs
/**
 * Support for the [GGUF file format](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md).
 *
 * Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
 */
package io.github.kotlinmania.candle.gguf

import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.NpyByteReader
import io.github.kotlinmania.candle.NpyByteWriter
import io.github.kotlinmania.candle.QTensor
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.ggml.qtensorFromGgml

const val DEFAULT_ALIGNMENT: ULong = 32uL

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
private fun remainingBytes(reader: NpyByteReader, fileSize: ULong): CandleResult<ULong> {
    val cur = reader.position.toULong()
    return CandleResult.Ok(if (fileSize >= cur) fileSize - cur else 0uL)
}

private fun readLength(reader: NpyByteReader, magic: VersionedMagic): CandleResult<ULong> {
    return when (magic) {
        VersionedMagic.GgufV1 -> {
            val b = ByteArray(4)
            when (val res = reader.readExact(b)) {
                is CandleResult.Ok -> {
                    val v = ((b[0].toInt() and 0xFF) or
                        ((b[1].toInt() and 0xFF) shl 8) or
                        ((b[2].toInt() and 0xFF) shl 16) or
                        ((b[3].toInt() and 0xFF) shl 24)).toUInt()
                    CandleResult.Ok(v.toULong())
                }
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }
        VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> {
            val b = ByteArray(8)
            when (val res = reader.readExact(b)) {
                is CandleResult.Ok -> {
                    var v = 0uL
                    for (i in 0 until 8) {
                        v = v or ((b[i].toLong() and 0xFFL).toULong() shl (i * 8))
                    }
                    CandleResult.Ok(v)
                }
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }
    }
}

private enum class Magic {
    Gguf;

    interface Error

    companion object {
        fun tryFrom(value: UInt): CandleResult<Magic> {
            val magic = when (value) {
                0x46554747u, 0x47475546u -> Gguf
                else -> return CandleResult.Err(CandleError.Msg("unknown magic 0x${value.toString(16)}").bt())
            }
            return CandleResult.Ok(magic)
        }
    }
}

enum class VersionedMagic {
    GgufV1,
    GgufV2,
    GgufV3;

    internal fun lengthPrefixSize(): ULong = when (this) {
        GgufV1 -> 4uL
        GgufV2, GgufV3 -> 8uL
    }

    companion object {
        internal fun read(reader: NpyByteReader): CandleResult<VersionedMagic> {
            val b4 = ByteArray(4)
            when (val res = reader.readExact(b4)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val rawMagic = ((b4[0].toInt() and 0xFF) or
                ((b4[1].toInt() and 0xFF) shl 8) or
                ((b4[2].toInt() and 0xFF) shl 16) or
                ((b4[3].toInt() and 0xFF) shl 24)).toUInt()
            val magic = when (val res = Magic.tryFrom(rawMagic)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            when (val res = reader.readExact(b4)) {
                is CandleResult.Ok -> {}
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val version = ((b4[0].toInt() and 0xFF) or
                ((b4[1].toInt() and 0xFF) shl 8) or
                ((b4[2].toInt() and 0xFF) shl 16) or
                ((b4[3].toInt() and 0xFF) shl 24)).toUInt()
            val versionedMagic = when {
                magic == Magic.Gguf && version == 1u -> GgufV1
                magic == Magic.Gguf && version == 2u -> GgufV2
                magic == Magic.Gguf && version == 3u -> GgufV3
                else -> return CandleResult.Err(
                    CandleError.Msg("gguf: unsupported magic/version $magic/$version").bt()
                )
            }
            return CandleResult.Ok(versionedMagic)
        }
    }
}

data class TensorInfo(
    val ggmlDtype: GgmlDType,
    val shape: Shape,
    val offset: ULong,
) {
    fun read(
        reader: NpyByteReader,
        fileSize: ULong,
        tensorDataOffset: ULong,
        device: Device,
    ): CandleResult<QTensor> {
        val tensorElems = this.shape.elemCount()
        val blockSize = this.ggmlDtype.blockSize()
        if (tensorElems % blockSize != 0) {
            return CandleResult.Err(
                CandleError.Msg(
                    "the number of elements $tensorElems is not divisible by the block size $blockSize"
                ).bt()
            )
        }
        val sizeInBytes = tensorElems / blockSize * this.ggmlDtype.typeSize()
        val tensorStart = tensorDataOffset + this.offset
        val remaining = if (fileSize >= tensorStart) fileSize - tensorStart else 0uL
        if (sizeInBytes.toULong() > remaining) {
            return CandleResult.Err(
                CandleError.Msg(
                    "tensor needs $sizeInBytes bytes at offset $tensorStart, only $remaining remaining in file"
                ).bt()
            )
        }
        val rawData = ByteArray(sizeInBytes)
        reader.position = tensorStart.toInt()
        when (val res = reader.readExact(rawData)) {
            is CandleResult.Ok -> {}
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return qtensorFromGgml(
            this.ggmlDtype,
            rawData,
            this.shape.dims().toList(),
            device,
        )
    }
}

private fun readString(
    reader: NpyByteReader,
    magic: VersionedMagic,
    fileSize: ULong,
): CandleResult<String> {
    val len = when (val res = readLength(reader, magic)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    if (len > GGUF_MAX_STRING_LENGTH) {
        return CandleResult.Err(
            CandleError.Msg("gguf: string length $len exceeds max $GGUF_MAX_STRING_LENGTH").bt()
        )
    }
    val remaining = when (val res = remainingBytes(reader, fileSize)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    if (len > remaining) {
        return CandleResult.Err(
            CandleError.Msg("gguf: string length $len exceeds remaining file bytes $remaining").bt()
        )
    }
    val v = ByteArray(len.toInt())
    when (val res = reader.readExact(v)) {
        is CandleResult.Ok -> {}
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    // GGUF strings are supposed to be non-null terminated but in practice this happens.
    var effectiveLen = v.size
    while (effectiveLen > 0 && v[effectiveLen - 1] == 0.toByte()) {
        effectiveLen -= 1
    }
    // GGUF strings are utf8 encoded but there are cases that don't seem to be valid.
    return CandleResult.Ok(v.decodeToString(0, effectiveLen))
}

enum class ValueType {
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
    Array;

    fun toU32(): UInt = when (this) {
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

    companion object {
        internal fun fromU32(v: UInt): CandleResult<ValueType> {
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
                else -> return CandleResult.Err(CandleError.Msg("unrecognized value-type 0x${v.toString(16)}").bt())
            }
            return CandleResult.Ok(vt)
        }
    }
}

sealed class Value {
    data class U8(val v: UByte) : Value()
    data class I8(val v: Byte) : Value()
    data class U16(val v: UShort) : Value()
    data class I16(val v: Short) : Value()
    data class U32(val v: UInt) : Value()
    data class I32(val v: Int) : Value()
    data class U64(val v: ULong) : Value()
    data class I64(val v: Long) : Value()
    data class F32(val v: Float) : Value()
    data class F64(val v: Double) : Value()
    data class Bool(val v: Boolean) : Value()
    data class String(val v: kotlin.String) : Value()
    data class Array(val v: List<Value>) : Value()

    fun valueType(): ValueType = when (this) {
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

    fun toU8(): CandleResult<UByte> = when (this) {
        is U8 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a u8 $this").bt())
    }

    fun toI8(): CandleResult<Byte> = when (this) {
        is I8 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a i8 $this").bt())
    }

    fun toU16(): CandleResult<UShort> = when (this) {
        is U16 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a u16 $this").bt())
    }

    fun toI16(): CandleResult<Short> = when (this) {
        is I16 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a i16 $this").bt())
    }

    fun toU32(): CandleResult<UInt> = when (this) {
        is U32 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a u32 $this").bt())
    }

    fun toI32(): CandleResult<Int> = when (this) {
        is I32 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a i32 $this").bt())
    }

    /**
     * This will also automatically upcast any integral types which will not truncate.
     */
    fun toU64(): CandleResult<ULong> = when (this) {
        is U64 -> CandleResult.Ok(this.v)
        // Autoupcast cases here
        is U8 -> CandleResult.Ok(this.v.toULong())
        is U16 -> CandleResult.Ok(this.v.toULong())
        is U32 -> CandleResult.Ok(this.v.toULong())
        is Bool -> CandleResult.Ok(if (this.v) 1uL else 0uL)
        else -> CandleResult.Err(CandleError.Msg("not a u64 or upcastable to u64 $this").bt())
    }

    fun toI64(): CandleResult<Long> = when (this) {
        is I64 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a i64 $this").bt())
    }

    fun toF32(): CandleResult<Float> = when (this) {
        is F32 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a f32 $this").bt())
    }

    fun toF64(): CandleResult<Double> = when (this) {
        is F64 -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a f64 $this").bt())
    }

    fun toBool(): CandleResult<Boolean> = when (this) {
        is Bool -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a bool $this").bt())
    }

    fun toVec(): CandleResult<List<Value>> = when (this) {
        is Array -> CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a vec $this").bt())
    }

    fun toString(validate: Boolean = true): CandleResult<kotlin.String> = when (this) {
        is String -> if (validate || this.v.isNotEmpty()) CandleResult.Ok(this.v) else CandleResult.Ok(this.v)
        else -> CandleResult.Err(CandleError.Msg("not a string $this").bt())
    }

    internal fun write(w: NpyByteWriter): CandleResult<Unit> {
        val writeU16Le = { v: Int ->
            w.writeAll(byteArrayOf((v and 0xFF).toByte(), ((v ushr 8) and 0xFF).toByte()))
        }
        val writeI32Le = { v: Int ->
            w.writeAll(
                byteArrayOf(
                    (v and 0xFF).toByte(),
                    ((v ushr 8) and 0xFF).toByte(),
                    ((v ushr 16) and 0xFF).toByte(),
                    ((v ushr 24) and 0xFF).toByte(),
                )
            )
        }
        val writeI64Le = { v: Long ->
            val b = ByteArray(8)
            for (i in 0 until 8) {
                b[i] = ((v ushr (i * 8)) and 0xFFL).toByte()
            }
            w.writeAll(b)
        }
        return when (this) {
            is U8 -> w.writeAll(byteArrayOf(this.v.toByte()))
            is I8 -> w.writeAll(byteArrayOf(this.v))
            is U16 -> writeU16Le(this.v.toInt())
            is I16 -> writeU16Le(this.v.toInt())
            is U32 -> writeI32Le(this.v.toInt())
            is I32 -> writeI32Le(this.v)
            is U64 -> writeI64Le(this.v.toLong())
            is I64 -> writeI64Le(this.v)
            is F32 -> writeI32Le(this.v.toRawBits())
            is F64 -> writeI64Le(this.v.toRawBits())
            is Bool -> w.writeAll(byteArrayOf(if (this.v) 1 else 0))
            is String -> writeString(w, this.v)
            is Array -> {
                // The `Value` type does not enforce that all the values in an Array have the same
                // type.
                val elemValueType = if (this.v.isEmpty()) {
                    // Doesn't matter, the array is empty.
                    ValueType.U32
                } else {
                    val types = this.v.map { it.valueType() }.toSet()
                    if (types.size != 1) {
                        return CandleResult.Err(
                            CandleError.Msg("multiple value-types in the same array $types").bt()
                        )
                    }
                    types.first()
                }
                writeI32Le(elemValueType.toU32().toInt())
                writeI64Le(this.v.size.toLong())
                for (elem in this.v) {
                    when (val res = elem.write(w)) {
                        is CandleResult.Ok -> {}
                        is CandleResult.Err -> return CandleResult.Err(res.error)
                    }
                }
                CandleResult.Ok(Unit)
            }
        }
    }

    companion object {
        internal fun read(
            reader: NpyByteReader,
            valueType: ValueType,
            magic: VersionedMagic,
            depth: Int,
            fileSize: ULong,
        ): CandleResult<Value> {
            if (depth > GGUF_MAX_VALUE_DEPTH) {
                return CandleResult.Err(
                    CandleError.Msg("gguf: value nesting depth exceeds max $GGUF_MAX_VALUE_DEPTH").bt()
                )
            }
            val readBytes = { count: Int ->
                val b = ByteArray(count)
                when (val res = reader.readExact(b)) {
                    is CandleResult.Ok -> CandleResult.Ok(b)
                    is CandleResult.Err -> CandleResult.Err(res.error)
                }
            }
            val v = when (valueType) {
                ValueType.U8 -> {
                    val b = when (val r = readBytes(1)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    U8(b[0].toUByte())
                }
                ValueType.I8 -> {
                    val b = when (val r = readBytes(1)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    I8(b[0])
                }
                ValueType.U16 -> {
                    val b = when (val r = readBytes(2)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    U16(((b[0].toInt() and 0xFF) or ((b[1].toInt() and 0xFF) shl 8)).toUShort())
                }
                ValueType.I16 -> {
                    val b = when (val r = readBytes(2)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    I16(((b[0].toInt() and 0xFF) or ((b[1].toInt() and 0xFF) shl 8)).toShort())
                }
                ValueType.U32 -> {
                    val b = when (val r = readBytes(4)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    val raw = (b[0].toInt() and 0xFF) or
                        ((b[1].toInt() and 0xFF) shl 8) or
                        ((b[2].toInt() and 0xFF) shl 16) or
                        ((b[3].toInt() and 0xFF) shl 24)
                    U32(raw.toUInt())
                }
                ValueType.I32 -> {
                    val b = when (val r = readBytes(4)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    val raw = (b[0].toInt() and 0xFF) or
                        ((b[1].toInt() and 0xFF) shl 8) or
                        ((b[2].toInt() and 0xFF) shl 16) or
                        ((b[3].toInt() and 0xFF) shl 24)
                    I32(raw)
                }
                ValueType.U64 -> {
                    val b = when (val r = readBytes(8)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    var raw = 0uL
                    for (i in 0 until 8) {
                        raw = raw or ((b[i].toLong() and 0xFFL).toULong() shl (i * 8))
                    }
                    U64(raw)
                }
                ValueType.I64 -> {
                    val b = when (val r = readBytes(8)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    var raw = 0L
                    for (i in 0 until 8) {
                        raw = raw or ((b[i].toLong() and 0xFFL) shl (i * 8))
                    }
                    I64(raw)
                }
                ValueType.F32 -> {
                    val b = when (val r = readBytes(4)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    val raw = (b[0].toInt() and 0xFF) or
                        ((b[1].toInt() and 0xFF) shl 8) or
                        ((b[2].toInt() and 0xFF) shl 16) or
                        ((b[3].toInt() and 0xFF) shl 24)
                    F32(Float.fromBits(raw))
                }
                ValueType.F64 -> {
                    val b = when (val r = readBytes(8)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    var raw = 0L
                    for (i in 0 until 8) {
                        raw = raw or ((b[i].toLong() and 0xFFL) shl (i * 8))
                    }
                    F64(Double.fromBits(raw))
                }
                ValueType.Bool -> {
                    val b = when (val r = readBytes(1)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    when (val flag = b[0].toInt() and 0xFF) {
                        0 -> Bool(false)
                        1 -> Bool(true)
                        else -> return CandleResult.Err(CandleError.Msg("unexpected bool value $flag").bt())
                    }
                }
                ValueType.String -> {
                    val s = when (val r = readString(reader, magic, fileSize)) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    String(s)
                }
                ValueType.Array -> {
                    val b4 = when (val r = readBytes(4)) { is CandleResult.Ok -> r.value; is CandleResult.Err -> return CandleResult.Err(r.error) }
                    val rawVt = ((b4[0].toInt() and 0xFF) or
                        ((b4[1].toInt() and 0xFF) shl 8) or
                        ((b4[2].toInt() and 0xFF) shl 16) or
                        ((b4[3].toInt() and 0xFF) shl 24)).toUInt()
                    val elemVt = when (val r = ValueType.fromU32(rawVt)) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    val len = when (val r = readLength(reader, magic)) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    if (len > GGUF_MAX_ARRAY_ELEMENTS) {
                        return CandleResult.Err(
                            CandleError.Msg("gguf: array length $len exceeds max $GGUF_MAX_ARRAY_ELEMENTS").bt()
                        )
                    }
                    val needed = len * elemVt.minDiskSize(magic)
                    val remaining = when (val r = remainingBytes(reader, fileSize)) {
                        is CandleResult.Ok -> r.value
                        is CandleResult.Err -> return CandleResult.Err(r.error)
                    }
                    if (needed > remaining) {
                        return CandleResult.Err(
                            CandleError.Msg(
                                "gguf: array of $len elements needs at least $needed bytes, only $remaining remaining"
                            ).bt()
                        )
                    }
                    val vs = ArrayList<Value>(len.toInt())
                    for (i in 0 until len.toInt()) {
                        val elem = when (val r = read(reader, elemVt, magic, depth + 1, fileSize)) {
                            is CandleResult.Ok -> r.value
                            is CandleResult.Err -> return CandleResult.Err(r.error)
                        }
                        vs.add(elem)
                    }
                    Array(vs)
                }
            }
            return CandleResult.Ok(v)
        }
    }
}

class Content(
    val magic: VersionedMagic,
    val metadata: Map<String, Value>,
    val tensorInfos: Map<String, TensorInfo>,
    val tensorDataOffset: ULong,
) {
    fun tensor(
        reader: NpyByteReader,
        fileSize: ULong,
        name: String,
        device: Device,
    ): CandleResult<QTensor> {
        val tensorInfo = this.tensorInfos[name]
            ?: return CandleResult.Err(CandleError.Msg("cannot find tensor info for $name").bt())
        return tensorInfo.read(reader, fileSize, this.tensorDataOffset, device)
    }

    companion object {
        fun read(reader: NpyByteReader, fileSize: ULong): CandleResult<Content> {
            // Capture the file size once so the bounds checks below don't have to
            // seek to the end and back on every length-prefixed read.
            val readU32 = {
                val b = ByteArray(4)
                when (val res = reader.readExact(b)) {
                    is CandleResult.Ok -> {
                        val v = (b[0].toInt() and 0xFF) or
                            ((b[1].toInt() and 0xFF) shl 8) or
                            ((b[2].toInt() and 0xFF) shl 16) or
                            ((b[3].toInt() and 0xFF) shl 24)
                        CandleResult.Ok(v.toUInt())
                    }
                    is CandleResult.Err -> CandleResult.Err(res.error)
                }
            }
            val readU64 = {
                val b = ByteArray(8)
                when (val res = reader.readExact(b)) {
                    is CandleResult.Ok -> {
                        var v = 0uL
                        for (i in 0 until 8) {
                            v = v or ((b[i].toLong() and 0xFFL).toULong() shl (i * 8))
                        }
                        CandleResult.Ok(v)
                    }
                    is CandleResult.Err -> CandleResult.Err(res.error)
                }
            }

            val magic = when (val r = VersionedMagic.read(reader)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val tensorCount = when (val r = readLength(reader, magic)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            val metadataKvCount = when (val r = readLength(reader, magic)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }

            if (tensorCount > GGUF_MAX_ARRAY_ELEMENTS) {
                return CandleResult.Err(
                    CandleError.Msg("gguf: tensorCount $tensorCount exceeds max $GGUF_MAX_ARRAY_ELEMENTS").bt()
                )
            }
            if (metadataKvCount > GGUF_MAX_ARRAY_ELEMENTS) {
                return CandleResult.Err(
                    CandleError.Msg("gguf: metadataKvCount $metadataKvCount exceeds max $GGUF_MAX_ARRAY_ELEMENTS").bt()
                )
            }

            // Reject header-declared counts that can't fit in the file at minimum size.
            // Per-entry minima: a metadata kv is at least `keyLenPrefix + u32 valueType
            // + 1 byte value`; a tensor info is at least `nameLenPrefix + u32 nDims
            // + u32 dtype + u64 offset`.
            val prefix = magic.lengthPrefixSize()
            val minPerKv = prefix + 4uL + 1uL
            val minPerTensor = prefix + 4uL + 4uL + 8uL
            val needed = metadataKvCount * minPerKv + tensorCount * minPerTensor
            val remaining = when (val r = remainingBytes(reader, fileSize)) {
                is CandleResult.Ok -> r.value
                is CandleResult.Err -> return CandleResult.Err(r.error)
            }
            if (needed > remaining) {
                return CandleResult.Err(
                    CandleError.Msg(
                        "gguf: header declares $tensorCount tensors and $metadataKvCount metadata entries, needs at least $needed bytes, only $remaining remaining"
                    ).bt()
                )
            }

            val metadata = HashMap<String, Value>()
            for (idx in 0 until metadataKvCount.toInt()) {
                val key = when (val r = readString(reader, magic, fileSize)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val rawVt = when (val r = readU32()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val valueType = when (val r = ValueType.fromU32(rawVt)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val value = when (val r = Value.read(reader, valueType, magic, 0, fileSize)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                metadata[key] = value
            }
            val tensorInfos = HashMap<String, TensorInfo>()
            for (idx in 0 until tensorCount.toInt()) {
                val tensorName = when (val r = readString(reader, magic, fileSize)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val nDimensions = when (val r = readU32()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                if (nDimensions > GGUF_MAX_TENSOR_DIMS) {
                    return CandleResult.Err(
                        CandleError.Msg(
                            "gguf: tensor '$tensorName' has $nDimensions dimensions, max is $GGUF_MAX_TENSOR_DIMS"
                        ).bt()
                    )
                }

                val dimensions = ArrayList<Int>(nDimensions.toInt())
                when (magic) {
                    VersionedMagic.GgufV1 -> {
                        for (d in 0 until nDimensions.toInt()) {
                            val dim = when (val r = readU32()) {
                                is CandleResult.Ok -> r.value.toInt()
                                is CandleResult.Err -> return CandleResult.Err(r.error)
                            }
                            dimensions.add(dim)
                        }
                    }
                    VersionedMagic.GgufV2, VersionedMagic.GgufV3 -> {
                        for (d in 0 until nDimensions.toInt()) {
                            val dim = when (val r = readU64()) {
                                is CandleResult.Ok -> r.value.toInt()
                                is CandleResult.Err -> return CandleResult.Err(r.error)
                            }
                            dimensions.add(dim)
                        }
                    }
                }

                dimensions.reverse()
                val rawGgmlDtype = when (val r = readU32()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val ggmlDtype = when (val r = GgmlDType.fromU32(rawGgmlDtype)) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                val offset = when (val r = readU64()) {
                    is CandleResult.Ok -> r.value
                    is CandleResult.Err -> return CandleResult.Err(r.error)
                }
                tensorInfos[tensorName] = TensorInfo(
                    ggmlDtype = ggmlDtype,
                    shape = Shape.fromDims(dimensions.toIntArray()),
                    offset = offset,
                )
            }
            val position = reader.position.toULong()
            val alignment = when (val alignVal = metadata["general.alignment"]) {
                is Value.U8 -> alignVal.v.toULong()
                is Value.U16 -> alignVal.v.toULong()
                is Value.U32 -> alignVal.v.toULong()
                is Value.I8 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                is Value.I16 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                is Value.I32 -> if (alignVal.v >= 0) alignVal.v.toULong() else DEFAULT_ALIGNMENT
                else -> DEFAULT_ALIGNMENT
            }
            val tensorDataOffset = ((position + alignment - 1uL) / alignment) * alignment
            return CandleResult.Ok(
                Content(
                    magic = magic,
                    metadata = metadata,
                    tensorInfos = tensorInfos,
                    tensorDataOffset = tensorDataOffset,
                )
            )
        }
    }
}

private fun writeString(w: NpyByteWriter, str: String): CandleResult<Unit> {
    val bytes = str.encodeToByteArray()
    val len = bytes.size.toLong()
    val lenBytes = ByteArray(8)
    for (i in 0 until 8) {
        lenBytes[i] = ((len ushr (i * 8)) and 0xFFL).toByte()
    }
    w.writeAll(lenBytes)
    w.writeAll(bytes)
    return CandleResult.Ok(Unit)
}

fun write(
    w: NpyByteWriter,
    metadata: List<Pair<String, Value>>,
    tensors: List<Pair<String, QTensor>>,
): CandleResult<Unit> {
    val writeU32 = { v: UInt ->
        val iv = v.toInt()
        w.writeAll(
            byteArrayOf(
                (iv and 0xFF).toByte(),
                ((iv ushr 8) and 0xFF).toByte(),
                ((iv ushr 16) and 0xFF).toByte(),
                ((iv ushr 24) and 0xFF).toByte(),
            )
        )
    }
    val writeU64 = { v: ULong ->
        val lv = v.toLong()
        val b = ByteArray(8)
        for (i in 0 until 8) {
            b[i] = ((lv ushr (i * 8)) and 0xFFL).toByte()
        }
        w.writeAll(b)
    }
    writeU32(0x46554747u)
    writeU32(2u) // version 2.
    writeU64(tensors.size.toULong())
    writeU64(metadata.size.toULong())
    for ((name, value) in metadata) {
        writeString(w, name)
        writeU32(value.valueType().toU32())
        when (val res = value.write(w)) {
            is CandleResult.Ok -> {}
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
    }
    var offset = 0
    val offsets = ArrayList<Int>(tensors.size)
    for ((name, tensor) in tensors) {
        writeString(w, name)
        val dims = tensor.shape().dims()
        writeU32(dims.size.toUInt())
        for (idx in dims.indices.reversed()) {
            writeU64(dims[idx].toULong())
        }
        writeU32(tensor.dtype().toU32())
        writeU64(offset.toULong())
        offsets.add(offset)
        val sizeInBytes = tensor.storageSizeInBytes()
        val padding = 31 - (31 + sizeInBytes) % 32
        offset += sizeInBytes + padding
    }
    val pos = w.size
    val padding = 31 - (31 + pos) % 32
    w.writeAll(ByteArray(padding))
    val tensorStartPos = w.size
    for (i in tensors.indices) {
        val expectedOffset = offsets[i]
        val tensor = tensors[i].second
        val curPos = w.size
        if (tensorStartPos + expectedOffset != curPos) {
            return CandleResult.Err(
                CandleError.Msg(
                    "internal error, unexpected current position $tensorStartPos $expectedOffset $curPos"
                ).bt()
            )
        }
        val data = when (val res = tensor.data()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val sizeInBytes = data.size
        w.writeAll(data)
        val tensorPad = 31 - (31 + sizeInBytes) % 32
        w.writeAll(ByteArray(tensorPad))
    }
    return CandleResult.Ok(Unit)
}
