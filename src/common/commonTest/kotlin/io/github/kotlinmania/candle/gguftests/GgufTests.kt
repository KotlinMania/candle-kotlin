@file:OptIn(ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.gguftests

import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.quantized.gguffile.Content

// Regression tests for the allocation caps added in huggingface/candle#3533.

public val GGUF_MAGIC: UByteArray = "GGUF".encodeToByteArray().toUByteArray()

public val GGUF_V3: UInt = 3u

public fun header(tensorCount: ULong, metadataKvCount: ULong): List<UByte> {
    val buf = mutableListOf<UByte>()
    buf.addAll(GGUF_MAGIC)
    buf.addAll(
        run {
            val rawBits = GGUF_V3
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    buf.addAll(
        run {
            val rawBits = tensorCount
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    buf.addAll(
        run {
            val rawBits = metadataKvCount
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    return buf
}

public fun lengthPrefixed(s: List<UByte>): List<UByte> {
    val buf =
        run {
            val rawBits = (s.size.toULong())
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        }.toVec()
    buf.extendFromSlice(s)
    return buf
}

/**
 * Pad past the upfront `count * minPerEntry` sufficient-bytes check so
 * the inner per-field cap is what fires.
 */
public fun pad(buf: List<UByte>, n: Int) {
    buf.resize(buf.size + n, 0)
}

public fun assertRejects(buf: List<UByte>, msgContains: String) {
    val cursor = Cursor.new(buf)
    val err = Content.read(cursor).expectErr("expected Err")
    val msg = "$err"
    check(msg.contains(msgContains)) { "unexpected error: $msg" }
}

@kotlin.test.Test
public fun emptyHeaderLoads() {
    val cursor = Cursor.new(header(0uL, 0uL))
    val content = Content.read(cursor).expect("empty header should parse")
    check(content.metadata.isEmpty())
    check(content.tensorInfos.isEmpty())
}

@kotlin.test.Test
public fun rejectsOversizedMetadataKvCount() {
    assertRejects(header(0uL, 1uL shl 31), "metadata_kv_count")
}

@kotlin.test.Test
public fun rejectsOversizedTensorCount() {
    assertRejects(header(1uL shl 31, 0uL), "tensor_count")
}

@kotlin.test.Test
public fun rejectsOversizedStringLength() {
    val buf = header(1uL, 0uL)
    buf.extendFromSlice(
        run {
            val rawBits = (1uL shl 31).toULong()
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    pad(buf, 64)
    assertRejects(buf, "string length")
}

@kotlin.test.Test
public fun rejectsOversizedArrayLength() {
    val buf = header(0uL, 1uL)
    buf.extend(lengthPrefixed("k".encodeToByteArray()))
    buf.extendFromSlice(
        run {
            val rawBits = 9u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // valueType = Array
    buf.extendFromSlice(
        run {
            val rawBits = 0u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // inner type = U8
    buf.extendFromSlice(
        run {
            val rawBits = (1uL shl 31).toULong()
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    // element count
    pad(buf, 64)
    assertRejects(buf, "array length")
}

@kotlin.test.Test
public fun rejectsOversizedNDimensions() {
    val buf = header(1uL, 0uL)
    buf.extend(lengthPrefixed("t".encodeToByteArray()))
    buf.extendFromSlice(UInt.MAX_VALUE.toLeBytes())
    pad(buf, 64)
    assertRejects(buf, "dimensions")
}

@kotlin.test.Test
public fun rejectsDeeplyNestedArrays() {
    // 256 nested arrays of arrays, well above GGUF_MAX_VALUE_DEPTH (64).
    // Each layer is type=Array(9) + innerType(u32) + len=1(u64) = 16 bytes
    // except the innermost which terminates with a U8 (type=0) + len=0(u64).
    val buf = header(0uL, 1uL)
    buf.extend(lengthPrefixed("k".encodeToByteArray()))
    buf.extendFromSlice(
        run {
            val rawBits = 9u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // outer valueType = Array
    for (unused in 0 until 256) {
        buf.extendFromSlice(
            run {
                val rawBits = 9u
                ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
            },
        )
        // inner type = Array
        buf.extendFromSlice(
            run {
                val rawBits = 1uL
                ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
            },
        )
        // len = 1
    }
    buf.extendFromSlice(
        run {
            val rawBits = 0u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // terminal inner type = U8
    buf.extendFromSlice(
        run {
            val rawBits = 0uL
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    // terminal len = 0
    assertRejects(buf, "nesting depth")
}

@kotlin.test.Test
public fun emptyV1HeaderLoads() {
    val buf = mutableListOf()
    buf.addAll(GGUF_MAGIC)
    buf.addAll(
        run {
            val rawBits = 1u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // version 1: u32 length prefixes
    buf.addAll(
        run {
            val rawBits = 0u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // tensorCount
    buf.addAll(
        run {
            val rawBits = 0u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // metadataKvCount
    val cursor = Cursor.new(buf)
    Content.read(cursor).expect("empty v1 header should parse")
}

@kotlin.test.Test
public fun rejectsTensorSizeExceedingFile() {
    // Create a gguf tensor that claims shape [1_073_741_824] F32 (4 GiB), but has no actual tensor data.
    val buf = header(1uL, 0uL)
    buf.extend(lengthPrefixed("t".encodeToByteArray()))
    buf.extendFromSlice(
        run {
            val rawBits = 1u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // n dims
    buf.extendFromSlice(
        run {
            val rawBits = 1073741824uL
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    // 1 GiB elements
    buf.extendFromSlice(
        run {
            val rawBits = 0u
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte())
        },
    )
    // F32
    buf.extendFromSlice(
        run {
            val rawBits = 0uL
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    // no offset
    val cursor = Cursor.new(buf)
    val content = Content.read(cursor).expect("header should parse")
    val err = content.tensor(cursor, "t", Device.Cpu).expectErr("expected Err from oversized tensor load")
    val msg = "$err"
    check(msg.contains("remaining")) { "unexpected error: $msg" }
}

@kotlin.test.Test
public fun rejectsStringLengthAboveRemainingFileBytes() {
    val buf = header(1uL, 0uL)
    buf.extendFromSlice(
        run {
            val rawBits = (1uL shl 20).toULong()
            ubyteArrayOf(rawBits.toUByte(), (rawBits shr 8).toUByte(), (rawBits shr 16).toUByte(), (rawBits shr 24).toUByte(), (rawBits shr 32).toUByte(), (rawBits shr 40).toUByte(), (rawBits shr 48).toUByte(), (rawBits shr 56).toUByte())
        },
    )
    // 1 MB, below cap, above file size
    pad(buf, 64)
    assertRejects(buf, "string length")
}
