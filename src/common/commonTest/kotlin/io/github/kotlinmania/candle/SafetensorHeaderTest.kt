package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFailsWith

class SafetensorHeaderTest {
    @Test
    fun readsTensorBytesFromTheWireFormat() {
        val bytes = safetensorBytes("""{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}   """, byteArrayOf(1, 3))
        val header = readSafetensorHeader(bytes)
        val tensor = header.tensors.single()
        assertEquals("x", tensor.name)
        assertEquals("U8", tensor.dtype)
        assertContentEquals(intArrayOf(2), tensor.shape)
        tensor.validateSize(8)
        assertContentEquals(byteArrayOf(1, 3), bytes.copyOfRange(header.dataStart + tensor.start, header.dataStart + tensor.end))
    }

    @Test
    fun readsEscapedNamesAndMetadataThroughSerde() {
        val header =
            readSafetensorHeader(
                safetensorBytes(
                    """{"__metadata__":{"description":"quote\" and } { \\ and \n"},"tensor\"\\\n😀":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}""",
                    byteArrayOf(7),
                ),
            )
        assertEquals("tensor\"\\\n😀", header.tensors.single().name)
        assertEquals(1, header.tensors.size)
    }

    @Test
    fun permitsJsonWhitespaceAndEmptyMetadata() {
        for (json in listOf("\t\n{}\r ", "{\"__metadata__\":null}", "{\"__metadata__\":{}}")) {
            assertEquals(emptyList(), readSafetensorHeader(safetensorBytes(json)).tensors)
        }
    }

    @Test
    fun rejectsMalformedJsonAndTrailingDocuments() {
        for (json in listOf("", "{", "[]", "null", "{}{}", "{} garbage", "{\"bad\\q\":{}}")) {
            assertFailsWith<IllegalArgumentException>(json) { readSafetensorHeader(safetensorBytes(json)) }
        }
    }

    @Test
    fun rejectsInvalidUtf8() {
        val bytes = safetensorBytes("{}")
        bytes[8] = 0xff.toByte()
        assertFails { readSafetensorHeader(bytes) }
    }

    @Test
    fun validatesTheFullUnsignedHeaderLengthBeforeNarrowing() {
        assertFailsWith<IllegalArgumentException> { readSafetensorHeader(ByteArray(7)) }
        for (length in listOf(3uL, 100_000_001uL, 4_294_967_298uL, ULong.MAX_VALUE)) {
            val bytes = safetensorBytes("{}")
            for (index in 0 until 8) bytes[index] = (length shr (index * 8)).toByte()
            assertFailsWith<IllegalArgumentException>("length $length") { readSafetensorHeader(bytes) }
        }
    }

    @Test
    fun rejectsInvalidFieldTypesAndMissingFields() {
        for (fields in listOf(
            "{}",
            "[]",
            """{"dtype":1,"shape":[],"data_offsets":[0,0]}""",
            """{"dtype":"U8","shape":{},"data_offsets":[0,0]}""",
            """{"dtype":"U8","shape":[],"data_offsets":{}}""",
            """{"dtype":"U8","shape":[],"data_offsets":[0]}""",
            """{"dtype":"U8","shape":[],"data_offsets":[0,0,0]}""",
        )) {
            assertFailsWith<IllegalArgumentException>(fields) { readSafetensorHeader(safetensorBytes("{\"x\":$fields}")) }
        }
        for (metadata in listOf("[]", "1", "{\"x\":1}", "{\"x\":null}")) {
            assertFailsWith<IllegalArgumentException>(metadata) {
                readSafetensorHeader(safetensorBytes("{\"__metadata__\":$metadata}"))
            }
        }
    }

    @Test
    fun rejectsNegativeFractionalAndUnrepresentableDimensions() {
        for (dimension in listOf("-1", "1.5", "1.0", "2147483648", "18446744073709551615", "\"1\"")) {
            val json = """{"x":{"dtype":"U8","shape":[$dimension],"data_offsets":[0,0]}}"""
            assertFailsWith<IllegalArgumentException>(dimension) { readSafetensorHeader(safetensorBytes(json)) }
        }
    }

    @Test
    fun rejectsInvalidOffsetsWithoutSlicingTheBuffer() {
        for (offsets in listOf("-1,0", "0,2", "1,0", "0,1.0", "0,2147483648", "0,\"1\"")) {
            val json = """{"x":{"dtype":"U8","shape":[1],"data_offsets":[$offsets]}}"""
            assertFailsWith<IllegalArgumentException>(offsets) { readSafetensorHeader(safetensorBytes(json, byteArrayOf(1))) }
        }
    }

    @Test
    fun ordersTensorRangesByOffsetAndAllowsEmptyTensors() {
        val header =
            readSafetensorHeader(
                safetensorBytes(
                    """{"second":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"first":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[1,1]}}""",
                    byteArrayOf(3, 7),
                ),
            )
        assertEquals(listOf("first", "empty", "second"), header.tensors.map { it.name })
        header.tensors.forEach { it.validateSize(8) }
    }

    @Test
    fun rejectsGapsOverlapsAndUnclaimedData() {
        for (ranges in listOf("0,1" to "2,3", "0,2" to "1,3")) {
            val json = """{"a":{"dtype":"U8","shape":[1],"data_offsets":[${ranges.first}]},"b":{"dtype":"U8","shape":[1],"data_offsets":[${ranges.second}]}}"""
            assertFailsWith<IllegalArgumentException> { readSafetensorHeader(safetensorBytes(json, ByteArray(3))) }
        }
        assertFailsWith<IllegalArgumentException> { readSafetensorHeader(safetensorBytes("{}", byteArrayOf(1))) }
        val json = """{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"""
        assertFailsWith<IllegalArgumentException> { readSafetensorHeader(safetensorBytes(json, ByteArray(2))) }
    }

    @Test
    fun validatesScalarEmptyAndPackedTensorSizes() {
        SafetensorHeaderEntry("scalar", "F32", intArrayOf(), 0, 4).validateSize(32)
        SafetensorHeaderEntry("empty", "F32", intArrayOf(2, 0), 0, 0).validateSize(32)
        SafetensorHeaderEntry("packed", "F4", intArrayOf(2), 0, 1).validateSize(4)
        SafetensorHeaderEntry("packed", "F6_E2M3", intArrayOf(4), 0, 3).validateSize(6)
        assertFailsWith<IllegalArgumentException> {
            SafetensorHeaderEntry("packed", "F4", intArrayOf(1), 0, 1).validateSize(4)
        }
        assertFailsWith<IllegalArgumentException> {
            SafetensorHeaderEntry("mismatch", "F32", intArrayOf(2, 2), 0, 4).validateSize(32)
        }
    }

    @Test
    fun rejectsDimensionAndBitCountOverflow() {
        for (shape in listOf(intArrayOf(Int.MAX_VALUE, Int.MAX_VALUE, Int.MAX_VALUE), intArrayOf(Int.MAX_VALUE, Int.MAX_VALUE))) {
            assertFailsWith<IllegalArgumentException> {
                SafetensorHeaderEntry("overflow", "F64", shape, 0, 0).validateSize(64)
            }
        }
    }
}

private fun safetensorBytes(
    json: String,
    payload: ByteArray = ByteArray(0),
): ByteArray {
    val header = json.encodeToByteArray()
    val bytes = ByteArray(8 + header.size + payload.size)
    for (index in 0 until 8) bytes[index] = (header.size.toLong() shr (index * 8)).toByte()
    header.copyInto(bytes, 8)
    payload.copyInto(bytes, 8 + header.size)
    return bytes
}
