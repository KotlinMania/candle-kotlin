package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertIs

class SafetensorsDeserializationTest {
    @Test
    fun loadsExistingWireFixtures() {
        val tensors = SafeTensors.deserialize(SafeTensors.sampleU8SafetensorsBytes).unwrap()
        val tensor = tensors.tensor("x").unwrap()
        assertEquals(SafeTensorDtype.U8, tensor.wireDtype)
        assertContentEquals(intArrayOf(2), tensor.tensorShape)
        assertContentEquals(byteArrayOf(1, 3), tensor.rawBytes)
        assertEquals(listOf("t"), SafeTensors.deserialize(SafeTensors.expectedSingleTensorBytes).unwrap().names())
        assertEquals(listOf("t", "u"), SafeTensors.deserialize(SafeTensors.expectedMultiTensorBytes).unwrap().names())
    }

    @Test
    fun decodesEscapedTensorNames() {
        val bytes =
            encodedFile(
                """{"tensor\"\\\n😀":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}""",
                byteArrayOf(9),
            )
        val tensor =
            SafeTensors
                .deserialize(bytes)
                .unwrap()
                .tensor("tensor\"\\\n😀")
                .unwrap()
        assertContentEquals(byteArrayOf(9), tensor.rawBytes)
    }

    @Test
    fun returnsErrorsForMalformedInputAndInconsistentShapes() {
        for (json in listOf(
            "{} trailing",
            """{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}""",
            """{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,2]}}""",
            """{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}""",
        )) {
            assertIs<CandleResult.Err>(SafeTensors.deserialize(encodedFile(json, byteArrayOf(1))), json)
        }
    }

    @Test
    fun retainsUnsupportedDtypeErrors() {
        val bytes = encodedFile("""{"x":{"dtype":"unknown","shape":[0],"data_offsets":[0,0]}}""")
        val error = assertIs<CandleResult.Err>(SafeTensors.deserialize(bytes))
        assertIs<CandleException.UnsupportedSafeTensorDtype>(error.error)
    }

    private fun encodedFile(
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
}
