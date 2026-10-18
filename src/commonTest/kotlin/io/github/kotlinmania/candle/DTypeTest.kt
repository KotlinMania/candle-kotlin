// port-lint: tests candle-core/src/dtype.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DTypeTest {
    @Test
    fun testDTypeProperties() {
        assertEquals("f32", DType.F32.asStr())
        assertEquals(4, DType.F32.sizeInBytes)
        assertTrue(DType.F32.isFloat())

        assertEquals("u32", DType.U32.asStr())
        assertEquals(4, DType.U32.sizeInBytes)
        assertTrue(DType.U32.isInt())

        assertEquals(DType.F16, DType.fromString("f16"))
        assertEquals(DType.BF16, DType.fromString("bf16"))
    }

    @Test
    fun testHalfConversions() {
        val f = 1.0f
        val f16Bits = Float16Utils.floatToF16Bits(f)
        val fBack = Float16Utils.f16BitsToFloat(f16Bits)
        assertEquals(1.0f, fBack)

        val bf16Bits = Float16Utils.floatToBF16Bits(f)
        val bfBack = Float16Utils.bf16BitsToFloat(bf16Bits)
        assertEquals(1.0f, bfBack)
    }
}
