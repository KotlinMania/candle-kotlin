// port-lint: tests candle-core/tests/layout_tests.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LayoutTest {
    @Test
    fun testLayoutOperations() {
        val shape = Shape(listOf(2, 3, 4))
        val layout = Layout.contiguous(shape)
        assertEquals(listOf(12, 4, 1), layout.stride)
        assertEquals(0, layout.startOffset)
        assertTrue(layout.isContiguous())

        val transposed = layout.transpose(0, 1)
        assertEquals(listOf(3, 2, 4), transposed.shape.dims)
        assertEquals(listOf(4, 12, 1), transposed.stride)

        val narrowed = layout.narrow(1, 1, 2)
        assertEquals(listOf(2, 2, 4), narrowed.shape.dims)
        assertEquals(4, narrowed.startOffset)
    }

    @Test
    fun testStridedBlocks() {
        val shape = Shape(listOf(2, 3, 4))
        val layout = Layout.contiguous(shape)
        val blocks = layout.stridedBlocks()
        assertTrue(blocks is StridedBlocks.SingleBlock)
        assertEquals(0, blocks.startOffset)
        assertEquals(24, blocks.len)
    }
}
