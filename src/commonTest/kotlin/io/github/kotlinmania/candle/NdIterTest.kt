// port-lint: tests candle-core/src/nditer.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NdIterTest {
    private fun layout(dims: List<Int>, strides: List<Int>): Layout =
        Layout(Shape(dims), strides, 0)

    @Test
    fun rank0Scalar() {
        val l = Layout.contiguous(Shape())
        val it = NdIter(l, l)
        assertEquals(1, it.innerSize)
        assertContentEquals(intArrayOf(0, 0), it.innerStrides)
        assertEquals(1, it.len)
        assertTrue(it.hasNext())
        assertContentEquals(intArrayOf(0, 0), it.next())
        assertFalse(it.hasNext())
    }

    @Test
    fun rank1ContiguousSingleBlock() {
        val l = Layout.contiguous(Shape(5))
        val it = NdIter(l, l)
        assertEquals(5, it.innerSize)
        assertContentEquals(intArrayOf(1, 1), it.innerStrides)
        assertEquals(1, it.len)
        assertTrue(it.hasNext())
        assertContentEquals(intArrayOf(0, 0), it.next())
        assertFalse(it.hasNext())
    }

    @Test
    fun rank2ContiguousMergesToOneBlock() {
        val l = Layout.contiguous(Shape(3, 4))
        val it = NdIter(l, l)
        assertEquals(12, it.innerSize)
        assertContentEquals(intArrayOf(1, 1), it.innerStrides)
        assertEquals(1, it.len)
    }

    @Test
    fun rank3ContiguousFullyMerged() {
        val l = Layout.contiguous(Shape(2, 3, 4))
        val it = NdIter(l)
        assertEquals(24, it.innerSize)
        assertContentEquals(intArrayOf(1), it.innerStrides)
        assertEquals(1, it.len)
    }

    @Test
    fun rank3OuterGapPartialMerge() {
        val l = layout(listOf(2, 3, 4), listOf(24, 4, 1))
        val it = NdIter(l)
        assertEquals(12, it.innerSize)
        assertContentEquals(intArrayOf(1), it.innerStrides)
        assertEquals(2, it.len)
        val offsets = it.asSequence().map { it.toList() }.toList()
        assertEquals(listOf(listOf(0), listOf(24)), offsets)
    }

    @Test
    fun rank2NoMerge() {
        val l = layout(listOf(3, 4), listOf(1, 3))
        val it = NdIter(l, l)
        assertEquals(4, it.innerSize)
        assertContentEquals(intArrayOf(3, 3), it.innerStrides)
        assertEquals(3, it.len)
        val offsets = it.asSequence().map { it.toList() }.toList()
        assertEquals(listOf(listOf(0, 0), listOf(1, 1), listOf(2, 2)), offsets)
    }

    @Test
    fun broadcastZerosMerge() {
        val l = layout(listOf(3, 4), listOf(0, 0))
        val it = NdIter(l, l)
        assertEquals(12, it.innerSize)
        assertContentEquals(intArrayOf(0, 0), it.innerStrides)
        assertEquals(1, it.len)
    }

    @Test
    fun mixedContiguousAndBroadcastMerge() {
        val lhs = Layout.contiguous(Shape(3, 4))
        val rhs = layout(listOf(3, 4), listOf(0, 0))
        val it = NdIter(lhs, rhs)
        assertEquals(12, it.innerSize)
        assertContentEquals(intArrayOf(1, 0), it.innerStrides)
        assertEquals(1, it.len)
    }

    @Test
    fun offsetsLhsContiguousRhsStrided() {
        val lhs = Layout.contiguous(Shape(2, 3))
        val rhs = layout(listOf(2, 3), listOf(1, 2))
        val it = NdIter(lhs, rhs)
        assertEquals(3, it.innerSize)
        assertContentEquals(intArrayOf(1, 2), it.innerStrides)
        assertEquals(2, it.len)
        val offsets = it.asSequence().map { it.toList() }.toList()
        assertEquals(listOf(listOf(0, 0), listOf(3, 1)), offsets)
    }

    @Test
    fun startOffsetReflectedInFirstIter() {
        val l = Layout.contiguousWithOffset(Shape(4), 7)
        val it = NdIter(l)
        assertTrue(it.hasNext())
        assertContentEquals(intArrayOf(7), it.next())
        assertFalse(it.hasNext())
    }

    @Test
    fun startOffsetAdvancesWithOuterDims() {
        val l = Layout(Shape(2, 3), listOf(4, 1), 10)
        val offsets = NdIter(l).asSequence().map { it.toList() }.toList()
        assertEquals(listOf(listOf(10), listOf(14)), offsets)
    }
}
