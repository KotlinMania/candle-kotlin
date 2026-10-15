// port-lint: tests shape.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ShapeTest {
    @Test
    fun stride() {
        val shape = Shape.SCALAR
        assertEquals(emptyList(), shape.strideContiguous())
        val s1 = Shape.from(42)
        assertEquals(listOf(1), s1.strideContiguous())
        val s2 = Shape.from(42, 1337)
        assertEquals(listOf(1337, 1), s2.strideContiguous())
        val s3 = Shape.from(299, 792, 458)
        assertEquals(listOf(458 * 792, 458, 1), s3.strideContiguous())
    }

    @Test
    fun testFromTuple() {
        val s1 = Shape.from(2)
        assertEquals(listOf(2), s1.dims())
        val s2 = Shape.from(2, 3)
        assertEquals(listOf(2, 3), s2.dims())
        val s3 = Shape.from(2, 3, 4)
        assertEquals(listOf(2, 3, 4), s3.dims())
        val s4 = Shape.from(2, 3, 4, 5)
        assertEquals(listOf(2, 3, 4, 5), s4.dims())
        val s5 = Shape.from(2, 3, 4, 5, 6)
        assertEquals(listOf(2, 3, 4, 5, 6), s5.dims())
        val s6 = Shape.from(2, 3, 4, 5, 6, 7)
        assertEquals(listOf(2, 3, 4, 5, 6, 7), s6.dims())
    }

    @Test
    fun testStride() {
        val shape = Shape(listOf(2, 3, 5, 7, 11))
        assertEquals(listOf(1155, 385, 77, 11, 1), shape.strideContiguous())

        val scalar = Shape.SCALAR
        assertEquals(emptyList(), scalar.strideContiguous())

        val dim1 = Shape(listOf(42))
        assertEquals(listOf(1), dim1.strideContiguous())
    }

    @Test
    fun testDims() {
        val shape = Shape(listOf(2, 3, 4))
        assertEquals(3, shape.rank)
        assertEquals(24, shape.elemCount())
        assertEquals(listOf(2, 3, 4), shape.dims)
        assertEquals(2, shape.dim(0))
        assertEquals(3, shape.dim(1))
        assertEquals(4, shape.dim(2))
        assertEquals(4, shape.dim(-1))
        assertEquals(3, shape.dim(-2))
        assertEquals(2, shape.dim(-3))
    }

    @Test
    fun testContiguous() {
        val shape = Shape(listOf(2, 3, 4))
        assertTrue(shape.isContiguous(listOf(12, 4, 1)))
        assertFalse(shape.isContiguous(listOf(1, 2, 6)))

        assertTrue(shape.isFortranContiguous(listOf(1, 2, 6)))
        assertFalse(shape.isFortranContiguous(listOf(12, 4, 1)))
    }

    @Test
    fun testBroadcast() {
        val s1 = Shape(listOf(2, 1, 4))
        val s2 = Shape(listOf(3, 1))
        val bcast = s1.broadcastShapeBinaryOp(s2, "test")
        assertEquals(listOf(2, 3, 4), bcast.dims)
    }

    @Test
    fun testHoleResolution() {
        val shape = listOf(2, -1, 4).intoShapeWithHole(24)
        assertEquals(listOf(2, 3, 4), shape.dims)
    }
}
