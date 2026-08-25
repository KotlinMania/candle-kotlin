// port-lint: tests candle-core/src/shape.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ShapeTest {
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
