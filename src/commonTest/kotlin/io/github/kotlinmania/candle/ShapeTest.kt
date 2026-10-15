// port-lint: tests shape.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertEquals

class ShapeTest {
    @Test internal fun stride() {
        var shape = Shape.from(Unit)
        assertEquals(shape.strideContiguous(), emptyList<Int>())
        shape = Shape.from(42)
        assertEquals(shape.strideContiguous(), listOf(1))
        shape = Shape.from(42, 1337)
        assertEquals(shape.strideContiguous(), listOf(1337, 1))
        shape = Shape.from(299, 792, 458)
        assertEquals(shape.strideContiguous(), listOf(458 * 792, 458, 1))
    }

    @Test internal fun testFromTuple() {
        var shape = Shape.from(2)
        assertEquals(shape.dims(), listOf(2))
        shape = Shape.from(2, 3)
        assertEquals(shape.dims(), listOf(2, 3))
        shape = Shape.from(2, 3, 4)
        assertEquals(shape.dims(), listOf(2, 3, 4))
        shape = Shape.from(2, 3, 4, 5)
        assertEquals(shape.dims(), listOf(2, 3, 4, 5))
        shape = Shape.from(2, 3, 4, 5, 6)
        assertEquals(shape.dims(), listOf(2, 3, 4, 5, 6))
        shape = Shape.from(2, 3, 4, 5, 6, 7)
        assertEquals(shape.dims(), listOf(2, 3, 4, 5, 6, 7))
    }
}
