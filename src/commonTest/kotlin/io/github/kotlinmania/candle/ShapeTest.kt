// port-lint: tests shape.rs
package io.github.kotlinmania.candle

import kotlin.test.Test

class ShapeTest {
    @Test internal fun stride() {
        var shape = Shape.from(Unit)
        assertEq(shape.strideContiguous(), vec<Int>())
        shape = Shape.from(42)
        assertEq(shape.strideContiguous(), vec(1))
        shape = Shape.from(42, 1337)
        assertEq(shape.strideContiguous(), vec(1337, 1))
        shape = Shape.from(299, 792, 458)
        assertEq(shape.strideContiguous(), vec(458 * 792, 458, 1))
    }

    @Test internal fun testFromTuple() {
        var shape = Shape.from(2)
        assertEq(shape.dims(), vec(2))
        shape = Shape.from(2, 3)
        assertEq(shape.dims(), vec(2, 3))
        shape = Shape.from(2, 3, 4)
        assertEq(shape.dims(), vec(2, 3, 4))
        shape = Shape.from(2, 3, 4, 5)
        assertEq(shape.dims(), vec(2, 3, 4, 5))
        shape = Shape.from(2, 3, 4, 5, 6)
        assertEq(shape.dims(), vec(2, 3, 4, 5, 6))
        shape = Shape.from(2, 3, 4, 5, 6, 7)
        assertEq(shape.dims(), vec(2, 3, 4, 5, 6, 7))
    }
}
