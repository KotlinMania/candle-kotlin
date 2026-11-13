// port-lint: tests shape.rs
package io.github.kotlinmania.candle.shape

// The shape of a tensor is a tuple with the size of each of its dimensions.

@kotlin.test.Test
public fun stride() {
    val shape = Shape.from()
    check(shape.strideContiguous() == mutableListOf())
    val shape1 = Shape.from(42)
    check(shape1.strideContiguous() == listOf(1))
    val shape2 = Shape.from(Pair(42, 1337))
    check(shape2.strideContiguous() == listOf(1337, 1))
    val shape3 = Shape.from(Triple(299, 792, 458))
    check(shape3.strideContiguous() == listOf(458 * 792, 458, 1))
}

@kotlin.test.Test
public fun testFromTuple() {
    val shape = Shape.from(2)
    check(shape.dims() == listOf(2))
    val shape1 = Shape.from(Pair(2, 3))
    check(shape1.dims() == listOf(2, 3))
    val shape2 = Shape.from(Triple(2, 3, 4))
    check(shape2.dims() == listOf(2, 3, 4))
    val shape3 = Shape.from(listOf(2, 3, 4, 5))
    check(shape3.dims() == listOf(2, 3, 4, 5))
    val shape4 = Shape.from(listOf(2, 3, 4, 5, 6))
    check(shape4.dims() == listOf(2, 3, 4, 5, 6))
    val shape5 = Shape.from(listOf(2, 3, 4, 5, 6, 7))
    check(shape5.dims() == listOf(2, 3, 4, 5, 6, 7))
}
