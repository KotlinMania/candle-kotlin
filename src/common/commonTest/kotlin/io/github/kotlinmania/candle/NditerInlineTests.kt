// port-lint: tests nditer.rs
package io.github.kotlinmania.candle.nditer

import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape

public fun layout(dims: List<ULong>, strides: List<ULong>): Layout = Layout.new(Shape.from(dims.toVec()), strides.toVec(), 0)

@kotlin.test.Test
public fun rank0Scalar() {
    val l = Layout.contiguous()
    val it = NdIter.new(listOf(l, l))
    check(it.innerSize == 1)
    check(it.innerStrides == listOf(0, 0))
    check(it.len() == 1)
    check(it.next() == listOf(0, 0))
    check(it.next() == null)
}

@kotlin.test.Test
public fun rank1ContiguousSingleBlock() {
    val l = Layout.contiguous(listOf(5))
    val it = NdIter.new(listOf(l, l))
    check(it.innerSize == 5)
    check(it.innerStrides == listOf(1, 1))
    check(it.len() == 1)
    check(it.next() == listOf(0, 0))
    check(it.next() == null)
}

@kotlin.test.Test
public fun rank2ContiguousMergesToOneBlock() {
    // [3, 4] strides[4, 1] -> fully merged, innerSize=12
    val l = Layout.contiguous(listOf(3, 4))
    val it = NdIter.new(listOf(l, l))
    check(it.innerSize == 12)
    check(it.innerStrides == listOf(1, 1))
    check(it.len() == 1)
}

@kotlin.test.Test
public fun rank3ContiguousFullyMerged() {
    val l = Layout.contiguous(listOf(2, 3, 4))
    val it = NdIter.new(listOf(l))
    check(it.innerSize == 24)
    check(it.innerStrides == listOf(1))
    check(it.len() == 1)
}

@kotlin.test.Test
public fun rank3OuterGapPartialMerge() {
    // shape [2, 3, 4]
    // strides [24, 4, 1]
    // dims 1 + 2 merge (4 == 1 * 4), dim 0 stays
    val l = layout(listOf(2, 3, 4), listOf(24, 4, 1))
    val it = NdIter.new(listOf(l))
    check(it.innerSize == 12)
    // 3 * 4 merged
    check(it.innerStrides == listOf(1))
    check(it.len() == 2)
    // outer: dim 0 has size 2
    val offsets: Vec = it.collect()
    check(offsets == mutableListOf(listOf(0), listOf(24)))
}

@kotlin.test.Test
public fun rank2NoMerge() {
    // shape [3, 4]
    // strides [1, 3]
    // stride[0] = 1 != stride[1] * dim[1] = 12 -> no merge
    val l = layout(listOf(3, 4), listOf(1, 3))
    val it = NdIter.new(listOf(l, l))
    check(it.innerSize == 4)
    check(it.innerStrides == listOf(3, 3))
    check(it.len() == 3)
    val offsets: Vec = it.collect()
    check(offsets == mutableListOf(listOf(0, 0), listOf(1, 1), listOf(2, 2)))
}

@kotlin.test.Test
public fun broadcastZerosMerge() {
    // shape [3, 4]
    // strides [0, 0]
    // 0 == 0 * 4 -> dims merge
    val l = layout(listOf(3, 4), listOf(0, 0))
    val it = NdIter.new(listOf(l, l))
    check(it.innerSize == 12)
    check(it.innerStrides == listOf(0, 0))
    check(it.len() == 1)
}

@kotlin.test.Test
public fun mixedContiguousAndBroadcastMerge() {
    // lhs [4, 1] contiguous
    // rhs [0,0] broadcast
    // both conditions pass -> merge
    val lhs = Layout.contiguous(listOf(3, 4))
    val rhs = layout(listOf(3, 4), listOf(0, 0))
    val it = NdIter.new(listOf(lhs, rhs))
    check(it.innerSize == 12)
    check(it.innerStrides == listOf(1, 0))
    check(it.len() == 1)
}

@kotlin.test.Test
public fun offsetsLhsContiguousRhsStrided() {
    // lhs row-major [2,3] strides [3,1]
    // rhs col-major [2,3] strides [1,2]
    // lhs can merge (3 == 1 * 3)
    // rhs can not (1 ≠ 2 * 3 = 6) -> no merge for the pair
    val lhs = Layout.contiguous(listOf(2, 3))
    val rhs = layout(listOf(2, 3), listOf(1, 2))
    val it = NdIter.new(listOf(lhs, rhs))
    check(it.innerSize == 3)
    check(it.innerStrides == listOf(1, 2))
    check(it.len() == 2)
    val offsets: Vec = it.collect()
    check(offsets == mutableListOf(listOf(0, 0), listOf(3, 1)))
}

@kotlin.test.Test
public fun startOffsetReflectedInFirstIter() {
    val l = Layout.contiguousWithOffset(4, 7)
    val it = NdIter.new(listOf(l))
    check(it.next() == listOf(7))
    check(it.next() == null)
}

@kotlin.test.Test
public fun startOffsetAdvancesWithOuterDims() {
    // shape [2, 3]
    // strides [4, 1]
    // startOffset=10
    // strides can't merge (4 != 1 * 3)
    // outer blocks at 10 and 14 (10 + 1 * outerStride)
    val l = Layout.new(Shape.from(mutableListOf(2, 3)), mutableListOf(4, 1), 10)
    val offsets: Vec = NdIter.new(listOf(l)).collect()
    check(offsets == mutableListOf(listOf(10), listOf(14)))
}

@kotlin.test.Test
public fun nthJumpsToOuterBlock() {
    val lhs = Layout.new(Shape.from(mutableListOf(2, 3, 4)), mutableListOf(40, 10, 1), 5)
    val rhs = Layout.new(Shape.from(mutableListOf(2, 3, 4)), mutableListOf(50, 12, 1), 7)
    val expected: Vec = NdIter.new(listOf(lhs, rhs)).collect()
    for ((n, offsets) in expected.iter().enumerate()) {
        check(NdIter.new(listOf(lhs, rhs)).nth(n) == offsets)
    }
    check(NdIter.new(listOf(lhs, rhs)).nth(expected.len()) == null)
    val iter = NdIter.new(listOf(lhs, rhs))
    check(iter.next() == expected[0])
    check(iter.nth(2) == expected[3])
    check(iter.next() == expected[4])
    check(iter.len() == expected.len() - 5)
}
