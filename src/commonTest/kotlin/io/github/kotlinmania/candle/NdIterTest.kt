// port-lint: tests candle-core/src/nditer.rs
package io.github.kotlinmania.candle

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class NdIterTest {
    private val assertEq: (Any?, Any?) -> Unit = { actual, expected ->
        if (actual is IntArray && expected is IntArray) {
            assertContentEquals(expected, actual)
        } else {
            assertEquals(expected, actual)
        }
    }
    private val nextItem: (NdIter) -> IntArray? = { it ->
        if (it.hasNext()) it.next() else null
    }
    private val collectVec: (NdIter) -> List<List<Int>> = { it ->
        it.asSequence().map { arr -> arr.toList() }.toList()
    }

    @Test fun layout(dims: List<Int> = listOf(1), strides: List<Int> = listOf(1)): Layout {
        return Layout.new(Shape.from(dims.toList()), strides.toList(), 0)
    }

    @Test fun rank0Scalar() {
        val l = Layout.contiguous(Shape())
        val it = NdIter.new(listOf(l, l))
        assertEq(it.innerSize, 1)
        assertEq(it.innerStrides, intArrayOf(0, 0))
        assertEq(it.len, 1)
        assertEq(nextItem(it), intArrayOf(0, 0))
        assertEq(nextItem(it), null)
    }

    @Test fun rank1ContiguousSingleBlock() {
        val l = Layout.contiguous(Shape(5))
        val it = NdIter.new(listOf(l, l))
        assertEq(it.innerSize, 5)
        assertEq(it.innerStrides, intArrayOf(1, 1))
        assertEq(it.len, 1)
        assertEq(nextItem(it), intArrayOf(0, 0))
        assertEq(nextItem(it), null)
    }

    @Test fun rank2ContiguousMergesToOneBlock() {
        // [3, 4] strides[4, 1] -> fully merged, innerSize=12
        val l = Layout.contiguous(Shape(3, 4))
        val it = NdIter.new(listOf(l, l))
        assertEq(it.innerSize, 12)
        assertEq(it.innerStrides, intArrayOf(1, 1))
        assertEq(it.len, 1)
    }

    @Test fun rank3ContiguousFullyMerged() {
        val l = Layout.contiguous(Shape(2, 3, 4))
        val it = NdIter.new(listOf(l))
        assertEq(it.innerSize, 24)
        assertEq(it.innerStrides, intArrayOf(1))
        assertEq(it.len, 1)
    }

    @Test fun rank3OuterGapPartialMerge() {
        // shape [2, 3, 4]
        // strides [24, 4, 1]
        // dims 1 + 2 merge (4 == 1 * 4), dim 0 stays
        val l = layout(listOf(2, 3, 4), listOf(24, 4, 1))
        val it = NdIter.new(listOf(l))
        assertEq(it.innerSize, 12) // 3 * 4 merged
        assertEq(it.innerStrides, intArrayOf(1))
        assertEq(it.len, 2) // outer: dim 0 has size 2
        val offsets = collectVec(it)
        assertEq(offsets, listOf(listOf(0), listOf(24)))
    }

    @Test fun rank2NoMerge() {
        // shape [3, 4]
        // strides [1, 3]
        // stride[0] = 1 != stride[1] * dim[1] = 12 -> no merge
        val l = layout(listOf(3, 4), listOf(1, 3))
        val it = NdIter.new(listOf(l, l))
        assertEq(it.innerSize, 4)
        assertEq(it.innerStrides, intArrayOf(3, 3))
        assertEq(it.len, 3)
        val offsets = collectVec(it)
        assertEq(offsets, listOf(listOf(0, 0), listOf(1, 1), listOf(2, 2)))
    }

    @Test fun broadcastZerosMerge() {
        // shape [3, 4]
        // strides [0, 0]
        // 0 == 0 * 4 -> dims merge
        val l = layout(listOf(3, 4), listOf(0, 0))
        val it = NdIter.new(listOf(l, l))
        assertEq(it.innerSize, 12)
        assertEq(it.innerStrides, intArrayOf(0, 0))
        assertEq(it.len, 1)
    }

    @Test fun mixedContiguousAndBroadcastMerge() {
        // lhs [4, 1] contiguous
        // rhs [0,0] broadcast
        // both conditions pass -> merge
        val lhs = Layout.contiguous(Shape(3, 4))
        val rhs = layout(listOf(3, 4), listOf(0, 0))
        val it = NdIter.new(listOf(lhs, rhs))
        assertEq(it.innerSize, 12)
        assertEq(it.innerStrides, intArrayOf(1, 0))
        assertEq(it.len, 1)
    }

    @Test fun offsetsLhsContiguousRhsStrided() {
        // lhs row-major [2,3] strides [3,1]
        // rhs col-major [2,3] strides [1,2]
        // lhs can merge (3 == 1 * 3)
        // rhs can not (1 != 2 * 3 = 6) -> no merge for the pair
        val lhs = Layout.contiguous(Shape(2, 3))
        val rhs = layout(listOf(2, 3), listOf(1, 2))
        val it = NdIter.new(listOf(lhs, rhs))
        assertEq(it.innerSize, 3)
        assertEq(it.innerStrides, intArrayOf(1, 2))
        assertEq(it.len, 2)
        val offsets = collectVec(it)
        assertEq(offsets, listOf(listOf(0, 0), listOf(3, 1)))
    }

    @Test fun startOffsetReflectedInFirstIter() {
        val l = Layout.contiguousWithOffset(Shape(4), 7)
        val it = NdIter.new(listOf(l))
        assertEq(nextItem(it), intArrayOf(7))
        assertEq(nextItem(it), null)
    }

    @Test fun startOffsetAdvancesWithOuterDims() {
        // shape [2, 3]
        // strides [4, 1]
        // startOffset=10
        // strides can't merge (4 != 1 * 3)
        // outer blocks at 10 and 14 (10 + 1 * outerStride)
        val l = Layout.new(Shape.from(listOf(2, 3)), listOf(4, 1), 10)
        val offsets = collectVec(NdIter.new(listOf(l)))
        assertEq(offsets, listOf(listOf(10), listOf(14)))
    }
}
