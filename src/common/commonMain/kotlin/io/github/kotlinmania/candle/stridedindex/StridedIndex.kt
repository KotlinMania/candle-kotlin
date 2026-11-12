// port-lint: source strided_index.rs
package io.github.kotlinmania.candle.stridedindex

import io.github.kotlinmania.candle.layout.Layout

public typealias Item = Int

/**
 * An iterator over offset position for items of an N-dimensional arrays stored in a
 * flat buffer using some potential strides.
 */
public open class StridedIndex(
    public var nextStorageIndex: Int?,
    public var multiIndex: List<Int>,
    public var dims: List<Int>,
    public var stride: List<Int>,
    public var remaining: Int,
) : Iterable<Int> {
    override fun iterator(): kotlin.collections.Iterator<Int> = object : AbstractIterator<Int>() {
        override fun computeNext() {
            when (val v = this@StridedIndex.next()) {
                null -> done()
                else -> setNext(v)
            }
        }
    }

    public open fun next(): Int? {
        val storageIndex = (this.nextStorageIndex ?: return null)
        var updated = false
        var nextStorageIndex = storageIndex
        for ((multiI, maxI, strideI) in ((this.multiIndex.zip(this.dims)).zip(this.stride)).rev()) {
            val nextI = multiI + 1
            if (nextI < maxI) {
                multiI = nextI
                updated = true
                nextStorageIndex += strideI
                break
            } else {
                nextStorageIndex -= multiI * strideI
                multiI = 0
            }
        }
        this.remaining -= 1
        this.nextStorageIndex = if (updated) {
            nextStorageIndex
        } else {
            null
        }
        return storageIndex
    }

    public open fun sizeHint(): Pair<Int, Int?> = Pair(this.remaining, this.remaining)

    public open fun len(): Int = this.remaining

    public companion object {
        internal fun new(dims: List<Int>, stride: List<Int>, startOffset: Int): StridedIndex {
            val elemCount: Int = dims.iter().product()
            val nextStorageIndex =
                if (elemCount == 0) {
                    null
                } else {
                    // This applies to the scalar case.
                    startOffset
                }
            return StridedIndex(nextStorageIndex = nextStorageIndex, multiIndex = mutableListOf(0, dims.size), dims = dims, stride = stride, remaining = elemCount)
        }

        internal fun fromLayout(l: Layout): StridedIndex = StridedIndex.new(l.dims(), l.stride(), l.startOffset())
    }
}

public sealed class StridedBlocks {
    public data class SingleBlock(public var startOffset: Int, public var len: Int) : StridedBlocks()
    public data class UniformBlocks(public var startOffset: Int, public var blockLen: Int, public var count: Int, public var srcStride: Int) : StridedBlocks()
    public data class MultipleBlocks(public var blockStartIndex: StridedIndex, public var blockLen: Int) : StridedBlocks()
}
