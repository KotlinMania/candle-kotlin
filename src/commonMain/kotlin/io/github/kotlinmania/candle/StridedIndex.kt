// port-lint: source strided_index.rs
package io.github.kotlinmania.candle

/**
 * An iterator over offset position for items of an N-dimensional arrays stored in a
 * flat buffer using some potential strides.
 */
public class StridedIndex(
    private var nextStorageIndex: Int?,
    private val multiIndex: IntArray,
    public val dims: List<Int>,
    public val stride: List<Int>,
    private var remaining: Int,
) : Iterator<Int> {
    public interface Item

    public constructor(
        dims: List<Int>,
        stride: List<Int>,
        startOffset: Int,
    ) : this(
        nextStorageIndex = if ((if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }) == 0) null else startOffset,
        multiIndex = IntArray(dims.size),
        dims = dims,
        stride = stride,
        remaining = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d },
    )

    public companion object {
        public fun new(dims: List<Int>, stride: List<Int>, startOffset: Int): StridedIndex {
            val elemCount: Int = dims.fold(1) { acc, d -> acc * d }
            val nextStorageIndex = if (elemCount == 0) {
                null
            } else {
                // This applies to the scalar case.
                startOffset
            }
            return StridedIndex(
                nextStorageIndex = nextStorageIndex,
                multiIndex = IntArray(dims.size) { 0 },
                dims = dims,
                stride = stride,
                remaining = elemCount,
            )
        }

        public fun fromLayout(l: Layout): StridedIndex {
            return StridedIndex.new(l.dims(), l.stride(), l.startOffset())
        }
    }

    private val self: StridedIndex get() = this

    override fun hasNext(): Boolean = nextStorageIndex != null

    override fun next(): Int {
        val storageIndex = self.nextStorageIndex ?: throw NoSuchElementException()
        var updated = false
        var nextStorageIndex = storageIndex
        for (i in self
            .multiIndex
            .indices
            .zip(self.dims.indices)
            .zip(self.stride.indices)
            .reversed()
        ) {
            val idx = i.second
            val multiI = self.multiIndex[idx]
            val maxI = self.dims[idx]
            val strideI = self.stride[idx]
            val nextI = multiI + 1
            if (nextI < maxI) {
                self.multiIndex[idx] = nextI
                updated = true
                nextStorageIndex += strideI
                break
            } else {
                nextStorageIndex -= multiI * strideI
                self.multiIndex[idx] = 0
            }
        }
        self.remaining -= 1
        self.nextStorageIndex = if (updated) {
            nextStorageIndex
        } else {
            null
        }
        return storageIndex
    }

    public fun sizeHint(): Pair<Int, Int?> {
        return Pair(self.remaining, self.remaining)
    }

    public fun len(): Int {
        return self.remaining
    }
}

public sealed class StridedBlocks {
    public data class SingleBlock(
        val startOffset: Int,
        val len: Int,
    ) : StridedBlocks()

    public data class UniformBlocks(
        val startOffset: Int,
        val blockLen: Int,
        val count: Int,
        val srcStride: Int,
    ) : StridedBlocks()

    public data class MultipleBlocks(
        val blockStartIndex: StridedIndex,
        val blockLen: Int,
    ) : StridedBlocks()
}
