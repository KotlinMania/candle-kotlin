// port-lint: source strided_index.rs
package io.github.kotlinmania.candle

/**
 * An iterator over offset position for items of an N-dimensional arrays stored in a
 * flat buffer using some potential strides.
 */
public class StridedIndex(
    public val dims: List<Int>,
    public val stride: List<Int>,
    public val startOffset: Int,
) : Iterator<Int> {
    private var nextStorageIndex: Int?
    private val multiIndex: IntArray = IntArray(dims.size)
    private var remaining: Int

    init {
        val elemCount = if (dims.isEmpty()) 1 else dims.fold(1) { acc, d -> acc * d }
        remaining = elemCount
        nextStorageIndex = if (elemCount == 0) {
            null
        } else {
            // This applies to the scalar case.
            startOffset
        }
    }

    public fun len(): Int = remaining

    public fun sizeHint(): Pair<Int, Int?> = Pair(remaining, remaining)

    override fun hasNext(): Boolean = nextStorageIndex != null

    override fun next(): Int {
        val storageIndex = nextStorageIndex ?: throw NoSuchElementException()
        var updated = false
        var nextIdx = storageIndex
        for (i in dims.indices.reversed()) {
            val nextI = multiIndex[i] + 1
            if (nextI < dims[i]) {
                multiIndex[i] = nextI
                updated = true
                nextIdx += stride[i]
                break
            } else {
                nextIdx -= multiIndex[i] * stride[i]
                multiIndex[i] = 0
            }
        }
        remaining--
        nextStorageIndex = if (updated) nextIdx else null
        return storageIndex
    }

    public companion object {
        public fun new(dims: List<Int>, stride: List<Int>, startOffset: Int): StridedIndex =
            StridedIndex(dims, stride, startOffset)

        public fun fromLayout(l: Layout): StridedIndex =
            new(l.dims(), l.stride(), l.startOffset())
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
