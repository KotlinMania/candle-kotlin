// port-lint: source candle-core/src/strided_index.rs
package io.github.kotlinmania.candle

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
        nextStorageIndex = if (elemCount == 0) null else startOffset
    }

    public fun len(): Int = remaining

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
        public fun fromLayout(layout: Layout): StridedIndex =
            StridedIndex(layout.dims, layout.stride, layout.startOffset)
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
