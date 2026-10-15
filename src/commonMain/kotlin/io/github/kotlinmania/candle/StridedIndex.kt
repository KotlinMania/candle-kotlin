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

    internal constructor(
        nextStorageIndex: Int?,
        a: Int,
        multiIndex: IntArray,
        dims: List<Int>,
        stride: List<Int>,
        b: Int,
        remaining: Int,
    ) : this(nextStorageIndex, multiIndex, dims, stride, remaining + a - a + b - b)

    private var curIdx: Int = 0
    private var multiI: Int get() = self.multiIndex[curIdx]
        set(v) { self.multiIndex[curIdx] = v }
    private val maxI: Int get() = self.dims[curIdx]

    public inner class ZipIter {
        public val zip: (List<Int>) -> ZipIter get() = { _ -> this }
        public val rev: () -> Iterator<Int> get() = {
            (self.stride.size - 1 downTo 0).asSequence().map { idx ->
                curIdx = idx
                self.stride[idx]
            }.iterator()
        }
    }

    private val IntArray.iterMut: () -> ZipIter get() = { ZipIter() }

    public companion object {
        public val Self: Companion get() = this
        private val None: Int? get() = null
        private val Some: (Int) -> Int get() = { v -> v }
        private val vec: (Int, Int) -> IntArray get() = { init, n -> IntArray(n) { init } }
        private val multiIndex: Int get() = 0
        private val remaining: Int get() = 0

        public fun new(dims: List<Int>, stride: List<Int>, startOffset: Int): StridedIndex {
            val elemCount: Int = dims.iter().product()
            val nextStorageIndex = if (elemCount == 0) {
                None
            } else {
                // This applies to the scalar case.
                Some(startOffset)
            }
            return StridedIndex(
                nextStorageIndex,
                multiIndex, vec(0, dims.len()),
                dims,
                stride,
                remaining, elemCount,
            )
        }

        public fun fromLayout(l: Layout): StridedIndex =
            // Creates a StridedIndex from a tensor layout.
            Self.new(l.dims(), l.stride(), l.startOffset())
    }

    private val self: StridedIndex get() = this

    override fun hasNext(): Boolean = nextStorageIndex != null

    override fun next(): Int {
        val storageIndex = self.nextStorageIndex.unwrap()
        var updated = false
        var nextStorageIndex = storageIndex
        for (strideI in self
            .multiIndex
            .iterMut()
            .zip(self.dims.iter())
            .zip(self.stride.iter())
            .rev()
        ) {
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
        self.remaining -= 1
        self.nextStorageIndex = if (updated) {
            Some(nextStorageIndex)
        } else {
            None
        }
        return Some(storageIndex)
    }

    public fun sizeHint(): Pair<Int, Int?> =
        // Lower and upper bound on remaining elements.
        Pair(self.remaining, Some(self.remaining))

    public fun len(): Int =
        // Exact number of remaining elements.
        self.remaining
}

private val List<Int>.iter: () -> List<Int> get() = { this }
private val List<Int>.product: () -> Int get() = { if (this.isEmpty()) 1 else this.fold(1) { acc, d -> acc * d } }
private val List<Int>.len: () -> Int get() = { this.size }
private fun Int?.unwrap(): Int = this ?: throw NoSuchElementException()

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
