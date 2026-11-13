// port-lint: source nditer.rs
package io.github.kotlinmania.candle.nditer

import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout

/**
 * Maximum tensor rank supported by [NdIter].
 */
public val MAX_DIMS: Int = 8

/**
 * Multi-dimensional iterator that walks `N` tensor layouts simultaneously.
 *
 * Internally, adjacent dimensions are merged whenever their strides allow it,
 * reducing the number of outer iterations. Each call to [Iterator.next] yields
 * `[Int; N]`, one base offset per layout, for one inner slice of `innerSize`
 * elements. Callers iterate that slice using `innerStrides`:
 *
 * ```kotlin
 * val ndIter = NdIter.new([lhsL, rhsL]); val innerSize = ndIter.innerSize; val [innerLs, innerRs] = ndIter.innerStrides; for [lhsOff, rhsOff] in ndIter { for i in 0 .. innerSize { val lhs = lhsBuf[lhsOff + i * innerLs]; val rhs = rhsBuf[rhsOff + i * innerRs]; ... } }
 * ```
 *
 * Note: `innerStrides[n]` can be 1 contiguous, but also 0 (broadcast/scalar) or > 1 (non-contiguous),
 * so callers must not assume contiguous element access within a slice.
 */
public class NdIter(
    public var innerSize: Int = 0,
    /**
     * Per-layout element stride within each inner slice: `elementOffset = base + i * stride`.
     */
    public var innerStrides: IntArray = IntArray((N).toInt()),
    public var outerDims: IntArray = IntArray(MAX_DIMS),
    public var outerStrides: List<IntArray> = List((N).toInt()) { IntArray(0) },
    public var outerLen: Int = 0,
    public var startOffsets: IntArray = IntArray((N).toInt()),
    public var offsets: IntArray = IntArray((N).toInt()),
    public var coords: IntArray = IntArray(MAX_DIMS),
    public var remaining: Int = 0,
) : Iterable<IntArray> {
    override fun iterator(): kotlin.collections.Iterator<IntArray> =
        object : AbstractIterator<IntArray>() {
            override fun computeNext() {
                when (val v = this@NdIter.next()) {
                    null -> done()
                    else -> setNext(v)
                }
            }
        }

    public fun next(): IntArray? {
        if (this.remaining == 0) {
            return null
        }
        val item = this.offsets
        this.remaining -= 1
        for (k in (0 until this.outerLen).rev()) {
            this.coords[k] += 1
            for (n in 0 until N) {
                this.offsets[n] += this.outerStrides[n][k]
            }
            if (this.coords[k] < this.outerDims[k]) {
                break
            }
            this.coords[k] = 0
            for (n1 in 0 until N) {
                this.offsets[n1] -= this.outerDims[k] * this.outerStrides[n1][k]
            }
        }
        return item
    }

    public fun nth(n: Int): IntArray? {
        if (n >= this.remaining) {
            this.remaining = 0
            return null
        }
        val total = ((this.outerDims).slice(0 until this.outerLen)).iter().product<Int>()
        var position = total - this.remaining + n
        this.offsets = this.startOffsets
        this.coords = IntArray(MAX_DIMS) { 0 }
        for (k in (0 until this.outerLen).rev()) {
            val coord = position % this.outerDims[k]
            position /= this.outerDims[k]
            this.coords[k] = coord
            for (layout in 0 until N) {
                this.offsets[layout] += coord * this.outerStrides[layout][k]
            }
        }
        this.remaining -= n
        return this.next()
    }

    public fun sizeHint(): Pair<Int, Int?> = Pair(this.remaining, this.remaining)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<NdIter> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<NdIter> {
                override val alignBytes: Long = 8L
                private val off2: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((8L + ((N).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).strideBytes)), 8L)
                private val off3: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off2 + ((MAX_DIMS).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).strideBytes)), 8L)
                private val off4: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes(
                            (
                                off3 + (
                                    (N).toLong() *
                                        (
                                            io.github.kotlinmania.rustlang.core.mem
                                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt())
                                        ).strideBytes
                                )
                            ),
                            8L,
                        )
                private val off5: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off4 + 8L), 8L)
                private val off6: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off5 + ((N).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).strideBytes)), 8L)
                private val off7: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off6 + ((N).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).strideBytes)), 8L)
                private val off8: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off7 + ((MAX_DIMS).toLong() * (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).strideBytes)), 8L)
                override val sizeBytes: Long =
                    io.github.kotlinmania.rustlang.core.mem
                        .alignUpBytes((off8 + 8L), 8L)
                override fun readNative(addr: Long): NdIter =
                    NdIter(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                        ).readNative(addr + 8L).toIntArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt())
                        ).readNative(addr + off2).toIntArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(
                                    io.github.kotlinmania.rustlang.core.mem
                                        .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt()),
                                    (N).toInt(),
                                )
                        ).readNative(addr + off3).map { it.toIntArray() },
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + off4),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                        ).readNative(addr + off5).toIntArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                        ).readNative(addr + off6).toIntArray(),
                        (
                            io.github.kotlinmania.rustlang.core.mem
                                .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt())
                        ).readNative(addr + off7).toIntArray(),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + off8),
                    )
                override fun writeNative(addr: Long, value: NdIter) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.innerSize)
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                    ).writeNative(addr + 8L, value.innerStrides.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt())
                    ).writeNative(addr + off2, value.outerDims.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(
                                io.github.kotlinmania.rustlang.core.mem
                                    .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt()),
                                (N).toInt(),
                            )
                    ).writeNative(addr + off3, value.outerStrides.map { it.toList() })
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + off4, value.outerLen)
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                    ).writeNative(addr + off5, value.startOffsets.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (N).toInt())
                    ).writeNative(addr + off6, value.offsets.toList())
                    (
                        io.github.kotlinmania.rustlang.core.mem
                            .FixedArrayTypeLayout(io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64, (MAX_DIMS).toInt())
                    ).writeNative(addr + off7, value.coords.toList())
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + off8, value.remaining)
                }
            }

        public fun new(layouts: List<Layout>, n: Int): NdIter {
            val dims = layouts[0].dims()
            check(dims.len() <= MAX_DIMS) { "rank ${dims.len()} exceeds MAX_DIMS=$MAX_DIMS" }
            for (l in layouts) {
                check(l.dims() == dims)
            }
            val rank = dims.len()
            val outDims = List(MAX_DIMS) { 0 }
            val outStrides = List(n) { List(MAX_DIMS) { 0 } }
            var outLen: Any?
            if (rank == 0) {
                outDims[0] = 1
                outLen = 1
            } else {
                outDims[0] = dims[0]
                for (n in 0 until n) {
                    outStrides[n][0] = layouts[n].stride()[0]
                }
                outLen = 1
                for ((i, d) in dims
                    .iter()
                    .mapIndexed { index, item -> Pair(index, item) }
                    .take(rank)
                    .skip(1)) {
                    val top = outLen - 1
                    val lastD = outDims[top]
                    val (canMerge1, useInner) =
                        if (lastD == 1) {
                            Pair(true, true)
                        } else if (d == 1) {
                            Pair(true, false)
                        } else {
                            val canMerge = (0 until n).all({ n1 -> outStrides[n1][top] == layouts[n1].stride()[i] * d })
                            Pair(canMerge, true)
                        }
                    if (canMerge1) {
                        outDims[top] = lastD * d
                        if (useInner) {
                            for (n1 in 0 until n) {
                                outStrides[n1][top] = layouts[n1].stride()[i]
                            }
                        }
                    } else {
                        outDims[outLen] = d
                        for (n2 in 0 until n) {
                            outStrides[n2][outLen] = layouts[n2].stride()[i]
                        }
                        outLen += 1
                    }
                }
            }
            // Update inner strides
            val innerIdx = outLen - 1
            val innerSize = outDims[innerIdx]
            var innerStrides = List(n) { 0 }
            for (n3 in 0 until n) {
                innerStrides[n3] = outStrides[n3][innerIdx]
            }
            // Update outer dims
            val outerLen = innerIdx
            var outerDims = List(MAX_DIMS) { 0 }
            ((outerDims).slice(0 until outerLen)).copyFromSlice(((outDims).slice(0 until outerLen)))
            // Update outer strides
            var outerStrides = List(n) { List(MAX_DIMS) { 0 } }
            for (n4 in 0 until n) {
                ((outerStrides[n4]).slice(0 until outerLen)).copyFromSlice(((outStrides[n4]).slice(0 until outerLen)))
            }
            // Update offsets
            var offsets = List(n) { 0 }
            for (n5 in 0 until n) {
                offsets[n5] = layouts[n5].startOffset()
            }
            // Number of outer blocks (product of all dims except the innermost).
            val remaining = ((outDims).slice(0 until outerLen)).iter().product()
            return NdIter(innerSize = innerSize, innerStrides = innerStrides, outerDims = outerDims, outerStrides = outerStrides, outerLen = outerLen, startOffsets = offsets, offsets = offsets, coords = IntArray(MAX_DIMS) { 0 }, remaining = remaining)
        }
    }
}
