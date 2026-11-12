// port-lint: source nditer.rs
package io.github.kotlinmania.candle.nditer

import io.github.kotlinmania.candle.layout.Layout

public typealias Item = List<Int>

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
public open class NdIter(
    public var innerSize: Int,
    /**
     * Per-layout element stride within each inner slice: `elementOffset = base + i * stride`.
     */
    public var innerStrides: List<Int>,
    public var outerDims: List<Int>,
    public var outerStrides: List<List<Int>>,
    public var outerLen: Int,
    public var startOffsets: List<Int>,
    public var offsets: List<Int>,
    public var coords: List<Int>,
    public var remaining: Int,
) : Iterable<List<Int>> {
    override fun iterator(): kotlin.collections.Iterator<List<Int>> = object : AbstractIterator<List<Int>>() {
        override fun computeNext() {
            when (val v = this@NdIter.next()) {
                null -> done()
                else -> setNext(v)
            }
        }
    }

    public open fun next(): List<Int>? {
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

    public open fun nth(n: Int): List<Int>? {
        if (n >= this.remaining) {
            this.remaining = 0
            return null
        }
        val total = this.outerDims[0 until this.outerLen].iter().product<Int>()
        var position = total - this.remaining + n
        this.offsets = this.startOffsets
        this.coords = List(MAX_DIMS) { 0 }
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

    public open fun sizeHint(): Pair<Int, Int?> = Pair(this.remaining, this.remaining)

    public companion object {
        public fun new(layouts: List<Layout>, n: Int): NdIter {
            val dims = layouts[0].dims()
            check(dims.len() <= MAX_DIMS) { "rank ${dims.len()} exceeds MAX_DIMS=$MAX_DIMS" }
            for (l in layouts) {
                check(l.dims() == dims)
            }
            val rank = dims.len()
            val outDims = List(MAX_DIMS) { 0 }
            val outStrides = List((N).toInt()) { List(MAX_DIMS) { 0 } }
            var outLen: Any?
            if (rank == 0) {
                outDims[0] = 1
                outLen = 1
            } else {
                outDims[0] = dims[0]
                for (n in 0 until N) {
                    outStrides[n][0] = layouts[n].stride()[0]
                }
                outLen = 1
                for ((i, d) in dims.iter().enumerate().take(rank).skip(1)) {
                    val top = outLen - 1
                    val lastD = outDims[top]
                    val (canMerge1, useInner) =
                        if (lastD == 1) {
                            Pair(true, true)
                        } else if (d == 1) {
                            Pair(true, false)
                        } else {
                            val canMerge = (0 until N).all({ n1 -> outStrides[n1][top] == layouts[n1].stride()[i] * d })
                            Pair(canMerge, true)
                        }
                    if (canMerge1) {
                        outDims[top] = lastD * d
                        if (useInner) {
                            for (n1 in 0 until N) {
                                outStrides[n1][top] = layouts[n1].stride()[i]
                            }
                        }
                    } else {
                        outDims[outLen] = d
                        for (n2 in 0 until N) {
                            outStrides[n2][outLen] = layouts[n2].stride()[i]
                        }
                        outLen += 1
                    }
                }
            }
            // Update inner strides
            val innerIdx = outLen - 1
            val innerSize = outDims[innerIdx]
            var innerStrides = List((N).toInt()) { 0 }
            for (n3 in 0 until N) {
                innerStrides[n3] = outStrides[n3][innerIdx]
            }
            // Update outer dims
            val outerLen = innerIdx
            var outerDims = List(MAX_DIMS) { 0 }
            outerDims[0 until outerLen].copyFromSlice(outDims[0 until outerLen])
            // Update outer strides
            var outerStrides = List((N).toInt()) { List(MAX_DIMS) { 0 } }
            for (n4 in 0 until N) {
                outerStrides[n4][0 until outerLen].copyFromSlice(outStrides[n4][0 until outerLen])
            }
            // Update offsets
            var offsets = List((N).toInt()) { 0 }
            for (n5 in 0 until N) {
                offsets[n5] = layouts[n5].startOffset()
            }
            // Number of outer blocks (product of all dims except the innermost).
            val remaining = outDims[0 until outerLen].iter().product()
            return NdIter(innerSize = innerSize, innerStrides = innerStrides, outerDims = outerDims, outerStrides = outerStrides, outerLen = outerLen, startOffsets = offsets, offsets = offsets, coords = List(MAX_DIMS) { 0 }, remaining = remaining)
        }
    }
}
