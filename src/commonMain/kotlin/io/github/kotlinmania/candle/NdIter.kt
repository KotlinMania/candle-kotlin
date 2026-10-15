// port-lint: source nditer.rs
package io.github.kotlinmania.candle

public typealias Item = IntArray

/**
 * Maximum tensor rank supported by [NdIter].
 */
public const val MAX_DIMS: Int = 8

/**
 * Multi-dimensional iterator that walks `N` tensor layouts simultaneously.
 *
 * Internally, adjacent dimensions are merged whenever their strides allow it,
 * reducing the number of outer iterations. Each call to [Iterator.next] yields
 * `IntArray`, one base offset per layout, for one inner slice of `innerSize`
 * elements. Callers iterate that slice using `innerStrides`:
 *
 * ```kotlin
 * val ndIter = NdIter.new(lhsL, rhsL)
 * val innerSize = ndIter.innerSize
 * val innerLs = ndIter.innerStrides[0]
 * val innerRs = ndIter.innerStrides[1]
 * for (offsets in ndIter) {
 *     val lhsOff = offsets[0]
 *     val rhsOff = offsets[1]
 *     for (i in 0 until innerSize) {
 *         val lhs = lhsBuf[lhsOff + i * innerLs]
 *         val rhs = rhsBuf[rhsOff + i * innerRs]
 *     }
 * }
 * ```
 *
 * Note: `innerStrides[n]` can be 1 (contiguous), but also 0 (broadcast/scalar) or > 1 (non-contiguous),
 * so callers must not assume contiguous element access within a slice.
 */
public class NdIter internal constructor(
    public val innerSize: Int,
    /** Per-layout element stride within each inner slice: `elementOffset = base + i * stride`. */
    public val innerStrides: IntArray,
    private val outerDims: IntArray,
    private val outerStrides: Array<IntArray>,
    public val outerLen: Int,
    private val offsets: IntArray,
    private val coords: IntArray,
    public var remaining: Int
) : Iterator<IntArray> {
    public constructor(layouts: List<Layout>) : this(new(layouts))
    public constructor(vararg layouts: Layout) : this(new(layouts.toList()))
    private constructor(other: NdIter) : this(
        other.innerSize,
        other.innerStrides,
        other.outerDims,
        other.outerStrides,
        other.outerLen,
        other.offsets,
        other.coords,
        other.remaining
    )

    public val n: Int = innerStrides.size
    private val self: NdIter get() = this

    public val len: Int get() = remaining

    public companion object {
        public fun new(layouts: List<Layout>): NdIter {
            val N = layouts.len()
            val dims = layouts[0].dims()
            debugAssert(
                dims.len() <= MAX_DIMS,
                "rank {} exceeds MAX_DIMS={}",
                dims.len(),
                MAX_DIMS,
            )
            for (l in layouts) {
                debugAssertEq(l.dims(), dims)
            }

            val rank = dims.len()
            val outDims = IntArray(MAX_DIMS)
            val outStrides = Array(N) { IntArray(MAX_DIMS) }
            var outLen: Int

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

                    val (canMerge, useInner) = if (lastD == 1) {
                        Pair(true, true)
                    } else if (d == 1) {
                        Pair(true, false)
                    } else {
                        val canMerge =
                            (0 until N).all { n -> outStrides[n][top] == layouts[n].stride()[i] * d }
                        Pair(canMerge, true)
                    }
                    if (canMerge) {
                        outDims[top] = lastD * d
                        if (useInner) {
                            for (n in 0 until N) {
                                outStrides[n][top] = layouts[n].stride()[i]
                            }
                        }
                    } else {
                        outDims[outLen] = d
                        for (n in 0 until N) {
                            outStrides[n][outLen] = layouts[n].stride()[i]
                        }
                        outLen += 1
                    }
                }
            }

            // Update inner strides
            val innerIdx = outLen - 1
            val innerSize = outDims[innerIdx]
            val innerStrides = IntArray(N)
            for (n in 0 until N) {
                innerStrides[n] = outStrides[n][innerIdx]
            }

            // Update outer dims
            val outerLen = innerIdx
            val outerDims = IntArray(MAX_DIMS)
            outDims.copyInto(outerDims, 0, 0, outerLen)

            // Update outer strides
            val outerStrides = Array(N) { IntArray(MAX_DIMS) }
            for (n in 0 until N) {
                outStrides[n].copyInto(outerStrides[n], 0, 0, outerLen)
            }

            // Update offsets
            val offsets = IntArray(N)
            for (n in 0 until N) {
                offsets[n] = layouts[n].startOffset()
            }

            // Number of outer blocks (product of all dims except the innermost).
            val remaining = outDims.take(outerLen).asSequence().product()

            return NdIter(
                innerSize,
                innerStrides,
                outerDims,
                outerStrides,
                outerLen,
                offsets,
                IntArray(MAX_DIMS),
                remaining,
            )
        }
    }

    override fun hasNext(): Boolean = remaining > 0

    override fun next(): IntArray {
        if (self.remaining == 0) {
            throw NoSuchElementException()
        }
        val item = self.offsets.copyOf()
        self.remaining -= 1

        for (k in (0 until self.outerLen).rev()) {
            self.coords[k] += 1
            for (n in 0 until this.n) {
                self.offsets[n] += self.outerStrides[n][k]
            }
            if (self.coords[k] < self.outerDims[k]) {
                break
            }
            self.coords[k] = 0
            for (n in 0 until this.n) {
                self.offsets[n] -= self.outerDims[k] * self.outerStrides[n][k]
            }
        }

        return Some(item)
    }

    public fun sizeHint(): Pair<Int, Int?> {
        return Pair(self.remaining, Some(self.remaining))
    }
}
