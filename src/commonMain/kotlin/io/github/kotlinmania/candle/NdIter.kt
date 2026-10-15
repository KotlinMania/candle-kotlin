// port-lint: source candle-core/src/nditer.rs
package io.github.kotlinmania.candle

/**
 * Maximum tensor rank supported by [NdIter].
 */
public const val MAX_DIMS: Int = 8

/**
 * Multi-dimensional iterator that walks `N` tensor layouts simultaneously.
 *
 * Internally, adjacent dimensions are merged whenever their strides allow it,
 * reducing the number of outer iterations. Each call to [next] yields
 * `IntArray`, one base offset per layout, for one inner slice of `innerSize`
 * elements. Callers iterate that slice using `innerStrides`.
 */
public class NdIter(
    layouts: List<Layout>,
) : Iterator<IntArray> {
    public constructor(vararg layouts: Layout) : this(layouts.toList())

    public val n: Int = layouts.size
    public val innerSize: Int
    public val innerStrides: IntArray

    private val outerDims: IntArray = IntArray(MAX_DIMS)
    private val outerStrides: Array<IntArray> = Array(n) { IntArray(MAX_DIMS) }
    public val outerLen: Int

    private val offsets: IntArray
    private val coords: IntArray = IntArray(MAX_DIMS)
    public var remaining: Int

    init {
        require(layouts.isNotEmpty()) { "layouts must not be empty" }
        val dims = layouts[0].dims
        require(dims.size <= MAX_DIMS) { "rank ${dims.size} exceeds MAX_DIMS=$MAX_DIMS" }
        for (l in layouts) {
            require(l.dims == dims) { "all layouts must have identical dimensions" }
        }

        val rank = dims.size
        val outDims = IntArray(MAX_DIMS)
        val outStrides = Array(n) { IntArray(MAX_DIMS) }
        var outLen: Int

        if (rank == 0) {
            outDims[0] = 1
            outLen = 1
        } else {
            outDims[0] = dims[0]
            for (i in 0 until n) {
                outStrides[i][0] = layouts[i].stride[0]
            }
            outLen = 1

            for (i in 1 until rank) {
                val d = dims[i]
                val top = outLen - 1
                val lastD = outDims[top]

                val (canMerge, useInner) = if (lastD == 1) {
                    Pair(true, true)
                } else if (d == 1) {
                    Pair(true, false)
                } else {
                    val canMerge = (0 until n).all { ni ->
                        outStrides[ni][top] == layouts[ni].stride[i] * d
                    }
                    Pair(canMerge, true)
                }

                if (canMerge) {
                    outDims[top] = lastD * d
                    if (useInner) {
                        for (ni in 0 until n) {
                            outStrides[ni][top] = layouts[ni].stride[i]
                        }
                    }
                } else {
                    outDims[outLen] = d
                    for (ni in 0 until n) {
                        outStrides[ni][outLen] = layouts[ni].stride[i]
                    }
                    outLen++
                }
            }
        }

        val innerIdx = outLen - 1
        innerSize = outDims[innerIdx]
        innerStrides = IntArray(n) { ni -> outStrides[ni][innerIdx] }

        outerLen = innerIdx
        for (i in 0 until outerLen) {
            outerDims[i] = outDims[i]
            for (ni in 0 until n) {
                outerStrides[ni][i] = outStrides[ni][i]
            }
        }

        offsets = IntArray(n) { ni -> layouts[ni].startOffset }

        var rem = 1
        for (i in 0 until outerLen) {
            rem *= outDims[i]
        }
        remaining = rem
    }

    public val len: Int get() = remaining

    override fun hasNext(): Boolean = remaining > 0

    override fun next(): IntArray {
        if (remaining == 0) {
            throw NoSuchElementException()
        }
        val item = offsets.copyOf()
        remaining--

        for (k in (outerLen - 1) downTo 0) {
            coords[k]++
            for (ni in 0 until n) {
                offsets[ni] += outerStrides[ni][k]
            }
            if (coords[k] < outerDims[k]) {
                break
            }
            coords[k] = 0
            for (ni in 0 until n) {
                offsets[ni] -= outerDims[k] * outerStrides[ni][k]
            }
        }

        return item
    }
}
