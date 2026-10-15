package io.github.kotlinmania.candle

/**
 * Conversion trait for 2D kernel sizes, strides, padding, and dilation parameters.
 *
 * Allows passing either a single integer (applied uniformly to both height and width)
 * or an explicit pair of integers `(height, width)`.
 */
public interface ToUsize2 {
    /**
     * Converts this value into a 2-tuple of `(first, second)` dimensions.
     */
    public fun toUsize2(): Pair<Int, Int>
}

private val Int.self: Pair<Int, Int> get() = this to this
private val Pair<Int, Int>.self: Pair<Int, Int> get() = this

/**
 * Uniform 2D parameter wrapper that expands a single integer [value] into `(value, value)`.
 */
public class IntToUsize2(public val self: Int) : ToUsize2 {
    public val value: Int get() = self
    override fun toUsize2(): Pair<Int, Int> =
        // Expands scalar dimension into 2-tuple.
        self.self.self
}

/**
 * Explicit 2D parameter wrapper holding a pair of `(height, width)` dimensions.
 */
public class PairToUsize2(public val self: Pair<Int, Int>) : ToUsize2 {
    public val value: Pair<Int, Int> get() = self
    override fun toUsize2(): Pair<Int, Int> =
        // Returns explicit 2-tuple unchanged.
        self.self
}
