// port-lint: source lib.rs
package io.github.kotlinmania.candle

public interface ToUsize2 {
    public fun toUsize2(): Pair<Int, Int>
}

public class IntToUsize2(public val value: Int) : ToUsize2 {
    override fun toUsize2(): Pair<Int, Int> = Pair(value, value)
}

public class PairToUsize2(public val value: Pair<Int, Int>) : ToUsize2 {
    override fun toUsize2(): Pair<Int, Int> = value
}

/**
 * Defining a module with forward method using a single argument.
 */
public fun interface Module : ModuleT {
    public fun forward(xs: Tensor): Tensor

    override fun forwardT(xs: Tensor, train: Boolean): Tensor = forward(xs)
}

/**
 * A single forward method using a single tensor argument and a flag to
 * separate the training and evaluation behaviors.
 */
public interface ModuleT {
    public fun forwardT(xs: Tensor, train: Boolean): Tensor
}

public object CandleCore {
    public const val VERSION: String = "0.8.2"
}
