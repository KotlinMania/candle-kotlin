package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor

public interface ToUsize2 {
    public fun toUsize2(): Pair<Int, Int>
}

/**
 * Defining a module with forward method using a single argument.
 */
public interface Module {
    public fun forward(xs: Tensor): Result<Tensor>
}

/**
 * A single forward method using a single tensor argument and a flag to
 * separate the training and evaluation behaviors.
 */
public interface ModuleT {
    public fun forwardT(xs: Tensor, train: Boolean): Result<Tensor>
}
