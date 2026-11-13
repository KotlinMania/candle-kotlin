package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor

public fun Int.toUsize2(): Pair<Int, Int> = Pair(this, this)

public fun Pair<Int, Int>.toUsize2(): Pair<Int, Int> = this

public fun ((Tensor) -> Result<Tensor>).forward(xs: Tensor): Result<Tensor> = this(xs)

public fun <M : Module> M?.forward(xs: Tensor): Result<Tensor> =
    when (this) {
        null -> Result.Ok(xs.clone())
        else -> {
            val m = this
            m.forward(xs)
        }
    }

public fun <M : Module> M.forwardT(xs: Tensor, train: Boolean): Result<Tensor> = this.forward(xs)
