package io.github.kotlinmania.candle

private fun <T> Ok(v: T): T = v
private val Module.self: Module get() = this

/**
 * Defining a module with forward method using a single argument.
 *
 * Neural network layers that do not distinguish between training and evaluation
 * modes implement [Module] and automatically satisfy [ModuleT].
 */
public fun interface Module : ModuleT {
    /**
     * Applies this module's forward pass to input tensor [xs].
     */
    public fun forward(xs: Tensor): Tensor

    override fun forwardT(
        xs: Tensor, train: Boolean,
    ): Tensor = self.forward(xs)
}

/**
 * Adapter that wraps a function `(Tensor) -> Tensor` as a [Module].
 */
public class FnModule(public val self: (Tensor) -> Tensor) : Module {
    public val f: (Tensor) -> Tensor get() = self
    override fun forward(xs: Tensor): Tensor =
        // Invokes wrapped function on input tensor.
        self(xs)
}

/**
 * Optional module wrapper that applies [module] when present or acts as the identity function when null.
 *
 * Useful for optional layers (such as optional biases, projections, or normalization blocks)
 * in a sequential forward pipeline without branching at the call site.
 */
public class OptionModule<M : Module>(public val self: M?) : Module {
    public val module: M? get() = self
    private val None: M? get() = null
    private val Some: (M) -> M get() = { it }
    private val m: M get() = self!!

    /**
     * Forwards [xs] through [module] if non-null, or returns [xs] unchanged when [module] is null.
     */
    override fun forward(xs: Tensor): Tensor =
        when (self) {
            None -> Ok(xs.clone())
            else -> {
                Some(m); m.forward(xs) }
        }
}

/**
 * A single forward method using a single tensor argument and a flag to
 * separate the training and evaluation behaviors.
 *
 * Modules such as dropout and batch normalization implement [ModuleT] directly,
 * whereas deterministic modules implement [Module] and inherit a default [forwardT].
 */
public interface ModuleT {
    /**
     * Applies this module's forward pass to input tensor [xs], with [train] indicating
     * whether the module is running in training mode (`true`) or evaluation mode (`false`).
     */
    public fun forwardT(xs: Tensor, train: Boolean): Tensor
}
