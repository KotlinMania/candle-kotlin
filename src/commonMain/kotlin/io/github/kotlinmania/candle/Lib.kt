// port-lint: source lib.rs
package io.github.kotlinmania.candle

/**
 * ML framework for Rust
 *
 * ```kotlin
 * val a = Tensor.arange(0f, 6f, Device.Cpu).reshape(2, 3)
 * val b = Tensor.arange(0f, 12f, Device.Cpu).reshape(3, 4)
 * val c = a.matmul(b)
 * ```
 *
 * ## Features
 *
 * - Simple syntax (looks and feels like PyTorch)
 * - CPU and Cuda backends (and M1 support)
 * - Enable serverless (CPU) small and fast deployments
 * - Model training
 * - Distributed computing (NCCL).
 * - Models out of the box (Llama, Whisper, Falcon, ...)
 *
 * ## FAQ
 *
 * - Why Candle?
 *
 * Candle stems from the need to reduce binary size in order to *enable serverless*
 * possible by making the whole engine smaller than PyTorch very large library volume
 *
 * And simply *removing Python* from production workloads.
 * Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
 *
 * Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)
 *
 * ## Other Crates
 *
 * Candle consists of a number of crates. This crate holds core the common data structures but you may wish
 * to look at the docs for the other crates which can be found here:
 *
 * - [candle-core](https://docs.rs/candle-core/). Core Datastructures and DataTypes.
 * - [candle-nn](https://docs.rs/candle-nn/). Building blocks for Neural Nets.
 * - [candle-datasets](https://docs.rs/candle-datasets/). Rust access to commonly used Datasets like MNIST.
 * - [candle-examples](https://docs.rs/candle-examples/). Examples of Candle in Use.
 * - [candle-onnx](https://docs.rs/candle-onnx/). Loading and using ONNX models.
 * - [candle-pyo3](https://docs.rs/candle-pyo3/). Access to Candle from Python.
 * - [candle-transformers](https://docs.rs/candle-transformers/). Candle implementation of many published transformer models.
 */
private val <T> T.Ok: (T) -> T get() = { it }
private val <T> T.Ok2: (T, Any?) -> T get() = { v, _ -> v }

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
private val ((Tensor) -> Tensor).self: (Tensor) -> Tensor get() = this

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
