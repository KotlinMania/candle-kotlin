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

public class FnModule(public val f: (Tensor) -> Tensor) : Module {
    override fun forward(xs: Tensor): Tensor = f(xs)
}

public class OptionModule<M : Module>(public val module: M?) : Module {
    override fun forward(xs: Tensor): Tensor =
        when (val m = module) {
            null -> xs
            else -> m.forward(xs)
        }
}

/**
 * A single forward method using a single single tensor argument and a flag to
 * separate the training and evaluation behaviors.
 */
public interface ModuleT {
    public fun forwardT(xs: Tensor, train: Boolean): Tensor
}
