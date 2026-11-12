// port-lint: source error.rs
package io.github.kotlinmania.candle.error

import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dummymetalbackend.MetalError
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.rustlang.core.result.Result
import io.github.kotlinmania.safetensors.Dtype
import io.github.kotlinmania.safetensors.SafeTensorError

// Candle-specific Error and Result

public data class MatMulUnexpectedStriding(
    public var lhsL: Layout = Layout(),
    public var rhsL: Layout = Layout(),
    public var bmnk0: Int = 0,
    public var bmnk1: Int = 0,
    public var bmnk2: Int = 0,
    public var bmnk3: Int = 0,
    public var msg: String = "",
)

/**
 * Main library error type.
 */
public sealed class Error {
    // === DType Errors ===
    public data class UnexpectedDType(
        public var msg: String,
        public var expected: DType,
        public var got: DType,
    ) : Error()
    public data class DTypeMismatchBinaryOp(
        public var lhs: DType,
        public var rhs: DType,
        public var op: String,
    ) : Error()
    public data class UnsupportedDTypeForOp(
        public var field0: DType,
        public var field1: String,
    ) : Error()

    // === Dimension Index Errors ===
    public data class DimOutOfRange(
        public var shape: Shape,
        public var dim: Int,
        public var op: String,
    ) : Error()
    public data class DuplicateDimIndex(
        public var shape: Shape,
        public var dims: List<Int>,
        public var op: String,
    ) : Error()

    // === Shape Errors ===
    public data class UnexpectedNumberOfDims(
        public var expected: Int,
        public var got: Int,
        public var shape: Shape,
    ) : Error()
    public data class UnexpectedShape(
        public var msg: String,
        public var expected: Shape,
        public var got: Shape,
    ) : Error()
    public data class ShapeMismatch(
        public var bufferSize: Int,
        public var shape: Shape,
    ) : Error()
    public data class ShapeMismatchBinaryOp(
        public var lhs: Shape,
        public var rhs: Shape,
        public var op: String,
    ) : Error()
    public data class ShapeMismatchCat(
        public var dim: Int,
        public var firstShape: Shape,
        public var n: Int,
        public var nthShape: Shape,
    ) : Error()
    public data class ShapeMismatchSplit(
        public var shape: Shape,
        public var dim: Int,
        public var nParts: Int,
    ) : Error()
    public data class OnlySingleDimension(
        public var op: String,
        public var dims: List<Int>,
    ) : Error()
    public data class EmptyTensor(
        public var op: String,
    ) : Error()

    // === Device Errors ===
    public data class DeviceMismatchBinaryOp(
        public var lhs: DeviceLocation,
        public var rhs: DeviceLocation,
        public var op: String,
    ) : Error()

    // === Op Specific Errors ===
    public data class NarrowInvalidArgs(
        public var shape: Shape,
        public var dim: Int,
        public var start: Int,
        public var len: Int,
        public var msg: String,
    ) : Error()
    public data class Conv1dInvalidArgs(
        public var inpShape: Shape,
        public var kShape: Shape,
        public var padding: Int,
        public var stride: Int,
        public var msg: String,
    ) : Error()
    public data class InvalidIndex(
        public var op: String,
        public var index: Int,
        public var size: Int,
    ) : Error()
    public data class BroadcastIncompatibleShapes(
        public var srcShape: Shape,
        public var dstShape: Shape,
    ) : Error()
    public data class CannotSetVar(
        public var msg: String,
    ) : Error()

    // Holder indirection to avoid large variant.
    public data class MatMulUnexpectedStriding(
        public var value: MatMulUnexpectedStriding,
    ) : Error()
    public data class RequiresContiguous(
        public var op: String,
    ) : Error()
    public data class OpRequiresAtLeastOneTensor(
        public var op: String,
    ) : Error()
    public data class OpRequiresAtLeastTwoTensors(
        public var op: String,
    ) : Error()
    public data class BackwardNotSupported(
        public var op: String,
    ) : Error()

    // === Other Errors ===
    public data object NotCompiledWithCudaSupport : Error()
    public data object NotCompiledWithMetalSupport : Error()
    public data class CannotFindTensor(
        public var path: String,
    ) : Error()

    // === Wrapped Errors ===
    public data class Cuda(
        public var value: Error,
    ) : Error()
    public data class Metal(
        public var value: MetalError,
    ) : Error()
    public data class TryFromIntError(
        public var value: TryFromIntError,
    ) : Error()
    public data class Npy(
        public var value: String,
    ) : Error()

    /**
     * Zip file format error.
     */
    public data class Zip(
        public var value: ZipError,
    ) : Error()

    /**
     * Integer parse error.
     */
    public data class ParseInt(
        public var value: ParseIntError,
    ) : Error()

    /**
     * Utf8 parse error.
     */
    public data class FromUtf8(
        public var value: FromUtf8Error,
    ) : Error()

    /**
     * I/O error.
     */
    public data class Io(
        public var value: Error,
    ) : Error()

    /**
     * SafeTensor error.
     */
    public data class SafeTensor(
        public var value: SafeTensorError,
    ) : Error()
    public data class UnsupportedSafeTensorDtype(
        public var value: Dtype,
    ) : Error()

    /**
     * Arbitrary errors wrapping.
     */
    public data class Wrapped(
        public var value: Any,
    ) : Error()

    /**
     * Arbitrary errors wrapping with context.
     */
    public data class WrappedContext(
        public var wrapped: Error,
        public var context: String,
    ) : Error()
    public data class Context(
        public var inner: Error,
        public var context: Any,
    ) : Error()

    /**
     * Adding path information to an error.
     */
    public data class WithPath(
        public var inner: Error,
        public var path: PathBuf,
    ) : Error()
    public data class WithBacktrace(
        public var inner: Error,
        public var backtrace: Backtrace,
    ) : Error()

    /**
     * User generated error message, typically created via `bail`.
     */
    public data class Msg(
        public var value: String,
    ) : Error()
    public data object UnwrapNone : Error()

    public fun fmt(f: StringBuilder) {
        f.append("$this")
    }

    public fun bt(): Error {
        val backtrace = Backtrace.capture()
        return when (backtrace.status()) {
            BacktraceStatus.Disabled, BacktraceStatus.Unsupported -> this
            else -> Error.WithBacktrace(inner = this, backtrace = backtrace)
        }
    }

    public fun <P : AsRef<Path>> withPath(p: P): Error = Error.WithPath(inner = this, path = p.asRef().toPathBuf())

    public fun context(c: Any): Error = Error.Context(inner = this, context = c)

    public companion object {
        public fun wrap(err: Any): Error = Error.Wrapped(err).bt()

        public fun msg(err: Any): Error = Error.Msg(err.toString()).bt()

        public fun debug(err: Any): Error = Error.Msg("$err").bt()
    }
}

public typealias Result<T> = io.github.kotlinmania.rustlang.core.result.Result<T, Error>

public fun <T, U> zip(r1: Result<T>, r2: Result<U>): Result<Pair<T, U>> =
    run {
        val matchVal = Pair(r1, r2)
        when {
            matchVal.first is Ok && matchVal.second is Ok -> {
                val (r11) = matchVal.first as Ok
                val (r21) = matchVal.second as Ok
                Ok(Pair(r11, r21))
            }
            matchVal.first is Err -> {
                val (e) = matchVal.first as Err
                Err(e)
            }
            else -> {
                val (e1) = matchVal.second as Err
                Err(e1)
            }
        }
    }

/**
 * Attach more context to an error.
 *
 * Inspired by [anyhow.Context].
 */
public interface Context<T, E> {
    /**
     * Wrap the error value with additional context.
     */
    public fun <C> context(context: C): io.github.kotlinmania.rustlang.core.result.Result<T, Error>

    /**
     * Wrap the error value with additional context that is evaluated lazily
     * only once an error does occur.
     */
    public fun <C> withContext(f: () -> C): io.github.kotlinmania.rustlang.core.result.Result<T, Error>
}

public fun <T, E, C> io.github.kotlinmania.rustlang.core.result.Result<T, E>.context(context: C): io.github.kotlinmania.rustlang.core.result.Result<T, Error> where E : Error {
    // Not using mapErr to save 2 useless frames off the captured backtrace
    // in extContext.
    return when (this) {
        is Result.Ok -> {
            val ok = this.value
            Ok(ok)
        }
        is Result.Err -> {
            val error = this.value
            Err(Error.WrappedContext(wrapped = error, context = context.toString()).bt())
        }
    }
}

public fun <T, E, C> io.github.kotlinmania.rustlang.core.result.Result<T, E>.withContext(context: () -> C): io.github.kotlinmania.rustlang.core.result.Result<T, Error> where E : Error =
    when (this) {
        is Result.Ok -> {
            val ok = this.value
            Ok(ok)
        }
        is Result.Err -> {
            val error = this.value
            Err(Error.WrappedContext(wrapped = error, context = context().toString()).bt())
        }
    }

public fun <T, C> T?.context(context: C): io.github.kotlinmania.rustlang.core.result.Result<T, Error> {
    // Not using okOrElse to save 2 useless frames off the captured
    // backtrace.
    return when (this) {
        null -> Err(Error.msg(context).bt())
        else -> {
            val ok = this
            Ok(ok)
        }
    }
}

public fun <T, C> T?.withContext(context: () -> C): io.github.kotlinmania.rustlang.core.result.Result<T, Error> =
    when (this) {
        null -> Err(Error.UnwrapNone.context(context()).bt())
        else -> {
            val v = this
            Ok(v)
        }
    }

private const val BAIL_TEMPLATE: String = "ret Err(Error.Msg(format(\$msg).into()).bt())"

private const val BAIL_TEMPLATE_1: String = "ret Err(Error.Msg(format(\$err).into()).bt())"

private const val BAIL_TEMPLATE_2: String = "ret Err(Error.Msg(format(\$fmt, \$(\$arg)*).into()).bt())"

public fun bail(msg: String = "", err: String = "", fmt: String = "", arg: String = ""): String =
    when {
        fmt.isNotEmpty() && arg.isNotEmpty() -> arrayOf("\$fmt" to fmt, "\$arg" to arg).fold(BAIL_TEMPLATE_2) { acc0, (k0, v0) -> acc0.replace(k0, v0) }
        msg.isNotEmpty() -> BAIL_TEMPLATE.replace("\$msg", msg)
        err.isNotEmpty() -> BAIL_TEMPLATE_1.replace("\$err", err)
        else -> ""
    }
