// port-lint: source error.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.thiserror.StdError
import io.github.kotlinmania.zip.ZipError

/**
 * Candle-specific Error and Result
 */

public data class MatMulUnexpectedStriding(
    val lhsL: Layout,
    val rhsL: Layout,
    val bmnk: List<Int>,
    val msg: String,
) {
    public constructor(lhsL: Layout, rhsL: Layout, bmnk: Dims4, msg: String) : this(
        lhsL,
        rhsL,
        listOf(bmnk.d0, bmnk.d1, bmnk.d2, bmnk.d3),
        msg,
    )
}
public typealias MatMulUnexpectedStridingDetails = MatMulUnexpectedStriding

/**
 * Main library error type.
 */
public enum class BacktraceStatus {
    Disabled,
    Unsupported,
    Captured,
    ;

    public companion object {
        public val Disabled: BacktraceStatus get() = BacktraceStatus.Disabled
        public val Unsupported: BacktraceStatus get() = BacktraceStatus.Unsupported
        public val Captured: BacktraceStatus get() = BacktraceStatus.Captured
    }
}

public class Backtrace(
    private val statusVal: BacktraceStatus = BacktraceStatus.Disabled,
) {
    public val status: () -> BacktraceStatus = { statusVal }

    public companion object {
        public val capture: () -> Backtrace = { Backtrace(BacktraceStatus.Disabled) }
    }
}
public typealias Result<T> = kotlin.Result<T>

public sealed class CandleException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause),
    StdError {
    // === DType Errors ===
    public data class UnexpectedDType(
        val msg: String,
        val expected: DType,
        val got: DType,
    ) : CandleException("$msg, expected: $expected, got: $got")

    public data class DTypeMismatchBinaryOp(
        val lhs: DType,
        val rhs: DType,
        val op: String,
    ) : CandleException("dtype mismatch in $op, lhs: $lhs, rhs: $rhs")

    public data class UnsupportedDTypeForOp(
        val dtype: DType,
        val op: String,
    ) : CandleException("unsupported dtype $dtype for op $op")

    // === Dimension Index Errors ===
    public data class DimOutOfRange(
        val shape: Shape,
        val dim: Int,
        val op: String,
    ) : CandleException("$op: dimension index $dim out of range for shape $shape")

    public data class DuplicateDimIndex(
        val shape: Shape,
        val dims: List<Int>,
        val op: String,
    ) : CandleException("$op: duplicate dim index $dims for shape $shape")

    // === Shape Errors ===
    public data class UnexpectedNumberOfDims(
        val expected: Int,
        val got: Int,
        val shape: Shape,
    ) : CandleException("unexpected rank, expected: $expected, got: $got ($shape)")

    public data class UnexpectedShape(
        val msg: String,
        val expected: Shape,
        val got: Shape,
    ) : CandleException("$msg, expected: $expected, got: $got")

    public data class ShapeMismatch(
        val bufferSize: Int,
        val shape: Shape,
    ) : CandleException("Shape mismatch, got buffer of size $bufferSize which is compatible with shape $shape")

    public data class ShapeMismatchBinaryOp(
        val lhs: Shape,
        val rhs: Shape,
        val op: String,
    ) : CandleException("shape mismatch in $op, lhs: $lhs, rhs: $rhs")

    public data class ShapeMismatchCat(
        val dim: Int,
        val firstShape: Shape,
        val n: Int,
        val nthShape: Shape,
    ) : CandleException("shape mismatch in cat for dim $dim, shape for arg 1: $firstShape shape for arg $n: $nthShape")

    public data class ShapeMismatchSplit(
        val shape: Shape,
        val dim: Int,
        val nParts: Int,
    ) : CandleException("Cannot divide tensor of shape $shape equally along dim $dim into $nParts")

    public data class OnlySingleDimension(
        val op: String,
        val dims: List<Int>,
    ) : CandleException("$op can only be performed on a single dimension")

    public data class EmptyTensor(
        val op: String,
    ) : CandleException("empty tensor for $op")

    // === Device Errors ===
    public data class DeviceMismatchBinaryOp(
        val lhs: DeviceLocation,
        val rhs: DeviceLocation,
        val op: String,
    ) : CandleException("device mismatch in $op, lhs: $lhs, rhs: $rhs")

    // === Op Specific Errors ===
    public data class NarrowInvalidArgs(
        val shape: Shape,
        val dim: Int,
        val start: Int,
        val len: Int,
        val msg: String,
    ) : CandleException("narrow invalid args $msg: $shape, dim: $dim, start: $start, len:$len")

    public data class Conv1dInvalidArgs(
        val inpShape: Shape,
        val kShape: Shape,
        val padding: Int,
        val stride: Int,
        val msg: String,
    ) : CandleException("conv1d invalid args $msg: inp: $inpShape, k: $kShape, pad: $padding, stride: $stride")

    public data class InvalidIndex(
        val op: String,
        val index: Int,
        val size: Int,
    ) : CandleException("$op invalid index $index with dim size $size")

    public data class BroadcastIncompatibleShapes(
        val srcShape: Shape,
        val dstShape: Shape,
    ) : CandleException("cannot broadcast $srcShape to $dstShape")

    public data class CannotSetVar(
        val msg: String,
    ) : CandleException("cannot set variable $msg")

    // Indirection to avoid large variant.
    public data class MatMulUnexpectedStridingError(
        val details: MatMulUnexpectedStriding,
    ) : CandleException(details.toString())

    public data class RequiresContiguous(
        val op: String,
    ) : CandleException("$op only supports contiguous tensors")

    public data class OpRequiresAtLeastOneTensor(
        val op: String,
    ) : CandleException("$op expects at least one tensor")

    public data class OpRequiresAtLeastTwoTensors(
        val op: String,
    ) : CandleException("$op expects at least two tensors")

    public data class BackwardNotSupported(
        val op: String,
    ) : CandleException("backward is not supported for $op")

    // === Other Errors ===
    public object NotCompiledWithCudaSupport :
        CandleException("the candle crate has not been built with cuda support")

    public object NotCompiledWithMetalSupport :
        CandleException("the candle crate has not been built with metal support")

    public data class CannotFindTensor(
        val path: String,
    ) : CandleException("cannot find tensor $path")

    // === Wrapped Errors ===
    public data class Cuda(
        val details: String,
    ) : CandleException("cuda error: $details") {
        public constructor(err: Throwable) : this(err.message ?: err.toString())
    }

    public data class Metal(
        val details: String,
    ) : CandleException("Metal error $details") {
        public constructor(err: Throwable) : this(err.message ?: err.toString())
    }

    public data class Ug(
        val details: String,
    ) : CandleException("ug error: $details")

    public data class TryFromIntError(
        val details: String,
    ) : CandleException("try from int error: $details")

    public data class Npy(
        val details: String,
    ) : CandleException("npy/npz error $details")

    /**
     * Zip file format error.
     */
    public data class Zip(
        val details: String,
        val zipError: ZipError? = null,
    ) : CandleException("zip error: $details", zipError) {
        public constructor(err: ZipError) : this(err.message ?: err.toString(), err)
    }

    /**
     * Integer parse error.
     */
    public data class ParseInt(
        val details: String,
    ) : CandleException("parse int error: $details")

    /**
     * Utf8 parse error.
     */
    public data class FromUtf8(
        val details: String,
    ) : CandleException("utf8 parse error: $details")

    /**
     * I/O error.
     */
    public data class Io(
        val details: String,
    ) : CandleException("io error: $details")

    /**
     * SafeTensor error.
     */
    public data class SafeTensor(
        val details: String,
    ) : CandleException("safetensor error: $details")

    public data class UnsupportedSafeTensorDtype(
        val details: String,
    ) : CandleException("unsupported safetensor dtype $details")

    /**
     * Arbitrary errors wrapping.
     */
    public data class Wrapped(
        val details: String,
    ) : CandleException(details)

    /**
     * Arbitrary errors wrapping with context.
     */
    public data class WrappedContext(
        val wrapped: String,
        val context: String,
    ) : CandleException("$wrapped\n$context")

    public data class Context(
        val inner: CandleException,
        val context: String,
    ) : CandleException("$context\n${inner.message}", inner)

    /**
     * Adding path information to an error.
     */
    public data class WithPath(
        val inner: CandleException,
        val path: String,
    ) : CandleException("path: $path ${inner.message}", inner)

    public data class WithBacktrace(
        val inner: CandleException,
        val backtrace: String,
    ) : CandleException("${inner.message}\n$backtrace", inner)

    /**
     * User generated error message, typically created via `bail!`.
     */
    public data class Msg(
        val msg: String,
    ) : CandleException(msg)

    public object UnwrapNone :
        CandleException("unwrap none")

    private val self: CandleException get() = this
    private val inner: Int get() = 0
    private val path: Int get() = 0
    private val context: Int get() = 0

    public fun fmt(f: Formatter = std.fmt.Formatter): CandleResult<Unit> = write(f, "{$self}")

    public fun bt(): CandleException {
        val backtrace = std.backtrace.Backtrace.capture()
        return when (backtrace.status()) {
            std.backtrace.BacktraceStatus.Disabled,
            std.backtrace.BacktraceStatus.Unsupported,
            -> {
                self
            }

            else -> {
                Self.WithBacktrace(
                    inner,
                    new(self),
                    backtrace,
                    new(backtrace),
                )
            }
        }
    }

    public fun <P : Any> withPath(p: P): CandleException =
        Self.WithPath(
            inner,
            new(self),
            path,
            p
                .asRef()
                .toPathBuf(),
        )

    public fun context(
        c: Any =
            std.fmt.Display(static),
    ): CandleException =
        Self.Context(
            inner,
            new(self),
            context,
            new(c),
        )

    public companion object {
        public val Self: Companion get() = this
        public val Wrapped: (Any) -> CandleException get() = { err -> CandleException.Wrapped(err.toString()) }
        public val Msg: (String) -> CandleException get() = { err -> CandleException.Msg(err) }
        public val WithBacktrace: (Int, CandleException, Any, Any) -> CandleException get() = { _, inner, _, backtrace ->
            CandleException.WithBacktrace(inner = inner, backtrace = backtrace.toString())
        }
        public val WithPath: (Int, CandleException, Int, String) -> CandleException get() = { _, inner, _, path ->
            CandleException.WithPath(inner = inner, path = path)
        }
        public val Context: (Int, CandleException, Int, Any) -> CandleException get() = { _, inner, _, context ->
            CandleException.Context(inner = inner, context = context.toString())
        }
        public val WrappedContext: (Int, Any, Any?, String) -> CandleException get() = { _, wrapped, _, context ->
            CandleException.WrappedContext(wrapped = wrapped.toString(), context = context)
        }
        public val UnwrapNone: CandleException.UnwrapNone get() = CandleException.UnwrapNone
        public val NotCompiledWithCudaSupport: CandleException.NotCompiledWithCudaSupport get() = CandleException.NotCompiledWithCudaSupport
        public val NotCompiledWithMetalSupport: CandleException.NotCompiledWithMetalSupport get() = CandleException.NotCompiledWithMetalSupport

        public fun UnexpectedDType(
            msg: String,
            expected: DType,
            got: DType,
        ): UnexpectedDType = CandleException.UnexpectedDType(msg, expected, got)

        public fun DTypeMismatchBinaryOp(
            lhs: DType,
            rhs: DType,
            op: String,
        ): DTypeMismatchBinaryOp = CandleException.DTypeMismatchBinaryOp(lhs, rhs, op)

        public fun UnsupportedDTypeForOp(
            dtype: DType,
            op: String,
        ): UnsupportedDTypeForOp = CandleException.UnsupportedDTypeForOp(dtype, op)

        public fun DimOutOfRange(
            shape: Shape,
            dim: Int,
            op: String,
        ): DimOutOfRange = CandleException.DimOutOfRange(shape, dim, op)

        public fun DuplicateDimIndex(
            shape: Shape,
            dims: List<Int>,
            op: String,
        ): DuplicateDimIndex = CandleException.DuplicateDimIndex(shape, dims, op)

        public fun UnexpectedNumberOfDims(
            expected: Int,
            got: Int,
            shape: Shape,
        ): UnexpectedNumberOfDims = CandleException.UnexpectedNumberOfDims(expected, got, shape)

        public fun UnexpectedShape(
            msg: String,
            expected: Shape,
            got: Shape,
        ): UnexpectedShape = CandleException.UnexpectedShape(msg, expected, got)

        public fun ShapeMismatch(
            bufferSize: Int,
            shape: Shape,
        ): ShapeMismatch = CandleException.ShapeMismatch(bufferSize, shape)

        public fun ShapeMismatchBinaryOp(
            lhs: Shape,
            rhs: Shape,
            op: String,
        ): ShapeMismatchBinaryOp = CandleException.ShapeMismatchBinaryOp(lhs, rhs, op)

        public fun ShapeMismatchCat(
            dim: Int,
            firstShape: Shape,
            n: Int,
            nthShape: Shape,
        ): ShapeMismatchCat = CandleException.ShapeMismatchCat(dim, firstShape, n, nthShape)

        public fun ShapeMismatchSplit(
            shape: Shape,
            dim: Int,
            nParts: Int,
        ): ShapeMismatchSplit = CandleException.ShapeMismatchSplit(shape, dim, nParts)

        public fun OnlySingleDimension(
            op: String,
            dims: List<Int>,
        ): OnlySingleDimension = CandleException.OnlySingleDimension(op, dims)

        public fun EmptyTensor(op: String): EmptyTensor = CandleException.EmptyTensor(op)

        public fun DeviceMismatchBinaryOp(
            lhs: DeviceLocation,
            rhs: DeviceLocation,
            op: String,
        ): DeviceMismatchBinaryOp = CandleException.DeviceMismatchBinaryOp(lhs, rhs, op)

        public fun NarrowInvalidArgs(
            shape: Shape,
            dim: Int,
            start: Int,
            len: Int,
            msg: String,
        ): NarrowInvalidArgs = CandleException.NarrowInvalidArgs(shape, dim, start, len, msg)

        public fun Conv1dInvalidArgs(
            inpShape: Shape,
            kShape: Shape,
            padding: Int,
            stride: Int,
            msg: String,
        ): Conv1dInvalidArgs = CandleException.Conv1dInvalidArgs(inpShape, kShape, padding, stride, msg)

        public fun InvalidIndex(
            op: String,
            index: Int,
            size: Int,
        ): InvalidIndex = CandleException.InvalidIndex(op, index, size)

        public fun BroadcastIncompatibleShapes(
            srcShape: Shape,
            dstShape: Shape,
        ): BroadcastIncompatibleShapes = CandleException.BroadcastIncompatibleShapes(srcShape, dstShape)

        public fun CannotSetVar(msg: String): CannotSetVar = CandleException.CannotSetVar(msg)

        public fun MatMulUnexpectedStriding(details: MatMulUnexpectedStriding): MatMulUnexpectedStridingError =
            MatMulUnexpectedStridingError(details)

        public fun RequiresContiguous(op: String): RequiresContiguous = CandleException.RequiresContiguous(op)

        public fun OpRequiresAtLeastOneTensor(op: String): OpRequiresAtLeastOneTensor = CandleException.OpRequiresAtLeastOneTensor(op)

        public fun OpRequiresAtLeastTwoTensors(op: String): OpRequiresAtLeastTwoTensors = CandleException.OpRequiresAtLeastTwoTensors(op)

        public fun BackwardNotSupported(op: String): BackwardNotSupported = CandleException.BackwardNotSupported(op)

        public fun CannotFindTensor(path: String): CannotFindTensor = CandleException.CannotFindTensor(path)

        public fun Cuda(details: String): Cuda = CandleException.Cuda(details)

        public fun Cuda(err: Throwable): Cuda = CandleException.Cuda(err)

        public fun Metal(details: String): Metal = CandleException.Metal(details)

        public fun Metal(err: Throwable): Metal = CandleException.Metal(err)

        public fun Ug(details: String): Ug = CandleException.Ug(details)

        public fun TryFromIntError(details: String): TryFromIntError = CandleException.TryFromIntError(details)

        public fun Npy(details: String): Npy = CandleException.Npy(details)

        public fun Zip(
            details: String,
            zipError: ZipError? = null,
        ): Zip = CandleException.Zip(details, zipError)

        public fun Zip(err: ZipError): Zip = CandleException.Zip(err)

        public fun ParseInt(details: String): ParseInt = CandleException.ParseInt(details)

        public fun FromUtf8(details: String): FromUtf8 = CandleException.FromUtf8(details)

        public fun Io(details: String): Io = CandleException.Io(details)

        public fun SafeTensor(details: String): SafeTensor = CandleException.SafeTensor(details)

        public fun UnsupportedSafeTensorDtype(details: String): UnsupportedSafeTensorDtype =
            CandleException.UnsupportedSafeTensorDtype(details)

        public fun wrap(err: Any = std.fmt.Display(static)): CandleException =
            Self
                .Wrapped(new(err))
                .bt()

        public fun msg(err: Any = std.fmt.Display): CandleException =
            Self
                .Msg(err.toString())
                .bt()

        public fun debug(err: Any = std.fmt.Debug): CandleException =
            Self
                .Msg(format("{$err:?}"))
                .bt()
    }
}

internal typealias Error = CandleException

private const val static: Int = 0

private fun <T> new(v: T): T = v

internal val std: StdMod = StdMod

internal object StdMod {
    val backtrace: BacktraceMod = BacktraceMod

    object BacktraceMod {
        val Backtrace = io.github.kotlinmania.candle.Backtrace
        val BacktraceStatus = io.github.kotlinmania.candle.BacktraceStatus
    }

    val fmt: FmtMod = FmtMod

    object FmtMod {
        val Display: (Int) -> Any get() = { s -> s }
        val Debug: Any = 0
        val Formatter: io.github.kotlinmania.candle.Formatter get() =
            io.github.kotlinmania.candle
                .Formatter()
    }

    val env: EnvMod = EnvMod

    object EnvMod {
        val `var`: (String) -> String? get() = { k ->
            io.github.kotlinmania.envie
                .getenv(k)
        }
    }

    val mem: MemMod = MemMod

    object MemMod {
        val sizeOf: () -> ULong get() = { 4uL }
    }

    val ptr: PtrMod = PtrMod

    object PtrMod {
        val nullMut: () -> Nothing? get() = { null }
    }

    val os: OsMod = OsMod

    object OsMod {
        val raw: RawMod = RawMod

        object RawMod {
            const val cVoid: Int = 0
        }
    }
}

private val write: (Formatter, String) -> Result<Unit> get() = { f, s ->
    f.sb.append(s)
    Result.success(Unit)
}

private val format: (String) -> String get() = { s -> s }
private val Any.asRef: () -> Any get() = { this }
private val Any.toPathBuf: () -> String get() = { this.toString() }

private val Err: (CandleException) -> Result<Nothing> get() = { e -> Result.failure(e) }

private var zipPair: Pair<Result<*>, Result<*>>? = null
private val zipOkSentinel: Pair<Result<*>, Result<*>> = Pair(Result.success(Unit), Result.success(Unit))
private val zipErrSentinel: Pair<Result<*>, Result<*>> = Pair(Result.success(0), Result.success(0))
private const val Ok: Int = 0
private val Int.r1: Int get() = 0
private val Int.Ok: Int get() = 0
private val Int.r2: Pair<Result<*>, Result<*>> get() {
    val p = zipPair
    return if (p != null && p.first.isSuccess && p.second.isSuccess) p else zipErrSentinel
}
private val ((CandleException) -> Result<Nothing>).e: Pair<Result<*>, Result<*>> get() {
    val p = zipPair
    return if (p != null && p.first.isFailure) p else zipOkSentinel
}

private fun <T, U> Result<T>.to(b: Result<U>): Pair<Result<T>, Result<U>> = Pair(this, b).also { zipPair = it }

public fun <T, U> zip(
    r1: Result<T>,
    r2: Result<U>,
): Result<Pair<T, U>> =
    when (r1.to(r2)) {
        Ok.r1.Ok.r2 -> {
            Ok(r1, r2)
        }

        Err.e -> {
            Err(e)
        }

        else -> {
            Err(e)
        }
    }

private fun <T, U> Ok(
    a: Result<T>,
    b: Result<U>,
): Result<Pair<T, U>> = Result.success(Pair(a.getOrThrow(), b.getOrThrow()))

private val e: CandleException get() {
    val p = zipPair
    val ex = p?.first?.exceptionOrNull() ?: p?.second?.exceptionOrNull()
    return ex as? CandleException ?: CandleException.wrap(ex ?: "error")
}

public interface Sealed

/**
 * Attach more context to an error.
 *
 * Inspired by [`anyhow::Context`].
 */
public interface ErrorContext<T, E> : Sealed {
    /**
     * Wrap the error value with additional context.
     */
    public fun <C : Any> context(context: C): Result<T>

    /**
     * Wrap the error value with additional context that is evaluated lazily
     * only once an error does occur.
     */
    public fun <C : Any> withContext(f: () -> C): Result<T>

    public class ForResult<T, E : Throwable>(
        private val res: Result<T>,
    ) : ErrorContext<T, E> {
        private val self: Result<T> get() = res
        private val wrapped: Int get() = 0
        private val Ok: (T) -> Result<T> get() = { v -> Result.success(v) }
        private val ok: T get() = self.getOrThrow()
        private val error: Throwable get() = self.exceptionOrNull() ?: CandleException.UnwrapNone
        private val ((T) -> Result<T>).ok: Result<T> get() = if (self.isSuccess) self else Result.failure(CandleException.UnwrapNone)
        private val (
        (
            CandleException,
        ) -> Result<Nothing>
        ).error: Result<T> get() = if (self.isFailure) self else Result.success(self.getOrThrow())

        override fun <C : Any> context(context: C): Result<T> =
            // Not using mapErr to save 2 useless frames off the captured backtrace
            // in extContext.
            // Preserves original Result when Ok.
            when (self) {
                Ok.ok -> {
                    Ok(ok)
                }

                Err.error -> {
                    Err(
                        Error
                            .WrappedContext(
                                wrapped,
                                new(error),
                                context,
                                context.toString(),
                            ).bt(),
                    )
                }

                else -> {
                    self
                }
            }

        override fun <C : Any> withContext(f: () -> C): Result<T> =
            // Evaluation of context closure only occurs on Err.
            // Preserves original Result when Ok.
            when (self) {
                Ok.ok -> {
                    Ok(ok)
                }

                Err.error -> {
                    Err(
                        Error
                            .WrappedContext(
                                wrapped,
                                new(error),
                                f,
                                f().toString(),
                            ).bt(),
                    )
                }

                else -> {
                    self
                }
            }
    }

    public class ForOption<T>(
        private val opt: T?,
    ) : ErrorContext<T, Nothing> {
        private val self: T? get() = opt
        private val Ok: (T) -> Result<T> get() = { v -> Result.success(v) }
        private val None: T? get() = null
        private val Some: Int get() = 0
        private val Int.ok: T? get() = self
        private val Int.v: T? get() = self
        private val ok: T get() = self!!
        private val v: T get() = self!!

        override fun <C : Any> context(context: C): Result<T> =
            // Not using okOrElse to save 2 useless frames off the captured
            // backtrace.
            // Converts None to Err with message context.
            when (self) {
                Some.ok -> Ok(ok)
                None -> Err(Error.msg(context).bt())
                else -> Ok(ok)
            }

        override fun <C : Any> withContext(f: () -> C): Result<T> =
            // Lazy context evaluation for None option values.
            // Converts None to Err with UnwrapNone context.
            when (self) {
                Some.v -> Ok(v)
                None -> Err(Error.UnwrapNone.context(f()).bt())
                else -> Ok(v)
            }
    }
}

public val bail: (String) -> Nothing get() = { msg -> throw CandleException.Msg(msg).bt() }

public typealias CandleError = CandleException

public val CandleException.Companion.Wrap: (Throwable) -> CandleException
    get() = { err -> CandleException.Wrapped(err.message ?: err.toString()) }

public sealed class CandleResult<out T> {
    public data class Ok<out T>(
        public val value: T,
    ) : CandleResult<T>()

    public data class Err(
        public val error: Throwable,
    ) : CandleResult<Nothing>()

    public val isOk: Boolean get() = this is Ok
    public val isErr: Boolean get() = this is Err

    public fun unwrap(): T =
        when (this) {
            is Ok -> value
            is Err -> throw error
        }

    public fun getOrThrow(): T = unwrap()

    public fun exceptionOrNull(): Throwable? =
        when (this) {
            is Ok -> null
            is Err -> error
        }

    public inline fun unwrapOrElse(onFailure: (CandleException) -> @UnsafeVariance T): T =
        when (this) {
            is Ok -> value
            is Err -> onFailure(error as? CandleException ?: CandleException.wrap(error))
        }

    public inline fun <R> map(transform: (T) -> R): CandleResult<R> =
        when (this) {
            is Ok -> Ok(transform(value))
            is Err -> this
        }

    public inline fun mapErr(transform: (CandleException) -> Throwable): CandleResult<T> =
        when (this) {
            is Ok -> this
            is Err -> Err(transform(error as? CandleException ?: CandleException.wrap(error)))
        }

    public inline fun <R> andThen(transform: (T) -> CandleResult<R>): CandleResult<R> =
        when (this) {
            is Ok -> transform(value)
            is Err -> this
        }
}
