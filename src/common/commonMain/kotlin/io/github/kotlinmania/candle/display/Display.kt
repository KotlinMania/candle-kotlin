// port-lint: source display.rs
package io.github.kotlinmania.candle.display

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.andThen

// Pretty printing of tensors
//
// This implementation should be in line with the [PyTorch version](https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py).

public fun <T : WithDType> Tensor.fmtDt(f: StringBuilder) {
    val deviceStr =
        when (val matchVal = this.device().location()) {
            Cpu -> "".toOwned()
            is DeviceLocation.Cuda -> {
                val gpuId = matchVal.gpuId
                ", cuda:$gpuId"
            }
            is DeviceLocation.Metal -> {
                val gpuId1 = matchVal.gpuId
                ", metal:$gpuId"
            }
        }
    f.append("Tensor[")
    run {
        val matchVal1 = this.dims()
        when {
            matchVal1.isEmpty() -> {
                when (val matchVal2 = this.toScalar<T>()) {
                    is Ok -> {
                        val v = matchVal2.value
                        f.append("$v")
                    }
                    else -> {}
                }
            }
            matchVal1.size == 1 && matchVal1[0] < 10 -> {
                val s = matchVal1[0]
                when (val matchVal3 = this.toVec1<T>()) {
                    is Ok -> {
                        val vs = matchVal3.value
                        for ((i, v1) in vs.iter().mapIndexed { index, item -> Pair(index, item) }) {
                            if (i > 0) {
                                f.append(", ")
                            }
                            f.append("$v")
                        }
                    }
                    else -> {}
                }
            }
            else -> {
                val dims = matchVal1
                f.append("dims ")
                for ((i1, d) in dims.iter().mapIndexed { index, item -> Pair(index, item) }) {
                    if (i1 > 0) {
                        f.append(", ")
                    }
                    f.append("$d")
                }
            }
        }
    }
    f.append("; ${this.dtype().asStr()}$deviceStr]")
}

public fun Tensor.fmt(f: StringBuilder) {
    when (this.dtype()) {
        DType.u8 -> this.fmtDt<UByte>(f)
        DType.u32 -> this.fmtDt<UInt>(f)
        DType.i16 -> this.fmtDt<Short>(f)
        DType.i32 -> this.fmtDt<Int>(f)
        DType.i64 -> this.fmtDt<Long>(f)
        DType.bf16 -> this.fmtDt<Bf16>(f)
        DType.f16 -> this.fmtDt<F16>(f)
        DType.f32 -> this.fmtDt<Float>(f)
        DType.f64 -> this.fmtDt<Double>(f)
        DType.f8e4m3 -> this.fmtDt<F8e4m3>(f)
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            f.append("Tensor[${this.shape()}; dtype=${this.dtype().asStr()}, unsupported dummy type]")
        }
    }
}

/**
 * Options for Tensor pretty printing
 */
public data class PrinterOptions(
    public var precision: Int = 0,
    public var threshold: Int = 0,
    public var edgeItems: Int = 0,
    public var lineWidth: Int = 0,
    public var sciMode: Boolean? = null,
) {
    public companion object {
        // We cannot use the default trait as it's not const.
        public fun constDefault(): PrinterOptions = PrinterOptions(precision = 4, threshold = 1000, edgeItems = 3, lineWidth = 80, sciMode = null)
    }
}

public var PRINT_OPTS: Mutex<PrinterOptions> = Mutex.new(PrinterOptions.constDefault())

public fun printOptions(): Mutex<PrinterOptions> = PRINT_OPTS

public fun setPrintOptions(options: PrinterOptions) {
    PRINT_OPTS.lock().unwrap() = options
}

public fun setPrintOptionsDefault() {
    PRINT_OPTS.lock().unwrap() = PrinterOptions.constDefault()
}

public fun setPrintOptionsShort() {
    PRINT_OPTS.lock().unwrap() = PrinterOptions(precision = 2, threshold = 1000, edgeItems = 2, lineWidth = 80, sciMode = null)
}

public fun setPrintOptionsFull() {
    PRINT_OPTS.lock().unwrap() = PrinterOptions(precision = 4, threshold = Int.MAX_VALUE, edgeItems = 3, lineWidth = 80, sciMode = null)
}

public fun setLineWidth(lineWidth: Int) {
    PRINT_OPTS.lock().unwrap().lineWidth = lineWidth
}

public fun setPrecision(precision: Int) {
    PRINT_OPTS.lock().unwrap().precision = precision
}

public fun setEdgeItems(edgeItems: Int) {
    PRINT_OPTS.lock().unwrap().edgeItems = edgeItems
}

public fun setThreshold(threshold: Int) {
    PRINT_OPTS.lock().unwrap().threshold = threshold
}

public fun setSciMode(sciMode: Boolean?) {
    PRINT_OPTS.lock().unwrap().sciMode = sciMode
}

public class FmtSize(
    public var currentSize: Int = 0,
) : Appendable {
    public fun finalSize(): Int = this.currentSize

    public fun writeStr(s: String) {
        this.currentSize += s.length
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<FmtSize> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<FmtSize> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): FmtSize =
                    FmtSize(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: FmtSize) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.currentSize)
                }
            }

        public fun new(): FmtSize = FmtSize(currentSize = 0)
    }
}

public interface TensorFormatter<Elem> {
    public fun <T : Appendable> fmt(v: Elem, maxW: Int, f: T)

    public fun maxWidth(toDisplay: Tensor): Int {
        var maxWidth = 1
        when (val matchVal = toDisplay.flattenAll().andThen({ t -> t.toVec1() })) {
            is Ok -> {
                val vs = matchVal.value
                for (v in vs.iter()) {
                    val fmtSize = FmtSize.new()
                    maxWidth = Int.max(maxWidth, fmtSize.finalSize())
                }
            }
            else -> {}
        }
        return maxWidth
    }

    public fun writeNewlineIndent(i: Int, f: StringBuilder) {
        f.appendLine()
        for (unused in 0 until i) {
            f.append(" ")
        }
    }

    public fun fmtTensor(t: Tensor, indent: Int, maxW: Int, summarize: Boolean, po: PrinterOptions, f: StringBuilder) {
        val dims = t.dims()
        val edgeItems = po.edgeItems
        f.append("[")
        when {
            dims.isEmpty() -> {
                when (val matchVal = t.toScalar<Elem>()) {
                    is Ok -> {
                        val v = matchVal.value
                        this.fmt(v, maxW, f)
                    }
                    else -> {}
                }
            }
            dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
                val v1 = dims[0]
                when (val matchVal1 = t.narrow(0, 0, edgeItems).andThen({ t1 -> t1.toVec1<Elem>() })) {
                    is Ok -> {
                        val vs = matchVal1.value
                        for (v2 in vs.intoIter()) {
                            this.fmt(v2, maxW, f)
                            f.append(", ")
                        }
                    }
                    else -> {}
                }
                f.append("...")
                when (val matchVal2 = t.narrow(0, v1 - edgeItems, edgeItems).andThen({ t1 -> t1.toVec1<Elem>() })) {
                    is Ok -> {
                        val vs1 = matchVal2.value
                        for (v3 in vs1.intoIter()) {
                            f.append(", ")
                            this.fmt(v3, maxW, f)
                        }
                    }
                    else -> {}
                }
            }
            dims.size == 1 -> {
                val elementsPerLine = Int.max(1, po.lineWidth / (maxW + 2))
                when (val matchVal3 = t.toVec1<Elem>()) {
                    is Ok -> {
                        val vs2 = matchVal3.value
                        for ((i, v4) in vs2.intoIter().mapIndexed { index, item -> Pair(index, item) }) {
                            if (i > 0) {
                                if (i % elementsPerLine == 0) {
                                    f.append(",")
                                    TensorFormatter.writeNewlineIndent(indent, f)
                                } else {
                                    f.append(", ")
                                }
                            }
                            this.fmt(v4, maxW, f)
                        }
                    }
                    else -> {}
                }
            }
            else -> {
                if (summarize && dims[0] > 2 * edgeItems) {
                    for (i1 in 0 until edgeItems) {
                        run {
                            val matchVal4 = t.get(i1)
                            when (matchVal4) {
                                is Result.Ok<*> -> {
                                    val t1 = matchVal4.value
                                    this.fmtTensor(t1, indent + 1, maxW, summarize, po, f)
                                }
                                is Result.Err<*> -> {
                                    val e = matchVal4.error
                                    f.append("$e")
                                }
                            }
                        }
                        f.append(",")
                        TensorFormatter.writeNewlineIndent(indent, f)
                    }
                    f.append("...")
                    TensorFormatter.writeNewlineIndent(indent, f)
                    for (i2 in dims[0] - edgeItems until dims[0]) {
                        run {
                            val matchVal5 = t.get(i2)
                            when (matchVal5) {
                                is Result.Ok<*> -> {
                                    val t2 = matchVal5.value
                                    this.fmtTensor(t2, indent + 1, maxW, summarize, po, f)
                                }
                                is Result.Err<*> -> {
                                    val e1 = matchVal5.error
                                    f.append("$e")
                                }
                            }
                        }
                        if (i2 + 1 != dims[0]) {
                            f.append(",")
                            TensorFormatter.writeNewlineIndent(indent, f)
                        }
                    }
                } else {
                    for (i3 in 0 until dims[0]) {
                        run {
                            val matchVal6 = t.get(i3)
                            when (matchVal6) {
                                is Result.Ok<*> -> {
                                    val t3 = matchVal6.value
                                    this.fmtTensor(t3, indent + 1, maxW, summarize, po, f)
                                }
                                is Result.Err<*> -> {
                                    val e2 = matchVal6.error
                                    f.append("$e")
                                }
                            }
                        }
                        if (i3 + 1 != dims[0]) {
                            f.append(",")
                            TensorFormatter.writeNewlineIndent(indent, f)
                        }
                    }
                }
            }
        }
        f.append("]")
    }
}

public class FloatFormatter<S : WithDType>(
    public var intMode: Boolean = false,
    public var sciMode: Boolean = false,
    public var precision: Int = 0,
    public var phantom: Unit = Unit,
) : TensorFormatter<S> {
    public override fun <T : Appendable> fmt(v: S, maxW: Int, f: T) {
        if (this.sciMode) {
            f.append("$v")
        } else if (this.intMode) {
            if (v.isFinite()) {
                f.append("$v.")
            } else {
                f.append("$v")
            }
        } else {
            f.append("$v")
        }
    }

    public companion object {
        public fun <S : WithDType> new(t: Tensor, po: PrinterOptions, fromSFn: (Any?) -> S, maxValueSFn: (Any?) -> Any?, minValueSFn: (Any?) -> Any?): Result<FloatFormatter<S>> where S : WithDType, S : io.github.kotlinmania.numtraits.Float {
            var intMode = true
            var sciMode = false
            // Rather than containing all values, this should only include
            // values that end up being displayed according to [threshold].
            val values =
                when (
                    val tryResult =
                        when (val tryResult = t.flattenAll()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.toVec1()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.intoIter().filter({ v: S -> v.isFinite() && !v.isZero() }).collect<Vec>()
            if (!values.isEmpty()) {
                var nonzeroFiniteMin = maxValueSFn()
                var nonzeroFiniteMax = minValueSFn()
                for (v in values.iter()) {
                    val v1 = v.abs()
                    if (v1 < nonzeroFiniteMin) {
                        nonzeroFiniteMin = v1
                    }
                    if (v1 > nonzeroFiniteMax) {
                        nonzeroFiniteMax = v1
                    }
                }
                for (value in values.iter()) {
                    if (value.ceil() != value) {
                        intMode = false
                        break
                    }
                }
                when (val matchVal = fromSFn(1000.0)) {
                    null -> {}
                    else -> {
                        val v1 = matchVal
                        when (val matchVal1 = fromSFn(1e8)) {
                            null -> {}
                            else -> {
                                val v2 = matchVal1
                                when (val matchVal2 = fromSFn(1e-4)) {
                                    null -> {}
                                    else -> {
                                        val v3 = matchVal2
                                        sciMode = nonzeroFiniteMax / nonzeroFiniteMin > v1 || nonzeroFiniteMax > v2 || nonzeroFiniteMin < v3
                                    }
                                }
                            }
                        }
                    }
                }
            }
            when (val matchVal3 = po.sciMode) {
                null -> {}
                else -> {
                    val v2 = matchVal3
                    sciMode = v2
                }
            }
            return Result.Ok(FloatFormatter(intMode = intMode, sciMode = sciMode, precision = po.precision, phantom = Unit))
        }
    }
}

public class IntFormatter<S : WithDType>(
    public var phantom: Unit = Unit,
) : TensorFormatter<S> {
    public override fun <T : Appendable> fmt(v: S, maxW: Int, f: T) {
        f.append("$v")
    }

    public companion object {
        public fun <S : WithDType> new(): IntFormatter<S> = IntFormatter(phantom = Unit)
    }
}

public fun getSummarizedData(t: Tensor, edgeItems: Int): Result<Tensor> {
    val dims = t.dims()
    return if (dims.isEmpty()) {
        Result.Ok(t.clone())
    } else if (dims.len() == 1) {
        if (dims[0] > 2 * edgeItems) {
            Tensor.cat(
                listOf(
                    when (val tryResult = t.narrow(0, 0, edgeItems)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                    when (val tryResult = t.narrow(0, dims[0] - edgeItems, edgeItems)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    },
                ),
                0,
            )
        } else {
            Result.Ok(t.clone())
        }
    } else if (dims[0] > 2 * edgeItems) {
        val vs: Vec =
            when (
                val tryResult =
                    (0 until edgeItems)
                        .map(
                            closure@{ i ->
                                getSummarizedData(
                                    when (val tryResult = t.get(i)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return@closure Result.Err(tryResult.error)
                                    },
                                    edgeItems,
                                )
                            },
                        ).collect<Result<Vec>>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (i in (dims[0] - edgeItems) until dims[0]) {
            vs.push(
                when (
                    val tryResult =
                        getSummarizedData(
                            when (val tryResult = t.get(i)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                            edgeItems,
                        )
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
            )
        }
        Tensor.cat(vs, 0)
    } else {
        val vs1: Vec =
            when (
                val tryResult =
                    (0 until dims[0])
                        .map(
                            closure1@{ i1 ->
                                getSummarizedData(
                                    when (val tryResult = t.get(i1)) {
                                        is Result.Ok<*> -> tryResult.value
                                        is Result.Err<*> -> return@closure1 Result.Err(tryResult.error)
                                    },
                                    edgeItems,
                                )
                            },
                        ).collect<Result<Vec>>()
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        Tensor.cat(vs1, 0)
    }
}
