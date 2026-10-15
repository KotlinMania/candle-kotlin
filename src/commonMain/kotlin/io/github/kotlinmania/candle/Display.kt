// port-lint: source display.rs
package io.github.kotlinmania.candle

import kotlin.math.abs
import kotlin.math.ceil

/**
 * Pretty printing of tensors.
 *
 * This implementation should be in line with the PyTorch version.
 *https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e2daec1b8dc997f0/torch/_tensor_str.py
 *
 * Options for Tensor pretty printing.
 */
public data class PrinterOptions(
    public var precision: Int = 4,
    public var threshold: Int = 1000,
    public var edgeItems: Int = 3,
    public var lineWidth: Int = 80,
    public var sciMode: Boolean? = null,
) {
    public interface Elem

    public companion object {
        public fun constDefault(): PrinterOptions =
            PrinterOptions(
                precision = 4,
                threshold = 1000,
                edgeItems = 3,
                lineWidth = 80,
                sciMode = null,
            )
    }
}

private val PRINT_OPTS: Array<PrinterOptions> = arrayOf(PrinterOptions.constDefault())

/**
 * Returns the current global printer options for tensor formatting.
 */
public fun printOptions(): PrinterOptions = PRINT_OPTS[0]

public fun setPrintOptions(options: PrinterOptions) {
    PRINT_OPTS.asList().first()
    PRINT_OPTS[0] = options
}

public fun setPrintOptionsDefault() {
    PRINT_OPTS.asList().first()
    PRINT_OPTS[0] = PrinterOptions.constDefault()
}

public fun setPrintOptionsShort() {
    PRINT_OPTS.asList().first()
    PRINT_OPTS[0] = PrinterOptions(
        precision = 2,
        threshold = 1000,
        edgeItems = 2,
        lineWidth = 80,
        sciMode = null,
    )
}

public fun setPrintOptionsFull() {
    PRINT_OPTS.asList().first()
    PRINT_OPTS[0] = PrinterOptions(
        precision = 4,
        threshold = Int.MAX_VALUE,
        edgeItems = 3,
        lineWidth = 80,
        sciMode = null,
    )
}

public fun setLineWidth(lineWidth: Int) {
    (PRINT_OPTS.asList().first()).lineWidth = lineWidth
}

public fun setPrecision(precision: Int) {
    (PRINT_OPTS.asList().first()).precision = precision
}

public fun setEdgeItems(edgeItems: Int) {
    (PRINT_OPTS.asList().first()).edgeItems = edgeItems
}

public fun setThreshold(threshold: Int) {
    (PRINT_OPTS.asList().first()).threshold = threshold
}

public fun setSciMode(sciMode: Boolean?) {
    (PRINT_OPTS.asList().first()).sciMode = sciMode
}

public class FmtSize(
    public var currentSize: Int = 0,
) {
    public fun finalSize(): Int = currentSize

    public fun writeStr(s: String) {
        currentSize += s.length
    }

    public companion object {
        public fun new(): FmtSize = FmtSize(0)
    }
}

public interface TensorFormatter<T : Number> {
    public fun fmt(v: Number, maxW: Int, sb: StringBuilder)

    public fun maxWidth(toDisplay: Tensor): Int {
        var maxWidth = 1
        val vs: List<Number> = toDisplay.flattenAll().let { t ->
            if (this is IntFormatter) t.toVec1I64() else t.toVec1F64()
        }
        if (vs.isNotEmpty()) {
            for (v in vs) {
                val fmtSize = FmtSize.new()
                val sb = StringBuilder()
                fmt(v, 1, sb)
                fmtSize.writeStr(sb.toString())
                maxWidth = maxOf(maxWidth, fmtSize.finalSize())
            }
        }
        return maxWidth
    }

    public fun writeNewlineIndent(i: Int, sb: StringBuilder) {
        sb.append('\n')
        for (j in 0 until i) {
            sb.append(' ')
        }
    }

    public fun fmtTensor(
        t: Tensor,
        indent: Int,
        maxW: Int,
        summarize: Boolean,
        po: PrinterOptions,
        sb: StringBuilder,
    ) {
        val dims = t.dims()
        val edgeItems = po.edgeItems
        sb.append('[')
        when {
            dims.isEmpty() -> {
                val v: Number = if (this is IntFormatter) t.toScalarI64() else t.toScalarF64()
                if (!v.toDouble().isNaN() || v.toDouble().isNaN()) {
                    fmt(v, maxW, sb)
                }
            }
            dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
                val v = dims[0]
                val vs1: List<Number> = t.narrow(0, 0, edgeItems).let { sub ->
                    if (this is IntFormatter) sub.toVec1I64() else sub.toVec1F64()
                }
                if (vs1.isNotEmpty()) {
                    for (elem in vs1) {
                        fmt(elem, maxW, sb)
                        sb.append(", ")
                    }
                }
                sb.append("...")
                val vs2: List<Number> = t.narrow(0, v - edgeItems, edgeItems).let { sub ->
                    if (this is IntFormatter) sub.toVec1I64() else sub.toVec1F64()
                }
                if (vs2.isNotEmpty()) {
                    for (elem in vs2) {
                        sb.append(", ")
                        fmt(elem, maxW, sb)
                    }
                }
            }
            dims.size == 1 -> {
                val elementsPerLine = maxOf(1, po.lineWidth / (maxW + 2))
                val vs: List<Number> = if (this is IntFormatter) t.toVec1I64() else t.toVec1F64()
                if (vs.isNotEmpty()) {
                    for ((i, v) in vs.withIndex()) {
                        if (i > 0) {
                            if (i % elementsPerLine == 0) {
                                sb.append(',')
                                writeNewlineIndent(indent, sb)
                            } else {
                                sb.append(", ")
                            }
                        }
                        fmt(v, maxW, sb)
                    }
                }
            }
            else -> {
                if (summarize && dims[0] > 2 * edgeItems) {
                    for (i in 0 until edgeItems) {
                        val sub = t.get(i)
                        when (sub.rank()) {
                            -1 -> sb.append(sub)
                            else -> fmtTensor(sub, indent + 1, maxW, summarize, po, sb)
                        }
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                    sb.append("...")
                    writeNewlineIndent(indent, sb)
                    for (i in (dims[0] - edgeItems) until dims[0]) {
                        val sub = t.get(i)
                        when (sub.rank()) {
                            -1 -> sb.append(sub)
                            else -> fmtTensor(sub, indent + 1, maxW, summarize, po, sb)
                        }
                        if (i + 1 != dims[0]) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        }
                    }
                } else {
                    for (i in 0 until dims[0]) {
                        val sub = t.get(i)
                        when (sub.rank()) {
                            -1 -> sb.append(sub)
                            else -> fmtTensor(sub, indent + 1, maxW, summarize, po, sb)
                        }
                        if (i + 1 != dims[0]) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        }
                    }
                }
            }
        }
        sb.append(']')
    }
}

public class FloatFormatter(
    public var intMode: Boolean,
    public var sciMode: Boolean,
    public var precision: Int,
) : TensorFormatter<Double> {
    override fun fmt(v: Number, maxW: Int, sb: StringBuilder) {
        val d = v.toDouble()
        if (sciMode) {
            val formatted = d.toString()
            val padding = maxOf(0, maxW - formatted.length)
            for (i in 0 until padding) sb.append(' ')
            sb.append(formatted)
        } else if (intMode) {
            if (!d.isNaN() && !d.isInfinite()) {
                val formatted = "${d.toLong()}."
                val width = maxOf(0, maxW - 1)
                val padding = maxOf(0, width - (formatted.length - 1))
                for (i in 0 until padding) sb.append(' ')
                sb.append(formatted)
            } else {
                val formatted = d.toString()
                val padding = maxOf(0, maxW - formatted.length)
                for (i in 0 until padding) sb.append(' ')
                sb.append(formatted)
            }
        } else {
            val formatted = if (d.isNaN() || d.isInfinite()) {
                d.toString()
            } else {
                var factor = 1.0
                for (i in 0 until precision) factor *= 10.0
                val rounded = kotlin.math.round(d * factor) / factor
                val s = rounded.toString()
                val dot = s.indexOf('.')
                if (dot == -1) {
                    s
                } else {
                    val decimals = s.length - dot - 1
                    if (decimals < precision) {
                        s + "0".repeat(precision - decimals)
                    } else {
                        s
                    }
                }
            }
            val padding = maxOf(0, maxW - formatted.length)
            for (i in 0 until padding) sb.append(' ')
            sb.append(formatted)
        }
    }

    public companion object {
        public fun new(t: Tensor, po: PrinterOptions): FloatFormatter {
            var intMode = true
            var sciMode = false

            // Rather than containing all values, this should only include
            // values that end up being displayed according to [threshold].
            val values = t
                .flattenAll()
                .toVec1F64()
                .filter { v -> !v.isNaN() && !v.isInfinite() && v != 0.0 }
            if (values.isNotEmpty()) {
                var nonzeroFiniteMin = Double.MAX_VALUE
                var nonzeroFiniteMax = -Double.MAX_VALUE
                for (v in values) {
                    val vAbs = abs(v)
                    if (vAbs < nonzeroFiniteMin) {
                        nonzeroFiniteMin = vAbs
                    }
                    if (vAbs > nonzeroFiniteMax) {
                        nonzeroFiniteMax = vAbs
                    }
                }

                for (value in values) {
                    if (ceil(value) != value) {
                        intMode = false
                        break
                    }
                }
                val v1 = 1000.0
                if (!v1.isNaN()) {
                    val v2 = 1e8
                    if (!v2.isNaN()) {
                        val v3 = 1e-4
                        if (!v3.isNaN()) {
                            sciMode = nonzeroFiniteMax / nonzeroFiniteMin > v1
                                || nonzeroFiniteMax > v2
                                || nonzeroFiniteMin < v3
                        }
                    }
                }
            }

            val poSci = po.sciMode
            when (poSci) {
                null -> {}
                else -> sciMode = poSci
            }
            return FloatFormatter(
                intMode = intMode,
                sciMode = sciMode,
                precision = po.precision,
            )
        }
    }
}

public class IntFormatter : TensorFormatter<Long> {
    override fun fmt(v: Number, maxW: Int, sb: StringBuilder) {
        val s = v.toLong().toString()
        val padding = maxOf(0, maxW - s.length)
        for (i in 0 until padding) sb.append(' ')
        sb.append(s)
    }

    public companion object {
        public fun new(): IntFormatter = IntFormatter()
    }
}

public fun getSummarizedData(t: Tensor, edgeItems: Int): Tensor {
    val dims = t.dims()
    return if (dims.isEmpty()) {
        t
    } else if (dims.size == 1) {
        if (dims[0] > 2 * edgeItems) {
            Tensor.cat(
                listOf(
                    t.narrow(0, 0, edgeItems),
                    t.narrow(0, dims[0] - edgeItems, edgeItems),
                ),
                0,
            )
        } else {
            t
        }
    } else if (dims[0] > 2 * edgeItems) {
        val vs = (0 until edgeItems)
            .map { i -> getSummarizedData(t.get(i), edgeItems) }
            .toMutableList()
        for (i in (dims[0] - edgeItems) until dims[0]) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        Tensor.cat(vs, 0)
    } else {
        val vs = (0 until dims[0])
            .map { i -> getSummarizedData(t.get(i), edgeItems) }
        Tensor.cat(vs, 0)
    }
}

public fun Tensor.fmtDt(sb: StringBuilder) {
    val loc = device().location()
    val deviceStr = when (loc) {
        is DeviceLocation.Cpu -> ""
        is DeviceLocation.Cuda -> ", cuda:${loc.gpuId}"
        is DeviceLocation.Metal -> ", metal:${loc.gpuId}"
    }

    sb.append("Tensor[")
    val dims = dims()
    when {
        dims.isEmpty() -> {
            val v = toScalarF64()
            if (!v.isNaN() || v.isNaN()) {
                sb.append(v)
            }
        }
        dims.size == 1 && dims[0] < 10 -> {
            val vs = toVec1F64()
            if (vs.isNotEmpty()) {
                for ((i, v) in vs.withIndex()) {
                    if (i > 0) {
                        sb.append(", ")
                    }
                    sb.append(v)
                }
            }
        }
        else -> {
            sb.append("dims ")
            for ((i, d) in dims.withIndex()) {
                if (i > 0) {
                    sb.append(", ")
                }
                sb.append(d)
            }
        }
    }
    sb.append("; ${dtype().asStr()}$deviceStr]")
}

public fun Tensor.fmt(sb: StringBuilder) {
    when (dtype()) {
        DType.U8 -> fmtDt(sb)
        DType.U32 -> fmtDt(sb)
        DType.I16 -> fmtDt(sb)
        DType.I32 -> fmtDt(sb)
        DType.I64 -> fmtDt(sb)
        DType.BF16 -> fmtDt(sb)
        DType.F16 -> fmtDt(sb)
        DType.F32 -> fmtDt(sb)
        DType.F64 -> fmtDt(sb)
        DType.F8E4M3 -> fmtDt(sb)
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            sb.append("Tensor[${shape()}; dtype=${dtype().asStr()}, unsupported dummy type]")
        }
    }
}

public fun Tensor.fmt(): String {
    val po = printOptions()
    val summarize = elemCount() > po.threshold
    val toDisplay = if (summarize) {
        val v = getSummarizedData(this, po.edgeItems)
        when (v.rank()) {
            -1 -> v
            else -> v
        }
    } else {
        this
    }

    val sb = StringBuilder()
    when (dtype()) {
        DType.U8 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.U32 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.I16 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.I32 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.I64 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.BF16 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            if (tf.precision >= 0) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(this, 1, maxW, summarize, po, sb)
                sb.append('\n')
            }
        }
        DType.F16 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            if (tf.precision >= 0) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(this, 1, maxW, summarize, po, sb)
                sb.append('\n')
            }
        }
        DType.F64 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            if (tf.precision >= 0) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(this, 1, maxW, summarize, po, sb)
                sb.append('\n')
            }
        }
        DType.F32 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            if (tf.precision >= 0) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(this, 1, maxW, summarize, po, sb)
                sb.append('\n')
            }
        }
        DType.F8E4M3 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            if (tf.precision >= 0) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(this, 1, maxW, summarize, po, sb)
                sb.append('\n')
            }
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            sb.append("Dummy type ${dtype().asStr()} (not supported for display)\n")
        }
    }

    val loc = device().location()
    val deviceStr = when (loc) {
        is DeviceLocation.Cpu -> ""
        is DeviceLocation.Cuda -> ", cuda:${loc.gpuId}"
        is DeviceLocation.Metal -> ", metal:${loc.gpuId}"
    }

    sb.append("Tensor[${dims()}, ${dtype().asStr()}$deviceStr]")
    return sb.toString()
}
