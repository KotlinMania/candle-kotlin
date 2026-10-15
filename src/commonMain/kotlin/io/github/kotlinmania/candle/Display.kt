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

private var globalPrintOptions: PrinterOptions = PrinterOptions.constDefault()

/**
 * Returns the current global printer options for tensor formatting.
 */
public fun printOptions(): PrinterOptions = globalPrintOptions

public fun setPrintOptions(options: PrinterOptions) {
    globalPrintOptions = options
}

public fun setPrintOptionsDefault() {
    globalPrintOptions = PrinterOptions.constDefault()
}

public fun setPrintOptionsShort() {
    globalPrintOptions = PrinterOptions(
        precision = 2,
        threshold = 1000,
        edgeItems = 2,
        lineWidth = 80,
        sciMode = null,
    )
}

public fun setPrintOptionsFull() {
    globalPrintOptions = PrinterOptions(
        precision = 4,
        threshold = Int.MAX_VALUE,
        edgeItems = 3,
        lineWidth = 80,
        sciMode = null,
    )
}

public fun setLineWidth(lineWidth: Int) {
    globalPrintOptions.lineWidth = lineWidth
}

public fun setPrecision(precision: Int) {
    globalPrintOptions.precision = precision
}

public fun setEdgeItems(edgeItems: Int) {
    globalPrintOptions.edgeItems = edgeItems
}

public fun setThreshold(threshold: Int) {
    globalPrintOptions.threshold = threshold
}

public fun setSciMode(sciMode: Boolean?) {
    globalPrintOptions.sciMode = sciMode
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

public interface TensorFormatter<T> {
    public fun fmt(v: T, maxW: Int, sb: StringBuilder)

    public fun maxWidth(toDisplay: Tensor): Int

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
    )
}

public class FloatFormatter(
    public var intMode: Boolean,
    public var sciMode: Boolean,
    public var precision: Int,
) : TensorFormatter<Double> {
    override fun fmt(v: Double, maxW: Int, sb: StringBuilder) {
        val formatted = if (sciMode) {
            formatScientific(v, precision)
        } else if (intMode) {
            if (!v.isNaN() && !v.isInfinite()) {
                "${v.toLong()}."
            } else {
                v.toString()
            }
        } else {
            formatDecimal(v, precision)
        }
        val padding = maxOf(0, maxW - formatted.length)
        for (i in 0 until padding) sb.append(' ')
        sb.append(formatted)
    }

    override fun maxWidth(toDisplay: Tensor): Int {
        var maxW = 1
        val vs = toDisplay.flattenAll().toVec1F64()
        for (v in vs) {
            val sb = StringBuilder()
            fmt(v, 1, sb)
            maxW = maxOf(maxW, sb.length)
        }
        return maxW
    }

    override fun fmtTensor(
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
                fmt(t.toScalarF64(), maxW, sb)
            }
            dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
                val start = t.narrow(0, 0, edgeItems).toVec1F64()
                for (v in start) {
                    fmt(v, maxW, sb)
                    sb.append(", ")
                }
                sb.append("...")
                val end = t.narrow(0, dims[0] - edgeItems, edgeItems).toVec1F64()
                for (v in end) {
                    sb.append(", ")
                    fmt(v, maxW, sb)
                }
            }
            dims.size == 1 -> {
                val elementsPerLine = maxOf(1, po.lineWidth / (maxW + 2))
                val vs = t.toVec1F64()
                for (i in vs.indices) {
                    if (i > 0) {
                        if (i % elementsPerLine == 0) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        } else {
                            sb.append(", ")
                        }
                    }
                    fmt(vs[i], maxW, sb)
                }
            }
            else -> {
                if (summarize && dims[0] > 2 * edgeItems) {
                    for (i in 0 until edgeItems) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                    sb.append("...")
                    writeNewlineIndent(indent, sb)
                    for (i in (dims[0] - edgeItems) until dims[0]) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
                        if (i + 1 != dims[0]) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        }
                    }
                } else {
                    for (i in 0 until dims[0]) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
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

    public companion object {
        public fun new(t: Tensor, po: PrinterOptions): FloatFormatter {
            var intMode = true
            var sciMode = false

            val values = t.flattenAll().toVec1F64().filter { !it.isNaN() && !it.isInfinite() && it != 0.0 }
            if (values.isNotEmpty()) {
                var nonzeroFiniteMin = Double.MAX_VALUE
                var nonzeroFiniteMax = -Double.MAX_VALUE
                for (v in values) {
                    val a = abs(v)
                    if (a < nonzeroFiniteMin) nonzeroFiniteMin = a
                    if (a > nonzeroFiniteMax) nonzeroFiniteMax = a
                }
                for (value in values) {
                    if (ceil(value) != value) {
                        intMode = false
                        break
                    }
                }
                val v1 = 1000.0
                val v2 = 1e8
                val v3 = 1e-4
                sciMode = nonzeroFiniteMax / nonzeroFiniteMin > v1 || nonzeroFiniteMax > v2 || nonzeroFiniteMin < v3
            }

            val poSciMode = po.sciMode
            if (poSciMode != null) {
                sciMode = poSciMode
            }

            return FloatFormatter(
                intMode = intMode,
                sciMode = sciMode,
                precision = po.precision,
            )
        }

        private fun formatScientific(v: Double, precision: Int): String {
            if (v.isNaN() || v.isInfinite() || precision < 0) return v.toString()
            return v.toString()
        }

        private fun formatDecimal(v: Double, precision: Int): String {
            if (v.isNaN() || v.isInfinite()) return v.toString()
            val factor = pow10(precision)
            val rounded = kotlin.math.round(v * factor) / factor
            val s = rounded.toString()
            val dot = s.indexOf('.')
            return if (dot == -1) {
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

        private fun pow10(n: Int): Double {
            var res = 1.0
            for (i in 0 until n) res *= 10.0
            return res
        }
    }
}

public class IntFormatter : TensorFormatter<Long> {
    override fun fmt(v: Long, maxW: Int, sb: StringBuilder) {
        val s = v.toString()
        val padding = maxOf(0, maxW - s.length)
        for (i in 0 until padding) sb.append(' ')
        sb.append(s)
    }

    override fun maxWidth(toDisplay: Tensor): Int {
        var maxW = 1
        val vs = toDisplay.flattenAll().toVec1I64()
        for (v in vs) {
            val sb = StringBuilder()
            fmt(v, 1, sb)
            maxW = maxOf(maxW, sb.length)
        }
        return maxW
    }

    override fun fmtTensor(
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
                fmt(t.toScalarI64(), maxW, sb)
            }
            dims.size == 1 && summarize && dims[0] > 2 * edgeItems -> {
                val start = t.narrow(0, 0, edgeItems).toVec1I64()
                for (v in start) {
                    fmt(v, maxW, sb)
                    sb.append(", ")
                }
                sb.append("...")
                val end = t.narrow(0, dims[0] - edgeItems, edgeItems).toVec1I64()
                for (v in end) {
                    sb.append(", ")
                    fmt(v, maxW, sb)
                }
            }
            dims.size == 1 -> {
                val elementsPerLine = maxOf(1, po.lineWidth / (maxW + 2))
                val vs = t.toVec1I64()
                for (i in vs.indices) {
                    if (i > 0) {
                        if (i % elementsPerLine == 0) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        } else {
                            sb.append(", ")
                        }
                    }
                    fmt(vs[i], maxW, sb)
                }
            }
            else -> {
                if (summarize && dims[0] > 2 * edgeItems) {
                    for (i in 0 until edgeItems) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
                        sb.append(',')
                        writeNewlineIndent(indent, sb)
                    }
                    sb.append("...")
                    writeNewlineIndent(indent, sb)
                    for (i in (dims[0] - edgeItems) until dims[0]) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
                        if (i + 1 != dims[0]) {
                            sb.append(',')
                            writeNewlineIndent(indent, sb)
                        }
                    }
                } else {
                    for (i in 0 until dims[0]) {
                        fmtTensor(t.get(i), indent + 1, maxW, summarize, po, sb)
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
        val vs = ArrayList<Tensor>()
        for (i in 0 until edgeItems) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        for (i in (dims[0] - edgeItems) until dims[0]) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        Tensor.cat(vs, 0)
    } else {
        val vs = ArrayList<Tensor>()
        for (i in 0 until dims[0]) {
            vs.add(getSummarizedData(t.get(i), edgeItems))
        }
        Tensor.cat(vs, 0)
    }
}

public fun Tensor.fmtDt(sb: StringBuilder) {
    val deviceStr = when (val loc = device().location()) {
        is DeviceLocation.Cpu -> ""
        is DeviceLocation.Cuda -> ", cuda:${loc.gpuId}"
        is DeviceLocation.Metal -> ", metal:${loc.gpuId}"
    }

    sb.append("Tensor[")
    val dims = dims()
    when {
        dims.isEmpty() -> {
            sb.append(toScalarF64())
        }
        dims.size == 1 && dims[0] < 10 -> {
            val vs = toVec1F64()
            for (i in vs.indices) {
                if (i > 0) sb.append(", ")
                sb.append(vs[i])
            }
        }
        else -> {
            sb.append("dims ")
            for (i in dims.indices) {
                if (i > 0) sb.append(", ")
                sb.append(dims[i])
            }
        }
    }
    sb.append("; ${dtype().asStr()}$deviceStr]")
}

public fun Tensor.fmt(sb: StringBuilder) {
    when (dtype()) {
        DType.U8, DType.U32, DType.I16, DType.I32, DType.I64,
        DType.BF16, DType.F16, DType.F32, DType.F64, DType.F8E4M3,
        -> fmtDt(sb)
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            sb.append("Tensor[${shape()}; dtype=${dtype().asStr()}, unsupported dummy type]")
        }
    }
}

public fun Tensor.fmt(): String = displayText()

public fun Tensor.displayText(): String {
    val po = printOptions()
    val summarize = elemCount() > po.threshold
    val toDisplay = if (summarize) {
        try {
            getSummarizedData(this, po.edgeItems)
        } catch (_: Exception) {
            this
        }
    } else {
        this
    }

    val sb = StringBuilder()
    when (dtype()) {
        DType.U8, DType.U32, DType.I16, DType.I32, DType.I64 -> {
            val tf = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.BF16, DType.F16, DType.F32, DType.F64, DType.F8E4M3 -> {
            val tf = FloatFormatter.new(toDisplay, po)
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(this, 1, maxW, summarize, po, sb)
            sb.append('\n')
        }
        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            sb.append("Dummy type ${dtype().asStr()} (not supported for display)\n")
        }
    }

    val deviceStr = when (val loc = device().location()) {
        is DeviceLocation.Cpu -> ""
        is DeviceLocation.Cuda -> ", cuda:${loc.gpuId}"
        is DeviceLocation.Metal -> ", metal:${loc.gpuId}"
    }

    sb.append("Tensor[${dims()}, ${dtype().asStr()}$deviceStr]")
    return sb.toString()
}
