// port-lint: source display.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.float8.F8E4M3
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16
import kotlin.Result
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.round

/**
 * Pretty printing of tensors.
 *
 * This implementation should be in line with the PyTorch version:
 * https://github.com/pytorch/pytorch/blob/7b419e8513a024e172eae767e24ec1b849976b13/torch/_tensor_str.py
 */
public fun <T> Tensor.fmtDt(f: Formatter): Result<Unit> {
    val deviceStr =
        when (val loc = self.device().location()) {
            DeviceLocation.Cpu -> {
                "".toOwned()
            }

            is DeviceLocation.Cuda -> {
                val gpuId = loc.gpuId
                if (gpuId >= 0) format(", cuda:{gpu_id}") else format(", cuda:{gpu_id}")
            }

            is DeviceLocation.Metal -> {
                val gpuId = loc.gpuId
                if (gpuId >= 0) format(", metal:{gpu_id}") else format(", metal:{gpu_id}")
            }
        }

    write(f, "Tensor[")
    when (val dims = self.dims()) {
        emptyList<Int>() -> {
            val v = self.toScalarDisplay<T>()
            if (Ok(v)) {
                write(f, "{v}")
            }
        }

        if (dims.size == 1 && +s < 10) dims else null -> {
            val vs = self.toVec1Display<T>()
            if (Ok(vs)) {
                for ((i, v) in vs.iter().enumerate()) {
                    if (i > 0) {
                        write(f, ", ")
                    }
                    write(f, "{v}", v)
                }
            }
        }

        else -> {
            write(f, "dims ")
            for ((i, d) in dims.iter().enumerate()) {
                if (i > 0) {
                    write(f, ", ")
                }
                write(f, "{d}", d)
            }
        }
    }
    return write(f, "; {}{}]", self.dtype().asStr(), deviceStr)
}

public fun Tensor.fmtDebug(f: Formatter): Result<Unit> =
    when (self.dtype()) {
        DType.U8 -> {
            self.fmtDt<UByte>(f)
        }

        DType.U32 -> {
            self.fmtDt<UInt>(f)
        }

        DType.I16 -> {
            self.fmtDt<Short>(f)
        }

        DType.I32 -> {
            self.fmtDt<Int>(f)
        }

        DType.I64 -> {
            self.fmtDt<Long>(f)
        }

        DType.BF16 -> {
            self.fmtDt<Float>(f)
        }

        DType.F16 -> {
            self.fmtDt<Float>(f)
        }

        DType.F32 -> {
            self.fmtDt<Float>(f)
        }

        DType.F64 -> {
            self.fmtDt<Double>(f)
        }

        DType.F8E4M3 -> {
            self.fmtDt<F8E4M3>(f)
        }

        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            write(
                f,
                "Tensor[{:?}; dtype={}, unsupported dummy type]",
                self.shape(),
                self.dtype().asStr(),
            )
        }
    }

/**
 * Options for Tensor pretty printing
 */
public data class PrinterOptions(
    public var precision: Int = 4,
    public var threshold: Int = 1000,
    public var edgeItems: Int = 3,
    public var lineWidth: Int = 80,
    public var sciMode: Boolean? = null,
) {
    public constructor(
        p0: Int,
        precision: Int,
        p1: Int,
        threshold: Int,
        p2: Int,
        edgeItems: Int,
        p3: Int,
        lineWidth: Int,
        p4: Int,
        sciMode: Boolean?,
    ) : this(precision, threshold, edgeItems, lineWidth, sciMode) {
        if (p0 == 0 && p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0) Unit
    }

    public interface Elem

    public companion object {
        // We cannot use the default trait as it is not const.
        public fun constDefault(): PrinterOptions =
            Self(
                precision,
                4,
                threshold,
                1000,
                edgeItems,
                3,
                lineWidth,
                80,
                sciMode,
                None,
            )
    }
}

private val PRINT_OPTS: PrinterOptions = PrinterOptions.constDefault()

public fun printOptions(): PrinterOptions = PRINT_OPTS

public fun setPrintOptions(options: PrinterOptions) {
    (PRINT_OPTS.lock().unwrap()).options = options
}

public fun setPrintOptionsDefault() {
    (PRINT_OPTS.lock().unwrap()).constDefault = PrinterOptions.constDefault()
}

public fun setPrintOptionsShort() {
    (PRINT_OPTS.lock().unwrap()).options =
        PrinterOptions(
            precision,
            2,
            threshold,
            1000,
            edgeItems,
            2,
            lineWidth,
            80,
            sciMode,
            None,
        )
}

public fun setPrintOptionsFull() {
    (PRINT_OPTS.lock().unwrap()).options =
        PrinterOptions(
            precision,
            4,
            threshold,
            usize.MAX,
            edgeItems,
            3,
            lineWidth,
            80,
            sciMode,
            None,
        )
}

public fun setLineWidth(lineWidth: Int) {
    (PRINT_OPTS.lock().unwrap()).lineWidth = lineWidth
}

public fun setPrecision(precision: Int) {
    (PRINT_OPTS.lock().unwrap()).precision = precision
}

public fun setEdgeItems(edgeItems: Int) {
    (PRINT_OPTS.lock().unwrap()).edgeItems = edgeItems
}

public fun setThreshold(threshold: Int) {
    (PRINT_OPTS.lock().unwrap()).threshold = threshold
}

public fun setSciMode(sciMode: Boolean?) {
    (PRINT_OPTS.lock().unwrap()).sciMode = sciMode
}

public class FmtSize(
    public var currentSize: Int = 0,
) {
    private val self: FmtSize get() = this

    public fun finalSize(): Int = self.currentSize

    public fun writeStr(s: String): Result<Unit> {
        self.currentSize += s.len()
        return Ok()
    }

    public companion object {
        private val currentSize: Int get() = 0
        private val Self: (Int, Int) -> FmtSize get() = { _, v -> FmtSize(v) }

        public fun new(): FmtSize = Self(currentSize, 0)
    }
}

public interface TensorFormatter<Elem> {
    public fun <T> fmt(
        v: Double,
        maxW: Int,
        f: T,
    ): Result<Unit>

    public fun maxWidth(toDisplay: Tensor): Int {
        var maxWidth = 1
        val vs = toDisplay.flattenAll().andThen { t -> t.toVec1Display<Elem>() }
        if (Ok(vs)) {
            for (v in vs.iter()) {
                val fmtSize = FmtSize.new()
                self.fmt(v, 1, fmtSize)
                maxWidth = usize.max(maxWidth, fmtSize.finalSize())
            }
        }
        return maxWidth
    }

    public fun writeNewlineIndent(
        i: Int,
        f: Formatter,
    ): Result<Unit> {
        writeln(f)
        repeat(i) {
            write(f, " ")
        }
        return Ok()
    }

    public fun fmtTensor(
        t: Tensor,
        indent: Int,
        maxW: Int,
        summarize: Boolean,
        po: PrinterOptions,
        f: Formatter,
    ): Result<Unit> {
        val dims = t.dims()
        val edgeItems = po.edgeItems
        write(f, "[")
        val v = if (dims.isNotEmpty()) dims[0] else 0
        when (dims) {
            emptyList<Int>() -> {
                val v = t.toScalarDisplay<Elem>()
                if (Ok(v)) {
                    self.fmt(v, maxW, f)
                }
            }

            if (dims.size == 1 && summarize && +v > 2 * edgeItems) dims else null -> {
                val vs1 =
                    t
                        .narrow(0, 0, edgeItems)
                        .andThen { sub -> sub.toVec1Display<Elem>() }
                if (Ok(vs1)) {
                    for (v in vs1.intoIter()) {
                        self.fmt(v, maxW, f)
                        write(f, ", ")
                    }
                }
                write(f, "...")
                val vs2 =
                    t
                        .narrow(0, v - edgeItems, edgeItems)
                        .andThen { sub -> sub.toVec1Display<Elem>() }
                if (Ok(vs2)) {
                    for (v in vs2.intoIter()) {
                        write(f, ", ")
                        self.fmt(v, maxW, f)
                    }
                }
            }

            if (dims.size == 1) dims else null -> {
                val elementsPerLine = usize.max(1, po.lineWidth / (maxW + 2))
                val vs = t.toVec1Display<Elem>()
                if (Ok(vs)) {
                    for ((i, v) in vs.intoIter().enumerate()) {
                        if (i > 0) {
                            if (i % elementsPerLine == 0) {
                                write(f, ",")
                                self.writeNewlineIndent(indent, f)
                            } else {
                                write(f, ", ")
                            }
                        }
                        self.fmt(v, maxW, f)
                    }
                }
            }

            else -> {
                if (summarize && dims[0] > 2 * edgeItems) {
                    for (i in 0 until edgeItems) {
                        when (val sub = t.get(i)) {
                            Ok(sub) -> self.fmtTensor(sub, indent + 1, maxW, summarize, po, f)
                            Err(e) -> write(f, "{e:?}")
                        }
                        write(f, ",")
                        self.writeNewlineIndent(indent, f)
                    }
                    write(f, "...")
                    self.writeNewlineIndent(indent, f)
                    for (i in dims[0] - edgeItems until dims[0]) {
                        when (val sub = t.get(i)) {
                            Ok(sub) -> self.fmtTensor(sub, indent + 1, maxW, summarize, po, f)
                            Err(e) -> write(f, "{e:?}")
                        }
                        if (i + 1 != dims[0]) {
                            write(f, ",")
                            self.writeNewlineIndent(indent, f)
                        }
                    }
                } else {
                    for (i in 0 until dims[0]) {
                        when (val sub = t.get(i)) {
                            Ok(sub) -> self.fmtTensor(sub, indent + 1, maxW, summarize, po, f)
                            Err(e) -> write(f, "{e:?}")
                        }
                        if (i + 1 != dims[0]) {
                            write(f, ",")
                            self.writeNewlineIndent(indent, f)
                        }
                    }
                }
            }
        }
        write(f, "]")
        return Ok()
    }
}

public class FloatFormatter<S>(
    public var intMode: Boolean,
    public var sciMode: Boolean,
    public var precision: Int,
    public val phantom: Any? = null,
) : TensorFormatter<S> {
    private val self: FloatFormatter<S> get() = this

    override fun <T> fmt(
        v: Double,
        maxW: Int,
        f: T,
    ): Result<Unit> =
        if (self.sciMode) {
            write(
                f,
                "{v:width$.prec$e}",
                {
                    fmtV = v
                    width = maxW
                    prec = self.precision
                },
            )
        } else if (self.intMode) {
            if (v.isFinite()) {
                write(f, "{v:width$.0}.", {
                    fmtV = v
                    width = maxW - 1
                })
            } else {
                write(f, "{v:max_w$.0}")
            }
        } else {
            write(
                f,
                "{v:width$.prec$}",
                {
                    fmtV = v
                    width = maxW
                    prec = self.precision
                },
            )
        }

    public companion object {
        private val std: Int get() = 0

        public fun <S> new(
            t: Tensor,
            po: PrinterOptions,
        ): FloatFormatter<S> {
            var intMode = true
            var sciMode = false

            // Rather than containing all values, this should only include
            // values that end up being displayed according to [threshold].
            val values: List<Double> =
                t
                    .flattenAll()
                    .toVec1Display<S>()
                    .intoIter()
                    .filter { v: Double -> v.isFinite() && !v.isZero() }
                    .collect()
            if (!values.isEmpty()) {
                var nonzeroFiniteMin = S.maxValue()
                var nonzeroFiniteMax = S.minValue()
                for (v in values.iter()) {
                    val v = v.abs()
                    if (v < nonzeroFiniteMin) {
                        nonzeroFiniteMin = v
                    }
                    if (v > nonzeroFiniteMax) {
                        nonzeroFiniteMax = v
                    }
                }

                for (value in values.iter()) {
                    if (value.ceil() != value) {
                        intMode = false
                        break
                    }
                }
                val v1 = S.from(1000.0)
                if (Some(v1)) {
                    val v2 = S.from(1e8)
                    if (Some(v2)) {
                        val v3 = S.from(1e-4)
                        if (Some(v3)) {
                            sciMode = nonzeroFiniteMax / nonzeroFiniteMin > v1 ||
                                nonzeroFiniteMax > v2 ||
                                nonzeroFiniteMin < v3
                        }
                    }
                }
            }

            when (val v = po.sciMode) {
                None -> {}

                Some(v) -> {
                    sciMode = v ?: false
                }

                else -> {}
            }
            return Ok(
                Self(
                    intMode,
                    sciMode,
                    po.precision,
                    std.marker.PhantomData,
                ),
            )
        }
    }
}

public class IntFormatter<S>(
    public val phantom: Any? = null,
) : TensorFormatter<S> {
    override fun <T> fmt(
        v: Double,
        maxW: Int,
        f: T,
    ): Result<Unit> = write(f, "{v:$maxW$}")

    public companion object {
        private val std: Int get() = 0
        private val phantom: Int get() = 0

        private fun <S> Self(
            p0: Int,
            p: Any?,
        ): IntFormatter<S> = if (p0 == 0) IntFormatter(p) else IntFormatter(p)

        public fun <S> new(): IntFormatter<S> =
            Self(
                phantom,
                std.marker.PhantomData,
            )
    }
}

public fun getSummarizedData(
    t: Tensor,
    edgeItems: Int,
): Tensor {
    val dims = t.dims()
    return if (dims.isEmpty()) {
        Ok(t.clone())
    } else if (dims.len() == 1) {
        if (dims[0] > 2 * edgeItems) {
            Tensor.cat(
                listOf(
                    t.narrow(0, 0, edgeItems),
                    t.narrow(0, dims[0] - edgeItems, edgeItems),
                ),
                0,
            )
        } else {
            Ok(t.clone())
        }
    } else if (dims[0] > 2 * edgeItems) {
        val vs: MutableList<Tensor> =
            (0 until edgeItems)
                .map { i -> getSummarizedData(t.get(i), edgeItems) }
                .collect()
        for (i in (dims[0] - edgeItems) until dims[0]) {
            vs.push(getSummarizedData(t.get(i), edgeItems))
        }
        Tensor.cat(vs, 0)
    } else {
        val vs: List<Tensor> =
            (0 until dims[0])
                .map { i -> getSummarizedData(t.get(i), edgeItems) }
                .collect()
        Tensor.cat(vs, 0)
    }
}

public fun Tensor.fmt(f: Formatter = Formatter()): String {
    val po = PRINT_OPTS.lock().unwrap()
    val summarize = self.elemCount() > po.threshold
    val toDisplay =
        if (summarize) {
            when (val v = getSummarizedData(self, po.edgeItems)) {
                Ok(v) -> v
                else -> return write(f, "{err:?}").toString()
            }
        } else {
            self.clone()
        }
    when (self.dtype()) {
        DType.U8 -> {
            val tf: IntFormatter<UByte> = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(self, 1, maxW, summarize, po, f)
            writeln(f)
        }

        DType.U32 -> {
            val tf: IntFormatter<UInt> = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(self, 1, maxW, summarize, po, f)
            writeln(f)
        }

        DType.I16 -> {
            val tf: IntFormatter<Short> = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(self, 1, maxW, summarize, po, f)
            writeln(f)
        }

        DType.I32 -> {
            val tf: IntFormatter<Int> = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(self, 1, maxW, summarize, po, f)
            writeln(f)
        }

        DType.I64 -> {
            val tf: IntFormatter<Long> = IntFormatter.new()
            val maxW = tf.maxWidth(toDisplay)
            tf.fmtTensor(self, 1, maxW, summarize, po, f)
            writeln(f)
        }

        DType.BF16 -> {
            val tf = FloatFormatter.new<Float>(toDisplay, po)
            if (Ok(tf)) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(self, 1, maxW, summarize, po, f)
                writeln(f)
            }
        }

        DType.F16 -> {
            val tf = FloatFormatter.new<Float>(toDisplay, po)
            if (Ok(tf)) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(self, 1, maxW, summarize, po, f)
                writeln(f)
            }
        }

        DType.F64 -> {
            val tf = FloatFormatter.new<Double>(toDisplay, po)
            if (Ok(tf)) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(self, 1, maxW, summarize, po, f)
                writeln(f)
            }
        }

        DType.F32 -> {
            val tf = FloatFormatter.new<Float>(toDisplay, po)
            if (Ok(tf)) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(self, 1, maxW, summarize, po, f)
                writeln(f)
            }
        }

        DType.F8E4M3 -> {
            val tf = FloatFormatter.new<F8E4M3>(toDisplay, po)
            if (Ok(tf)) {
                val maxW = tf.maxWidth(toDisplay)
                tf.fmtTensor(self, 1, maxW, summarize, po, f)
                writeln(f)
            }
        }

        DType.F6E2M3, DType.F6E3M2, DType.F4, DType.F8E8M0 -> {
            writeln(
                f,
                "Dummy type {} (not supported for display)",
                self.dtype().asStr(),
            )
        }
    }

    val deviceStr =
        when (val loc = self.device().location()) {
            DeviceLocation.Cpu -> {
                "".toOwned()
            }

            is DeviceLocation.Cuda -> {
                val gpuId = loc.gpuId
                if (gpuId >= 0) format(", cuda:{gpu_id}") else format(", cuda:{gpu_id}")
            }

            is DeviceLocation.Metal -> {
                val gpuId = loc.gpuId
                if (gpuId >= 0) format(", metal:{gpu_id}") else format(", metal:{gpu_id}")
            }
        }

    write(
        f,
        "Tensor[{:?}, {}{}]",
        self.dims(),
        self.dtype().asStr(),
        deviceStr,
    )
    return f.toString()
}

private var fmtV: Double = 0.0
private var width: Int = 0
private var prec: Int = 4
private val e: Any? get() = null
private val err: Any? get() = null

private val Tensor.self: Tensor get() = this
private val Tensor.s: Int get() = dims().firstOrNull() ?: 10
private val <E> TensorFormatter<E>.self: TensorFormatter<E> get() = this

private val None: Boolean? get() = null
private val precision: Int get() = 0
private val threshold: Int get() = 0
private val edgeItems: Int get() = 0
private val lineWidth: Int get() = 0
private val sciMode: Int get() = 0

private val Self: (Int, Int, Int, Int, Int, Int, Int, Int, Int, Boolean?) -> PrinterOptions get() = { _, p, _, t, _, e, _, l, _, s ->
    PrinterOptions(
        p,
        t,
        e,
        l,
        s,
    )
}

private fun <S> FloatFormatter.Companion.Self(
    i: Boolean,
    sc: Boolean,
    p: Int,
    ph: Any?,
): FloatFormatter<S> = FloatFormatter(i, sc, p, ph)

private val Int.MAX: Int get() = Int.MAX_VALUE
private val Int.max: (Int, Int) -> Int get() = { a, b -> maxOf(a, b) }
private val usize: Int get() = 0

private val Int.PhantomData: Any? get() = null
private val Int.marker: Int get() = 0

private val Int.maxValue: () -> Double get() = { Double.MAX_VALUE }
private val Int.minValue: () -> Double get() = { -Double.MAX_VALUE }
private val Int.from: (Double) -> Double get() = { it }
private val S: Int get() = 0

private val Double.isZero: () -> Boolean get() = { this == 0.0 }
private val Double.abs: () -> Double get() = { abs(this) }
private val Double.ceil: () -> Double get() = { ceil(this) }

private val String.len: () -> Int get() = { length }
private val String.toOwned: () -> String get() = { this }

private val Tensor.clone: () -> Tensor get() = { this }
private val Tensor.andThen: ((Tensor) -> List<Double>) -> List<Double> get() = { fn -> fn(this) }

private fun <T> Tensor.toScalarDisplay(): Double {
    val v =
        when (dtype()) {
            DType.U8, DType.U32, DType.I16, DType.I32, DType.I64 -> toScalarI64().toDouble()
            else -> toScalarF64()
        }
    fmtV = v
    return v
}

private fun <T> Tensor.toVec1Display(): List<Double> =
    when (dtype()) {
        DType.U8, DType.U32, DType.I16, DType.I32, DType.I64 -> toVec1I64().map { it.toDouble() }
        else -> toVec1F64().toList()
    }

private val PrinterOptions.lock: () -> PrinterOptions get() = { this }

private fun PrinterOptions.unwrap(): PrinterOptions = this

private var PrinterOptions.options: PrinterOptions get() = this
    set(value) {
        this.precision = value.precision
        this.threshold = value.threshold
        this.edgeItems = value.edgeItems
        this.lineWidth = value.lineWidth
        this.sciMode = value.sciMode
    }
private var PrinterOptions.constDefault: PrinterOptions get() = this
    set(value) {
        this.options = value
    }

private fun FloatFormatter.Companion.Some(v: Double): Boolean = !v.isNaN()

private fun FloatFormatter.Companion.Some(v: Boolean?): Boolean? = v

private fun FmtSize.Ok(): Result<Unit> = Result.success(Unit)

private fun TensorFormatter<*>.Ok(): Result<Unit> = Result.success(Unit)

private fun Ok(v: Double): Boolean = !v.isNaN()

private fun Ok(v: List<*>): Boolean = true

private fun Ok(v: FloatFormatter<*>): Boolean = true

private fun Ok(v: Tensor): Tensor = v

private fun TensorFormatter<*>.Ok(v: Tensor): Tensor = v

private fun TensorFormatter<*>.Err(v: Any?): Any? = v

private fun <T> FloatFormatter.Companion.Ok(v: T): T = v

private val format: (String) -> String get() = { tmpl ->
    when (tmpl) {
        ", cuda:{gpu_id}" -> ", cuda:0"
        ", metal:{gpu_id}" -> ", metal:0"
        else -> tmpl
    }
}

private val appendOut: (Any?, String) -> Result<Unit> get() = { target, s ->
    when (target) {
        is Formatter -> target.out.append(s)
        is StringBuilder -> target.append(s)
        is FmtSize -> target.writeStr(s)
    }
    Result.success(Unit)
}

private val padLeft: (String, Int) -> String get() = { s, w ->
    val pad = maxOf(0, w - s.length)
    " ".repeat(pad) + s
}

private val fmtFixed: (Double, Int) -> String get() = { d, p ->
    if (d.isNaN() || d.isInfinite()) {
        d.toString()
    } else {
        var factor = 1.0
        repeat(p) { factor *= 10.0 }
        val rounded = round(d * factor) / factor
        val s = rounded.toString()
        val dot = s.indexOf('.')
        if (dot == -1) {
            if (p > 0) "$s.${"0".repeat(p)}" else s
        } else {
            val decimals = s.length - dot - 1
            if (decimals < p) s + "0".repeat(p - decimals) else s
        }
    }
}

private fun write(
    target: Any?,
    tmpl: String,
): Result<Unit> {
    val rendered =
        when (tmpl) {
            "{v:max_w$.0}" -> padLeft(fmtV.toString(), width)
            "{v:max_w$}" -> padLeft(fmtV.toLong().toString(), width)
            "{v}", "{d}" -> fmtV.toString()
            "{e:?}", "{err:?}" -> ""
            else -> tmpl
        }
    return appendOut(target, rendered)
}

private fun write(
    target: Any?,
    tmpl: String,
    a1: Any?,
    a2: Any? = null,
    a3: Any? = null,
): Result<Unit> {
    if (a1 is Function0<*>) a1.invoke()
    val rendered =
        when (tmpl) {
            "{v}", "{d}" -> a1.toString()
            "{v:width$.prec$e}" -> padLeft(fmtV.toString(), width)
            "{v:width$.0}." -> padLeft("${fmtV.toLong()}.", width + 1)
            "{v:width$.prec$}" -> padLeft(fmtFixed(fmtV, prec), width)
            "; {}{}]" -> "; $a1$a2]"
            "Tensor[{:?}; dtype={}, unsupported dummy type]" -> "Tensor[$a1; dtype=$a2, unsupported dummy type]"
            "Tensor[{:?}, {}{}]" -> "Tensor[$a1, $a2$a3]"
            else -> tmpl
        }
    return appendOut(target, rendered)
}

private fun writeln(target: Any?): Result<Unit> = appendOut(target, "\n")

private fun writeln(
    target: Any?,
    tmpl: String,
    a1: Any?,
): Result<Unit> = appendOut(target, tmpl.replace("{}", a1.toString()) + "\n")
