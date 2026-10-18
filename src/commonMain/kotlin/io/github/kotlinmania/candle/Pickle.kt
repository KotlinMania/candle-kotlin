// port-lint: source pickle.rs
/**
 * Just enough pickle support to be able to read PyTorch checkpoints.
 */
// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.
package io.github.kotlinmania.candle

import io.github.kotlinmania.byteorder.BigEndian
import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.CandleException
import io.github.kotlinmania.candle.CandleResult.Err
import io.github.kotlinmania.candle.CandleResult.Ok
import io.github.kotlinmania.candle.Zip.ZipArchive
import io.github.kotlinmania.half.BF16
import io.github.kotlinmania.half.F16

private fun Err(v: UByte): CandleResult<Nothing> = Err(CandleException.Msg("unknown op-code $v"))

private fun Err(obj: Object): CandleResult<Nothing> = Err(from(obj))

private fun Object.clone(): Object = this

private fun String.stripSuffix(suffix: String): String? = if (this.endsWith(suffix)) this.removeSuffix(suffix) else null

private const val VERBOSE: Boolean = false

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
public enum class OpCode(
    public val code: UByte,
) {
    // https://github.com/python/cpython/blob/ed25f097160b5cbb0c9a1f9a746d2f1bbc96515a/Lib/pickletools.py#L2123
    Proto(0x80u),
    Global('c'.code.toUByte()),
    BinPut('q'.code.toUByte()),
    LongBinPut('r'.code.toUByte()),
    EmptyTuple(')'.code.toUByte()),
    Reduce('R'.code.toUByte()),
    Mark('('.code.toUByte()),
    BinUnicode('X'.code.toUByte()),
    BinInt('J'.code.toUByte()),
    Tuple('t'.code.toUByte()),
    BinPersId('Q'.code.toUByte()),
    BinInt1('K'.code.toUByte()),
    BinInt2('M'.code.toUByte()),
    Tuple1(0x85u),
    Tuple2(0x86u),
    Tuple3(0x87u),
    NewTrue(0x88u),
    NewFalse(0x89u),
    None('N'.code.toUByte()),
    BinGet('h'.code.toUByte()),
    LongBinGet('j'.code.toUByte()),
    SetItem('s'.code.toUByte()),
    SetItems('u'.code.toUByte()),
    EmptyDict('}'.code.toUByte()),
    Dict('d'.code.toUByte()),
    Build('b'.code.toUByte()),
    Stop('.'.code.toUByte()),
    NewObj(0x81u),
    EmptyList(']'.code.toUByte()),
    BinFloat('G'.code.toUByte()),
    Append('a'.code.toUByte()),
    Appends('e'.code.toUByte()),
    Long1(0x8au),
    ;

    public interface CandleException

    public interface OResult

    public companion object {
        // Avoid using FromPrimitive so as not to drag another dependency.
        public fun tryFrom(value: UByte): CandleResult<OpCode> =
            when (value.toInt()) {
                0x80 -> Ok(Proto)
                'c'.code -> Ok(Global)
                'q'.code -> Ok(BinPut)
                'r'.code -> Ok(LongBinPut)
                ')'.code -> Ok(EmptyTuple)
                'R'.code -> Ok(Reduce)
                '('.code -> Ok(Mark)
                'X'.code -> Ok(BinUnicode)
                'J'.code -> Ok(BinInt)
                't'.code -> Ok(Tuple)
                'Q'.code -> Ok(BinPersId)
                'K'.code -> Ok(BinInt1)
                'M'.code -> Ok(BinInt2)
                'N'.code -> Ok(None)
                0x85 -> Ok(Tuple1)
                0x86 -> Ok(Tuple2)
                0x87 -> Ok(Tuple3)
                0x88 -> Ok(NewTrue)
                0x89 -> Ok(NewFalse)
                'h'.code -> Ok(BinGet)
                'j'.code -> Ok(LongBinGet)
                's'.code -> Ok(SetItem)
                'u'.code -> Ok(SetItems)
                '}'.code -> Ok(EmptyDict)
                'd'.code -> Ok(EmptyDict)
                'b'.code -> Ok(Build)
                '.'.code -> Ok(Stop)
                0x81 -> Ok(NewObj)
                ']'.code -> Ok(EmptyList)
                'G'.code -> Ok(BinFloat)
                'a'.code -> Ok(Append)
                'e'.code -> Ok(Appends)
                0x8a -> Ok(Long1)
                else -> Err(value)
            }
    }
}

private fun <R : NpyByteReader> readToNewline(r: R): CandleResult<ByteArray> {
    val data = Vec.withCapacity<Byte>(32)
    r.readUntil('\n'.code.toByte(), data).unwrap()
    data.pop()
    if (data.isNotEmpty() && data.last() == '\r'.code.toByte()) {
        data.pop()
    }
    return Ok(data.toByteArray())
}

public sealed class Object {
    public class Class(
        public val moduleName: String,
        internal val className: String,
    ) : Object() {
        public val clsName: String get() = className

        public fun copy(
            moduleName: String = this.moduleName,
            className: String = this.className,
        ): Class = Class(moduleName, className)

        override fun equals(other: Any?): Boolean = other is Class && moduleName == other.moduleName && className == other.className

        override fun hashCode(): kotlin.Int = 31 * moduleName.hashCode() + className.hashCode()

        override fun toString(): String = "Class(moduleName=$moduleName, className=$className)"
    }

    public data class Int(
        public val value: kotlin.Int,
    ) : Object()

    public data class Long(
        public val value: kotlin.Long,
    ) : Object()

    public data class Float(
        public val value: Double,
    ) : Object()

    public data class Unicode(
        public val value: String,
    ) : Object()

    public data class Bool(
        public val value: Boolean,
    ) : Object()

    public data object None : Object()

    @ConsistentCopyVisibility
    public data class Tuple internal constructor(
        internal val values: MutableList<Object>,
    ) : Object()

    @ConsistentCopyVisibility
    public data class List internal constructor(
        internal val values: MutableList<Object>,
    ) : Object()

    public data object Mark : Object()

    @ConsistentCopyVisibility
    public data class Dict internal constructor(
        internal val entries: MutableList<Pair<Object, Object>>,
    ) : Object()

    public data class Reduce(
        public val callable: Object,
        public val args: Object,
    ) : Object()

    public data class Build(
        public val callable: Object,
        public val args: Object,
    ) : Object()

    public data class PersistentLoad(
        public val value: Object,
    ) : Object()

    public fun unicode(): CandleResult<String> =
        when (this) {
            is Unicode -> Ok(this.value)
            else -> Err(this)
        }

    public fun reduce(): CandleResult<Pair<Object, Object>> =
        when (this) {
            is Reduce -> Ok(Pair(this.callable, this.args))
            else -> Err(this)
        }

    public fun none(): CandleResult<Unit> =
        when (this) {
            is None -> Ok(Unit)
            else -> Err(this)
        }

    public fun persistentLoad(): CandleResult<Object> =
        when (this) {
            is PersistentLoad -> Ok(this.value)
            else -> Err(this)
        }

    public fun bool(): CandleResult<Boolean> =
        when (this) {
            is Bool -> Ok(this.value)
            else -> Err(this)
        }

    public fun int(): CandleResult<kotlin.Int> =
        when (this) {
            is Int -> Ok(this.value)
            else -> Err(this)
        }

    public fun intOrLong(): CandleResult<kotlin.Long> =
        when (this) {
            is Int -> Ok(this.value.toLong())
            is Long -> Ok(this.value)
            else -> Err(this)
        }

    internal fun tuple(): CandleResult<MutableList<Object>> =
        when (this) {
            is Tuple -> Ok(this.values)
            else -> Err(this)
        }

    internal fun dict(): CandleResult<MutableList<Pair<Object, Object>>> =
        when (this) {
            is Dict -> Ok(this.entries)
            else -> Err(this)
        }

    public fun class_(): CandleResult<Pair<String, String>> =
        when (this) {
            is Class -> Ok(Pair(this.moduleName, this.className))
            else -> Err(this)
        }

    public fun intoTensorInfo(
        nameObj: Object,
        dirName: String,
    ): CandleResult<TensorInfo?> {
        val nameRes = nameObj.unicode()
        val name =
            when (nameRes) {
                is Ok -> nameRes.value
                is Err -> return Ok(null)
            }
        val reduceRes = this.reduce()
        val callableArgs =
            when (reduceRes) {
                is Ok -> reduceRes.value
                else -> return Ok(null)
            }
        var callable = callableArgs.first
        var args = callableArgs.second
        when {
            callable is Class && callable.moduleName == "torch._tensor" && callable.className == "_rebuild_from_type_v2" -> {
                val tupleArgs = args.tuple().unwrap()
                val nextCallable = tupleArgs.removeAt(0)
                val nextArgs = tupleArgs.removeAt(1)
                callable = nextCallable
                args = nextArgs
            }

            callable is Class && callable.moduleName == "torch._utils" && callable.className == "_rebuild_parameter" -> {
                val tupleArgs = args.tuple().unwrap()
                val reduced = tupleArgs.removeAt(0).reduce().unwrap()
                callable = reduced.first
                args = reduced.second
            }
        }
        when {
            callable is Class && callable.moduleName == "torch._utils" && callable.className == "_rebuild_tensor_v2" -> {}

            else -> {
                return Ok(null)
            }
        }
        val rebuilt = rebuildArgs(args).unwrap()
        val layout = rebuilt.first
        val dtype = rebuilt.second
        val filePath = rebuilt.third
        val storageSize = rebuilt.fourth
        return Ok(
            TensorInfo(
                name = name,
                dtype = dtype,
                layout = layout,
                path = format("{}/{}", dirName.toStringLossy(), filePath),
                storageSize = storageSize,
            ),
        )
    }
}

public fun tryFrom(value: Object): CandleResult<String> =
    when (value) {
        is Object.Unicode -> Ok(value.value)
        else -> Err(value)
    }

public fun tryFrom(
    value: Object,
    allowZero: Boolean = true,
): CandleResult<Int> =
    when (value) {
        is Object.Int -> if (value.value >= 0 || !allowZero) Ok(value.value) else Err(value)
        else -> Err(value)
    }

public fun <T> tryFrom(
    value: Object,
    elemConv: (Object) -> CandleResult<T>,
): CandleResult<List<T>> =
    when (value) {
        is Object.Tuple -> {
            // This does not return the appropriate value in the error case but instead return
            // the object related to the first error.
            Ok(
                value.values
                    .intoIter()
                    .map { v -> elemConv(v).unwrap() }
                    .collect(),
            )
        }

        else -> {
            Err(value)
        }
    }

public class Stack internal constructor(
    private val stack: MutableList<Object> = Vec.withCapacity(512),
    private val memo: MutableMap<UInt, Object> = HashMap.new(),
) {
    public fun stack(): List<Object> = this.stack.asSlice()

    public fun <R : NpyByteReader> readLoop(r: R): CandleResult<Unit> {
        while (true) {
            if (this.read(r).unwrap()) {
                break
            }
        }
        return Ok(Unit)
    }

    public fun finalize(): CandleResult<Object> = this.pop()

    private fun push(obj: Object) {
        this.stack.push(obj)
    }

    private fun pop(): CandleResult<Object> {
        val obj = this.stack.pop()
        return when (obj) {
            null -> bail("unexpected empty stack")
            else -> Ok(obj)
        }
    }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    private fun build(): CandleResult<Unit> {
        val args = this.pop().unwrap()
        val obj = this.pop().unwrap()
        val built =
            when {
                obj is Object.Dict && args is Object.Dict -> {
                    obj.entries.append(args.entries)
                    Object.Dict(obj.entries)
                }

                else -> {
                    Object.Build(
                        callable = obj,
                        args = args,
                    )
                }
            }
        this.push(built)
        return Ok(Unit)
    }

    private fun reduce(): CandleResult<Unit> {
        val args = this.pop().unwrap()
        val callable = this.pop().unwrap()
        val reduced =
            when (callable) {
                is Object.Class -> {
                    if (callable.moduleName == "collections" &&
                        (callable.className == "OrderedDict" || callable.className == "defaultdict")
                    ) {
                        Object.Dict(Vec.new())
                    } else {
                        null
                    }
                }

                else -> {
                    null
                }
            }
        val finalReduced =
            reduced ?: Object.Reduce(
                callable = callable,
                args = args,
            )
        this.push(finalReduced)
        return Ok(Unit)
    }

    private fun last(): CandleResult<Object> {
        val obj = this.stack.lastMut()
        return when (obj) {
            null -> bail("unexpected empty stack")
            else -> Ok(obj)
        }
    }

    private fun memoGet(id: UInt): CandleResult<Object> {
        val obj = this.memo.get(id)
        return when (obj) {
            null -> {
                bail("missing object in memo $id")
            }

            else -> {
                // Maybe we should use refcounting rather than doing potential large clones here.
                Ok(obj.clone())
            }
        }
    }

    private fun memoPut(id: UInt): CandleResult<Unit> {
        val obj = this.last().unwrap().clone()
        this.memo.insert(id, obj)
        return Ok(Unit)
    }

    private fun persistentLoad(id: Object): CandleResult<Object> = Ok(Object.PersistentLoad(id))

    private fun newObj(
        classObj: Object,
        args: Object,
    ): CandleResult<Object> =
        Ok(
            Object.Reduce(
                callable = classObj,
                args = args,
            ),
        )

    private fun popToMarker(): CandleResult<MutableList<Object>> {
        var markIdx: Int? = null
        for ((idx, obj) in this.stack
            .iter()
            .enumerate()
            .rev()) {
            if (obj == Object.Mark) {
                markIdx = idx
                break
            }
        }
        return when (markIdx) {
            null -> {
                bail("marker object not found")
            }

            else -> {
                val objs = this.stack.splitOff(markIdx + 1)
                this.stack.pop()
                Ok(objs)
            }
        }
    }

    public fun <R : NpyByteReader> read(r: R): CandleResult<Boolean> {
        val opRes = OpCode.tryFrom(r.readU8().unwrap())
        val opCode =
            when (opRes) {
                is Ok -> opRes.value
                is Err -> bail("unknown op-code ${opRes.error}")
            }
        // println("op: $opCode")
        // println("${this.stack}")
        when (opCode) {
            OpCode.Proto -> {
                val version = r.readU8().unwrap()
                if (VERBOSE) {
                    println("proto $version")
                }
            }

            OpCode.Global -> {
                val rawModuleName = readToNewline(r).unwrap()
                val rawClassName = readToNewline(r).unwrap()
                val moduleName = String.fromUtf8Lossy(rawModuleName)
                val className = String.fromUtf8Lossy(rawClassName)
                this.push(
                    Object.Class(
                        moduleName = moduleName,
                        className = className,
                    ),
                )
            }

            OpCode.BinInt1 -> {
                val arg = r.readU8().unwrap()
                this.push(Object.Int(arg.toInt()))
            }

            OpCode.BinInt2 -> {
                val arg = r.readU16<LittleEndian>().unwrap()
                this.push(Object.Int(arg.toInt()))
            }

            OpCode.BinInt -> {
                val arg = r.readI32<LittleEndian>().unwrap()
                this.push(Object.Int(arg))
            }

            OpCode.BinFloat -> {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                // https://github.com/pytorch/pytorch/blob/372d078f361e726bb4ac0884ac334b04c58179ef/torch/_weights_only_unpickler.py#L243
                val arg = r.readF64<BigEndian>().unwrap()
                this.push(Object.Float(arg))
            }

            OpCode.BinUnicode -> {
                val len = r.readU32<LittleEndian>().unwrap()
                val rawData = ByteArray(len.toInt())
                r.readExact(rawData).unwrap()
                val data = String.fromUtf8(rawData).mapErr { e -> CandleException.wrap(e) }.unwrap()
                this.push(Object.Unicode(data))
            }

            OpCode.BinPersId -> {
                val id = this.pop().unwrap()
                val obj = this.persistentLoad(id).unwrap()
                this.push(obj)
            }

            OpCode.Tuple -> {
                val objs = this.popToMarker().unwrap()
                this.push(Object.Tuple(objs))
            }

            OpCode.Tuple1 -> {
                val obj = this.pop().unwrap()
                this.push(Object.Tuple(mutableListOf(obj)))
            }

            OpCode.Tuple2 -> {
                val obj2 = this.pop().unwrap()
                val obj1 = this.pop().unwrap()
                this.push(Object.Tuple(mutableListOf(obj1, obj2)))
            }

            OpCode.Tuple3 -> {
                val obj3 = this.pop().unwrap()
                val obj2 = this.pop().unwrap()
                val obj1 = this.pop().unwrap()
                this.push(Object.Tuple(mutableListOf(obj1, obj2, obj3)))
            }

            OpCode.NewTrue -> {
                this.push(Object.Bool(true))
            }

            OpCode.NewFalse -> {
                this.push(Object.Bool(false))
            }

            OpCode.Append -> {
                val value = this.pop().unwrap()
                val pylist = this.last().unwrap()
                if (pylist is Object.List) {
                    pylist.values.push(value)
                } else {
                    bail("expected a list, got $pylist")
                }
            }

            OpCode.Appends -> {
                val objs = this.popToMarker().unwrap()
                val pylist = this.last().unwrap()
                if (pylist is Object.List) {
                    pylist.values.extend(objs)
                } else {
                    bail("expected a list, got $pylist")
                }
            }

            OpCode.SetItem -> {
                val value = this.pop().unwrap()
                val key = this.pop().unwrap()
                val pydict = this.last().unwrap()
                if (pydict is Object.Dict) {
                    pydict.entries.push(Pair(key, value))
                } else {
                    bail("expected a dict, got $pydict")
                }
            }

            OpCode.SetItems -> {
                val objs = this.popToMarker().unwrap()
                val pydict = this.last().unwrap()
                if (pydict is Object.Dict) {
                    if (objs.len() % 2 != 0) {
                        bail("setitems: not an even number of objects")
                    }
                    while (objs.isNotEmpty()) {
                        val value = objs.pop()!!
                        val key = objs.pop().context("empty objs").unwrap()
                        pydict.entries.push(Pair(key, value))
                    }
                } else {
                    bail("expected a dict, got $pydict")
                }
            }

            OpCode.None -> {
                this.push(Object.None)
            }

            OpCode.Stop -> {
                return Ok(true)
            }

            OpCode.Build -> {
                this.build().unwrap()
            }

            OpCode.EmptyDict -> {
                this.push(Object.Dict(Vec.new()))
            }

            OpCode.Dict -> {
                val objs = this.popToMarker().unwrap()
                val pydict = Vec.new<Pair<Object, Object>>()
                if (objs.len() % 2 != 0) {
                    bail("setitems: not an even number of objects")
                }
                while (objs.isNotEmpty()) {
                    val value = objs.pop()!!
                    val key = objs.pop().context("empty objs").unwrap()
                    pydict.push(Pair(key, value))
                }
                this.push(Object.Dict(pydict))
            }

            OpCode.Mark -> {
                this.push(Object.Mark)
            }

            OpCode.Reduce -> {
                this.reduce().unwrap()
            }

            OpCode.EmptyTuple -> {
                this.push(Object.Tuple(Vec.new()))
            }

            OpCode.EmptyList -> {
                this.push(Object.List(Vec.new()))
            }

            OpCode.BinGet -> {
                val arg = r.readU8().unwrap()
                val obj = this.memoGet(arg.toUInt()).unwrap()
                this.push(obj)
            }

            OpCode.LongBinGet -> {
                val arg = r.readU32<LittleEndian>().unwrap()
                val obj = this.memoGet(arg).unwrap()
                this.push(obj)
            }

            OpCode.BinPut -> {
                val arg = r.readU8().unwrap()
                this.memoPut(arg.toUInt()).unwrap()
            }

            OpCode.LongBinPut -> {
                val arg = r.readU32<LittleEndian>().unwrap()
                this.memoPut(arg).unwrap()
            }

            OpCode.NewObj -> {
                val args = this.pop().unwrap()
                val classObj = this.pop().unwrap()
                val obj = this.newObj(classObj, args).unwrap()
                this.push(obj)
            }

            OpCode.Long1 -> {
                val nBytes = r.readU8().unwrap().toInt()
                var v = 0L
                // Decode the next n bytes in little endian
                for (i in 0 until nBytes) {
                    v = v or (r.readU8().unwrap().toLong() shl (i * 8))
                }
                this.push(Object.Long(v))
            }
        }
        return Ok(false)
    }

    public companion object {
        public fun empty(): Stack =
            Stack(
                stack = Vec.withCapacity(512),
                memo = HashMap.new(),
            )
    }
}

public fun from(value: Object): CandleException = CandleException.Msg(format("conversion error on $value"))

// https://github.com/pytorch/pytorch/blob/4eac43d046ded0f0a5a5fa8db03eb40f45bf656e/torch/_utils.py#L198
// Arguments: storage, storageOffset, size, stride, requiresGrad, backwardHooks
private fun rebuildArgs(argsObj: Object): CandleResult<Quadruple<Layout, DType, String, Int>> {
    val args = argsObj.tuple().unwrap()
    val stride = tryFrom(args.removeAt(3)) { tryFrom(it, true) }.unwrap()
    val size = tryFrom(args.removeAt(2)) { tryFrom(it, true) }.unwrap()
    val offset =
        args
            .removeAt(1)
            .intOrLong()
            .unwrap()
            .toInt()
    val storageObj = args.removeAt(0).persistentLoad().unwrap()
    val storage = storageObj.tuple().unwrap()
    val storageSize =
        storage
            .removeAt(4)
            .intOrLong()
            .unwrap()
            .toInt()
    val path = storage.removeAt(2).unicode().unwrap()
    val classPair = storage.removeAt(1).class_().unwrap()
    val className = classPair.second
    val dtype =
        when (className.asStr()) {
            "FloatStorage" -> {
                DType.F32
            }

            "DoubleStorage" -> {
                DType.F64
            }

            "HalfStorage" -> {
                DType.F16
            }

            "BFloat16Storage" -> {
                DType.BF16
            }

            "ByteStorage" -> {
                DType.U8
            }

            "LongStorage" -> {
                DType.I64
            }

            else -> {
                bail("unsupported storage type $className")
            }
        }
    val layout =
        Layout.new(
            Shape.from(size),
            stride,
            offset * dtype.sizeInBytes(),
        )
    return Ok(Quadruple(layout, dtype, path, storageSize))
}

public data class TensorInfo(
    public val name: String,
    public val dtype: DType,
    public val layout: Layout,
    public val path: String,
    public val storageSize: Int,
)

/**
 * Read the tensor info from a .pth file.
 *
 * # Arguments
 * * `file` - The path to the .pth file.
 * * `verbose` - Whether to print debug information.
 * * `key` - Optional key to retrieve `stateDict` from the pth file.
 */
public fun <P : Any> readPthTensorInfo(
    file: P,
    verbose: Boolean,
    key: String?,
): CandleResult<List<TensorInfo>> {
    val openedFile = File.open(file).unwrap()
    val zipReader = BufReader.new(openedFile)
    val zip = ZipArchive.new(zipReader).unwrap()
    val zipFileNames =
        zip
            .fileNames()
            .collect()

    val tensorInfos = Vec.new<TensorInfo>()
    for (fileName in zipFileNames.iter()) {
        if (!fileName.endsWith("data.pkl")) {
            continue
        }
        val dirName = PathBuf.from(fileName.stripSuffix(".pkl").context("no .pkl").unwrap())
        val rawReader = zip.byName(fileName).unwrap()
        val reader = BufReader.new(rawReader)
        val stack = Stack.empty()
        stack.readLoop(reader).unwrap()
        var obj = stack.finalize().unwrap()
        if (VERBOSE || verbose) {
            println("$obj")
        }

        obj =
            when (obj) {
                is Object.Build -> {
                    val buildCallable = obj.callable
                    when (buildCallable) {
                        is Object.Reduce -> {
                            val reduceCallable = buildCallable.callable
                            when {
                                reduceCallable is Object.Class &&
                                    reduceCallable.moduleName == "__torch__" &&
                                    reduceCallable.className == "Module" -> obj.args

                                else -> continue
                            }
                        }

                        else -> {
                            continue
                        }
                    }
                }

                else -> {
                    obj
                }
            }

        // If key is provided, then we need to extract the stateDict from the object.
        obj =
            if (key != null) {
                if (obj is Object.Dict) {
                    obj.entries
                        .intoIter()
                        .find { (k, _) -> k == Object.Unicode(key.toOwned()) }
                        ?.second
                        .okOrElse { CandleException.Msg(format("key $key not found")) }
                        .unwrap()
                } else {
                    obj
                }
            } else {
                obj
            }

        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is stateDict by this stage.
        if (obj is Object.Dict) {
            for ((name, value) in obj.entries.intoIter()) {
                val infoRes = value.intoTensorInfo(name, dirName)
                when (infoRes) {
                    is Ok -> {
                        val tensorInfo = infoRes.value
                        if (tensorInfo != null) {
                            tensorInfos.push(tensorInfo)
                        }
                    }

                    is Err -> {
                        eprintln("skipping: ${infoRes.error}")
                    }
                }
            }
        }
    }
    return Ok(tensorInfos)
}

/**
 * Lazy tensor loader.
 */
public class PthTensors(
    private val tensorInfos: Map<String, TensorInfo>,
    private val path: String,
    // We do not store a zip reader as it needs mutable access to extract data. Instead we
    // re-create a zip reader for each tensor.
) {
    public fun tensorInfos(): Map<String, TensorInfo> = this.tensorInfos

    public fun get(name: String): CandleResult<Tensor?> {
        val info = this.tensorInfos.get(name)
        val tensorInfo =
            when (info) {
                null -> return Ok(null)
                else -> info
            }
        // We hope that the file has not changed since first reading it.
        val zipReader = BufReader.new(File.open(this.path).unwrap())
        val zip = ZipArchive.new(zipReader).unwrap()
        val reader = zip.byName(tensorInfo.path).unwrap()
        val isFortranContiguous = tensorInfo.layout.isFortranContiguous()
        val rank = tensorInfo.layout.shape().rank()

        // Reading the data is a bit tricky as it can be strided, for now only support the basic
        // case and when the tensor is fortran contiguous.
        if (!tensorInfo.layout.isContiguous() && !isFortranContiguous) {
            bail(
                "cannot retrieve non-contiguous tensors ${tensorInfo.layout}",
            )
        }
        val startOffset = tensorInfo.layout.startOffset()
        if (startOffset > 0) {
            ioCopy(
                reader.byRef().take(startOffset.toULong()),
                ioSink(),
            ).unwrap()
        }
        val tensor =
            Tensor.fromReader(
                tensorInfo.layout.shape().clone(),
                tensorInfo.dtype,
                reader,
            )

        return if (rank > 1 && isFortranContiguous) {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            val shapeReversed: List<Int> =
                tensorInfo.layout
                    .dims()
                    .toList()
                    .iter()
                    .rev()
                    .cloned()
                    .collect()
            val reshaped = tensor.reshape(Shape.from(shapeReversed))

            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            val dimIndicesReversed: List<Int> = (0 until rank).rev().collect()
            val permuted = reshaped.permute(dimIndicesReversed)
            Ok(permuted)
        } else {
            Ok(tensor)
        }
    }

    public companion object {
        public fun <P : Any> new(
            path: P,
            key: String?,
        ): CandleResult<PthTensors> {
            val pathStr = path.toString()
            val rawInfos = readPthTensorInfo(pathStr, false, key).unwrap()
            val tensorInfos =
                rawInfos
                    .intoIter()
                    .map { ti -> Pair(ti.name, ti) }
                    .collectMap()
            val ownedPath = pathStr.toOwned()
            return Ok(PthTensors(tensorInfos, ownedPath))
        }
    }
}

/**
 * Read all the tensors from a PyTorch pth file with a given key.
 *
 * # Arguments
 * * `path` - Path to the pth file.
 * * `key` - Optional key to retrieve `stateDict` from the pth file. Sometimes the pth file
 *   contains multiple objects and the stateDict is the one we are interested in.
 */
public fun <P : Any> readAllWithKey(
    path: P,
    key: String?,
): CandleResult<List<Pair<String, Tensor>>> {
    val pth = PthTensors.new(path, key).unwrap()
    val tensorNames = pth.tensorInfos().keys
    val tensors = Vec.withCapacity<Pair<String, Tensor>>(tensorNames.len())
    for (name in tensorNames) {
        val tensor = pth.get(name).unwrap()
        if (tensor != null) {
            tensors.push(Pair(name, tensor))
        }
    }
    return Ok(tensors)
}

/**
 * Read all the tensors from a PyTorch pth file.
 *
 * # Arguments
 * * `path` - Path to the pth file.
 */
public fun <P : Any> readAll(path: P): CandleResult<List<Pair<String, Tensor>>> = readAllWithKey(path, null)
