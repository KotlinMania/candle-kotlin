// port-lint: source pickle.rs
/**
 * Just enough pickle support to be able to read PyTorch checkpoints.
 */
package io.github.kotlinmania.candle

// This hardcodes objects that are required for tensor reading, we may want to make this a bit more
// composable/tensor agnostic at some point.

private const val VERBOSE: Boolean = false

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
enum class OpCode(val code: UByte) {
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
    Long1(0x8au);

    companion object {
        // Avoid using FromPrimitive so as not to drag another dependency.
        fun tryFrom(value: UByte): CandleResult<OpCode> {
            return when (value.toInt()) {
                0x80 -> CandleResult.Ok(Proto)
                'c'.code -> CandleResult.Ok(Global)
                'q'.code -> CandleResult.Ok(BinPut)
                'r'.code -> CandleResult.Ok(LongBinPut)
                ')'.code -> CandleResult.Ok(EmptyTuple)
                'R'.code -> CandleResult.Ok(Reduce)
                '('.code -> CandleResult.Ok(Mark)
                'X'.code -> CandleResult.Ok(BinUnicode)
                'J'.code -> CandleResult.Ok(BinInt)
                't'.code -> CandleResult.Ok(Tuple)
                'Q'.code -> CandleResult.Ok(BinPersId)
                'K'.code -> CandleResult.Ok(BinInt1)
                'M'.code -> CandleResult.Ok(BinInt2)
                'N'.code -> CandleResult.Ok(None)
                0x85 -> CandleResult.Ok(Tuple1)
                0x86 -> CandleResult.Ok(Tuple2)
                0x87 -> CandleResult.Ok(Tuple3)
                0x88 -> CandleResult.Ok(NewTrue)
                0x89 -> CandleResult.Ok(NewFalse)
                'h'.code -> CandleResult.Ok(BinGet)
                'j'.code -> CandleResult.Ok(LongBinGet)
                's'.code -> CandleResult.Ok(SetItem)
                'u'.code -> CandleResult.Ok(SetItems)
                '}'.code -> CandleResult.Ok(EmptyDict)
                'd'.code -> CandleResult.Ok(EmptyDict)
                'b'.code -> CandleResult.Ok(Build)
                '.'.code -> CandleResult.Ok(Stop)
                0x81 -> CandleResult.Ok(NewObj)
                ']'.code -> CandleResult.Ok(EmptyList)
                'G'.code -> CandleResult.Ok(BinFloat)
                'a'.code -> CandleResult.Ok(Append)
                'e'.code -> CandleResult.Ok(Appends)
                0x8a -> CandleResult.Ok(Long1)
                else -> CandleResult.Err(CandleError.Msg("unknown op-code $value"))
            }
        }
    }
}

private fun readToNewline(r: NpyByteReader): CandleResult<ByteArray> {
    val data = ArrayList<Byte>(32)
    val oneByte = ByteArray(1)
    while (true) {
        when (val res = r.readExact(oneByte)) {
            is CandleResult.Ok -> {
                val b = oneByte[0]
                data.add(b)
                if (b == '\n'.code.toByte()) break
            }
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
    }
    if (data.isNotEmpty()) {
        data.removeAt(data.size - 1)
    }
    if (data.isNotEmpty() && data[data.size - 1] == '\r'.code.toByte()) {
        data.removeAt(data.size - 1)
    }
    return CandleResult.Ok(ByteArray(data.size) { i -> data[i] })
}

sealed class Object {
    data class Class(
        val moduleName: String,
        val className: String,
    ) : Object()

    data class Int(val value: kotlin.Int) : Object()
    data class Long(val value: kotlin.Long) : Object()
    data class Float(val value: Double) : Object()
    data class Unicode(val value: String) : Object()
    data class Bool(val value: Boolean) : Object()
    data object None : Object()
    data class Tuple(val values: MutableList<Object>) : Object()
    data class List(val values: MutableList<Object>) : Object()
    data object Mark : Object()
    data class Dict(val entries: MutableList<Pair<Object, Object>>) : Object()
    data class Reduce(
        val callable: Object,
        val args: Object,
    ) : Object()

    data class Build(
        val callable: Object,
        val args: Object,
    ) : Object()

    data class PersistentLoad(val value: Object) : Object()

    fun unicode(): CandleResult<String> = when (this) {
        is Unicode -> CandleResult.Ok(this.value)
        else -> CandleResult.Err(from(this))
    }

    fun reduce(): CandleResult<Pair<Object, Object>> = when (this) {
        is Reduce -> CandleResult.Ok(this.callable to this.args)
        else -> CandleResult.Err(from(this))
    }

    fun none(): CandleResult<Unit> = when (this) {
        is None -> CandleResult.Ok(Unit)
        else -> CandleResult.Err(from(this))
    }

    fun persistentLoad(): CandleResult<Object> = when (this) {
        is PersistentLoad -> CandleResult.Ok(this.value)
        else -> CandleResult.Err(from(this))
    }

    fun bool(): CandleResult<Boolean> = when (this) {
        is Bool -> CandleResult.Ok(this.value)
        else -> CandleResult.Err(from(this))
    }

    fun int(): CandleResult<kotlin.Int> = when (this) {
        is Int -> CandleResult.Ok(this.value)
        else -> CandleResult.Err(from(this))
    }

    fun intOrLong(): CandleResult<kotlin.Long> = when (this) {
        is Int -> CandleResult.Ok(this.value.toLong())
        is Long -> CandleResult.Ok(this.value)
        else -> CandleResult.Err(from(this))
    }

    fun tuple(): CandleResult<MutableList<Object>> = when (this) {
        is Tuple -> CandleResult.Ok(this.values)
        else -> CandleResult.Err(from(this))
    }

    fun dict(): CandleResult<MutableList<Pair<Object, Object>>> = when (this) {
        is Dict -> CandleResult.Ok(this.entries)
        else -> CandleResult.Err(from(this))
    }

    fun `class`(): CandleResult<Pair<String, String>> = when (this) {
        is Class -> CandleResult.Ok(this.moduleName to this.className)
        else -> CandleResult.Err(from(this))
    }

    fun intoTensorInfo(
        nameObj: Object,
        dirName: String,
    ): CandleResult<TensorInfo?> {
        val name = when (val res = nameObj.unicode()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Ok(null)
        }
        val callableArgs = when (val res = this.reduce()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Ok(null)
        }
        var callable = callableArgs.first
        var args = callableArgs.second
        if (callable is Class && callable.moduleName == "torch._tensor" && callable.className == "_rebuild_from_type_v2") {
            val tupleArgs = when (val res = args.tuple()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            callable = tupleArgs.removeAt(0)
            args = tupleArgs.removeAt(0)
        } else if (callable is Class && callable.moduleName == "torch._utils" && callable.className == "_rebuild_parameter") {
            val tupleArgs = when (val res = args.tuple()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val innerPair = when (val res = tupleArgs.removeAt(0).reduce()) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            callable = innerPair.first
            args = innerPair.second
        }
        if (!(callable is Class && callable.moduleName == "torch._utils" && callable.className == "_rebuild_tensor_v2")) {
            return CandleResult.Ok(null)
        }
        val rebuilt = when (val res = rebuildArgs(args)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        return CandleResult.Ok(
            TensorInfo(
                name = name,
                dtype = rebuilt.dtype,
                layout = rebuilt.layout,
                path = "$dirName/${rebuilt.path}",
                storageSize = rebuilt.storageSize,
            )
        )
    }
}

fun tryFrom(value: Object): CandleResult<String> = when (value) {
    is Object.Unicode -> CandleResult.Ok(value.value)
    else -> CandleResult.Err(from(value))
}

fun tryFrom(value: Object, allowZero: Boolean = true): CandleResult<Int> = when (value) {
    is Object.Int -> if (value.value >= 0 || !allowZero) CandleResult.Ok(value.value) else CandleResult.Err(from(value))
    else -> CandleResult.Err(from(value))
}

fun <T> tryFrom(
    value: Object,
    elemConv: (Object) -> CandleResult<T>,
): CandleResult<List<T>> = when (value) {
    is Object.Tuple -> {
        // This does not return the appropriate value in the error case but instead return
        // the object related to the first error.
        val out = ArrayList<T>(value.values.size)
        for (v in value.values) {
            when (val res = elemConv(v)) {
                is CandleResult.Ok -> out.add(res.value)
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
        }
        CandleResult.Ok(out)
    }
    else -> CandleResult.Err(from(value))
}

class Stack(
    private val stackList: MutableList<Object> = ArrayList(512),
    private val memo: MutableMap<UInt, Object> = HashMap(),
) {
    fun stack(): List<Object> = this.stackList

    fun readLoop(r: NpyByteReader): CandleResult<Unit> {
        while (true) {
            val done = when (val res = this.read(r)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            if (done) break
        }
        return CandleResult.Ok(Unit)
    }

    fun finalize(): CandleResult<Object> = this.pop()

    private fun push(obj: Object) {
        this.stackList.add(obj)
    }

    private fun pop(): CandleResult<Object> {
        if (this.stackList.isEmpty()) {
            return CandleResult.Err(CandleError.Msg("unexpected empty stack").bt())
        }
        return CandleResult.Ok(this.stackList.removeAt(this.stackList.size - 1))
    }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    private fun build(): CandleResult<Unit> {
        val args = when (val res = this.pop()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val obj = when (val res = this.pop()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val built = if (obj is Object.Dict && args is Object.Dict) {
            obj.entries.addAll(args.entries)
            Object.Dict(obj.entries)
        } else {
            Object.Build(callable = obj, args = args)
        }
        this.push(built)
        return CandleResult.Ok(Unit)
    }

    private fun reduce(): CandleResult<Unit> {
        val args = when (val res = this.pop()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val callable = when (val res = this.pop()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val reduced = if (
            callable is Object.Class &&
            callable.moduleName == "collections" &&
            (callable.className == "OrderedDict" || callable.className == "defaultdict")
        ) {
            Object.Dict(ArrayList())
        } else {
            Object.Reduce(callable = callable, args = args)
        }
        this.push(reduced)
        return CandleResult.Ok(Unit)
    }

    private fun last(): CandleResult<Object> {
        if (this.stackList.isEmpty()) {
            return CandleResult.Err(CandleError.Msg("unexpected empty stack").bt())
        }
        return CandleResult.Ok(this.stackList[this.stackList.size - 1])
    }

    private fun memoGet(id: UInt): CandleResult<Object> {
        val obj = this.memo[id]
            ?: return CandleResult.Err(CandleError.Msg("missing object in memo $id").bt())
        // Maybe we should use refcounting rather than doing potential large clones here.
        return CandleResult.Ok(obj)
    }

    private fun memoPut(id: UInt): CandleResult<Unit> {
        val obj = when (val res = this.last()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        this.memo[id] = obj
        return CandleResult.Ok(Unit)
    }

    private fun persistentLoad(id: Object): CandleResult<Object> =
        CandleResult.Ok(Object.PersistentLoad(id))

    private fun newObj(classObj: Object, args: Object): CandleResult<Object> =
        CandleResult.Ok(Object.Reduce(callable = classObj, args = args))

    private fun popToMarker(): CandleResult<MutableList<Object>> {
        var markIdx = -1
        for (idx in this.stackList.indices.reversed()) {
            if (this.stackList[idx] == Object.Mark) {
                markIdx = idx
                break
            }
        }
        if (markIdx < 0) {
            return CandleResult.Err(CandleError.Msg("marker object not found").bt())
        }
        val objs = ArrayList<Object>(this.stackList.size - markIdx - 1)
        for (i in (markIdx + 1) until this.stackList.size) {
            objs.add(this.stackList[i])
        }
        while (this.stackList.size > markIdx) {
            this.stackList.removeAt(this.stackList.size - 1)
        }
        return CandleResult.Ok(objs)
    }

    fun read(r: NpyByteReader): CandleResult<Boolean> {
        val readU8 = {
            val b = ByteArray(1)
            when (val res = r.readExact(b)) {
                is CandleResult.Ok -> CandleResult.Ok(b[0].toUByte())
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }
        val readU16Le = {
            val b = ByteArray(2)
            when (val res = r.readExact(b)) {
                is CandleResult.Ok -> {
                    val v = (b[0].toInt() and 0xFF) or ((b[1].toInt() and 0xFF) shl 8)
                    CandleResult.Ok(v.toUShort())
                }
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }
        val readI32Le = {
            val b = ByteArray(4)
            when (val res = r.readExact(b)) {
                is CandleResult.Ok -> {
                    val v = (b[0].toInt() and 0xFF) or
                        ((b[1].toInt() and 0xFF) shl 8) or
                        ((b[2].toInt() and 0xFF) shl 16) or
                        ((b[3].toInt() and 0xFF) shl 24)
                    CandleResult.Ok(v)
                }
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }
        val readF64Be = {
            val b = ByteArray(8)
            when (val res = r.readExact(b)) {
                is CandleResult.Ok -> {
                    var bits = 0L
                    for (i in 0 until 8) {
                        bits = (bits shl 8) or (b[i].toLong() and 0xFFL)
                    }
                    CandleResult.Ok(Double.fromBits(bits))
                }
                is CandleResult.Err -> CandleResult.Err(res.error)
            }
        }

        val rawOp = when (val res = readU8()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        val opCode = when (val res = OpCode.tryFrom(rawOp)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        when (opCode) {
            OpCode.Proto -> {
                val version = when (val res = readU8()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                if (VERBOSE) {
                    println("proto $version")
                }
            }
            OpCode.Global -> {
                val moduleBytes = when (val res = readToNewline(r)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val classBytes = when (val res = readToNewline(r)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val moduleName = moduleBytes.decodeToString()
                val className = classBytes.decodeToString()
                this.push(Object.Class(moduleName = moduleName, className = className))
            }
            OpCode.BinInt1 -> {
                val arg = when (val res = readU8()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Int(arg.toInt()))
            }
            OpCode.BinInt2 -> {
                val arg = when (val res = readU16Le()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Int(arg.toInt()))
            }
            OpCode.BinInt -> {
                val arg = when (val res = readI32Le()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Int(arg))
            }
            OpCode.BinFloat -> {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                val arg = when (val res = readF64Be()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Float(arg))
            }
            OpCode.BinUnicode -> {
                val len = when (val res = readI32Le()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val data = ByteArray(len)
                when (val res = r.readExact(data)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Unicode(data.decodeToString()))
            }
            OpCode.BinPersId -> {
                val id = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj = when (val res = this.persistentLoad(id)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(obj)
            }
            OpCode.Tuple -> {
                val objs = when (val res = this.popToMarker()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Tuple(objs))
            }
            OpCode.Tuple1 -> {
                val obj = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Tuple(mutableListOf(obj)))
            }
            OpCode.Tuple2 -> {
                val obj2 = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj1 = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Tuple(mutableListOf(obj1, obj2)))
            }
            OpCode.Tuple3 -> {
                val obj3 = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj2 = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj1 = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(Object.Tuple(mutableListOf(obj1, obj2, obj3)))
            }
            OpCode.NewTrue -> this.push(Object.Bool(true))
            OpCode.NewFalse -> this.push(Object.Bool(false))
            OpCode.Append -> {
                val value = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val pylist = when (val res = this.last()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                if (pylist is Object.List) {
                    pylist.values.add(value)
                } else {
                    return CandleResult.Err(CandleError.Msg("expected a list, got $pylist").bt())
                }
            }
            OpCode.Appends -> {
                val objs = when (val res = this.popToMarker()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val pylist = when (val res = this.last()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                if (pylist is Object.List) {
                    pylist.values.addAll(objs)
                } else {
                    return CandleResult.Err(CandleError.Msg("expected a list, got $pylist").bt())
                }
            }
            OpCode.SetItem -> {
                val value = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val key = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val pydict = when (val res = this.last()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                if (pydict is Object.Dict) {
                    pydict.entries.add(key to value)
                } else {
                    return CandleResult.Err(CandleError.Msg("expected a dict, got $pydict").bt())
                }
            }
            OpCode.SetItems -> {
                val objs = when (val res = this.popToMarker()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val pydict = when (val res = this.last()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                if (pydict is Object.Dict) {
                    if (objs.size % 2 != 0) {
                        return CandleResult.Err(CandleError.Msg("setitems: not an even number of objects").bt())
                    }
                    while (objs.isNotEmpty()) {
                        val value = objs.removeAt(objs.size - 1)
                        val key = objs.removeAt(objs.size - 1)
                        pydict.entries.add(key to value)
                    }
                } else {
                    return CandleResult.Err(CandleError.Msg("expected a dict, got $pydict").bt())
                }
            }
            OpCode.None -> this.push(Object.None)
            OpCode.Stop -> return CandleResult.Ok(true)
            OpCode.Build -> {
                when (val res = this.build()) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
            }
            OpCode.EmptyDict -> this.push(Object.Dict(ArrayList()))
            OpCode.Dict -> {
                val objs = when (val res = this.popToMarker()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val pydict = ArrayList<Pair<Object, Object>>()
                if (objs.size % 2 != 0) {
                    return CandleResult.Err(CandleError.Msg("setitems: not an even number of objects").bt())
                }
                while (objs.isNotEmpty()) {
                    val value = objs.removeAt(objs.size - 1)
                    val key = objs.removeAt(objs.size - 1)
                    pydict.add(key to value)
                }
                this.push(Object.Dict(pydict))
            }
            OpCode.Mark -> this.push(Object.Mark)
            OpCode.Reduce -> {
                when (val res = this.reduce()) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
            }
            OpCode.EmptyTuple -> this.push(Object.Tuple(ArrayList()))
            OpCode.EmptyList -> this.push(Object.List(ArrayList()))
            OpCode.BinGet -> {
                val arg = when (val res = readU8()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj = when (val res = this.memoGet(arg.toUInt())) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(obj)
            }
            OpCode.LongBinGet -> {
                val arg = when (val res = readI32Le()) {
                    is CandleResult.Ok -> res.value.toUInt()
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj = when (val res = this.memoGet(arg)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(obj)
            }
            OpCode.BinPut -> {
                val arg = when (val res = readU8()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                when (val res = this.memoPut(arg.toUInt())) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
            }
            OpCode.LongBinPut -> {
                val arg = when (val res = readI32Le()) {
                    is CandleResult.Ok -> res.value.toUInt()
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                when (val res = this.memoPut(arg)) {
                    is CandleResult.Ok -> {}
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
            }
            OpCode.NewObj -> {
                val args = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val classObj = when (val res = this.pop()) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                val obj = when (val res = this.newObj(classObj, args)) {
                    is CandleResult.Ok -> res.value
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                this.push(obj)
            }
            OpCode.Long1 -> {
                val nBytes = when (val res = readU8()) {
                    is CandleResult.Ok -> res.value.toInt()
                    is CandleResult.Err -> return CandleResult.Err(res.error)
                }
                var v = 0L
                // Decode the next n bytes in little endian
                for (i in 0 until nBytes) {
                    val b = when (val res = readU8()) {
                        is CandleResult.Ok -> res.value.toLong()
                        is CandleResult.Err -> return CandleResult.Err(res.error)
                    }
                    v = v or (b shl (i * 8))
                }
                this.push(Object.Long(v))
            }
        }
        return CandleResult.Ok(false)
    }

    companion object {
        fun empty(): Stack = Stack()
    }
}

fun from(value: Object): CandleError =
    CandleError.Msg("conversion error on $value")

private data class RebuiltTensorArgs(
    val layout: Layout,
    val dtype: DType,
    val path: String,
    val storageSize: Int,
)

// Arguments: storage, storageOffset, size, stride, requiresGrad, backwardHooks
private fun rebuildArgs(args: Object): CandleResult<RebuiltTensorArgs> {
    val tupleArgs = when (val res = args.tuple()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val stride = when (val res = tryFrom(tupleArgs.removeAt(3)) { tryFrom(it, true) }) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val size = when (val res = tryFrom(tupleArgs.removeAt(2)) { tryFrom(it, true) }) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val offset = when (val res = tupleArgs.removeAt(1).intOrLong()) {
        is CandleResult.Ok -> res.value.toInt()
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val storageObj = when (val res = tupleArgs.removeAt(0).persistentLoad()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val storage = when (val res = storageObj.tuple()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val storageSize = when (val res = storage.removeAt(4).intOrLong()) {
        is CandleResult.Ok -> res.value.toInt()
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val path = when (val res = storage.removeAt(2).unicode()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val classPair = when (val res = storage.removeAt(1).`class`()) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val className = classPair.second
    val dtype = when (className) {
        "FloatStorage" -> DType.F32
        "DoubleStorage" -> DType.F64
        "HalfStorage" -> DType.F16
        "BFloat16Storage" -> DType.BF16
        "ByteStorage" -> DType.U8
        "LongStorage" -> DType.I64
        else -> return CandleResult.Err(CandleError.Msg("unsupported storage type $className").bt())
    }
    val layout = Layout.new(
        Shape.fromDims(size.toIntArray()),
        stride,
        offset * dtype.sizeInBytes(),
    )
    return CandleResult.Ok(RebuiltTensorArgs(layout, dtype, path, storageSize))
}

data class TensorInfo(
    val name: String,
    val dtype: DType,
    val layout: Layout,
    val path: String,
    val storageSize: Int,
)

private val inMemoryPthArchives: MutableMap<String, LinkedHashMap<String, ByteArray>> = LinkedHashMap()

/**
 * Read the tensor info from a .pth file.
 *
 * # Arguments
 * * `file` - The path to the .pth file.
 * * `verbose` - Whether to print debug information.
 * * `key` - Optional key to retrieve `stateDict` from the pth file.
 */
fun readPthTensorInfo(
    file: String,
    verbose: Boolean,
    key: String?,
): CandleResult<List<TensorInfo>> {
    val zip = inMemoryPthArchives[file]
        ?: return CandleResult.Err(CandleError.Msg("pth file not found: $file").withPath(file))
    val zipFileNames = zip.keys.toList()

    val tensorInfos = ArrayList<TensorInfo>()
    for (fileName in zipFileNames) {
        if (!fileName.endsWith("data.pkl")) {
            continue
        }
        val dirName = fileName.removeSuffix(".pkl")
        val pklBytes = zip[fileName]
            ?: return CandleResult.Err(CandleError.Msg("missing entry $fileName").bt())
        val reader = NpyByteReader(pklBytes)
        val stack = Stack.empty()
        when (val res = stack.readLoop(reader)) {
            is CandleResult.Ok -> {}
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        var obj = when (val res = stack.finalize()) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (VERBOSE || verbose) {
            println(obj)
        }

        if (obj is Object.Build) {
            val buildCallable = obj.callable
            if (buildCallable is Object.Reduce) {
                val reduceCallable = buildCallable.callable
                if (reduceCallable is Object.Class && reduceCallable.moduleName == "__torch__" && reduceCallable.className == "Module") {
                    obj = obj.args
                } else {
                    continue
                }
            } else {
                continue
            }
        }

        // If key is provided, then we need to extract the stateDict from the object.
        if (key != null && obj is Object.Dict) {
            val found = obj.entries.firstOrNull { (k, _) -> k == Object.Unicode(key) }?.second
                ?: return CandleResult.Err(CandleError.Msg("key $key not found"))
            obj = found
        }

        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is stateDict by this stage.
        if (obj is Object.Dict) {
            for ((name, value) in obj.entries) {
                when (val res = value.intoTensorInfo(name, dirName)) {
                    is CandleResult.Ok -> {
                        val info = res.value
                        if (info != null) {
                            tensorInfos.add(info)
                        }
                    }
                    is CandleResult.Err -> {
                        println("skipping: ${res.error}")
                    }
                }
            }
        }
    }
    return CandleResult.Ok(tensorInfos)
}

/**
 * Lazy tensor loader.
 */
class PthTensors private constructor(
    private val tensorInfosMap: Map<String, TensorInfo>,
    private val path: String,
    // We do not store a zip reader as it needs mutable access to extract data. Instead we
    // re-create a zip reader for each tensor.
) {
    fun tensorInfos(): Map<String, TensorInfo> = this.tensorInfosMap

    fun get(name: String): CandleResult<Tensor?> {
        val tensorInfo = this.tensorInfosMap[name] ?: return CandleResult.Ok(null)
        // We hope that the file has not changed since first reading it.
        val zip = inMemoryPthArchives[this.path]
            ?: return CandleResult.Err(CandleError.Msg("pth file not found: ${this.path}").withPath(this.path))
        val entryBytes = zip[tensorInfo.path]
            ?: return CandleResult.Err(CandleError.Msg("missing tensor data ${tensorInfo.path}").bt())
        val reader = NpyByteReader(entryBytes)
        val isFortranContiguous = tensorInfo.layout.isFortranContiguous()
        val rank = tensorInfo.layout.shape().rank()

        // Reading the data is a bit tricky as it can be strided, for now only support the basic
        // case and when the tensor is fortran contiguous.
        if (!tensorInfo.layout.isContiguous() && !isFortranContiguous) {
            return CandleResult.Err(
                CandleError.Msg("cannot retrieve non-contiguous tensors ${tensorInfo.layout}").bt()
            )
        }
        val startOffset = tensorInfo.layout.startOffset()
        if (startOffset > 0) {
            reader.position += startOffset
        }
        val tensor = when (val res = fromReader(tensorInfo.layout.shape(), tensorInfo.dtype, reader)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }

        return if (rank > 1 && isFortranContiguous) {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            val shapeReversed = tensorInfo.layout.dims().reversed().toIntArray()
            val reshaped = when (val res = tensor.reshape(Shape.fromDims(shapeReversed))) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }

            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            val dimIndicesReversed = (0 until rank).reversed().toList()
            val permuted = when (val res = reshaped.permute(dimIndicesReversed)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            CandleResult.Ok(permuted)
        } else {
            CandleResult.Ok(tensor)
        }
    }

    companion object {
        fun new(path: String, key: String?): CandleResult<PthTensors> {
            val infosList = when (val res = readPthTensorInfo(path, false, key)) {
                is CandleResult.Ok -> res.value
                is CandleResult.Err -> return CandleResult.Err(res.error)
            }
            val tensorInfos = LinkedHashMap<String, TensorInfo>(infosList.size)
            for (ti in infosList) {
                tensorInfos[ti.name] = ti
            }
            return CandleResult.Ok(PthTensors(tensorInfos, path))
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
fun readAllWithKey(
    path: String,
    key: String?,
): CandleResult<List<Pair<String, Tensor>>> {
    val pth = when (val res = PthTensors.new(path, key)) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }
    val tensorNames = pth.tensorInfos().keys
    val tensors = ArrayList<Pair<String, Tensor>>(tensorNames.size)
    for (name in tensorNames) {
        val tensor = when (val res = pth.get(name)) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }
        if (tensor != null) {
            tensors.add(name to tensor)
        }
    }
    return CandleResult.Ok(tensors)
}

/**
 * Read all the tensors from a PyTorch pth file.
 *
 * # Arguments
 * * `path` - Path to the pth file.
 */
fun readAll(path: String): CandleResult<List<Pair<String, Tensor>>> =
    readAllWithKey(path, null)
