// port-lint: source pickle.rs
package io.github.kotlinmania.candle.pickle

import io.github.kotlinmania.byteorder.BigEndian
import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.rustlang.core.F16
import io.github.kotlinmania.rustlang.core.result.unwrapOrElse
import io.github.kotlinmania.candle.error.Error as E

// Just enough pickle support to be able to read PyTorch checkpoints.

public val VERBOSE: Boolean = false

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
public enum class OpCode(
    public val value: UByte,
) {
    // https://github.com/python/cpython/blob/ed25f097160b5cbb0c9a1f9a746d2f1bbc96515a/Lib/pickletools.py#L2123
    Proto(0x80u),
    Global(99u),
    BinPut(113u),
    LongBinPut(114u),
    EmptyTuple(41u),
    Reduce(82u),
    Mark(40u),
    BinUnicode(88u),
    BinInt(74u),
    Tuple(116u),
    BinPersId(81u),
    BinInt1(75u),
    BinInt2(77u),
    Tuple1(0x85u),
    Tuple2(0x86u),
    Tuple3(0x87u),
    NewTrue(0x88u),
    NewFalse(0x89u),
    None(78u),
    BinGet(104u),
    LongBinGet(106u),
    SetItem(115u),
    SetItems(117u),
    EmptyDict(125u),
    Dict(100u),
    Build(98u),
    Stop(46u),
    NewObj(0x81u),
    EmptyList(93u),
    BinFloat(71u),
    Append(97u),
    Appends(101u),
    Long1(0x8au),
    ;

    public companion object {
        // Avoid using FromPrimitive so as not to drag another dependency.
        public fun tryFrom(value: UByte): Result<OpCode, UByte> =
            when (value) {
                0x80 -> Result.Ok(OpCode.Proto)
                99u -> Result.Ok(OpCode.Global)
                113u -> Result.Ok(OpCode.BinPut)
                114u -> Result.Ok(OpCode.LongBinPut)
                41u -> Result.Ok(OpCode.EmptyTuple)
                82u -> Result.Ok(OpCode.Reduce)
                40u -> Result.Ok(OpCode.Mark)
                88u -> Result.Ok(OpCode.BinUnicode)
                74u -> Result.Ok(OpCode.BinInt)
                116u -> Result.Ok(OpCode.Tuple)
                81u -> Result.Ok(OpCode.BinPersId)
                75u -> Result.Ok(OpCode.BinInt1)
                77u -> Result.Ok(OpCode.BinInt2)
                78u -> Result.Ok(OpCode.None)
                0x85 -> Result.Ok(OpCode.Tuple1)
                0x86 -> Result.Ok(OpCode.Tuple2)
                0x87 -> Result.Ok(OpCode.Tuple3)
                0x88 -> Result.Ok(OpCode.NewTrue)
                0x89 -> Result.Ok(OpCode.NewFalse)
                104u -> Result.Ok(OpCode.BinGet)
                106u -> Result.Ok(OpCode.LongBinGet)
                115u -> Result.Ok(OpCode.SetItem)
                117u -> Result.Ok(OpCode.SetItems)
                125u -> Result.Ok(OpCode.EmptyDict)
                100u -> Result.Ok(OpCode.EmptyDict)
                98u -> Result.Ok(OpCode.Build)
                46u -> Result.Ok(OpCode.Stop)
                0x81 -> Result.Ok(OpCode.NewObj)
                93u -> Result.Ok(OpCode.EmptyList)
                71u -> Result.Ok(OpCode.BinFloat)
                97u -> Result.Ok(OpCode.Append)
                101u -> Result.Ok(OpCode.Appends)
                0x8a -> Result.Ok(OpCode.Long1)
                else -> {
                    val value1 = value
                    Result.Err(value1)
                }
            }
    }
}

public fun <R : BufRead> readToNewline(r: R): Result<List<UByte>> {
    val data: List<UByte> = ArrayList(32)
    when (val tryResult = r.readUntil(10u, data)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    data.removeLastOrNull()
    if (data.last() == 13u) {
        data.removeLastOrNull()
    }
    return Result.Ok(data)
}

public sealed class Object {
    public data class Class(
        public var moduleName: String,
        public var clsName: String,
    ) : Object()
    public data class Int(
        public var value: Int,
    ) : Object()
    public data class Long(
        public var value: Long,
    ) : Object()
    public data class Float(
        public var value: Double,
    ) : Object()
    public data class Unicode(
        public var value: String,
    ) : Object()
    public data class Bool(
        public var value: Boolean,
    ) : Object()
    public data object None : Object()
    public data class Tuple(
        public var value: List<Object>,
    ) : Object()
    public data class List(
        public var value: List<Object>,
    ) : Object()
    public data object Mark : Object()
    public data class Dict(
        public var value: List<Pair<Object, Object>>,
    ) : Object()
    public data class Reduce(
        public var callable: Object,
        public var args: Object,
    ) : Object()
    public data class Build(
        public var callable: Object,
        public var args: Object,
    ) : Object()
    public data class PersistentLoad(
        public var value: Object,
    ) : Object()

    public fun rustEquals(other: Object): Boolean = this == other

    public fun unicode(): OResult<String> =
        when (this) {
            is Unicode -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun reduce(): OResult<Pair<Object, Object>> =
        when (this) {
            is Reduce -> {
                val callable = this.callable
                val args = this.args
                Result.Ok(Pair(callable, args))
            }
            else -> Result.Err(this)
        }

    public fun none(): OResult<Unit> =
        when (this) {
            Object.None -> Result.Ok(Unit)
            else -> Result.Err(this)
        }

    public fun persistentLoad(): OResult<Object> =
        when (this) {
            is PersistentLoad -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun bool(): OResult<Boolean> =
        when (this) {
            is Bool -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun int(): OResult<Int> =
        when (this) {
            is Int -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun intOrLong(): OResult<Long> =
        when (this) {
            is Int -> {
                val (t) = this
                Result.Ok(t.toLong())
            }
            is Long -> {
                val (t1) = this
                Result.Ok(t1)
            }
            else -> Result.Err(this)
        }

    public fun tuple(): OResult<List<Object>> =
        when (this) {
            is Tuple -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun dict(): OResult<List<Pair<Object, Object>>> =
        when (this) {
            is Dict -> {
                val (t) = this
                Result.Ok(t)
            }
            else -> Result.Err(this)
        }

    public fun cls(): OResult<Pair<String, String>> =
        when (this) {
            is Class -> {
                val moduleName = this.moduleName
                val className = this.clsName
                Result.Ok(Pair(moduleName, className))
            }
            else -> Result.Err(this)
        }

    public fun intoTensorInfo(name: Object, dirName: Path): Result<TensorInfo?> {
        val name2 =
            run {
                val matchVal = name.unicode()
                when (matchVal) {
                    is Result.Ok<*> -> {
                        val name1 = matchVal.value
                        name1
                    }
                    is Result.Err<*> -> {
                        return Result.Ok(null)
                    }
                }
            }
        val (callable, args) =
            run {
                val matchVal1 = this.reduce()
                when (matchVal1) {
                    is Result.Ok<*> -> {
                        val callableArgs = matchVal1.value
                        callableArgs
                    }
                    is Result.Err<*> -> {
                        return Result.Ok(null)
                    }
                }
            }
        val (callable2, args4) =
            when {
                callable is Object.Class && (callable as Object.Class).moduleName == "torch._tensor" && (callable as Object.Class).clsName == "_rebuild_from_type_v2" -> {
                    val moduleName = callable.moduleName
                    val className = callable.clsName
                    var args1 =
                        when (val tryResult = args.tuple()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    val callable1 = args1.remove(0)
                    val args2 = args1.remove(1)
                    Pair(callable1, args2)
                }
                callable is Object.Class && (callable as Object.Class).moduleName == "torch._utils" && (callable as Object.Class).clsName == "_rebuild_parameter" -> {
                    val moduleName1 = callable.moduleName
                    val className1 = callable.clsName
                    var args3 =
                        when (val tryResult = args.tuple()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    when (val tryResult = args3.remove(0).reduce()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                else -> Pair(callable, args)
            }
        when {
            callable2 is Object.Class && (callable2 as Object.Class).moduleName == "torch._utils" && (callable2 as Object.Class).clsName == "_rebuild_tensor_v2" -> {
                val moduleName2 = callable2.moduleName
                val className2 = callable2.clsName
            }
            else -> {
                return Result.Ok(null)
            }
        }
        val (layout, dtype, filePath, storageSize) =
            when (val tryResult = rebuildArgs(args4)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        return Result.Ok(TensorInfo(name = name2, dtype = dtype, layout = layout, path = "${dirName.toStringLossy()}/$filePath", storageSize = storageSize))
    }
}

public typealias OResult<T> = Result<T, Object>

public fun tryFromString(value: Object): Result<String, Object> =
    when (value) {
        is Object.Unicode -> {
            val (s) = value
            Result.Ok(s)
        }
        else -> {
            val other = value
            Result.Err(other)
        }
    }

public fun tryFromInt(value: Object): Result<Int, Object> =
    when {
        value is Object.Int && (value as Object.Int).value >= 0 -> {
            val (s) = value
            Result.Ok(s)
        }
        else -> {
            val other = value
            Result.Err(other)
        }
    }

public fun <T> tryFromVecT(value: Object, tryFromTFn: (Any?) -> Any?): Result<List<T>, Object> =
    when (value) {
        is Object.Tuple -> {
            val (values) = value
            // This does not return the appropriate value in the error case but instead return
            // the object related to the first error.
            values.map({ v -> tryFromTFn(v) }).collect<Result<List<T>, Object>>()
        }
        else -> {
            val other = value
            Result.Err(other)
        }
    }

public data class Stack(
    public var stack: MutableList<Object> = mutableListOf(),
    public var memo: MutableMap<UInt, Object> = MutableMap<UInt, Object>(),
) {
    public fun <R : BufRead> readLoop(r: R): Result<Unit> {
        while (true) {
            if (when (val tryResult = this.read(r)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            ) {
                break
            }
        }
        return Result.Ok(Unit)
    }

    public fun finalize(): Result<Object> = this.pop()

    public fun push(obj: Object) {
        this.stack.add(obj)
    }

    public fun pop(): Result<Object> =
        when (val matchVal = this.stack.removeLastOrNull()) {
            null -> bail("unexpected empty stack")
            else -> {
                val obj = matchVal
                Result.Ok(obj)
            }
        }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    public fun build(): Result<Unit> {
        val args =
            when (val tryResult = this.pop()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val obj =
            when (val tryResult = this.pop()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val obj3 =
            run {
                val matchVal = Pair(obj, args)
                when {
                    matchVal.first is Object.Dict && matchVal.second is Object.Dict -> {
                        val (obj1) = matchVal.first as Object.Dict
                        val (args1) = matchVal.second as Object.Dict
                        obj1.append(args1)
                        Object.Dict(obj1)
                    }
                    else -> {
                        val obj2 = matchVal.first
                        val args2 = matchVal.second
                        Object.Build(callable = obj2, args = args2)
                    }
                }
            }
        this.push(obj3)
        return Result.Ok(Unit)
    }

    public fun reduce(): Result<Unit> {
        val args =
            when (val tryResult = this.pop()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val callable =
            when (val tryResult = this.pop()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val reduced =
            when (callable) {
                is Object.Class -> {
                    val moduleName = callable.moduleName
                    val className = callable.clsName
                    if (moduleName == "collections" && (className == "OrderedDict" || className == "defaultdict")) {
                        // Note: have a separate ordered dict and a separate default dict.
                        Object.Dict(mutableListOf())
                    } else {
                        null
                    }
                }
                else -> null
            }
        val reduced1 = reduced.unwrapOrElse({ Object.Reduce(callable = callable, args = args) })
        this.push(reduced1)
        return Result.Ok(Unit)
    }

    public fun last(): Result<Object> =
        when (val matchVal = this.stack.lastOrNull()) {
            null -> bail("unexpected empty stack")
            else -> {
                val obj = matchVal
                Result.Ok(obj)
            }
        }

    public fun memoGet(id: UInt): Result<Object> =
        when (val matchVal = this.memo.get(id)) {
            null -> bail("missing object in memo {id}")
            else -> {
                val obj = matchVal
                // Maybe we should use refcounting rather than doing potential large clones here.
                Result.Ok(obj.clone())
            }
        }

    public fun memoPut(id: UInt): Result<Unit> {
        val obj =
            when (val tryResult = this.last()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }.clone()
        this.memo.put(id, obj)
        return Result.Ok(Unit)
    }

    public fun persistentLoad(id: Object): Result<Object> = Result.Ok(Object.PersistentLoad(id))

    public fun newObj(cls: Object, args: Object): Result<Object> = Result.Ok(Object.Reduce(callable = cls, args = args))

    public fun popToMarker(): Result<List<Object>> {
        var markIdx = null
        for ((idx, obj) in this.stack.mapIndexed { index, item -> Pair(index, item) }.rev()) {
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
                val markIdx1 = markIdx
                val objs = this.stack.splitOff(markIdx1 + 1)
                this.stack.removeLastOrNull()
                Result.Ok(objs)
            }
        }
    }

    public fun <R : BufRead> read(r: R): Result<Boolean> {
        val opCode2 =
            run {
                val matchVal =
                    OpCode.tryFrom(
                        when (val tryResult = r.readU8()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                when (matchVal) {
                    is Result.Ok<*> -> {
                        val opCode = matchVal.value
                        opCode
                    }
                    is Result.Err<*> -> {
                        val opCode1 = matchVal.error
                        bail("unknown op-code {op_code}")
                    }
                }
            }
        // println("op: {opCode:?}");
        // println("{:?}", self.stack);
        when (opCode2) {
            OpCode.Proto -> {
                val version =
                    when (val tryResult = r.readU8()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (VERBOSE) {
                    println("proto $version")
                }
            }
            OpCode.Global -> {
                val moduleName =
                    when (val tryResult = readToNewline(r)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val className =
                    when (val tryResult = readToNewline(r)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val moduleName1 = String.fromUtf8Lossy(moduleName).toString()
                val className1 = String.fromUtf8Lossy(className).toString()
                this.push(Object.Class(moduleName = moduleName1, clsName = className1))
            }
            OpCode.BinInt1 -> {
                val arg =
                    when (val tryResult = r.readU8()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Int(arg.toInt()))
            }
            OpCode.BinInt2 -> {
                val arg1 =
                    when (val tryResult = r.readU16<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Int(arg1.toInt()))
            }
            OpCode.BinInt -> {
                val arg2 =
                    when (val tryResult = r.readI32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Int(arg2))
            }
            OpCode.BinFloat -> {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                // https://github.com/pytorch/pytorch/blob/372d078f361e726bb4ac0884ac334b04c58179ef/torch/_weights_only_unpickler.py#L243
                val arg3 =
                    when (val tryResult = r.readF64<BigEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Float(arg3))
            }
            OpCode.BinUnicode -> {
                val len =
                    when (val tryResult = r.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val data = mutableListOf(0u.toUByte(), len.toInt())
                when (val tryResult = r.readExact(data)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
                val data1 =
                    when (val tryResult = String.fromUtf8(data).mapErr(E::wrap)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Unicode(data1))
            }
            OpCode.BinPersId -> {
                val id =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj =
                    when (val tryResult = this.persistentLoad(id)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(obj)
            }
            OpCode.Tuple -> {
                val objs =
                    when (val tryResult = this.popToMarker()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Tuple(objs))
            }
            OpCode.Tuple1 -> {
                val obj1 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Tuple(mutableListOf(obj1)))
            }
            OpCode.Tuple2 -> {
                val obj2 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj1 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Tuple(mutableListOf(obj1, obj2)))
            }
            OpCode.Tuple3 -> {
                val obj3 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj21 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj11 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(Object.Tuple(mutableListOf(obj11, obj21, obj3)))
            }
            OpCode.NewTrue -> this.push(Object.Bool(true))
            OpCode.NewFalse -> this.push(Object.Bool(false))
            OpCode.Append -> {
                val value =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val pylist =
                    when (val tryResult = this.last()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (pylist is Object.List) {
                    val (d) = pylist
                    d.push(value)
                } else {
                    bail("expected a list, got {pylist:?}")
                }
            }
            OpCode.Appends -> {
                val objs1 =
                    when (val tryResult = this.popToMarker()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val pylist1 =
                    when (val tryResult = this.last()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (pylist1 is Object.List) {
                    val (d1) = pylist1
                    d1.extend(objs1)
                } else {
                    bail("expected a list, got {pylist:?}")
                }
            }
            OpCode.SetItem -> {
                val value1 =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val key =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val pydict =
                    when (val tryResult = this.last()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (pydict is Object.Dict) {
                    val (d2) = pydict
                    d2.push(Pair(key, value1))
                } else {
                    bail("expected a dict, got {pydict:?}")
                }
            }
            OpCode.SetItems -> {
                val objs2 =
                    when (val tryResult = this.popToMarker()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val pydict1 =
                    when (val tryResult = this.last()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                if (pydict1 is Object.Dict) {
                    val (d3) = pydict1
                    if (objs2.size % 2 != 0) {
                        bail("setitems: not an even number of objects")
                    }
                    while (true) {
                        val matchVal1 = objs2.pop()
                        if (matchVal1 == null) break
                        val value2 = matchVal1
                        val key1 =
                            when (val tryResult = objs2.pop().context("empty objs")) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            }
                        d3.push(Pair(key1, value2))
                    }
                } else {
                    bail("expected a dict, got {pydict:?}")
                }
            }
            OpCode.None -> this.push(Object.None)
            OpCode.Stop -> {
                return Result.Ok(true)
            }
            OpCode.Build ->
                when (val tryResult = this.build()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            OpCode.EmptyDict -> this.push(Object.Dict(mutableListOf()))
            OpCode.Dict -> {
                val objs3 =
                    when (val tryResult = this.popToMarker()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val pydict2 = mutableListOf()
                if (objs3.size % 2 != 0) {
                    bail("setitems: not an even number of objects")
                }
                while (true) {
                    val matchVal2 = objs3.pop()
                    if (matchVal2 == null) break
                    val value3 = matchVal2
                    val key2 =
                        when (val tryResult = objs3.pop().context("empty objs")) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    pydict2.add(Pair(key2, value3))
                }
                this.push(Object.Dict(pydict2))
            }
            OpCode.Mark -> this.push(Object.Mark)
            OpCode.Reduce ->
                when (val tryResult = this.reduce()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            OpCode.EmptyTuple -> this.push(Object.Tuple(mutableListOf()))
            OpCode.EmptyList -> this.push(Object.List(mutableListOf()))
            OpCode.BinGet -> {
                val arg4 =
                    when (val tryResult = r.readU8()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj2 =
                    when (val tryResult = this.memoGet(arg4.toUInt())) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(obj2)
            }
            OpCode.LongBinGet -> {
                val arg5 =
                    when (val tryResult = r.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj3 =
                    when (val tryResult = this.memoGet(arg5)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(obj3)
            }
            OpCode.BinPut -> {
                val arg6 =
                    when (val tryResult = r.readU8()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = this.memoPut(arg6.toUInt())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            OpCode.LongBinPut -> {
                val arg7 =
                    when (val tryResult = r.readU32<LittleEndian>()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                when (val tryResult = this.memoPut(arg7)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
            OpCode.NewObj -> {
                val args =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val cls =
                    when (val tryResult = this.pop()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                val obj4 =
                    when (val tryResult = this.newObj(cls, args)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                this.push(obj4)
            }
            OpCode.Long1 -> {
                val nBytes =
                    when (val tryResult = r.readU8()) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                var v = 0
                // Decode the next n bytes in little endian
                for (i in 0 until nBytes) {
                    v = v or (
                        (
                            (
                                when (val tryResult = r.readU8()) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }
                            ).toLong()
                        ) shl (i * 8).toInt()
                    )
                }
                this.push(Object.Long(v))
            }
        }
        return Result.Ok(false)
    }

    public companion object {
        public fun empty(): Stack = Stack(stack = ArrayList(512), memo = mutableMapOf())
    }
}

public fun fromE(value: Object): E = E.Msg("conversion error on $value")

// https://github.com/pytorch/pytorch/blob/4eac43d046ded0f0a5a5fa8db03eb40f45bf656e/torch/_utils.py#L198
// Arguments: storage, storageOffset, size, stride, requiresGrad, backwardHooks
public fun rebuildArgs(args: Object): Result<List<Any?>> {
    val args1 =
        when (val tryResult = args.tuple()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val stride =
        when (val tryResult = Vec.tryFrom(args1.remove(3))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val size =
        when (val tryResult = Vec.tryFrom(args1.remove(2))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val offset =
        (
            when (val tryResult = args1.remove(1).intOrLong()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        ).toInt()
    val storage =
        when (val tryResult = args1.remove(0).persistentLoad()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val storage1 =
        when (val tryResult = storage.tuple()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val storageSize =
        (
            when (val tryResult = storage1.remove(4).intOrLong()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        ).toInt()
    val path =
        when (val tryResult = storage1.remove(2).unicode()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val (moduleName, className) =
        when (val tryResult = storage1.remove(1).cls()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dtype =
        when (className) {
            "FloatStorage" -> DType.F32
            "DoubleStorage" -> DType.F64
            "HalfStorage" -> DType.F16
            "BFloat16Storage" -> DType.BF16
            "ByteStorage" -> DType.U8
            "LongStorage" -> DType.I64
            else -> {
                val other = className
                bail("unsupported storage type {other}")
            }
        }
    val layout = Layout.new(Shape.from(size), stride, offset * dtype.sizeInBytes())
    return Result.Ok(listOf(layout, dtype, path, storageSize))
}

public data class TensorInfo(
    public var name: String = "",
    public var dtype: DType = DType.entries.first(),
    public var layout: Layout = Layout(),
    public var path: String = "",
    public var storageSize: Int = 0,
)

/**
 * Read the tensor info from a .pth file.
 *
 * # Arguments
 * * `file` - The path to the .pth file.
 * * `verbose` - Whether to print debug information.
 * * `key` - Optional key to retrieve `stateDict` from the pth file.
 */
public fun readPthTensorInfo(file: Path, verbose: Boolean, key: String?): Result<List<TensorInfo>> {
    val file1 =
        when (val tryResult = File.open(file)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val zipReader = BufReader.new(file1)
    val zip =
        when (val tryResult = ZipArchive.new(zipReader)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val zipFileNames = zip.fileNames().map({ f -> f.toString() }).toMutableList()
    val tensorInfos = mutableListOf<TensorInfo>()
    for (fileName in ArrayDeque(zipFileNames)) {
        if (!fileName.endsWith("data.pkl")) {
            continue
        }
        val dirName =
            PathBuf.from(
                when (val tryResult = fileName.takeIf { it.endsWith(".pkl") }?.removeSuffix(".pkl").context("no .pkl")) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
            )
        val reader =
            when (val tryResult = zip.byName(fileName)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val reader1 = BufReader.new(reader)
        val stack = Stack.empty()
        when (val tryResult = stack.readLoop(reader1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        val obj =
            when (val tryResult = stack.finalize()) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        if (VERBOSE || verbose) {
            println("$obj")
        }
        val obj2 =
            when (obj) {
                is Object.Build -> {
                    val callable = obj.callable
                    val args = obj.args
                    when (callable) {
                        is Object.Reduce -> {
                            val callable1 = callable.callable
                            when {
                                callable1 is Object.Class && (callable1 as Object.Class).moduleName == "__torch__" && (callable1 as Object.Class).clsName == "Module" -> {
                                    val moduleName = callable1.moduleName
                                    val className = callable1.clsName
                                    args
                                }
                                else -> {
                                    continue
                                }
                            }
                        }
                        else -> {
                            continue
                        }
                    }
                }
                else -> {
                    val obj1 = obj
                    obj1
                }
            }
        // If key is provided, then we need to extract the stateDict from the object.
        val obj3 =
            if (key != null) {
                val key1 = key
                if (obj2 is Object.Dict) {
                    val (keyValues) = obj2
                    when (val tryResult = ((keyValues.indexOf({ (k, _) -> k == Object.Unicode(key1.toOwned()) }).takeIf { it >= 0 })?.let({ (_, v) -> v })?.let { Result.Ok(it) } ?: Result.Err(E.Msg("key $key not found")))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                } else {
                    obj2
                }
            } else {
                obj2
            }
        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is stateDict by this stage.
        if (obj3 is Object.Dict) {
            val (keyValues1) = obj3
            for ((name, value) in ArrayDeque((keyValues1).toList())) {
                run {
                    val matchVal = value.intoTensorInfo(name, dirName)
                    when {
                        matchVal is Result.Ok<*> && matchVal.value != null -> tensorInfos.add(tensorInfo)
                        matchVal is Result.Ok<*> && matchVal.value == null -> {
                            val None = (matchVal as Result.Ok<*>).value
                        }
                        else -> {
                            val err = (matchVal as Result.Err<*>).error
                            println("skipping: $err")
                        }
                    }
                }
            }
        }
    }
    return Result.Ok(tensorInfos)
}

/**
 * Lazy tensor loader.
 */
public data class PthTensors(
    public var tensorInfos: MutableMap<String, TensorInfo> = MutableMap<String, TensorInfo>(),
    public var path: PathBuf = PathBuf(),
) {
    public fun get(name: String): Result<Tensor?> {
        val tensorInfo1 =
            when (val matchVal = this.tensorInfos.get(name)) {
                null -> {
                    return Result.Ok(null)
                }
                else -> {
                    val tensorInfo = matchVal
                    tensorInfo
                }
            }
        // We hope that the file has not changed since first reading it.
        val zipReader =
            BufReader.new(
                when (val tryResult = File.open(this.path)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                },
            )
        val zip =
            when (val tryResult = ZipArchive.new(zipReader)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val reader =
            when (val tryResult = zip.byName(tensorInfo1.path)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val isFortranContiguous = tensorInfo1.layout.isFortranContiguous()
        val rank = tensorInfo1.layout.shape().rank()
        // Reading the data is a bit tricky as it can be strided, for now only support the basic
        // case and when the tensor is fortran contiguous.
        if (!tensorInfo1.layout.isContiguous() && !isFortranContiguous) {
            bail("cannot retrieve non-contiguous tensors {:?}", tensorInfo1.layout)
        }
        val startOffset = tensorInfo1.layout.startOffset()
        if (startOffset > 0) {
            when (val tryResult = copy(reader.byRef().take(startOffset.toULong()), sink())) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        }
        val tensor =
            when (val tryResult = Tensor.fromReader(tensorInfo1.layout.shape().clone(), tensorInfo1.dtype, reader)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        return if (rank > 1 && isFortranContiguous) {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            val shapeReversed: Vec =
                tensorInfo1.layout
                    .dims()
                    .iter()
                    .rev()
                    .cloned()
                    .collect()
            val tensor1 =
                when (val tryResult = tensor.reshape(shapeReversed)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            val dimIndicesReversed: Vec = (0 until rank).rev().collect()
            val tensor2 =
                when (val tryResult = tensor1.permute(dimIndicesReversed)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            Result.Ok(tensor2)
        } else {
            Result.Ok(tensor)
        }
    }

    public companion object {
        public fun new(path: Path, key: String?): Result<PthTensors> {
            val tensorInfos =
                when (val tryResult = readPthTensorInfo(path, false, key)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val tensorInfos1 = tensorInfos.intoIter().map({ ti -> Pair(ti.name.toString(), ti) }).collect()
            val path1 = path.toOwned()
            return Result.Ok(PthTensors(tensorInfos = tensorInfos1, path = path1))
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
public fun readAllWithKey(path: Path, key: String?): Result<List<Pair<String, Tensor>>> {
    val pth =
        when (val tryResult = PthTensors.new(path, key)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val tensorNames = pth.tensorInfos.keys()
    val tensors = ArrayList<Pair<String, Tensor>>(tensorNames.len())
    for (name in tensorNames) {
        when (
            val matchVal =
                when (val tryResult = pth.get(name)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
        ) {
            null -> {}
            else -> {
                val tensor = matchVal
                tensors.add(Pair(name.toString(), tensor))
            }
        }
    }
    return Result.Ok(tensors)
}

/**
 * Read all the tensors from a PyTorch pth file.
 *
 * # Arguments
 * * `path` - Path to the pth file.
 */
public fun readAll(path: Path): Result<List<Pair<String, Tensor>>> = readAllWithKey(path, null)
