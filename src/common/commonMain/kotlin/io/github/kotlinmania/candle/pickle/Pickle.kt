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
import io.github.kotlinmania.candle.error.Error as E

// Just enough pickle support to be able to read PyTorch checkpoints.

public val VERBOSE: Boolean = false

// https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/
public sealed class OpCode(
    public val value: UByte,
) : TryFrom<UByte> {
    // https://github.com/python/cpython/blob/ed25f097160b5cbb0c9a1f9a746d2f1bbc96515a/Lib/pickletools.py#L2123
    public data object Proto : OpCode(0x80u)
    public data object Global : OpCode(99u)
    public data object BinPut : OpCode(113u)
    public data object LongBinPut : OpCode(114u)
    public data object EmptyTuple : OpCode(41u)
    public data object Reduce : OpCode(82u)
    public data object Mark : OpCode(40u)
    public data object BinUnicode : OpCode(88u)
    public data object BinInt : OpCode(74u)
    public data object Tuple : OpCode(116u)
    public data object BinPersId : OpCode(81u)
    public data object BinInt1 : OpCode(75u)
    public data object BinInt2 : OpCode(77u)
    public data object Tuple1 : OpCode(0x85u)
    public data object Tuple2 : OpCode(0x86u)
    public data object Tuple3 : OpCode(0x87u)
    public data object NewTrue : OpCode(0x88u)
    public data object NewFalse : OpCode(0x89u)
    public data object None : OpCode(78u)
    public data object BinGet : OpCode(104u)
    public data object LongBinGet : OpCode(106u)
    public data object SetItem : OpCode(115u)
    public data object SetItems : OpCode(117u)
    public data object EmptyDict : OpCode(125u)
    public data object Dict : OpCode(100u)
    public data object Build : OpCode(98u)
    public data object Stop : OpCode(46u)
    public data object NewObj : OpCode(0x81u)
    public data object EmptyList : OpCode(93u)
    public data object BinFloat : OpCode(71u)
    public data object Append : OpCode(97u)
    public data object Appends : OpCode(101u)
    public data object Long1 : OpCode(0x8au)

    public fun __rustEq(other: OpCode): Boolean = this == other

    public companion object {
        // Avoid using FromPrimitive so as not to drag another dependency.
        public fun tryFrom(value: UByte): Result<OpCode> =
            when (value) {
                0x80 -> Result.success(OpCode.Proto)
                99u -> Result.success(OpCode.Global)
                113u -> Result.success(OpCode.BinPut)
                114u -> Result.success(OpCode.LongBinPut)
                41u -> Result.success(OpCode.EmptyTuple)
                82u -> Result.success(OpCode.Reduce)
                40u -> Result.success(OpCode.Mark)
                88u -> Result.success(OpCode.BinUnicode)
                74u -> Result.success(OpCode.BinInt)
                116u -> Result.success(OpCode.Tuple)
                81u -> Result.success(OpCode.BinPersId)
                75u -> Result.success(OpCode.BinInt1)
                77u -> Result.success(OpCode.BinInt2)
                78u -> Result.success(OpCode.None)
                0x85 -> Result.success(OpCode.Tuple1)
                0x86 -> Result.success(OpCode.Tuple2)
                0x87 -> Result.success(OpCode.Tuple3)
                0x88 -> Result.success(OpCode.NewTrue)
                0x89 -> Result.success(OpCode.NewFalse)
                104u -> Result.success(OpCode.BinGet)
                106u -> Result.success(OpCode.LongBinGet)
                115u -> Result.success(OpCode.SetItem)
                117u -> Result.success(OpCode.SetItems)
                125u -> Result.success(OpCode.EmptyDict)
                100u -> Result.success(OpCode.EmptyDict)
                98u -> Result.success(OpCode.Build)
                46u -> Result.success(OpCode.Stop)
                0x81 -> Result.success(OpCode.NewObj)
                93u -> Result.success(OpCode.EmptyList)
                71u -> Result.success(OpCode.BinFloat)
                97u -> Result.success(OpCode.Append)
                101u -> Result.success(OpCode.Appends)
                0x8a -> Result.success(OpCode.Long1)
                else -> {
                    val value1 = value
                    Result.failure(value1)
                }
            }
    }
}

public fun <R : BufRead> readToNewline(r: R): Result<List<UByte>> {
    val data: List<UByte>  = ArrayList(32)
    r.readUntil(10u, data).getOrElse { return Result.failure(it) }
    data.removeLastOrNull()
    if (data.last() == 13u) {
        data.removeLastOrNull()
    }
    return Result.success(data)
}

public sealed class Object {
    public data class Class(public var moduleName: String, public var className: String) : Object()
    public data class Int(public var value: Int) : Object()
    public data class Long(public var value: Long) : Object()
    public data class Float(public var value: Double) : Object()
    public data class Unicode(public var value: String) : Object()
    public data class Bool(public var value: Boolean) : Object()
    public data object None : Object()
    public data class Tuple(public var value: List<Object>) : Object()
    public data class List(public var value: List<Object>) : Object()
    public data object Mark : Object()
    public data class Dict(public var value: List<Pair<Object, Object>>) : Object()
    public data class Reduce(public var callable: Object, public var args: Object) : Object()
    public data class Build(public var callable: Object, public var args: Object) : Object()
    public data class PersistentLoad(public var value: Object) : Object()

    public fun __rustEq(other: Object): Boolean = this == other

    public open fun unicode(): OResult<String> =
        when (this) {
            is Unicode -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun reduce(): OResult<Pair<Object, Object>> =
        when (this) {
            is Reduce -> {
                val callable = this.callable
                val args = this.args
                Result.success(Pair(callable, args))
            }
            else -> Result.failure(this)
        }

    public open fun none(): OResult<Unit> =
        when (this) {
            Object.None -> Result.success(Unit)
            else -> Result.failure(this)
        }

    public open fun persistentLoad(): OResult<Object> =
        when (this) {
            is PersistentLoad -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun bool(): OResult<Boolean> =
        when (this) {
            is Bool -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun int(): OResult<Int> =
        when (this) {
            is Int -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun intOrLong(): OResult<Long> =
        when (this) {
            is Int -> {
                val (t) = this
                Result.success(t.toLong())
            }
            is Long -> {
                val (t1) = this
                Result.success(t1)
            }
            else -> Result.failure(this)
        }

    public open fun tuple(): OResult<List<Object>> =
        when (this) {
            is Tuple -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun dict(): OResult<List<Pair<Object, Object>>> =
        when (this) {
            is Dict -> {
                val (t) = this
                Result.success(t)
            }
            else -> Result.failure(this)
        }

    public open fun cls(): OResult<Pair<String, String>> =
        when (this) {
            is Class -> {
                val moduleName = this.moduleName
                val className = this.className
                Result.success(Pair(moduleName, className))
            }
            else -> Result.failure(this)
        }

    public open fun intoTensorInfo(name: Object, dirName: Path): Result<TensorInfo?> {
        val name2 =
            run {
                val matchVal = name.unicode()
                if (matchVal.isSuccess) {
                    val name1 = matchVal.getOrThrow()
                    name1
                } else {
                    return Result.success(null)
                }
            }
        val (callable, args) =
            run {
                val matchVal1 = this.reduce()
                if (matchVal1.isSuccess) {
                    val callableArgs = matchVal1.getOrThrow()
                    callableArgs
                } else {
                    return Result.success(null)
                }
            }
        val (callable2, args4) =
            when {
                callable is Object.Class && (callable as Object.Class).moduleName == "torch._tensor" && (callable as Object.Class).className == "_rebuild_from_type_v2" -> {
                    val moduleName = callable.moduleName
                    val className = callable.className
                    var args1 = args.tuple().getOrElse { return Result.failure(it) }
                    val callable1 = args1.remove(0)
                    val args2 = args1.remove(1)
                    Pair(callable1, args2)
                }
                callable is Object.Class && (callable as Object.Class).moduleName == "torch._utils" && (callable as Object.Class).className == "_rebuild_parameter" -> {
                    val moduleName1 = callable.moduleName
                    val className1 = callable.className
                    var args3 = args.tuple().getOrElse { return Result.failure(it) }
                    args3.remove(0).reduce().getOrElse { return Result.failure(it) }
                }
                else -> Pair(callable, args)
            }
        when {
            callable2 is Object.Class && (callable2 as Object.Class).moduleName == "torch._utils" && (callable2 as Object.Class).className == "_rebuild_tensor_v2" -> {
                val moduleName2 = callable2.moduleName
                val className2 = callable2.className
            }
            else -> { return Result.success(null) }
        }
        val (layout, dtype, filePath, storageSize) = rebuildArgs(args4).getOrElse { return Result.failure(it) }
        return Result.success(TensorInfo(name = name2, dtype = dtype, layout = layout, path = "${dirName.toStringLossy()}/$filePath", storageSize = storageSize))
    }
}

public typealias OResult<T> = Result<T>

public fun tryFromString(value: Object): Result<String> =
    when (value) {
        is Object.Unicode -> {
            val (s) = value
            Result.success(s)
        }
        else -> {
            val other = value
            Result.failure(other)
        }
    }

public fun tryFromInt(value: Object): Result<Int> =
    when {
        value is Object.Int && (value as Object.Int).value >= 0 -> {
            val (s) = value
            Result.success(s)
        }
        else -> {
            val other = value
            Result.failure(other)
        }
    }

public fun <T : TryFrom<Object, Object>> tryFromVecT(value: Object, tryFromTFn: (Any?) -> Any?): Result<List<T>> {
    return when (value) {
        is Object.Tuple -> {
            val (values) = value
            // This does not return the appropriate value in the error case but instead return
            // the object related to the first error.
            values.map({ v -> tryFromTFn(v) }).collect<Result<List<T>>>()
        }
        else -> {
            val other = value
            Result.failure(other)
        }
    }
}

public open class Stack(
    public var stack: MutableList<Object>,
    public var memo: Map<UInt, Object>,
) {
    public open fun stack(): List<Object> = this.stack.asSlice()

    public open fun <R : BufRead> readLoop(r: R): Result<Unit> {
        while (true) {
            if (this.read(r).getOrElse { return Result.failure(it) }) {
                break
            }
        }
        return Result.success(Unit)
    }

    public open fun finalize(): Result<Object> = this.pop()

    public open fun push(obj: Object) {
        this.stack.add(obj)
    }

    public open fun pop(): Result<Object> =
        when (val matchVal = this.stack.removeLastOrNull()) {
            null -> bail("unexpected empty stack")
            else -> {
                val obj = matchVal
                Result.success(obj)
            }
        }

    // https://docs.juliahub.com/Pickle/LAUNc/0.1.0/opcode/#Pickle.OpCodes.BUILD
    public open fun build(): Result<Unit> {
        val args = this.pop().getOrElse { return Result.failure(it) }
        val obj = this.pop().getOrElse { return Result.failure(it) }
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
        return Result.success(Unit)
    }

    public open fun reduce(): Result<Unit> {
        val args = this.pop().getOrElse { return Result.failure(it) }
        val callable = this.pop().getOrElse { return Result.failure(it) }
        val reduced =
            when (callable) {
                is Object.Class -> {
                    val moduleName = callable.moduleName
                    val className = callable.className
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
        return Result.success(Unit)
    }

    public open fun last(): Result<Object> =
        when (val matchVal = this.stack.lastOrNull()) {
            null -> bail("unexpected empty stack")
            else -> {
                val obj = matchVal
                Result.success(obj)
            }
        }

    public open fun memoGet(id: UInt): Result<Object> =
        when (val matchVal = this.memo.get(id)) {
            null -> bail("missing object in memo {id}")
            else -> {
                val obj = matchVal
                // Maybe we should use refcounting rather than doing potential large clones here.
                Result.success(obj.clone())
            }
        }

    public open fun memoPut(id: UInt): Result<Unit> {
        val obj = this.last().getOrElse { return Result.failure(it) }.clone()
        this.memo.insert(id, obj)
        return Result.success(Unit)
    }

    public open fun persistentLoad(id: Object): Result<Object> = Result.success(Object.PersistentLoad(id))

    public open fun newObj(cls: Object, args: Object): Result<Object> = Result.success(Object.Reduce(callable = cls, args = args))

    public open fun popToMarker(): Result<List<Object>> {
        var markIdx = null
        for ((idx, obj) in this.stack.enumerate().rev()) {
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
                Result.success(objs)
            }
        }
    }

    public open fun <R : BufRead> read(r: R): Result<Boolean> {
        val opCode2 =
            run {
                val matchVal = OpCode.tryFrom(r.readU8().getOrElse { return Result.failure(it) })
                if (matchVal.isSuccess) {
                    val opCode = matchVal.getOrThrow()
                    opCode
                } else {
                    val opCode1 = matchVal.exceptionOrNull()!!
                    bail("unknown op-code {op_code}")
                }
            }
        // println("op: {opCode:?}");
        // println("{:?}", self.stack);
        when (opCode2) {
            OpCode.Proto -> {
                val version = r.readU8().getOrElse { return Result.failure(it) }
                if (VERBOSE) {
                    println("proto $version")
                }
            }
            OpCode.Global -> {
                val moduleName = readToNewline(r).getOrElse { return Result.failure(it) }
                val className = readToNewline(r).getOrElse { return Result.failure(it) }
                val moduleName1 = String.fromUtf8Lossy(moduleName).toString()
                val className1 = String.fromUtf8Lossy(className).toString()
                this.push(Object.Class(moduleName = moduleName1, className = className1))
            }
            OpCode.BinInt1 -> {
                val arg = r.readU8().getOrElse { return Result.failure(it) }
                this.push(Object.Int(arg.toInt()))
            }
            OpCode.BinInt2 -> {
                val arg1 = r.readU16<LittleEndian>().getOrElse { return Result.failure(it) }
                this.push(Object.Int(arg1.toInt()))
            }
            OpCode.BinInt -> {
                val arg2 = r.readI32<LittleEndian>().getOrElse { return Result.failure(it) }
                this.push(Object.Int(arg2))
            }
            OpCode.BinFloat -> {
                // Somehow floats are encoded using BigEndian whereas int types use LittleEndian.
                // https://github.com/python/cpython/blob/0c80da4c14d904a367968955544dd6ae58c8101c/Lib/pickletools.py#L855
                // https://github.com/pytorch/pytorch/blob/372d078f361e726bb4ac0884ac334b04c58179ef/torch/_weights_only_unpickler.py#L243
                val arg3 = r.readF64<BigEndian>().getOrElse { return Result.failure(it) }
                this.push(Object.Float(arg3))
            }
            OpCode.BinUnicode -> {
                val len = r.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                val data = mutableListOf(0u.toUByte(), len.toInt())
                r.readExact(data).getOrElse { return Result.failure(it) }
                val data1 = String.fromUtf8(data).mapErr(E::wrap).getOrElse { return Result.failure(it) }
                this.push(Object.Unicode(data1))
            }
            OpCode.BinPersId -> {
                val id = this.pop().getOrElse { return Result.failure(it) }
                val obj = this.persistentLoad(id).getOrElse { return Result.failure(it) }
                this.push(obj)
            }
            OpCode.Tuple -> {
                val objs = this.popToMarker().getOrElse { return Result.failure(it) }
                this.push(Object.Tuple(objs))
            }
            OpCode.Tuple1 -> {
                val obj1 = this.pop().getOrElse { return Result.failure(it) }
                this.push(Object.Tuple(mutableListOf(obj1)))
            }
            OpCode.Tuple2 -> {
                val obj2 = this.pop().getOrElse { return Result.failure(it) }
                val obj1 = this.pop().getOrElse { return Result.failure(it) }
                this.push(Object.Tuple(mutableListOf(obj1, obj2)))
            }
            OpCode.Tuple3 -> {
                val obj3 = this.pop().getOrElse { return Result.failure(it) }
                val obj21 = this.pop().getOrElse { return Result.failure(it) }
                val obj11 = this.pop().getOrElse { return Result.failure(it) }
                this.push(Object.Tuple(mutableListOf(obj11, obj21, obj3)))
            }
            OpCode.NewTrue -> this.push(Object.Bool(true))
            OpCode.NewFalse -> this.push(Object.Bool(false))
            OpCode.Append -> {
                val value = this.pop().getOrElse { return Result.failure(it) }
                val pylist = this.last().getOrElse { return Result.failure(it) }
                if (pylist is Object.List) {
                    val (d) = pylist
                    d.push(value)
                } else {
                    bail("expected a list, got {pylist:?}")
                }
            }
            OpCode.Appends -> {
                val objs1 = this.popToMarker().getOrElse { return Result.failure(it) }
                val pylist1 = this.last().getOrElse { return Result.failure(it) }
                if (pylist1 is Object.List) {
                    val (d1) = pylist1
                    d1.extend(objs1)
                } else {
                    bail("expected a list, got {pylist:?}")
                }
            }
            OpCode.SetItem -> {
                val value1 = this.pop().getOrElse { return Result.failure(it) }
                val key = this.pop().getOrElse { return Result.failure(it) }
                val pydict = this.last().getOrElse { return Result.failure(it) }
                if (pydict is Object.Dict) {
                    val (d2) = pydict
                    d2.push(Pair(key, value1))
                } else {
                    bail("expected a dict, got {pydict:?}")
                }
            }
            OpCode.SetItems -> {
                val objs2 = this.popToMarker().getOrElse { return Result.failure(it) }
                val pydict1 = this.last().getOrElse { return Result.failure(it) }
                if (pydict1 is Object.Dict) {
                    val (d3) = pydict1
                    if (objs2.size % 2 != 0) {
                        bail("setitems: not an even number of objects")
                    }
                    while (true) {
                        val matchVal1 = objs2.pop()
                        if (matchVal1 == null) break
                        val value2 = matchVal1
                        val key1 = objs2.pop().context("empty objs").getOrElse { return Result.failure(it) }
                        d3.push(Pair(key1, value2))
                    }
                } else {
                    bail("expected a dict, got {pydict:?}")
                }
            }
            OpCode.None -> this.push(Object.None)
            OpCode.Stop -> {
                return Result.success(true)
            }
            OpCode.Build -> this.build().getOrElse { return Result.failure(it) }
            OpCode.EmptyDict -> this.push(Object.Dict(mutableListOf()))
            OpCode.Dict -> {
                val objs3 = this.popToMarker().getOrElse { return Result.failure(it) }
                val pydict2 = mutableListOf()
                if (objs3.size % 2 != 0) {
                    bail("setitems: not an even number of objects")
                }
                while (true) {
                    val matchVal2 = objs3.pop()
                    if (matchVal2 == null) break
                    val value3 = matchVal2
                    val key2 = objs3.pop().context("empty objs").getOrElse { return Result.failure(it) }
                    pydict2.add(Pair(key2, value3))
                }
                this.push(Object.Dict(pydict2))
            }
            OpCode.Mark -> this.push(Object.Mark)
            OpCode.Reduce -> this.reduce().getOrElse { return Result.failure(it) }
            OpCode.EmptyTuple -> this.push(Object.Tuple(mutableListOf()))
            OpCode.EmptyList -> this.push(Object.List(mutableListOf()))
            OpCode.BinGet -> {
                val arg4 = r.readU8().getOrElse { return Result.failure(it) }
                val obj2 = this.memoGet(arg4.toUInt()).getOrElse { return Result.failure(it) }
                this.push(obj2)
            }
            OpCode.LongBinGet -> {
                val arg5 = r.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                val obj3 = this.memoGet(arg5).getOrElse { return Result.failure(it) }
                this.push(obj3)
            }
            OpCode.BinPut -> {
                val arg6 = r.readU8().getOrElse { return Result.failure(it) }
                this.memoPut(arg6.toUInt()).getOrElse { return Result.failure(it) }
            }
            OpCode.LongBinPut -> {
                val arg7 = r.readU32<LittleEndian>().getOrElse { return Result.failure(it) }
                this.memoPut(arg7).getOrElse { return Result.failure(it) }
            }
            OpCode.NewObj -> {
                val args = this.pop().getOrElse { return Result.failure(it) }
                val cls = this.pop().getOrElse { return Result.failure(it) }
                val obj4 = this.newObj(cls, args).getOrElse { return Result.failure(it) }
                this.push(obj4)
            }
            OpCode.Long1 -> {
                val nBytes = r.readU8().getOrElse { return Result.failure(it) }
                var v = 0
                // Decode the next n bytes in little endian
                for (i in 0 until nBytes) {
                    v = v or (((r.readU8().getOrElse { return Result.failure(it) }).toLong()) shl (i * 8).toInt())
                }
                this.push(Object.Long(v))
            }
        }
        return Result.success(false)
    }

    public companion object {
        public fun empty(): Stack = Stack(stack = ArrayList(512), memo = HashMap.new())
    }
}

public fun fromE(value: Object): E = E.Msg("conversion error on $value")

// https://github.com/pytorch/pytorch/blob/4eac43d046ded0f0a5a5fa8db03eb40f45bf656e/torch/_utils.py#L198
// Arguments: storage, storageOffset, size, stride, requiresGrad, backwardHooks
public fun rebuildArgs(args: Object): Result<List<Any?>> {
    val args1 = args.tuple().getOrElse { return Result.failure(it) }
    val stride = Vec.tryFrom(args1.remove(3)).getOrElse { return Result.failure(it) }
    val size = Vec.tryFrom(args1.remove(2)).getOrElse { return Result.failure(it) }
    val offset = (args1.remove(1).intOrLong().getOrElse { return Result.failure(it) }).toInt()
    val storage = args1.remove(0).persistentLoad().getOrElse { return Result.failure(it) }
    val storage1 = storage.tuple().getOrElse { return Result.failure(it) }
    val storageSize = (storage1.remove(4).intOrLong().getOrElse { return Result.failure(it) }).toInt()
    val path = storage1.remove(2).unicode().getOrElse { return Result.failure(it) }
    val (moduleName, className) = storage1.remove(1).cls().getOrElse { return Result.failure(it) }
    val dtype =
        when (val matchVal = className.asStr()) {
            "FloatStorage" -> DType.F32
            "DoubleStorage" -> DType.F64
            "HalfStorage" -> DType.F16
            "BFloat16Storage" -> DType.BF16
            "ByteStorage" -> DType.U8
            "LongStorage" -> DType.I64
            else -> {
                val other = matchVal
                bail("unsupported storage type {other}")
            }
        }
    val layout = Layout.new(Shape.from(size), stride, offset * dtype.sizeInBytes())
    return Result.success(listOf(layout, dtype, path, storageSize))
}

public open class TensorInfo(
    public var name: String,
    public var dtype: DType,
    public var layout: Layout,
    public var path: String,
    public var storageSize: Int,
)

/**
 * Read the tensor info from a .pth file.
 *
 * # Arguments
 * * `file` - The path to the .pth file.
 * * `verbose` - Whether to print debug information.
 * * `key` - Optional key to retrieve `stateDict` from the pth file.
 */
public fun <P : AsRef<Path>> readPthTensorInfo(file: P, verbose: Boolean, key: String?): Result<List<TensorInfo>> {
    val file1 = File.open(file).getOrElse { return Result.failure(it) }
    val zipReader = BufReader.new(file1)
    val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
    val zipFileNames = zip.fileNames().map({ f -> f.toString() }).collect<List<String>>()
    val tensorInfos = mutableListOf<TensorInfo>()
    for (fileName in zipFileNames.iter()) {
        if (!fileName.endsWith("data.pkl")) {
            continue
        }
        val dirName = PathBuf.from(fileName.takeIf { it.endsWith(".pkl") }?.removeSuffix(".pkl").context("no .pkl").getOrElse { return Result.failure(it) })
        val reader = zip.byName(fileName).getOrElse { return Result.failure(it) }
        val reader1 = BufReader.new(reader)
        val stack = Stack.empty()
        stack.readLoop(reader1).getOrElse { return Result.failure(it) }
        val obj = stack.finalize().getOrElse { return Result.failure(it) }
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
                                callable1 is Object.Class && (callable1 as Object.Class).moduleName == "__torch__" && (callable1 as Object.Class).className == "Module" -> {
                                    val moduleName = callable1.moduleName
                                    val className = callable1.className
                                    args
                                }
                                else -> { continue }
                            }
                        }
                        else -> { continue }
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
                    (keyValues.indexOf({ (k, _) -> k == Object.Unicode(key1.toOwned()) }).takeIf { it >= 0 })?.let({ (_, v) -> v }).okOrElse({ E.Msg("key $key not found") }).getOrElse { return Result.failure(it) }
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
                        matchVal is Ok && matchVal.value != null -> {
                            val tensorInfo = matchVal.value
                            tensorInfos.add(tensorInfo)
                        }
                        matchVal is Ok && matchVal.value == null -> {
                            val (None) = matchVal
                        }
                        else -> {
                            val (err) = matchVal
                            println("skipping: $err")
                        }
                    }
                }
            }
        }
    }
    return Result.success(tensorInfos)
}

/**
 * Lazy tensor loader.
 */
public open class PthTensors(
    public var tensorInfos: Map<String, TensorInfo>,
    public var path: PathBuf,
) {
    public open fun tensorInfos(): Map<String, TensorInfo> = this.tensorInfos

    public open fun get(name: String): Result<Tensor?> {
        val tensorInfo1 =
            when (val matchVal = this.tensorInfos.get(name)) {
                null -> { return Result.success(null) }
                else -> {
                    val tensorInfo = matchVal
                    tensorInfo
                }
            }
        // We hope that the file has not changed since first reading it.
        val zipReader = BufReader.new(File.open(this.path).getOrElse { return Result.failure(it) })
        val zip = ZipArchive.new(zipReader).getOrElse { return Result.failure(it) }
        val reader = zip.byName(tensorInfo1.path).getOrElse { return Result.failure(it) }
        val isFortranContiguous = tensorInfo1.layout.isFortranContiguous()
        val rank = tensorInfo1.layout.shape().rank()
        // Reading the data is a bit tricky as it can be strided, for now only support the basic
        // case and when the tensor is fortran contiguous.
        if (!tensorInfo1.layout.isContiguous() && !isFortranContiguous) {
            bail("cannot retrieve non-contiguous tensors {:?}", tensorInfo1.layout)
        }
        val startOffset = tensorInfo1.layout.startOffset()
        if (startOffset > 0) {
            copy(reader.byRef().take(startOffset.toULong()), sink()).getOrElse { return Result.failure(it) }
        }
        val tensor = Tensor.fromReader(tensorInfo1.layout.shape().clone(), tensorInfo1.dtype, reader).getOrElse { return Result.failure(it) }
        return if (rank > 1 && isFortranContiguous) {
            // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
            val shapeReversed: Vec = tensorInfo1.layout.dims().iter().rev().cloned().collect()
            val tensor1 = tensor.reshape(shapeReversed).getOrElse { return Result.failure(it) }
            // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
            val dimIndicesReversed: Vec = (0 until rank).rev().collect()
            val tensor2 = tensor1.permute(dimIndicesReversed).getOrElse { return Result.failure(it) }
            Result.success(tensor2)
        } else {
            Result.success(tensor)
        }
    }

    public companion object {
        public fun <P : AsRef<Path>> new(path: P, key: String?): Result<PthTensors> {
            val tensorInfos = readPthTensorInfo(path.asRef(), false, key).getOrElse { return Result.failure(it) }
            val tensorInfos1 = tensorInfos.intoIter().map({ ti -> Pair(ti.name.toString(), ti) }).collect()
            val path1 = path.asRef().toOwned()
            return Result.success(PthTensors(tensorInfos = tensorInfos1, path = path1))
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
public fun <P : AsRef<Path>> readAllWithKey(path: P, key: String?): Result<List<Pair<String, Tensor>>> {
    val pth = PthTensors.new(path, key).getOrElse { return Result.failure(it) }
    val tensorNames = pth.tensorInfos.keys()
    val tensors = ArrayList<Pair<String, Tensor>>(tensorNames.len())
    for (name in tensorNames) {
        when (val matchVal = pth.get(name).getOrElse { return Result.failure(it) }) {
            null -> {}
            else -> {
                val tensor = matchVal
                tensors.add(Pair(name.toString(), tensor))
            }
        }
    }
    return Result.success(tensors)
}

/**
 * Read all the tensors from a PyTorch pth file.
 *
 * # Arguments
 * * `path` - Path to the pth file.
 */
public fun <P : AsRef<Path>> readAll(path: P): Result<List<Pair<String, Tensor>>> = readAllWithKey(path, null)
