// port-lint: source safetensors.rs
package io.github.kotlinmania.candle.safetensors

import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.dummydtype.F4
import io.github.kotlinmania.candle.dummydtype.F6e2m3
import io.github.kotlinmania.candle.dummydtype.F6e3m2
import io.github.kotlinmania.candle.dummydtype.F8e8m0
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.metalbackend.MetalStorage
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.candle.tensor.fromStorage
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.memmap2.Mmap
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.borrow.Cow
import io.github.kotlinmania.rustlang.std.path.Path
import io.github.kotlinmania.yoke.Yoke
import io.github.kotlinmania.safetensors.tensor as st

// Module to load `safetensor` files into CPU/GPU memory.
//
// There are multiple ways to load tensors from safetensor files:
// - `load` function for loading directly into memory and returning a HashMap of tensors
// - `MmapedSafetensors` for memory mapping files and avoiding full allocation
// - `SliceSafetensors` for working with in-memory buffers
// - `BufferedSafetensors` for owning a buffer of data
//
// Tensors can also be serialized to safetensor format using the `save` function or
// `Tensor.saveSafetensors` method.

public fun fromDtype(value: DType): st.Dtype =
    when (value) {
        DType.u8 -> st.Dtype.U8
        DType.u32 -> st.Dtype.U32
        DType.i16 -> st.Dtype.I16
        DType.i32 -> st.Dtype.I32
        DType.i64 -> st.Dtype.I64
        DType.bf16 -> st.Dtype.BF16
        DType.f16 -> st.Dtype.F16
        DType.f32 -> st.Dtype.F32
        DType.f64 -> st.Dtype.F64
        DType.f8e4m3 -> st.Dtype.F8_E4M3
        DType.f6e2m3 -> st.Dtype.F6_E2M3
        DType.f6e3m2 -> st.Dtype.F6_E3M2
        DType.f4 -> st.Dtype.F4
        DType.f8e8m0 -> st.Dtype.F8_E8M0
    }

public fun tryFromDType(value: st.Dtype): Result<DType> =
    when (value) {
        st.Dtype.u8 -> Ok(DType.U8)
        st.Dtype.u32 -> Ok(DType.U32)
        st.Dtype.i16 -> Ok(DType.I16)
        st.Dtype.i32 -> Ok(DType.I32)
        st.Dtype.i64 -> Ok(DType.I64)
        st.Dtype.bf16 -> Ok(DType.BF16)
        st.Dtype.f16 -> Ok(DType.F16)
        st.Dtype.f32 -> Ok(DType.F32)
        st.Dtype.f64 -> Ok(DType.F64)
        st.Dtype.f8E4m3 -> Ok(DType.F8E4M3)
        st.Dtype.f6E2m3 -> Ok(DType.F6E2M3)
        st.Dtype.f6E3m2 -> Ok(DType.F6E3M2)
        st.Dtype.f4 -> Ok(DType.F4)
        st.Dtype.f8E8m0 -> Ok(DType.F8E8M0)
        else -> {
            val dtype = value
            Err(Error.UnsupportedSafeTensorDtype(dtype))
        }
    }

public fun Tensor.dtypeView(): st.Dtype = this.dtype().into()

public fun Tensor.shapeView(): List<Int> = this.shape().dims()

public fun Tensor.data(): Cow<List<UByte>> {
    // This copies data from GPU to CPU.
    // Note: Avoid the unwrap here.
    return Cow.Owned(convertBack(this).unwrap())
}

public fun Tensor.dataLen(): Int {
    val n: Int = this.shape().elemCount()
    val bytesPerElement = this.dtype().sizeInBytes()
    return n * bytesPerElement
}

public fun Tensor.dtype(): st.Dtype = this.dtype().into()

public fun Tensor.shape(): List<Int> = this.dims()

public fun <P : AsRef<Path>> Tensor.saveSafetensors(name: String, filename: P): Result<Unit> {
    val data = listOf(Pair(name, this.clone()))
    return Ok(
        when (val tryResult = st.serializeToFile(data, null, filename.asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        },
    )
}

public fun <T : WithDType> convertSlice(data: List<UByte>, shape: List<Int>, device: Device): Result<Tensor> {
    val sizeInBytes = T.DTYPE.sizeInBytes()
    val elemCount = data.size / sizeInBytes
    return if ((data.asPtr().toInt()).isMultipleOf(sizeInBytes)) {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        val data1: List<T> = fromRawParts(constPtrFromRustCast(data.asPtr()), elemCount)
        Tensor.fromSlice(data1, shape, device)
    } else {
        // Note: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        val c: List<T> = ArrayList(elemCount)
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        run {
            copyNonoverlapping(data.asPtr(), mutPtrFromRustCast(c.asMutPtr()), data.size)
            c.setLen(elemCount)
        }
        Tensor.fromSlice(c, shape, device)
    }
}

public fun <T, U : WithDType> convertSliceWithCast(data: List<UByte>, shape: List<Int>, device: Device, conv: (T) -> Result<U>, typeLayoutT: TypeLayout<T>): Result<Tensor> {
    val sizeInBytes = sizeOf<T>(typeLayoutT = typeLayoutT)
    val elemCount = data.size / sizeInBytes
    return if ((data.asPtr().toInt()).isMultipleOf(sizeInBytes)) {
        // SAFETY This is safe because we just checked that this
        // was correctly aligned.
        val data1: List<T> = fromRawParts(constPtrFromRustCast(data.asPtr()), elemCount)
        val data2 =
            when (val tryResult = data1.map({ t -> conv(t) }).collect<Result<Vec>>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        Tensor.fromVec(data2, shape, device)
    } else {
        // Note: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        val c: List<T> = ArrayList(elemCount)
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        run {
            copyNonoverlapping(data.asPtr(), mutPtrFromRustCast(c.asMutPtr()), data.size)
            c.setLen(elemCount)
        }
        val c1 =
            when (val tryResult = c.map(conv).collect<Result<Vec>>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        Tensor.fromVec(c1, shape, device)
    }
}

public fun <T, U : WithDType> convertWithCast(view: st.TensorView, device: Device, conv: (T) -> Result<U>, typeLayoutT: TypeLayout<T>): Result<Tensor> = convertSliceWithCast<T, U, (T) -> Result<U>>(view.data(), view.shape(), device, conv, typeLayoutT = typeLayoutT)

public fun <T : WithDType> convert(view: st.TensorView, device: Device): Result<Tensor> = convertSlice<T>(view.data(), view.shape(), device)

public fun <T : WithDType> convertBack(vs: List<T>): List<UByte> {
    val sizeInBytes = T.DTYPE.sizeInBytes()
    val length = vs.size * sizeInBytes
    val capacity = vs.capacity() * sizeInBytes
    val ptr = mutPtrFromRustCast(vs.asMutPtr())
    // Don't run the destructor for Vec<T>
    forget(vs)
    // SAFETY:
    //
    // Every T is larger than u8, so there is no issue regarding alignment.
    // This re-interpret the Vec<T> as a Vec<u8>.
    return Vec.fromRawParts(ptr, length, capacity)
}

public interface Load {
    public fun load(device: Device): Result<Tensor>
}

public fun st.TensorView.load(device: Device): Result<Tensor> = convert(this, device)

public fun Tensor.Companion.fromRawBuffer(data: List<UByte>, dtype: DType, shape: List<Int>, device: Device): Result<Tensor> {
    return when (dtype) {
        DType.u8 -> convertSlice<UByte>(data, shape, device)
        DType.u32 -> convertSlice<UInt>(data, shape, device)
        DType.i16 -> convertSlice<Short>(data, shape, device)
        DType.i32 -> convertSlice<Int>(data, shape, device)
        DType.i64 -> convertSlice<Long>(data, shape, device)
        DType.bf16 -> convertSlice<Bf16>(data, shape, device)
        DType.f16 -> convertSlice<F16>(data, shape, device)
        DType.f32 -> convertSlice<Float>(data, shape, device)
        DType.f64 -> convertSlice<Double>(data, shape, device)
        DType.f8e4m3 -> convertSlice<F8e4m3>(data, shape, device)
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            // For dummy types, create storage with raw bytes
            val storage2 =
                when (device) {
                    Device.Cpu -> {
                        val cpuStorage =
                            when (dtype) {
                                DType.f6e2m3 -> F6e2m3(data.toVec())
                                DType.f6e3m2 -> F6e3m2(data.toVec())
                                DType.f4 -> F4(data.toVec())
                                DType.f8e8m0 -> F8e8m0(data.toVec())
                                else -> error("unreachable")
                            }
                        Storage.Cpu(cpuStorage)
                    }
                    is Device.Cuda -> {
                        val (device1) = device
                        var slice =
                            when (val tryResult = device1.alloc<UByte>(data.size)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        when (val tryResult = device1.memcpyHtod(data, slice)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                        val slice1 =
                            when (dtype) {
                                DType.f6e2m3 -> F6e2m3(slice)
                                DType.f6e3m2 -> F6e3m2(slice)
                                DType.f4 -> F4(slice)
                                DType.f8e8m0 -> F8e8m0(slice)
                                else -> error("unreachable")
                            }
                        val storage = CudaStorage(slice = slice1, device = device1.clone())
                        Storage.Cuda(storage)
                    }
                    is Device.Metal -> {
                        val (device2) = device
                        val buffer =
                            when (
                                val tryResult =
                                    device2
                                        .newBufferBuilder()
                                        .withData(data)
                                        .withLabel("safetensors_view")
                                        .build()
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        val storage1 = MetalStorage.new(buffer, device2.clone(), data.size, dtype)
                        Storage.Metal(storage1)
                    }
                }
            val op = BackpropOp.none()
            Ok(fromStorage(storage2, shape, op, false))
        }
    }
}

public fun convertDummy(view: st.TensorView, device: Device): Result<Tensor> {
    // For dummy types, we'll create the appropriate storage variant that preserves
    // both the raw data and the correct dtype
    val (dtype, dtypeName) =
        when (view.dtype()) {
            st.Dtype.f6E2m3 -> Pair(DType.F6E2M3, "F6_E2M3 (MX6)")
            st.Dtype.f6E3m2 -> Pair(DType.F6E3M2, "F6_E3M2 (MX6)")
            st.Dtype.f4 -> Pair(DType.F4, "F4 (MX4)")
            st.Dtype.f8E8m0 -> Pair(DType.F8E8M0, "F8_E8M0")
            else -> error("convert_dummy called with non-dummy dtype")
        }
    // Load the raw bytes
    val data = view.data()
    val shape = view.shape()
    // Create storage with the appropriate dummy type variant
    val storage2 =
        when (device) {
            Device.Cpu -> {
                val cpuStorage =
                    when (dtype) {
                        DType.f6e2m3 -> F6e2m3(data.toVec())
                        DType.f6e3m2 -> F6e3m2(data.toVec())
                        DType.f4 -> F4(data.toVec())
                        DType.f8e8m0 -> F8e8m0(data.toVec())
                        else -> error("unreachable")
                    }
                Storage.Cpu(cpuStorage)
            }
            is Device.Cuda -> {
                val (device1) = device
                var slice =
                    when (val tryResult = device1.alloc<UByte>(data.len())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = device1.memcpyHtod(data, slice)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                val slice1 =
                    when (dtype) {
                        DType.f6e2m3 -> F6e2m3(slice)
                        DType.f6e3m2 -> F6e3m2(slice)
                        DType.f4 -> F4(slice)
                        DType.f8e8m0 -> F8e8m0(slice)
                        else -> error("unreachable")
                    }
                val storage = CudaStorage(slice = slice1, device = device1.clone())
                Storage.Cuda(storage)
            }
            is Device.Metal -> {
                val (device2) = device
                val buffer =
                    when (
                        val tryResult =
                            device2
                                .newBufferBuilder()
                                .withData(data)
                                .withLabel("safetensors_load")
                                .build()
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val storage1 = MetalStorage.new(buffer, device2.clone(), data.len(), dtype)
                Storage.Metal(storage1)
            }
        }
    // Create tensor with correct dtype
    val op = BackpropOp.none()
    return Ok(fromStorage(storage2, shape, op, false))
}

public fun convertBack(tensor: Tensor): Result<List<UByte>> {
    // Note: This makes an unnecessary copy when the tensor is on the cpu.
    val tensor1 =
        when (val tryResult = tensor.flattenAll()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return when (tensor1.dtype()) {
        DType.u8 ->
            Ok(
                convertBack<UByte>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.u32 ->
            Ok(
                convertBack<UInt>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.i16 ->
            Ok(
                convertBack<Short>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.i32 ->
            Ok(
                convertBack<Int>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.i64 ->
            Ok(
                convertBack<Long>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.f16 ->
            Ok(
                convertBack<F16>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.bf16 ->
            Ok(
                convertBack<Bf16>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.f32 ->
            Ok(
                convertBack<Float>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.f64 ->
            Ok(
                convertBack<Double>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.f8e4m3 ->
            Ok(
                convertBack<F8e4m3>(
                    when (val tryResult = tensor1.toVec1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                ),
            )
        DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
            Err(Error.Msg("Internal error: dtype mismatch in storage".toString()).bt())
        }
    }
}

public fun <P : AsRef<Path>> load(filename: P, device: Device): Result<Map<String, Tensor>> {
    val data =
        when (val tryResult = read(filename.asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return loadBuffer(data, device)
}

public fun loadBuffer(data: List<UByte>, device: Device): Result<Map<String, Tensor>> {
    val st =
        when (val tryResult = SafeTensors.deserialize(data)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    return st
        .tensors()
        .intoIter()
        .map(
            closure@{ (name, view) ->
                Ok(
                    Pair(
                        name,
                        when (val tryResult = view.load(device)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return@closure Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            },
        ).collect()
}

public fun <K : AsRef<String>, P : AsRef<Path>> save(tensors: Map<K, Tensor>, filename: P): Result<Unit> {
    return Ok(
        when (val tryResult = st.serializeToFile(tensors, null, filename.asRef())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        },
    )
}

public data class SafeTensors(
    public var value: SafeTensors = SafeTensors(),
) {
    public var field0: SafeTensors
        get() = value
        set(v) {
            value = v
        }
}

public class MmapedSafetensors internal constructor(
    safetensorsCtorArg: List<Yoke<SafeTensors, Mmap>>,
    public var routing: Map<String, Int>?,
) {
    public var safetensors: List<Yoke<SafeTensors, Mmap>> = safetensorsCtorArg
        internal set

    public constructor() : this(
        safetensorsCtorArg = emptyList(),
        routing = null,
    )

    public operator fun component1(): List<Yoke<SafeTensors, Mmap>> = this.safetensors

    public operator fun component2(): Map<String, Int>? = this.routing

    public fun copy(safetensors: List<Yoke<SafeTensors, Mmap>> = this.safetensors, routing: Map<String, Int>? = this.routing): MmapedSafetensors = MmapedSafetensors(safetensorsCtorArg = safetensors, routing = routing)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is MmapedSafetensors) return false
        if (this.safetensors != other.safetensors) return false
        if (this.routing != other.routing) return false
        return true
    }

    override fun hashCode(): Int {
        var result = this.safetensors.hashCode()
        result = 31 * result + this.routing.hashCode()
        return result
    }

    public fun load(name: String, dev: Device): Result<Tensor> {
        return when (val tryResult = this.get(name)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.load(dev)
    }

    public fun tensors(): List<Pair<String, st.TensorView>> {
        val tensors = mutableListOf<Pair<String, st.TensorView>>()
        for (safetensors in ArrayDeque(this.safetensors)) {
            tensors.add(safetensors.get().first.tensors())
        }
        return ArrayDeque(tensors).flatten().collect()
    }

    public fun get(name: String): Result<st.TensorView> {
        val index1 =
            when (val matchVal = this.routing) {
                null -> 0
                else -> {
                    val routing = matchVal
                    val index =
                        when (
                            val tryResult =
                                routing.get(name).okOrElse({
                                    Error.CannotFindTensor(path = name.toString()).bt()
                                })
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    index
                }
            }
        return Ok(
            when (
                val tryResult =
                    this.safetensors[index1]
                        .get()
                        .first
                        .tensor(name)
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    }

    public companion object {
        public operator fun invoke(
            safetensors: List<Yoke<SafeTensors, Mmap>> = emptyList(),
            routing: Map<String, Int>? = null,
        ): MmapedSafetensors =
            MmapedSafetensors(
                safetensorsCtorArg = safetensors,
                routing = routing,
            )

        /**
         * Creates a wrapper around a memory mapped file and deserialize the safetensors header.
         *
         * # Safety
         *
         * The unchecked is inherited from [memmap2.MmapOptions].
         */
        public fun <P : AsRef<Path>> new(p: P): Result<MmapedSafetensors> {
            val p1 = p.asRef()
            val file =
                when (val tryResult = File.open(p1).mapErr({ e -> Error.from(e).withPath(p1) })) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val file1 =
                when (val tryResult = MmapOptions.new().map(file).mapErr({ e -> Error.from(e).withPath(p1) })) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val safetensors =
                when (
                    val tryResult =
                        Yoke.tryAttachToCart(
                            file1,
                            closure1@{ data: List<UByte> ->
                                val st =
                                    when (val tryResult = SafeTensors.deserialize(data).mapErr({ e -> Error.from(e).withPath(p1) })) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return@closure1 Result.fromResidual(Result.Err(tryResult.value))
                                    }
                                Ok(SafeTensors(st))
                            },
                        )
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(MmapedSafetensors(safetensors = mutableListOf(safetensors), routing = null))
        }

        /**
         * Creates a wrapper around multiple memory mapped file and deserialize the safetensors headers.
         *
         * If a tensor name appears in multiple files, the last entry is returned.
         *
         * # Safety
         *
         * The unchecked is inherited from [memmap2.MmapOptions].
         */
        public fun <P : AsRef<Path>> multi(paths: List<P>): Result<MmapedSafetensors> {
            var routing = HashMap.new()
            var safetensors = mutableListOf()
            for ((index, p) in paths.enumerate()) {
                val p1 = p.asRef()
                val file =
                    when (val tryResult = File.open(p1).mapErr({ e -> Error.from(e).withPath(p1) })) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val file1 =
                    when (val tryResult = MmapOptions.new().map(file).mapErr({ e -> Error.from(e).withPath(p1) })) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val data =
                    when (
                        val tryResult =
                            Yoke.tryAttachToCart(
                                file1,
                                closure2@{ data: List<UByte> ->
                                    val st =
                                        when (val tryResult = SafeTensors.deserialize(data).mapErr({ e -> Error.from(e).withPath(p1) })) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return@closure2 Result.fromResidual(Result.Err(tryResult.value))
                                        }
                                    Ok(SafeTensors(st))
                                },
                            )
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (k in data.get().first.names()) {
                    routing.insert(k.toString(), index)
                }
                safetensors.add(data)
            }
            return Ok(MmapedSafetensors(safetensors = safetensors, routing = routing))
        }
    }
}

public data class SliceSafetensors(
    public var safetensors: SafeTensors = SafeTensors(),
) {
    public fun load(name: String, dev: Device): Result<Tensor> {
        return when (val tryResult = this.safetensors.tensor(name)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.load(dev)
    }

    public fun tensors(): List<Pair<String, st.TensorView>> = this.safetensors.tensors()

    public fun get(name: String): Result<st.TensorView> {
        return Ok(
            when (val tryResult = this.safetensors.tensor(name)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    }

    public companion object {
        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        public fun new(buffer: List<UByte>): Result<SliceSafetensors> {
            val safetensors =
                when (val tryResult = SafeTensors.deserialize(buffer)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(SliceSafetensors(safetensors = safetensors))
        }
    }
}

public class BufferedSafetensors internal constructor(
    safetensorsCtorArg: Yoke<SafeTensors, List<UByte>>,
) {
    public var safetensors: Yoke<SafeTensors, List<UByte>> = safetensorsCtorArg
        internal set

    public constructor() : this(
        safetensorsCtorArg = Yoke<SafeTensors, List<UByte>>(),
    )

    public operator fun component1(): Yoke<SafeTensors, List<UByte>> = this.safetensors

    public fun copy(safetensors: Yoke<SafeTensors, List<UByte>> = this.safetensors): BufferedSafetensors = BufferedSafetensors(safetensorsCtorArg = safetensors)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BufferedSafetensors) return false
        if (this.safetensors != other.safetensors) return false
        return true
    }

    override fun hashCode(): Int = this.safetensors.hashCode()

    public fun load(name: String, dev: Device): Result<Tensor> {
        return when (val tryResult = this.get(name)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.load(dev)
    }

    public fun tensors(): List<Pair<String, st.TensorView>> =
        this.safetensors
            .get()
            .first
            .tensors()

    public fun get(name: String): Result<st.TensorView> {
        return Ok(
            when (
                val tryResult =
                    this.safetensors
                        .get()
                        .first
                        .tensor(name)
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        )
    }

    public companion object {
        public operator fun invoke(
            safetensors: Yoke<SafeTensors, List<UByte>> = Yoke<SafeTensors, List<UByte>>(),
        ): BufferedSafetensors =
            BufferedSafetensors(
                safetensorsCtorArg = safetensors,
            )

        /**
         * Creates a wrapper around a binary buffer and deserialize the safetensors header.
         */
        public fun new(buffer: List<UByte>): Result<BufferedSafetensors> {
            val safetensors =
                when (
                    val tryResult =
                        Yoke.tryAttachToCart(
                            buffer,
                            closure3@{ data: List<UByte> ->
                                val st =
                                    when (val tryResult = SafeTensors.deserialize(data)) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return@closure3 Result.fromResidual(Result.Err(tryResult.value))
                                    }
                                Ok(SafeTensors(st))
                            },
                        )
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(BufferedSafetensors(safetensors = safetensors))
        }
    }
}

public data class MmapedFile(
    public var path: PathBuf = PathBuf(),
    public var inner: Mmap = Mmap(),
) {
    public fun deserialize(): Result<SafeTensors> {
        val st =
            when (val tryResult = SafeTensors.deserialize(this.inner).mapErr({ e -> Error.from(e).withPath(this.path) })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(st)
    }

    public companion object {
        /**
         * Creates a wrapper around a memory mapped file from which you can retrieve
         * tensors using [MmapedFile.deserialize]
         *
         * # Safety
         *
         * The unchecked is inherited from [memmap2.MmapOptions].
         */
        public fun <P : AsRef<Path>> new(p: P): Result<MmapedFile> {
            val p1 = p.asRef()
            val file =
                when (val tryResult = File.open(p1).mapErr({ e -> Error.from(e).withPath(p1) })) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val inner =
                when (val tryResult = MmapOptions.new().map(file).mapErr({ e -> Error.from(e).withPath(p1) })) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(MmapedFile(inner = inner, path = p1.toPathBuf()))
        }
    }
}
