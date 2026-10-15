// port-lint: source candle-core/src/device.rs
package io.github.kotlinmania.candle

public sealed class DeviceLocation {
    public object Cpu : DeviceLocation() {
        override fun toString(): String = "cpu"
    }

    public data class Cuda(
        val gpuId: Int,
    ) : DeviceLocation() {
        override fun toString(): String = "cuda:$gpuId"
    }

    public data class Metal(
        val gpuId: Int,
    ) : DeviceLocation() {
        override fun toString(): String = "metal:$gpuId"
    }
}

public sealed class Device {
    public object Cpu : Device() {
        override fun toString(): String = "Device::Cpu"
    }

    public data class Cuda(
        val device: CudaDevice,
    ) : Device() {
        override fun toString(): String = "Device::Cuda(${device.gpuId})"
    }

    public data class Metal(
        val device: MetalDevice,
    ) : Device() {
        override fun toString(): String = "Device::Metal(${device.gpuId})"
    }

    public fun location(): DeviceLocation =
        when (this) {
            is Cpu -> DeviceLocation.Cpu
            is Cuda -> DeviceLocation.Cuda(device.gpuId)
            is Metal -> DeviceLocation.Metal(device.gpuId)
        }

    public fun isCpu(): Boolean = this is Cpu

    public fun isCuda(): Boolean = this is Cuda

    public fun isMetal(): Boolean = this is Metal

    public fun sameDevice(other: Device): Boolean =
        when (this) {
            is Cpu -> other is Cpu
            is Cuda -> other is Cuda && device.gpuId == other.device.gpuId
            is Metal -> other is Metal && device.gpuId == other.device.gpuId
        }

    public fun zeros(shape: Shape, dtype: DType): Storage {
        val count = shape.elemCount()
        return when (this) {
            is Cpu -> {
                val cpuStorage = when (dtype) {
                    DType.U8 -> CpuStorage.U8(UByteArray(count))
                    DType.U32 -> CpuStorage.U32(UIntArray(count))
                    DType.I16 -> CpuStorage.I16(ShortArray(count))
                    DType.I32 -> CpuStorage.I32(IntArray(count))
                    DType.I64 -> CpuStorage.I64(LongArray(count))
                    DType.BF16 -> CpuStorage.BF16(ShortArray(count))
                    DType.F16 -> CpuStorage.F16(ShortArray(count))
                    DType.F32 -> CpuStorage.F32(FloatArray(count))
                    DType.F64 -> CpuStorage.F64(DoubleArray(count))
                    DType.F8E4M3 -> CpuStorage.F8E4M3(ByteArray(count))
                }
                Storage.Cpu(cpuStorage)
            }
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
    }

    public fun allocUninit(shape: Shape, dtype: DType): Storage = zeros(shape, dtype)

    public companion object {
        public fun newCuda(gpuId: Int): Device = Cuda(CudaDevice(gpuId))

        public fun newMetal(gpuId: Int): Device = Metal(MetalDevice(gpuId))

        public fun cudaIfAvailable(gpuId: Int): Device = Cpu

        public fun metalIfAvailable(gpuId: Int): Device = Cpu
    }
}
