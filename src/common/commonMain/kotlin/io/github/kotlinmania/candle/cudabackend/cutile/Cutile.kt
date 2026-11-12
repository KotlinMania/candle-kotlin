// port-lint: source cuda_backend/cutile.rs
package io.github.kotlinmania.candle.cudabackend.cutile

import io.github.kotlinmania.candle.cudabackend.CudaDType
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.CudaSlice
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.DeviceRepr
import io.github.kotlinmania.candle.cudabackend.cudarc.driver.SyncOnDrop
import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cutile.cudaasync.devicebuffer.DevicePointer
import io.github.kotlinmania.cutile.cudacore.Stream
import io.github.kotlinmania.cutile.cutilecompiler.cudatileruntimeutils.getGpuName
import io.github.kotlinmania.cutile.cutilecompiler.cudatileruntimeutils.tileirasBinary
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.path.Path
import io.github.kotlinmania.rustlang.std.path.PathBuf
import io.github.kotlinmania.rustlang.std.process.Command
import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.OnceLock

// cuTile kernel authoring and CUDA stream interop.
//
// ```kotlin
// import candlecore.cutile
// import cutile.core.*
// import cutile.cutileCompiler
//
// @entry
// public fun noop() {}
// ```

/**
 * A cuTile launch context backed by Candle's current CUDA stream.
 */
public data class CutileContext(
    public var candleStream: CudaStream = CudaStream(),
    public var cutileStream: Stream = Stream(),
) {
    /**
     * The borrowed cuTile stream used by `compileOn` and `asyncOn`.
     */
    public fun stream(): Stream = this.cutileStream

    /**
     * Borrows a typed Candle allocation for a cuTile kernel read.
     */
    public fun <T : DeviceRepr> read(slice: CudaSlice<T>, offset: Int, typeLayoutT: TypeLayout<T>): Result<CutileRead<T>> {
        if (offset > slice.len()) {
            return Err(Error.msg("cuTile read offset $offset exceeds allocation length ${slice.len()}"))
        }
        val (pointer, guard) = slice.devicePtr(this.candleStream)
        val byteOffset =
            when (val tryResult = ((offset).toLong() * (sizeOf<T>(typeLayoutT = typeLayoutT)).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt().okOrElse({ Error.msg("cuTile read offset overflow") })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val pointer1 =
            when (val tryResult = ((pointer).toLong() + (byteOffset.toULong()).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt().okOrElse({ Error.msg("cuTile read pointer overflow") })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CutileRead(pointer = DevicePointer.fromCuDeviceptr(pointer1), guard = guard))
    }

    /**
     * Borrows a typed Candle allocation for a cuTile kernel write.
     */
    public fun <T : DeviceRepr> write(slice: CudaSlice<T>, offset: Int, typeLayoutT: TypeLayout<T>): Result<CutileWrite<T>> {
        if (offset > slice.len()) {
            return Err(Error.msg("cuTile write offset $offset exceeds allocation length ${slice.len()}"))
        }
        val (pointer, guard) = slice.devicePtrMut(this.candleStream)
        val byteOffset =
            when (val tryResult = ((offset).toLong() * (sizeOf<T>(typeLayoutT = typeLayoutT)).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt().okOrElse({ Error.msg("cuTile write offset overflow") })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val pointer1 =
            when (val tryResult = ((pointer).toLong() + (byteOffset.toULong()).toLong()).takeIf { it in Int.MIN_VALUE..Int.MAX_VALUE }?.toInt().okOrElse({ Error.msg("cuTile write pointer overflow") })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(CutileWrite(pointer = DevicePointer.fromCuDeviceptr(pointer1), guard = guard))
    }

    /**
     * Borrows a typed Candle storage and applies its layout offset.
     */
    public fun <T> readStorage(storage: CudaStorage, layout: Layout): Result<CutileRead<T>> where T : CudaDType, T : DeviceRepr {
        return this.read(
            when (val tryResult = storage.asCudaSlice<T>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
            layout.startOffset(),
        )
    }

    /**
     * Mutably borrows a typed Candle storage and applies its layout offset.
     */
    public fun <T> writeStorage(storage: CudaStorage, layout: Layout): Result<CutileWrite<T>> where T : CudaDType, T : DeviceRepr {
        return this.write(
            when (val tryResult = storage.asCudaSliceMut<T>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
            layout.startOffset(),
        )
    }

    public companion object {
        /**
         * Borrows Candle's CUDA context and stream without transferring ownership.
         */
        public fun new(device: CudaDevice): Result<CutileContext> {
            val candleStream = device.cudaStream()
            val candleContext = candleStream.context()
            when (val tryResult = candleContext.bindToThread().w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            when (val tryResult = checkDeviceSupport(candleContext.ordinal())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            val cutileDevice = Device.borrowWithOwner(mutPtrFromRustCast(candleContext.cuCtx()), candleContext.cuDevice().toInt(), candleContext.ordinal(), candleContext.clone())
            val cutileStream = Stream.borrowWithOwner(mutPtrFromRustCast(candleStream.cuStream()), cutileDevice, candleStream.clone())
            return Ok(CutileContext(candleStream = candleStream, cutileStream = cutileStream))
        }
    }
}

/**
 * Fails when the installed `tileiras` cannot target the device, if it can be probed.
 */
public fun checkDeviceSupport(ordinal: Int): Result<Unit> {
    val tileiras = tileirasBinary()
    val supported = tileirasGpuNames(tileiras) ?: return Ok(Unit)
    return ensureGpuSupported(ordinal, getGpuName(ordinal), tileiras, supported)
}

public fun ensureGpuSupported(ordinal: Int, gpuName: String, tileiras: Path, supported: List<String>): Result<Unit> {
    if (supported.any({ name -> name == gpuName })) {
        return Ok(Unit)
    }
    return Err(Error.msg("cuTile cannot target CUDA device " + (ordinal).toString() + ": it is " + (gpuName).toString() + ", but " + (tileiras.display()).toString() + " supports only " + (supported.join(", ")).toString() + ". Use a supported GPU or a CUDA toolkit whose tileiras supports " + (gpuName).toString() + "."))
}

public typealias GpuNames = List<String>?

/**
 * Architectures accepted by `tileiras - - gpu - name`, probed once per binary.
 */
public fun tileirasGpuNames(tileiras: Path): GpuNames {
    val CACHE: OnceLock<Mutex<Map<PathBuf, GpuNames>>> = OnceLock.new()
    val cache = CACHE.getOrInit(Default::default).lock().unwrapOrElse({ poisoned -> poisoned.intoInner() })
    when (val matchVal = cache.get(tileiras)) {
        null -> {}
        else -> {
            val cached = matchVal
            return cached.clone()
        }
    }
    val names = probeTileirasGpuNames(tileiras)
    cache.insert(tileiras.toPathBuf(), names.clone())
    return names
}

public fun probeTileirasGpuNames(tileiras: Path): GpuNames {
    val output =
        when (
            val tryResult =
                Command
                    .new(tileiras)
                    .arg("--help")
                    .output()
                    .ok()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val help = String.fromUtf8Lossy(output.stdout).intoOwned()
    help.pushStr(String.fromUtf8Lossy(output.stderr))
    val names = parseTileirasGpuNames(help)
    return (!names.isEmpty()).then({ names.into() })
}

/**
 * Extracts the `= smXy` choices listed under `- - gpu - name` in `tileiras - - help`.
 */
public fun parseTileirasGpuNames(help: String): List<String> {
    val names = mutableListOf<String>()
    var inGpuName = false
    for (line in help.lines()) {
        val line1 = line.trimStart()
        if (line1.startsWith("--gpu-name")) {
            inGpuName = true
        } else if (inGpuName) {
            when (val matchVal = (line1.takeIf { it.startsWith('=') }?.removePrefix('='))?.let({ rest -> rest.splitWhitespace().next() })) {
                null -> inGpuName = false
                else -> {
                    val name = matchVal
                    names.add(name.toString())
                }
            }
        }
    }
    return names
}

/**
 * A typed cuTile pointer that records a Candle read after it is dropped.
 */
public class CutileRead<T>(
    public var pointer: DevicePointer<T> = DevicePointer<T>(),
    public var guard: SyncOnDrop = SyncOnDrop(),
) {
    /**
     * Returns the pointer accepted by raw-pointer cuTile entry functions.
     */
    public fun devicePointer(): DevicePointer<T> = DevicePointer.fromCuDeviceptr(this.pointer.cuDeviceptr())
}

/**
 * A typed cuTile pointer that records a Candle write after it is dropped.
 */
public class CutileWrite<T>(
    public var pointer: DevicePointer<T> = DevicePointer<T>(),
    public var guard: SyncOnDrop = SyncOnDrop(),
) {
    /**
     * Returns the pointer accepted by raw-pointer cuTile entry functions.
     */
    public fun devicePointer(): DevicePointer<T> = DevicePointer.fromCuDeviceptr(this.pointer.cuDeviceptr())
}

/**
 * Converts a cuTile error or panic into a Candle error.
 */
public fun <T, E> kernel(operation: String, f: () -> Result<T, E>): Result<T> =
    run {
        val matchVal = catchUnwind(AssertUnwindSafe(f))
        when {
            matchVal is Ok && matchVal.value is Ok -> {
                val (value) = matchVal.value as Ok
                Ok(value)
            }
            matchVal is Ok && matchVal.value is Err -> {
                val (error) = matchVal.value as Err
                Err(Error.msg("cuTile $operation: $error"))
            }
            else -> {
                val (payload) = matchVal
                val message =
                    payload
                        .downcastRef<String>()
                        .map(String::asStr)
                        .orElse({ payload.downcastRef<String>().copied() })
                        .unwrapOr("non-string panic")
                Err(Error.msg("cuTile $operation panicked: $message"))
            }
        }
    }
