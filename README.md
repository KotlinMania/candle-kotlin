# candle-kotlin in Kotlin

[![GitHub link](https://img.shields.io/badge/GitHub-KotlinMania%2Fcandle--kotlin-blue.svg)](https://github.com/KotlinMania/candle-kotlin)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.kotlinmania/candle-kotlin)](https://central.sonatype.com/artifact/io.github.kotlinmania/candle-kotlin)
[![Build status](https://img.shields.io/github/actions/workflow/status/KotlinMania/candle-kotlin/ci.yml?branch=main)](https://github.com/KotlinMania/candle-kotlin/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A pure Kotlin Multiplatform line-by-line transliteration port of Hugging Face's [`candle`](https://github.com/huggingface/candle) (specifically `candle-core`), bringing a minimalist, high-performance tensor computing engine and model runtime to Kotlin across mobile, desktop, server, browser, and embedded platforms.

**Original Project:** This port is based on [`huggingface/candle`](https://github.com/huggingface/candle) by Laurent Mazare and the Hugging Face team. All design credit and algorithmic intent belong to the upstream authors; this repository provides a faithful, type-safe port to Kotlin Multiplatform without JVM-only or third-party Java ecosystem dependencies.

---

## Installation

### Gradle (Kotlin DSL)

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("io.github.kotlinmania:candle-kotlin:0.2.1")
}
```

### Gradle (Groovy DSL)

```groovy
repositories {
    mavenCentral()
}

dependencies {
    implementation 'io.github.kotlinmania:candle-kotlin:0.2.1'
}
```

### Maven (`pom.xml`)

```xml
<dependency>
    <groupId>io.github.kotlinmania</groupId>
    <artifactId>candle-kotlin</artifactId>
    <version>0.2.1</version>
</dependency>
```

---

## Supported Targets

`candle-kotlin` compiles to pure native machine code, JVM bytecode, Android AAR, JavaScript, and WebAssembly with zero Java standard library dependencies in common code:

| Platform Category | Target Identifiers |
|---|---|
| **JVM & Android** | `jvm` (Java 21+ toolchain), `android` (AAR, API 21+) |
| **Apple Silicon & iOS** | `macosArm64`, `iosArm64`, `iosSimulatorArm64`, `tvosArm64`, `tvosSimulatorArm64`, `watchosDeviceArm64`, `watchosSimulatorArm64` |
| **Linux & Windows** | `linuxX64`, `linuxArm64`, `mingwX64` (Windows x64) |
| **Android Native** | `androidNativeArm64`, `androidNativeX64` |
| **Web & WASM** | `js` (Node.js & Browser), `wasmJs` (Browser & Node), `wasmWasi` (WASI CLI runtimes) |

---

## Kotlin Usage Patterns

### 1. Creating Tensors & Basic Matrix Multiplication

Tensors are multi-dimensional arrays parameterized by their shape, data type, and target compute device:

```kotlin
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Tensor

fun main() {
    val device = Device.Cpu

    // Create 2x2 matrices on the CPU
    val a = Tensor.new(
        floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f),
        Shape(listOf(2, 2)),
        device
    )
    val b = Tensor.new(
        floatArrayOf(2.0f, 0.0f, 1.0f, 2.0f),
        Shape(listOf(2, 2)),
        device
    )

    // Compute matrix multiplication: C = A @ B
    val c = a.matmul(b)
    println("Result shape: ${c.dims()}") // [2, 2]

    // Read back values
    val data = c.toVec1F32()
    println("Result values: ${data.joinToString()}") // 4.0, 4.0, 10.0, 8.0
}
```

### 2. Zeros, Ones & Random Initializers

```kotlin
// 2x3 zeros matrix
val zeros = Tensor.zeros(Shape(listOf(2, 3)), DType.F32, Device.Cpu)

// 3x3 identity or ones matrix
val ones = Tensor.ones(Shape(listOf(3, 3)), DType.F32, Device.Cpu)

// Gaussian / standard normal random distribution
val rand = Tensor.randn(0.0f, 1.0f, Shape(listOf(4, 4)), Device.Cpu)
```

### 3. Element-Wise Arithmetic & Broadcasting

All standard operators are implemented with automatic shape broadcasting:

```kotlin
val a = Tensor.new(floatArrayOf(1f, 2f, 3f, 4f), Shape(listOf(2, 2)), Device.Cpu)
val b = Tensor.new(floatArrayOf(10f, 20f, 30f, 40f), Shape(listOf(2, 2)), Device.Cpu)

val sum = a + b        // Element-wise addition
val diff = b - a       // Element-wise subtraction
val prod = a * b       // Element-wise multiplication
val scaled = a * 2.5f  // Scalar multiplication
```

### 4. Reshaping, Slicing & Transposition

```kotlin
val t = Tensor.ones(Shape(listOf(2, 3, 4)), DType.F32, Device.Cpu)

// Reshape to 2D
val flattened = t.reshape(Shape(listOf(6, 4)))

// Transpose dimensions 0 and 1
val transposed = flattened.transpose(0, 1)

// Narrowing along a dimension (dim, start, length)
val slice = t.narrow(1, 0, 2)
```

### 5. Automatic Differentiation & Gradients

Track computation graphs and compute backward passes using `Variable`:

```kotlin
import io.github.kotlinmania.candle.Variable

val x = Variable(Tensor.new(floatArrayOf(3.0f), Shape(listOf(1)), Device.Cpu))

// y = x^2 + 2x + 1
val y = (x * x) + (x * 2.0f) + 1.0f

// Backward pass
val grads = y.backward()
val dx = grads.get(x) // dy/dx = 2x + 2 = 8.0
```

### 6. Quantized Models (GGUF & GGML)

Direct support for quantized tensor storage and dequantization kernels across formats:

```kotlin
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.QTensor

// Supports standard quantization formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2K, Q3K, Q4K, Q5K, Q6K
val qTensor = QTensor.fromStorage(quantizedStorage, shape)
val dequantized = qTensor.dequantize(Device.Cpu)
```

### 7. Reading SafeTensors Headers

Safetensors metadata and wire formats are parsed with full header validation:

```kotlin
import io.github.kotlinmania.candle.readSafetensorHeader
import io.github.kotlinmania.candle.SafeTensors

val header = readSafetensorHeader(rawSafetensorByteArray)
for (entry in header.tensors) {
    println("Tensor: ${entry.name}, DType: ${entry.dtype}, Shape: ${entry.shape.contentToString()}")
}

// Or deserialize tensors directly
val tensors = SafeTensors.deserialize(rawSafetensorByteArray).unwrap()
val weights = tensors.tensor("model.embed_tokens.weight").unwrap()
```

---

## Architectural Scope & Execution Characteristics

- **Core Scope:** This module ports `candle-core`. Higher-level libraries from the Candle ecosystem (`candle-nn`, `candle-transformers`, `candle-datasets`, `candle-onnx`) are separate modular projects.
- **CPU Backend & SIMD:** Powered by a pure Kotlin Multiplatform CPU storage engine featuring SIMD vectorization paths (ARM Neon, Apple Accelerate framework integration, x86 AVX routines).
- **Concurrency:** `CandlePool` operations currently execute synchronously in common code across all targets without platform thread spawners.
- **IO & Memory Mapping:** SafeTensors and NPZ/Pickle file readers utilize in-memory byte buffers. `MmapOptions` maps memory in-place from provided byte buffers rather than establishing OS kernel-level page mappings.
- **Hardware Acceleration Hooks:** GPU abstraction points (`Device.Cuda`, `Device.Metal`) and corresponding dummy backend stubs exist in `commonMain` to preserve multiplatform API parity while hardware-specific native runners remain optional.

---

## Swift Export & Apple Platforms

`candle-kotlin` integrates Kotlin Multiplatform's official Swift Export pipeline. 

- **Deployment Target:** Requires macOS 15.0+ / iOS 18.0+ deployment target when using modern Swift Concurrency (`Synchronization.Mutex` in `KotlinCoroutineSupport`).
- **Xcode Integration:** Built via `embedSwiftExportForXcode` to generate native Swift package modules with strict Swift 6 mode compliance.

---

## Upstream Project & Attribution

This library is a derivative work of Hugging Face's [`candle`](https://github.com/huggingface/candle) repository, developed primarily by Laurent Mazare and contributors.

- **Upstream Repository:** [https://github.com/huggingface/candle](https://github.com/huggingface/candle)
- **Upstream License:** Dual licensed under the MIT License and Apache License, Version 2.0.
- **Port Authorship:** Kotlin transliteration and multiplatform maintenance by Sydney Renee and the KotlinMania team.

---

## License

`candle-kotlin` is dual-licensed under:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))
