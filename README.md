# MetalJIT

Compilacion y despacho JIT de shaders Metal con Zero-Copy sobre memoria unificada para Apple Silicon.

## Overview

MetalJIT compila codigo MSL en tiempo de ejecucion y lo ejecuta directamente sobre buffers del usuario sin copias CPU-GPU, aprovechando la arquitectura UMA de Apple Silicon.

- Compilacion JIT via `MTLLibrary`.
- Despacho Zero-Copy sobre `MTLBuffer` sin copias intermedias.
- Ruteo automatico CPU/GPU.
- Kernels predefinidos con fallback CPU.
- API Swift, C y Python.
- Distribucion como XCFramework firmado y notarizado.

## Requirements

- macOS 13.0 or later
- Apple Silicon
- Swift 5.9 or later

## Getting Started

Add MetalJIT to your project with Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/Alektronnik/MetalJIT.git", from: "1.0.0")
]
```

Or clone and build:

```bash
git clone https://github.com/Alektronnik/MetalJIT.git
cd MetalJIT
swift build
swift run MetalJITRunner
```

## Usage

```swift
import MetalJIT

let compiler = JITCompiler()
let dispatcher = ComputeDispatcher()

let shader = """
#include <metal_stdlib>
using namespace metal;
kernel void doble(device const uint64_t* in  [[buffer(0)]],
                  device       uint64_t* out [[buffer(1)]],
                  uint id [[thread_position_in_grid]]) {
    out[id] = in[id] * 2;
}
"""

let pipeline = try compiler.compile(shaderSource: shader, kernelName: "doble")

var input:  [UInt64] = [1, 2, 3]
var output: [UInt64] = [0, 0, 0]

try input.withUnsafeMutableBufferPointer { inPtr in
    try output.withUnsafeMutableBufferPointer { outPtr in
        try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
    }
}
```

Kernels predefinidos:

```swift
let crypto   = try compiler.compile(builtIn: .crypto)
let tensor   = try compiler.compile(builtIn: .tensor)
let logic    = try compiler.compile(builtIn: .logic)
let physics  = try compiler.compile(builtIn: .physics)
let topology = try compiler.compile(builtIn: .topology)
```

Async dispatch:

```swift
try await dispatcher.dispatchAsync(pipeline: pipeline, input: &in, output: &out)
```

C API:

```c
#include "MetalJITCore.h"
int h = mjit_compile_pipeline(shader, "k", err, sizeof(err));
mjit_dispatch_uint64(h, input, output, count);
mjit_destroy_pipeline(h);
```

## Documentation

- [User Manual](Docs/METALJIT_MANUAL_USUARIO.md) -- guia completa de la API, arquitectura y buenas practicas.
- [Contributing](CONTRIBUTING.md) -- como configurar el entorno, ejecutar tests y enviar cambios.

## Distribution

Los XCFrameworks firmados y notarizados se publican en [GitHub Releases](https://github.com/Alektronnik/MetalJIT/releases) como `MetalJIT.dmg`.

Para construir desde fuente:

```bash
./build_framework.sh release        # Debug o Release
./build_framework.sh release sign   # Firmado + notarizado (requiere Developer ID)
```

## License

Apache 2.0. See [LICENSE](LICENSE).
