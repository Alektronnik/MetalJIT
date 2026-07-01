# Getting Started with MetalJIT

Compila tu primer shader Metal en tiempo de ejecucion y despachalo con Zero-Copy.

## Prerequisites

- macOS 13.0 or later
- Apple Silicon
- Swift 5.9 or later

## Add MetalJIT to your project

```swift
dependencies: [
    .package(url: "https://github.com/Alektronnik/MetalJIT.git", from: "1.0.0")
]
```

## Compile a shader

```swift
import MetalJIT

let compiler = JITCompiler()

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
```

## Dispatch

```swift
let dispatcher = ComputeDispatcher()

var input:  [UInt64] = [1, 2, 3, 4, 5]
var output: [UInt64] = [0, 0, 0, 0, 0]

try input.withUnsafeMutableBufferPointer { inPtr in
    try output.withUnsafeMutableBufferPointer { outPtr in
        try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
    }
}

print(output) // [2, 4, 6, 8, 10]
```

## Built-in kernels

```swift
let cryptoPipeline = try compiler.compile(builtIn: .crypto)
let tensorPipeline = try compiler.compile(builtIn: .tensor)
```

## Next steps

- ``JITCompiler`` -- todas las variantes de compilacion
- ``ComputeDispatcher`` -- dispatch, batch y async
- ``BuiltInKernel`` -- catalogo de kernels predefinidos
