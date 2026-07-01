# MetalJIT

Compilacion y despacho JIT de shaders Metal con Zero-Copy sobre memoria unificada para Apple Silicon.

## Overview

MetalJIT compila codigo MSL en tiempo de ejecucion y lo ejecuta directamente sobre buffers del usuario sin copias CPU-GPU, aprovechando la arquitectura UMA de Apple Silicon.

- Compilacion JIT via `MTLLibrary`.
- Despacho Zero-Copy sobre `MTLBuffer` sin copias intermedias.
- Ruteo automatico CPU/GPU.
- Kernels predefinidos con fallback CPU.
- API Swift, C y Python.

## Topics

### Compilacion

- ``JITCompiler``
- ``Pipeline``
- ``BuiltInKernel``
- ``JITCompilerError``

### Despacho

- ``ComputeDispatcher``
- ``ComputeDispatcherError``

### Tipos de dato

- ``DataTypeTag``

### Kernels predefinidos

- ``BuiltInKernel/crypto``
- ``BuiltInKernel/tensor``
- ``BuiltInKernel/logic``
- ``BuiltInKernel/physics``
- ``BuiltInKernel/topology``
