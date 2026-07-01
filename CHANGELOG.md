# Changelog

All notable changes to MetalJIT will be documented in this file.

## [1.0.0] - 2026-07-01

### Added

- Initial release of MetalJIT.
- JIT compilation of Metal Shading Language via `newLibraryWithSource`.
- Zero-Copy dispatch over Unified Memory Architecture (UMA) on Apple Silicon.
- Swift API: `JITCompiler`, `ComputeDispatcher`, `Pipeline` with `async/await` support.
- C ABI: `mjit_compile_pipeline`, `mjit_dispatch`, `mjit_dispatch_batch`, `mjit_dispatch_async`.
- Python binding via `ctypes`.
- Six data types: `UInt64`, `Float32`, `Float64`, `Int32`, `Int64`, `Float16`.
- Automatic CPU/GPU routing (GCD for <10k elements, Metal for >=10k).
- Five built-in kernels: crypto, tensor, logic, physics, topology.
- `MTLHeap` wrapper for GPU buffer pooling.
- Shader source cache with validation.
- `MTLBinaryArchive` integration for PSO caching.
- Multi-device GPU selection.
- XCFramework distribution with notarization and stapling via DMG.
- `xcodegen` project definition for reproducible Xcode builds.
- 136 unit tests (XCTest) plus C and Python integration tests.
- User manual and API reference.
