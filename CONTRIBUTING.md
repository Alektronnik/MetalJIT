# Contributing to MetalJIT

MetalJIT welcomes contributions. This document describes how to set up your environment, run tests, and submit changes.

## Prerequisites

- macOS 13.0 or later
- Apple Silicon Mac
- Xcode 16.0 or later
- `xcodegen` (`brew install xcodegen`) for Xcode project generation

## Getting Started

```bash
git clone https://github.com/Alektronnik/MetalJIT.git
cd MetalJIT
swift build
swift run MetalJITRunner
```

## Running Tests

```bash
# SPM tests (requires Xcode with XCTest)
swift test

# C integration tests
cd Tests
clang++ -std=c++17 -O0 -fobjc-arc -x objective-c++ MetalJITBindingTest.mm \
    -I ../Sources/MetalJITCore/Headers \
    ../Sources/MetalJITCore/Modules/*.mm \
    -framework Metal -framework Foundation \
    -o /tmp/MetalJITBindingTest && /tmp/MetalJITBindingTest

# Python integration tests
METALJIT_LIB=.build/debug/libMetalJITCore.dylib python3 Tests/MetalJITBindingTest.py
```

## Project Structure

```
Sources/
  MetalJIT/            Swift public API
  MetalJITCore/
    Headers/            Public C headers
    Modules/            C++/ObjC implementation
    Resources/          PrivacyInfo.xcprivacy
    Utils/              Python binding
  MetalJITExamples/     Example code (Swift, ObjC++, Python)
Tests/                  XCTest suite + integration tests
Benchmark/              Standalone benchmarks
Docs/                   User manual and documentation
Xcode/                  xcodegen project definition
```

## Code Style

- Swift: Follow [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/).
- C/C++: `clang-format` with LLVM style, 4-space indent.
- Objective-C++: ARC enabled, `NS_ASSUME_NONNULL_BEGIN`/`END` on public headers.
- No emojis in code, documentation, or commit messages.

## Pull Requests

1. Fork the repository.
2. Create a feature branch.
3. Ensure `swift build` passes.
4. Ensure all tests pass.
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Submit a pull request with a clear description.
