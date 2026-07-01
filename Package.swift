// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalJIT",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "MetalJIT",
            targets: ["MetalJIT"]
        ),
        .library(
            name: "MetalJITCore",
            type: .dynamic,
            targets: ["MetalJITCore"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0")
    ],
    targets: [
        // Capa 1: C++ / ObjC / Metal JIT Engine
        .target(
            name: "MetalJITCore",
            path: "Sources/MetalJITCore",
            sources: ["Modules"],
            resources: [
                .copy("Resources/PrivacyInfo.xcprivacy")
            ],
            publicHeadersPath: "Headers",
            cxxSettings: [
                .headerSearchPath("Headers"),
                .define("NDEBUG", .when(configuration: .release))
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Foundation")
            ]
        ),
        // Capa 2: Swift Public API
        .target(
            name: "MetalJIT",
            dependencies: ["MetalJITCore"],
            path: "Sources/MetalJIT"
        ),
        // Capa 3: Unit Tests
        .testTarget(
            name: "MetalJITTests",
            dependencies: ["MetalJIT"],
            path: "Tests",
            exclude: ["MetalJITBindingTest.mm", "MetalJITBindingTest.py"]
        ),
        // Capa 4: Demo Runner
        .executableTarget(
            name: "MetalJITRunner",
            dependencies: ["MetalJIT"],
            path: "Sources/MetalJITExamples",
            exclude: ["MetalJITExample.mm", "MetalJITExample.py"],
            sources: ["MetalJITExample.swift"]
        )
    ],
    cxxLanguageStandard: .cxx17
)
