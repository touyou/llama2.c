// swift-tools-version: 6.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "llama-swift",
    platforms: [
        .macOS(.v14)
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "llama-swift",
            cSettings: [
                // Swiftコンパイラに渡す設定
                // ACCELERATE_NEW_LAPACK=1 を定義
                .define("ACCELERATE_NEW_LAPACK"),
                // ACCELERATE_LAPACK_ILP64=1 を定義
                .define("ACCELERATE_LAPACK_ILP64"),
            ],
            linkerSettings: [
                // Accelerateフレームワークをリンク
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
