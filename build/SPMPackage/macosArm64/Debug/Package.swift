// swift-tools-version: 5.9

import PackageDescription
let package = Package(
    name: "Candle",
    platforms: [.macOS(.v14)],
    products: [
        .library(
            name: "CandleLibrary",
            targets: ["ExportedKotlinPackages", "KotlinRuntimeSupport", "KotlinCoroutineSupport", "Candle"]
        )
    ],
    targets: [
        .target(
            name: "ExportedKotlinPackages",
            dependencies: ["KotlinRuntime"]
        ),
        .target(
            name: "KotlinRuntimeSupport",
            dependencies: ["KotlinRuntimeSupportBridge", "KotlinRuntime"]
        ),
        .target(
            name: "KotlinRuntimeSupportBridge"
        ),
        .target(
            name: "KotlinCoroutineSupport",
            dependencies: ["KotlinRuntimeSupport", "KotlinCoroutineSupportBridge", "KotlinRuntime"]
        ),
        .target(
            name: "KotlinCoroutineSupportBridge"
        ),
        .target(
            name: "Candle",
            dependencies: ["KotlinRuntimeSupport", "KotlinCoroutineSupport", "SharedBridge_Candle", "KotlinRuntime"]
        ),
        .target(
            name: "SharedBridge_Candle"
        ),
        .target(
            name: "KotlinRuntime"
        )
    ]
)
