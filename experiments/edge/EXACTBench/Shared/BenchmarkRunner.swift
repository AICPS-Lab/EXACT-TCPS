import Foundation
import CoreML
#if os(watchOS)
import WatchKit
#else
import UIKit
#endif

/// Times Core ML inference of the exported segmentation models on a fixed IMU window.
/// Shared by the iOS and watchOS targets.
struct BenchmarkResult: CustomStringConvertible {
    let model: String, device: String, os: String, computeUnits: String, thermal: String, power: String
    let window: Int, channels: Int, iterations: Int, loadMs: Double
    let meanMs: Double, p50Ms: Double, p95Ms: Double, p99Ms: Double, minMs: Double, maxMs: Double
    var description: String {
        """
        on-device benchmark  model: \(model)
          device: \(device)  os: \(os)  thermal: \(thermal)  power: \(power)
          compute units: \(computeUnits)  window: \(window)x\(channels)  iterations: \(iterations)  load+compile: \(fmt(loadMs)) ms
          latency ms  mean \(fmt(meanMs))  p50 \(fmt(p50Ms))  p95 \(fmt(p95Ms))  p99 \(fmt(p99Ms))  min \(fmt(minMs))  max \(fmt(maxMs))
        """
    }
    private func fmt(_ v: Double) -> String { String(format: "%.3f", v) }
}

/// Streaming run: one window every `strideS` seconds for `durationS` seconds (the deployment cadence).
struct SustainedResult: CustomStringConvertible {
    let model: String, device: String, os: String, computeUnits: String
    let thermalStart: String, thermalEnd: String, powerStart: String, powerEnd: String
    let strideS: Double, durationS: Double, windows: Int, missedDeadlines: Int
    let perMinuteMeanMs: [Double], perMinuteP95Ms: [Double], meanMs: Double, p95Ms: Double, p99Ms: Double, maxMs: Double
    var description: String {
        """
        sustained streaming benchmark  model: \(model)
          device: \(device)  os: \(os)  compute units: \(computeUnits)
          thermal start/end: \(thermalStart)/\(thermalEnd)  power start/end: \(powerStart) / \(powerEnd)
          stride \(strideS) s  duration \(Int(durationS)) s  windows \(windows)  missed 500 ms deadlines \(missedDeadlines)
          per-minute mean ms: \(perMinuteMeanMs.map { String(format: "%.3f", $0) }.joined(separator: " "))
          per-minute p95 ms:  \(perMinuteP95Ms.map { String(format: "%.3f", $0) }.joined(separator: " "))
          overall ms  mean \(String(format: "%.3f", meanMs))  p95 \(String(format: "%.3f", p95Ms))  p99 \(String(format: "%.3f", p99Ms))  max \(String(format: "%.3f", maxMs))
        """
    }
}

enum BenchmarkError: Error { case modelNotFound(String), badInput, badOutput }

final class BenchmarkRunner {
    /// Order in which bundled models are benchmarked (names without extension; Xcode compiles .mlpackage -> .mlmodelc).
    static let defaultModels = ["EXACT_physiq", "EXACT_physiq_win250", "EXACT_physiq_win750",
                                "unet_physiq", "cnn_physiq", "transformer_physiq", "segmenter_physiq"]

    static func now() -> Double { Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1_000_000.0 }   // ms, monotonic

    /// Loads a bundled model and builds a fixed pseudo-random input matching its declared "window" shape.
    private func load(_ name: String, _ units: MLComputeUnits) throws -> (MLModel, MLDictionaryFeatureProvider, Int, Int, Double) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { throw BenchmarkError.modelNotFound(name) }
        let config = MLModelConfiguration(); config.computeUnits = units
        let t0 = Self.now()
        let model = try MLModel(contentsOf: url, configuration: config)
        guard let shape = model.modelDescription.inputDescriptionsByName["window"]?.multiArrayConstraint?.shape, shape.count == 3
        else { throw BenchmarkError.badInput }
        let window = shape[1].intValue, channels = shape[2].intValue
        let input = try MLMultiArray(shape: shape, dataType: .float32)
        var state: UInt64 = 0x9E3779B97F4A7C15                       // same seed every run -> comparable across devices
        for i in 0..<(window * channels) {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            input[i] = NSNumber(value: Float((Double(state >> 11) / Double(1 << 53) - 0.5) * 2.0))
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: ["window": MLFeatureValue(multiArray: input)])
        _ = try model.prediction(from: provider)                      // first call finishes compilation
        return (model, provider, window, channels, Self.now() - t0)
    }

    func run(model name: String, computeUnits: MLComputeUnits, warmup: Int = 50, iterations: Int = 300) throws -> BenchmarkResult {
        let (model, provider, window, channels, loadMs) = try load(name, computeUnits)
        for _ in 0..<warmup { _ = try model.prediction(from: provider) }
        var times: [Double] = []; times.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let t0 = Self.now()
            let out = try model.prediction(from: provider)
            let t1 = Self.now()
            guard out.featureValue(for: "logits")?.multiArrayValue != nil else { throw BenchmarkError.badOutput }
            times.append(t1 - t0)
        }
        let s = times.sorted()
        func pct(_ p: Double) -> Double { s[min(s.count - 1, Int(Double(s.count) * p))] }
        return BenchmarkResult(model: name, device: Self.deviceModel(), os: Self.osVersion(), computeUnits: Self.name(computeUnits),
                               thermal: Self.thermal(), power: Self.power(), window: window, channels: channels, iterations: iterations,
                               loadMs: loadMs, meanMs: times.reduce(0, +) / Double(times.count),
                               p50Ms: pct(0.5), p95Ms: pct(0.95), p99Ms: pct(0.99), minMs: s.first ?? 0, maxMs: s.last ?? 0)
    }

    func runSustained(model name: String, computeUnits: MLComputeUnits, durationS: Double, strideS: Double = 0.5,
                      progress: ((Int, Double) -> Void)? = nil) throws -> SustainedResult {
        let (model, provider, _, _, _) = try load(name, computeUnits)
        let thermal0 = Self.thermal(), power0 = Self.power()
        var times: [Double] = [], missed = 0
        let start = Self.now(); var k = 0
        while true {
            let due = start + Double(k) * strideS * 1000.0
            if due - start >= durationS * 1000.0 { break }
            let wait = due - Self.now()
            if wait > 0 { Thread.sleep(forTimeInterval: wait / 1000.0) }
            let t0 = Self.now(); _ = try model.prediction(from: provider); let dt = Self.now() - t0
            times.append(dt); if dt > strideS * 1000.0 { missed += 1 }
            k += 1
            if k % 20 == 0 { progress?(k, dt) }
        }
        func stats(_ a: [Double]) -> (Double, Double, Double, Double) {
            let s = a.sorted(); func pct(_ p: Double) -> Double { s.isEmpty ? 0 : s[min(s.count - 1, Int(Double(s.count) * p))] }
            return (a.isEmpty ? 0 : a.reduce(0, +) / Double(a.count), pct(0.95), pct(0.99), s.last ?? 0)
        }
        let perMinute = Int(60.0 / strideS)
        var mins: [Double] = [], p95s: [Double] = []
        var i = 0
        while i < times.count { let c = Array(times[i..<min(i + perMinute, times.count)]); let st = stats(c); mins.append(st.0); p95s.append(st.1); i += perMinute }
        let all = stats(times)
        return SustainedResult(model: name, device: Self.deviceModel(), os: Self.osVersion(), computeUnits: Self.name(computeUnits),
                               thermalStart: thermal0, thermalEnd: Self.thermal(), powerStart: power0, powerEnd: Self.power(),
                               strideS: strideS, durationS: durationS, windows: times.count, missedDeadlines: missed,
                               perMinuteMeanMs: mins, perMinuteP95Ms: p95s, meanMs: all.0, p95Ms: all.1, p99Ms: all.2, maxMs: all.3)
    }

    static func deviceModel() -> String {
        var sys = utsname(); uname(&sys)
        return withUnsafePointer(to: &sys.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) } }
    }
    static func osVersion() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)"
    }
    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"; case .fair: return "fair"; case .serious: return "serious"; case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
    /// Battery state + level, so the report proves the device was not on the charger.
    static func power() -> String {
        #if os(watchOS)
        let d = WKInterfaceDevice.current(); d.isBatteryMonitoringEnabled = true
        let state = d.batteryState, level = d.batteryLevel
        #else
        let d = UIDevice.current; d.isBatteryMonitoringEnabled = true
        let state = d.batteryState, level = d.batteryLevel
        #endif
        let name: String
        switch state {
        case .unplugged: name = "unplugged"; case .charging: name = "charging"; case .full: name = "full (on charger)"
        case .unknown: name = "unknown"; @unknown default: name = "other"
        }
        return level < 0 ? name : "\(name) \(Int(level * 100))%"
    }
    static func name(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly: return "cpuOnly"; case .cpuAndGPU: return "cpuAndGPU"; case .all: return "all (Neural Engine allowed)"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        @unknown default: return "other"
        }
    }
}
