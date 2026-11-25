import SwiftUI
import CoreML

/// Two buttons: the short benchmark over every bundled model (both compute modes) and a sustained
/// streaming run of EXACT at the deployment cadence. Reports go to the screen, the console and
/// Documents/benchmark.txt (appended, one stamped block per run).
///
/// Launch arguments (e.g. `devicectl device process launch ... --autorun --exit --sustain 300`):
///   --autorun            run the short benchmark on appear
///   --sustain <seconds>  after it, run the sustained benchmark for that long (0 = skip)
///   --iterations <n>     timed iterations per model/mode (default 300)
///   --models a,b,c       restrict the short benchmark to these bundled model names
///   --exit               quit when done, so a `--console` launch returns
struct ContentView: View {
    @State private var report = "Short: every model, both compute modes. Sustained: EXACT at one window / 0.5 s. Keep the watch on the wrist, off the charger, display awake."
    @State private var running = false
    @State private var status = ""

    private var args: [String] { CommandLine.arguments }
    private func arg(_ key: String) -> String? { args.firstIndex(of: key).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button(running ? "Running…" : "Run benchmark") { run(sustainS: Double(arg("--sustain") ?? "0") ?? 0) }.disabled(running)
                    Button("Sustained 60 s") { run(shortToo: false, sustainS: 60) }.disabled(running)
                }
                if !status.isEmpty { Text(status).font(.footnote) }
                #if os(watchOS)
                Text(report).font(.system(.footnote, design: .monospaced))
                #else
                Text(report).font(.system(.footnote, design: .monospaced)).textSelection(.enabled)
                #endif
            }.padding()
        }
        .onAppear { if args.contains("--autorun") && !running { run(sustainS: Double(arg("--sustain") ?? "0") ?? 0) } }
    }

    /// Appends one stamped block to Documents/benchmark.txt right away, so a run that watchOS suspends
    /// half-way (display off) still leaves the models it finished.
    private static func persist(_ block: String) {
        guard let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = docs.appendingPathComponent("benchmark.txt")
        let stamped = "### run at \(ISO8601DateFormatter().string(from: Date()))\n" + block + "\n\n"
        if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(Data(stamped.utf8)); h.closeFile() }
        else { try? stamped.write(to: url, atomically: true, encoding: .utf8) }
    }

    private func run(shortToo: Bool = true, sustainS: Double) {
        running = true; report = ""; status = "starting"
        let iterations = Int(arg("--iterations") ?? "300") ?? 300
        let models = arg("--models")?.split(separator: ",").map(String.init) ?? BenchmarkRunner.defaultModels
        DispatchQueue.global(qos: .userInitiated).async {
            var lines: [String] = []
            let runner = BenchmarkRunner()
            if shortToo {
                for name in models {
                    for units in [MLComputeUnits.all, MLComputeUnits.cpuOnly] {
                        DispatchQueue.main.async { status = "\(name) \(BenchmarkRunner.name(units))" }
                        do { let r = try runner.run(model: name, computeUnits: units, iterations: iterations); lines.append(r.description); print(r); Self.persist(r.description) }
                        catch { lines.append("FAILED \(name) (\(BenchmarkRunner.name(units))): \(error)"); print(error) }
                    }
                }
            }
            if sustainS > 0 {
                for units in [MLComputeUnits.cpuOnly, MLComputeUnits.all] {
                    do {
                        let r = try runner.runSustained(model: "EXACT_physiq", computeUnits: units, durationS: sustainS) { k, dt in
                            DispatchQueue.main.async { status = String(format: "sustained %@: window %d, %.2f ms", BenchmarkRunner.name(units), k, dt) }
                        }
                        lines.append(r.description); print(r); Self.persist(r.description)
                    } catch { lines.append("FAILED sustained (\(BenchmarkRunner.name(units))): \(error)"); print(error) }
                }
            }
            let text = lines.joined(separator: "\n\n")
            DispatchQueue.main.async { report = text; status = "done"; running = false }
            if args.contains("--exit") { print("BENCHMARK DONE"); fflush(stdout); exit(0) }
        }
    }
}
