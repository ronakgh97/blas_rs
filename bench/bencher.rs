use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::f64::consts::PI;
use std::hint::black_box;
use std::time::Instant;

//TODO: bench will use `cblas` for openBLAS ffi, to compare against, but for now I will just use Intel VTune

//TODO: metrics to bench: size/gflops, size/toc, size/bdps, size/cpuc, size/effciency, arithmetic intensity, etc

#[allow(unused)]
/// Bench a function `f` by running it `runs` times and returning the average time per run in seconds
fn bench<F: FnMut()>(mut f: F, runs: usize, warmup: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }

    let start = Instant::now();

    for _ in 0..runs {
        f();
    }

    let dur = start.elapsed();
    dur.as_secs_f64() / runs as f64
}

#[allow(unused)]
enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    TimeofComplete(Vec<(f64, f64)>),
    Bandwidths(Vec<(f64, f64)>),
    CpuCycles(Vec<(f64, f64)>),
    // Efficiency(Vec<(f64, f64)>),
    ArithmeticIntensity(Vec<(f64, f64)>),
}

fn main() {
    let m = 4096;
    let n = 2048;

    let mut a = vec![0.0f32; m * n];
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; m];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    gen_fill(&mut a);
    gen_fill(&mut x);
    y.fill(1.0);

    // Warmup phase
    for i in 0..1024 {
        gemv(
            m,
            n,
            (i % 2) as f32,
            &a,
            m,
            &x,
            1,
            (i % 3) as f32,
            &mut y,
            1,
            false,
        );
        black_box(());

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);
    }

    drop(a);
    drop(x);
    drop(y);

    // BENCH TOTAL TAKE: `target_time` * vary_size.len() seconds
    let target_time = PI * PI * PI; // big enough i think
    let vary_size = [256, 512, 1024, 2048, 4096, 8192, 16384];

    let mut metrics: Vec<BenchMetrics> = Vec::new();

    let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
    let mut timeofcomplete_acc: Vec<(f64, f64)> = Vec::new();
    let mut bandwidths_acc: Vec<(f64, f64)> = Vec::new();
    let mut cpu_cycles_acc: Vec<(f64, f64)> = Vec::new();
    let mut arithmetic_intensity_acc: Vec<(f64, f64)> = Vec::new();

    for i in vary_size {
        let mut run_times = 0u64;
        let mut a_buf = vec![0.0f32; i * i];
        let mut x_buf = vec![0.0f32; i];
        let mut y_buf = vec![0.0f32; i];

        black_box(&mut a_buf);
        black_box(&mut x_buf);
        black_box(&mut y_buf);

        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);
        y_buf.fill(1.0);

        // Start time bound bench
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < target_time {
            gemv(
                i, // m
                i, // n
                5.0, &a_buf, // matrix
                i,      // lda
                &x_buf, // vec
                1, 7.0, &mut y_buf, // result y
                1, false,
            );
            run_times += 1;
        }
        let toc = start.elapsed().as_secs_f64();
        let gflops = 2.0 * i.pow(2) as f64 * run_times as f64 / toc / 1e9;
        let bandwidth =
            (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 * run_times as f64
                / (toc)
                / 1e9;
        let cpu_cycles = toc * 4e9 / run_times as f64; // 4GHz CPU, SHUT UP, DON'T QUESTION ME
        // This should approach to 0.5, which it does, that means IM CORRECT.
        let arithmetic_intensity = (2.0 * i as f64 * i as f64)
            / ((i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64); // FLOPs per byte

        gflops_acc.push((i as f64, gflops));
        timeofcomplete_acc.push((i as f64, toc));
        bandwidths_acc.push((i as f64, bandwidth));
        cpu_cycles_acc.push((i as f64, cpu_cycles));
        arithmetic_intensity_acc.push((i as f64, arithmetic_intensity));

        println!(
            "Size: {}, Runs: {}, Time: {:.4} sec, GFLOPS: {:.4}, Bandwidth: {:.4} GB/s, CPU Cycles: {:.2}, Arithmetic Intensity: {:.4} FLOPs/Byte",
            i, run_times, toc, gflops, bandwidth, cpu_cycles, arithmetic_intensity
        );

        // reset, avoid hot cache possibility, we don't do TRUTHS ME BRO AI BENCH
        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);
        y_buf.fill(1.0);
    }
    metrics.push(BenchMetrics::Gflops(gflops_acc));
    metrics.push(BenchMetrics::TimeofComplete(timeofcomplete_acc));
    metrics.push(BenchMetrics::Bandwidths(bandwidths_acc));
    metrics.push(BenchMetrics::CpuCycles(cpu_cycles_acc));
    metrics.push(BenchMetrics::ArithmeticIntensity(arithmetic_intensity_acc));
}

#[allow(unused)]
fn plot_bench(bench_metrics: Vec<BenchMetrics>) {
    use plotters::prelude::*;
}
