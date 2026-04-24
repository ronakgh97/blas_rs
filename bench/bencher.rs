use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::error::Error;
use std::f64::consts::PI;
use std::hint::black_box;
use std::path::Path;
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

enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    RunCount(Vec<(f64, f64)>),
    Bandwidths(Vec<(f64, f64)>),
    ArithmeticIntensity(Vec<(f64, f64)>),
    // Efficiency(Vec<(f64, f64)>),
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
    let target_time = PI * PI * PI; // big enough to run atleast some runs in largest samples
    let vary_size = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

    let mut metrics: Vec<BenchMetrics> = Vec::new();

    let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
    let mut runcount_acc: Vec<(f64, f64)> = Vec::new();
    let mut bandwidths_acc: Vec<(f64, f64)> = Vec::new();
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
        // This should approach to 0.5, which it does, that means I'M CORRECT.
        let arithmetic_intensity = (2.0 * i as f64 * i as f64)
            / ((i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64); // FLOPs per byte

        gflops_acc.push(((i as f64).log10(), gflops));
        runcount_acc.push((i as f64, (run_times as f64).log10()));
        bandwidths_acc.push(((i as f64).log10(), bandwidth));
        arithmetic_intensity_acc.push(((i as f64).log10(), arithmetic_intensity));

        println!(
            "Size: {}, Runs: {}, Time: {} sec, GFLOPS: {}, Bandwidth: {} GB/s, Arithmetic Intensity: {} FLOPs/Byte",
            i, run_times, toc, gflops, bandwidth, arithmetic_intensity
        );

        // reset, avoid hot cache possibility, we don't do TRUTHS ME BRO AI BENCH
        gen_fill(&mut a_buf);
        gen_fill(&mut x_buf);
        y_buf.fill(1.0);
    }
    metrics.push(BenchMetrics::Gflops(gflops_acc));
    metrics.push(BenchMetrics::RunCount(runcount_acc));
    metrics.push(BenchMetrics::Bandwidths(bandwidths_acc));
    metrics.push(BenchMetrics::ArithmeticIntensity(arithmetic_intensity_acc));

    match plot_bench(&metrics, Path::new("./bench/plot.png")) {
        Ok(_) => println!("Benchmark plot saved to bench/plot.png"),
        Err(err) => {
            eprintln!("failed to render benchmark plot: {err}");
        }
    }
}

fn plot_bench(bench_metrics: &[BenchMetrics], output: &Path) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new(&output, (1200, 900)).into_drawing_area();
    root.fill(&BLACK)?;

    let chart_area = root.split_evenly((2, 2)).into_iter().enumerate();

    for (idx, area) in chart_area {
        let (x_label, y_label, color, label, points): (&str, &str, RGBColor, &str, &[(f64, f64)]) =
            match bench_metrics.get(idx) {
                Some(BenchMetrics::Gflops(points)) => {
                    ("log(Size)", "GFLOPS", RED, "GFLOPS", points.as_slice())
                }
                Some(BenchMetrics::RunCount(points)) => (
                    "Size",
                    "log(Run Count)",
                    BLUE,
                    "Throughput",
                    points.as_slice(),
                ),
                Some(BenchMetrics::Bandwidths(points)) => (
                    "log(Size)",
                    "Bandwidth (GB/s)",
                    GREEN,
                    "Bandwidth",
                    points.as_slice(),
                ),
                Some(BenchMetrics::ArithmeticIntensity(points)) => (
                    "log(Size)",
                    "FLOPs / Byte",
                    MAGENTA,
                    "Arithmetic Intensity",
                    points.as_slice(),
                ),
                None => ("X", "Y", WHITE, "N/A", &[]),
            };

        let (x_range, y_range) = if points.is_empty() {
            (0.0..1.0, 0.0..1.0)
        } else {
            let (mut x_min, mut x_max) = (points[0].0, points[0].0);
            let (mut y_min, mut y_max) = (points[0].1, points[0].1);

            for &(x, y) in points.iter().skip(1) {
                x_min = x_min.min(x);
                x_max = x_max.max(x);
                y_min = y_min.min(y);
                y_max = y_max.max(y);
            }

            let x_span = (x_max - x_min).abs();
            let y_span = (y_max - y_min).abs();
            let x_padding = if x_span == 0.0 {
                x_min.abs().max(1.0) * 0.1
            } else {
                x_span * 0.1
            };
            let y_padding = if y_span == 0.0 {
                y_min.abs().max(1.0) * 0.1
            } else {
                y_span * 0.1
            };

            (
                (x_min - x_padding)..(x_max + x_padding),
                (y_min - y_padding)..(y_max + y_padding),
            )
        };

        let mut chart = ChartBuilder::on(&area)
            .caption(label, ("0xProto Nerd Font", 20).into_font().color(&color))
            .x_label_area_size(40)
            .y_label_area_size(60)
            .margin(20)
            .build_cartesian_2d(x_range, y_range)?;

        chart
            .configure_mesh()
            .label_style(("0xProto Nerd Font", 14).into_font().color(&WHITE))
            .axis_desc_style(("0xProto Nerd Font", 16).into_font().color(&WHITE))
            .x_desc(x_label)
            .y_desc(y_label)
            .bold_line_style(WHITE.mix(0.3))
            .light_line_style(WHITE.mix(0.15))
            .draw()?;

        if !points.is_empty() {
            chart.draw_series(LineSeries::new(points.iter().copied(), color))?;

            chart.draw_series(
                points
                    .iter()
                    .map(|(x, y)| (*x, *y))
                    .map(|pos| Circle::new(pos, 5, color.filled())),
            )?;
        }
    }

    root.present()?;

    Ok(())
}
