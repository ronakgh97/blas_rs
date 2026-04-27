mod harness;
mod utils;

use crate::harness::{MetricSet, run_bench};
use blas_rs::lvl1::{axpy, dot};
use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::error::Error;
use std::f64::consts::PI;
use std::hint::black_box;
use std::path::Path;

extern crate openblas_src;

// Bench uses this as ref: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark as ref

fn warmup(r: usize, s: usize) {
    let mut a = vec![0.0f32; s * s];
    let mut x = vec![0.0f32; s];
    let mut y = vec![0.0f32; s];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    gen_fill(&mut a);
    gen_fill(&mut x);
    y.fill(1.0);

    // Warmup phase
    for i in 0..r {
        gemv(
            s,
            s,
            (i % 3) as f32,
            &a,
            s,
            &x,
            1,
            (i % 5) as f32,
            &mut y,
            1,
            false,
        );

        gemv(
            s,
            s,
            (i % 3) as f32,
            &a,
            s,
            &x,
            1,
            (i % 5) as f32,
            &mut y,
            1,
            true,
        );
        axpy(s, (i % 7) as f32, &y, 1, &mut vec![0.0f32; s], 1);

        dot(s, &y, 1, &vec![0.0f32; s], 1);

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);
    }
}

enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    AvgTimePerCall(Vec<(f64, f64)>),
    CacheEfficiency(Vec<(f64, f64)>), // LLC fit proxy
    // a bit heuristic, but we can see how much performance cost we pay per flop as working set grows, higher means more cache miss
    CacheMiss(Vec<(f64, f64)>),
    // Efficiency(Vec<(f64, f64)>), <- we need openBlas for that, grand finale
}

fn main() {
    unsafe {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }

    warmup(12, 8192);

    // bit yap about axpy and dot, axpy is slightly overhead, due to` _mm256_storeu_ps`, write heavy,
    // so its write-bandwidth bound, even it whole vectors fits in cache and also "cache" does not help much since those 8 lanes are not reused.
    // while dot is just most read heavy, and accumulated in 4 sum register, less overhead, finally the `cache miss` is heuristic metrics as say done

    // BENCH TOTAL `target_time` * sample_len * no of bench seconds
    let target_time = PI * PI; // big enough to run least some runs in largest samples
    let size_sample = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

    let plot_path = Path::new("./bench");

    // axpy
    {
        let mut bench_metrics: Vec<BenchMetrics> = Vec::new();

        let mut metrics_collector = MetricSet::new();

        for i in size_sample {
            let mut x_buf = vec![0.0f32; i];
            let mut y_buf = vec![0.0f32; i];

            black_box(&mut x_buf);
            black_box(&mut y_buf);

            gen_fill(&mut x_buf);
            y_buf.fill(1.0);

            // Start time bound bench
            let rc = run_bench(|| axpy(i, 3.0, &x_buf, 1, &mut y_buf, 1), target_time);

            let total_flops = 2.0 * i as f64 * rc; // 2n FLOPs per axpy call
            let working_kb = 3.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write

            let (gflops, avg_time_per_call, cache_eff, ns_per_flop) =
                MetricSet::derive(rc, target_time, total_flops, working_kb);

            metrics_collector.collect(
                ((i as f64).log10(), gflops),
                (i as f64, avg_time_per_call.log10()),
                ((i as f64).log10(), cache_eff),
                (working_kb.log10(), ns_per_flop),
            );

            println!(
                "S: {}, R: {}, GFLOPS: {}, Cache Efficiency: {} %",
                i, rc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility, we don't do TRUST ME BRO BENCH
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }
        metrics_collector.finalize(&mut bench_metrics);

        match plot_bench(&bench_metrics, &plot_path.join("axpy.png")) {
            Ok(_) => println!("Exported"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }
    }

    // dot
    {
        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut metrics_collector = MetricSet::new();

        for i in size_sample {
            let mut x_buf = vec![0.0f32; i];
            let mut y_buf = vec![0.0f32; i];

            black_box(&mut x_buf);
            black_box(&mut y_buf);

            gen_fill(&mut x_buf);
            gen_fill(&mut y_buf);

            // Start time bound bench

            let rc = run_bench(
                || {
                    dot(i, &x_buf, 1, &y_buf, 1);
                },
                target_time,
            );

            let total_flops = 2.0 * i as f64 * rc;
            let working_kb = 2.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read

            let (gflops, avg_time_per_call, cache_eff, ns_per_flop) =
                MetricSet::derive(rc, target_time, total_flops, working_kb);

            metrics_collector.collect(
                ((i as f64).log10(), gflops),
                (i as f64, avg_time_per_call.log10()),
                ((i as f64).log10(), cache_eff),
                (working_kb.log10(), ns_per_flop),
            );

            println!(
                "S: {}, R: {}, GFLOPS: {}, Cache Efficiency: {} %",
                i, rc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility, we don't do 'TRUST ME BRO' BENCH
            gen_fill(&mut x_buf);
            gen_fill(&mut y_buf);
        }

        metrics_collector.finalize(&mut metrics);

        match plot_bench(&metrics, &plot_path.join("dot.png")) {
            Ok(_) => println!("Exported"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }
    }

    // Gemv & Gemv_t
    {
        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut metrics_collector = MetricSet::new();

        for i in size_sample {
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
            let rc = run_bench(
                || {
                    gemv(
                        i, // m
                        i, // n
                        5.0, &a_buf, // matrix
                        i,      // lda
                        &x_buf, // vec
                        1, 7.0, &mut y_buf, // result y
                        1, false,
                    );
                },
                target_time,
            );

            let total_flops = 2.0 * i.pow(2) as f64 * rc;
            // no of (matrix + vector) element * write/read ops (2)
            let working_kbyte =
                (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;

            let (gflops, avg_time_per_call, cache_eff, ns_per_flop) =
                MetricSet::derive(rc, target_time, total_flops, working_kbyte);

            metrics_collector.collect(
                ((i as f64).log10(), gflops),
                (i as f64, avg_time_per_call.log10()),
                ((i as f64).log10(), cache_eff),
                (working_kbyte.log10(), ns_per_flop),
            );

            println!(
                "S: {}, R: {}, GFLOPS: {}, Cache Efficiency: {} %",
                i, rc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility
            gen_fill(&mut a_buf);
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }

        metrics_collector.finalize(&mut metrics);

        match plot_bench(&metrics, &plot_path.join("gemv.png")) {
            Ok(_) => println!("Exported"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }
    }

    {
        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut metrics_collector = MetricSet::new();

        for i in size_sample {
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
            let rc = run_bench(
                || {
                    gemv(
                        i, // m
                        i, // n
                        5.0, &a_buf, // matrix
                        i,      // lda
                        &x_buf, // vec
                        1, 7.0, &mut y_buf, // result y
                        1, true,
                    );
                },
                target_time,
            );
            let total_flops = 2.0 * i.pow(2) as f64 * rc;
            let working_kbyte =
                (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;

            let (gflops, avg_time_per_call, cache_eff, ns_per_flop) =
                MetricSet::derive(rc, target_time, total_flops, working_kbyte);

            metrics_collector.collect(
                ((i as f64).log10(), gflops),
                (i as f64, avg_time_per_call.log10()),
                ((i as f64).log10(), cache_eff),
                (working_kbyte.log10(), ns_per_flop),
            );

            println!(
                "S: {}, R: {}, GFLOPS: {}, Cache Efficiency: {} %",
                i, rc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility
            gen_fill(&mut a_buf);
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }

        metrics_collector.finalize(&mut metrics);

        match plot_bench(&metrics, &plot_path.join("gemv_t.png")) {
            Ok(_) => println!("Exported"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }
    }
}

#[inline(always)]
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
                Some(BenchMetrics::AvgTimePerCall(points)) => (
                    "Size",
                    "log(Time per Call (ns))",
                    BLUE,
                    "Latency",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheEfficiency(points)) => (
                    "log(Size)",
                    "L1+L2 Fit Proxy (%)",
                    GREEN,
                    "Cache Fit",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheMiss(points)) => (
                    "log(Working Set KB)",
                    "ns / FLOP",
                    CYAN,
                    "Perf Cost",
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
