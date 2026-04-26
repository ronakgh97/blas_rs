use blas_rs::lvl1::{axpy, dot};
use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::error::Error;
use std::f64::consts::PI;
use std::hint::black_box;
use std::path::Path;
use std::sync::LazyLock;
use std::time::Instant;
// Bench uses this: https://github.com/OpenMathLib/OpenBLAS/tree/develop/benchmark as ref

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

/// Returns the cache sizes (L1, L2, L3) in KB for the current CPU using CPUID
pub fn get_cache_size() -> (usize, usize, usize) {
    let mut l1 = 0;
    let mut l2 = 0;
    let mut l3 = 0;

    let mut i = 0;

    loop {
        let res = std::arch::x86_64::__cpuid_count(4, i);

        let cache_type = res.eax & 0x1F;
        if cache_type == 0 {
            break;
        }

        let level = (res.eax >> 5) & 0x7;

        let ways = ((res.ebx >> 22) & 0x3FF) + 1;
        let partitions = ((res.ebx >> 12) & 0x3FF) + 1;
        let line_size = (res.ebx & 0xFFF) + 1;
        let sets = res.ecx + 1;

        let size_kb = (ways * partitions * line_size * sets) as usize / 1024;

        match (level, cache_type) {
            (1, 1) => l1 = size_kb,
            (2, 3) => l2 = size_kb,
            (3, 3) => l3 = size_kb,
            _ => {}
        }

        i += 1;
    }

    (l1, l2, l3)
}

static MAX_LLC_KB: LazyLock<f64> = LazyLock::new(|| {
    let (l1, l2, _l3) = get_cache_size();
    // TODO: taking just l1 & l2
    (l1 + l2) as f64 // total cache capacity in KB (used as a fit proxy, not bandwidth)
});
enum BenchMetrics {
    Gflops(Vec<(f64, f64)>),
    AvgTimePerCall(Vec<(f64, f64)>),
    CacheEfficiency(Vec<(f64, f64)>),
    CacheMiss(Vec<(f64, f64)>),
    // Efficiency(Vec<(f64, f64)>), <- we need openBlas for that, grand finale
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
    for i in 0..4096 {
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

        axpy(m, (i % 5) as f32, &y, 1, &mut vec![0.0f32; m], 1);
        black_box(());

        dot(m, &y, 1, &vec![0.0f32; m], 1);
        black_box(());

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);
    }

    drop(a);
    drop(x);
    drop(y);

    // BENCH TOTAL `target_time` * sample_len * no of bench seconds
    let target_time = PI * PI; // big enough to run least some runs in largest samples
    let size_sample = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768];

    let plot_path = Path::new("./bench");

    // axpy & dot
    {
        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
        let mut avgcalls_acc: Vec<(f64, f64)> = Vec::new();
        let mut cache_eff_acc: Vec<(f64, f64)> = Vec::new();
        let mut cachemiss_acc: Vec<(f64, f64)> = Vec::new();

        for i in size_sample {
            let mut x_buf = vec![0.0f32; i];
            let mut y_buf = vec![0.0f32; i];

            let mut run_times = 0f64;
            black_box(&mut x_buf);
            black_box(&mut y_buf);

            gen_fill(&mut x_buf);
            y_buf.fill(1.0);

            // Start time bound bench
            let start = Instant::now();
            while start.elapsed().as_secs_f64() < target_time {
                axpy(i, 3.0, &x_buf, 1, &mut y_buf, 1);
                black_box(());
                run_times += 1.0;
            }
            let toc = start.elapsed().as_secs_f64();
            let total_flops = 2.0 * i as f64 * run_times; // 2n FLOPs per axpy call
            let gflops = total_flops / toc / 1e9;
            let working_kbyte = 3.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read/write
            let cache_eff = ((*MAX_LLC_KB / working_kbyte) * 100.0).min(100.0); // LLC fit proxy
            let avg_time_per_call = toc / run_times * 1e9; // nanoseconds
            let ns_per_flop = (toc * 1e9) / total_flops;

            gflops_acc.push(((i as f64).log10(), gflops));
            avgcalls_acc.push((i as f64, avg_time_per_call.log10()));
            cache_eff_acc.push(((i as f64).log10(), cache_eff));
            cachemiss_acc.push((working_kbyte.log10(), ns_per_flop)); // seconds per flop, higher means more cache miss

            println!(
                "Size: {}, Runs: {}, Time: {} sec, GFLOPS: {}, Cache Efficiency: {} %",
                i, run_times, toc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility, we don't do TRUST ME BRO BENCH
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }
        metrics.push(BenchMetrics::Gflops(gflops_acc));
        metrics.push(BenchMetrics::AvgTimePerCall(avgcalls_acc));
        metrics.push(BenchMetrics::CacheEfficiency(cache_eff_acc));
        metrics.push(BenchMetrics::CacheMiss(cachemiss_acc));

        match plot_bench(&metrics, &plot_path.join("axpy.png")) {
            Ok(_) => println!("Export to bench/axpy.png"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }

        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
        let mut avgcalls_acc: Vec<(f64, f64)> = Vec::new();
        let mut cache_eff_acc: Vec<(f64, f64)> = Vec::new();
        let mut cachemiss_acc: Vec<(f64, f64)> = Vec::new();

        for i in size_sample {
            let mut run_times = 0u64;
            let mut x_buf = vec![0.0f32; i];
            let mut y_buf = vec![0.0f32; i];

            black_box(&mut x_buf);
            black_box(&mut y_buf);

            gen_fill(&mut x_buf);
            gen_fill(&mut y_buf);

            // Start time bound bench
            let start = Instant::now();
            while start.elapsed().as_secs_f64() < target_time {
                black_box(dot(i, &x_buf, 1, &y_buf, 1));
                run_times += 1;
            }
            let toc = start.elapsed().as_secs_f64();
            let total_flops = 2.0 * i as f64 * run_times as f64;
            let gflops = total_flops / toc / 1e9;
            let working_kbyte = 2.0 * i as f64 * size_of::<f32>() as f64 / 1024.0; // x read + y read
            let cache_eff = ((*MAX_LLC_KB / working_kbyte) * 100.0).min(100.0); // LLC fit proxy
            let avg_time_per_call = toc / run_times as f64 * 1e9; // nanoseconds
            let ns_per_flop = (toc * 1e9) / total_flops;

            gflops_acc.push(((i as f64).log10(), gflops));
            avgcalls_acc.push((i as f64, avg_time_per_call.log10()));
            cache_eff_acc.push(((i as f64).log10(), cache_eff));
            cachemiss_acc.push((working_kbyte.log10(), ns_per_flop));

            println!(
                "Size: {}, Runs: {}, Time: {} sec, GFLOPS: {}, Cache Efficiency: {} %",
                i, run_times, toc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility, we don't do 'TRUST ME BRO' BENCH
            gen_fill(&mut x_buf);
            gen_fill(&mut y_buf);
        }

        metrics.push(BenchMetrics::Gflops(gflops_acc));
        metrics.push(BenchMetrics::AvgTimePerCall(avgcalls_acc));
        metrics.push(BenchMetrics::CacheEfficiency(cache_eff_acc));
        metrics.push(BenchMetrics::CacheMiss(cachemiss_acc));

        match plot_bench(&metrics, &plot_path.join("dot.png")) {
            Ok(_) => println!("Export to bench/dot.png"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }
    }

    // Gemv & Gemv_t
    {
        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
        let mut avgcalls_acc: Vec<(f64, f64)> = Vec::new();
        let mut cache_eff_acc: Vec<(f64, f64)> = Vec::new();
        let mut cachemiss_acc: Vec<(f64, f64)> = Vec::new();

        for i in size_sample {
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
            let total_flops = 2.0 * i.pow(2) as f64 * run_times as f64;
            let gflops = total_flops / toc / 1e9;
            let working_kbyte =
                (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;
            let cache_eff = ((*MAX_LLC_KB / working_kbyte) * 100.0).min(100.0); // LLC fit proxy
            let avg_time_per_call = toc / run_times as f64 * 1e9; // nanoseconds
            let ns_per_flop = (toc * 1e9) / total_flops;

            gflops_acc.push(((i as f64).log10(), gflops));
            avgcalls_acc.push((i as f64, avg_time_per_call.log10()));
            cache_eff_acc.push(((i as f64).log10(), cache_eff));
            cachemiss_acc.push((working_kbyte.log10(), ns_per_flop));

            println!(
                "Size: {}, Runs: {}, Time: {} sec, GFLOPS: {}, Cache Efficiency: {} %",
                i, run_times, toc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility
            gen_fill(&mut a_buf);
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }
        metrics.push(BenchMetrics::Gflops(gflops_acc));
        metrics.push(BenchMetrics::AvgTimePerCall(avgcalls_acc));
        metrics.push(BenchMetrics::CacheEfficiency(cache_eff_acc));
        metrics.push(BenchMetrics::CacheMiss(cachemiss_acc));

        match plot_bench(&metrics, &plot_path.join("gemv.png")) {
            Ok(_) => println!("Export to bench/gemv.png"),
            Err(err) => {
                eprintln!("failed to render benchmark plot: {err}");
            }
        }

        let mut metrics: Vec<BenchMetrics> = Vec::new();

        let mut gflops_acc: Vec<(f64, f64)> = Vec::new();
        let mut avgcalls_acc: Vec<(f64, f64)> = Vec::new();
        let mut cache_eff_acc: Vec<(f64, f64)> = Vec::new();
        let mut cachemiss_acc: Vec<(f64, f64)> = Vec::new();

        for i in size_sample {
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
                    1, true,
                );
                run_times += 1;
            }
            let toc = start.elapsed().as_secs_f64();
            let total_flops = 2.0 * i.pow(2) as f64 * run_times as f64;
            let gflops = total_flops / toc / 1e9;
            let working_kbyte =
                (i.pow(2) as f64 + 3.0 * i as f64) * size_of::<f32>() as f64 / 1024.0;
            let cache_eff = ((*MAX_LLC_KB / working_kbyte) * 100.0).min(100.0); // LLC fit proxy
            let avg_time_per_call = toc / run_times as f64 * 1e9; // nanoseconds
            let ns_per_flop = (toc * 1e9) / total_flops;

            gflops_acc.push(((i as f64).log10(), gflops));
            avgcalls_acc.push((i as f64, avg_time_per_call.log10()));
            cache_eff_acc.push(((i as f64).log10(), cache_eff));
            cachemiss_acc.push((working_kbyte.log10(), ns_per_flop));

            println!(
                "Size: {}, Runs: {}, Time: {} sec, GFLOPS: {}, Cache Efficiency: {} %",
                i, run_times, toc, gflops, cache_eff
            );

            // reset, avoid hot cache possibility, we don't do TRUTHS ME BRO AI BENCH
            gen_fill(&mut a_buf);
            gen_fill(&mut x_buf);
            y_buf.fill(1.0);
        }
        metrics.push(BenchMetrics::Gflops(gflops_acc));
        metrics.push(BenchMetrics::AvgTimePerCall(avgcalls_acc));
        metrics.push(BenchMetrics::CacheEfficiency(cache_eff_acc));
        metrics.push(BenchMetrics::CacheMiss(cachemiss_acc));

        match plot_bench(&metrics, &plot_path.join("gemv_t.png")) {
            Ok(_) => println!("Export to bench/gemv_t.png"),
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
                    "Avg Time Per Call",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheEfficiency(points)) => (
                    "log(Size)",
                    "L1 L2 Fit Proxy (%)",
                    GREEN,
                    "Cache Fit",
                    points.as_slice(),
                ),
                Some(BenchMetrics::CacheMiss(points)) => (
                    "log(Working Set KB)",
                    "ns / FLOP",
                    MAGENTA,
                    "Time Per FLOP Proxy",
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
