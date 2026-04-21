use blas_rs::lvl2::gemv;
use blas_rs::utils::*;
use std::f32::consts::PI;
use std::hint::black_box;
use std::time::Instant;

//TODO: bench will use `cblas` for openBLAS ffi, to compare against, but for now I will just use Intel VTune

#[allow(unused)]
fn bench<F: FnMut()>(mut f: F, iters: usize, warmup: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }

    let start = Instant::now();

    for _ in 0..iters {
        f();
    }

    let dur = start.elapsed();
    dur.as_secs_f64() / iters as f64
}

fn main() {
    let m = 4096;
    let n = 4096;

    let mut a = vec![0.0f32; m * n];
    let mut x = vec![0.0f32; n];
    let mut y = vec![0.0f32; m];

    black_box(&mut a);
    black_box(&mut x);
    black_box(&mut y);

    gen_fill(&mut a);
    gen_fill(&mut x);
    y.fill(1.0);

    // warmup
    for _ in 0..12 {
        gemv(m, n, 8.0, &a, m, &x, 1, 6.0, &mut y, 1, false);

        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0);
    }
    let mut iters = 0u64;

    let timer = Instant::now();

    // classic time bound bench
    loop {
        gemv(m, n, 8.0, &a, m, &x, 1, 6.0, &mut y, 1, false);
        iters += 1;
        // fill with random data to avoid hot cache
        gen_fill(&mut a);
        gen_fill(&mut x);
        y.fill(1.0); // reset y to avoid hot cache
        if timer.elapsed().as_secs_f32() >= PI.powi(2) {
            println!(
                "gflops: {}, ran: {}times",
                2.0 * (m as f32) * (n as f32) * (iters as f32)
                    / timer.elapsed().as_secs_f32()
                    / 1e9,
                iters
            );
            // timer = Instant::now();
            // iters = 0;
            break; // <- cmt this, if you wanna get horny
        }
    }
}
