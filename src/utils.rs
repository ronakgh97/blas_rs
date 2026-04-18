use std::arch::x86_64::{__m256, _mm256_storeu_ps};

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f32], parallel: bool) {
    if parallel {
        use rayon::prelude::*;
        buf.par_iter_mut()
            .for_each_init(fastrand::Rng::new, |rng, x| {
                *x = rng.f32() * 2.0 - 1.0;
            });
    } else {
        for x in buf.iter_mut() {
            *x = fastrand::f32() * 2.0 - 1.0;
        }
    }
}

/// Performs a horizontal add (reduction) of an `__m256` vector and returns the result as a `f32`
#[inline(always)]
pub fn from_m256(v: __m256) -> f32 {
    unsafe {
        let tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_ptr() as *mut f32, v);
        tmp.iter().sum()
    }
}

#[test]
fn test_gen_fill() {
    let mut buf = vec![0.0f32; 8192];
    let seq = std::time::Instant::now();
    gen_fill(&mut buf, false);
    let seq_ep = seq.elapsed();

    buf.fill(0.0);

    let pl = std::time::Instant::now();
    gen_fill(&mut buf, true);
    let pl_ep = pl.elapsed();

    println!(
        "Sequential: {:.6} seconds, Parallel: {:.6} seconds",
        seq_ep.as_secs_f64(),
        pl_ep.as_secs_f64()
    );
}
