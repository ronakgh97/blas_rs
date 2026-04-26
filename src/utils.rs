#[allow(unused)]
use std::arch::x86_64::{__cpuid_count, __m256, _mm256_storeu_ps};

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline(always)]
pub fn gen_fill(buf: &mut [f32]) {
    for x in buf.iter_mut() {
        *x = fastrand::f32() * 2.0 - 1.0;
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
    let mut buf = vec![0.0f32; 99999999];
    let strt = std::time::Instant::now();
    gen_fill(&mut buf);
    let elp = strt.elapsed();
    println!(
        "Generated {} random numbers in {:?} seconds",
        buf.len(),
        elp.as_secs_f32()
    );
    assert!(buf.iter().all(|&x| (-1.0..=1.0).contains(&x)));
}
