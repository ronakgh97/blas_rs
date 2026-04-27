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

#[inline]
/// Returns the cache sizes (L1, L2, L3) in KB for the current CPU using CPUID
pub fn get_cache_size() -> (usize, usize, usize) {
    let mut l1 = 0;
    let mut l2 = 0;
    let mut l3 = 0;

    let mut i = 0;

    loop {
        let res = __cpuid_count(4, i);

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
