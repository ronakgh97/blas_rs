use wide::f32x8;

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f32]) {
    use rayon::prelude::*;
    buf.par_iter_mut()
        .for_each_init(fastrand::Rng::new, |rng, x| {
            *x = rng.f32() * 2.0 - 1.0;
        });
}

/// Performs a horizontal add of an `f32x8` vector, returning the sum of all elements.
#[inline(always)]
pub fn from_f32x8(v: f32x8) -> f32 {
    let a = v.to_array();
    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}

#[test]
fn test_gen_fill() {
    let mut buf = vec![0.0f32; 16];
    gen_fill(&mut buf);
    for &x in &buf {
        assert!((-1.0..=1.0).contains(&x), "Value {} is out of range", x);
    }
    println!("Generated values: {:?}", buf);
}
