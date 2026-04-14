use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use wide::f32x8;

/// Fills an existing buffer with random values in range `[-1.0, 1.0]`
#[inline]
pub fn gen_fill(buf: &mut [f32]) {
    buf.par_iter_mut().for_each(|x| {
        *x = fastrand::f32() * 2.0 - 1.0;
    });
}

/// Sums all elements in a f32x8 vector and returns the result as a single f32 value
#[inline(always)]
pub fn from_f32x8(v: f32x8) -> f32 {
    let a = v.to_array();
    a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7]
}
