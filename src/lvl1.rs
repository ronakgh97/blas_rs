use std::ops::{Add, Mul};
use std::ptr::{read_unaligned, write_unaligned};
use wide::f32x8;

#[inline(always)]
pub fn axpy(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 {
        return;
    }

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        // If both increments are 1, we can use simd, data are contiguous
        if incx == 1 && incy == 1 {
            // Process 8 elements at a time using simd
            let mut i = 0;
            while i + 8 <= n {
                let x_chunk = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let y_chunk = read_unaligned(y_ptr.add(i) as *const [f32; 8]);

                let x_vec = f32x8::from(x_chunk);
                let y_vec = f32x8::from(y_chunk);

                // Y = alpha * X + Y
                let res = y_vec.add(f32x8::splat(alpha).mul(x_vec));
                write_unaligned(y_ptr.add(i) as *mut [f32; 8], res.to_array()); // Write the result back to Y

                i += 8;
            }

            // Handle remaining elements
            for i in i..n {
                *y_ptr.add(i) += alpha * *x_ptr.add(i);
            }
        } else {
            // Start from the end of the array and move backwards, if inc* are negative
            let mut ix = if incx < 0 {
                (n as isize - 1) * -(incx as isize)
            } else {
                0
            };
            let mut iy = if incy < 0 {
                (n as isize - 1) * -(incy as isize)
            } else {
                0
            };

            for _ in 0..n {
                // Y = alpha * X + Y
                *y_ptr.offset(iy) += alpha * *x_ptr.offset(ix);
                ix += incx as isize;
                iy += incy as isize;
            }
        }
    }
}

#[test]
fn test_axpy() {
    use crate::utils::gen_fill;
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0.0; 8];

    // Basic test
    axpy(8, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    y.fill(0.0);

    // Test with negative increments
    axpy(8, 2.0, &x, 1, &mut y, -1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);
    axpy(8, 2.0, &x, -1, &mut y, 1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    // Test with non-unit increments
    axpy(4, 2.0, &x, 2, &mut y, 2);
    assert_eq!(y, vec![0.0, 0.0, 4.0, 0.0, 8.0, 0.0, 12.0, 0.0]);
    y.fill(0.0);

    // Test with zero length
    axpy(0, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    // Test with alpha = 0
    axpy(8, 0.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    let mut gen_x = vec![0.0f32; 1024];
    let mut gen_y = vec![0.0f32; 1024];

    gen_fill(&mut gen_x);
    gen_fill(&mut gen_y);

    // Test glops
    let start = std::time::Instant::now();
    for _ in 0..1024 {
        axpy(gen_x.len(), 9.0, &gen_x, 1, &mut gen_y, 1);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let gflops = (2.0 * gen_x.len() as f64 * 1024.0) / (elapsed * 1e9);

    println!(
        "Elapsed time: {:.6} seconds, GFLOPS: {:.2}",
        elapsed, gflops
    );
}
