use std::ops::{Add, Mul};
use std::ptr::{read_unaligned, write_unaligned};
use wide::f32x8;

#[inline(always)]
pub fn axpy(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            let mut i = 0;

            let alpha_x8 = f32x8::splat(alpha);

            // Process 16 elements at a time
            while i + 16 <= n {
                let x0 = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let y0 = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                let x1 = f32x8::from(read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]));
                let y1 = f32x8::from(read_unaligned(y_ptr.add(i + 8) as *const [f32; 8]));

                let res0 = y0.add(alpha_x8.mul(x0));
                let res1 = y1.add(alpha_x8.mul(x1));

                write_unaligned(y_ptr.add(i) as *mut [f32; 8], res0.to_array());
                write_unaligned(y_ptr.add(i + 8) as *mut [f32; 8], res1.to_array());

                i += 16;
            }

            // Handle 8 elements if left after processing 16
            while i + 8 <= n {
                let x_chunk = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let y_chunk = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                let res = y_chunk.add(alpha_x8.mul(x_chunk));
                write_unaligned(y_ptr.add(i) as *mut [f32; 8], res.to_array());
                i += 8;
            }

            // Handle remaining elements
            while i < n {
                *y_ptr.add(i) += alpha * *x_ptr.add(i);
                i += 1;
            }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 {
                (n as isize - 1) * incx.abs()
            } else {
                0
            };
            let mut iy = if incy < 0 {
                (n as isize - 1) * incy.abs()
            } else {
                0
            };

            for _ in 0..n {
                // Y += alpha * X
                *y_ptr.offset(iy) += alpha * *x_ptr.offset(ix);
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
pub fn scal(n: usize, alpha: f32, x: &mut [f32], incx: i32) {
    if n == 0 || alpha == 1.0 {
        return;
    }

    if incx == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    unsafe {
        let x_ptr = x.as_mut_ptr();

        if incx == 1 {
            if alpha == 0.0 {
                std::ptr::write_bytes(x_ptr, 0u8, n);
                // for i in 0..n {
                //     *x_ptr.add(i) = 0.0;
                // }
                return;
            }

            let alpha_x8 = f32x8::splat(alpha);
            let mut i = 0;

            while i + 16 <= n {
                let x0 = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let x1 = f32x8::from(read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]));

                write_unaligned(x_ptr.add(i) as *mut [f32; 8], alpha_x8.mul(x0).to_array());
                write_unaligned(
                    x_ptr.add(i + 8) as *mut [f32; 8],
                    alpha_x8.mul(x1).to_array(),
                );

                i += 16;
            }

            while i + 8 <= n {
                let x_chunk = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                write_unaligned(
                    x_ptr.add(i) as *mut [f32; 8],
                    alpha_x8.mul(x_chunk).to_array(),
                );
                i += 8;
            }

            while i < n {
                *x_ptr.add(i) *= alpha;
                i += 1;
            }
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 {
                (n as isize - 1) * -incx
            } else {
                0
            };

            for _ in 0..n {
                *x_ptr.offset(ix) *= alpha;
                ix += incx;
            }
        }
    }
}

#[inline(always)]
pub fn copy(n: usize, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        let x_ptr = x.as_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            // Contiguous memory allows for a simple bulk copy
            std::ptr::copy_nonoverlapping(x_ptr, y_ptr, n);

            // let mut i = 0;
            // while i + 16 <= n {
            //     let x0 = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
            //     let x1 = read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]);
            //
            //     write_unaligned(y_ptr.add(i) as *mut [f32; 8], x0);
            //     write_unaligned(y_ptr.add(i + 8) as *mut [f32; 8], x1);
            //
            //     i += 16;
            // }
            //
            // while i + 8 <= n {
            //     let x_chunk = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
            //     write_unaligned(y_ptr.add(i) as *mut [f32; 8], x_chunk);
            //     i += 8;
            // }
            //
            // while i < n {
            //     *y_ptr.add(i) = *x_ptr.add(i);
            //     i += 1;
            // }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 {
                (n as isize - 1) * -incx
            } else {
                0
            };
            let mut iy = if incy < 0 {
                (n as isize - 1) * -incy
            } else {
                0
            };

            for _ in 0..n {
                *y_ptr.offset(iy) = *x_ptr.offset(ix);
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[inline(always)]
pub fn swap(n: usize, x: &mut [f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    unsafe {
        let x_ptr = x.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();

        if incx == 1 && incy == 1 {
            let mut i = 0;

            // let x_addr = x_ptr as usize;
            // let y_addr = y_ptr as usize;
            // let byte_len = n * size_of::<f32>();
            //
            // if x_addr == y_addr {
            // } else if x_addr + byte_len <= y_addr || y_addr + byte_len <= x_addr {
            //     std::ptr::swap_nonoverlapping(x_ptr, y_ptr, n);
            // } else {
            //     for i in 0..n {
            //         std::ptr::swap(x_ptr.add(i), y_ptr.add(i));
            //     }
            // }

            while i + 16 <= n {
                let x0 = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let y0 = read_unaligned(y_ptr.add(i) as *const [f32; 8]);
                let x1 = read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]);
                let y1 = read_unaligned(y_ptr.add(i + 8) as *const [f32; 8]);

                write_unaligned(y_ptr.add(i) as *mut [f32; 8], x0);
                write_unaligned(x_ptr.add(i) as *mut [f32; 8], y0);
                write_unaligned(y_ptr.add(i + 8) as *mut [f32; 8], x1);
                write_unaligned(x_ptr.add(i + 8) as *mut [f32; 8], y1);

                i += 16;
            }

            while i + 8 <= n {
                let x_chunk = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let y_chunk = read_unaligned(y_ptr.add(i) as *const [f32; 8]);
                write_unaligned(y_ptr.add(i) as *mut [f32; 8], x_chunk);
                write_unaligned(x_ptr.add(i) as *mut [f32; 8], y_chunk);
                i += 8;
            }

            while i < n {
                let tmp = *x_ptr.add(i);
                *x_ptr.add(i) = *y_ptr.add(i);
                *y_ptr.add(i) = tmp;
                i += 1;
            }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 {
                (n as isize - 1) * -incx
            } else {
                0
            };
            let mut iy = if incy < 0 {
                (n as isize - 1) * -incy
            } else {
                0
            };

            for _ in 0..n {
                let tmp = *x_ptr.offset(ix);
                *x_ptr.offset(ix) = *y_ptr.offset(iy);
                *y_ptr.offset(iy) = tmp;
                ix += incx;
                iy += incy;
            }
        }
    }
}

#[test]
fn test_axpy() {
    use crate::utils::gen_fill;
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0.0; 8];

    // Test with positive inc
    axpy(8, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    y.fill(0.0);

    // Test with negative inc
    axpy(8, 2.0, &x, 1, &mut y, -1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);
    axpy(8, 2.0, &x, -1, &mut y, 1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    // Test with non-unit inc
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
        axpy(gen_x.len(), 8.0, &gen_x, 1, &mut gen_y, 1);
    }
    let elapsed = start.elapsed().as_secs_f64();

    let gflops = (2.0 * gen_x.len() as f64 * 1024.0) / (elapsed * 1e9);

    println!(
        "Elapsed time: {:.6} seconds, GFLOPS: {:.2}",
        elapsed, gflops
    );
}

#[test]
fn test_scal() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 2.0, &mut x, 1);
    assert_eq!(x, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 1.0, &mut x, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(5, -1.0, &mut x, -1);
    assert_eq!(x, vec![-1.0, -2.0, -3.0, -4.0, -5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(4, 3.0, &mut x, 2);
    assert_eq!(x, vec![3.0, 2.0, 9.0, 4.0, 15.0, 6.0, 21.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(0, 2.0, &mut x, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 0.0, &mut x, 1);
    assert_eq!(x, vec![0.0; 8]);
}

#[test]
fn test_copy() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut y = vec![0.0; 8];
    copy(8, &x, 1, &mut y, 1);
    assert_eq!(y, x);

    let mut y = vec![0.0; 8];
    copy(8, &x, -1, &mut y, 1);
    assert_eq!(y, vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, 2, &mut y, 2);
    assert_eq!(y, vec![1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, -2, &mut y, 2);
    assert_eq!(y, vec![7.0, 0.0, 5.0, 0.0, 3.0, 0.0, 1.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(4, &x, 2, &mut y, -1);
    assert_eq!(y, vec![7.0, 5.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(3, &x, -1, &mut y, -1);
    assert_eq!(y, vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let mut y = vec![0.0; 8];
    copy(0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
}

#[test]
fn test_swap() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    swap(8, &mut x, 1, &mut y, 1);
    assert_eq!(x, vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(y, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    swap(4, &mut x, 2, &mut y, 2);
    assert_eq!(x, vec![9.0, 2.0, 11.0, 4.0, 13.0, 6.0, 15.0, 8.0]);
    assert_eq!(y, vec![1.0, 10.0, 3.0, 12.0, 5.0, 14.0, 7.0, 16.0]);

    x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    swap(0, &mut x, 1, &mut y, 1);
    assert_eq!(x, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y, vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
}
