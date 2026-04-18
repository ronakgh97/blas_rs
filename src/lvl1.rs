use crate::utils::from_f32x8;
#[allow(unused_imports)]
use std::ops::{Add, Mul};
use std::ptr::{read_unaligned, write_unaligned};
use wide::f32x8;

// x[ix], x[ix + incx], x[ix + 2*incx], ..., x[ix + (n-1)*incx]

// TODO: Need to handle to overflows for f32, using scale^2 * ( (x1/scale)^2 + (x1/scale)^2 + ... )

#[inline(always)]
pub fn axpy(n: usize, alpha: f32, x: &[f32], incx: i32, y: &mut [f32], incy: i32) {
    if n == 0 || alpha == 0.0 {
        return;
    }

    if incx == 0 || incy == 0 {
        panic!("Increment values must be non-zero");
    }

    // Bound checks
    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    // Bound checks
    if y.len() < 1 + (n - 1) * incy.unsigned_abs() as usize {
        panic!("Length of y does not match expected size based on n and incy");
    }

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_mut_ptr();

    if incx == 1 && incy == 1 {
        let mut i = 0;

        let alpha_x8 = f32x8::splat(alpha);

        // Process 16 elements at a time
        unsafe {
            while i + 16 <= n {
                let x0 = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let y0 = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                let x1 = f32x8::from(read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]));
                let y1 = f32x8::from(read_unaligned(y_ptr.add(i + 8) as *const [f32; 8]));

                // FMA
                let res0 = alpha_x8.mul_add(x0, y0);
                let res1 = alpha_x8.mul_add(x1, y1);

                write_unaligned(y_ptr.add(i) as *mut [f32; 8], res0.to_array());
                write_unaligned(y_ptr.add(i + 8) as *mut [f32; 8], res1.to_array());

                i += 16;

                if i + 32 > n {
                    core::arch::x86_64::_mm_prefetch(
                        x_ptr.add(i + 32) as *const i8,
                        core::arch::x86_64::_MM_HINT_NTA,
                    );
                    core::arch::x86_64::_mm_prefetch(
                        y_ptr.add(i + 32) as *const i8,
                        core::arch::x86_64::_MM_HINT_NTA,
                    );
                }
            }

            // Handle 8 elements if left after processing 16
            while i + 8 <= n {
                let x_chunk = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let y_chunk = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                let res = alpha_x8.mul_add(x_chunk, y_chunk);
                write_unaligned(y_ptr.add(i) as *mut [f32; 8], res.to_array());
                i += 8;
            }

            // Handle remaining elements
            while i < n {
                *y_ptr.add(i) += alpha * *x_ptr.add(i);
                i += 1;
            }
        }
    } else {
        let incx = incx as isize;
        let incy = incy as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

        unsafe {
            for _ in 0..n {
                // Y += alpha * X
                *y_ptr.offset(iy) += alpha * *x_ptr.offset(ix);
                ix += incx;
                iy += incy;

                let pfx = ix + 16 * incx;
                if pfx >= 0 && pfx < x.len() as isize {
                    core::arch::x86_64::_mm_prefetch(
                        x_ptr.offset(pfx) as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }

                let pfy = ix + 16 * incy;
                if pfy >= 0 && pfy < y.len() as isize {
                    core::arch::x86_64::_mm_prefetch(
                        y_ptr.offset(pfy) as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
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
                //     *x_ptr.add(i) = 0.0  ;
                // }

                return;
            }

            let alpha_x8 = core::arch::x86_64::_mm256_set1_ps(alpha);
            //let alpha_x8 = f32x8::splat(alpha);
            let mut i = 0;

            while i + 16 <= n {
                let mut v0 = core::arch::x86_64::_mm256_loadu_ps(x_ptr.add(i));
                let mut v1 = core::arch::x86_64::_mm256_loadu_ps(x_ptr.add(i + 8));

                v0 = core::arch::x86_64::_mm256_mul_ps(alpha_x8, v0);
                v1 = core::arch::x86_64::_mm256_mul_ps(alpha_x8, v1);

                core::arch::x86_64::_mm256_storeu_ps(x_ptr.add(i), v0);
                core::arch::x86_64::_mm256_storeu_ps(x_ptr.add(i + 8), v1);

                // let x0 = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                // let x1 = f32x8::from(read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]))
                //
                // write_unaligned(x_ptr.add(i) as *mut [f32; 8], alpha_x8.mul(x0).to_array());
                // write_unaligned(
                //     x_ptr.add(i + 8) as *mut [f32; 8],
                //     alpha_x8.mul(x1).to_array(),
                // );

                i += 16;

                if i + 64 < n {
                    core::arch::x86_64::_mm_prefetch(
                        x_ptr.add(i + 32) as *const i8,
                        core::arch::x86_64::_MM_HINT_NTA,
                    );
                }
            }

            while i + 8 <= n {
                // let x_chunk = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                // write_unaligned(
                //     x_ptr.add(i) as *mut [f32; 8],
                //     alpha_x8.mul(x_chunk).to_array(),
                // );

                let v = core::arch::x86_64::_mm256_loadu_ps(x_ptr.add(i));
                let res = core::arch::x86_64::_mm256_mul_ps(alpha_x8, v);
                core::arch::x86_64::_mm256_storeu_ps(x_ptr.add(i), res);
                i += 8;
            }

            while i < n {
                *x_ptr.add(i) *= alpha;
                i += 1;
            }
        } else {
            let incx = incx as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

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
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

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
            let x_addr = x_ptr as usize;
            let y_addr = y_ptr as usize;
            let byte_len = n * size_of::<f32>();

            if x_addr == y_addr {
            } else if x_addr + byte_len <= y_addr || y_addr + byte_len <= x_addr {
                std::ptr::swap_nonoverlapping(x_ptr, y_ptr, n);
            } else {
                for i in 0..n {
                    std::ptr::swap(x_ptr.add(i), y_ptr.add(i));
                }
            }
        } else {
            let incx = incx as isize;
            let incy = incy as isize;
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

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

#[inline(always)]
pub fn dot(n: usize, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    if n == 0 {
        return 0.0;
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

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    if incx == 1 && incy == 1 {
        let mut sum0 = f32x8::ZERO;
        let mut sum1 = f32x8::ZERO;
        let mut sum2 = f32x8::ZERO;
        let mut sum3 = f32x8::ZERO;
        let mut i = 0;

        unsafe {
            while i + 32 <= n {
                let x0 = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let x1 = f32x8::from(read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]));
                let x2 = f32x8::from(read_unaligned(x_ptr.add(i + 16) as *const [f32; 8]));
                let x3 = f32x8::from(read_unaligned(x_ptr.add(i + 24) as *const [f32; 8]));

                let y0 = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                let y1 = f32x8::from(read_unaligned(y_ptr.add(i + 8) as *const [f32; 8]));
                let y2 = f32x8::from(read_unaligned(y_ptr.add(i + 16) as *const [f32; 8]));
                let y3 = f32x8::from(read_unaligned(y_ptr.add(i + 24) as *const [f32; 8]));

                sum0 = x0.mul_add(y0, sum0);
                sum1 = x1.mul_add(y1, sum1);
                sum2 = x2.mul_add(y2, sum2);
                sum3 = x3.mul_add(y3, sum3);

                i += 32;
            }

            while i + 8 <= n {
                let x_chunk = f32x8::from(read_unaligned(x_ptr.add(i) as *const [f32; 8]));
                let y_chunk = f32x8::from(read_unaligned(y_ptr.add(i) as *const [f32; 8]));
                sum0 = x_chunk.mul_add(y_chunk, sum0);
                i += 8;
            }
        }

        let sum = sum0 + sum1 + sum2;

        let mut result = from_f32x8(sum);
        while i < n {
            result += x[i] * y[i];
            i += 1;
        }

        result
    } else {
        let incx = incx as isize;
        let incy = incy as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut iy = if incy < 0 { (1 - n as isize) * incy } else { 0 };

        let mut sum0 = 0.0;
        let mut sum1 = 0.0;
        for _ in (0..n).step_by(2) {
            unsafe {
                sum0 += *x_ptr.offset(ix) * *y_ptr.offset(iy);
                ix += incx;
                iy += incy;
                sum1 += *x_ptr.offset(ix) * *y_ptr.offset(iy);
                ix += incx;
                iy += incy;
            }
        }
        sum0 + sum1
    }
}

#[inline(always)]
pub fn nrm2(n: usize, x: &[f32], incx: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    let (start, end) = if incx >= 0 {
        (0, (n as isize - 1) * incx as isize)
    } else {
        ((1 - n as isize) * incx as isize, 0)
    };

    if start < 0 || end as usize >= x.len() {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if incx == 1 {
        let mut i = 0;
        let mut sum = f32x8::ZERO;

        let x_ptr = x.as_ptr();

        while i + 32 <= n {
            unsafe {
                let x0 = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let x1 = read_unaligned(x_ptr.add(i + 8) as *const [f32; 8]);
                let x2 = read_unaligned(x_ptr.add(i + 16) as *const [f32; 8]);
                let x3 = read_unaligned(x_ptr.add(i + 24) as *const [f32; 8]);

                let va = f32x8::from(x0);
                let vb = f32x8::from(x1);
                let vc = f32x8::from(x2);
                let vd = f32x8::from(x3);

                sum = va.mul_add(va, sum);
                sum = vb.mul_add(vb, sum);
                sum = vc.mul_add(vc, sum);
                sum = vd.mul_add(vd, sum);
                i += 32;
            }
        }

        while i + 8 <= n {
            unsafe {
                let x0 = read_unaligned(x_ptr.add(i) as *const [f32; 8]);
                let va = f32x8::from(x0);
                sum = va.mul_add(va, sum);
                i += 8;
            }
        }

        let mut result = from_f32x8(sum);
        while i < n {
            result += x[i] * x[i];
            i += 1;
        }

        result.sqrt()
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let mut sum: f32 = 0.0;
        let x_ptr = x.as_ptr();
        for _ in 0..n {
            unsafe {
                sum += *x_ptr.offset(ix) * *x_ptr.offset(ix);
                ix += incx;
            }
        }
        sum.sqrt()
    }
}
