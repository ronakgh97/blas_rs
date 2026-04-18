use crate::utils::from_m256;
use std::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
#[allow(unused)]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_hadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
    _mm256_permute_ps, _mm256_setzero_ps, _mm256_storeu_ps,
};
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

        // Handle 4 AVX registers at a time
        unsafe {
            let alpha_x8 = core::arch::x86_64::_mm256_set1_ps(alpha);
            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                let r0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                let r1 = _mm256_fmadd_ps(alpha_x8, x1, y1);
                let r2 = _mm256_fmadd_ps(alpha_x8, x2, y2);
                let r3 = _mm256_fmadd_ps(alpha_x8, x3, y3);

                _mm256_storeu_ps(y_ptr.add(i), r0);
                _mm256_storeu_ps(y_ptr.add(i + 8), r1);
                _mm256_storeu_ps(y_ptr.add(i + 16), r2);
                _mm256_storeu_ps(y_ptr.add(i + 24), r3);

                i += 32;

                if i + 32 < n {
                    _mm_prefetch(x_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                    _mm_prefetch(y_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                }
            }

            // Handle one AVX register at a time.
            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let res0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                _mm256_storeu_ps(y_ptr.add(i), res0);
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

        unsafe {
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
                //     *x_ptr.add(i) = 0.0  ;
                // }

                return;
            }

            let alpha_x8 = core::arch::x86_64::_mm256_set1_ps(alpha);
            let mut i = 0;

            while i + 16 <= n {
                let mut v0 = _mm256_loadu_ps(x_ptr.add(i));
                let mut v1 = _mm256_loadu_ps(x_ptr.add(i + 8));

                v0 = _mm256_mul_ps(alpha_x8, v0);
                v1 = _mm256_mul_ps(alpha_x8, v1);

                _mm256_storeu_ps(x_ptr.add(i), v0);
                _mm256_storeu_ps(x_ptr.add(i + 8), v1);

                i += 16;

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 32) as *const i8, _MM_HINT_NTA);
                }
            }

            while i + 8 <= n {
                let v = _mm256_loadu_ps(x_ptr.add(i));
                let res = _mm256_mul_ps(alpha_x8, v);
                _mm256_storeu_ps(x_ptr.add(i), res);
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
        unsafe {
            let mut sum0 = _mm256_setzero_ps();
            let mut sum1 = _mm256_setzero_ps();
            let mut sum2 = _mm256_setzero_ps();
            let mut sum3 = _mm256_setzero_ps();
            let mut i = 0;

            while i + 32 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                sum0 = _mm256_fmadd_ps(x0, y0, sum0);
                sum1 = _mm256_fmadd_ps(x1, y1, sum1);
                sum2 = _mm256_fmadd_ps(x2, y2, sum2);
                sum3 = _mm256_fmadd_ps(x3, y3, sum3);

                i += 32;
            }

            while i + 8 <= n {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let y = _mm256_loadu_ps(y_ptr.add(i));
                sum0 = _mm256_fmadd_ps(x, y, sum0);
                i += 8;
            }

            let sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

            let mut result = from_m256(sum);
            while i < n {
                result += x[i] * y[i];
                i += 1;
            }

            result
        }
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

    if incx == 0 {
        panic!("Increment value must be non-zero");
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
        let mut sum = unsafe { _mm256_setzero_ps() };

        let x_ptr = x.as_ptr();

        while i + 32 <= n {
            unsafe {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                sum = _mm256_fmadd_ps(x0, x0, sum);
                sum = _mm256_fmadd_ps(x1, x1, sum);
                sum = _mm256_fmadd_ps(x2, x2, sum);
                sum = _mm256_fmadd_ps(x3, x3, sum);

                i += 32;
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                sum = _mm256_fmadd_ps(x, x, sum);
                i += 8;
            }
        }

        let mut result = from_m256(sum);
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
