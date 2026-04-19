use crate::utils::from_m256;
#[allow(unused)]
use std::arch::x86_64::{
    __m256i, _CMP_GT_OQ, _MM_HINT_NTA, _mm_prefetch, _mm256_add_epi32, _mm256_add_ps,
    _mm256_and_ps, _mm256_blendv_epi8, _mm256_blendv_ps, _mm256_castps_si256, _mm256_castsi256_ps,
    _mm256_cmp_ps, _mm256_fmadd_ps, _mm256_hadd_ps, _mm256_load_ps, _mm256_loadu_ps, _mm256_mul_ps,
    _mm256_permute_ps, _mm256_set_epi32, _mm256_set1_epi32, _mm256_set1_ps, _mm256_setzero_ps,
    _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_storeu_ps, _mm256_storeu_si256,
};

// TODO: x[ix], x[ix + incx], x[ix + 2*incx], ..., x[ix + (n-1)*incx]
// TODO: Need to handle to overflows for f32, using scale^2 * ( (x1/scale)^2 + (x1/scale)^2 + ... )

//#[inline(always)]
#[inline(never)]
/// Performs the AXPY operation, Y = alpha * X + Y, [ref](https://www.netlib.org/lapack/explore-html/d5/d4b/group__axpy.html)
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
            let alpha_x8 = _mm256_set1_ps(alpha);
            while i + 32 <= n {
                // Load from x
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Load from y
                let y0 = _mm256_loadu_ps(y_ptr.add(i));
                let y1 = _mm256_loadu_ps(y_ptr.add(i + 8));
                let y2 = _mm256_loadu_ps(y_ptr.add(i + 16));
                let y3 = _mm256_loadu_ps(y_ptr.add(i + 24));

                // FMA, Y += alpha * X
                let r0 = _mm256_fmadd_ps(alpha_x8, x0, y0);
                let r1 = _mm256_fmadd_ps(alpha_x8, x1, y1);
                let r2 = _mm256_fmadd_ps(alpha_x8, x2, y2);
                let r3 = _mm256_fmadd_ps(alpha_x8, x3, y3);

                // Store results back to y
                _mm256_storeu_ps(y_ptr.add(i), r0);
                _mm256_storeu_ps(y_ptr.add(i + 8), r1);
                _mm256_storeu_ps(y_ptr.add(i + 16), r2);
                _mm256_storeu_ps(y_ptr.add(i + 24), r3);

                i += 32;

                // I don't know, how to rightfully use this
                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                    _mm_prefetch(y_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
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

//#[inline(always)]
#[inline(never)]
/// Performs the SCAL operation, X = alpha * X, [ref](https://www.netlib.org/lapack/explore-html/d2/de8/group__scal.html)
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

            let alpha_x8 = _mm256_set1_ps(alpha);
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

//#[inline(always)]
#[inline(never)]
/// Performs the COPY operation, Y = X, [ref](https://www.netlib.org/lapack/explore-html/d5/d2b/group__copy.html)
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

//#[inline(always)]
#[inline(never)]
/// Performs the SWAP operation, X <-> Y, [ref](https://www.netlib.org/lapack/explore-html/d7/d51/group__swap.html)
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

//#[inline(always)]
#[inline(never)]
/// Performs the DOT operation, returns X^T * Y, [ref](https://www.netlib.org/lapack/explore-html/d1/dcc/group__dot.html)
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

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut i = 0usize;

        // Have two accumulators to allow for some instruction-level parallelism
        while i + 1 < n {
            unsafe {
                sum0 += *x_ptr.offset(ix) * *y_ptr.offset(iy);
                ix += incx;
                iy += incy;
                sum1 += *x_ptr.offset(ix) * *y_ptr.offset(iy);
                ix += incx;
                iy += incy;
            }
            i += 2;
        }

        // Handle the last element if n is odd
        if i < n {
            unsafe {
                sum0 += *x_ptr.offset(ix) * *y_ptr.offset(iy);
            }
        }

        sum0 + sum1
    }
}

//#[inline(always)]
#[inline(never)]
/// Performs the NRM2 operation, returns ||X||_2, [ref](https://www.netlib.org/lapack/explore-html/d1/d2a/group__nrm2.html)
pub fn nrm2(n: usize, x: &[f32], incx: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
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

//#[inline(always)]
#[inline(never)]
/// Performs the ASUM operation, returns sum of absolute values of elements in X, [ref](https://www.netlib.org/lapack/explore-html/d5/d72/group__asum.html)
pub fn asum(n: usize, x: &[f32], incx: i32) -> f32 {
    if n == 0 {
        return 0.0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if incx == 1 {
        let mut i = 0;
        let x_ptr = x.as_ptr();
        let mut sum0 = unsafe { _mm256_setzero_ps() };
        let mut sum1 = unsafe { _mm256_setzero_ps() };
        let mut sum2 = unsafe { _mm256_setzero_ps() };
        let mut sum3 = unsafe { _mm256_setzero_ps() };

        // Mask to clear the sign bit, effectively computing absolute value: [0x7fffffff, 0x7fffffff, ...8 times]
        let mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)) };

        while i + 32 <= n {
            unsafe {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Compute absolute values using AND with mask
                let abs_x0 = _mm256_and_ps(x0, mask);
                let abs_x1 = _mm256_and_ps(x1, mask);
                let abs_x2 = _mm256_and_ps(x2, mask);
                let abs_x3 = _mm256_and_ps(x3, mask);

                sum0 = _mm256_add_ps(sum0, abs_x0);
                sum1 = _mm256_add_ps(sum1, abs_x1);
                sum2 = _mm256_add_ps(sum2, abs_x2);
                sum3 = _mm256_add_ps(sum3, abs_x3);

                i += 32;

                if i + 64 < n {
                    _mm_prefetch(x_ptr.add(i + 64) as *const i8, _MM_HINT_NTA);
                }
            }
        }

        while i + 8 <= n {
            unsafe {
                let x = _mm256_loadu_ps(x_ptr.add(i));
                let abs_x = _mm256_and_ps(x, _mm256_set1_ps(f32::from_bits(0x7FFFFFFF)));
                sum0 = _mm256_add_ps(sum0, abs_x);
                i += 8;
            }
        }

        let sum = unsafe { _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3)) };

        let mut result = from_m256(sum);
        while i < n {
            result += x[i].abs();
            i += 1;
        }
        result
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };

        let x_ptr = x.as_ptr();

        let mut sum = 0.0f32;

        for _ in 0..n {
            unsafe {
                sum += (*x_ptr.offset(ix)).abs();
                ix += incx;
            }
        }
        sum
    }
}

//#[inline(always)]
#[inline(never)]
/// Performs the IAMAX operation, returns the index of the element with the maximum absolute value in X, [ref](https://www.netlib.org/lapack/explore-html/dd/d52/group__iamax.html)
pub fn i_amax(n: usize, x: &[f32], incx: i32) -> usize {
    if n == 0 {
        return 0;
    }

    if incx == 0 {
        panic!("Increment value must be non-zero");
    }

    if x.len() < 1 + (n - 1) * incx.unsigned_abs() as usize {
        panic!("Length of x does not match expected size based on n and incx");
    }

    if incx == 1 {
        unsafe {
            let x_ptr = x.as_ptr();

            // Create mask [0x7fffffff, 0x7fffffff, ...8 times]
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

            let base_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            // Init max trackers to -1.0 so any valid absolute value overwrites it.
            let mut la_vals0 = _mm256_set1_ps(-1.0);
            let mut la_idxs0 = _mm256_setzero_si256();

            let mut la_vals1 = _mm256_set1_ps(-1.0);
            let mut la_idxs1 = _mm256_setzero_si256();

            let mut la_vals2 = _mm256_set1_ps(-1.0);
            let mut la_idxs2 = _mm256_setzero_si256();

            let mut la_vals3 = _mm256_set1_ps(-1.0);
            let mut la_idxs3 = _mm256_setzero_si256();

            let mut i = 0usize;

            while i + 32 <= n {
                // Load Values
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let x1 = _mm256_loadu_ps(x_ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(x_ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(x_ptr.add(i + 24));

                // Compute absolute values using AND with mask
                let xabs0 = _mm256_and_ps(x0, mask);
                let xabs1 = _mm256_and_ps(x1, mask);
                let xabs2 = _mm256_and_ps(x2, mask);
                let xabs3 = _mm256_and_ps(x3, mask);

                // Create index vectors for current chunks by adding 'i' to base index
                // Each contains [i, i+1, i+2, i+3, i+4, i+5, i+6, i+7]...
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32)); // TODO: Broadcast overhead
                let idx1 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 8) as i32));
                let idx2 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 16) as i32));
                let idx3 = _mm256_add_epi32(base_idx, _mm256_set1_epi32((i + 24) as i32));

                // Create comparison masks > current max for each lane
                // Either returns 0xFFFFFFFF (true) or 0x00000000 (false) for each lane
                let cmp0 = _mm256_cmp_ps(xabs0, la_vals0, _CMP_GT_OQ);
                let cmp1 = _mm256_cmp_ps(xabs1, la_vals1, _CMP_GT_OQ);
                let cmp2 = _mm256_cmp_ps(xabs2, la_vals2, _CMP_GT_OQ);
                let cmp3 = _mm256_cmp_ps(xabs3, la_vals3, _CMP_GT_OQ);

                // Blend values and indices using the mask,
                // take compute cmp mask for each lane, check against abs lane,
                // keep either the old max or update with new value and index
                // For example
                //  old val = [5.0, 3.0, 6.0, 2.0, 4.0, 1.0, 7.0, 0.5], new val = [4.0, 8.0, 5.0, 1.0, 3.0, 2.0, 6.0, 9.0]
                //  cmp mask = [0, 0xFFFFFFFF, 0, 0, 0, 0, 0, 0xFFFFFFFF]
                //  result = [5.0, 8.0, 6.0, 2.0, 4.0, 1.0, 7.0, 9.0]
                // Same goes for indices
                la_vals0 = _mm256_blendv_ps(la_vals0, xabs0, cmp0);
                la_idxs0 = _mm256_blendv_epi8(la_idxs0, idx0, _mm256_castps_si256(cmp0));

                la_vals1 = _mm256_blendv_ps(la_vals1, xabs1, cmp1);
                la_idxs1 = _mm256_blendv_epi8(la_idxs1, idx1, _mm256_castps_si256(cmp1));

                la_vals2 = _mm256_blendv_ps(la_vals2, xabs2, cmp2);
                la_idxs2 = _mm256_blendv_epi8(la_idxs2, idx2, _mm256_castps_si256(cmp2));

                la_vals3 = _mm256_blendv_ps(la_vals3, xabs3, cmp3);
                la_idxs3 = _mm256_blendv_epi8(la_idxs3, idx3, _mm256_castps_si256(cmp3));

                i += 32;
            }

            let cmp01 = _mm256_cmp_ps(la_vals1, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals1, cmp01);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs1, _mm256_castps_si256(cmp01));

            let cmp02 = _mm256_cmp_ps(la_vals2, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals2, cmp02);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs2, _mm256_castps_si256(cmp02));

            let cmp03 = _mm256_cmp_ps(la_vals3, la_vals0, _CMP_GT_OQ);
            la_vals0 = _mm256_blendv_ps(la_vals0, la_vals3, cmp03);
            la_idxs0 = _mm256_blendv_epi8(la_idxs0, la_idxs3, _mm256_castps_si256(cmp03));

            while i + 8 <= n {
                let x0 = _mm256_loadu_ps(x_ptr.add(i));
                let abs0 = _mm256_and_ps(x0, mask);
                let idx0 = _mm256_add_epi32(base_idx, _mm256_set1_epi32(i as i32));

                let cmp0 = _mm256_cmp_ps(abs0, la_vals0, _CMP_GT_OQ);
                la_vals0 = _mm256_blendv_ps(la_vals0, abs0, cmp0);
                la_idxs0 = _mm256_blendv_epi8(la_idxs0, idx0, _mm256_castps_si256(cmp0));

                i += 8;
            }

            let mut tmp_vals = [0.0f32; 8];
            let mut tmp_idxs = [0i32; 8];
            // Store the SIMD registers to temporary arrays for reduction
            _mm256_storeu_ps(tmp_vals.as_mut_ptr(), la_vals0);
            _mm256_storeu_si256(tmp_idxs.as_mut_ptr() as *mut __m256i, la_idxs0);

            let mut la_val = -1.0f32;
            let mut la_idx = 0usize;

            for j in 0..8 {
                // Check if the current value is greater than the max found so far, or if it's a tie,
                // check if the index is smaller (to ensure we return the first occurrence of the max value)
                if tmp_vals[j] > la_val
                    || (tmp_vals[j] == la_val && (tmp_idxs[j] as usize) < la_idx)
                {
                    la_val = tmp_vals[j];
                    la_idx = tmp_idxs[j] as usize;
                } // On tie, we ignore the new index since
                // we want the first occurrence (lower index) of the max value
            }

            while i < n {
                let val = (*x_ptr.add(i)).abs();
                if val > la_val || (val == la_val && i < la_idx) {
                    la_val = val;
                    la_idx = i;
                }
                i += 1;
            }

            la_idx
        }
    } else {
        let incx = incx as isize;
        let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
        let x_ptr = x.as_ptr();
        let mut la_idx: usize = 0;
        let mut la_val = unsafe { (*x_ptr.offset(ix)).abs() };
        for _ in 1..n {
            ix += incx;
            let val = unsafe { (*x_ptr.offset(ix)).abs() };
            if val > la_val {
                la_val = val;
                la_idx = ix as usize;
            }
        }
        la_idx
    }
}
