use crate::lvl1::{axpy, dot, scal};
use std::slice::from_raw_parts;

// TODO: the order of checks can be improved, if done smartly to avoid unnecessary compute
//#[inline(always)
#[inline(never)]
#[allow(clippy::too_many_arguments)]
/// The gemv routines compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a general matrix.
/// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemv.html)
pub fn gemv(
    m: usize,   // row of mat
    n: usize,   // col of mat
    alpha: f32, // scaling for product
    a: &[f32],  // matrix buf
    lda: usize, // leading dim of a row or col major depends, we follow `column major`
    x: &[f32],  // mul vector buf
    incx: i32,
    beta: f32,     // y scaling
    y: &mut [f32], // resultant buf
    incy: i32,
    is_trans: bool, // Fortran uses 'T', 'N', but we will use bool
) {
    // get the effective dimensions of x and y based on transposition
    let (x_len, y_len) = if is_trans { (m, n) } else { (n, m) };

    if incx == 0 || incy == 0 {
        panic!("incx and incy must be non-zero");
    }
    if lda == 0 || lda < m {
        panic!("lda must be >= m and non-zero");
    }

    if m == 0 || n == 0 {
        panic!("Matrix dimensions must be greater than zero");
    }

    // `(n - 1) * lda` is start of last col, since we are col major,
    // so we added m to get the last element of that col
    if a.len() < (n - 1) * lda + m {
        panic!("Matrix A is too short for the given dimensions and leading dimension");
    }

    // check inner dim
    if (x.len() < (1 + (x_len - 1) * incx.unsigned_abs() as usize))
        || (y.len() < (1 + (y_len - 1) * incy.unsigned_abs() as usize))
    {
        panic!("Vector x is too short for the given dimensions, increment and transposition");
    }

    // we use `scal` to handle the beta scaling and zeroing out y if beta is 0, as per BLAS spec
    scal(y_len, beta, y, incy);

    // y is already scaled with beta, we are done here, no need to do any mul computation
    // also returning here means `gemv` just became `scal` :0
    if alpha == 0.0 {
        return;
    }

    // TODO: as always we are memory-bound, need block caching here

    // We have taken `!trans` because, default is column major,
    // so for simd we need contigous memory, and this is fine place to use `axpy` from lvl1
    if !is_trans {
        let incx = incx as isize;
        unsafe {
            let mut ix = if incx < 0 { (1 - n as isize) * incx } else { 0 };
            // iter all col here
            for idx in 0..n {
                // Get base pointer, we wil move it down the column for each iteration,
                // using lda is the leading dimension (row stride for column major)
                let col_ptr = a.as_ptr().add(idx * lda); // <- e.g c1, c2, c3..n times
                let col_buf = from_raw_parts(col_ptr, m); // every col has m elements, and they are contiguous in memory, so we can create a slice buf for it
                // Partial result for this column,
                // res_buf <- alpha * col[0] * x[0] + ... + alpha * col[n] * x[n].
                let x_val = *x.as_ptr().offset(ix);
                axpy(m, alpha * x_val, col_buf, 1, y, incy); // <- we take incx as 1, because its column major, simd can be applied :9
                ix += incx;
            }
        }
    } else {
        // SYMMETRY!!!
        // ok now it transposed, same logic but opposite behavior, we have "row-major" and since its row are contiguous in memory,
        // we can apply simd, and this is good place to use `dot` from lvl1
        let incy = incy as isize;
        let mut iy = if incy < 0 {
            (1 - y_len as isize) * incy
        } else {
            0
        };

        // base pointer for y, we will offset it for each row
        let y_ptr = y.as_mut_ptr();

        unsafe {
            // iter over all rows
            for i in 0..n {
                // Get base pointer for the current row, we will move it across the row for each iteration,
                let col_ptr = a.as_ptr().add(i * lda);
                // Get buf from on that idx, we use `from_raw_parts` since it contiguous and look `n` ahead
                let col_buf = from_raw_parts(col_ptr, m);

                // use `dot` from lvl1
                let dot_val = dot(m, col_buf, 1, x, incx); // <- again incx is 1 here, same reason

                // Apply alpha and add to y
                *y_ptr.offset(iy) = alpha.mul_add(dot_val, *y_ptr.offset(iy));

                iy += incy;
            } // move to the next row
        }
    }
}
