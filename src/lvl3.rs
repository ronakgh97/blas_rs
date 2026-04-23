#[allow(unused)]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
///The gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemm.html)
pub fn gemm(
    m: usize,      // row of mat A
    n: usize,      // col of mat A
    k: usize,      // col of mat B
    alpha: f32,    // scaling for product
    a: &[f32],     // matrix A
    lda: usize,    // leading dim of A, row or col major, we follow `column major`
    b: &[f32],     // matrix B
    ldb: usize,    // leading dim of B `column major`
    beta: f32,     // result scaling
    c: &mut [f32], // resultant mat
    ldc: usize,    // leading dim of C `column major`
    is_trans_a: bool,
    is_trans_b: bool,
) {
    todo!("PTSD");
}
