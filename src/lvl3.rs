use crate::lvl2::gemv;

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

// TODO: Super experimental
#[allow(clippy::too_many_arguments)]
#[inline(always)]
///The gemm routines compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, with general matrices.
// [ref](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2025-2/gemm.html)
pub fn gemm_exp(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
    is_trans_a: bool,
    is_trans_b: bool,
) {
    use rayon::prelude::*;

    c.par_chunks_mut(ldc) // process column in parallel
        .enumerate()
        .take(n)
        .for_each(|(j, c_col)| {
            // get the col slice buf
            let b_col = if !is_trans_b { &b[j * ldb..] } else { &b[j..] };

            // get computed col from using gemv
            gemv(
                m,
                k,
                alpha,
                a,
                lda,
                b_col,
                if !is_trans_b { 1 } else { ldb as i32 },
                beta,
                c_col,
                1,          // every col of c is contiguous
                is_trans_a, // gemv handles this
            );
        });
}

#[test]
fn gemm_exp_test() {
    use crate::utils::gen_fill;
    use std::hint::black_box;

    let run = 4;
    let size = 8192;

    let mut a = vec![0.0f32; size * size];
    let mut b = vec![0.0f32; size * size];
    let mut c = vec![0.0f32; size * size];

    gen_fill(&mut a);
    gen_fill(&mut b);

    black_box((&a, &b, &mut c));

    let start = std::time::Instant::now();
    for _ in 0..run {
        gemm_exp(
            size, size, size, 6.0, &a, size, &b, size, 8.0, &mut c, size, false, false,
        );
    }
    let elapsed = start.elapsed();

    let gflops = 2.0 * (size as f64).powi(3) / 1e9;

    println!(
        "gemm_exp: size={}x{}, avg time={:.3} sec, GFLOPS={:.2}",
        size,
        size,
        elapsed.as_secs_f64() / run as f64,
        gflops / (elapsed.as_secs_f64() / run as f64)
    );
}
