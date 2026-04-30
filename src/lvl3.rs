use crate::lvl1::{axpy, dot, scal};
use std::slice::{from_raw_parts, from_raw_parts_mut};

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn gemm(
    m: usize,      // rows of C (and A when not transposed)
    n: usize,      // cols of C (and B when not transposed)
    k: usize,      // inner dimension (cols of A when not transposed, rows of B when not transposed)
    alpha: f32,    // scaling for product
    a: &[f32],     // matrix A: m×k when is_trans_a=false, k×m when is_trans_a=true (column major)
    lda: usize,    // leading dim of A (column major storage)
    b: &[f32],     // matrix B: k×n when is_trans_b=false, n×k when is_trans_b=true (column major)
    ldb: usize,    // leading dim of B (column major storage)
    beta: f32,     // result scaling
    c: &mut [f32], // resultant mat C: m×n (column major)
    ldc: usize,    // leading dim of C (column major storage)
    is_trans_a: bool,
    is_trans_b: bool,
) {
    // README: Go through `gemv` implementation in lvl2.rs before reading this

    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let a_rows = if is_trans_a { k } else { m };
    let a_cols = if is_trans_a { m } else { k };

    let b_rows = if is_trans_b { n } else { k };
    let b_cols = if is_trans_b { k } else { n };

    if lda == 0 || lda < a_rows {
        panic!("lda must be >= rows of stored A and non-zero");
    }

    if ldb == 0 || ldb < b_rows {
        panic!("ldb must be >= rows of stored B and non-zero");
    }

    if ldc == 0 || ldc < m {
        panic!("ldc must be >= rows of C and non-zero");
    }

    if a.len() < (a_cols - 1) * lda + a_rows {
        panic!("Matrix A buffer too small");
    }
    if b.len() < (b_cols - 1) * ldb + b_rows {
        panic!("Matrix B buffer too small");
    }
    if c.len() < (n - 1) * ldc + m {
        panic!("Matrix C buffer too small");
    }

    // Properly scale C by beta
    for j in 0..n {
        let start = j * ldc;
        let col = unsafe { from_raw_parts_mut(c.as_mut_ptr().add(start), m) };
        scal(m, beta, col, 1);
    }

    if alpha == 0.0 {
        return;
    }

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    // Number of columns of B to process in one block
    let block_n = { (n / 4).clamp(1, 16_348) };
    // Number of rows of A to process in one block
    let block_m = { (m / 4).clamp(1, 16_348) };
    // Number of elements in the inner dimension to process in one block,
    let block_k = { (k / 4).clamp(1, 16_348) };

    match (is_trans_a, is_trans_b) {
        // C = alpha * A * B + beta * C
        // C = alpha * A * B^T + beta * C (B transposed)
        // we need JUST a scalar from b, either B[i][j] or B[j][i] and use do OUTER PRODUCT MATMUL
        // while A & C are column contiguous, so using axpy is the right choice here
        (false, false) | (false, true) => {
            // iter over block of col of B and C, then inner dim and finally block of row of A and C
            for j_b in (0..n).step_by(block_n) {
                let j_max = (j_b + block_n).min(n); // <- max col of B and C to process in this block
                for k_b in (0..k).step_by(block_k) {
                    let k_max = (k_b + block_k).min(k); // shared dim, max col of A and row of B to process in this block
                    for i_b in (0..m).step_by(block_m) {
                        let i_max = (i_b + block_m).min(m); // <- max row of A and C to process in this block
                        // num of element in col of A and C to process in this block
                        let curr_e = i_max - i_b;
                        // iter over col of B and C in block,
                        // for each col, get col_buf of C to update in INNER loop (cached!!)
                        for j in j_b..j_max {
                            // MATRIX C: row (i_b..i_max) + col (j) * ldc
                            let c_idx = i_b + (j * ldc); // <- column index, c0, c1, ... c(n-1)
                            let c_col_ptr = unsafe { c_ptr.add(c_idx) };
                            // get mut col_buf by reading `curr_e` i.e no of current element in that rows
                            let c_col_buf = unsafe { from_raw_parts_mut(c_col_ptr, curr_e) };

                            // iter over shared dim in block, for each col in C
                            // for each element, scale by alpha and perform `axpy` on col of A and C
                            for kk in k_b..k_max {
                                // grab each element in this col and scale
                                let b_idx = if is_trans_b {
                                    j + (kk * ldb) // element B[j][k] in n×k storage (transposed)
                                } else {
                                    kk + (j * ldb) // element B[k][j] in k×n storage (not transposed)
                                };
                                let b_val = unsafe { *b_ptr.add(b_idx) };

                                // grab an entire col buf, using k since we are moving down in THAT BLOCK
                                let a_idx = i_b + (kk * lda); // points to starting A[i_b][k], stride by `lda` since we are moving down in col of A
                                let a_col_ptr = unsafe { a_ptr.add(a_idx) };
                                let a_col_buf = unsafe { from_raw_parts(a_col_ptr, curr_e) }; // <-- MATRIX A: row (i_b..i_max) + col (k) * lda

                                // `axpy` C_col = C_col + (scaled_b * A_col)
                                axpy(curr_e, alpha * b_val, a_col_buf, 1, c_col_buf, 1); // <- inc* are 1, since cols of A & C are contiguous no matter
                            }
                        }
                    }
                }
            }
        }
        // C = alpha * A^T * B + beta * C (A transposed)
        // compute dot between a row of A^T (contiguous) and a column of B, (INNER PRODUCT)
        (true, false) => {
            for j_b in (0..n).step_by(block_n) {
                let j_max = (j_b + block_n).min(n);
                for k_b in (0..k).step_by(block_k) {
                    let k_max = (k_b + block_k).min(k);
                    for i_b in (0..m).step_by(block_m) {
                        let i_max = (i_b + block_m).min(m);
                        let curr_r = i_max - i_b;

                        // iter over all column of B & C in block
                        for j in j_b..j_max {
                            // C [i_b][j], since we are moving down in col of C, and col is j, so row is i_b
                            let c_idx = i_b + (j * ldc); // <- column index, c0, c1, ... c(n-1)
                            let c_col_ptr = unsafe { c_ptr.add(c_idx) };

                            // for each row r in i_b block,
                            // we compute partial dot over shared dim (k_b...k_max), and update c[r][j]
                            // compute buf len is same for both buf, since its inner dim, and for dot
                            let buf_len = k_max - k_b;
                            for r_off in 0..curr_r {
                                // absolute row index = start row index of the block + local offset (0...curr_r)
                                let r = i_b + r_off;

                                // get a row from A in this BLOCK, since A is transposed, we are moving down in row,
                                // reading ahead from starting index of col i.e. first row index, gives us entire row_buf in current BLOCK
                                let a_idx = k_b + (r * lda); // <- points to starting element of col
                                let a_row_ptr = unsafe { a_ptr.add(a_idx) }; // pointer to the start of row r within this k-block for A^T
                                let a_row_buf = unsafe { from_raw_parts(a_row_ptr, buf_len) };

                                // get stored column r of A, which is logical row r of A^T
                                let b_idx = k_b + (j * ldb); // element B[k][j] in k×n storage (not transposed)
                                let b_col_ptr = unsafe { b_ptr.add(b_idx) }; // pointer to the start of column j within this k-block for B
                                let b_col_buf = unsafe { from_raw_parts(b_col_ptr, buf_len) };

                                // compute & apply, using `mul_add` FMA
                                let partial = dot(buf_len, a_row_buf, 1, b_col_buf, 1);
                                unsafe {
                                    *c_col_ptr.add(r_off) =
                                        alpha.mul_add(partial, *c_col_ptr.add(r_off)); // use rdx because c_col_ptr already points at row i_b
                                }
                            }
                        }
                    }
                }
            }
        }
        // C = alpha * A^T * B^T + beta * C (both transposed) (OUTER PRODUCT)
        // well, axpy can use here with stride, using col of A^T with stride, and a scalar element from B (INNER PRODUCT)
        (true, true) => {
            for j_b in (0..n).step_by(block_n) {
                let j_max = (j_b + block_n).min(n);
                for k_b in (0..k).step_by(block_k) {
                    let k_max = (k_b + block_k).min(k);
                    for i_b in (0..m).step_by(block_m) {
                        let i_max = (i_b + block_m).min(m);
                        let curr_r = i_max - i_b;

                        for j in j_b..j_max {
                            let c_idx = i_b + j * ldc;
                            let c_col_ptr = unsafe { c_ptr.add(c_idx) };

                            for rdx in 0..curr_r {
                                let i = i_b + rdx;
                                let buf_len = k_max - k_b;

                                // Row i of A^T = column i of stored A (contiguous)
                                let a_idx = k_b + i * lda;
                                let a_row_buf =
                                    unsafe { from_raw_parts(a_ptr.add(a_idx), buf_len) };

                                // Column j of B^T = row j of stored B (strided by ldb)
                                let b_start = j + k_b * ldb;
                                let b_slice = unsafe {
                                    from_raw_parts(b_ptr.add(b_start), (buf_len - 1) * ldb + 1)
                                };

                                let partial = dot(buf_len, a_row_buf, 1, b_slice, ldb as i32);
                                unsafe {
                                    *c_col_ptr.add(rdx) =
                                        alpha.mul_add(partial, *c_col_ptr.add(rdx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gemm_test() {
    use crate::utils::gen_fill;
    use std::hint::black_box;
    use std::time::Instant;

    let size = 1024;
    let runs = 12;
    let warmup = 6;

    let mut a = vec![0.0f32; size * size];
    let mut b = vec![0.0f32; size * size];
    let mut c = vec![0.0f32; size * size];

    black_box(&mut a);
    black_box(&mut b);
    black_box(&mut c);

    gen_fill(&mut a);
    gen_fill(&mut b);

    // correctness check, by comparing with `gemm_checker`, which is a straightforward implementation of gemm without blocking, and definitely not optimized
    {
        let mut c1 = vec![0.0f32; size * size];
        let mut c2 = vec![0.0f32; size * size];

        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c1, size, false, false,
        );

        gemm_checker(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c2, size, false, false,
        );

        assert!(
            c1.iter()
                .zip(c2.iter())
                .all(|(&x, &y)| (x - y).abs() < 1e-3)
        );
    }

    for _ in 0..warmup {
        c.fill(0.0);
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, false,
        );
    }

    let start = Instant::now();
    for _ in 0..runs {
        c.fill(0.0);
        gemm(
            size, size, size, 4.0, &a, size, &b, size, 2.0, &mut c, size, false, false,
        );
    }
    let elapsed = start.elapsed();

    let gflops = 2.0 * size.pow(3) as f64 * runs as f64 / elapsed.as_secs_f64() / 1e9;

    println!(
        "gemm_perf: {}x{}, {} runs, {:.3}s, {:.2} GFLOPS",
        size,
        size,
        runs,
        elapsed.as_secs_f64(),
        gflops
    );
}

#[allow(clippy::too_many_arguments)]
pub fn gemm_checker(
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
    assert!(c.len() >= ldc * n);

    // Scale C first by beta
    for col in 0..n {
        for row in 0..m {
            c[row + col * ldc] *= beta;
        }
    }

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;

            for p in 0..k {
                let a_ip = if !is_trans_a {
                    // A(i, p), A is (m x k) when not transposed
                    a[i + p * lda]
                } else {
                    // A^T(i, p) = A(p, i), A is (k x m) when transposed
                    a[p + i * lda]
                };

                let b_pj = if !is_trans_b {
                    // B(p, j), B is (k x n) when not transposed
                    b[p + j * ldb]
                } else {
                    // B^T(p, j) = B(j, p), B is (n x k) when transposed
                    b[j + p * ldb]
                };

                sum += a_ip * b_pj;
            }

            c[i + j * ldc] += alpha * sum;
        }
    }
}
