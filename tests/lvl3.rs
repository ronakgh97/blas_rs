use blas_rs::lvl3::gemm;

#[test]
fn test_gemm_no_trans_basic() {
    let m = 3_usize;
    let n = 3_usize;
    let k = 3_usize;

    let a = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let b = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let mut c = vec![0.0; m * n];
    let mut c_expected = vec![0.0; m * n];

    gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m, false, false);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        0.0,
        &mut c_expected,
        m,
        false,
        false,
    );

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_trans_a() {
    let m = 3_usize;
    let n = 3_usize;
    let k = 2_usize;
    let lda = m;

    // Need A: (m-1)*lda + k = 2*3 + 2 = 8 elements for transposed A
    let a = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 0.0, 0.0];
    let b = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    let mut c = vec![0.0; m * n];
    let mut c_expected = vec![0.0; m * n];

    gemm(m, n, k, 1.0, &a, lda, &b, k, 0.0, &mut c, m, true, false);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        lda,
        &b,
        k,
        0.0,
        &mut c_expected,
        m,
        true,
        false,
    );

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_trans_b() {
    let m = 2_usize;
    let n = 3_usize;
    let k = 2_usize;
    let lda = m;
    let ldb = n; // Must be >= max(k,n) = max(2,3) = 3

    let a = vec![1.0, 4.0, 2.0, 5.0];
    let b = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0]; // n*ldb = 9 elements
    let mut c = vec![0.0; m * n];
    let mut c_expected = vec![0.0; m * n];

    gemm(m, n, k, 1.0, &a, lda, &b, ldb, 0.0, &mut c, m, false, true);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        lda,
        &b,
        ldb,
        0.0,
        &mut c_expected,
        m,
        false,
        true,
    );

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_trans_ab() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 2_usize;
    let lda = k;
    let ldb = n;

    let a = vec![1.0, 3.0, 2.0, 4.0];
    let b = vec![1.0, 3.0, 2.0, 4.0];
    let mut c = vec![0.0; m * n];
    let mut c_expected = vec![0.0; m * n];

    gemm(m, n, k, 1.0, &a, lda, &b, ldb, 0.0, &mut c, m, true, true);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        lda,
        &b,
        ldb,
        0.0,
        &mut c_expected,
        m,
        true,
        true,
    );

    for i in 0..(m * n) {
        assert!(
            (c[i] - c_expected[i]).abs() < 1e-5,
            "mismatch at {}: got {}, expected {}",
            i,
            c[i],
            c_expected[i]
        );
    }
}

#[test]
fn test_gemm_alpha_zero_beta_scale_only() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 3_usize;

    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut c = vec![10.0, 20.0, 30.0, 40.0];
    let mut c_expected = [10.0, 20.0, 30.0, 40.0];

    gemm(m, n, k, 0.0, &a, m, &b, k, 2.0, &mut c, m, false, false);

    for item in c_expected.iter_mut().take(m * n) {
        *item *= 2.0;
    }

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_beta_zero() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 2_usize;

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![100.0, 200.0, 300.0, 400.0];
    let mut c_expected = vec![100.0, 200.0, 300.0, 400.0];

    gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m, false, false);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        0.0,
        &mut c_expected,
        m,
        false,
        false,
    );

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_beta_scale_and_add() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 2_usize;

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![1.0, 1.0, 1.0, 1.0];
    let mut c_expected = vec![1.0, 1.0, 1.0, 1.0];

    gemm(m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m, false, false);
    gemm_checker(
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        1.0,
        &mut c_expected,
        m,
        false,
        false,
    );

    for i in 0..(m * n) {
        assert!((c[i] - c_expected[i]).abs() < 1e-5, "mismatch at {}", i);
    }
}

#[test]
fn test_gemm_zero_m() {
    let m = 0_usize;
    let n = 3_usize;
    let k = 2_usize;

    let a = vec![];
    let b = vec![];
    let mut c = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let c_orig = c.clone();

    gemm(m, n, k, 1.0, &a, m, &b, k, 2.0, &mut c, n, false, false);

    assert_eq!(c, c_orig);
}

#[test]
fn test_gemm_zero_n() {
    let m = 3_usize;
    let n = 0_usize;
    let k = 2_usize;

    let a = vec![];
    let b = vec![];
    let mut c = vec![1.0, 2.0, 3.0];
    let c_orig = c.clone();

    gemm(m, n, k, 1.0, &a, m, &b, k, 2.0, &mut c, n, false, false);

    assert_eq!(c, c_orig);
}

#[test]
fn test_gemm_zero_k() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 0_usize;

    let a = vec![];
    let b = vec![];
    let mut c = vec![1.0, 2.0, 3.0, 4.0];
    let c_orig = c.clone();

    gemm(m, n, k, 1.0, &a, m, &b, k, 2.0, &mut c, m, false, false);

    assert_eq!(c, c_orig);
}

#[test]
#[should_panic]
fn test_gemm_lda_zero_panic() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 2_usize;
    let a = vec![];
    let b = vec![];
    let mut c = vec![];

    gemm(m, n, k, 1.0, &a, 0, &b, k, 0.0, &mut c, m, false, false);
}

#[test]
#[should_panic]
fn test_gemm_ldb_zero_panic() {
    let m = 2_usize;
    let n = 2_usize;
    let k = 2_usize;
    let a = vec![];
    let b = vec![];
    let mut c = vec![];

    gemm(m, n, k, 1.0, &a, m, &b, 0, 0.0, &mut c, m, false, false);
}

#[test]
fn test_gemm_various_sizes() {
    for size in [2, 3, 4, 8, 16] {
        let m = size;
        let n = size;
        let k = size;

        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0; m * n];
        let mut c_expected = vec![0.0; m * n];

        gemm(m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m, false, false);
        gemm_checker(
            m,
            n,
            k,
            1.0,
            &a,
            m,
            &b,
            k,
            0.0,
            &mut c_expected,
            m,
            false,
            false,
        );

        for i in 0..(m * n) {
            assert!(
                (c[i] - c_expected[i]).abs() < 1e-4,
                "size {} mismatch at {}",
                size,
                i
            );
        }
    }
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
