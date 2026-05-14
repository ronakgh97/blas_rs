use blas_rs::lvl2::gemv;
use std::panic::catch_unwind;

#[test]
fn test_gemv_no_trans() {
    // A (2x3), column-major:
    // [1 2 3]
    // [4 5 6]
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0, 0.0];

    gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, false);
    assert_eq!(y, vec![6.0, 15.0]);

    let mut y = vec![10.0, 20.0];
    gemv(2, 3, 1.0, &a, 2, &x, 1, 1.0, &mut y, 1, false);
    assert_eq!(y, vec![16.0, 35.0]);
}

#[test]
fn test_gemv_trans_basic_non_square() {
    // A (2x3), column-major:
    // [1 2 3]
    // [4 5 6]
    // A^T * [1, 1] = [5, 7, 9]
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 1.0];
    let mut y = vec![0.0, 0.0, 0.0];

    gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, true);
    assert_eq!(y, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_gemv_alpha_zero_beta_scale_only() {
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![10.0, 20.0];
    gemv(2, 3, 0.0, &a, 2, &x, 1, 0.5, &mut y, 1, false);
    assert_eq!(y, vec![5.0, 10.0]);

    let x_t = vec![1.0, 1.0];
    let mut y_t = vec![2.0, 4.0, 6.0];
    gemv(2, 3, 0.0, &a, 2, &x_t, 1, 2.0, &mut y_t, 1, true);
    assert_eq!(y_t, vec![4.0, 8.0, 12.0]);
}

#[test]
fn test_gemv_with_lda_padding() {
    // m=2, n=3, lda=4 (2 valid + 2 padding each column)
    // columns: [1,4], [2,5], [3,6]
    let a = vec![1.0, 4.0, 99.0, 99.0, 2.0, 5.0, 99.0, 99.0, 3.0, 6.0];

    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 4, &x, 1, 0.0, &mut y, 1, false);
    assert_eq!(y, vec![6.0, 15.0]);

    let x_t = vec![1.0, 1.0];
    let mut y_t = vec![0.0, 0.0, 0.0];
    gemv(2, 3, 1.0, &a, 4, &x_t, 1, 0.0, &mut y_t, 1, true);
    assert_eq!(y_t, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_gemv_stride_cases() {
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

    // non-trans, positive strides (incx=2, incy=2)
    // effective x = [1,2,3], A*x = [14, 32]
    let x = vec![1.0, 99.0, 2.0, 99.0, 3.0];
    let mut y = vec![0.0, 99.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x, 2, 0.0, &mut y, 2, false);
    assert_eq!(y, vec![14.0, 99.0, 32.0]);

    // non-trans, negative incx
    // x indices visited: 2,1,0 -> [3,2,1]
    // A*x = [10, 28]
    let x_neg = vec![1.0, 2.0, 3.0];
    let mut y_neg = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x_neg, -1, 0.0, &mut y_neg, 1, false);
    assert_eq!(y_neg, vec![10.0, 28.0]);

    // non-trans, negative incy
    // y indices visited: 1,0, so result order is reversed in backing buffer
    let x_ones = vec![1.0, 1.0, 1.0];
    let mut y_rev = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x_ones, 1, 0.0, &mut y_rev, -1, false);
    assert_eq!(y_rev, vec![15.0, 6.0]);

    // trans, mixed strides (incx=-1, incy=2)
    // effective x = [2,1], A^T*x = [6,9,12]
    let x_t = vec![1.0, 2.0];
    let mut y_t = vec![0.0, 99.0, 0.0, 99.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x_t, -1, 0.0, &mut y_t, 2, true);
    assert_eq!(y_t, vec![6.0, 99.0, 9.0, 99.0, 12.0]);
}

#[test]
#[should_panic]
fn test_gemv_incx_zero_panic() {
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x, 0, 0.0, &mut y, 1, false);
}

#[test]
#[should_panic]
fn test_gemv_incy_zero_panic() {
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 0, false);
}

#[test]
#[should_panic]
fn test_gemv_bad_lda_panic() {
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let x = vec![1.0, 1.0, 1.0];
    let mut y = vec![0.0, 0.0];
    gemv(2, 3, 1.0, &a, 1, &x, 1, 0.0, &mut y, 1, false);
}

#[test]
#[should_panic]
fn test_gemv_zero_m_panic() {
    let a = vec![1.0];
    let x = vec![1.0];
    let mut y = vec![1.0];
    gemv(0, 1, 1.0, &a, 1, &x, 1, 1.0, &mut y, 1, false);
}

#[test]
#[should_panic]
fn test_gemv_zero_n_panic() {
    let a = vec![1.0];
    let x = vec![1.0];
    let mut y = vec![1.0];
    gemv(1, 0, 1.0, &a, 1, &x, 1, 1.0, &mut y, 1, false);
}

#[test]
fn test_gemv_bounds_error() {
    let result = catch_unwind(|| {
        let a = vec![1.0, 4.0, 2.0, 5.0]; // too short for m=2,n=3,lda=2
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0, 0.0];
        gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, false);
    });
    assert!(result.is_err());

    let result = catch_unwind(|| {
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 1.0]; // too short for non-trans n=3
        let mut y = vec![0.0, 0.0];
        gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, false);
    });
    assert!(result.is_err());

    let result = catch_unwind(|| {
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0, 1.0, 1.0];
        let mut y = vec![0.0]; // too short for non-trans m=2
        gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, false);
    });
    assert!(result.is_err());

    let result = catch_unwind(|| {
        let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let x = vec![1.0]; // too short for trans m=2
        let mut y = vec![0.0, 0.0, 0.0];
        gemv(2, 3, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1, true);
    });
    assert!(result.is_err());
}
