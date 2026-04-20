use blas_rs::lvl1::*;
use blas_rs::utils::gen_fill;

#[test]
fn test_axpy() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let mut y = vec![0.0; 8];

    axpy(8, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    y.fill(0.0);

    axpy(8, 2.0, &x, 1, &mut y, -1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    axpy(8, 2.0, &x, -1, &mut y, 1);
    assert_eq!(y, vec![14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0]);
    y.fill(0.0);

    axpy(4, 2.0, &x, 2, &mut y, 2);
    assert_eq!(y, vec![0.0, 0.0, 4.0, 0.0, 8.0, 0.0, 12.0, 0.0]);
    y.fill(0.0);

    axpy(0, 2.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    axpy(8, 0.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0; 8]);
    y.fill(0.0);

    axpy(8, -1.0, &x, 1, &mut y, 1);
    assert_eq!(y, vec![0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]);
    y.fill(0.0);

    let mut gen_x = vec![0.0f32; 1024];
    let mut gen_y = vec![0.0f32; 1024];

    gen_fill(&mut gen_x);
    gen_fill(&mut gen_y);

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
#[should_panic]
fn test_axpy_incx_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    axpy(4, 2.0, &x, 0, &mut y, 1);
}

#[test]
#[should_panic]
fn test_axpy_incy_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    axpy(4, 2.0, &x, 1, &mut y, 0);
}

#[test]
fn test_axpy_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 4];
        axpy(4, 2.0, &x, 1, &mut y, 1);
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 2];
        axpy(4, 2.0, &x, 1, &mut y, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_axpy_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut y = vec![0.0; n.max(1)];
        let mut expected = x.clone();

        axpy(n, 2.0, &x, 1, &mut y, 1);

        for exp in expected.iter_mut().take(n) {
            *exp *= 2.0;
        }
        assert_eq!(y[..n], expected[..n], "failed at n={}", n);
    }
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

    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    scal(8, 0.5, &mut x, 1);
    assert_eq!(x, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
#[should_panic]
fn test_scal_incx_zero_panic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    scal(4, 2.0, &mut x, 0);
}

#[test]
fn test_scal_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let mut x = vec![1.0, 2.0];
        scal(4, 2.0, &mut x, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_scal_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let expected: Vec<f32> = x.iter().map(|v| v * 2.0).collect();

        scal(n, 2.0, &mut x, 1);

        assert_eq!(x[..n], expected[..n], "failed at n={}", n);
    }
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
#[should_panic]
fn test_copy_incx_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    copy(4, &x, 0, &mut y, 1);
}

#[test]
#[should_panic]
fn test_copy_incy_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    copy(4, &x, 1, &mut y, 0);
}

#[test]
fn test_copy_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        let mut y = vec![0.0; 4];
        copy(4, &x, 1, &mut y, 1);
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 2];
        copy(4, &x, 1, &mut y, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_copy_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let mut y = vec![0.0; n.max(1)];

        copy(n, &x, 1, &mut y, 1);

        assert_eq!(y[..n], x[..n], "failed at n={}", n);
    }
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

#[test]
#[should_panic]
fn test_swap_incx_zero_panic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    swap(4, &mut x, 0, &mut y, 1);
}

#[test]
#[should_panic]
fn test_swap_incy_zero_panic() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; 4];
    swap(4, &mut x, 1, &mut y, 0);
}

#[test]
fn test_swap_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![0.0; 4];
        swap(4, &mut x, 1, &mut y, 1);
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 2];
        swap(4, &mut x, 1, &mut y, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_swap_overlapping_memory() {
    let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (x_slice, y_slice) = data.split_at_mut(4);
    swap(4, x_slice, 1, y_slice, 1);
    assert_eq!(x_slice, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(y_slice, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_swap_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32] {
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let mut y: Vec<f32> = (0..n).map(|i| i as f32 + 100.0).collect();
        let orig_x = x.clone();
        let orig_y = y.clone();

        swap(n, &mut x, 1, &mut y, 1);

        assert_eq!(x[..n], orig_y[..n], "failed at n={}", n);
        assert_eq!(y[..n], orig_x[..n], "failed at n={}", n);
    }
}

#[test]
fn test_dot() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

    let res = dot(8, &x, 1, &y, 1);
    let exp = simple_dot(&x, &y);
    assert_eq!(res, exp);

    let res = dot(4, &x, 2, &y, 2);
    let exp = x[0] * y[0] + x[2] * y[2] + x[4] * y[4] + x[6] * y[6];
    assert_eq!(res, exp);

    let res = dot(0, &x, 1, &y, 1);
    assert_eq!(res, 0.0);

    let res = dot(4, &x, -1, &y, 1);
    let exp = x[3] * y[0] + x[2] * y[1] + x[1] * y[2] + x[0] * y[3];
    assert_eq!(res, exp);

    let res = dot(4, &x, 1, &y, -1);
    let exp = x[0] * y[3] + x[1] * y[2] + x[2] * y[1] + x[3] * y[0];
    assert_eq!(res, exp);
}

#[test]
#[should_panic]
fn test_dot_incx_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    dot(4, &x, 0, &y, 1);
}

#[test]
#[should_panic]
fn test_dot_incy_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    dot(4, &x, 1, &y, 0);
}

#[test]
fn test_dot_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        dot(4, &x, 1, &y, 1);
    });
    assert!(result.is_err());

    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0];
        dot(4, &x, 1, &y, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_dot_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let y: Vec<f32> = (0..n).map(|i| i as f32 + 10.0).collect();

        let res = dot(n, &x, 1, &y, 1);
        let exp = simple_dot(&x, &y);

        let rel_diff = if exp != 0.0 {
            (res - exp).abs() / exp.abs()
        } else {
            (res - exp).abs()
        };
        assert!(
            rel_diff < 1e-5,
            "failed at n={}: res={}, exp={}",
            n,
            res,
            exp
        );
    }
}

fn simple_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let common = a.len().min(b.len());
    for i in 0..common {
        sum += a[i] * b[i];
    }
    sum
}

#[test]
fn test_nrm2() {
    let x = vec![3.0, 4.0];
    let res = nrm2(2, &x, 1);
    assert_eq!(res, 5.0);

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = nrm2(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert_eq!(res, exp);

    let res = nrm2(0, &x, 1);
    assert_eq!(res, 0.0);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = nrm2(4, &x, 2);
    let exp = (1.0_f32.powi(2) + 2.0_f32.powi(2) + 3.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();
    assert_eq!(res, exp);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = nrm2(4, &x, -1);
    let exp = (4.0_f32.powi(2) + 3.0_f32.powi(2) + 2.0_f32.powi(2) + 1.0_f32.powi(2)).sqrt();
    assert_eq!(res, exp);
}

#[test]
fn test_nrm2_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        nrm2(4, &x, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_nrm2_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();

        let res = nrm2(n, &x, 1);
        let exp: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let rel_diff = if exp != 0.0 {
            (res - exp).abs() / exp.abs()
        } else {
            (res - exp).abs()
        };
        assert!(
            rel_diff < 1e-5,
            "failed at n={}: res={}, exp={}",
            n,
            res,
            exp
        );
    }
}

#[test]
fn test_nrm2_large_vectors() {
    for n in [1024, 2048, 4096] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

        let res = nrm2(n, &x, 1);
        let exp: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();

        let rel_diff = if exp != 0.0 {
            (res - exp).abs() / exp.abs()
        } else {
            (res - exp).abs()
        };
        assert!(rel_diff < 1e-5, "failed at n={}", n);
    }
}

#[test]
fn test_asum() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = asum(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v.abs()).sum();
    assert_eq!(res, exp);

    let x = vec![-1.0, -2.0, -3.0, -4.0];
    let res = asum(4, &x, 1);
    assert_eq!(res, 10.0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = asum(8, &x, 1);
    let exp: f32 = x.iter().map(|v| v.abs()).sum();
    assert_eq!(res, exp);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = asum(4, &x, 2);
    assert_eq!(res, 10.0);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = asum(4, &x, -1);
    assert_eq!(res, 10.0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = asum(0, &x, 1);
    assert_eq!(res, 0.0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = asum(4, &x, 1);
    assert_eq!(res, 10.0);
}

#[test]
#[should_panic]
fn test_asum_incx_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    asum(4, &x, 0);
}

#[test]
fn test_asum_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        asum(4, &x, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_asum_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32) / 2.0).collect();

        let res = asum(n, &x, 1);
        let exp: f32 = x.iter().map(|v| v.abs()).sum();

        let rel_diff = if exp != 0.0 {
            (res - exp).abs() / exp.abs()
        } else {
            (res - exp).abs()
        };
        assert!(
            rel_diff < 1e-5,
            "failed at n={}: res={}, exp={}",
            n,
            res,
            exp
        );
    }
}

#[test]
fn test_i_amax() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
    let res = i_amax(4, &x, 2);
    assert_eq!(res, 6);

    let x = vec![4.0, 3.0, 2.0, 1.0];
    let res = i_amax(4, &x, -1);
    assert_eq!(res, 0);

    let x = vec![1.0, 2.0, 3.0, 4.0];
    let res = i_amax(0, &x, 1);
    assert_eq!(res, 0);

    let x = vec![5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);
}

#[test]
#[should_panic]
fn test_i_amax_incx_zero_panic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    i_amax(4, &x, 0);
}

#[test]
fn test_i_amax_bounds_error() {
    let result = std::panic::catch_unwind(|| {
        let x = vec![1.0, 2.0];
        i_amax(4, &x, 1);
    });
    assert!(result.is_err());
}

#[test]
fn test_i_amax_various_sizes() {
    for n in [1, 7, 8, 9, 15, 16, 17, 24, 32, 1024] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();

        let res = i_amax(n, &x, 1);
        assert_eq!(res, n - 1, "failed at n={}", n);
    }
}

#[test]
fn test_i_amax_first_occurrence() {
    let x = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);
}

#[test]
fn test_i_amax_negative_values() {
    let x = vec![-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 0);

    let x = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);

    let x = vec![-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -8.0];
    let res = i_amax(8, &x, 1);
    assert_eq!(res, 7);
}
