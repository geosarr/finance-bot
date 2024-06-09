#[cfg(test)]
mod test {
    use super::super::*;
    use ndarray::array;
    #[test]
    fn test_is_orthonormal() {
        let ortho = array![
            [1. / 2_f64.sqrt(), -1. / 2_f64.sqrt()],
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()]
        ];
        let not_ortho = array![[1., 1.], [1., -1.]];
        assert!(is_orthonormal(1e-6, &ortho));
        assert!(!is_orthonormal(1e-6, &not_ortho));
    }

    #[test]
    fn test_normalize() {
        let a = array![[1., 1.], [1., 1.]];
        let a_norm = array![
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()],
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()]
        ];
        assert_eq!(normalize(&a, 0), a_norm);
        let b = array![[1., 2.], [3., 4.]];
        let b_norm = array![
            [1. / 10_f64.sqrt(), 2. / 20_f64.sqrt()],
            [3. / 10_f64.sqrt(), 4. / 20_f64.sqrt()]
        ];
        assert_eq!(normalize(&b, 0), b_norm);

        let c = array![[5., 6., 2.], [7., 8., 1.], [9., 10., 0.]];
        let c_norm = array![
            [5. / 65_f64.sqrt(), 6. / 114_f64.sqrt(), 2. / 181_f64.sqrt()],
            [7. / 65_f64.sqrt(), 8. / 114_f64.sqrt(), 1. / 181_f64.sqrt()],
            [9. / 65_f64.sqrt(), 10. / 114_f64.sqrt(), 0.],
        ];
        assert_eq!(normalize(&c, 1), c_norm);
    }

    #[test]
    fn test_calc_metric() {
        let eps: f64 = 1e-6;
        let stiefel = array![[1., 1.], [1., 1.], [1., 1.], [1., 1.]];
        let beta = array![[1.], [0.]];
        assert_eq!(-1.0, calc_metric(eps, &stiefel, &beta, &stiefel, &stiefel));
        let stiefel = array![
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()],
            [1. / 2_f64.sqrt(), -1. / 2_f64.sqrt()],
        ];
        let beta = array![[1.], [0.]];
        let x_train = array![
            [-0.0192, -0.0083],
            [-0.0130, -0.0222],
            [-0.0083, -0.0084],
            [0.0222, -0.0136],
            [-0.0084, -0.0234],
            [-0.0136, -0.0070]
        ];
        let y_train = array![[-0.0084, -0.0234, 0.0051], [-0.0136, -0.0070, -0.0038]];
        let metric = calc_metric(eps, &stiefel, &beta, &x_train, &y_train);
        assert_eq!(0.45544479334867666, metric);
    }

    #[test]
    fn test_schwartz_rutishauser_qr() {
        let matrix = array![
            [-0.24498023, -0.144783],
            [0.014980094, 0.5980434],
            [-0.45020490, 1.0214345],
            [0.689404899, 0.4580322],
            [-0.075092312, -0.232462],
            [0.064296445, 0.9320769]
        ];
        let (q, r) = schwartz_rutishauser_qr(&matrix);
        let error = q.dot(&r) - matrix;
        let eps = 1e-10;
        assert!(error.iter().all(|err| err.abs() <= eps));
        assert!(is_orthonormal(eps, &q));
    }
}
