#[cfg(test)]
mod test {
    use super::super::*;
    use ndarray::{array, Dim};
    #[test]
    fn test_is_orthonormal() {
        let a = array![
            [1. / 2_f64.sqrt(), -1. / 2_f64.sqrt()],
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()]
        ];
        assert!(is_orthonormal(1e-6, &a));
    }

    #[test]
    fn test_normalize() {
        let a = array![[1., 1.], [1., 1.]];
        let a_norm = array![
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()],
            [1. / 2_f64.sqrt(), 1. / 2_f64.sqrt()]
        ];
        assert_eq!(normalize(&a), a_norm);

        let b = array![[1., 2.], [3., 4.]];
        let b_norm = array![
            [1. / 10_f64.sqrt(), 2. / 20_f64.sqrt()],
            [3. / 10_f64.sqrt(), 4. / 20_f64.sqrt()]
        ];
        assert_eq!(normalize(&b), b_norm);
    }

    #[test]
    fn test_calc_metric() {
        let eps: f64 = 1e-6;
        let stiefel = array![[1., 1.], [1., 1.], [1., 1.], [1., 1.]];
        let beta = array![[1.], [0.]];
        assert_eq!(-1.0, calc_metric(eps, &stiefel, &beta, &stiefel, &stiefel));
    }
}
