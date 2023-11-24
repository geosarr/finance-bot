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
        assert!(is_orthonormal(1e-6, a));
    }
}
