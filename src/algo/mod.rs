mod context;
mod ema;
mod error;
mod ma;
mod skip_nan_window;

use std::fmt::{Debug, Display};

pub use context::Context;
pub use ema::*;
pub use error::Error;
pub use ma::*;
use num_traits::Float;

#[inline]
pub fn is_normal<T: Float>(a: &T) -> bool {
  !a.is_nan()
}

pub fn assert_vec_eq_nan<T: Float + Display + Debug>(a: &[T], b: &[T]) {
  assert!(
    a.len() == b.len(),
    "length mismatch, a.len(): {}, b.len(): {}",
    a.len(),
    b.len()
  );
  for i in 0..a.len() {
    assert!(
      (a[i] - b[i]).abs() < T::from(0.000001).unwrap() || (!is_normal(&a[i]) && !is_normal(&b[i])),
      "index {} mismatch, a: {}, b: {}\nvec a: {:?}\nvec b: {:?}",
      i,
      a[i],
      b[i],
      a,
      b
    );
  }
}
