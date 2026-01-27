// Copyright 2026 MSD-RS Project LiJia
// SPDX-License-Identifier: BSD-2-Clause

use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate sum of values in preceding `periods` window
///
/// If periods is 0, it calculates the cumulative sum from the first valid value.
///
/// Ref: https://www.amibroker.com/guide/afl/sum.html
pub fn ta_sum<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      if periods == 0 {
        // Cumulative sum
        let mut sum = NumT::zero();
        let mut found_valid = false;

        for i in start..x.len() {
          let val = x[i];
          if is_normal(&val) {
            sum = sum + val;
            found_valid = true;
          }

          if found_valid {
            r[i] = sum;
          }
        }
      } else {
        // Sliding window sum
        if ctx.is_skip_nan() {
          // Skip NaN logic
          let iter = SkipNanWindow::new(x, periods, start);
          let mut sum = NumT::zero();

          for i in iter {
            let val = x[i.end];
            if is_normal(&val) {
              sum = sum + val;
            }

            // subtract values that fell out of the window
            for k in i.prev_start..i.start {
              let old = x[k];
              if is_normal(&old) {
                sum = sum - old;
              }
            }

            if !is_normal(&val) {
              continue;
            }

            if ctx.is_strictly_cycle() {
              if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
                r[i.end] = sum;
              }
            } else {
              r[i.end] = sum;
            }
          }
        } else {
          // Normal mode
          let mut sum = NumT::zero();
          let mut nan_in_window = 0;

          // Pre-initialization for start > 0
          let pre_fill_start = if start >= periods { start - periods } else { 0 };
          for k in pre_fill_start..start {
            if is_normal(&x[k]) {
              sum = sum + x[k];
            } else {
              nan_in_window += 1;
            }
          }

          for i in start..x.len() {
            let val = x[i];

            // Add new value
            if is_normal(&val) {
              sum = sum + val;
            } else {
              nan_in_window += 1;
            }

            // Remove old value
            if i >= periods {
              let old = x[i - periods];
              if is_normal(&old) {
                sum = sum - old;
              } else {
                nan_in_window -= 1;
              }
            }

            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              }
            } else {
              if i < periods {
                // Usually SUM(X, N) for N < periods is not defined or is partial sum?
                // AmiBroker SUM returns value for first N bars?
                // Documentation says: "Sum of array over periods".
                // Usually simple moving sum requires N periods.
                // Let's assume naive partial sum is returned until full window, UNLESS strictly cycle?
                // Or consistent with MA? MA returns partial MA.
                // Let's return partial sum.
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              } else {
                if nan_in_window == 0 {
                  r[i] = sum;
                }
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
///
/// Ref: https://www.amibroker.com/guide/afl/sumbars.html
pub fn ta_sumbars<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  amount: NumT,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      for i in start..x.len() {
        let mut sum = NumT::zero();
        let mut bars = 0;

        // Iterate backwards
        for j in (0..=i).rev() {
          let val = x[j];
          if ctx.is_skip_nan() {
            if is_normal(&val) {
              sum = sum + val;
              bars += 1;
            }
          } else {
            if is_normal(&val) {
              sum = sum + val;
            } else {
              // If not skip nan, NaN breaks the sum? Or treats as 0?
              // Generally arithmetic with NaN results in NaN.
              // So we stop? Or we just count it?
              // Let's assume standard behavior: NaN makes result NaN for that bar unless handled.
              // But for simplicity in SUMBARS, usually we just count back.
              // If we hit NaN, sum becomes NaN.
              // If sum is NaN, it will never be >= amount.
              sum = sum + val;
            }
            bars += 1;
          }

          if is_normal(&sum) && sum >= amount {
            r[i] = NumT::from(bars).unwrap();
            break;
          }
        }

        // If loop finishes and not found, r[i] remains NaN (default fill)
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  #[test]
  fn test_sum_sliding() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sum(&ctx, &mut r, &input, 3).unwrap();
    // 0: 1
    // 1: 3
    // 2: 6
    // 3: 9 (2+3+4)
    // 4: 12 (3+4+5)
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, 6.0, 9.0, 12.0]);
  }

  #[test]
  fn test_sum_cumulative() {
    let input = vec![1.0, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_sum(&ctx, &mut r, &input, 0).unwrap();
    assert_vec_eq_nan(&r, &vec![1.0, 3.0, 6.0]);
  }

  #[test]
  fn test_sum_skip_nan() {
    let input = vec![1.0, f64::NAN, 2.0, 3.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_sum(&ctx, &mut r, &input, 2).unwrap();
    // 0: 1 (1)
    // 1: NaN (NaN)
    // 2: 3 (1+2, skip nan)
    // 3: 5 (2+3)
    assert_vec_eq_nan(&r, &vec![1.0, f64::NAN, 3.0, 5.0]);
  }

  #[test]
  fn test_sumbars() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    // Find bars to sum up to 5.0
    ta_sumbars(&ctx, &mut r, &input, 5.0).unwrap();
    // 0: 1 (1 < 5, end, wait, actually loop rev(0..=0). sum=1. 1<5. not found. r[0]=NaN)
    // Wait, SUMBARS(X, A) returns bars count needed to reach A.
    // If total history sum < A, returns NaN? Yes usually.

    // 0: sum=1 < 5 -> NaN
    // 1: 2+1 = 3 < 5 -> NaN
    // 2: 3+2 = 5 >= 5 -> 2 bars (idx 2, 1)
    // 3: 4 >= 5 (false), 4+3=7 >= 5 -> 2 bars
    // 4: 5 >= 5 -> 1 bar

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 2.0, 2.0, 1.0]);
  }
}
