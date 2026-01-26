use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Calculate Variance over a moving window
///
/// Variance = (SumSq - (Sum^2)/N) / (N - 1)
pub fn ta_var<NumT: Float + Send + Sync>(
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

      if ctx.is_skip_nan() {
        let iter = SkipNanWindow::new(x, periods, start);
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();

        for i in iter {
          let val = x[i.end];
          if val.is_normal() {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          }

          for k in i.prev_start..i.start {
            let old = x[k];
            if old.is_normal() {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          let count = NumT::from(i.no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if i.no_nan_count != periods || (i.end - i.start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if i.no_nan_count > 1 {
              let var_num = sum_sq - (sum * sum / count);
              let var = var_num / (count - NumT::one());
              // Precision check
              if var < NumT::zero() {
                r[i.end] = NumT::zero();
              } else {
                r[i.end] = var;
              }
            } else {
              r[i.end] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut sum_sq = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val = x[k];
          if val.is_normal() {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..x.len() {
          let val = x[i];

          if val.is_normal() {
            sum = sum + val;
            sum_sq = sum_sq + val * val;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old = x[i - periods];
            if old.is_normal() {
              sum = sum - old;
              sum_sq = sum_sq - old * old;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !val.is_normal() {
            // Result NaN
          } else {
            let output_idx = i;
            let mut valid = false;
            if ctx.is_strictly_cycle() {
              if i >= periods - 1 {
                valid = true;
              }
            } else {
              if i >= periods - 1 {
                valid = true;
              }
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let var_num = sum_sq - (sum * sum / count);
                let var = var_num / (count - NumT::one());
                if var < NumT::zero() {
                  r[output_idx] = NumT::zero();
                } else {
                  r[output_idx] = var;
                }
              } else {
                r[output_idx] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate Covariance over a moving window
///
/// Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
pub fn ta_cov<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != x.len() || x.len() != y.len() {
    return Err(Error::LengthMismatch(r.len(), x.len()));
  }

  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(x.par_chunks(chunk_size))
    .zip(y.par_chunks(chunk_size))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut no_nan_count = 0;

        let mut win_start = start;

        for i in start..r.len() {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = val_x.is_normal() && val_y.is_normal();

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if old_x.is_normal() && old_y.is_normal() {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(x[win_start].is_normal() && y[win_start].is_normal()) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let count = NumT::from(no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if no_nan_count > 1 {
              let cov_num = sum_xy - (sum_x * sum_y / count);
              let cov = cov_num / (count - NumT::one());
              r[i] = cov;
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if val_x.is_normal() && val_y.is_normal() {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..r.len() {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = val_x.is_normal() && val_y.is_normal();

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if old_x.is_normal() && old_y.is_normal() {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            if i >= periods - 1 {
              valid = true;
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let cov_num = sum_xy - (sum_x * sum_y / count);
                let cov = cov_num / (count - NumT::one());
                r[i] = cov;
              } else {
                r[i] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

/// Calculate Correlation over a moving window
///
/// Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
pub fn ta_corr<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  x: &[NumT],
  y: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != x.len() || x.len() != y.len() {
    return Err(Error::LengthMismatch(r.len(), x.len()));
  }

  let chunk_size = ctx.chunk_size(r.len());

  r.par_chunks_mut(chunk_size)
    .zip(x.par_chunks(chunk_size))
    .zip(y.par_chunks(chunk_size))
    .for_each(|((r, x), y)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());

      if ctx.is_skip_nan() {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut no_nan_count = 0;

        let mut win_start = start;

        for i in start..r.len() {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = val_x.is_normal() && val_y.is_normal();

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
            no_nan_count += 1;
          }

          while no_nan_count > periods {
            let old_x = x[win_start];
            let old_y = y[win_start];
            if old_x.is_normal() && old_y.is_normal() {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              sum_y2 = sum_y2 - old_y * old_y;
              no_nan_count -= 1;
            }
            win_start += 1;
          }

          while win_start <= i && !(x[win_start].is_normal() && y[win_start].is_normal()) {
            win_start += 1;
          }

          if !is_valid {
            continue;
          }

          let count = NumT::from(no_nan_count).unwrap();

          let mut should_output = true;
          if ctx.is_strictly_cycle() {
            if no_nan_count != periods || (i - win_start + 1) != periods {
              should_output = false;
            }
          }

          if should_output {
            if no_nan_count > 1 {
              let numerator = sum_xy - (sum_x * sum_y / count);

              let mut var_x_part = sum_x2 - (sum_x * sum_x / count);
              if var_x_part < NumT::zero() {
                var_x_part = NumT::zero();
              }

              let mut var_y_part = sum_y2 - (sum_y * sum_y / count);
              if var_y_part < NumT::zero() {
                var_y_part = NumT::zero();
              }

              let denominator = (var_x_part * var_y_part).sqrt();

              if denominator > NumT::epsilon() {
                r[i] = numerator / denominator;
              } else {
                r[i] = NumT::nan();
              }
            } else {
              r[i] = NumT::nan();
            }
          }
        }
      } else {
        let mut sum_x = NumT::zero();
        let mut sum_y = NumT::zero();
        let mut sum_xy = NumT::zero();
        let mut sum_x2 = NumT::zero();
        let mut sum_y2 = NumT::zero();
        let mut nan_in_window = 0;

        let pre_fill_start = if start >= periods { start - periods } else { 0 };

        for k in pre_fill_start..start {
          let val_x = x[k];
          let val_y = y[k];
          if val_x.is_normal() && val_y.is_normal() {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..r.len() {
          let val_x = x[i];
          let val_y = y[i];
          let is_valid = val_x.is_normal() && val_y.is_normal();

          if is_valid {
            sum_x = sum_x + val_x;
            sum_y = sum_y + val_y;
            sum_xy = sum_xy + val_x * val_y;
            sum_x2 = sum_x2 + val_x * val_x;
            sum_y2 = sum_y2 + val_y * val_y;
          } else {
            nan_in_window += 1;
          }

          if i >= periods {
            let old_x = x[i - periods];
            let old_y = y[i - periods];
            if old_x.is_normal() && old_y.is_normal() {
              sum_x = sum_x - old_x;
              sum_y = sum_y - old_y;
              sum_xy = sum_xy - old_x * old_y;
              sum_x2 = sum_x2 - old_x * old_x;
              sum_y2 = sum_y2 - old_y * old_y;
            } else {
              nan_in_window -= 1;
            }
          }

          if nan_in_window > 0 || !is_valid {
            // Result NaN
          } else {
            let mut valid = false;
            // Standard check i >= periods - 1
            if i >= periods - 1 {
              valid = true;
            }

            if valid {
              let count = NumT::from(periods).unwrap();
              if periods > 1 {
                let numerator = sum_xy - (sum_x * sum_y / count);

                let mut var_x_part = sum_x2 - (sum_x * sum_x / count);
                if var_x_part < NumT::zero() {
                  var_x_part = NumT::zero();
                }

                let mut var_y_part = sum_y2 - (sum_y * sum_y / count);
                if var_y_part < NumT::zero() {
                  var_y_part = NumT::zero();
                }

                let denominator = (var_x_part * var_y_part).sqrt();

                if denominator > NumT::epsilon() {
                  r[i] = numerator / denominator;
                } else {
                  r[i] = NumT::nan();
                }
              } else {
                r[i] = NumT::nan();
              }
            }
          }
        }
      }
    });

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::algo::{assert_vec_eq_nan, context::FLAG_SKIP_NAN};

  #[test]
  fn test_var() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_var(&ctx, &mut r, &input, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_cov() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.0, 3.0, 4.0, 5.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0]);
  }

  #[test]
  fn test_cov_neg() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, -1.0]);
  }

  #[test]
  fn test_cov_skip_nan() {
    let x = vec![1.0, f64::NAN, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let periods = 2; // small window
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);

    ta_cov(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 0.5]);
  }

  #[test]
  fn test_corr() {
    // x: [1, 2, 3] mean=2, std=1
    // y: [2, 3, 4] mean=3, std=1
    // Cov=1. Corr=1/1 = 1.
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_corr(&ctx, &mut r, &x, &y, periods).unwrap();

    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, 1.0, 1.0, 1.0]);
  }

  #[test]
  fn test_corr_neg() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![3.0, 2.0, 1.0];
    let periods = 3;
    let mut r = vec![0.0; x.len()];
    let ctx = Context::new(0, 0, 0);
    ta_corr(&ctx, &mut r, &x, &y, periods).unwrap();
    assert_vec_eq_nan(&r, &vec![f64::NAN, f64::NAN, -1.0]);
  }
}
