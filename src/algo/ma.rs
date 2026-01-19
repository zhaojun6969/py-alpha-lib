use num_traits::Float;
use rayon::prelude::*;

use crate::algo::{Context, Error, is_normal, skip_nan_window::SkipNanWindow};

/// Moving Average
///
/// https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
///
///
pub fn ta_ma<NumT: Float + Send + Sync>(
  ctx: &Context,
  r: &mut [NumT],
  input: &[NumT],
  periods: usize,
) -> Result<(), Error> {
  if r.len() != input.len() {
    return Err(Error::LengthMismatch(r.len(), input.len()));
  }

  if periods == 1 {
    r.copy_from_slice(input);
    return Ok(());
  }

  r.par_chunks_mut(ctx.chunk_size(r.len()))
    .zip(input.par_chunks(ctx.chunk_size(input.len())))
    .for_each(|(r, x)| {
      let start = ctx.start(r.len());
      r.fill(NumT::nan());
      if ctx.skip_nan() {
        let iter = SkipNanWindow::new(x, periods, start);
        let mut sum = NumT::zero();
        for i in iter {
          let val = x[i.end];
          if val.is_normal() {
            sum = sum + val;
          }

          // subtract values that fell out of the window
          for k in i.prev_start..i.start {
            let old = x[k];
            if old.is_normal() {
              sum = sum - old;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.strictly_cycle() {
            // strict cycle with skip_nan implies we want 'periods' valid numbers,
            // BUT existing behavior implies we return NaN if there are any NaNs in the window.
            if i.no_nan_count == periods && (i.end - i.start + 1) == periods {
              r[i.end] = sum / NumT::from(periods).unwrap();
            }
          } else {
            r[i.end] = sum / NumT::from(i.no_nan_count).unwrap();
          }
        }
      } else {
        let mut sum = NumT::zero();
        let mut nan_in_window = 0;

        // Pre-initialization for start > 0
        let pre_fill_start = if start >= periods { start - periods } else { 0 };
        for k in pre_fill_start..start {
          if x[k].is_normal() {
            sum = sum + x[k];
          } else {
            nan_in_window += 1;
          }
        }

        for i in start..x.len() {
          let val = x[i];

          // Add new value
          if val.is_normal() {
            sum = sum + val;
          } else {
            nan_in_window += 1;
          }

          // Remove old value
          if i >= periods {
            let old = x[i - periods];
            if old.is_normal() {
              sum = sum - old;
            } else {
              nan_in_window -= 1;
            }
          }

          if !is_normal(&val) {
            continue;
          }

          if ctx.strictly_cycle() {
            if i >= periods - 1 {
              if nan_in_window == 0 {
                r[i] = sum / NumT::from(periods).unwrap();
              }
            }
          } else {
            if i < periods {
              if nan_in_window == 0 {
                let count = if i < periods { i + 1 } else { periods };
                r[i] = sum / NumT::from(count).unwrap();
              }
            } else {
              if nan_in_window == 0 {
                r[i] = sum / NumT::from(periods).unwrap();
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
  use crate::algo::{
    assert_vec_eq_nan,
    context::{FLAG_SKIP_NAN, FLAG_STRICTLY_CYCLE},
  };

  use super::*;

  #[test]
  fn test_ta_ma() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];
    let ctx = Context::new(0, 0, 0);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_eq!(r, vec![1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    let ctx = Context::new(0, 0, FLAG_STRICTLY_CYCLE);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![f64::NAN, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
  }

  #[test]
  fn test_ta_ma_skip_nan() {
    let input = vec![1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let periods = 3;
    let mut r = vec![0.0; input.len()];

    // don't skip nan
    let ctx = Context::new(0, 0, 0);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        1.0,
        1.5,
        2.0,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );

    // skip nan
    let ctx = Context::new(0, 0, FLAG_SKIP_NAN);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        1.0,
        1.5,
        2.0,
        f64::NAN,
        3.333333,
        4.666667,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );

    // skip nan and strictly cycle
    let ctx = Context::new(2, 0, FLAG_SKIP_NAN | FLAG_STRICTLY_CYCLE);
    ta_ma(&ctx, &mut r, &input, periods).unwrap();
    assert_vec_eq_nan(
      &r,
      &vec![
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        6.0,
        7.0,
        8.0,
        9.0,
      ],
    );
  }
}
