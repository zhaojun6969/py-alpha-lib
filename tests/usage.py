import alpha as al
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# Calculate 3-period moving average, note that first 2 values are average of available values
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float64)
result = al.MA(data, 3)
print(result)
# Output: [1.  1.5 2.  3.  4.  5.  6.  7.  8.  9. ]

# Calculate 3-period exponential moving average, first 2 values are NaN
al.set_ctx(flags=al.FLAG_STRICTLY_CYCLE)
result = al.MA(data, 3)
print(result)
# Output: [nan nan  2.  3.  4.  5.  6.  7.  8.  9.]


# Calculate 3-period exponential moving average, skipping NaN values
al.set_ctx(flags=al.FLAG_SKIP_NAN)
data_with_nan = np.array([1, 2, None, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
result = al.MA(data_with_nan, 3)
print(result)
# Output: [1.    1.5     nan 2.333 3.667 5.    6.    7.    8.    9.   ]