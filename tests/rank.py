import pandas as pd
import numpy as np
import alpha


df = pd.DataFrame(
  data={
    "Animal": ["cat", "penguin", "dog", "spider", "snake"],
    "Number_legs": [4, 2, 4, 8, 0],
  }
)
df["rank"] = df["Number_legs"].rank(pct=True)

print(df)

_ALGO_CTX_ = alpha.Context(5)
print(alpha.RANK(df["Number_legs"].to_numpy().astype(np.float64)))
