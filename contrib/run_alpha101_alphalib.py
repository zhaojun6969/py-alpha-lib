# !/usr/bin/env python3
# DolphinDB Inc.
# @Author: DolphinDB
# @Last modification time: 2022.09.01
# @FileName: wq101alphaPyTime.py

# This script is to test the performance of the implementation of WorldQuant 101 alpha in python.
# You will need to put alpha101_adjusted.py and this wq101alphaPyTime.py in the same folder.
# You will need to use dataPerformance.csv. Don't forget to change your directory.
# The overall time cost is about 42 minutes.

import time
import pandas as pd
from tqdm import tqdm
from alpha101_alphalib import Alphas
import traceback
import logging
import numpy as np

np.set_printoptions(precision=3, suppress=True)


class Context:
  def __init__(self, start=0, groups=1, flags=0):
    self.start = start
    self.groups = groups
    self.flags = flags


_ALGO_CTX_ = Context()

FORMAT = "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"

logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger("MAIN")

logger.info("Loading data")
data = pd.read_csv("/home/jia/Downloads/101_data/dataPerformance.csv")
# df = data.pivot(index="tradetime", columns="securityid")
df = data.set_index(["securityid", "tradetime"])
_ALGO_CTX_.groups = df.index.get_level_values("securityid").value_counts().shape[0]

logger.info("Data loaded")

stock = Alphas(df)

a1 = getattr(Alphas, "alpha00" + str(1))
times = []

nofunc = [48, 56, 58, 59, 63, 67, 69, 70, 76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]

start = 1
end = 2

for i in tqdm(range(start, end)):
  if i in nofunc:
    times.append("no function")
    continue
  else:
    factor = getattr(Alphas, "alpha{:03d}".format(i))
  try:
    t1 = time.time()
    res = factor(stock)
    t2 = time.time()
    times.append(t2 - t1)
    print(res[0:261])
  except Exception:
    traceback.print_exc()
    times.append("error")

timeRes = pd.DataFrame({"alphaName": list(range(start, end)), "timeCost": times})
# timeRes.to_csv("test/pyPerformance.txt", index=False)
print(timeRes)
