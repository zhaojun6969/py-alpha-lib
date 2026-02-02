import pandas as pd
import numpy as np
import alpha


def returns(a: np.ndarray):
  return a / alpha.REF(a, 1) - 1


class ExecContext:
  def __init__(self, data: pd.DataFrame, securities: int, trades: int):
    self.OPEN = data["open"].to_numpy()
    self.HIGH = data["high"].to_numpy()
    self.LOW = data["low"].to_numpy()
    self.CLOSE = data["close"].to_numpy()
    self.VOLUME = data["vol"].to_numpy().astype(np.float64)
    self.RET = returns(data["close"].to_numpy())
    self.VWAP = data["vwap"].to_numpy()
    self.BANCHMARKINDEXCLOSE = np.repeat(self.CLOSE[0:trades], securities)
    self.BANCHMARKINDEXOPEN = np.repeat(self.OPEN[0:trades], securities)
    self.DTM = self._DTM()
    self.DBM = self._DBM()
    self.TR = self._TR()
    self.HD = self._HD()
    self.LD = self._LD()

  def __call__(self, name: str) -> np.ndarray:
    if name.startswith("ADV"):
      n = name[3:]
      if len(n) == 0:
        return self.VOLUME
      else:
        w = int(n)
        return self.SMA(self.VOLUME, w)
    return getattr(self, name)

  def _DTM(self):
    # (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
    return np.where(
      self.OPEN <= self.DELAY(self.OPEN, 1),
      0,
      self.MAX(self.HIGH - self.OPEN, self.OPEN - self.DELAY(self.OPEN, 1)),
    )

  def _DBM(self):
    # (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
    return np.where(
      self.OPEN >= self.DELAY(self.OPEN, 1),
      0,
      self.MAX(self.OPEN - self.LOW, self.OPEN - self.DELAY(self.OPEN, 1)),
    )

  def _TR(self):
    # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
    return self.MAX(
      self.MAX(
        self.HIGH - self.LOW,
        np.abs(self.HIGH - self.DELAY(self.CLOSE, 1)),
      ),
      np.abs(self.LOW - self.DELAY(self.CLOSE, 1)),
    )

  def _HD(self):
    return self.HIGH - self.DELAY(self.HIGH, 1)

  def _LD(self):
    return self.LOW - self.DELAY(self.LOW, 1)

  def SUM(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  def SUMAC(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.SUM(a, int(w))

  def SMA(self, a: np.ndarray, m: int, n: int) -> np.ndarray:
    return alpha.SMA(a, int(m), int(n))

  def STDDEV(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  def CORR(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.CORR(a, b, int(w))

  def COV(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.COV(a, b, int(w))

  def STD(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.STDDEV(a, int(w))

  def TSRANK(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.TS_RANK(a, int(w))

  def PRODUCT(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.PRODUCT(a, int(w))

  def TSMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LLV(a, int(w))

  def MIN(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

  def TSMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.HHV(a, int(w))

  def MAX(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

  def DELTA(self, a: np.ndarray, p: int) -> np.ndarray:
    return a - alpha.REF(a, int(p))

  def DELAY(self, a: np.ndarray, p: int) -> np.ndarray:
    return alpha.REF(a, int(p))

  def SCALE(self, a: np.ndarray, k: int = 1) -> np.ndarray:
    sum = np.abs(a).sum()
    return a * k / sum

  def RANK(self, a: np.ndarray) -> np.ndarray:
    return alpha.RANK(a)

  def TS_ARGMAX(self, a: np.ndarray, w: int) -> np.ndarray:
    return w - alpha.HHVBARS(a, int(w))

  def TS_ARGMIN(self, a: np.ndarray, w: int) -> np.ndarray:
    return w - alpha.LLVBARS(a, int(w))

  def DECAY_LINEAR(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.LWMA(a, int(w))

  def SIGNEDPOWER(self, a: np.ndarray, p: float | np.ndarray) -> np.ndarray:
    return np.power(a, p)

  def LOG(self, a: np.ndarray) -> np.ndarray:
    return np.log(a)

  def ABS(self, a: np.ndarray) -> np.ndarray:
    return np.abs(a)

  def SIGN(self, a: np.ndarray) -> np.ndarray:
    return np.sign(a)

  def MEAN(self, a: np.ndarray, w: int) -> np.ndarray:
    return alpha.MA(a, int(w))

  def REGBETA(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGBETA(a, b, int(w))

  def REGRESI(self, a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    return alpha.REGRESI(a, b, int(w))

  def SEQUENCE(self, w: int) -> np.ndarray:
    return np.arange(w + 1).astype(np.float64)
