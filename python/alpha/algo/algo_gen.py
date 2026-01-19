# THIS FILE IS AUTO-GENERATED, DO NOT EDIT

import numpy as np
from . import _algo

def DMA(
  input: np.ndarray | list[np.ndarray], alpha: float
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  
  current = alpha * current + (1 - alpha) * previous
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.dma(r, input, alpha)
    return r
  else:
    r = np.empty_like(input)
    _algo.dma(r, input, alpha)
    return r

def MA(
  input: np.ndarray | list[np.ndarray], periods: int
) -> np.ndarray | list[np.ndarray]:
  """
  Moving Average
  
  https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average
  
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.ma(r, input, periods)
    return r
  else:
    r = np.empty_like(input)
    _algo.ma(r, input, periods)
    return r

def SMA(
  input: np.ndarray | list[np.ndarray], n: int, m: int
) -> np.ndarray | list[np.ndarray]:
  """
  Exponential Moving Average (variant of EMA)
  
  alpha = m / n
  
  https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
  """
  if isinstance(input, list):
    r = [np.empty_like(x) for x in input]
    _algo.sma(r, input, n, m)
    return r
  else:
    r = np.empty_like(input)
    _algo.sma(r, input, n, m)
    return r

