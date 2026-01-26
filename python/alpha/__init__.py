from .algo import *
from .lang import *


class Context:
  def __init__(self, groups=1, start=0, flags=0):
    self.groups = groups
    self.start = start
    self.flags = flags
