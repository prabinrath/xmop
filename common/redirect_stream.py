import os
import ctypes
import sys

# https://github.com/bulletphysics/bullet3/discussions/3441
class RedirectStream(object):
  @staticmethod
  def _flush_c_stream(stream):
    streamname = stream.name[1:-1]
    libc = ctypes.CDLL(None)
    libc.fflush(ctypes.c_void_p.in_dll(libc, streamname))

  def __init__(self, file=os.devnull):
    self.stream = sys.stdout
    sys.stderr = open(os.devnull, "w+")
    self.file = file

  def __enter__(self):
    self.stream.flush()  # ensures python stream unaffected 
    self.fd = open(self.file, "w+")
    self.dup_stream = os.dup(self.stream.fileno())
    os.dup2(self.fd.fileno(), self.stream.fileno()) # replaces stream
  
  def __exit__(self, type, value, traceback):
    RedirectStream._flush_c_stream(self.stream)  # ensures C stream buffer empty
    os.dup2(self.dup_stream, self.stream.fileno()) # restores stream
    os.close(self.dup_stream)
    self.fd.close()