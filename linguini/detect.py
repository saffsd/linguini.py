"""
Implementation of the linguini language detector.

Marco Lui, April 2013
"""

import base64, bz2, cPickle
import numpy as np
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)

class Linguini(object):

  @classmethod
  def from_modelpath(cls, path, *args, **kwargs):
    with open(path) as f:
      return cls.from_modelstring(f.read(), *args, **kwargs)

  @classmethod
  def from_modelstring(cls, string, *args, **kwargs):
    model = cPickle.loads(bz2.decompress(base64.b64decode(string)))
    ilf, lprot, langs, tk_nextmove, tk_output = model

    # convert to np arrays
    ilf = np.array(ilf)
    lprot = np.array(lprot).reshape(len(lprot) / len(ilf), len(ilf))

    return cls(ilf, lprot, langs, tk_nextmove, tk_output, *args, **kwargs)

  def __init__(self, ilf, lprot, langs, tk_nextmove, tk_output):
    self.ilf = ilf
    self.lprot = lprot
    self.langs = langs
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output

    self.n_langs, self.n_feats = lprot.shape

  def instance2fv(self, text):
    """
    Map an instance into the feature space of the trained model.
    """
    if isinstance(text, unicode):
      text = text.encode('utf8')

    arr = np.zeros((self.n_feats,), dtype='uint32')

    # Convert the text to a sequence of ascii values
    ords = map(ord, text)

    # Count the number of times we enter each state
    state = 0
    statecount = defaultdict(int)
    for letter in ords:
      state = self.tk_nextmove[(state << 8) + letter]
      statecount[state] += 1

    # Update all the productions corresponding to the state
    for state in statecount:
      for index in self.tk_output.get(state, []):
        arr[index] += statecount[state]

    # The returned vector is the TFxIDF vector. The IDF for the
    # linguini system is actually the inv-lang-freq, and this is
    # pre-computed from the training data. We also normalize to len 1
    # at this stage.
    retval = arr * self.ilf
    return retval

  def detect(self, text):
    logger.debug("detect on an instance of len {0}".format(len(text)))
    fv = self.instance2fv(text)
    vector_len = np.sqrt(fv.dot(fv)) 
    fv /= vector_len # TODO: This may not be neccessary for ranking candidate langs
    lang_index = self.lprot.dot(fv).argmax()
    lang = self.langs[lang_index]
    return lang


import sys
def main(args):
  logger.info("reading model from: {0}".format(args.model))
  x = Linguini.from_modelpath(args.model)

  def _process(text):
    return x.detect(text)

  if sys.stdin.isatty():
    # Interactive mode
    while True:
      try:
        text = raw_input(">>> ")
      except Exception:
        break
      print _process(text)
  else:
    # Redirected
    if options.line:
      for line in sys.stdin.readlines():
        print _process(line)
    else:
      print _process(sys.stdin.read())



