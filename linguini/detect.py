"""
Implementation of the linguini language detector.

Marco Lui, April 2013
"""

import base64, bz2, cPickle
import numpy as np
import logging

from collections import defaultdict
from itertools import combinations

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

  def __init__(self, ilf, lprot, langs, tk_nextmove, tk_output, topn=5):
    self.ilf = ilf
    self.lprot = lprot
    self.langs = langs
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output

    # scalar products between all the language prototypes
    self.ldot = lprot.dot(lprot.T)

    self.n_langs, self.n_feats = lprot.shape
    self.topn = topn

    logger.debug("initialized a Linguini instance")
    logger.debug("n_langs={0} n_feats={1} topn={2}".format(self.n_langs, self.n_feats, self.topn))


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
    #vector_len = np.sqrt(fv.dot(fv)) 
    #fv /= vector_len # TODO: This may not be neccessary for ranking candidate langs
    #lang_index = self.lprot.dot(fv).argmax()
    #fv = self.lprot[12] # FOR TESTING OMLY
    fdot = self.lprot.dot(fv) 
    lang_order = np.arange(self.n_langs)[fdot.argsort()]

    best = {lang_order[-1] : 1.0}
    best_score = fdot[lang_order[-1]]

    candidates = lang_order[::-1][:self.topn]
    for l1, l2 in combinations(candidates, 2):
      fifj = self.ldot[l1,l2]

      # Alpha can be outside the range 0-1. This simply indicates that the projection
      # of d onto the fi-fj plane is not between fi and fj.
      alpha_num =  fdot[l1] - fdot[l2] * fifj 
      alpha_den = (1. - fifj) * (fdot[l1] + fdot[l2])
      alpha = alpha_num / alpha_den
      #logger.debug("alpha[{0},{1}]: {2:.2f} / {3:.2f} = {4:.2f}".format(self.langs[l1], self.langs[l2], alpha_num, alpha_den, alpha))

      dk = alpha * fdot[l1] + (1-alpha) * fdot[l2]
      score = dk / (alpha * alpha + 2 * alpha * (1.-alpha) * fifj + (1-alpha) * (1-alpha) )

      if score > best_score:
        new_best = { l1:alpha, l2:(1-alpha)}

        logger.debug("replacing {0}({1}) with {2}({3})".format([self.langs[c] for c in best], best_score, [self.langs[c] for c in new_best], score))
        best_score = score
        best = new_best
    
    retval = dict( (self.langs[c],best[c]) for c in best )
    return retval 


import sys
def main(args):
  logger.info("reading model from: {0}".format(args.model))
  identifier = Linguini.from_modelpath(args.model, topn=args.topn)

  def _process(text):
    return identifier.detect(text)

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
    if args.line:
      for line in sys.stdin.readlines():
        print _process(line)
    else:
      print _process(sys.stdin.read())



