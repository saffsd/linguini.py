"""
Implementation of the linguini language detector.

Marco Lui, April 2013
"""

import base64, bz2, cPickle
import numpy as np
import logging
import sys
import pkgutil 

from collections import defaultdict
from itertools import combinations

logger = logging.getLogger(__name__)

class Linguini(object):

  @classmethod
  def from_package(cls, name, *args, **kwargs):
    modelstr = pkgutil.get_data('linguini','models/'+name)
    return cls.from_modelstring(modelstr, *args, **kwargs)
   
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

  def __init__(self, ilf, lprot, langs, tk_nextmove, tk_output, topn=5, detect_multilingual=True):
    self.ilf = ilf
    self.lprot = lprot
    self.langs = langs
    self.tk_nextmove = tk_nextmove
    self.tk_output = tk_output
    self.detect_multilingual = detect_multilingual

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

    if not self.detect_multilingual:
      logger.debug("skipping test for multilinguality")
    else:
      candidates = lang_order[::-1][:self.topn]
      # Consider 2-lang combinations of the topn candidates
      for l1, l2 in combinations(candidates, 2):
        A = self.ldot[l1,l2]
        X = fdot[l1]
        Y = fdot[l2]

        # Alpha can be outside the range 0-1. This simply indicates that the projection
        # of d onto the fi-fj plane is not between fi and fj.
        a_num =  X - Y * A 
        a_den = (1 - A) * (X + Y)
        a = a_num / a_den
        #logger.debug("a[{0},{1}]: {2:.2f} / {3:.2f} = {4:.2f}".format(self.langs[l1], self.langs[l2], a_num, a_den, a))

        dk = a * X + (1-a) * Y
        kmag = np.sqrt(a * a + 2 * a * (1-a) * A + (1-a) * (1-a) )

        ## Only for validation of the implementation
        #k = a * self.lprot[l1] + (1-a) * self.lprot[l2]
        #dk_alt = fv.dot(k)
        #kmag_alt = np.sqrt(k.dot(k))
        #print "ORD2", "DK", dk, "DKALT", dk_alt, "KMAG", kmag, "KMAGALT", kmag_alt

        score = dk / kmag

        if 0 < a < 1 and score > best_score:
          new_best = { l1:a, l2:(1-a)}

          logger.debug("replacing {0}({1}) with {2}({3})".format([self.langs[c] for c in best], best_score, [self.langs[c] for c in new_best], score))
          best_score = score
          best = new_best

      # Consider 3-lang combinations of the topn candidates
      for l1, l2, l3 in combinations(candidates, 3):
        A = self.ldot[l1,l2]
        B = self.ldot[l2,l3]
        C = self.ldot[l1,l3]
        X = fdot[l1]
        Y = fdot[l2]
        Z = fdot[l3]

        # The formulations of a and g were obtained using sympy to solve the simultaneous equations 
        # described by Prager (1999)
        a_num = (-A*B*Z + A*Y + B**2*X - B*C*Y + C*Z - X)
        a_den = (A**2*Z - A*B*X - A*B*Z - A*C*Y - A*C*Z + A*X + A*Y + B**2*X - B*C*X - B*C*Y + B*Y + B*Z + C**2*Y + C*X + C*Z - X - Y - Z)
        a = a_num / a_den

        g_num =(C + 1 - (X + Z)*(A*C - B)/((C - 1)*(X*(A - B)/(C - 1) + Y)))
        g_den =(-A - B + C + 1 + (X + Z)*((A - B)*(A - C) - (B - 1)*(C - 1))/((C - 1)*(X*(A - B)/(C - 1) + Y)))
        g = g_num/g_den 

        dk = a * fdot[l1] + g * fdot[l2] + (1 - a - g) * fdot[l3]
        kmag = np.sqrt( a**2 + g**2 + (1-a-g)**2 + 2*(a*g*A + a*(1-a-g)*C + g*(1-a-g)*B) )

        #k = a * self.lprot[l1] + g * self.lprot[l2] + (1-a-g) * self.lprot[l3]
        #dk_alt = fv.dot(k)
        #kmag_alt = np.sqrt(k.dot(k))
        #print "ORD3", "DK", dk - dk_alt, "KMAG", kmag - kmag_alt

        score = dk / kmag
        #logger.debug("considering {0},{1},{2}: a={3} g={4} score={5}".format(l1,l2,l3,a,g,score))

        if (0 < a < 1) and (0 < g < 1) and (0 < (1-a-g) < 1) and score > best_score:
          new_best = { l1:a, l2:g, l3:(1-a-g) }

          logger.debug("replacing {0}({1}) with {2}({3})".format([self.langs[c] for c in best], best_score, [self.langs[c] for c in new_best], score))
          best_score = score
          best = new_best
    
    retval = dict( (self.langs[c],best[c]) for c in best )
    return retval 


def main(args):
  # TODO: Tweak interface to allow for specifying multiple files at the commandline, and to provide CSV output
  if args.model:
    logger.info("reading model from: {0}".format(args.model))
    identifier = Linguini.from_modelpath(args.model, topn=args.topn, detect_multilingual=args.multilingual)
  else:
    logger.info("using default model")
    identifier = Linguini.from_package('default', topn=args.topn, detect_multilingual=args.multilingual)

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



