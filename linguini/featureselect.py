"""
Feature selection as detailed in Prager(199). The paragraph of interest reads
as follows:
'If feature i occurred mi times in  a language training set, the value we 
stored was the integral part of k.mi/ni. This allowed us to suppress 
infrequently occurring words, especially if they occurred in many other 
languages, and varying k allowed us to control this filter.  Specifically, 
a word would not be stored if its occurrence count mi < ni * k. We 
experimented with values from .1 to 10, and found that values in the region 
of .3 to .5 work best.'
"""

import os, sys, argparse
import collections
import csv
import shutil
import tempfile
import marshal
import random
import numpy
import cPickle
import multiprocessing as mp
import atexit
import logging
from itertools import tee, imap, islice
from collections import defaultdict
from datetime import datetime
from contextlib import closing

from common import Enumerator, unmarshal_iter, MapPool, write_features, write_weights
from defaults import MAX_NGRAM_ORDER, TOKENS_PER_ORDER

logger = logging.getLogger(__name__)

def pass_sum_lf(bucket):
  """
  Compute 'document frequency' as defined by Prager(1999) - the number of
  languages a term appears in.
  """
  term_langs = defaultdict(set)

  for path in os.listdir(bucket):
    if path.endswith('.lang'):
      for key, lang, _ in unmarshal_iter(os.path.join(bucket,path)):
        term_langs[key].add(lang)
  
  retval = dict( (k,len(v)) for k,v in term_langs.iteritems() ) 
  return retval

def tally_lf(bucketlist, jobs=None):
  """
  Sum up k,v pairs across all buckets. This builds a global mapping of
  terms to the number of languages the terms occur in
  """

  lang_count = {}
  with MapPool(jobs) as f:
    pass_sum_lf_out = f(pass_sum_lf, bucketlist)

    for i, v in enumerate(pass_sum_lf_out):
      lang_count.update(v)
      logger.debug( "processed bucket ({0}/{1}) [{2} terms]".format(i+1, len(bucketlist), len(v)))

  return lang_count
  
def prager_select(bucketlist, lang_count, k, jobs=None):
  """
  Compute the feature selection score according to Prager (1999).
  This is basically a tf-idf computation (where the 'df' used is
  number of languages a term occurs in rather than number of training
  documents). 

  @param k threshold value for selection. We select when score > k
  """
  features = set()
  with MapPool(jobs, setup_pass_select, (lang_count,k)) as f:
    pass_select_out = f(pass_select, bucketlist)

    for i, feats in enumerate(pass_select_out):
      features |= feats
      logger.debug( "processed bucket ({0}/{1}) [selected {2}]".format(i+1, len(bucketlist), len(feats)))

  return features

def setup_pass_select(lang_count, k):
  global __lang_count, __k
  __lang_count = lang_count # mapping (term) -> lang_freq
  __k = k # inclusion threshold for features

def pass_select(bucket):
  """
  Select features from a chunk that meet our selection criteria.
  """
  global __lang_count, __k

  # Compute the term-language frequency first
  term_lang_count = defaultdict(int)
  for path in os.listdir(bucket):
    if path.endswith('.lang'):
      for key, lang, value in unmarshal_iter(os.path.join(bucket,path)):
        term_lang_count[key, lang] += value
    
  features = set()
  for (term, lang), count in term_lang_count.iteritems():
    if count >= __k * __lang_count[term]:
      features.add(term)
  return features

def main(args):
  if args.features:
    feature_path = args.features
  else:
    feature_path = os.path.join(args.model, 'PragerFeats')

  bucketlist_path = os.path.join(args.model, 'bucketlist')

  # display paths
  logger.info("buckets path: %s", bucketlist_path)
  logger.info("features output path: %s", feature_path)
  logger.info("k = {0}".format(args.k))

  with open(bucketlist_path) as f:
    bucketlist = map(str.strip, f)

  lang_count = tally_lf(bucketlist, args.jobs)
  total_feats = len(lang_count)
  logger.info("unique features: {0}".format(total_feats))

  feats = prager_select(bucketlist, lang_count, args.k, args.jobs)
  logger.info("selected features: {0} / {1} ({2:.2f}%)".format(len(feats), total_feats, 100. * len(feats) / total_feats))

  write_features(feats, feature_path)
  logger.info('wrote features to "%s"', feature_path )

  
