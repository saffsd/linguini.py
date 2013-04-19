#!/usr/bin/env python
"""
tokenize.py - 
Tokenizer for langid.py training system. This takes a list of files and tokenizes them
in parallel.

Marco Lui, January 2013

Copyright 2013 Marco Lui <saffsd@gmail.com>. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of the copyright holder.
"""

import os, sys, argparse
import csv
import shutil
import tempfile
import marshal
import multiprocessing as mp
import random
import atexit
import logging

from itertools import tee, izip, ifilter
from collections import defaultdict

from common import makedir, chunk, MapPool
from defaults import MAX_NGRAM_ORDER, TOP_DOC_FREQ, NUM_BUCKETS, CHUNKSIZE

logger = logging.getLogger(__name__)

class NGramTokenizer(object):
  def __init__(self, min_order=1, max_order=3):
    self.min_order = min_order
    self.max_order = max_order

  def __call__(self, seq):
    min_order = self.min_order
    max_order = self.max_order
    t = tee(seq, max_order)
    for i in xrange(max_order):
      for j in xrange(i):
        # advance iterators, ignoring result
        t[i].next()
    while True:
      token = ''.join(tn.next() for tn in t)
      if len(token) < max_order: break
      for n in xrange(min_order-1, max_order):
        yield token[:n+1]
    for a in xrange(max_order-1):
      for b in xrange(min_order, max_order-a):
        yield token[a:a+b]

class PragerTokenizer(object):
  """
  Implementation of a tokenizer that produces tokens according
  to those described in Prager (1999). The key excerpt from the
  paper is as follows:
  'The features we chose to examine were N-grams (sequences of N
  consecutive characters, not spanning words but possibly including
  word-ending spaces) with N ranging from 2 to 5, and words of
  two different size groups, either words of 4 charaters or less,
  or words of any length.'
  """

  def __init__(self, n=4, use_words=False, use_short_words=False, word_maxlen=30):
    if use_short_words:
      raise NotImplementedError("not yet implemented")
    self.n = n
    self.use_words = use_words
    self.word_maxlen = word_maxlen

  def __call__(self, seq):
    if self.use_words:
      # We generate words first. We use str.split, with the constraint
      # that words cannot have the same length as n-gram order N, to
      # avoid double-generation thereof. We enforce an additional
      # upper bound on word length as a naive heuristic for dealing with
      # non-whitespace-segmented languages and non-language data.
      for word in str.split(seq):
        if word and len(word) != self.n and len(word) <= self.word_maxlen:
          yield tuple(word)

    # Set up N iterators in order to generate n-grams
    t = tee(seq, self.n)
    # advance iterators, ignoring result
    [ t[i].next() for i in xrange(self.n) for j in xrange(i) ]

    ngram_tokens = izip(*t)

    for token in ngram_tokens:
      # Prager's definition requires that we do not span word boundaries.
      # To achieve this, we discard any tokens that contain whitespace
      # (unless the only whitespace is at the end of the word, as 
      # explicitly allowed by Prager)
      if not any(str.isspace(t) for t in token[:-1]):
        yield token



   

@atexit.register
def cleanup():
  global b_dirs, complete
  try:
    if not complete:
      for d in b_dirs:
        shutil.rmtree(d)
  except NameError:
    # Failed before globals defined, nothing to clean
    pass

def setup_pass_tokenize(tokenizer, b_dirs, sample_count, sample_size):
  global __tokenizer, __b_dirs, __sample_count, __sample_size
  __tokenizer = tokenizer
  __b_dirs = b_dirs
  __sample_count = sample_count
  __sample_size = sample_size

def pass_tokenize(chunk_items):
  """
  Chunk files into a doc->term mapping,
  and simultaneously build a term->df count.
  The term->df counts are redistributed to
  buckets via python's in-built hash function.
  This is basically an inversion step, so that 
  now we are chunked on the term axis rather
  than the document axis.
  """
  global __maxorder, __b_dirs, __extractor, __sample_count, __sample_size
  __procname = mp.current_process().name
  b_freq_lang = [tempfile.mkstemp(prefix=__procname+'-', suffix='.lang', dir=p)[0] for p in __b_dirs]
  b_freq_domain = [tempfile.mkstemp(prefix=__procname+'-', suffix='.domain', dir=p)[0] for p in __b_dirs]
  
  extractor = __tokenizer
  term_lng_freq = defaultdict(lambda: defaultdict(int))
  term_dom_freq = defaultdict(lambda: defaultdict(int))

  for domain_id, lang_id, path in chunk_items:
    with open(path) as f:
      if __sample_count:
        # sampling tokenization
        text = f.read()
        poss = max(1,len(text) - __sample_size) # possibe start locations
        count = min(poss, __sample_count) # reduce number of samples if document is too short
        offsets = random.sample(xrange(poss), count)
        for offset in offsets:
          tokenset = set(extractor(text[offset: offset+__sample_size]))
          for token in tokenset:
            term_lng_freq[token][lang_id] += 1
            term_dom_freq[token][domain_id] += 1
          
      else:
        # whole-document tokenization
        tokenset = set(extractor(f.read()))
        for token in tokenset:
          term_lng_freq[token][lang_id] += 1
          term_dom_freq[token][domain_id] += 1

  for term in term_lng_freq:
    bucket_index = hash(term) % len(b_freq_lang)
    for lang, count in term_lng_freq[term].iteritems():
      os.write(b_freq_lang[bucket_index], marshal.dumps((term, lang, count)))
    for domain, count in term_dom_freq[term].iteritems():
      os.write(b_freq_domain[bucket_index], marshal.dumps((term, domain, count)))

  # Close all the open files
  for f in b_freq_lang + b_freq_domain:
    os.close(f)

  return len(term_lng_freq)

def build_index(items, tokenizer, outdir, buckets=NUM_BUCKETS, jobs=None, chunksize=CHUNKSIZE, sample_count=None, sample_size=None):
  """
  @param items a list of (domain, language, path) tuples
  """
  global b_dirs, complete

  # Our exitfunc uses this to know whether to delete the tokenized files
  complete = False 

  if jobs is None:
    jobs = mp.cpu_count() + 4

  b_dirs = [ tempfile.mkdtemp(prefix="tokenize-",suffix='-{0}'.format(tokenizer.__class__.__name__), dir=outdir) for i in range(buckets) ]

  # PASS 1: Tokenize documents into sets of terms
   
  # If there are few items, make the chunk size such that each job
  # will have 2 chunks
  chunk_size = max(1,min(len(items) / (jobs * 2), chunksize))
  item_chunks = list(chunk(items, chunk_size))
  pass_tokenize_globals = (tokenizer, b_dirs, sample_count, sample_size)

  with MapPool(jobs, setup_pass_tokenize, pass_tokenize_globals) as f:
    pass_tokenize_out = f(pass_tokenize, item_chunks)


    doc_count = defaultdict(int)
    chunk_count = len(item_chunks)
    logger.info("chunk size: {0} ({1} chunks)".format(chunk_size, chunk_count))
    logger.info("job count: {0}".format(jobs))

    if sample_count:
      logger.info("sampling-based tokenization: size {0} count {1}".format(sample_size, sample_count))
    else:
      logger.info("whole-document tokenization")

    for i, keycount in enumerate(pass_tokenize_out):
      logger.debug("tokenized chunk (%d/%d) [%d keys]" % (i+1,chunk_count, keycount))

  complete = True

  return b_dirs

def main(args):
  if args.temp:
    buckets_dir = args.temp
  else:
    buckets_dir = os.path.join(args.model, 'buckets')
  makedir(buckets_dir)

  bucketlist_path = os.path.join(args.model, 'bucketlist')
  index_path = os.path.join(args.model, 'paths')

  # display paths
  logger.info("index path: %s", index_path)
  logger.info("bucketlist path: %s", bucketlist_path)
  logger.info("buckets path: %s", buckets_dir)

  with open(index_path) as f:
    reader = csv.reader(f)
    items = list(reader)

  # Tokenize
  logger.info("will tokenize %d files" % len(items))
  if args.scanner:
    from scanner import Scanner
    tokenizer = Scanner.from_file(args.scanner)
    logger.info("using provided scanner: ", args.scanner)
  elif args.prager:
    tokenizer = PragerTokenizer(args.order, use_words=True)
    logger.info("using str.split to tokenize")
  else:
    tokenizer = NGramTokenizer(args.min_order,args.max_order)
    logger.info("using n-gram tokenizer: order {0}-{1}".format(args.min_order, args.max_order))
  b_dirs = build_index(items, tokenizer, buckets_dir, args.buckets, args.jobs, args.chunksize, args.sample_count, args.sample_size)

  # output the paths to the buckets
  with open(bucketlist_path,'w') as f:
    for d in b_dirs:
      f.write(d+'\n')
