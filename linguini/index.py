#!/usr/bin/env python
"""
index.py - 
Index a corpus that is stored in a directory hierarchy as follows:

- corpus
  - domain1
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - domain2
    - language1
      - file1
      - file2
      - ...
    - language2
    - ...
  - ...

This produces 3 files: 
* index: a list of paths, together with the langid and domainid as integers
* lang_index: a list of languages in ascending order of id, with the count for each
* domain_index: a list of domains in ascending order of id, with the count for each

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
import random
import numpy
import logging
from itertools import tee, imap, islice
from collections import defaultdict

from common import Enumerator, makedir
from defaults import TRAIN_PROP, MIN_DOMAIN

logger = logging.getLogger(__name__)

class CorpusIndexer(object):
  """
  Class to index the contents of a corpus
  """
  def __init__(self, root, min_domain=MIN_DOMAIN, proportion=TRAIN_PROP, langs=None, domains=None):
    self.root = root
    self.min_domain = min_domain
    self.proportion = proportion 

    if langs is None:
      self.lang_index = defaultdict(Enumerator())
    else:
      # pre-specified lang set
      self.lang_index = dict((k,v) for v,k in enumerate(langs))

    if domains is None:
      self.domain_index = defaultdict(Enumerator())
    else:
      # pre-specified domain set
      self.domain_index = dict((k,v) for v,k in enumerate(domains))

    self.coverage_index = defaultdict(set)
    self.items = list()

    self.index(root)
    self.prune_min_domain(self.min_domain)

  def index(self, root):
    if os.path.isdir(root):
      # root supplied was the root of a directory structure
      candidates = []
      for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for docname in filenames:
          candidates.append((dirpath, docname))
    else:
      # root supplied was a file, interpet as list of paths
      candidates = [os.path.split(str.strip(l)) for l in open(root)]

    # build a list of paths
    paths = []
    for dirpath, docname in candidates:
      if random.random() < self.proportion:
        # Each file has 'proportion' chance of being selected.
        path = os.path.join(dirpath, docname)

        # split the dirpath into identifying components
        d, lang = os.path.split(dirpath)
        d, domain = os.path.split(d)

        # index the language and the domain
        try:
          # TODO: If lang is pre-specified but not domain, we can end up 
          #       enumerating empty domains.
          domain_id = self.domain_index[domain]
          lang_id = self.lang_index[lang]
        except KeyError:
          # lang or domain outside a pre-specified set so
          # skip this document.
          continue

        # add the domain-lang relation to the coverage index
        self.coverage_index[domain].add(lang)

        # add the item to our list
        self.items.append((domain_id,lang_id,docname,path))

  def prune_min_domain(self, min_domain):
    # prune files for all languages that do not occur in at least min_domain 
     
    # Work out which languages to reject as they are not present in at least 
    # the required number of domains
    lang_domain_count = defaultdict(int)
    for langs in self.coverage_index.values():
      for lang in langs:
        lang_domain_count[lang] += 1
    reject_langs = set( l for l in lang_domain_count if lang_domain_count[l] < min_domain)

    # Remove the languages from the indexer
    if reject_langs:
      logger.debug("reject (<{0} domains): {1}".format(min_domain, sorted(reject_langs)))
      reject_ids = set(self.lang_index[l] for l in reject_langs)
    
      new_lang_index = defaultdict(Enumerator())
      lm = dict()
      for k,v in self.lang_index.items():
        if v not in reject_ids:
          new_id = new_lang_index[k]
          lm[v] = new_id

      # Eliminate all entries for the languages
      self.items = [ (d, lm[l], n, p) for (d, l, n, p) in self.items if l in lm]

      self.lang_index = new_lang_index


  @property
  def dist_lang(self):
    """
    @returns A vector over frequency counts for each language
    """
    retval = numpy.zeros((len(self.lang_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[l] += 1
    return retval

  @property
  def dist_domain(self):
    """
    @returns A vector over frequency counts for each domain 
    """
    retval = numpy.zeros((len(self.domain_index),), dtype='int')
    for d, l, n, p in self.items:
      retval[d] += 1
    return retval

  # TODO: Remove this as it should no longer be needed
  @property 
  def classmaps(self):
    num_instances = len(self.items)
    if num_instances == 0:
      raise ValueError("no items indexed!")
    cm_domain = numpy.zeros((num_instances, len(self.domain_index)), dtype='bool')
    cm_lang = numpy.zeros((num_instances, len(self.lang_index)), dtype='bool')

    # Populate the class maps
    for docid, (domain_id, lang_id, docname, path) in enumerate(self.items):
      cm_domain[docid, domain_id] = True
      cm_lang[docid, lang_id] = True
    return cm_domain, cm_lang

  @property
  def paths(self):
    return [ p for (d,l,n,p) in self.items ]



def main(args):
  corpus_name = os.path.basename(args.corpus)
  if args.model:
    model_dir = args.model
  else:
    model_dir = os.path.join('.', corpus_name+'.model')

  makedir(model_dir)

  langs_path = os.path.join(model_dir, 'lang_index')
  domains_path = os.path.join(model_dir, 'domain_index')
  index_path = os.path.join(model_dir, 'paths')

  # display paths
  logging.info("corpus path: {0}".format(args.corpus))
  logging.info("model path: {0}".format(model_dir))
  logging.info("writing langs to: {0}".format(langs_path))
  logging.info("writing domains to: {0}".format(domains_path))
  logging.info("writing index to: {0}".format(index_path))

  indexer = CorpusIndexer(args.corpus, min_domain=args.min_domain, proportion=args.proportion,
                          langs = args.lang, domains = args.domain)

  # Compute mappings between files, languages and domains
  lang_dist = indexer.dist_lang
  lang_index = indexer.lang_index
  lang_info = ' '.join(("{0}({1})".format(k, lang_dist[v]) for k,v in lang_index.items()))
  logging.info("langs({0}): {1}".format(len(lang_dist), lang_info))

  domain_dist = indexer.dist_domain
  domain_index = indexer.domain_index
  domain_info = ' '.join(("{0}({1})".format(k, domain_dist[v]) for k,v in domain_index.items()))
  logging.info("domains({0}): {1}".format(len(domain_dist), domain_info))

  logging.info("identified {0} files".format(len(indexer.items)))

  # output the language index
  with open(langs_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows((l, lang_dist[lang_index[l]]) 
        for l in sorted(lang_index.keys(), key=lang_index.get))

  # output the domain index
  with open(domains_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows((d, domain_dist[domain_index[d]]) 
        for d in sorted(domain_index.keys(), key=domain_index.get))

  # output items found
  with open(index_path,'w') as f:
    writer = csv.writer(f)
    writer.writerows( (d,l,p) for (d,l,n,p) in indexer.items )

