"""
model.py - compute parameters for cosine-based linguini classifier

Marco Lui, April 2013
"""

import base64, bz2, cPickle
import os, sys, argparse, csv
import array
import numpy as np
import tempfile
import marshal
import atexit, shutil
import multiprocessing as mp
import logging
from collections import deque, defaultdict
from contextlib import closing
from itertools import compress

from common import chunk, unmarshal_iter, read_features, index, MapPool
from defaults import MAX_CHUNK_SIZE

logger = logging.getLogger(__name__)

def offsets(chunks):
  # Work out the path chunk start offsets
  chunk_offsets = [0]
  for c in chunks:
    chunk_offsets.append(chunk_offsets[-1] + len(c))
  return chunk_offsets

def state_trace(path):
  """
  Returns counts of how often each state was entered
  """
  global __nm_arr
  c = defaultdict(int)
  state = 0
  with open(path) as f:
    text = f.read()
    for letter in map(ord,text):
      state = __nm_arr[(state << 8) + letter]
      c[state] += 1
  return c

def setup_pass_tokenize(nm_arr, output_states, tk_output, b_dirs):
  """
  Set the global next-move array used by the aho-corasick scanner
  """
  global __nm_arr, __output_states, __tk_output, __b_dirs
  __nm_arr = nm_arr
  __output_states = output_states
  __tk_output = tk_output
  __b_dirs = b_dirs

def pass_tokenize(arg):
  """
  Tokenize documents and do counts for each feature
  Split this into buckets chunked over features rather than documents
  """
  global __output_states, __tk_output, __b_dirs
  chunk_offset, chunk_paths = arg
  term_freq = defaultdict(int)
  __procname = mp.current_process().name
  __buckets = [tempfile.mkstemp(prefix=__procname, suffix='.index', dir=p)[0] for p in __b_dirs]

  # Tokenize each document and add to a count of (doc_id, f_id) frequencies
  for doc_count, path in enumerate(chunk_paths):
    doc_id = doc_count + chunk_offset
    count = state_trace(path)
    for state in (set(count) & __output_states):
      for f_id in __tk_output[state]:
        term_freq[doc_id, f_id] += count[state]

  # Distribute the aggregated counts into buckets
  bucket_count = len(__buckets)
  for doc_id, f_id in term_freq:
    bucket_index = hash(f_id) % bucket_count
    count = term_freq[doc_id, f_id]
    item = ( f_id, doc_id, count )
    os.write(__buckets[bucket_index], marshal.dumps(item))

  for f in __buckets:
    os.close(f)

  return len(term_freq)

def setup_pass_ftc(cm, num_instances):
  global __cm, __num_instances
  __cm = cm
  __num_instances = num_instances

def pass_ftc(b_dir):
  """
  Take a bucket, form a feature map, compute the count of
  each feature in each class.
  @param b_dir path to the bucket directory
  @returns (read_count, f_ids, prod) 
  """
  global __cm, __num_instances

  terms = defaultdict(lambda : np.zeros((__num_instances,), dtype='int'))

  read_count = 0
  for path in os.listdir(b_dir):
    if path.endswith('.index'):
      for f_id, doc_id, count in unmarshal_iter(os.path.join(b_dir, path)):
        terms[f_id][doc_id] = count
        read_count += 1

  f_ids, f_vs = zip(*terms.items())
  fm = np.vstack(f_vs)
  prod = np.dot(fm, __cm)
  return read_count, f_ids, prod

def generate_cm(items, num_classes):
  """
  @param items (class id, path) pairs
  @param num_classes The number of classes present
  """
  num_instances = len(items)

  # Generate the class map
  cm = np.zeros((num_instances, num_classes), dtype='bool')
  for docid, (lang_id, path) in enumerate(items):
    cm[docid, lang_id] = True

  return cm

def learn_ftc(paths, tk_nextmove, tk_output, cm, temp_path, args):
  global b_dirs
  num_instances = len(paths)
  num_features = max( i for v in tk_output.values() for i in v) + 1

  # Generate the feature map
  nm_arr = mp.Array('i', tk_nextmove, lock=False)

  if args.jobs:
    chunksize = min(len(paths) / (args.jobs*2), args.chunksize)
  else:
    chunksize = min(len(paths) / (mp.cpu_count()*2), args.chunksize)

  # TODO: Set the output dir
  b_dirs = [ tempfile.mkdtemp(prefix="train-",suffix='-bucket', dir=temp_path) for i in range(args.buckets) ]

  output_states = set(tk_output)
  
  path_chunks = list(chunk(paths, chunksize))
  pass_tokenize_arg = zip(offsets(path_chunks), path_chunks)
  
  pass_tokenize_params = (nm_arr, output_states, tk_output, b_dirs) 
  with MapPool(args.jobs, setup_pass_tokenize, pass_tokenize_params) as f:
    pass_tokenize_out = f(pass_tokenize, pass_tokenize_arg)

  write_count = sum(pass_tokenize_out)
  logger.info("wrote a total of %d keys", write_count)

  # TODO: Report on the progress of this pass
  pass_ftc_params = (cm, num_instances)
  with MapPool(args.jobs, setup_pass_ftc, pass_ftc_params) as f:
    pass_ftc_out = f(pass_ftc, b_dirs)

  reads, ids, prods = zip(*pass_ftc_out)
  read_count = sum(reads)
  logger.info("read a total of %d keys (%d short)", read_count, write_count - read_count)

  # Re-order the weights into a single ndarray
  term_lang_counts = np.zeros((num_features, cm.shape[1]), dtype=int)
  term_lang_counts[np.concatenate(ids)] = np.vstack(prods)
  return term_lang_counts

@atexit.register
def cleanup():
  global b_dirs
  try:
    for d in b_dirs:
      shutil.rmtree(d)
  except NameError:
    # Failed before b_dirs is defined, nothing to clean
    pass

def main(args):
  if args.temp:
    temp_path = args.temp
  else:
    temp_path = os.path.join(args.model, 'buckets')

  if args.scanner:
    scanner_path = args.scanner
  else:
    scanner_path = os.path.join(args.model, 'PragerFeats.scanner')

  if args.output:
    output_path = args.output
  else:
    output_path = os.path.join(args.model, 'model')

  index_path = os.path.join(args.model, 'paths')
  lang_path = os.path.join(args.model, 'lang_index')

  # display paths
  logger.info("model path: %s", args.model)
  logger.info("temp path: %s", temp_path)
  logger.info("scanner path: %s", scanner_path)
  logger.info("output path: %s", output_path)

  # read list of training files
  with open(index_path) as f:
    reader = csv.reader(f)
    items = [ (l,p) for _,l,p in reader ]

  # read scanner
  with open(scanner_path) as f:
    tk_nextmove, tk_output, _ = cPickle.load(f)

  # read list of languages in order
  with open(lang_path) as f:
    reader = csv.reader(f)
    langs = zip(*reader)[0]
  

  paths = zip(*items)[1]
  cm = generate_cm(items, len(langs))
  ftc = learn_ftc(paths, tk_nextmove, tk_output, cm, temp_path, args)

  # use only feats that appear in at least one lang
  lang_freq = (ftc > 0).sum(axis=1) # num langs feat appears in
  logger.debug("{0} feats ({1} used)".format(len(lang_freq), (lang_freq>0).sum()))
  #ftc = ftc[lang_freq > 0] 
  #lang_freq = lang_freq[lang_freq > 0]
  with np.errstate(divide='ignore'):
    ilf = 1. / lang_freq
  ilf[np.isinf(ilf)] = 0 # set to 0 to effectively ignore these features

  # It is possible for documents to not contain -any- of the selected features.
  # It is also possible for all of the documents in a language to not contain any of the 
  # features, particularly because the selection is TF-based. We hence have to discard
  # languages for which we effectively have no training vectors.
  nz = ftc.sum(axis=0) > 0
  ftc = ftc[:,nz]

  unused = list(compress(langs,np.logical_not(nz)))
  if len(unused) > 0:
    logger.warning("{0} unused classes: {1}".format(len(unused), unused))

  # TF-IDF calculation with renormalization to unit vectors
  w = np.vstack([row / np.sqrt(row.dot(row)) for row in ftc.T * ilf])

  lprot = array.array('d')
  for dist in w.tolist():
    lprot.extend(dist)

  # ensure that langs is a list of Python strings
  # and concurrently remove the unused classes
  langs = map(str,compress(langs,nz))

  assert len(langs) == w.shape[0]

  # ilf needs to be a python array
  ilf = array.array('d', ilf)

  # output the model
  model = ilf, lprot, langs, tk_nextmove, tk_output
  string = base64.b64encode(bz2.compress(cPickle.dumps(model)))
  with open(output_path, 'w') as f:
    f.write(string)
  logger.info("wrote model to %s (%d bytes)", output_path, len(string))
