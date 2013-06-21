"""
Command-line interface for linguini.py implementation.

Marco Lui, April 2013
"""

import argparse
import logging
import sys

from defaults import TRAIN_PROP, MIN_DOMAIN, MAX_CHUNK_SIZE
from defaults import MAX_NGRAM_ORDER, TOP_DOC_FREQ, NUM_BUCKETS, CHUNKSIZE, SAMPLE_SIZE
from index import main as c_index
from tokenize import main as c_tokenize
from featureselect import main as c_featsel
from scanner import main as c_scanner
from model import main as c_model
from detect import detect as c_detect, score as c_score

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-v","--verbose", action='store_true',
      help="produce verbose output")
  parser.add_argument("-j","--jobs", type=int, metavar='N', 
      help="spawn N processes (set to 1 for no paralleization)")
  parser.add_argument("-t", "--temp", metavar='TEMP_DIR', 
      help="store temporary files in TEMP_DIR")

  subp = parser.add_subparsers(help="available commands")

  ############
  # Indexing #
  ############
  index_p = subp.add_parser('index', help = "index a target corpus")
  index_p.set_defaults(func=c_index)

  index_p.add_argument("-p","--proportion", type=float, default=TRAIN_PROP,
      help="proportion of training data to use" )
  index_p.add_argument("-m","--model", help="save output to MODEL_DIR", metavar="MODEL_DIR")
  index_p.add_argument("-d","--domain", metavar="DOMAIN", action='append',
      help="use DOMAIN - can be specified multiple times (uses all domains found if not specified)")
  index_p.add_argument("-l","--lang", metavar="LANG", action='append',
      help="use LANG - can be specified multiple times (uses all langs found if not specified)")
  index_p.add_argument("--min_domain", type=int, default=MIN_DOMAIN,
      help="minimum number of domains a language must be present in" )
  index_p.add_argument("corpus", help="read corpus from CORPUS (interpret file as list of paths)", metavar="CORPUS")

  ################
  # Tokenization #
  ################
  tk_p = subp.add_parser('tokenize', help = "tokenize an indexed corpus")
  tk_p.set_defaults(func=c_tokenize)

  tk_p.add_argument("--buckets", type=int, metavar='N', default=NUM_BUCKETS,
      help="distribute features into N buckets")
  tk_p.add_argument("--chunksize", type=int, default=CHUNKSIZE,
      help="max chunk size (number of files to tokenize at a time - smaller should reduce memory use)")
  tk_p.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")

  tk_g = tk_p.add_argument_group("tokenization")
  tk_type = tk_g.add_mutually_exclusive_group(required=True)
  tk_type.add_argument("--ngram", action='store_true',
      help="use mixed character n-gram tokenization")
  tk_type.add_argument("--prager", action='store_true', 
      help="use the tokenization described in Prager (1999)")
  tk_type.add_argument("--scanner", metavar='SCANNER', 
      help="use SCANNER for tokenizing")

  tk_g.add_argument("--min_order", type=int, default=1,
      help="minimum n-gram order (for ngram tokenization)")
  tk_g.add_argument("--max_order", type=int, default=4,
      help="maximum n-gram order (for ngram tokenization)")
  tk_g.add_argument("--order", type=int, default=4,
      help="n-gram order (for Prager tokenization)")
  tk_g.add_argument("--words", action="store_true",
      help="include words (for Prager tokenization)")

  group = tk_g.add_argument_group('sampling')
  group.add_argument("--sample_size", type=int, default=SAMPLE_SIZE,
      help="size of sample for sampling-based tokenization")
  group.add_argument("--sample_count", type=int, default=None,
      help="number of samples for sampling-based tokenization")

  #####################
  # Feature Selection #
  #####################
  fs_p = subp.add_parser('featsel', help = "perform feature selection")
  fs_p.set_defaults(func=c_featsel)

  fs_p.add_argument("-f","--features", metavar='FEATURE_FILE', 
      help="output features to FEATURE_FILE")

  fs_type = fs_p.add_mutually_exclusive_group(required=True)
  fs_type.add_argument("-k", type=float,
      help="K value for Prager-style feature selection")
  fs_type.add_argument("-c", "--count", type=int,
      help="Count for top-N feature selection using Prager-style TFILF metric")

  fs_p.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")

  #################
  # Token Scanner #
  #################
  sc_p = subp.add_parser('scanner', help = "build document scanner used to tokenize test documents")
  sc_p.set_defaults(func=c_scanner)

  sc_p.add_argument("-o","--output", help="output scanner to OUTFILE", metavar="OUTFILE")
  sc_p.add_argument("input", metavar="INPUT", help="build a scanner for INPUT. If input is a directory, read INPUT/PragerFeats")

  ###################
  # Model Generator #
  ###################
  mod_p = subp.add_parser('model', help = "build model to be used in linguini classifier")
  mod_p.set_defaults(func=c_model)

  mod_p.add_argument("-s", "--scanner", metavar='SCANNER', help="use SCANNER for feature counting")
  mod_p.add_argument("-o", "--output", metavar='OUTPUT', help="output model to OUTPUT")
  mod_p.add_argument("--chunksize", type=int, help='maximum chunk size (number of files)', default=MAX_CHUNK_SIZE)
  mod_p.add_argument("--buckets", type=int, metavar='N', help="distribute features into N buckets", default=NUM_BUCKETS)
  mod_p.add_argument("model", metavar='MODEL_DIR', help="read index and produce output in MODEL_DIR")

  #####################
  # Language Detector #
  #####################
  det_p = subp.add_parser('detect', help = "apply linguini")
  det_p.set_defaults(func=c_detect)

  det_p.add_argument("-m", "--model", help="read model from")
  det_p.add_argument("-t", "--topn", type=int, default=5, help ="consider the top N languages for multilinguality")
  det_p.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout, metavar="OUTFILE", help ="write output to OUTFILE(csv format)")
  det_p.add_argument("--max_order", type=int, default=1, help ="maximum order of documents to detect (1 to 3)")
  det_p.add_argument("--line", action="store_true", help ="apply langid line-by-line")
  det_p.add_argument('docs', metavar='FILE', help='files to process (interactive mode if blank)', nargs='*')

  ######################
  # Cosine Calculation #
  ######################
  cos_p = subp.add_parser('score', help = "calculate per-language scores for documents")
  cos_p.set_defaults(func=c_score)

  cos_p.add_argument("-m", "--model", help="read model from")
  cos_p.add_argument("-o", "--output", type=argparse.FileType('w'), default=sys.stdout, metavar="OUTFILE", help ="write output to OUTFILE(csv format)")
  cos_p.add_argument('docs', metavar='FILE', help='files to process (interactive mode if blank)', nargs='+')

  args = parser.parse_args()



  logging.basicConfig(level = logging.DEBUG if args.verbose else logging.WARNING)

  args.func(args)
