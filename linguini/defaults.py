# index.py
TRAIN_PROP = 1.0 # probability than any given document is selected
MIN_DOMAIN = 1 # minimum number of domains a language must be present in to be included

# tokenize.py
MAX_NGRAM_ORDER = 4 # largest order of n-grams to consider
TOP_DOC_FREQ = 15000 # number of tokens to consider for each order
NUM_BUCKETS = 64 # number of buckets to use in k-v pair generation
CHUNKSIZE = 50 # maximum size of chunk (number of files tokenized - less = less memory use)
SAMPLE_SIZE = 140 # size of sample in bytes for sampling-based tokenization

# featureselect.py
TOKENS_PER_ORDER = 15000 # number of tokens to consider for each order

# model.py
MAX_CHUNK_SIZE = 100 # maximum number of files to tokenize at once
