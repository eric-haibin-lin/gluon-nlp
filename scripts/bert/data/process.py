from multiprocessing import Pool
import argparse
import glob
import os
import io
import time
import logging
import re

parser = argparse.ArgumentParser(description='BERT tokenizer')
parser.add_argument('--data', type=str, default='~/book-corpus-large/*.txt',
                    help='Input files. Default is "~/book-corpus-large/*.txt"')
parser.add_argument('--out-dir', type=str, default=None)
parser.add_argument('--nworker', type=int, default=72,
                    help='Number of workers for parallel processing.')

args = parser.parse_args()

args.data = str.replace(args.data, ',', '')
input_files = sorted(glob.glob(os.path.expanduser(args.data)))
num_files = len(input_files)
num_workers = args.nworker
logging.basicConfig(level=logging.INFO)
logging.info("Number of input files to process = %d"%(num_files))

os.makedirs(args.out_dir, exist_ok=True)

# Remove html tags utility
html_pattern = re.compile('<.*?>')

# Characters to ignore in data processing
IGNORE_CHARACTERS = '-=()/%"*[];,&@#:\'+”'

# Newline characters to split the sentence into 2 sentences
NEWLINE_CHARACTERS = '.?'

# Separator for final output data
OUTPUT_DATA_SEPARATOR = '\n'

# Minimum Sentence length for data preparation
# The value is chosen to avoid one word and empty sentences
MIN_SENTENCE_LENGTH = 5

def isNotBlank(text):
    if text and text.strip() and len(text)>MIN_SENTENCE_LENGTH:
        #text is not None AND text is not empty or blank or text length lesser than threshold
        return True
    #text is None OR text is empty or blank
    return False

def removeEnclosingQuotes(text):
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]

def main(args_dict):
    files = []
    indices = [int(x) for x in str(args_dict['indices']).split(',')]
    for input_pattern in args_dict['input_file']:
        files.extend(glob.glob(os.path.expanduser(input_pattern)))
    for f in files:
        processData = PrepareTSVData(f, indices, args_dict['is_header_present'], args_dict['remove_punctuations'])
        if args_dict['prepare_sequences']:
            data = processData.process_dataset_with_sequences()
            data_joined = OUTPUT_DATA_SEPARATOR.join(data) + OUTPUT_DATA_SEPARATOR
        else:
            data = processData.process_dataset_without_sequences()
            record_separator = OUTPUT_DATA_SEPARATOR + OUTPUT_DATA_SEPARATOR
            data_joined = record_separator.join(data)
        logging.info("Writing file {}".format(os.path.basename(f)))
        get_file_utils(args_dict['output_dir']).write_to_file(args_dict['output_dir'] + os.path.basename(f), data_joined, "w+")

def f(input_file):
    with io.open(input_file, 'r', encoding="utf-8") as fin:
        curr_count = 0
        num_products = 0
        bins = {}
        out_file_name = input_file.split('/')[-1]
        out_file_name = os.path.join(args.out_dir, out_file_name)
        with io.open(out_file_name, 'w', encoding="utf-8") as fout:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    # matched_count += 1 # len(re.findall(html_pattern, line))
                    if curr_count not in bins:
                        bins[curr_count] = 1
                    else:
                        bins[curr_count] += 1
                    curr_count = 0
                    num_products += 1
                else:
                    curr_count += len(line.split(' '))
                    # line = re.sub(html_pattern, '', line)
                    # fout.write(line + '\n')
        bins_key = sorted(bins.keys())
        acc = 0
        for b in bins_key:
            ratio = bins[b] * 100.0 / num_products
            acc += ratio
            print(b, bins[b], ratio, acc)
        #print(matched_count * 1.0 / total_count)

if __name__ == '__main__':
    tic = time.time()
    p = Pool(num_workers)
    p.map(f, input_files)
    toc = time.time()
    logging.info("Processed %s in %.2f sec"%(args.data, toc-tic))
