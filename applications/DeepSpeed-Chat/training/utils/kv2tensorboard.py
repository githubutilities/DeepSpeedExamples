import os
import sys
import json
import pickle
import logging
import argparse
import collections

from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def _main():
    parser = argparse.ArgumentParser(description='kv2tensorboard')
    parser.add_argument('--key_prefix', type=str, default='')
    parser.add_argument('-o', '--output_dir', type=str, default='./runs/')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    writer = SummaryWriter(logdir=args.output_dir)

    key2idx = collections.defaultdict(lambda: 0)
    def record(key, value, fn=float):
        try:
            print(key, value)
            value = fn(value)
            key2idx[key] += 1
            writer.add_scalar(key, value, key2idx[key])
        except:
            return None
    for idx, l in enumerate(sys.stdin):
        #json
        for line in l.split('\r'):
            try:
                o = json.loads(line.strip().replace("'", "\""))
                for k in o.keys():
                    key = args.key_prefix + 'json_' + k
                    record(k, o[k])
            except Exception as e:
                print(str(e))
                pass
        #tsv
        if True and len(l) <= 50:
            l_sp = l.replace(':', '\t').rstrip('\n').split('\t')
            if len(l_sp) < 2:
                continue
            key = l_sp[0].replace(' ', '_')
            key = args.key_prefix + key
            record(key, l_sp[1])

if __name__ == '__main__':
    _main()

