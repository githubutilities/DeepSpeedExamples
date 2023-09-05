import os
import sys
import json
import pickle
import logging
import argparse
import base64

logger = logging.getLogger(__name__)

def encode_world_info(world_info):
    world_info_json = json.dumps(world_info).encode('utf-8')
    world_info_base64 = base64.urlsafe_b64encode(world_info_json).decode('utf-8')
    return world_info_base64

def hostfile2world_info(fn):
    ret = {}
    with open(fn) as f:
        for l in f:
            ip, slot = l.split()
            ret[ip] = list(range( int(slot.lstrip('slots=')) ))
    return encode_world_info(ret)

def _main():
    parser = argparse.ArgumentParser(description='hostfile2world_info')
    parser.add_argument('-i', '--input_fn', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    print(hostfile2world_info(args.input_fn))

if __name__ == '__main__':
    _main()

