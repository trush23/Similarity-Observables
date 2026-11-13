"""Source-disjoint Fashion-MNIST transfer with the frozen Stage 8 protocol."""
import argparse
import gzip
import hashlib
import json
import os
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from .stage6 import CONDITIONS, view
from .stage8 import replication_galleries, run_classical, run_frozen, summarize

SEED = 9327
CHECKSUMS = {
    't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
    't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310',
}


def load_fashion(path=None):
    folder = Path(path or os.environ.get('FASHION_MNIST_DIR',Path.home()/'.keras/datasets/fashion-mnist'))
    arrays = {}
    for name,checksum in CHECKSUMS.items():
        payload = (folder/name).read_bytes()
        if hashlib.md5(payload).hexdigest()!=checksum:
            raise ValueError(f'Unexpected Fashion-MNIST digest for {name}')
        raw = gzip.decompress(payload)
        if 'images' in name:
            if struct.unpack('>4I',raw[:16])!=(2051,10000,28,28):
                raise ValueError('Unexpected Fashion-MNIST image header')
            arrays['x_test'] = np.frombuffer(raw,np.uint8,offset=16).reshape(10000,28,28)
        else:
            if struct.unpack('>2I',raw[:8])!=(2049,10000):
                raise ValueError('Unexpected Fashion-MNIST label header')
            arrays['y_test'] = np.frombuffer(raw,np.uint8,offset=8)
    return arrays


def observations(data,excluded):
    rng = np.random.default_rng(SEED+1)
    for panel,group,ids,images,labels in replication_galleries(data,excluded,seed=SEED,panel='fashion'):
        queries = [view(x,condition,rng) for x in images for condition in CONDITIONS]
        yield panel,group,ids,images,labels,queries


def cross_dataset_contrast(fashion,mnist,left,right,endpoint,seed=9361):
    """Independent-gallery bootstrap of the difference between paired method effects."""
    rng = np.random.default_rng(seed)
    deltas,draws = [],[]
    for panel in (fashion,mnist):
        means = panel.groupby(['gallery','method'])[endpoint].mean().unstack('method')
        delta = (means[left]-means[right]).to_numpy()
        deltas.append(delta.mean())
        draws.append(rng.choice(delta,(4000,len(delta)),replace=True).mean(1))
    return deltas[0]-deltas[1],*np.quantile(draws[0]-draws[1],[.025,.975])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['classical','frozen','summarize'])
    parser.add_argument('--data',type=Path)
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1]/'results'/'stage9'
    exclusions = output/'excluded_test_ids.csv'
    excluded = set(pd.read_csv(exclusions).source)
    data = load_fashion(args.data)
    if args.part=='classical':
        run_classical(data,excluded,output,generate=observations)
    elif args.part=='frozen':
        run_frozen(data,excluded,output,generate=observations)
    else:
        decisions,summary = summarize(None,output,data,shortlist_seed=SEED)
        ids = pd.read_csv(output/'membership.csv').source
        assert ids.is_unique and len(ids)==480 and set(ids).isdisjoint(excluded)
        actual = {data['x_test'][i].tobytes() for i in ids}
        assert len(actual)==480 and actual.isdisjoint(data['x_test'][i].tobytes() for i in excluded)
        metadata = dict(dataset='Fashion-MNIST official test',seed=SEED,checksums_md5=CHECKSUMS,
                        exclusions_sha256=hashlib.sha256(exclusions.read_bytes()).hexdigest(),
                        excluded=len(excluded),galleries=12,sources=480,methods=27)
        (output/'manifest.json').write_text(json.dumps(metadata,indent=2))
        print(summary.query("gallery_type=='full'").round(4).to_string(index=False))
