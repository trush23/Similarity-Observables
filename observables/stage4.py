"""Frozen ResNet controls. Gallery geometry and model weights remain fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version
from torchvision.models import ResNet50_Weights, resnet50
from .core import relative_distance
from .patterns import homometric
from .stage3 import (aligned_distances, cyclic_statistics, orbit, patch_array,
                     query_images, render, summarize, tile_library)

WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
RMS_SCALE = .25 / np.max(np.abs(tile_library('windowed wave')))


def unit(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 1e-12)


class Encoder:
    def __init__(self):
        torch.set_num_threads(1)
        self.model = resnet50(weights=WEIGHTS).eval()
        self.model.requires_grad_(False)
        transform = WEIGHTS.transforms()
        self.mean = torch.tensor(transform.mean)[None,:,None,None]
        self.std = torch.tensor(transform.std)[None,:,None,None]
        self.clipped = 0
        self.input_samples = 0

    @torch.inference_mode()
    def maps(self, images, size):
        x = .5 + RMS_SCALE*torch.tensor(np.asarray(images)[:,None],dtype=torch.float32)
        self.clipped += int(((x<0)|(x>1)).sum())
        self.input_samples += x.numel()
        x = x.clamp(0,1)
        if x.shape[-2:] != (size,size):
            x = F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
        x = (x-self.mean)/self.std
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        early = m.layer1(x)
        late = m.layer4(m.layer3(m.layer2(early)))
        return early,late

    def patch_embeddings(self, patches, symmetrize=True, batch_size=128):
        patches = np.asarray(patches)
        outputs = []
        for start in range(0,len(patches),batch_size):
            x = patches[start:start+batch_size]
            batch = np.concatenate([x,-x]) if symmetrize else x
            _,late = self.maps(batch,32)
            z = unit(late.mean((2,3)).numpy().astype(float))
            if symmetrize:
                z = unit(z[:len(x)]+z[len(x):])
            outputs.append(z)
        return np.concatenate(outputs)

    def whole_embeddings(self, strips, batch_size=16):
        fields = [[],[]]
        for start in range(0,len(strips),batch_size):
            square = np.tile(np.asarray(strips[start:start+batch_size]),(1,12,1))
            for out, feature_map in zip(fields,self.maps(square,96)):
                out.append(feature_map.mean((2,3)).numpy().astype(float))
        return [unit(np.concatenate(x)) for x in fields]


def reference_images():
    library = tile_library('windowed wave')
    return [render(x,library) for x in homometric()]


def mean_q_distance(a,b):
    """Frobenius distance between traceless second moments via small Gram matrices."""
    value = np.mean((a@a.T)**2)+np.mean((b@b.T)**2)-2*np.mean((a@b.T)**2)
    return np.sqrt(max(0.,value))


def patch_projection(features, reference_tiles):
    """Fixed axis determined by the two clean reference motifs."""
    left,right = reference_tiles
    direction = right-left
    if direction@direction < 1e-12:
        raise ValueError('Reference tiles are indistinguishable in the encoder')
    return 2*((features-(left+right)/2)@direction)/(direction@direction)


def q_scores(features,gallery):
    return np.array([mean_q_distance(features,g) for g in gallery])


def top1(distance,label):
    tied = np.isclose(distance,distance.min(),atol=1e-7,rtol=0)
    return float(tied[label]/tied.sum())


def evaluate(encoder, keep=16):
    if not 1 <= keep <= 48:
        raise ValueError('Use 1–48 Stage 3 trajectories per class')
    gallery_images = reference_images()
    reference = encoder.patch_embeddings(tile_library('windowed wave'))
    gallery = [reference[x.astype(int)] for x in homometric()]
    gallery_proj = [patch_projection(g,reference) for g in gallery]
    signatures = [cyclic_statistics(g) for g in gallery_proj]
    full_orbit = np.array([orbit(g) for g in gallery])
    phase_gallery = [np.roll(im,8*k,axis=1) for im in gallery_images for k in range(12)]
    whole = [z.reshape(2,12,-1) for z in encoder.whole_embeddings(phase_gallery)]
    records = [(key,img) for key,img in query_images() if key[1]<keep]
    images = [img for _,img in records]
    patches = np.stack([patch_array(im).reshape(12,8,8) for im in images])
    isolated = encoder.patch_embeddings(patches.reshape(-1,8,8)).reshape(len(records),12,-1)
    whole_queries = encoder.whole_embeddings(images)
    rows = []
    for i,((key,_),features) in enumerate(zip(records,isolated)):
        label = key[0]
        q = patch_projection(features,reference)
        pair,triple = cyclic_statistics(q)
        distances = {
            'isolated mean':np.array([relative_distance(features.mean(0),g.mean(0)) for g in gallery]),
            'isolated Q':q_scores(features,gallery),
            'isolated pair':np.array([relative_distance(pair,s[0]) for s in signatures]),
            'isolated triple':np.array([relative_distance(triple,s[1]) for s in signatures]),
            'isolated alignment':aligned_distances(features,full_orbit),
        }
        for name,references,query in zip(('early GAP','late GAP'),whole,whole_queries):
            all_distances = 1-references@query[i]
            distances[name] = all_distances[:,0]
            distances[name+' · phase bank'] = all_distances.min(1)
        for method,distance in distances.items():
            rows.append((*key,method,top1(distance,label),float(distance[1-label]-distance[label])))
    return pd.DataFrame(rows,columns=['label','trajectory','motif','condition','method','top1','margin'])


def run(output, keep=16):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    encoder = Encoder()
    rows = evaluate(encoder,keep)
    summary = summarize(rows)
    rows.to_csv(output/'retrieval_queries.csv',index=False)
    summary.to_csv(output/'retrieval_summary.csv',index=False)
    checkpoint = Path(torch.hub.get_dir())/'checkpoints'/WEIGHTS.url.rsplit('/',1)[-1]
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    info = dict(weights=WEIGHTS.name,weights_url=WEIGHTS.url,sha256=digest,
                torch=torch.__version__,torchvision=torchvision_version,
                per_class_trajectories=keep,stage3_seed=1507,
                input_scale=float(RMS_SCALE),fraction_clipped=encoder.clipped/encoder.input_samples)
    (output/'manifest.json').write_text(json.dumps(info,indent=2))
    return rows,summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keep',type=int,default=16)
    args = parser.parse_args()
    _,summary = run(Path(__file__).resolve().parents[1]/'results'/'stage4',args.keep)
    print(summary.pivot(index=['motif','condition'],columns='method',values='top1').round(3).to_string())
