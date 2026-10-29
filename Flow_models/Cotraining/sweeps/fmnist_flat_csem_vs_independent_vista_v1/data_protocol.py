"""A fixed class-stratified 50k/10k split; test remains inaccessible to selection."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset,Subset
from torchvision import transforms
from pinned import stacked_two_block as old
from paper_io import load_prepared_pair
from protocol import atomic,object_sha

def split_indices(targets,seed=20261028,per_class=1000):
    labels=np.asarray(targets);rng=np.random.default_rng(seed);valid=[];train=[]
    for label in range(10):
        ids=np.flatnonzero(labels==label);ids=rng.permutation(ids)
        if len(ids)<=per_class:raise ValueError('Insufficient examples for stratified split')
        valid.extend(ids[:per_class].tolist());train.extend(ids[per_class:].tolist())
    return rng.permutation(train).tolist(),rng.permutation(valid).tolist()

def datasets(args,synthetic=False):
    if synthetic:
        g=torch.Generator().manual_seed(1700)
        data=TensorDataset(torch.rand(8,1,32,32,generator=g)*2-1,torch.arange(8)%10)
        return data,data,dict(synthetic=True,train_count=8,reference_count=8,hash='synthetic-smoke-only')
    tf=transforms.Compose([transforms.Pad(2),transforms.ToTensor(),transforms.Normalize((.5,),(.5,))])
    full,test=load_prepared_pair(old.ref.DATASET_INFO['FMNIST']['class'],args.data_root,'FMNIST',transform=tf)
    if args.split=='historical_full60k':
        if args.eval_split!='test':raise ValueError('Historical provenance uses the original test contract')
        return full,test,dict(protocol=args.split,hash='historical_full60k',train_count=len(full),reference_count=len(test))
    ti,vi=split_indices(full.targets,args.split_seed,args.validation_per_class)
    record=dict(protocol=args.split,split_seed=args.split_seed,train_ids=ti,validation_ids=vi,
                test_ids=list(range(len(test))))
    h=object_sha(record);dest=Path(args.output).parent.parent/'splits'/f'{h}.json'
    if not dest.exists():atomic(dest,record)
    train=Subset(full,ti);reference=Subset(full,vi) if args.eval_split=='validation' else test
    if args.eval_split not in ('validation','test'):raise ValueError('Unknown evaluation split')
    return train,reference,dict(protocol=args.split,hash=h,split_manifest=str(dest),train_count=len(train),
                               reference_count=len(reference),reference=args.eval_split)

def loaders(args,synthetic=False):
    tr,ref,meta=datasets(args,synthetic)
    # Dedicated data-order stream; workers only perform deterministic transforms.
    g=torch.Generator().manual_seed(args.seed+10001)
    return (DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=args.num_workers,drop_last=True,generator=g),
            DataLoader(ref,batch_size=args.eval_batch,shuffle=False,num_workers=args.num_workers),g,meta)
