#!/usr/bin/env python3
"""Validate an LRM1 checkpoint and emit a stripped, sharded public model."""
import argparse, hashlib, json, struct
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint',type=Path)
p.add_argument('--out',type=Path,default=Path('dist/models'))
a=p.parse_args()
with a.checkpoint.open('rb') as f:
    if f.read(4)!=b'LRM1': raise SystemExit('Expected LRM1 checkpoint')
    n=struct.unpack('<I',f.read(4))[0]
    if n>50*1024*1024: raise SystemExit('Invalid header length')
    m=json.loads(f.read(n)); start=8+n
    if m.get('formatVersion')!=1 or m.get('endianness')!='little' or m.get('version')!=3 or m.get('tokenizer',{}).get('type')!='bpe': raise SystemExit('Expected v3 BPE LRM')
    sha=hashlib.sha256()
    while b:=f.read(1024*1024): sha.update(b)
    if sha.hexdigest()!=m['payloadSHA256'].lower(): raise SystemExit('Checksum mismatch')
    size=a.checkpoint.stat().st_size-start; offset=0; public=[]; payload=bytearray()
    for t in m['tensors']:
        if t['category'] not in ('model','optimizer') or t['dtype']!='float32' or t['offset']!=offset or any(type(d)!=int or d<1 for d in t['shape']): raise SystemExit('Invalid tensor descriptor')
        count=1
        for d in t['shape']: count*=d
        if t['bytes']!=count*4 or offset+t['bytes']>size: raise SystemExit('Invalid tensor size')
        offset+=t['bytes']
        if t['category']=='model':
            f.seek(start+t['offset']); b=f.read(t['bytes'])
            public.append(dict(t,offset=len(payload)));payload.extend(b)
    if offset!=size: raise SystemExit('Unexpected payload bytes')
# Strict allowlist: no corpus, validation text, optimizer or arbitrary metadata.
meta={k:m[k] for k in ('version','architecture','tokenizer','modelId','revision','steps') if k in m}
meta.update(format='Labroom Model',formatVersion=1,endianness='little',tensors=public,payloadSHA256=hashlib.sha256(payload).hexdigest())
header=json.dumps(meta,separators=(',',':'),ensure_ascii=False).encode()
blob=b'LRM1'+struct.pack('<I',len(header))+header+payload
version=hashlib.sha256(blob).hexdigest()[:16]; shard=4*1024*1024
# Versioned URLs allow immutable caching and atomic manifest changes.
a.out.mkdir(parents=True,exist_ok=True);files=[]
for i,at in enumerate(range(0,len(blob),shard)):
    name=f'glyph-{version}-{i:03d}.bin';(a.out/name).write_bytes(blob[at:at+shard]);files.append(name)
(a.out/'model.json').write_text(json.dumps({'version':version,'bytes':len(blob),'files':files},indent=2)+'\n')
print(f'Published {len(blob)/1048576:.2f} MiB in {len(files)} shard(s) to {a.out}')
