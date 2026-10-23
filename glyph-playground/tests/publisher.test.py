import hashlib,json,struct,subprocess,tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);payload=struct.pack('<fff',1,2,99)
    meta=dict(version=3,formatVersion=1,endianness='little',architecture={},tokenizer={'type':'bpe','vocab':256,'merges':[]},corpus='PRIVATE CORPUS',validationText='PRIVATE VALIDATION',optimizer=['PRIVATE'],tensors=[dict(category='model',name='embedding',dtype='float32',shape=[2],offset=0,bytes=8),dict(category='optimizer',name='adam',dtype='float32',shape=[1],offset=8,bytes=4)],payloadSHA256=hashlib.sha256(payload).hexdigest())
    header=json.dumps(meta).encode();source=root/'test.lrm';source.write_bytes(b'LRM1'+struct.pack('<I',len(header))+header+payload)
    subprocess.run(['python','tools/publish_model.py',str(source),'--out',str(root/'out')],check=True)
    manifest=json.loads((root/'out/model.json').read_text());blob=b''.join((root/'out'/p).read_bytes() for p in manifest['files']);n=struct.unpack('<I',blob[4:8])[0];public=json.loads(blob[8:8+n]);weights=blob[8+n:]
    assert len(blob)==manifest['bytes']
    assert not {'corpus','validationText','optimizer'} & public.keys()
    assert b'PRIVATE' not in blob
    assert weights==payload[:8]
    assert hashlib.sha256(weights).hexdigest()==public['payloadSHA256']
    source.write_bytes(source.read_bytes()[:-1]+b'\x01')
    result=subprocess.run(['python','tools/publish_model.py',str(source),'--out',str(root/'bad')],capture_output=True)
    assert result.returncode!=0
print('PASS: publisher strips private fields and optimizer tensors; preserves weights; rejects corruption.')
