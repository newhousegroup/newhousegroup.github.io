const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const source=fs.readFileSync('dist/core.js','utf8');
const ctx=vm.createContext({TextEncoder,TextDecoder,structuredClone,setTimeout,console,Blob,status:()=>{}});
vm.runInContext(source+`\n globalThis.api={encodeBPE,decodeTokens,rebuildTokenizer,plusScore,plusTopPSample,IncrementalSHA256,inspectLRM,verifyLRMStream};`,ctx);
const a=ctx.api;
for(const text of ['', 'hello world', 'สวัสดีครับ 🌏', 'a\n\tb']){
 a.rebuildTokenizer({type:'bpe',vocab:258,merges:[[104,101],[256,108]]});
 assert.equal(a.decodeTokens(a.encodeBPE(text,{merges:[[104,101],[256,108]]})),text);
}
assert.deepEqual(Array.from(a.encodeBPE('hel',{merges:[[104,101],[256,108]]})),[257]);
for(const n of [0,1,55,56,63,64,65,1000,524289]){
 const bytes=crypto.randomBytes(n),h=new a.IncrementalSHA256();for(let i=0;i<n;i+=37)h.update(bytes.subarray(i,i+37));assert.equal(h.hex(),crypto.createHash('sha256').update(bytes).digest('hex'));
}
const related=a.plusScore('The captain sailed his ship across the ocean.','','The sailor watched the sea.',-2);
assert.ok(related.anchor>0);assert.ok(related.continuity>0);
const repetitive=a.plusScore('the road','','walk walk walk walk walk walk',-2);
assert.ok(repetitive.repetitionPart<0);
const chosen=a.plusTopPSample([0,100,-100],.75,.9,[],1.2,3);assert.equal(chosen.id,1);assert.ok(Number.isFinite(chosen.logprob));
(async()=>{
 const bytes=Buffer.from(new Float32Array([1,2]).buffer),meta={formatVersion:1,endianness:'little',tensors:[{category:'model',dtype:'float32',name:'test',shape:[2],offset:0,bytes:8}],payloadSHA256:crypto.createHash('sha256').update(bytes).digest('hex')},header=Buffer.from(JSON.stringify(meta)),prefix=Buffer.alloc(8);prefix.write('LRM1');prefix.writeUInt32LE(header.length,4);const file=new Blob([prefix,header,bytes]);const inspected=await a.inspectLRM(file);await a.verifyLRMStream(file,inspected.start,meta.payloadSHA256);await assert.rejects(a.verifyLRMStream(file,inspected.start,'0'.repeat(64)),/integrity/);
 console.log('PASS: BPE Unicode roundtrips, ranked merges, incremental SHA-256 boundaries, lexical scores, nucleus sampling, LRM manifest and corruption detection.');
})().catch(e=>{console.error(e);process.exitCode=1});
