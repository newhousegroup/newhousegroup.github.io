// Test real TF.js kernels with synthetic weights only; never published as a model.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const tf=require('../dist/vendor/tf.min.js');
const messages=[];
const context=vm.createContext({tf,TextEncoder,TextDecoder,structuredClone,setTimeout,performance,Blob,console,importScripts:()=>{},postMessage:m=>messages.push(m)});
vm.runInContext(fs.readFileSync('dist/core.js','utf8')+'\n'+fs.readFileSync('dist/worker.js','utf8')+`\n globalThis.test={setup:()=>{model=new TinyGPT(256,{blocks:2,dim:32,heads:4,context:32,dropout:0,lr:.001});rebuildTokenizer({type:'bpe',vocab:256,merges:[]})},get:()=>model,generate,candidate,cancel:()=>{cancelled=true},dispose:()=>model.dispose()};`,context);
(async()=>{
 await tf.setBackend('cpu');context.test.setup();const model=context.test.get(),ids=[32,70,111,111];
 const full=model.prefill(ids);let cache=null,last=null;
 for(let i=0;i<ids.length;i++){const r=model.step(ids[i],i,cache);last?.dispose();if(cache)for(const b of cache){b.k.dispose();b.v.dispose()}last=r.logits;cache=r.cache}
 const a=await full.logits.data(),b=await last.data();let diff=0;for(let i=0;i<a.length;i++)diff=Math.max(diff,Math.abs(a[i]-b[i]));assert.ok(diff<.0001,`KV difference ${diff}`);
 full.logits.dispose();last.dispose();for(const list of [full.cache,cache])for(const b of list){b.k.dispose();b.v.dispose()}
 const options={length:65,candidates:2,temperature:.75,repeat:1.2,gram:3,mode:'fast'};
 await context.test.generate('Foo',options);assert.deepEqual(messages.filter(x=>x.type==='chunk').map(x=>x.tokens),[32,64,65]);
 // No UI consumer or reveal callback runs during this test. The worker must
 // still progress into later groups and finish independently of presentation.
 const first=messages.findIndex(x=>x.type==='chunk');const second=messages.findIndex((x,i)=>i>first&&x.type==='chunk');
 assert.ok(messages.slice(first+1,second).some(x=>x.type==='progress'&&x.segment===2&&x.token===4));
 assert.ok(messages[first].nextGroupMs>0);assert.equal(messages.filter(x=>x.type==='chunk').at(-1).nextGroupMs,0);
 const tensors=tf.memory().numTensors;messages.length=0;
 await context.test.generate('Foo',{...options,length:35,mode:'exact'});assert.equal(tf.memory().numTensors,tensors,'No tensor growth after exact overflow');
 messages.length=0;const job=context.test.generate('Foo',{...options,length:1024});setTimeout(()=>context.test.cancel(),5);await job;
 assert.equal(messages.at(-1).cancelled,true);assert.equal(tf.memory().numTensors,tensors,'No tensor growth after cancellation');
 context.test.dispose();assert.equal(tf.memory().numTensors,0);
 console.log('PASS: real TF.js CPU prefill/step parity, multi-candidate segmented generation, both context overflow modes, cancellation, tensor cleanup.');
})().catch(e=>{console.error(e);process.exitCode=1});
