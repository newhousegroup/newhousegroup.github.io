// Optional integration test: node tests/checkpoint.test.cjs path/to/real.lrm
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const tf=require('../dist/vendor/tf.min.js');const messages=[];
const ctx=vm.createContext({tf,TextEncoder,TextDecoder,structuredClone,setTimeout,performance,Blob,console,importScripts:()=>{},postMessage:m=>messages.push(m)});
vm.runInContext(fs.readFileSync('dist/core.js','utf8')+'\n'+fs.readFileSync('dist/worker.js','utf8')+'\nglobalThis.api={load,generate,dispose:()=>model.dispose()};',ctx);
(async()=>{
 await tf.setBackend('cpu');await ctx.api.load(new Blob([fs.readFileSync(process.argv[2])]));
 const ready=messages.find(x=>x.type==='ready');assert.ok(ready);console.log('Loaded:',JSON.stringify(ready));messages.length=0;
 await ctx.api.generate('The captain looked out across the ocean.',{length:32,candidates:2,temperature:.75,repeat:1.2,gram:3,mode:'fast'});
 const chunk=messages.find(x=>x.type==='chunk');assert.equal(chunk.tokens,32);assert.ok(Number.isFinite(chunk.score.score));console.log('Actual continuation:',JSON.stringify(chunk.text));console.log('Seconds:',chunk.elapsed.toFixed(2));
 ctx.api.dispose();assert.equal(tf.memory().numTensors,0);console.log('PASS: real checkpoint validation, weight loading, 2-candidate Plus generation, complete tensor cleanup.');
})().catch(e=>{console.error(e);process.exitCode=1});
