/* All model tensors and inference live in this worker. */
importScripts('vendor/tf.min.js','core.js');
let model=null,cancelled=false,active=false;
const send=(type,data={})=>postMessage({type,...data});
const status=message=>send('status',{message});
const sameShape=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
async function backend(){
 for(const name of ['webgl','wasm','cpu'])try{
  if(name==='wasm'){
   importScripts('vendor/tf-backend-wasm.js');
   tf.wasm.setWasmPaths(new URL('vendor/',self.location.href).href);tf.env().set('WASM_HAS_MULTITHREAD_SUPPORT',false);
  }
  if(await tf.setBackend(name)){await tf.ready();const t=tf.tidy(()=>tf.matMul(tf.ones([2,2]),tf.ones([2,2])));await t.data();t.dispose();return name}
 }catch(e){status(name+' unavailable; trying fallback…')}
 throw Error('No supported TensorFlow.js backend.');
}
function expectedShapes(c,v){const d=c.dim,out=[['embedding',[v,d]],['position',[c.context,d]]];for(let i=0;i<c.blocks;i++){const n='b'+i+'_';out.push([n+'g1',[d]],[n+'b1',[d]],[n+'qkv',[d,3*d]],[n+'qkvb',[3*d]],[n+'proj',[d,d]],[n+'projb',[d]],[n+'g2',[d]],[n+'b2',[d]],[n+'fc1',[d,4*d]],[n+'fc1b',[4*d]],[n+'fc2',[4*d,d]],[n+'fc2b',[d]])}return out.concat([['lastG',[d]],['lastB',[d]],['head',[d,v]],['headB',[v]]])}
async function load(file){
 const {meta,start}=await inspectLRM(file),c=validateArchitecture(meta.architecture||{}),spec=meta.tokenizer;
 if(meta.version!==3||spec?.type!=='bpe')throw Error('Glyph Plus requires a version 3 BPE LRM checkpoint.');
 if(!Array.isArray(spec.merges)||spec.vocab!==256+spec.merges.length||spec.vocab>4096)throw Error('Invalid BPE vocabulary.');
 spec.merges.forEach((p,i)=>{if(!Array.isArray(p)||p.length!==2||p.some(x=>!Number.isInteger(x)||x<0||x>=256+i))throw Error('Invalid BPE merge.');});
 const weights=meta.tensors.filter(t=>t.category==='model'),expected=expectedShapes(c,spec.vocab);
 if(weights.length!==expected.length||weights.some((w,i)=>w.name!==expected[i][0]||!sameShape(w.shape,expected[i][1])))throw Error('Checkpoint architecture/tensor mismatch.');
 await verifyLRMStream(file,start,meta.payloadSHA256);
 model?.dispose();model=null;
 let next;
 try{
  next=new TinyGPT(spec.vocab,c);
  for(let i=0;i<weights.length;i++){
   const w=weights[i],bytes=await file.slice(start+w.offset,start+w.offset+w.bytes).arrayBuffer();
   const values=new Float32Array(w.bytes/4),view=new DataView(bytes);
   for(let j=0;j<values.length;j++){values[j]=view.getFloat32(j*4,true);if(!Number.isFinite(values[j]))throw Error('Non-finite model weight.');}
   const t=tf.tensor(values,w.shape);try{next.variables[i].assign(t)}finally{t.dispose()}
   status(`Loading weights ${i+1}/${weights.length}`);await importYield();
  }
  rebuildTokenizer(spec);model=next;
  send('ready',{name:meta.modelId||'glyph-1x',backend:tf.getBackend(),params:model.paramCount,context:c.context});
 }catch(e){next?.dispose();throw e}
}
async function candidate(base,count,o,onProgress){
 let cache=null,logits=null,used=0;const ids=base.slice(),generated=[],T=model.config.context;let logSum=0;
 const clear=()=>{logits?.dispose();logits=null;disposeCache(cache);cache=null;used=0};
 const prefill=window=>{clear();const r=model.prefill(window);logits=r.logits;cache=r.cache;used=window.length};
 try{
  prefill(ids.slice(-T));
  for(let i=0;i<count;i++){
   if(cancelled)return null;
   const scores=await logits.data();if(cancelled)return null;
   if(Array.from(scores).some(x=>!Number.isFinite(x)))throw Error('Non-finite inference logits; reload model or try another device.');
   const chosen=plusTopPSample(scores,o.temperature,.9,ids,o.repeat,o.gram);
   ids.push(chosen.id);generated.push(chosen.id);logSum+=chosen.logprob;
   if((i+1)%4===0||i+1===count)onProgress?.(i+1);
   if(i<count-1){if(used<T){const r=model.step(chosen.id,used,cache);logits.dispose();disposeCache(cache);logits=r.logits;cache=r.cache;used++}else prefill(ids.slice(-(o.mode==='exact'?T:Math.floor(T/2))))}
   // Yield every token so Stop messages are processed promptly, including on CPU.
   await importYield();
  }
  return {ids:generated,meanLogprob:logSum/count};
 }finally{clear()}
}
async function generate(prompt,o){
 if(!model)throw Error('Load a trained checkpoint first.');
 if(!Number.isInteger(o.length)||o.length<1||o.length>2048||![1,2,3,4].includes(o.candidates)||o.temperature<.1||o.temperature>2||o.repeat<1||o.repeat>3||![2,3,4].includes(o.gram)||!['exact','fast'].includes(o.mode))throw Error('Invalid generation settings.');
 cancelled=false;const history=tokensFor(prompt).ids;if(!history.length)history.push(32);const initial=history.length,begin=performance.now();let generated=0;
 for(let pos=0;pos<o.length&&!cancelled;pos+=PLUS_CHUNK){
  const count=Math.min(PLUS_CHUNK,o.length-pos),context=decodeTokens(history.slice(initial)),scored=[],groupStarted=performance.now();
  for(let j=0;j<o.candidates&&!cancelled;j++){
   status(`Segment ${1+Math.floor(pos/32)} · candidate ${j+1}/${o.candidates}`);
   const c=await candidate(history,count,o,token=>send('progress',{segment:1+Math.floor(pos/PLUS_CHUNK),candidate:j+1,candidates:o.candidates,token,count}));if(!c)break;
   const metrics=plusScore(prompt,context,decodeTokens(c.ids),c.meanLogprob);scored.push({...c,metrics});
  }
  if(cancelled)break;
  scored.sort((a,b)=>b.metrics.score-a.metrics.score);const winner=scored[0];history.push(...winner.ids);generated+=count;
  const nextCount=Math.min(PLUS_CHUNK,Math.max(0,o.length-generated));
  // Fire-and-forget: the next group never waits for display/animation callbacks.
  send('chunk',{text:decodeTokens(history.slice(initial)),tokens:generated,elapsed:(performance.now()-begin)/1000,score:winner.metrics,nextGroupMs:nextCount?(performance.now()-groupStarted)*nextCount/count:0});await importYield();
 }
 send('done',{cancelled,tokens:generated,elapsed:(performance.now()-begin)/1000,tensors:tf.memory().numTensors});
}
let initialized=false;
onmessage=async({data:d})=>{
 if(d.type==='stop'){cancelled=true;return}
 if(active)return;active=true;
 try{if(!initialized){await backend();initialized=true}if(d.type==='load')await load(d.file);else if(d.type==='generate')await generate(d.prompt,d.options)}
 catch(e){send('error',{message:e.message,ready:!!model})}finally{active=false}
};
