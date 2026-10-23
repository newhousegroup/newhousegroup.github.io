// Inference primitives extracted from Labroom 2.7.1. No training UI or dataset.
function makeV(name,shape,scale=.02,zeros=false,ones=false){const t=zeros?tf.zeros(shape):ones?tf.ones(shape):tf.randomNormal(shape,0,scale,'float32');const v=tf.variable(t,true,name);t.dispose();return v}
function layerNorm(x,g,b){const mu=x.mean(-1,true),diff=x.sub(mu);return diff.mul(tf.rsqrt(diff.square().mean(-1,true).add(1e-5))).mul(g).add(b)}
function linear(x,w,b){const dims=x.shape.slice(0,-1);return tf.matMul(x.reshape([-1,x.shape[x.rank-1]]),w).add(b).reshape([...dims,w.shape[1]])}
function gelu(x){return x.mul(.5).mul(tf.tanh(x.add(x.pow(tf.scalar(3)).mul(.044715)).mul(Math.sqrt(2/Math.PI))).add(1))}
function dropout(x,rate){if(!rate)return x;return x.mul(tf.randomUniform(x.shape).greaterEqual(rate).toFloat()).mul(1/(1-rate))}
class TinyGPT{
 constructor(vocab,c){this.config={...c};this.vocab=vocab;const {dim:d,context:T,blocks:B}=c;this.emb=makeV('embedding',[vocab,d]);this.pos=makeV('position',[T,d]);this.blocks=[];
  for(let i=0;i<B;i++){const n='b'+i+'_';this.blocks.push({g1:makeV(n+'g1',[d],.02,false,true),b1:makeV(n+'b1',[d],.02,true),qkv:makeV(n+'qkv',[d,3*d]),qkvb:makeV(n+'qkvb',[3*d],.02,true),proj:makeV(n+'proj',[d,d]),projb:makeV(n+'projb',[d],.02,true),g2:makeV(n+'g2',[d],.02,false,true),b2:makeV(n+'b2',[d],.02,true),fc1:makeV(n+'fc1',[d,4*d]),fc1b:makeV(n+'fc1b',[4*d],.02,true),fc2:makeV(n+'fc2',[4*d,d]),fc2b:makeV(n+'fc2b',[d],.02,true)})}
  this.lastG=makeV('lastG',[d],.02,false,true);this.lastB=makeV('lastB',[d],.02,true);this.head=makeV('head',[d,vocab]);this.headB=makeV('headB',[vocab],.02,true);
  this.variables=[this.emb,this.pos,...this.blocks.flatMap(b=>Object.values(b)),this.lastG,this.lastB,this.head,this.headB];this.paramCount=this.variables.reduce((a,v)=>a+v.size,0);this.masks=new Map();
 }
 forward(ids,training=false,withCache=false){const {dim:d,heads:H,context:T,dropout:rate}=this.config;const [batch,len]=ids.shape;if(len>T)throw Error(`Sequence length ${len} exceeds context ${T}`);const dh=d/H;
  let x=tf.matMul(tf.oneHot(ids.reshape([-1]),this.vocab),this.emb).reshape([batch,len,d]).add(this.pos.slice([0,0],[len,d]).reshape([1,len,d]));const cached=[];
  for(const b of this.blocks){const a=layerNorm(x,b.g1,b.b1),qkv=linear(a,b.qkv,b.qkvb),[q0,k0,v0]=tf.split(qkv,3,-1);const re=t=>t.reshape([batch,len,H,dh]).transpose([0,2,1,3]);const q=re(q0),k=re(k0),v=re(v0);
   if(withCache)cached.push({k,v});let mask=this.masks.get(len);if(!mask){const vals=new Float32Array(len*len);for(let i=0;i<len;i++)for(let j=i+1;j<len;j++)vals[i*len+j]=-1e9;for(const old of this.masks.values())old.dispose();this.masks.clear();mask=tf.keep(tf.tensor(vals,[1,1,len,len]));this.masks.set(len,mask)}
   const scores=tf.matMul(q,k,false,true).mul(1/Math.sqrt(dh)).add(mask);const attention=tf.softmax(scores,-1);
   let att=linear(tf.matMul(attention,v).transpose([0,2,1,3]).reshape([batch,len,d]),b.proj,b.projb);
   x=x.add(training?dropout(att,rate):att);const n=layerNorm(x,b.g2,b.b2);let ff=linear(gelu(linear(n,b.fc1,b.fc1b)),b.fc2,b.fc2b);x=x.add(training?dropout(ff,rate):ff);
  }
  const logits=linear(layerNorm(x,this.lastG,this.lastB),this.head,this.headB);
  return withCache?{logits,cache:cached}:logits;
 }
 step(token,position,prev){const {dim:d,heads:H,context:T}=this.config;if(position>=T)throw Error('KV cache full; recompute the context window.');const dh=d/H;
  return tf.tidy(()=>{let x=tf.matMul(tf.oneHot(tf.tensor1d([token],'int32'),this.vocab),this.emb).reshape([1,1,d]).add(this.pos.slice([position,0],[1,d]).reshape([1,1,d]));const next=[];
   for(let i=0;i<this.blocks.length;i++){const b=this.blocks[i],old=prev?.[i];const qkv=linear(layerNorm(x,b.g1,b.b1),b.qkv,b.qkvb);const [q0,k0,v0]=tf.split(qkv,3,-1);const re=t=>t.reshape([1,1,H,dh]).transpose([0,2,1,3]);const q=re(q0),k=re(k0),v=re(v0);const keys=old?tf.concat([old.k,k],2):k,values=old?tf.concat([old.v,v],2):v;const att=tf.matMul(tf.softmax(tf.matMul(q,keys,false,true).mul(1/Math.sqrt(dh)),-1),values).transpose([0,2,1,3]).reshape([1,1,d]);x=x.add(linear(att,b.proj,b.projb));x=x.add(linear(gelu(linear(layerNorm(x,b.g2,b.b2),b.fc1,b.fc1b)),b.fc2,b.fc2b));next.push({k:keys,v:values})}
   return {logits:linear(layerNorm(x,this.lastG,this.lastB),this.head,this.headB).reshape([this.vocab]),cache:next};
  });
 }
 prefill(ids){return tf.tidy(()=>{const n=ids.length,t=tf.tensor2d(ids,[1,n],'int32'),o=this.forward(t,false,true);return {logits:o.logits.slice([0,n-1,0],[1,1,this.vocab]).reshape([this.vocab]),cache:o.cache}})}
 dispose(){this.variables.forEach(v=>v.dispose());this.masks.forEach(v=>v.dispose())}
}

let charset=[],tokMap=new Map(),tokenizerSpec={type:'bpe',vocab:256,merges:[]};
const encText=new TextEncoder(),decText=new TextDecoder();
function validateModelId(id){if(!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(id))throw Error('Model ID: 1–64 characters; letters, numbers, dots, underscores and hyphens only, starting with a letter or number.');return id}
function bpePieces(text){return text.match(/\p{L}+|\p{N}+|[^\p{L}\p{N}]+/gu)||[]}
function encodeBPE(text,spec){const ranks=new Map(spec.merges.map(([a,b],i)=>[a+','+b,i]));const output=[];for(const piece of bpePieces(text)){const bytes=encText.encode(piece);let ids=Array.from(bytes);while(ids.length>1){let rank=Infinity,index=-1;for(let i=0;i<ids.length-1;i++){const r=ranks.get(ids[i]+','+ids[i+1]);if(r!==undefined&&r<rank){rank=r;index=i}}if(index<0)break;ids.splice(index,2,256+rank)}output.push(...ids)}return output}
function rebuildTokenizer(spec){if(spec.type==='char'){charset=[...spec.charset];tokMap=new Map(charset.map((v,i)=>[v,i]));if(charset.length<2||charset.length>256)throw Error('Unsupported character vocabulary.')}else if(spec.type==='bpe'){if(!Array.isArray(spec.merges)||spec.vocab!==256+spec.merges.length||spec.vocab>4096)throw Error('Invalid BPE tokenizer.');charset=[];tokMap=new Map();}else throw Error('Unknown tokenizer type.') ;tokenizerSpec=structuredClone(spec)}
function tokensFor(text,strict=false){if(tokenizerSpec.type==='bpe')return {ids:encodeBPE(text,tokenizerSpec),unknown:0};let ids=[],unknown=0;const fallback=tokMap.get(' ')??0;for(const ch of text){const id=tokMap.get(ch);if(id!==undefined)ids.push(id);else if(strict)throw Error('New character '+JSON.stringify(ch)+' is not in this model’s vocabulary. Remove/replace it or create a new model.');else{ids.push(fallback);unknown++}}return {ids,unknown}}
function decodeTokens(ids){if(tokenizerSpec.type==='char')return ids.map(id=>charset[id]||'').join('');const cache=[];function bs(n){if(cache[n])return cache[n];if(n<256)return cache[n]=[n];const pair=tokenizerSpec.merges[n-256];if(!pair)throw Error('Invalid BPE token '+n);return cache[n]=[...bs(pair[0]),...bs(pair[1])]}let bytes=[];for(const id of ids)bytes.push(...bs(id));return decText.decode(Uint8Array.from(bytes))}
function disposeCache(kv){if(kv)for(const b of kv){b.k.dispose();b.v.dispose()}}
// Glyph 1x Plus: inference-only candidate selection. No weights, tokenizer, or checkpoint changes.
const PLUS_CHUNK=32,PLUS_TEMPERATURE=.75,PLUS_TOP_P=.9;
const PLUS_STOPWORDS=new Set('the a an and or but for nor so yet to of in on at by with from into over under as is are was were be been being it its this that these those i you he she we they my your his her their our me him them us not no do did does had has have would could should can may might will shall if then than when while where who what which how there here very just also all one two up down out about after before again only more most some any each every same such own other once upon time said says say'.split(' '));
function plusWords(s){return (s.toLowerCase().match(/[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu)||[]).filter(w=>w.length>2)}
function plusAnchors(s){const words=plusWords(s).filter(w=>!PLUS_STOPWORDS.has(w));const counts=new Map();for(const w of words.slice(-100))counts.set(w,(counts.get(w)||0)+1);return [...counts].sort((a,b)=>b[1]-a[1]).slice(0,16).map(x=>x[0])}
function plusTopPSample(logits,temperature,topP,ids,penalty,gram){
 const scores=Float64Array.from(logits);if(penalty>1&&ids.length>=gram-1){const prefix=ids.slice(-(gram-1));const start=Math.max(0,ids.length-64-gram);const count=new Map();for(let j=start;j+gram<=ids.length;j++){let matches=true;for(let k=0;k<gram-1;k++)if(ids[j+k]!==prefix[k]){matches=false;break}if(matches)count.set(ids[j+gram-1],(count.get(ids[j+gram-1])||0)+1)}for(const [id,n] of count)scores[id]-=Math.log(penalty)*Math.min(n,6)}
 let high=-Infinity;for(const v of scores)if(v>high)high=v;const entries=[];let total=0;for(let i=0;i<scores.length;i++){const p=Math.exp((scores[i]-high)/temperature);if(p>0&&Number.isFinite(p)){entries.push([i,p]);total+=p}}
 entries.sort((a,b)=>b[1]-a[1]);let mass=0,cut=0;for(let i=0;i<entries.length;i++){mass+=entries[i][1]/total;cut=i+1;if(mass>=topP)break}
 let eligible=0;for(let i=0;i<cut;i++)eligible+=entries[i][1];let roll=Math.random()*eligible;let chosen=entries[0][0];for(let i=0;i<cut;i++){roll-=entries[i][1];if(roll<=0){chosen=entries[i][0];break}}
 // The probability is from the original unpenalized softmax (for quality ranking).
 const rawHigh=Math.max(...logits);let rawTotal=0;for(let i=0;i<logits.length;i++)rawTotal+=Math.exp((logits[i]-rawHigh)/temperature);
 const logprob=(logits[chosen]-rawHigh)/temperature-Math.log(rawTotal);
 return {id:chosen,logprob};
}
// Lightweight, inspectable lexical guidance. No external model or network requests.
// The relevance score measures content-word coverage, including inflections and a small
// semantic-family lexicon; it is NOT a semantic understanding model.
const PLUS_FAMILIES=[
 'ship boat vessel sailor captain crew deck mast sail sea ocean wave shore coast harbor harbour port voyage journey travel storm wind rain',
 'house home room door window floor wall hall building mansion castle',
 'king queen prince princess throne crown kingdom royal palace',
 'forest woods tree branch leaf leaves garden flower field meadow',
 'fear horror terror dread afraid frightened scared danger peril',
 'death dead die died dying kill killed murder grave funeral',
 'love loved loving affection heart dear beloved wife husband marriage',
 'fight fought battle war soldier army sword weapon enemy attack',
 'walk walked walking run ran running road path street journey',
 'speak spoke spoken say said tell told talk voice whisper cry',
 'look looked see saw seen watch watched gaze stare sight eyes',
 'think thought know knew known mind remember memory believe',
 'night dark darkness moon stars evening shadow dawn morning daylight sun',
 'water river lake stream rain ocean sea wave wet',
 'fire flame burning burned heat smoke blaze',
 'happy joy joyful delight pleasure smile laugh',
 'sad sorrow grief misery mourn tears weep crying',
 'find found discover discovered search seek sought lost missing'
];
const PLUS_FAMILY_INDEX=new Map();for(let i=0;i<PLUS_FAMILIES.length;i++)for(const w of PLUS_FAMILIES[i].split(' ')){const set=PLUS_FAMILY_INDEX.get(w)||new Set();set.add(i);PLUS_FAMILY_INDEX.set(w,set)}
function plusStem(w){if(w.length>6&&w.endsWith('ing')){let v=w.slice(0,-3);if(v.length>2&&v.at(-1)===v.at(-2))v=v.slice(0,-1);return v}if(w.length>5&&w.endsWith('ies'))return w.slice(0,-3)+'y';if(w.length>5&&w.endsWith('ed'))return w.slice(0,-2);if(w.length>4&&w.endsWith('es'))return w.slice(0,-2);if(w.length>4&&w.endsWith('s'))return w.slice(0,-1);return w}
function plusRelated(a,b){if(a===b)return 1;const sa=plusStem(a),sb=plusStem(b);if(sa===sb||sa+'e'===sb||sb+'e'===sa)return .75;const aa=PLUS_FAMILY_INDEX.get(a)||PLUS_FAMILY_INDEX.get(sa),bb=PLUS_FAMILY_INDEX.get(b)||PLUS_FAMILY_INDEX.get(sb);if(aa&&bb)for(const k of aa)if(bb.has(k))return .4;return 0}
function plusKeyTerms(s,max=18){const words=plusWords(s).filter(w=>!PLUS_STOPWORDS.has(w)).slice(-140);const counts=new Map();for(const w of words)counts.set(w,(counts.get(w)||0)+1);return [...counts].sort((a,b)=>b[1]-a[1]).slice(0,max).map(([word,freq])=>({word,weight:1+Math.min(1.2,Math.log2(1+freq)*.35)+Math.min(1.2,word.length/12)}))}
function plusRelevance(terms,candidateWords){if(!terms.length||!candidateWords.length)return 0;let total=0,matched=0;for(const term of terms){total+=term.weight;let best=0;for(const w of candidateWords)best=Math.max(best,plusRelated(term.word,w));matched+=term.weight*best}return total?matched/total:0}
function plusScore(prompt,historyText,fragment,meanLogprob){
 const words=plusWords(fragment),meaningful=words.filter(w=>!PLUS_STOPWORDS.has(w));
 const anchor=plusRelevance(plusKeyTerms(prompt),meaningful);
 // Use prompt as continuity reference for the first segment (history is empty).
 const recent=historyText.trim()?historyText.slice(-650):prompt.slice(-650);
 const continuity=plusRelevance(plusKeyTerms(recent,12),meaningful);
 const novelty=words.length?new Set(words).size/words.length:0;
 const previous=plusWords((prompt+' '+historyText).slice(-1500)).slice(-60),all=previous.concat(words);let duplicateNgrams=0;
 for(const n of [2,3,4]){const seen=new Set();for(let i=0;i+n<=all.length;i++){const gram=all.slice(i,i+n).join(' ');if(seen.has(gram)&&i+n>previous.length)duplicateNgrams++;seen.add(gram)}}
 let runs=0;for(let i=1;i<words.length;i++)if(words[i]===words[i-1])runs++;
 const charRepeat=(fragment.match(/([!?.,;:\-])\1{2,}/g)||[]).length;
 const likelihoodPart=meanLogprob*.55;
 // Coverage rewards related ideas but is bounded to avoid keyword stuffing.
 const anchorPart=Math.min(.85,anchor*1.25),continuityPart=Math.min(.35,continuity*.65);
 const diversityPart=novelty*.18,repetitionPart=-(Math.min(2.5,duplicateNgrams*.42)+runs*.7+charRepeat*.35);
 const score=likelihoodPart+anchorPart+continuityPart+diversityPart+repetitionPart;
 return {score,anchor,continuity,novelty,duplicateNgrams,meanLogprob,likelihoodPart,anchorPart,continuityPart,diversityPart,repetitionPart};
}
function validateArchitecture(a){const c={blocks:Number(a.blocks),dim:Number(a.dim),context:Number(a.context),heads:Number(a.heads)||4,dropout:Number(a.dropout)||0,lr:Number(a.lr||a.learningRate)||.001};if(!Number.isInteger(c.blocks)||c.blocks<1||c.blocks>16||!Number.isInteger(c.dim)||c.dim<32||c.dim>768||!Number.isInteger(c.context)||c.context<32||c.context>1024||!Number.isInteger(c.heads)||c.dim%c.heads||c.lr<=0||c.lr>.05||c.dropout<0||c.dropout>=1)throw Error('Unsupported checkpoint architecture.');return c}
const SHA256_K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];;
class IncrementalSHA256 {
 constructor(){this.h=new Uint32Array([0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19]);this.block=new Uint8Array(64);this.words=new Uint32Array(64);this.used=0;this.total=0;}
 static get K(){return SHA256_K}

 compress(bytes,offset){const w=this.words,h=this.h,k=IncrementalSHA256.K;for(let i=0;i<16;i++){const j=offset+i*4;w[i]=((bytes[j]<<24)|(bytes[j+1]<<16)|(bytes[j+2]<<8)|bytes[j+3])>>>0}for(let i=16;i<64;i++){const x=w[i-15],y=w[i-2],s0=((x>>>7)|(x<<25))^((x>>>18)|(x<<14))^(x>>>3),s1=((y>>>17)|(y<<15))^((y>>>19)|(y<<13))^(y>>>10);w[i]=(w[i-16]+s0+w[i-7]+s1)>>>0}let [a,b,c,d,e,f,g,hh]=h;for(let i=0;i<64;i++){const s1=((e>>>6)|(e<<26))^((e>>>11)|(e<<21))^((e>>>25)|(e<<7)),ch=(e&f)^(~e&g),t1=(hh+s1+ch+k[i]+w[i])>>>0,s0=((a>>>2)|(a<<30))^((a>>>13)|(a<<19))^((a>>>22)|(a<<10)),maj=(a&b)^(a&c)^(b&c),t2=(s0+maj)>>>0;hh=g;g=f;f=e;e=(d+t1)>>>0;d=c;c=b;b=a;a=(t1+t2)>>>0}h[0]=(h[0]+a)>>>0;h[1]=(h[1]+b)>>>0;h[2]=(h[2]+c)>>>0;h[3]=(h[3]+d)>>>0;h[4]=(h[4]+e)>>>0;h[5]=(h[5]+f)>>>0;h[6]=(h[6]+g)>>>0;h[7]=(h[7]+hh)>>>0}
 update(data){this.total+=data.length;let i=0;if(this.used){const take=Math.min(64-this.used,data.length);this.block.set(data.subarray(0,take),this.used);this.used+=take;i=take;if(this.used===64){this.compress(this.block,0);this.used=0}}while(i+64<=data.length){this.compress(data,i);i+=64}if(i<data.length){this.block.set(data.subarray(i),0);this.used=data.length-i}return this}
 hex(){const bits=this.total*8;this.block[this.used++]=128;if(this.used>56){this.block.fill(0,this.used,64);this.compress(this.block,0);this.used=0}this.block.fill(0,this.used,56);const hi=Math.floor(bits/4294967296),lo=bits>>>0;const dv=new DataView(this.block.buffer);dv.setUint32(56,hi,false);dv.setUint32(60,lo,false);this.compress(this.block,0);return Array.from(this.h,x=>x.toString(16).padStart(8,'0')).join('')}
}
const importYield=()=>new Promise(resolve=>setTimeout(resolve,0));
async function inspectLRM(file){
 if(file.size<8)throw Error('File too short.');
 const prefix=new DataView(await file.slice(0,8).arrayBuffer());
 if(prefix.getUint8(0)!==76||prefix.getUint8(1)!==82||prefix.getUint8(2)!==77||prefix.getUint8(3)!==49)throw Error('Not LRM1.');
 const n=prefix.getUint32(4,true);if(n>file.size-8||n>50*1048576)throw Error('Invalid LRM header.');
 status('Reading checkpoint manifest…');await importYield();
 const meta=JSON.parse(decText.decode(await file.slice(8,8+n).arrayBuffer()));
 if(meta.formatVersion!==1||meta.endianness!=='little'||!Array.isArray(meta.tensors)||!Array.isArray(meta.tensors.filter(t=>t.category==='model')))throw Error('Unsupported LRM format.');
 if(!/^[0-9a-f]{64}$/i.test(meta.payloadSHA256||''))throw Error('Missing checkpoint SHA-256.');
 const start=8+n,expectedSize=file.size-start;let offset=0;
 for(const t of meta.tensors){
  if(!['model','optimizer'].includes(t.category)||t.dtype!=='float32'||!Number.isSafeInteger(t.offset)||t.offset!==offset||!Number.isSafeInteger(t.bytes)||t.bytes<0||t.bytes%4||t.bytes>expectedSize-offset||!Array.isArray(t.shape)||t.shape.some(d=>!Number.isSafeInteger(d)||d<1)||t.shape.reduce((a,b)=>a*b,1)!==t.bytes/4)throw Error('Invalid tensor manifest.');
  offset+=t.bytes;
 }
 if(offset!==expectedSize)throw Error('Checkpoint payload length mismatch.');
 return {meta,start};
}
async function verifyLRMStream(file,start,expected){
 const sha=new IncrementalSHA256(),chunkSize=256*1024,total=file.size-start;
 for(let off=0;off<total;off+=chunkSize){
  const bytes=new Uint8Array(await file.slice(start+off,start+Math.min(total,off+chunkSize)).arrayBuffer());
  sha.update(bytes);
  if((Math.floor(off/chunkSize)%8)===0){status(`Verifying checkpoint: ${Math.round(100*Math.min(total,off+chunkSize)/total)}%…`);await importYield()}
 }
 if(sha.hex().toLowerCase()!==expected.toLowerCase())throw Error('Checkpoint integrity failure: SHA-256 mismatch.');
}
