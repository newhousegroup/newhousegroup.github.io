const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const WordReveal=require('../dist/reveal.js');
for(const text of ['Hello, world!\nAnother idea.','สวัสดีครับ วันนี้เป็นอย่างไร','🌏 a\n\t b','   ','']){
 const r=new WordReveal();r.setTarget(text);let n=0,last='';while(r.pending){r.step();assert.ok(r.visible.length>last.length);assert.ok(text.startsWith(r.visible));last=r.visible;assert.ok(++n<100)}assert.equal(r.visible,text);
}
const r=new WordReveal();r.setTarget('one two');assert.equal(r.step(),'one ');r.setTarget('one two three');while(r.pending)r.step();assert.equal(r.visible,'one two three');r.setTarget('one two three four');assert.equal(r.flush(),r.target);
r.setTarget('a�');r.flush();r.setTarget('aé');assert.equal(r.visible,'a');assert.equal(r.step(),'aé');
const elements=new Map(),timeouts=new Map(),intervals=new Map();let next=1;
function el(id){if(!elements.has(id))elements.set(id,{value:'',placeholder:'Begin…',children:[],scrollHeight:500,scrollTop:0,clientHeight:500,replaceChildren(...c){this.children=c},append(c){this.children.push(c)},removeAttribute(){}});return elements.get(id)}
const ctx=vm.createContext({WordReveal,console,Blob,AbortController,URL,performance:{now:()=>0},document:{getElementById:el,createTextNode:text=>({textContent:text}),createElement:()=>({})},window:{addEventListener(){}},Worker:class{postMessage(d){this.sent=d}terminate(){}},setTimeout:f=>{const id=next++;timeouts.set(id,f);return id},clearTimeout:id=>timeouts.delete(id),setInterval:f=>{const id=next++;intervals.set(id,f);return id},clearInterval:id=>intervals.delete(id)});
vm.runInContext(fs.readFileSync('dist/app.js','utf8')+`\nglobalThis.test={start:()=>{spawn();worker.onmessage({data:{type:'ready',name:'test',params:1,context:32,backend:'cpu'}})},event:d=>worker.onmessage({data:d}),busy:()=>busy};`,ctx);
for(const [id,value] of Object.entries({length:'64',temperature:'.75',candidates:'2',repeat:'1.2',gram:'3',mode:'fast'}))el(id).value=value;
const event=d=>ctx.test.event(d),text=()=>el('highlight').children.map(x=>x.textContent).join('');
function tick(){const [id,fn]=timeouts.entries().next().value;timeouts.delete(id);fn()}
ctx.test.start();el('editor').value='Prompt: ';el('generate').onclick();assert.equal(el('editor').value,'Prompt: ');assert.ok(text().includes('.'));
for(const fn of intervals.values())fn();assert.ok(text().includes('..'));assert.equal(el('editor').value,'Prompt: ');
event({type:'chunk',text:'one two ',tokens:32,elapsed:1,score:{}});assert.equal(el('editor').value,'Prompt: ');tick();assert.equal(el('editor').value,'Prompt: one ');
event({type:'chunk',text:'one two three four',tokens:64,elapsed:2,score:{}});event({type:'done',tokens:64,elapsed:2});assert.equal(ctx.test.busy(),true);while(timeouts.size)tick();assert.equal(ctx.test.busy(),false);assert.equal(el('editor').value,'Prompt: one two three four');assert.ok(!text().includes('...'));
el('regenerate').onclick();assert.equal(el('editor').value,'Prompt: ');event({type:'chunk',text:'new result here',tokens:32,elapsed:1,score:{}});tick();el('stop').onclick();assert.equal(el('editor').value,'Prompt: new result here');assert.equal(timeouts.size,0);event({type:'done',cancelled:true,tokens:32,elapsed:1});assert.equal(ctx.test.busy(),false);
el('continue').onclick();el('stop').onclick();assert.ok(!text().endsWith('.\n'));event({type:'done',cancelled:true,tokens:0,elapsed:1});assert.equal(el('editor').value,'Prompt: new result here');assert.equal(intervals.size,0);
console.log('PASS: multilingual word reveal, queue extension, UTF-8 tail repair, first-group dots, no dots in editor, next-group buffering, final drain, stop flush, regenerate, timer cleanup.');
const html=fs.readFileSync('dist/index.html','utf8');assert.match(html,/<select id="candidates"><option>1<\/option><option>2<\/option><option selected>3<\/option><option>4<\/option>/);
assert.match(fs.readFileSync('dist/app.js','utf8'),/Math.max\(300,Math.min\(900/);
// Finishing inference must unlock Continue even with an unrevealed tail.
el('generate').onclick();event({type:'chunk',text:'buffered words not yet visible',tokens:32,elapsed:1,score:{},nextGroupMs:3000});
event({type:'status',message:'Segment 2 · candidate 1/3'});
event({type:'progress',segment:2,candidate:1,candidates:3,token:4,count:32});assert.equal(el('status').textContent,'Segment 2 · candidate 1/3');assert.equal(el('state').textContent,'Generating');
event({type:'done',tokens:32,elapsed:1});assert.equal(el('continue').disabled,false);
const previous=el('editor').value;assert.ok(!previous.endsWith('not yet visible'));
el('continue').onclick();assert.equal(el('editor').value,'Prompt: new result herebuffered words not yet visible');
el('stop').onclick();event({type:'done',cancelled:true,tokens:0,elapsed:1});assert.equal(timeouts.size,0);
// Both streamed downloads and the no-stream fallback update percentage progress.
ctx.location={href:'https://example.test/'};let fetchIndex=0;const progress=[];
Object.defineProperty(el('status'),'textContent',{get(){return this.current},set(value){this.current=value;progress.push(value)}});
ctx.fetch=async()=>{
 fetchIndex++;
 if(fetchIndex===1)return {ok:true,json:async()=>({files:['a.bin','b.bin'],bytes:4})};
 if(fetchIndex===3)return {ok:true,body:null,blob:async()=>new Blob([new Uint8Array(2)])};
 let reads=0;return {ok:true,body:{getReader:()=>({read:async()=>++reads<=2?{done:false,value:new Uint8Array(1)}:{done:true}})}};
};
vm.runInContext('published()',ctx).then(()=>{
 assert.deepEqual(progress.filter(x=>x.startsWith('Downloading model')),['Downloading model · 0%','Downloading model · 25%','Downloading model · 50%','Downloading model · 100%']);
 console.log('PASS: adaptive reveal, simple sidebar ignores live progress, Continue before animation ends, default 3 candidates, percentage progress including no-stream fallback.');
}).catch(e=>{console.error(e);process.exitCode=1});
