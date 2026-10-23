// Presentation only: reveals already-selected model output, never candidate drafts.
class WordReveal {
 constructor(){this.target='';this.visible='';this.segmenter=typeof Intl.Segmenter==='function'?new Intl.Segmenter(undefined,{granularity:'word'}):null}
 setTarget(text){
  this.target=text;
  // A cumulative UTF-8 decode may repair an incomplete byte sequence at a
  // segment boundary. Never duplicate text or leave an obsolete replacement.
  if(!text.startsWith(this.visible)){
   let n=0;while(n<this.visible.length&&n<text.length&&this.visible[n]===text[n])n++;
   this.visible=this.visible.slice(0,n);
  }
 }
 get pending(){return this.visible!==this.target}
 stepsLeft(){const rest=this.target.slice(this.visible.length);if(!rest)return 0;if(!this.segmenter)return (rest.match(/\S+/gu)||['']).length;let words=0;for(const part of this.segmenter.segment(rest))if(part.isWordLike)words++;return Math.max(1,words)}
 step(){
  if(!this.pending)return this.visible;
  const rest=this.target.slice(this.visible.length);let end=0;
  if(this.segmenter){
   let found=false;
   for(const part of this.segmenter.segment(rest)){
    if(found&&part.isWordLike)break;
    end=part.index+part.segment.length;
    if(part.isWordLike)found=true;
   }
  }else end=(rest.match(/^\s*\S+\s*|^\s+/u)||[rest])[0].length;
  this.visible+=rest.slice(0,end);return this.visible;
 }
 flush(){this.visible=this.target;return this.visible}
}
if(typeof module!=='undefined')module.exports=WordReveal;
