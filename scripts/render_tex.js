// Renders every <span class="tex" data-tex="..."> (add data-display for display math) into inline SVG, idempotently.
const fs=require('fs');
const {mathjax}=require('mathjax-full/js/mathjax.js');
const {TeX}=require('mathjax-full/js/input/tex.js');
const {SVG}=require('mathjax-full/js/output/svg.js');
const {liteAdaptor}=require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler}=require('mathjax-full/js/handlers/html.js');
const {AllPackages}=require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor=liteAdaptor(); RegisterHTMLHandler(adaptor);
const doc=mathjax.document('',{InputJax:new TeX({packages:AllPackages}),OutputJax:new SVG({fontCache:'none'})});
const unesc=s=>s.replace(/&quot;/g,'"').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&');
const f=process.argv[2]; let h=fs.readFileSync(f,'utf8'); let n=0;
h=h.replace(/<span class="tex"( data-display)? data-tex="([^"]*)">[\s\S]*?<\/span><!--\/tex-->/g,(m,disp,tex)=>{
  const node=doc.convert(unesc(tex),{display:!!disp}); let svg=adaptor.innerHTML(node);
  svg=svg.replace(/<svg /,'<svg aria-label="'+tex.replace(/"/g,'&quot;')+'" ');
  n++; return `<span class="tex"${disp||''} data-tex="${tex}">${svg}</span><!--/tex-->`;});
fs.writeFileSync(f,h); console.log('rendered',n);
