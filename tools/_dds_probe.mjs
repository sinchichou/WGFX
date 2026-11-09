import fs from 'fs';
function h2f(h){const s=(h&0x8000)>>15,e=(h&0x7c00)>>10,f=h&0x3ff;
 if(e===0)return (s?-1:1)*Math.pow(2,-14)*(f/1024);
 if(e===0x1f)return f?NaN:(s?-Infinity:Infinity);
 return (s?-1:1)*Math.pow(2,e-15)*(1+f/1024);}
for (const n of ['Coef_Scale','Coef_USM']) {
  const b=fs.readFileSync('D:/WGFX/effects_src/NIS/'+n+'.dds');
  console.log('=== '+n);
  for (let row=0;row<6;row++){
    const vals=[];
    for (let col=0;col<2;col++){
      const o=128+(row*2+col)*8;
      const q=[b.readUInt16LE(o),b.readUInt16LE(o+2),b.readUInt16LE(o+4),b.readUInt16LE(o+6)];
      vals.push('['+q.map(x=>h2f(x).toFixed(4)).join(', ')+']');
    }
    console.log(' row'+row+'  tex0='+vals[0]+'  tex1='+vals[1]);
  }
}
