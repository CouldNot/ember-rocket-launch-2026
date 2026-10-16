'use strict';
const components = [
 {name:'Air inlet',tag:'01 / AIRFLOW',color:'#b8c8a4',text:'The inlet admits nearby air into the optical measurement path. In the wearable concept, air is sampled near the face. Keep the opening clear; any added tube or inlet needs airflow validation.',output:'No independent reading. Brings the air sample into the sensor.'},
 {name:'Laser & optics',tag:'02 / ILLUMINATION',color:'#dc9b70',text:'A laser illuminates particles moving through the sensing region. Particles scatter that light. The optics and light-sensitive detector form the optical measurement system.',output:'Scattered-light signal used to estimate particle concentrations. The internal geometry shown here is conceptual.'},
 {name:'Optical detector',tag:'03 / DETECTION',color:'#91cbb9',text:'The detector receives light scattered by airborne particles. The sensor electronics interpret the optical signal to estimate particle concentration and size characteristics.',output:'Optical signal—not chemical identity, CO concentration, radiation, or a measurement of what entered the lungs.'},
 {name:'Fan & outlet',tag:'04 / AIRFLOW',color:'#95a5b8',text:'The internal fan moves air through the sensor and out of the outlet. Periodic fan cleaning helps clear accumulated dust. The outlet must remain unobstructed.',output:'Maintains sampling airflow. The current data contract does not contain a measured airflow rate.'},
 {name:'Processing board',tag:'05 / PROCESSING',color:'#829866',text:'The sensor electronics process the optical measurements into digital particle outputs. The prototype reads PM2.5 once per second; an external controller summarizes and transmits records every 10 seconds.',output:'pm25_ug_m3 · PM2.5 mass concentration in µg/m³. Additional SPS30 channels are listed below.'},
 {name:'Power & data port',tag:'06 / INTERFACE',color:'#d6c597',text:'The five-pin connector provides power and digital communication with the wearable controller. Pin 1: 5 V supply; pin 2: RX/SDA; pin 3: TX/SCL; pin 4: interface select; pin 5: ground.',output:'UART or I²C particle data. Location and identity come from the phone/relay, not this connector or the SPS30 itself.'}
];
const canvas=document.querySelector('#model'),ctx=canvas.getContext('2d');
let yaw=-.55,pitch=.65,zoom=1,exploded=false,enclosed=false,selected=0,faces=[],width=0,height=0,drag=null;
const detail=document.querySelector('#detail'),parts=document.querySelector('#parts');
components.forEach((p,i)=>{const b=document.createElement('button');b.innerHTML='<small>'+p.tag+'</small><strong>'+p.name+'</strong>';b.setAttribute('aria-pressed',String(i===selected));b.addEventListener('click',()=>select(i));parts.append(b)});
function select(i){selected=i;const p=components[i];detail.innerHTML='<span class="part-number">'+p.tag+'</span><h2 class="detail-title">'+p.name+'</h2><p class="detail-copy">'+p.text+'</p><div class="detail-output"><small>MEASUREMENT ROLE</small><p>'+p.output+'</p></div>';[...parts.children].forEach((b,j)=>b.setAttribute('aria-pressed',String(i===j)));draw()}
function transform(v){const [x,y,z]=v;const a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw);return[a,y*Math.cos(pitch)-b*Math.sin(pitch),y*Math.sin(pitch)+b*Math.cos(pitch)]}
function project(v){const [x,y,z]=transform(v),scale=Math.min(width/6.5,height/5.6)*zoom*8/(8+z);return[width/2+x*scale,height*.49-y*scale,z]}
function box(x,y,z,w,h,d,color,id){const v=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(a=>[x+a[0]*w/2,y+a[1]*h/2,z+a[2]*d/2]);[[0,1,2,3],[4,7,6,5],[0,4,5,1],[3,2,6,7],[1,5,6,2],[0,3,7,4]].forEach((f,i)=>face(f.map(j=>v[j]),color,id,[.65,.8,.58,1,.76,.88][i]))}
function face(vertices,color,id,shade=1){const pts=vertices.map(project);faces.push({pts,color,id,shade,depth:pts.reduce((a,p)=>a+p[2],0)/pts.length})}
function cylinder(x,y,z,r,h,color,id){for(let i=0;i<32;i++){const a=i/32*Math.PI*2,b=(i+1)/32*Math.PI*2;face([[x+Math.cos(a)*r,y-h/2,z+Math.sin(a)*r],[x+Math.cos(b)*r,y-h/2,z+Math.sin(b)*r],[x+Math.cos(b)*r,y+h/2,z+Math.sin(b)*r],[x+Math.cos(a)*r,y+h/2,z+Math.sin(a)*r]],color,id,.7+.2*Math.sin(a))}face(Array.from({length:32},(_,i)=>[x+Math.cos(i/32*Math.PI*2)*r,y+h/2,z+Math.sin(i/32*Math.PI*2)*r]),color,id)}
function tint(hex,s){const n=parseInt(hex.slice(1),16);return 'rgb('+Math.round((n>>16)*s)+','+Math.round(((n>>8)&255)*s)+','+Math.round((n&255)*s)+')'}
function draw(){if(!width)return;ctx.clearRect(0,0,width,height);faces=[];const e=exploded?1:0;
 ctx.lineWidth=.6;ctx.strokeStyle='#46543e55';for(let k=-6;k<=6;k++){for(const v of [[[k*.6,-1,-3.6],[k*.6,-1,3.6]],[[-3.6,-1,k*.6],[3.6,-1,k*.6]]]){const a=project(v[0]),b=project(v[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke()}}
 box(0,-.42,0,4.1,.18,4.1,'#747d72',-1);box(0,-.28,0,3.85,.12,3.85,'#536b43',4);
 for(let i=0;i<7;i++)box(-1.4+i*.4,-.2,.55,.025,.02,1.6,'#b2b27b',4);
 box(-1.15,.03+e*.45,-1.72,.95,.55,.62,components[0].color,0);for(let i=0;i<5;i++)box(-1.5+i*.175,.05+e*.45,-2.041,.09,.32,.025,'#26352b',0);
 box(-1.03,.22+e*1.2,-.65,.65,.6,.7,components[1].color,1);cylinder(-1.03,.56+e*1.2,-.65,.19,.08,'#f6bc89',1);
 box(.03,.12+e*.7,-.3,.55,.45,.65,components[2].color,2);box(.03,.36+e*.7,-.3,.33,.05,.35,'#214c43',2);
 cylinder(1.03,.04+e*.55,-.8,.76,.48,'#444e50',3);cylinder(1.03,.29+e*.55,-.8,.63,.025,'#202b28',3);
 for(let i=0;i<8;i++){const a=i*Math.PI/4;face([[1.03+Math.cos(a)*.15,.32+e*.55,-.8+Math.sin(a)*.15],[1.03+Math.cos(a+.2)*.6,.32+e*.55,-.8+Math.sin(a+.2)*.6],[1.03+Math.cos(a+.65)*.54,.32+e*.55,-.8+Math.sin(a+.65)*.54]],components[3].color,3)}cylinder(1.03,.34+e*.55,-.8,.17,.07,'#bcc8bf',3);box(1.15,0+e*.55,-1.85,.8,.4,.36,'#68777a',3);
 box(-.45,.02+e*.35,1,.85,.4,.72,'#303d2b',4);for(let i=0;i<6;i++){box(-.78+i*.13,-.03+e*.35,.58,.06,.06,.14,'#b0bba4',4);box(-.78+i*.13,-.03+e*.35,1.42,.06,.06,.14,'#b0bba4',4)}
 box(.65,-.02+e*.5,1.82,1.15,.4,.5,components[5].color,5);for(let i=0;i<5;i++)box(.22+i*.215,-.02+e*.5,2.1,.095,.14,.15,'#dcc478',5);
 if(enclosed){box(0,.63+e*1.7,0,4.1,.13,4.1,'#b3b9ad',-1);box(-2,.03,0,.1,1.1,4.1,'#919c8a',-1);box(2,.03,0,.1,1.1,4.1,'#919c8a',-1)}
 faces.sort((a,b)=>b.depth-a.depth);faces.forEach(f=>{ctx.beginPath();f.pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();ctx.fillStyle=tint(f.color,f.shade);ctx.fill();ctx.strokeStyle=f.id===selected?'#dcffa4':'#111a1366';ctx.lineWidth=f.id===selected?1.35:.45;ctx.stroke()});
 const anchors=[[-1.15,.52+e*.45,-1.72],[-1.03,.85+e*1.2,-.65],[.03,.65+e*.7,-.3],[1.03,.66+e*.55,-.8],[-.45,.47+e*.35,1],[.65,.5+e*.5,1.82]];
 markers=anchors.map(project);markers.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],12,0,Math.PI*2);ctx.fillStyle=selected===i?'#d0ef92':'#162016';ctx.fill();ctx.strokeStyle='#a8bf93';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle=selected===i?'#15200d':'#ddebd3';ctx.font='10px monospace';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1).padStart(2,'0'),p[0],p[1])});
}
let markers=[];
function resize(){const r=canvas.getBoundingClientRect();width=r.width;height=r.height;const d=Math.min(devicePixelRatio||1,2);canvas.width=width*d;canvas.height=height*d;ctx.setTransform(d,0,0,d,0,0);draw()}
function hit(x,y){for(let i=markers.length-1;i>=0;i--)if(Math.hypot(markers[i][0]-x,markers[i][1]-y)<17)return i;for(let i=faces.length-1;i>=0;i--){const f=faces[i];ctx.beginPath();f.pts.forEach((p,j)=>j?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.closePath();const d=Math.min(devicePixelRatio||1,2);if(ctx.isPointInPath(x*d,y*d))return f.id}return-1}
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,moved:false};canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4)drag.moved=true;if(drag.moved){yaw+=dx*.009;pitch=Math.max(-1.3,Math.min(1.3,pitch+dy*.009));draw()}drag.x=e.clientX;drag.y=e.clientY});
canvas.addEventListener('pointerup',e=>{if(drag&&!drag.moved){const r=canvas.getBoundingClientRect(),id=hit(e.clientX-r.left,e.clientY-r.top);if(id>=0)select(id)}drag=null});canvas.addEventListener('pointercancel',()=>drag=null);
function changeZoom(delta){zoom=Math.max(.6,Math.min(1.7,zoom+delta));draw()}
canvas.addEventListener('wheel',e=>{e.preventDefault();changeZoom(-e.deltaY*.001)},{passive:false});
canvas.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','Home'].includes(e.key)){e.preventDefault();if(e.key==='ArrowLeft')yaw-=.15;if(e.key==='ArrowRight')yaw+=.15;if(e.key==='ArrowUp')pitch=Math.min(1.3,pitch+.15);if(e.key==='ArrowDown')pitch=Math.max(-1.3,pitch-.15);if(e.key==='+')changeZoom(.1);if(e.key==='-')changeZoom(-.1);if(e.key==='Home')reset();draw()}});
function reset(){yaw=-.55;pitch=.65;zoom=1;draw()}
document.querySelector('#reset').onclick=reset;document.querySelector('#plus').onclick=()=>changeZoom(.1);document.querySelector('#minus').onclick=()=>changeZoom(-.1);
document.querySelector('#explode').onclick=e=>{exploded=!exploded;e.currentTarget.setAttribute('aria-pressed',String(exploded));e.currentTarget.textContent=exploded?'Assemble parts':'Explode parts';draw()};
document.querySelector('#shell').onclick=e=>{enclosed=!enclosed;e.currentTarget.setAttribute('aria-pressed',String(enclosed));e.currentTarget.textContent=enclosed?'Hide enclosure':'Show enclosure';document.querySelector('#viewLabel').textContent=enclosed?'ENCLOSURE VIEW':'CUTAWAY VIEW';draw()};
new ResizeObserver(resize).observe(canvas);select(0);resize();
