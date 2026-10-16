'use strict';
// Fictional scenario, not a physical fire model or validated operational rule.
function buildScenario(threshold=150){
 const samples=[];let sustained=0,trigger=null,delivery=null,exposure=0;
 for(let t=0;t<=120;t++){
  const retreat=delivery===null?Infinity:delivery+6;
  let x=t<=10?160:Math.min(650,160+(t-10)*10);
  if(t>=retreat){const start=Math.min(650,160+(retreat-10)*10);x=Math.max(160,start-(t-retreat)*13)}
  const pm=12+460*Math.pow(Math.max(0,(x-240)/410),1.5);
  if(t>0)exposure+=(samples[t-1].pm+pm)/2/60;
  sustained=pm>=threshold?sustained+1:0;
  if(trigger===null&&sustained>=15){trigger=t;delivery=Math.ceil(t/10)*10}
  samples.push({t,x,pm,exposure,sustained,inside:x>=380});
 }
 return {samples,trigger,delivery,threshold};
}
if(typeof module!=='undefined'&&module.exports)module.exports={buildScenario};
if(typeof document!=='undefined'){
 const $=id=>document.getElementById(id);let scenario=buildScenario(),time=0,playing=false,acknowledged=false,speed=4,last=null,frame=0;
 const clock=t=>'14:'+String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0')+' UTC';
 const elapsed=t=>String(Math.floor(t/60)).padStart(2,'0')+':'+String(t%60).padStart(2,'0');
 function render(){
  const tick=Math.min(120,Math.floor(time)),s=scenario.samples[tick],next=scenario.samples[Math.min(tick+1,120)];
  const x=s.x+(next.x-s.x)*(time-tick),delivered=scenario.delivery!==null&&tick>=scenario.delivery,retreat=delivered&&tick>=scenario.delivery+6,outsideAfter=retreat&&!s.inside;
  const packet=Math.floor(tick/10)*10,window=scenario.samples.slice(Math.max(0,packet-9),packet+1),mean=window.reduce((n,a)=>n+a.pm,0)/window.length,p=scenario.samples[packet];
  $('pm').textContent=s.pm.toFixed(0);$('pm').style.color=s.pm>=scenario.threshold?'#e9af7a':'#d0ef92';$('dose').textContent=s.exposure.toFixed(1);$('distance').textContent=Math.abs((s.x-380)*.2).toFixed(0);$('distanceUnit').textContent=s.inside?'m inside':'m outside';$('packetAge').textContent=tick-packet;$('packetTime').textContent='Last received · '+clock(packet);
  $('pmTrend').textContent=tick===0?'Ambient particle mass concentration':s.pm>scenario.samples[tick-1].pm+.1?'↑ Rising with smoke exposure':s.pm<scenario.samples[tick-1].pm-.1?'↓ Falling during withdrawal':'→ Steady at this position';
  $('simClock').textContent='T+'+elapsed(tick);$('elapsed').textContent=tick+' / 120 s';$('timeline').value=String(tick);
  $('walker').setAttribute('transform','translate('+x.toFixed(2)+' 373)');$('trail').setAttribute('d','M160 373H'+x.toFixed(2));$('sensorCallout').setAttribute('opacity',tick<=10?'1':'0');$('signal').setAttribute('opacity',tick%10<2&&tick>0?'1':'0');
  $('field').classList.toggle('running',playing);$('field').classList.toggle('walking',tick>10&&tick<120&&Math.abs(next.x-s.x)>.1);
  let stage,title,description;
  if(tick<=10){stage='01 / GEAR CHECK';title='Clip in. Pair up.';description='The inlet sits near the face. A short tube leads to the body module holding the SPS30. The paired phone adds location.'}
  else if(outsideAfter){stage='04 / BEYOND THE PERIMETER';title='Lower readings. Exposure retained.';description='The firefighter has left the demo perimeter. PM2.5 falls, but accumulated exposure and the alert history remain. Incident command still needs a check-in.'}
  else if(retreat){stage='04 / WITHDRAWAL';title='Turn back. Leave the perimeter.';description='The figure follows the demo withdrawal instruction toward staging. The phone keeps the alert visible while measurements continue.'}
  else if(delivered){stage='03 / ALERT DELIVERED';title='The phone receives the alert.';description='A sustained rise triggered the device rule at T+'+elapsed(scenario.trigger)+'. The next summary packet at T+'+elapsed(scenario.delivery)+' delivers the demonstration withdrawal instruction.'}
  else if(scenario.trigger!==null&&tick>=scenario.trigger){stage='03 / ALERT QUEUED';title='Sustained rise detected.';description='Fifteen consecutive samples exceeded the demo threshold. The alert is queued for the next 10-second phone summary.'}
  else{stage='02 / ENTERING SMOKE';title=s.inside?'The concentration climbs.':'Approaching the perimeter.';description='PM2.5 rises along this fictional smoke gradient. '+s.sustained+' of 15 consecutive samples are currently at or above '+scenario.threshold+' µg/m³. Location is supplied by the phone.'}
  $('stageNumber').textContent=stage;$('stageTitle').textContent=title;$('stageDescription').textContent=description;
  $('phonePM').innerHTML=mean.toFixed(0)+' <small>µg/m³</small>';$('phoneTime').textContent=clock(packet)+' · simulated';$('phoneDose').textContent=p.exposure.toFixed(1)+' µg·min/m³';$('phoneLocation').textContent=p.inside?'Inside perimeter':'Outside perimeter';
  $('phoneAlert').classList.toggle('active',delivered);$('ack').hidden=!delivered;$('ack').disabled=acknowledged;$('ack').textContent=acknowledged?'✓ Alert acknowledged':'Acknowledge alert';
  $('phoneStatus').textContent=delivered?(acknowledged?'Acknowledged':'Alert active'):'Monitoring';
  $('alertTitle').textContent=delivered?'DEMO: leave the perimeter':'Monitoring ambient air';
  $('alertMessage').textContent=delivered?(outsideAfter?'Outside perimeter on the latest scenario position. Alert retained; report to incident command for a check-in.':'PM2.5 stayed ≥ '+scenario.threshold+' µg/m³ for 15 samples. Withdraw toward staging and check in with incident command. Demo instruction only.'):'No demo alert yet. Incoming summaries appear every 10 simulated seconds.';
  const pts=scenario.samples.slice(0,packet+1).filter((_,i)=>i%2===0).map(a=>(a.t/120*240).toFixed(1)+','+(70-a.pm/500*65).toFixed(1)).join(' ');$('chartLine').setAttribute('points',pts);$('thresholdLine').setAttribute('d','M0 '+(70-scenario.threshold/500*65)+'H240');
  $('thresholdValue').textContent=scenario.threshold+' µg/m³';$('ruleOutcome').textContent='Device flag: T+'+elapsed(scenario.trigger)+' · Phone alert: T+'+elapsed(scenario.delivery)+' · Source: simulated';
  $('play').textContent=playing?'Ⅱ Pause':tick>=120?'↻ Replay simulation':tick>0?'▶ Resume simulation':'▶ Run simulation';
 }
 function pause(){playing=false;cancelAnimationFrame(frame);last=null;render()}
 function animate(now){if(!playing)return;if(last!==null)time=Math.min(120,time+Math.min((now-last)/1000,.25)*speed);last=now;if(time>=120){playing=false;last=null}render();if(playing)frame=requestAnimationFrame(animate)}
 function start(){if(time>=120){time=0;acknowledged=false}playing=true;last=null;render();frame=requestAnimationFrame(animate)}
 function seek(t){pause();time=t;acknowledged=false;render()}
 $('play').onclick=()=>playing?pause():start();$('restart').onclick=()=>seek(0);$('speed').onchange=e=>{speed=Number(e.target.value)};
 $('timeline').oninput=e=>seek(Number(e.target.value));document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>seek(Number(b.dataset.jump)));$('jumpAlert').onclick=()=>seek(scenario.delivery);
 $('threshold').oninput=e=>{scenario=buildScenario(Number(e.target.value));seek(0)};$('ack').onclick=()=>{acknowledged=true;render()};
 document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)pause()});render();
}
