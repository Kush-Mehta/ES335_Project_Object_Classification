// Local preview connects to the local API automatically. Set VMRD_API_URL before app.js for a hosted service.
const localPreview = ["localhost", "127.0.0.1"].includes(window.location.hostname) || window.location.protocol === "file:";
const API_URL = window.VMRD_API_URL || (localPreview ? "http://127.0.0.1:8000" : "");

const classNames = ["apple", "cans", "cup", "stapler", "toothpaste"];
const classCounts = { apple: 867, cans: 877, cup: 916, stapler: 636, toothpaste: 892 };
const modelNames = { lr: "Logistic Regression", svm: "SVM (RBF)", rf: "Random Forest" };
const results = {
  lr: {
    accuracy: .9177, macro: .9185,
    matrix: [[168,1,3,0,1],[0,161,10,2,3],[4,9,156,4,10],[1,2,5,117,2],[1,3,5,3,167]],
    rows: [
      {name:"apple",tp:168,fp:6,fn:5,tn:659,p:.9655,r:.9711,f1:.9683,fpr:.0090,fnr:.0289},
      {name:"cans",tp:161,fp:15,fn:15,tn:647,p:.9148,r:.9148,f1:.9148,fpr:.0227,fnr:.0852},
      {name:"cup",tp:156,fp:23,fn:27,tn:632,p:.8715,r:.8525,f1:.8619,fpr:.0351,fnr:.1475},
      {name:"stapler",tp:117,fp:9,fn:10,tn:702,p:.9286,r:.9213,f1:.9249,fpr:.0127,fnr:.0787},
      {name:"toothpaste",tp:167,fp:16,fn:12,tn:643,p:.9126,r:.9330,f1:.9227,fpr:.0243,fnr:.0670}
    ]
  },
  svm: {
    accuracy: .9177, macro: .9181,
    matrix: [[165,0,7,0,1],[0,156,9,5,6],[2,7,166,1,7],[0,5,8,113,1],[0,2,6,2,169]],
    rows: [
      {name:"apple",tp:165,fp:2,fn:8,tn:663,p:.9880,r:.9538,f1:.9706,fpr:.0030,fnr:.0462},
      {name:"cans",tp:156,fp:14,fn:20,tn:648,p:.9176,r:.8864,f1:.9017,fpr:.0211,fnr:.1136},
      {name:"cup",tp:166,fp:30,fn:17,tn:625,p:.8469,r:.9071,f1:.8760,fpr:.0458,fnr:.0929},
      {name:"stapler",tp:113,fp:8,fn:14,tn:703,p:.9339,r:.8898,f1:.9113,fpr:.0113,fnr:.1102},
      {name:"toothpaste",tp:169,fp:15,fn:10,tn:644,p:.9185,r:.9441,f1:.9311,fpr:.0228,fnr:.0559}
    ]
  },
  rf: {
    accuracy: .8831, macro: .8826,
    matrix: [[165,1,4,0,3],[0,152,10,1,13],[3,8,160,0,12],[0,5,17,97,8],[0,6,6,1,166]],
    rows: [
      {name:"apple",tp:165,fp:3,fn:8,tn:662,p:.9821,r:.9538,f1:.9677,fpr:.0045,fnr:.0462},
      {name:"cans",tp:152,fp:20,fn:24,tn:642,p:.8837,r:.8636,f1:.8736,fpr:.0302,fnr:.1364},
      {name:"cup",tp:160,fp:37,fn:23,tn:618,p:.8122,r:.8743,f1:.8421,fpr:.0565,fnr:.1257},
      {name:"stapler",tp:97,fp:2,fn:30,tn:709,p:.9798,r:.7638,f1:.8584,fpr:.0028,fnr:.2362},
      {name:"toothpaste",tp:166,fp:36,fn:13,tn:623,p:.8218,r:.9274,f1:.8714,fpr:.0546,fnr:.0726}
    ]
  }
};
const v1Results = JSON.parse(JSON.stringify(results));
const v2Report = window.VMRD_V2_RESULTS;
let resultsVersion = v2Report ? "v2" : "v1";

const number = (x, digits=2) => Number(x).toFixed(digits);
const percent = (x, digits=1) => `${(x*100).toFixed(digits)}%`;

// Class-count chart: switch between raw counts and share of the 4,188 crops.
let countMode = "number";
function renderClassCounts() {
  const root = document.querySelector("#class-bars");
  const maximum = countMode === "number" ? Math.max(...Object.values(classCounts)) : Math.max(...Object.values(classCounts).map(x=>x/4188));
  root.innerHTML = classNames.map(label => {
    const count = classCounts[label];
    const value = countMode === "number" ? count : count/4188;
    const shown = countMode === "number" ? count.toLocaleString() : percent(value,1);
    return `<div class="class-bar-row"><span class="class-bar-label">${label}</span><div class="class-bar-track"><div class="class-bar-fill" style="width:${(value/maximum*100).toFixed(2)}%"></div></div><b>${shown}</b><small>${count.toLocaleString()} crops</small></div>`;
  }).join("");
}
document.querySelectorAll("[data-count-mode]").forEach(button=>button.addEventListener("click",()=>{
  countMode=button.dataset.countMode;
  document.querySelectorAll("[data-count-mode]").forEach(b=>b.classList.toggle("active",b===button));
  renderClassCounts();
}));
renderClassCounts();

// HOG direction illustration: the edge is fixed; the arrow shows its gradient projection.
const orientation=document.querySelector("#orientation");
if(orientation){
  const edgeAngle= Math.atan2(142,150)*180/Math.PI;
  const strongestGradientAngle=edgeAngle+90;
  function updateHogDirection(){
    const bin=Number(orientation.value),degrees=10+20*bin;
    const radians=degrees*Math.PI/180;
    const response=Math.abs(Math.cos((degrees-strongestGradientAngle)*Math.PI/180));
    const strength=Math.round(response*100);
    document.querySelector("#orientation-value").value=`${degrees}°`;
    document.querySelector("#svg-angle").textContent=`${degrees}°`;
    document.querySelector("#svg-strength").textContent=`${strength}%`;
    const arrow=document.querySelector("#gradient-arrow"),cx=83,cy=88,length=49;
    arrow.setAttribute("x1",cx);arrow.setAttribute("y1",cy);
    arrow.setAttribute("x2",cx+Math.cos(radians)*length);
    arrow.setAttribute("y2",cy-Math.sin(radians)*length);
    document.querySelector("#orientation-explanation").textContent=strength>=90
      ?`The arrow crosses the fixed edge almost perpendicularly. Brightness changes strongly along this direction (${strength}% of the illustration’s maximum).`
      :strength<=10
        ?`The arrow runs almost along the fixed edge. There is very little brightness change in this direction (${strength}% of the illustration’s maximum).`
        :`The arrow crosses the edge at an angle, so brightness changes partly along this direction (${strength}% of the illustration’s maximum).`;
  }
  orientation.addEventListener("input",updateHogDirection);
  updateHogDirection();
}

// Metric guide: a small selection explains the statistical terms where they appear.
const metricCopy={
  accuracy:{title:"Accuracy",text:"Of all held-out crops, what fraction received the correct label? It gives a useful overall summary, but it can hide a class that is often missed."},
  macro:{title:"Macro-F1",text:"For each class, F1 balances precision and recall. Macro-F1 averages those five class scores equally, so a small class counts as much as a large class."},
  precision:{title:"Precision",text:"Among examples predicted as a particular class, what fraction really belonged to it? Low precision means other classes are often incorrectly assigned that label."},
  recall:{title:"Recall",text:"Among all true examples of a class, what fraction did the model find? Low recall means the model missed many examples of that class."},
  fpr:{title:"False-positive rate (FPR)",text:"Among examples that do not belong to a class, what fraction were incorrectly labelled as that class? This is a one-class-versus-the-rest rate."},
  fnr:{title:"False-negative rate (FNR)",text:"Among examples that do belong to a class, what fraction were labelled as something else? Here it is the class miss rate: FN ÷ (TP + FN)."},
  coverage:{title:"Coverage",text:"In the rejection analysis, coverage is the fraction of test examples whose top score meets a chosen threshold. Lower coverage means more cases are deferred rather than automatically accepted."}
};
function renderMetric(key){const m=metricCopy[key];document.querySelector("#metric-explanation").innerHTML=`<b>${m.title}</b><p>${m.text}</p>`;}
document.querySelectorAll("[data-metric]").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll("[data-metric]").forEach(b=>b.classList.toggle("active",b===button));renderMetric(button.dataset.metric);
}));
renderMetric("accuracy");

// Confusion matrix heatmap and per-class metrics are built from the actual saved report.
let matrixMode="count";
let selectedClass="cup";
function renderClassDetail(modelKey,label){
  const row=results[modelKey].rows.find(x=>x.name===label);
  selectedClass=label;
  document.querySelector("#class-detail").innerHTML=`<div class="selected-class">${label}<span>${modelNames[modelKey]}</span></div><div class="detail-stats"><div><small>Correctly found</small><b>${row.tp}</b></div><div><small>Missed</small><b>${row.fn}</b></div><div><small>Wrong predictions as this class</small><b>${row.fp}</b></div><div><small>True examples</small><b>${row.tp+row.fn}</b></div></div><div class="detail-metrics"><div><span>Precision</span><b>${number(row.p,4)}</b></div><div><span>Recall</span><b>${number(row.r,4)}</b></div><div><span>F1</span><b>${number(row.f1,4)}</b></div><div><span>FPR</span><b>${number(row.fpr,4)}</b></div><div><span>FNR</span><b>${number(row.fnr,4)}</b></div></div>`;
}
function renderMatrix(){
  const modelKey=document.querySelector("#matrix-model").value;
  const matrix=results[modelKey].matrix;
  const largest=Math.max(...matrix.flat());
  const heading=`<table class="heatmap"><thead><tr><th scope="col">True ↓ / Predicted →</th>${classNames.map(x=>`<th scope="col">${x}</th>`).join("")}</tr></thead><tbody>`;
  const body=matrix.map((row,i)=>`<tr><th scope="row"><button class="row-select ${classNames[i]===selectedClass?'chosen':''}" data-row-class="${classNames[i]}">${classNames[i]}</button></th>${row.map((value,j)=>{
    const share=value/row.reduce((a,b)=>a+b,0);
    const intensity=matrixMode==="count"?value/largest:share;
    const text=matrixMode==="count"?String(value):percent(share,1);
    const tone=intensity>.54?"dark":"";
    return `<td><button class="heat-cell ${tone} ${i===j?'diagonal':''}" style="--heat:${intensity}" data-row-class="${classNames[i]}" aria-label="True ${classNames[i]}, predicted ${classNames[j]}: ${text}"><span>${text}</span><small>${classNames[j]}</small></button></td>`;
  }).join("")}</tr>`).join("");
  document.querySelector("#matrix-chart").innerHTML=heading+body+"</tbody></table>";
  document.querySelectorAll("[data-row-class]").forEach(el=>el.addEventListener("click",()=>{
    renderClassDetail(modelKey,el.dataset.rowClass);
    document.querySelectorAll(".row-select").forEach(button=>button.classList.toggle("chosen",button.dataset.rowClass===el.dataset.rowClass));
  }));
  document.querySelectorAll(".heat-cell").forEach((el,index)=>{
    const row=Math.floor(index/classNames.length),col=index%classNames.length;
    const total=matrix[row].reduce((a,b)=>a+b,0),value=matrix[row][col];
    const message=`True ${classNames[row]} → predicted ${classNames[col]}: ${value} crop${value===1?"":"s"} (${percent(value/total)} of this true class).`;
    el.addEventListener("mouseenter",()=>document.querySelector("#matrix-tooltip").textContent=message);
    el.addEventListener("focus",()=>document.querySelector("#matrix-tooltip").textContent=message);
  });
  document.querySelector("#matrix-caption").textContent=matrixMode==="count"?`Each row sums to the number of held-out examples in that true class. The diagonal cells are correct predictions. Overall test accuracy: ${percent(results[modelKey].accuracy,2)}.`:"Each row is normalized to 100%. A diagonal value is the recall for that true class; off-diagonal values show where that class was sent instead.";
  renderClassDetail(modelKey,selectedClass);
}
document.querySelector("#matrix-model").addEventListener("change",renderMatrix);
document.querySelectorAll("[data-matrix-mode]").forEach(button=>button.addEventListener("click",()=>{
  matrixMode=button.dataset.matrixMode;
  document.querySelectorAll("[data-matrix-mode]").forEach(b=>b.classList.toggle("active",b===button));renderMatrix();
}));
document.querySelector("#matrix-model").value="rf";
renderMatrix();

// Small, data-backed model comparison; the selector switches the measured quantity.
function renderComparison(){
  const metric=document.querySelector("#comparison-metric").value;
  const key=metric==="accuracy"?"accuracy":"macro";
  const values=["lr","svm","rf"].map(k=>({key:k,value:results[k][key]}));
  const minimum=0, maximum=1;
  let axis=document.querySelector("#comparison-axis");
  if(!axis){axis=document.createElement("p");axis.id="comparison-axis";axis.className="chart-foot";document.querySelector("#model-comparison").before(axis);}
  document.querySelector("#model-comparison").innerHTML=values.map(item=>`<div class="compare-row"><span>${modelNames[item.key]}</span><div class="compare-track"><i style="width:${((item.value-minimum)/(maximum-minimum)*100).toFixed(2)}%"></i></div><b>${number(item.value,4)}</b></div>`).join("");
  document.querySelector("#comparison-explanation").textContent=metric==="accuracy"?`Accuracy is the share of all 838 held-out crops labelled correctly (${resultsVersion.toUpperCase()} results).`:`Macro-F1 averages the five per-class F1 scores equally. These are the ${resultsVersion.toUpperCase()} held-out results.`;
  axis.textContent="Scale: 0.00 to 1.00 (0% to 100%); bar lengths start at zero.";
}
document.querySelector("#comparison-metric").addEventListener("change",renderComparison);renderComparison();

function applyResultsVersion(version){
  resultsVersion=version==="v2"&&v2Report?"v2":"v1";
  if(resultsVersion==="v1"){
    Object.assign(results,JSON.parse(JSON.stringify(v1Results)));
    document.querySelector("#results-version-note").textContent="V1 stretched every crop and combined HOG and HSV features in one vector.";
    document.querySelector("#finding-notes").innerHTML='<article><b>01 · CUP AND CANS</b><p>These cylindrical classes were recurring sources of confusion. Similar outlines and metallic or printed surfaces could make their edge-and-color descriptions overlap.</p></article><article><b>02 · STAPLERS</b><p>On the fixed held-out VMRD test set, v1 Random Forest missed 30 of 127 staplers (23.62%). Logistic Regression missed 10 (7.87%) and SVM missed 14 (11.02%). These benchmark counts are separate from the small exploratory checks on real photographs.</p></article><article><b>03 · APPLES</b><p>Apples had a low miss rate in these held-out results. Color may help, but these numbers did not isolate which visual clue caused a prediction.</p></article>';
  }else{
    const map={lr:"Logistic_Regression",svm:"SVM_RBF",rf:"Random_Forest"};
    for(const [key,reportKey] of Object.entries(map)){
      const m=v2Report.models[reportKey];
      results[key]={accuracy:m.accuracy,macro:m.macro_f1,matrix:m.matrix,rows:m.rows};
      document.querySelector(`#score-${key}`).textContent=percent(m.accuracy,2);
      document.querySelector(`#score-${key}-sub`).textContent=`test accuracy · macro-F1 ${number(m.macro_f1,4)}`;
    }
    document.querySelector("#results-version-note").textContent="V2 preserved crop shape and fused separate HOG and color model scores.";
    document.querySelector("#finding-notes").innerHTML='<article><b>01 · OVERALL ACCURACY</b><p>SVM and Random Forest each reached 92.00% accuracy on the held-out crops. Their per-class errors differed, so the class-level view still mattered.</p></article><article><b>02 · STAPLERS</b><p>On the fixed VMRD test set, v2 SVM had the lowest stapler miss rate across the two versions: 9 of 127 (7.09%). V2 Random Forest missed 13 (10.24%) and Logistic Regression missed 15 (11.81%). The separate real-photo checks motivated this investigation; they were exploratory and were not included in these benchmark metrics.</p></article><article><b>03 · WHAT IMPROVED</b><p>Random Forest rose from 88.31% to 92.00% accuracy, while its stapler false-negative rate fell from 23.62% to 10.24%. Both preprocessing and fusion changed, so the comparison could not attribute the gain to one change alone.</p></article>';
  }
  for(const key of ["lr","svm","rf"]){
    if(resultsVersion==="v1"){
      document.querySelector(`#score-${key}`).textContent=percent(results[key].accuracy,2);
      document.querySelector(`#score-${key}-sub`).textContent=`test accuracy · macro-F1 ${number(results[key].macro,4)}`;
    }
  }
  document.querySelector("#results-version").value=resultsVersion;
  renderMatrix();renderComparison();
  if(window.renderRejectionVersion)window.renderRejectionVersion();
}
document.querySelector("#results-version").addEventListener("change",event=>applyResultsVersion(event.target.value));

// Exact threshold series from saved held-out predictions for both versions.
function loadRejectionData(){
  const slider=document.querySelector("#curve-threshold");
  const note=document.querySelector("#rejection-plot");
  const label=document.querySelector("#curve-threshold-value");
  const output=document.querySelector("#curve-point");
  document.querySelector("#curve-model").value="Random_Forest";
  try{
    if(!window.VMRD_V1_REJECTION||!v2Report?.models)throw new Error("missing exact result data");
    const displayNames={Logistic_Regression:"Logistic Regression",SVM_RBF:"SVM (RBF)",Random_Forest:"Random Forest"};
    function currentData(){
      if(resultsVersion==="v1")return window.VMRD_V1_REJECTION;
      return Object.fromEntries(Object.entries(v2Report.models).map(([key,model])=>[key,{thresholds:model.thresholds,coverage:model.coverage,accuracy:model.accepted_accuracy}]));
    }
    function allData(){
      const v1=window.VMRD_V1_REJECTION;
      const v2=Object.fromEntries(Object.entries(v2Report.models).map(([key,model])=>[key,{thresholds:model.thresholds,coverage:model.coverage,accuracy:model.accepted_accuracy}]));
      return {v1,v2};
    }
    function referenceIndex(series){
      const valid=series.thresholds.map((t,i)=>({t,c:series.coverage[i],a:series.accuracy[i],i})).filter(p=>p.a!==null);
      if(!valid.length)return 0;
      const highestAccuracy=Math.max(...valid.map(p=>p.a));
      const plateauEdge=valid.filter(p=>Math.abs(p.a-highestAccuracy)<1e-9).reduce((best,p)=>p.c>best.c?p:best);
      return plateauEdge.i;
    }
    function show(resetToReference=true){
      const data=currentData();
      const both=allData();
      const summaryRows=Object.entries(both).flatMap(([version,versionData])=>Object.entries(versionData).map(([key,series])=>{
        const ref=series.thresholds[referenceIndex(series)];
        const ri=referenceIndex(series);
        const full=series.thresholds.map((t,i)=>({t,c:series.coverage[i],a:series.accuracy[i]})).reduce((best,p)=>p.c>best.c?p:best);
        return `<tr><td>${version.toUpperCase()}</td><td>${displayNames[key]||key}</td><td>${number(ref,2)}</td><td>${percent(series.coverage[ri])}</td><td>${percent(series.accuracy[ri])}</td><td>${percent(full.a)}</td></tr>`;
      })).join("");
      document.querySelector("#threshold-summary").innerHTML=`<div class="table-scroll"><table><thead><tr><th>Version</th><th>Model</th><th>Threshold at plateau edge</th><th>Coverage at plateau edge</th><th>Accuracy at plateau edge</th><th>Accuracy at 100% coverage</th></tr></thead><tbody>${summaryRows}</tbody></table></div>`;
      const series=data[document.querySelector("#curve-model").value];
      if(resetToReference)slider.value=series.thresholds[referenceIndex(series)];
      if(!series?.thresholds?.length)throw new Error("series not found");
      const desired=Number(slider.value);let idx=0;
      for(let i=1;i<series.thresholds.length;i++)if(Math.abs(series.thresholds[i]-desired)<Math.abs(series.thresholds[idx]-desired))idx=i;
      const points=series.thresholds.map((t,i)=>({t,c:series.coverage[i],a:series.accuracy[i]}));
      const W=740,H=340,L=58,R=18,T=18,B=100;
      const x=v=>L+v*(W-L-R), y=v=>T+(1-v)*(H-T-B);
      const axes=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}"/><line x1="${L}" y1="${y(0)}" x2="${L}" y2="${y(1)}"/>`;
      const grid=[0,.25,.5,.75,1].map(v=>`<line class="gridline" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`).join("");
      const xGrid=[0,.25,.5,.75,1].map(v=>`<line class="gridline" x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${y(0)}"/><text x="${x(v)}" y="${y(0)+16}" text-anchor="middle">${Math.round(v*100)}%</text>`).join("");
      const acceptedAccuracy=points.filter(p=>p.a!==null).map(p=>`${x(p.c)},${y(p.a)}`).join(" ");
      const point=points[idx];
      const selectedDot=point.a===null?"":`<circle class="selection-dot" cx="${x(point.c)}" cy="${y(point.a)}" r="6"/>`;
      note.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${resultsVersion.toUpperCase()} held-out rejection curve: accuracy on accepted examples versus coverage. The selected confidence threshold is highlighted."><g class="chart-axis">${grid}${xGrid}${axes}<text x="${(L+W-R)/2}" y="${H-57}" text-anchor="middle">Coverage: fraction of held-out crops accepted</text><text transform="translate(17 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">Accuracy among accepted crops</text></g><polyline class="accepted-line" points="${acceptedAccuracy}"/><circle class="selection-dot" cx="${x(point.c)}" cy="${point.a===null?y(0):y(point.a)}" r="6"/><g class="legend"><line x1="88" y1="${H-25}" x2="112" y2="${H-25}" class="accepted-line"/><text x="119" y="${H-21}">${document.querySelector("#curve-model").selectedOptions[0].text} · ${resultsVersion.toUpperCase()} held-out set</text></g></svg>`;
      label.value=number(point.t,2);
      output.textContent=point.a==null?`Threshold ${number(point.t,2)} accepts ${percent(point.c)} of test crops; no accepted cases are available to calculate accuracy.`:`At threshold ${number(point.t,2)}, coverage is ${percent(point.c)} and accuracy among accepted crops is ${percent(point.a)}.`;
    }
    slider.disabled=false;
    slider.addEventListener("input",()=>show(false));
    document.querySelector("#curve-model").addEventListener("change",()=>show(true));
    window.renderRejectionVersion=()=>show(true);
    show(true);
  }catch{
    note.innerHTML="<p>The exact held-out threshold data could not be loaded.</p>";
  }
}
loadRejectionData();
applyResultsVersion(resultsVersion);

// Keep a fixed square input frame. Visitors can pan the source image and
// stretch it independently in each direction, matching training's 128×128 input.
const upload=document.querySelector("#upload");
const cropCanvas=document.querySelector("#crop-canvas");
const cropCtx=cropCanvas.getContext("2d");
const cropPlaceholder=document.querySelector("#crop-placeholder");
const imageScale=document.querySelector("#image-scale");
const resetImage=document.querySelector("#reset-image");
const predictButton=document.querySelector("#predict");
const apiStatus=document.querySelector("#api-status");
const prediction=document.querySelector("#prediction");
const threshold=document.querySelector("#threshold");
const frameSize=384;
let sourceImage=null, baseFit=1, offsetX=0, offsetY=0, dragging=false, lastPointer=null;
function drawImageTo(ctx,size){
  ctx.clearRect(0,0,size,size);
  if(!sourceImage)return;
  const factor=size/frameSize;
  const w=sourceImage.naturalWidth*baseFit*(Number(imageScale.value)/100)*size/300;
  const h=sourceImage.naturalHeight*baseFit*(Number(imageScale.value)/100)*size/300;
  const x=(size-w)/2+offsetX*factor,y=(size-h)/2+offsetY*factor;
  ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality="high";
  ctx.drawImage(sourceImage,x,y,w,h);
  // For a contained, non-square crop, extend edge pixels into the unused frame
  // area as BORDER_REPLICATE does in the model's v2 preprocessing.
  if(h<=size&&y>=0&&y+h<=size){
    if(y>0)ctx.drawImage(sourceImage,0,0,sourceImage.naturalWidth,1,x,0,w,y);
    if(y+h<size)ctx.drawImage(sourceImage,0,sourceImage.naturalHeight-1,sourceImage.naturalWidth,1,x,y+h,w,size-y-h);
  }
  if(w<=size&&x>=0&&x+w<=size){
    if(x>0)ctx.drawImage(sourceImage,0,0,1,sourceImage.naturalHeight,0,y,x,h);
    if(x+w<size)ctx.drawImage(sourceImage,sourceImage.naturalWidth-1,0,1,sourceImage.naturalHeight,x+w,y,size-x-w,h);
  }
  if(w<=size&&h<=size&&x>=0&&y>=0&&x+w<=size&&y+h<=size){
    if(x>0&&y>0)ctx.drawImage(sourceImage,0,0,1,1,0,0,x,y);
    if(x+w<size&&y>0)ctx.drawImage(sourceImage,sourceImage.naturalWidth-1,0,1,1,x+w,0,size-x-w,y);
    if(x>0&&y+h<size)ctx.drawImage(sourceImage,0,sourceImage.naturalHeight-1,1,1,0,y+h,x,size-y-h);
    if(x+w<size&&y+h<size)ctx.drawImage(sourceImage,sourceImage.naturalWidth-1,sourceImage.naturalHeight-1,1,1,x+w,y+h,size-x-w,size-y-h);
  }
}
function renderCrop(){
  drawImageTo(cropCtx,frameSize);
  document.querySelector("#image-scale-value").value=`${imageScale.value}%`;
}
function fitImage(){
  if(!sourceImage)return;
  baseFit=300/Math.max(sourceImage.naturalWidth,sourceImage.naturalHeight);
  imageScale.value="100";offsetX=0;offsetY=0;renderCrop();
}
function setImageFile(file){
  if(!file)return;
  const image=new Image();
  image.onload=()=>{
    sourceImage=image;cropPlaceholder.hidden=true;
    imageScale.disabled=false;resetImage.disabled=false;predictButton.disabled=false;
    fitImage();prediction.hidden=true;document.querySelector("#prediction-empty").hidden=false;
    apiStatus.textContent="Image ready. Adjust it inside the fixed 128 × 128 frame, then predict.";
  };
  image.onerror=()=>{apiStatus.textContent="The selected file could not be opened as an image.";};
  image.src=URL.createObjectURL(file);
}
upload.addEventListener("change",()=>setImageFile(upload.files?.[0]));
imageScale.addEventListener("input",renderCrop);
resetImage.addEventListener("click",fitImage);
cropCanvas.addEventListener("pointerdown",event=>{if(!sourceImage)return;dragging=true;lastPointer={x:event.clientX,y:event.clientY};cropCanvas.setPointerCapture(event.pointerId);});
cropCanvas.addEventListener("pointermove",event=>{if(!dragging||!lastPointer)return;const rect=cropCanvas.getBoundingClientRect();offsetX+=(event.clientX-lastPointer.x)*frameSize/rect.width;offsetY+=(event.clientY-lastPointer.y)*frameSize/rect.height;lastPointer={x:event.clientX,y:event.clientY};renderCrop();});
cropCanvas.addEventListener("pointerup",()=>{dragging=false;lastPointer=null;});
cropCanvas.addEventListener("pointercancel",()=>{dragging=false;lastPointer=null;});
function updateDecision(){
  if(prediction.hidden||!prediction.dataset.confidence)return;
  const accepted=Number(prediction.dataset.confidence)>=Number(threshold.value);
  const badge=document.querySelector("#decision");
  badge.className=`decision ${accepted?"pass":"review"}`;
  badge.textContent=accepted?"GRASP · identity score meets threshold":"FLAG FOR HUMAN REVIEW · score below threshold";
}
threshold.addEventListener("input",()=>{
  document.querySelector("#threshold-value").value=Number(threshold.value).toFixed(2);
  updateDecision();
});
document.querySelector("#demo-model").addEventListener("change",()=>{if(sourceImage&&API_URL)predictButton.click();});
predictButton.addEventListener("click",async()=>{
  if(!sourceImage)return;
  if(!API_URL){apiStatus.textContent="No prediction API address is configured.";return;}
  predictButton.disabled=true;apiStatus.textContent="Sending the selected crop to the prediction API…";
  try{
    // Rasterize the fixed display frame at exactly 128×128 model pixels.
    const output=document.createElement("canvas");output.width=128;output.height=128;
    const outCtx=output.getContext("2d");outCtx.imageSmoothingEnabled=true;outCtx.imageSmoothingQuality="high";drawImageTo(outCtx,128);
    const blob=await new Promise((resolve,reject)=>output.toBlob(b=>b?resolve(b):reject(new Error("Could not create crop")),"image/jpeg",.92));
    const form=new FormData();form.append("image",blob,"object-crop.jpg");form.append("model",document.querySelector("#demo-model").value);form.append("threshold",threshold.value);
    const response=await fetch(`${API_URL.replace(/\/$/,"")}/predict`,{method:"POST",body:form});
    const data=await response.json();if(!response.ok)throw new Error(data.detail||`Request failed (${response.status})`);if(data.pipeline_version!=="v2")throw new Error("The API is still running the v1 model. Restart it to load the v2 models.");
    prediction.dataset.confidence=data.confidence;prediction.dataset.predicted=data.predicted_class;
    const sorted=Object.entries(data.all_class_probabilities).sort((a,b)=>b[1]-a[1]);
    document.querySelector("#probabilities").innerHTML=sorted.map(([label,value])=>`<div class="prob-row ${label===data.predicted_class?"top":""}"><span>${label===data.predicted_class?"★ ":""}${label}</span><div class="prob-track"><i style="width:${Math.max(0,Math.min(100,value*100))}%"></i></div><b>${percent(value)}</b></div>`).join("");
    prediction.hidden=false;document.querySelector("#prediction-empty").hidden=true;
    apiStatus.textContent=`V2 prediction received from ${modelNames[data.model_used]||data.model_used}. Moving the threshold below does not change these scores.`;updateDecision();
  }catch(error){apiStatus.textContent=`The API may be waking up or unreachable: ${error.message}`;}
  finally{predictButton.disabled=false;}
});
if(!API_URL)apiStatus.textContent="No prediction API address is configured.";
else fetch(`${API_URL}/health`).then(response=>{if(!response.ok)throw new Error("API unavailable");return response.json();}).then(data=>{if(data.pipeline!=="v2")throw new Error("The old API version is running");apiStatus.textContent="V2 prediction API connected on this computer. Choose an image to begin.";}).catch(()=>{apiStatus.textContent="The v2 prediction service is not ready. Restart the local API, then reopen this page.";});
