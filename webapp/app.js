// One raw scan in, four 3D views out. Everything, the model included, runs in this browser.
import {complete, loadModel, parsePointFile} from './pipeline.js';
import {toSurface} from './surface.js';

const $ = id => document.getElementById(id);
const dark = matchMedia('(prefers-color-scheme: dark)').matches, ink = dark ? '#a0a5ad' : '#62676f', grid = dark ? '#30333a' : '#e4e4e0';
ort.env.wasm.wasmPaths = new URL('vendor/', location.href).href;
ort.env.wasm.numThreads = 1;      // a plain local server has no cross-origin isolation
ort.env.wasm.proxy = true;        // the model runs in a worker, so the page stays responsive
let session = null, stage = 'final', busy = false;

const say = (text, err) => { $('msg').textContent = text; $('msg').className = err ? 'err' : ''; };
const axis = title => ({title, color: ink, gridcolor: grid, zerolinecolor: grid});
const layout = () => ({margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
  scene: {aspectmode: 'data', dragmode: 'orbit', camera: {eye: {x: 2.1, y: -2.3, z: 1.0}, up: {x: 0, y: 0, z: 1}}, xaxis: axis('x (m)'), yaxis: axis('y (m)'), zaxis: axis('z (m, up)')}});
const cfg = {displaylogo: false, responsive: true, displayModeBar: false};
const xyz = f => { const n = f.length / 3, a = [new Array(n), new Array(n), new Array(n)]; for (let i = 0; i < 3 * n; i++) a[i % 3][i / 3 | 0] = f[i]; return a; };
const cloud = (f, marker) => { const [x, y, z] = xyz(f); return {type: 'scatter3d', mode: 'markers', x, y, z, hoverinfo: 'skip', marker: {...marker, ...(marker.scale ? {color: z, colorscale: 'Viridis'} : {})}}; };
const size = f => { const s = xyz(f).map(a => { const v = [...a].sort((p, q) => p - q); return 100 * (v[v.length * 0.99 | 0] - v[v.length * 0.01 | 0]); }); return s.map(v => v.toFixed(1)).join(' × ') + ' cm'; };

function clearViews() { for (const i of [0, 1, 2, 3]) { Plotly.purge(`p${i}`); $(`p${i}`).innerHTML = ''; $(`t${i}`).textContent = ''; } }

function show(raw, r, secs) {
  $('grid').hidden = false; clearViews();
  $('t0').textContent = `${(raw.length / 3).toLocaleString()} points, as recorded`;
  Plotly.newPlot('p0', [cloud(raw, {size: 1.5, scale: true})], layout(), cfg);
  if (r.status !== 'ok') { for (const i of [1, 2, 3]) $(`t${i}`).textContent = 'Not available for this scan.'; say(`No object found: ${r.reason}`, true); return; }
  $('t1').textContent = `${(r.observed.length / 3).toLocaleString()} points · seen ${size(r.observed)}`;
  Plotly.newPlot('p1', [cloud(r.observed, {size: 2.4, color: dark ? '#ecedef' : '#17191d'})], layout(), cfg);
  $('t2').textContent = `${(r.completed.length / 3).toLocaleString()} points · ${size(r.completed)}`;
  Plotly.newPlot('p2', [cloud(r.completed, {size: 1.3, color: '#ea580c'})], layout(), cfg);
  const m = toSurface(r.completed);
  $('t3').textContent = `${m.i.length.toLocaleString()} triangles`;
  Plotly.newPlot('p3', [{type: 'mesh3d', ...m, color: '#3b82f6', flatshading: false, hoverinfo: 'skip', lighting: {ambient: 0.45, diffuse: 0.85, specular: 0.25, roughness: 0.6}, lightposition: {x: 300, y: -500, z: 600}}], layout(), cfg);
  say(`Done in ${secs} s.`);
}

async function run(file) {
  if (!file || busy || !session) return;
  busy = true; $('pick').classList.add('off'); say(`Reconstructing ${file.name}…`);
  await new Promise(r => setTimeout(r, 30));
  try {
    const t0 = performance.now(), raw = parsePointFile(file.name, await file.arrayBuffer()), r = await complete(session, raw, stage);
    show(raw, r, ((performance.now() - t0) / 1000).toFixed(1));
  } catch (e) { console.error(e); say(e.message || 'This scan could not be processed.', true); }
  busy = false; $('pick').classList.remove('off');
}

$('file').onchange = e => { run(e.target.files[0]); e.target.value = ''; };
const k = $('pick');
['dragover', 'dragenter'].forEach(n => k.addEventListener(n, e => { e.preventDefault(); k.classList.add('over'); }));
['dragleave', 'drop'].forEach(n => k.addEventListener(n, e => { e.preventDefault(); k.classList.remove('over'); }));
k.addEventListener('drop', e => run(e.dataTransfer.files[0]));

const samples = [...document.querySelectorAll('[data-sample]')];
samples.forEach(b => { b.onclick = async () => {
  try {
    const res = await fetch(b.dataset.sample);
    if (!res.ok) throw new Error(`The sample scan could not be downloaded (${res.status}).`);
    run(new File([await res.arrayBuffer()], b.dataset.sample.split('/').pop()));
  } catch (e) { console.error(e); say(e.message, true); }
}; });

try {
  const m = await loadModel('model/', p => say(`Downloading the model to this device: ${Math.round(100 * p)}%`));
  session = m.session; stage = m.manifest.stage || stage;
  $('pick').classList.remove('off'); samples.forEach(b => { b.disabled = false; }); say('Model ready on this device. Choose a scan.');
} catch (e) { console.error(e); say(`The model could not be loaded on this device: ${e.message}`, true); }
