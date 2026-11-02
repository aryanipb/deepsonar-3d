// Browser pipeline: ONE scan file -> complete 3D object, or a rejection with its reason. Everything runs on-device.
//
//   file (N x 3, z = UP)
//     -> thin()            raw metric map: 1 cm voxels (exports pass unchanged)
//     -> clean()           the dwelled-on surface; otherwise the file is REJECTED and the model is not run
//     -> toCanonical()     sensor-aligned, longest seen side 0.9
//     -> sample2048()      the model input
//     -> model             (1, 2048, 3) -> (1, 16384, 3)
//     -> fromCanonical()   metres, sensor at the origin
//
// Contract with Python (majproj/infer.py): for one file this makes the same cleaner decision (status, reason, object
// points) and the same float32 model input as `majproj.infer.Completer`. Nothing is random. scripts/build_web.py
// writes the Python outputs into samples/expected.json; `node web/test/parity.mjs` and `index.html?selftest` compare.
import {clean, PORTED_FROM_CLEANER_SHA256} from './clean.js';

export {clean, PORTED_FROM_CLEANER_SHA256};
export const N_IN = 2048;            // model input points
export const SEEN_FACE_SPAN = 0.9;   // longest seen side in the canonical frame
const EXPORT_MAX_ROWS = 4096, VOXEL = 0.01, MIN_OBJECT_PTS = 20, CM = 100;

// ---------- file parsing ----------
// Every parser returns a flat Float64Array [x0, y0, z0, x1, ...] holding only the finite rows.
export function parsePointFile(name, buffer) {
  // Android pickers can hand over files without their extension (content:// names), so sniff the content too.
  const lower = name.toLowerCase();
  const head = new Uint8Array(buffer, 0, Math.min(6, buffer.byteLength));
  if (lower.endsWith('.npy') || (head[0] === 0x93 && String.fromCharCode(...head.slice(1, 6)) === 'NUMPY')) return parseNpy(buffer);
  if (lower.endsWith('.ply') || String.fromCharCode(...head.slice(0, 3)) === 'ply') return parsePly(buffer);
  return parseText(new TextDecoder().decode(buffer));
}

function parseNpy(buffer) {
  const u8 = new Uint8Array(buffer);
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('not a .npy file');
  const major = u8[6];
  const dv = new DataView(buffer);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*True/.test(header);              // the device writes its raw maps this way
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1].split(',').map(s => s.trim()).filter(Boolean).map(Number);
  if (shape.length !== 2 || shape[1] !== 3) throw new Error(`expected an N x 3 array, got shape (${shape})`);
  const off = hstart + hlen;
  let data;
  if (descr === '<f4') data = Float64Array.from(new Float32Array(buffer.slice(off, off + shape[0] * 12)));
  else if (descr === '<f8') data = new Float64Array(buffer.slice(off, off + shape[0] * 24));
  else throw new Error(`unsupported dtype ${descr} (use float32 or float64)`);
  if (fortran) {
    const n = shape[0], rows = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) rows[3 * i + a] = data[a * n + i];
    data = rows;
  }
  return toPoints(data);
}

function parsePly(buffer) {
  const u8 = new Uint8Array(buffer);
  const headEnd = new TextDecoder().decode(u8.slice(0, Math.min(u8.length, 4096))).indexOf('end_header');
  if (headEnd < 0) throw new Error('invalid .ply header');
  const headerText = new TextDecoder().decode(u8.slice(0, headEnd));
  const bodyStart = headEnd + 'end_header'.length + 1 + (u8[headEnd + 'end_header'.length] === 13 ? 1 : 0);
  const lines = headerText.split(/\r?\n/);
  const format = lines.find(l => l.startsWith('format')).split(/\s+/)[1];
  let count = 0, inVertex = false;
  const props = [];
  for (const l of lines) {
    const t = l.trim().split(/\s+/);
    if (t[0] === 'element') { inVertex = t[1] === 'vertex'; if (inVertex) count = +t[2]; }
    else if (t[0] === 'property' && inVertex) props.push({type: t[1], name: t[t.length - 1]});
  }
  const ix = ['x', 'y', 'z'].map(n => props.findIndex(p => p.name === n));
  if (ix.some(i => i < 0)) throw new Error('.ply has no x/y/z vertex properties');
  const out = new Float64Array(count * 3);
  if (format === 'ascii') {
    const body = new TextDecoder().decode(u8.slice(bodyStart)).split(/\r?\n/);
    for (let i = 0; i < count; i++) {
      const t = body[i].trim().split(/\s+/);
      for (let k = 0; k < 3; k++) out[i * 3 + k] = +t[ix[k]];
    }
  } else if (format === 'binary_little_endian') {
    const size = {char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4,
      int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8};
    const offs = [];
    let stride = 0;
    for (const p of props) { offs.push(stride); stride += size[p.type]; }
    const dv = new DataView(buffer, bodyStart);
    for (let i = 0; i < count; i++) {
      for (let k = 0; k < 3; k++) {
        const p = props[ix[k]], o = i * stride + offs[ix[k]];
        out[i * 3 + k] = p.type.startsWith('double') || p.type === 'float64' ? dv.getFloat64(o, true) : dv.getFloat32(o, true);
      }
    }
  } else throw new Error(`unsupported .ply format ${format}`);
  return toPoints(out);
}

function parseText(text) {
  const vals = [];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/[\s,;]+/).map(Number);
    if (t.length >= 3 && t.slice(0, 3).every(Number.isFinite)) vals.push(t[0], t[1], t[2]);
  }
  if (!vals.length) throw new Error('no "x y z" rows found in the file');
  return toPoints(Float64Array.from(vals));
}

function toPoints(flat) {
  const out = [];
  for (let i = 0; i + 2 < flat.length; i += 3) {
    if (Number.isFinite(flat[i]) && Number.isFinite(flat[i + 1]) && Number.isFinite(flat[i + 2])) {
      out.push(flat[i], flat[i + 1], flat[i + 2]);
    }
  }
  if (out.length < 150) throw new Error(`only ${out.length / 3} valid points; need at least 50`);
  return Float64Array.from(out);
}

export function fnv1a(bytes) {  // FNV-1a 32 bit, used to fingerprint float32 arrays exactly
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193) >>> 0;
  return h >>> 0;
}
const fingerprint = f32 => fnv1a(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength));
export const partialChecksum = fingerprint;

// ---------- file type and thinning ----------
// Exports are divided by their largest coordinate, so it is exactly 1 (see majproj.infer.is_export).
export function isExport(pts) {
  let m = 0;
  for (let i = 0; i < pts.length; i++) { const v = Math.abs(pts[i]); if (v > m) m = v; }
  return pts.length / 3 <= EXPORT_MAX_ROWS && Math.abs(m - 1) < 1e-5;
}

// scripts.run_pipeline.thin: one point per 1 cm voxel, the lexicographically first row of each voxel, in voxel order.
export function thin(flat) {
  const n = flat.length / 3, cells = new Float64Array(flat.length);
  for (let i = 0; i < flat.length; i++) cells[i] = Math.floor(flat[i] / VOXEL);
  const rows = Array.from({length: n}, (_, i) => i);
  rows.sort((a, b) => {                    // voxel first, then the point itself, so the first row of a voxel comes first
    for (let k = 0; k < 3; k++) { const d = cells[3 * a + k] - cells[3 * b + k]; if (d) return d; }
    for (let k = 0; k < 3; k++) { const u = flat[3 * a + k], w = flat[3 * b + k]; if (u < w) return -1; if (u > w) return 1; }
    return 0;
  });
  const kept = [];
  for (let r = 0; r < n; r++) {
    const a = rows[r], b = rows[r - 1];
    if (r === 0 || cells[3 * a] !== cells[3 * b] || cells[3 * a + 1] !== cells[3 * b + 1] || cells[3 * a + 2] !== cells[3 * b + 2]) kept.push(a);
  }
  const out = new Float64Array(kept.length * 3);
  kept.forEach((i, r) => { out[3 * r] = flat[3 * i]; out[3 * r + 1] = flat[3 * i + 1]; out[3 * r + 2] = flat[3 * i + 2]; });
  return out;
}

// ---------- canonical frame (majproj/canon.py) ----------
// sqrt(x^2 + y^2), closer to glibc's hypot (which numpy calls) than Math.hypot: 13 % against 36 % one-ulp mismatches on random
// pairs here. Not identical; the float32 model input was still bit-exact on all 42 reference files.
export function hypot2(x, y) {
  let ax = Math.abs(x), ay = Math.abs(y);
  if (ax < ay) [ax, ay] = [ay, ax];
  let h = Math.sqrt(ax * ax + ay * ay), t1, t2;
  if (h <= 2 * ay) { const d = h - ay; t1 = ax * (2 * d - ax); t2 = (d - 2 * (ax - ay)) * d; }
  else { const d = h - ax; t1 = 2 * d * (ax - 2 * ay); t2 = (4 * d - ay) * ay + d * d; }
  h -= (t1 + t2) / (2 * h);
  return h;
}

// Seen object (flat, z up) -> {canon: Float64Array, T}. T holds the same fields as canon.to_canonical.
export function toCanonical(obj, origin = [0, 0, 0]) {
  const n = obj.length / 3, mean = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) mean[a] += obj[3 * i + a];
  const hx = mean[0] / n - origin[0], hy = mean[1] / n - origin[1], len = hypot2(hx, hy);
  const view = len > 1e-9 ? [hx / len, hy / len, 0] : [0, 1, 0];
  const R = [[-view[1], view[0], 0], [0, 0, 1], view];             // rows: up x view, up, view
  const rotated = new Float64Array(obj.length), lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const px = obj[3 * i] - origin[0], py = obj[3 * i + 1] - origin[1], pz = obj[3 * i + 2] - origin[2];
    for (let a = 0; a < 3; a++) {
      const v = (px * R[a][0] + py * R[a][1]) + pz * R[a][2];
      rotated[3 * i + a] = v;
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  const side = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const T = {R, centre: lo.map((l, a) => (l + hi[a]) / 2), scale: side > 0 ? side / SEEN_FACE_SPAN : 1.0, origin: [...origin]};
  const canon = new Float64Array(obj.length);
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) canon[3 * i + a] = (rotated[3 * i + a] - T.centre[a]) / T.scale;
  return {canon, T};
}

// Exact inverse for model output (flat, canonical) -> flat Float32Array in the file frame.
export function fromCanonical(q, T) {
  const n = q.length / 3, out = new Float32Array(q.length), {R, centre, scale, origin} = T;
  for (let i = 0; i < n; i++) {
    const a = q[3 * i] * scale + centre[0], b = q[3 * i + 1] * scale + centre[1], c = q[3 * i + 2] * scale + centre[2];
    for (let k = 0; k < 3; k++) out[3 * i + k] = ((a * R[0][k] + b * R[1][k]) + c * R[2][k]) + origin[k];
  }
  return out;
}

// canon.sample_2048: farthest point sampling from index 0 on squared distances; clouds of at most 2048 points are tiled.
export function sample2048(p) {
  const m = p.length / 3, out = new Float32Array(N_IN * 3);
  if (m === 0) throw new Error('no points to sample');
  if (m <= N_IN) {
    for (let i = 0; i < N_IN; i++) for (let a = 0; a < 3; a++) out[3 * i + a] = p[3 * (i % m) + a];
    return out;
  }
  const dist = new Float64Array(m).fill(Infinity);
  let far = 0;
  for (let i = 0; i < N_IN; i++) {
    for (let a = 0; a < 3; a++) out[3 * i + a] = p[3 * far + a];
    const fx = p[3 * far], fy = p[3 * far + 1], fz = p[3 * far + 2];
    let top = -Infinity, arg = 0;
    for (let j = 0; j < m; j++) {
      const dx = p[3 * j] - fx, dy = p[3 * j + 1] - fy, dz = p[3 * j + 2] - fz, d = (dx * dx + dy * dy) + dz * dz;
      if (d < dist[j]) dist[j] = d;
      if (dist[j] > top) { top = dist[j]; arg = j; }
    }
    far = arg;
  }
  return out;
}

// 1-99 percentile extents along the canonical axes in metres or file units (majproj.infer.extents).
export function extents(canon, scale) {
  const n = canon.length / 3;
  return [0, 1, 2].map(a => {
    const v = Float64Array.from({length: n}, (_, i) => canon[3 * i + a]).sort();
    const at = q => { const x = (n - 1) * q, i = Math.floor(x), f = x - i; return i + 1 < n ? v[i] + (v[i + 1] - v[i]) * f : v[i]; };
    return (at(0.99) - at(0.01)) * scale;
  });
}

// ---------- model ----------
export async function loadModel(baseUrl, onProgress) {
  const manifest = await (await fetch(`${baseUrl}manifest.json`)).json();
  const buf = new Uint8Array(manifest.size);
  const cacheName = `adapointr-${manifest.version || 'unversioned'}`;
  let done = 0;
  for (const chunk of manifest.chunks) {
    const bytes = await fetchWithCache(cacheName, `${baseUrl}${chunk}`, n => onProgress?.((done + n) / manifest.size));
    buf.set(bytes, done);
    done += bytes.length;
    onProgress?.(done / manifest.size);
  }
  const session = await ort.InferenceSession.create(buf, {executionProviders: ['wasm'], graphOptimizationLevel: 'basic'});
  return {session, manifest};
}

async function fetchWithCache(cacheName, url, onBytes) {
  let cache = null;
  const native = window.Capacitor?.isNativePlatform?.();  // the Android app bundles the model: no cache copy
  try {
    if (!native) {
      cache = await caches.open(cacheName);
      for (const k of await caches.keys()) if (k.startsWith('adapointr-') && k !== cacheName) caches.delete(k);  // stale models
    }
  } catch (e) { /* Cache API unavailable */ }
  const hit = cache && await cache.match(url).catch(() => null);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not download ${url} (${res.status})`);
  const reader = res.body.getReader();
  const parts = [];
  let n = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onBytes?.(n);
  }
  const bytes = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { bytes.set(p, o); o += p.length; }
  try { await cache?.put(url, new Response(bytes)); } catch (e) { /* quota: fine, just not cached */ }
  return bytes;
}

// Stand-in sensor position for an export: one unit along the patch normal, whose sign is a convention.
const exportOrigin = m => [m.centre[0] + m.normal[0], m.centre[1] + m.normal[1], m.centre[2]];

function caveats(res, stage) {
  if (res.status !== 'ok') return ['The cleaner found no object in this file, so the model was not run.'];
  const out = [];
  if (stage !== 'final') out.push(`Model stage: ${stage}. These are not final results.`);
  out.push('Depth and the back of the object are predicted by the network, not measured: one scan never sees them.');
  out.push(res.metric ? 'Measured width is blurred by the 5.6 degree sensor zones, so narrow objects read wider than they are.'
    : 'This is an export: metric scale and the sensor position are gone. Sizes are in file units, the predicted back may lie on the wrong side of the face, and training never saw this format.');
  if (res.meta.layer_mode === 'drift_slab') out.push('The hand drifted along the line of sight in this scan. The face position is the depth held longest, not the true surface depth.');
  return out;
}

// Cleaner + canonical frame, no model: one file (flat Float64Array from parsePointFile) -> result with status, reason,
// confidence, observed (cleaned object), background and, for accepted scans, the 2048-point model input.
export function prepare(pts) {
  const metric = !isExport(pts);
  const c = clean(metric ? thin(pts) : pts, metric ? [0, 0, 0] : null);
  const meta = c.meta;
  let status = meta.status, reason = meta.reason || '', observed = c.object, background = c.removed;
  if (status === 'ok' && observed.length / 3 < MIN_OBJECT_PTS) {
    status = 'rejected'; reason = `too few object points (${observed.length / 3})`;
    background = Float32Array.from([...background, ...observed]); observed = new Float32Array(0);
  }
  const res = {status, metric, reason: status === 'ok' ? '' : (reason || 'the cleaner found no object'), confidence: meta.confidence,
    observed, background, completed: null, partial: null, completedCanon: null, T: null, sizes: null, warnings: [],
    checksum: null, objectChecksum: fingerprint(observed), meta};
  if (status === 'ok') {
    const {canon, T} = toCanonical(observed, metric ? [0, 0, 0] : exportOrigin(meta));
    res.partial = sample2048(canon); res.T = T; res.checksum = fingerprint(res.partial);
  }
  return res;
}

// Whole pipeline: prepare, then (accepted scans only) the model and the inverse transform back into the file frame.
export async function complete(session, pts, stage, ortLib = globalThis.ort) {
  const res = prepare(pts);
  res.warnings = caveats(res, stage);
  if (res.status !== 'ok') return res;
  // the proxied (web worker) runtime transfers the input buffer and detaches it, so hand it a copy
  const out = await session.run({partial: new ortLib.Tensor('float32', res.partial.slice(), [1, N_IN, 3])});
  res.completedCanon = Float32Array.from(out.complete.data);               // 16384 x 3 canonical
  res.completed = fromCanonical(res.completedCanon, res.T);
  const k = res.metric ? CM : 1, e = extents(res.completedCanon, res.T.scale), m = res.meta;
  res.sizes = {unit: res.metric ? 'cm' : 'file units',
    measured: {face_width: k * m.width, face_height: k * m.height, distance: res.metric && m.range_m != null ? k * m.range_m : null},
    predicted: {width: k * e[0], height: k * e[1], depth: k * e[2]}};
  return res;
}

// ---------- self-test helpers (used by index.html?selftest and the Node parity harness) ----------
// Cleaner decision, object points and the float32 model input must reproduce Python exactly; the model output must
// agree with PyTorch in one-sided Chamfer-L1 (canonical frame) up to float32 / wasm numerical noise. The model is chaotic
// on many inputs (see scripts/export_onnx.py), so the tolerance is max(cd, 2 x torch's own sensitivity to 1e-6 noise).
export const SELFTEST_TOL = {pre: 1e-9, cd: 2e-3};

export function oneSidedChamfer(a, flat) {            // a: array of [x, y, z]; mean distance to the nearest point of flat
  let acc = 0;
  const n = flat.length / 3;
  for (const p of a) {
    let best = Infinity;
    for (let j = 0; j < n; j++) {
      const dx = p[0] - flat[3 * j], dy = p[1] - flat[3 * j + 1], dz = p[2] - flat[3 * j + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < best) best = d;
    }
    acc += Math.sqrt(best);
  }
  return acc / a.length;
}

const relDiff = (a, b) => (a == null || Number.isNaN(a)) ? ((b == null || Number.isNaN(b)) ? 0 : Infinity)
  : Math.abs(a - b) / Math.max(1e-12, Math.abs(b));

// res = complete(...) result, e = expected.files[name] written by scripts/build_web.py from majproj.infer (Python)
export function compareWithExpected(res, e, tol = SELFTEST_TOL) {
  const m = res.meta;
  const decision = res.status === e.status && (res.reason || null) === (e.reason || null) && res.metric === e.metric
    && (m.layer_mode ?? null) === (e.layer_mode ?? null) && m.n_input === e.n_input && m.n_unique === e.n_unique && res.observed.length / 3 === e.n_object && res.objectChecksum === (e.object_fnv1a ?? res.objectChecksum)
    && (e.status !== 'ok' || relDiff(res.confidence, e.confidence) <= 1e-6 || Math.abs(res.confidence - e.confidence) <= 1e-6);
  const base = {decision, status: res.status, nObj: res.observed.length / 3};
  const dec = `${res.status}/${e.status}, object points ${base.nObj}/${e.n_object}, confidence ${res.confidence}/${e.confidence}`
    + (res.reason || e.reason ? `, reason "${res.reason}" / "${e.reason}"` : '');
  if (e.status !== 'ok' || res.status !== 'ok') return {...base, pass: decision, pre: decision, exact: decision, cd: 0, line: dec};
  const dCentre = Math.max(...[0, 1, 2].map(k => Math.abs(res.T.centre[k] - e.T.centre[k])));
  const dScale = relDiff(res.T.scale, e.T.scale);
  let dHead = 0;
  e.partial_head.forEach((p, i) => { for (let k = 0; k < 3; k++) dHead = Math.max(dHead, Math.abs(res.partial[3 * 16 * i + k] - p[k])); });
  const exact = res.checksum === e.partial_fnv1a && res.objectChecksum === e.object_fnv1a;
  const cd = res.completedCanon ? oneSidedChamfer(e.completed_canon, res.completedCanon) : 0;   // 0 = model not run
  const cdTol = Math.max(tol.cd, 2 * (e.torch_sensitivity || 0));
  const pre = decision && dCentre <= tol.pre && dScale <= tol.pre && exact && dHead <= 1e-6;
  return {...base, pass: pre && cd < cdTol, pre, exact, cd, cdTol, dCentre, dScale, dHead,
    line: `${dec}, input bit-exact ${exact}, |d centre| ${dCentre.toExponential(1)}, rel d scale ${dScale.toExponential(1)}, model CD ${cd.toExponential(2)} (tol ${cdTol.toExponential(1)})`};
}
