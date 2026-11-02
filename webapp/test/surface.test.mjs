// node webapp/test/surface.test.mjs : the browser surface is closed and sits on the points.
import assert from 'node:assert';
import {toSurface} from '../surface.js';
let seed = 1; const rnd = () => (seed = seed * 16807 % 2147483647) / 2147483647;
const p = [];
for (let i = 0; i < 16384; i++) { const t = 6.2832 * rnd(); p.push(0.2 + 0.035 * Math.cos(t), 0.3 + 0.035 * Math.sin(t), 0.25 * rnd()); }
const m = toSurface(new Float32Array(p));
const edges = new Map();
for (let t = 0; t < m.i.length; t++) for (const [a, b] of [[m.i[t], m.j[t]], [m.j[t], m.k[t]], [m.k[t], m.i[t]]]) { const e = a < b ? `${a}_${b}` : `${b}_${a}`; edges.set(e, (edges.get(e) || 0) + 1); }
assert([...edges.values()].every(c => c === 2), 'every edge is shared by two triangles');
const wall = m.x.map((x, i) => Math.hypot(x - 0.2, m.y[i] - 0.3)).slice(0, -2).sort((a, b) => a - b);
assert(Math.abs(wall[wall.length >> 1] - 0.035) < 0.002, `radius ${wall[wall.length >> 1]}`);
assert(Math.abs(Math.max(...m.z) - Math.min(...m.z) - 0.25) < 0.02);
try { toSurface(new Float32Array(30)); assert.fail('short input'); } catch (e) { assert.match(e.message, /too few/); }
console.log('surface ok:', m.x.length, 'vertices', m.i.length, 'triangles');
