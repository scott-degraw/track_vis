// track_vis event display: ray-marched voxel volumes, polylines and markers in three.js.
// Expects THREE (r147 classic build) with OrbitControls and LineSegments2 on window.
// Entry point: TrackVis.render(containerElement, spec). The spec is built by
// track_vis/display.py (EventDisplay).
(function () {
  if (window.TrackVis) return;

  const VERT = `
    out vec3 vOrigin;
    out vec3 vDirection;
    void main() {
      vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
      vDirection = position - vOrigin;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  // Emission-absorption ray marching in the unit box [-0.5, 0.5]^3.
  // density is the optical depth of one voxel pitch at the top of the scale.
  const FRAG = `
    precision highp float;
    precision highp sampler3D;
    uniform sampler3D map;
    uniform sampler2D cmap;
    uniform vec3 shape;
    uniform float thresh;
    uniform float density;
    uniform float gamma;
    uniform int mode;
    uniform bool sharp;
    in vec3 vOrigin;
    in vec3 vDirection;
    out vec4 outColor;

    vec2 hitBox(vec3 o, vec3 d) {
      vec3 inv = 1.0 / d;
      vec3 t0 = (vec3(-0.5) - o) * inv;
      vec3 t1 = (vec3(0.5) - o) * inv;
      vec3 tmin = min(t0, t1);
      vec3 tmax = max(t0, t1);
      return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
    }

    vec4 classify(float v, float len) {
      // colour and opacity of a path of len voxel pitches through value v
      vec3 c = texture(cmap, vec2(v, 0.5)).rgb;
      float a = 1.0 - exp(-density * pow(v, gamma) * len);
      return vec4(c, a);
    }

    void main() {
      vec3 d = normalize(vDirection);
      vec2 b = hitBox(vOrigin, d);
      if (b.x > b.y) discard;
      b.x = max(b.x, 0.0);
      float vox = length(d * shape);          // voxels per unit of t

      if (sharp) {
        // Exact voxel traversal (Amanatides & Woo): each voxel is a uniform
        // cube, composited with its true path length. No sampling noise.
        vec3 D = d * shape;
        D = sign(D) * max(abs(D), vec3(1e-6)) + vec3(equal(D, vec3(0.0))) * 1e-6;
        vec3 P0 = (vOrigin + d * b.x + 0.5) * shape;
        ivec3 n = ivec3(shape);
        ivec3 cell = clamp(ivec3(floor(P0)), ivec3(0), n - 1);
        ivec3 stp = ivec3(sign(D));
        vec3 tDelta = abs(1.0 / D);
        vec3 tMax = b.x + (vec3(cell) + max(vec3(stp), 0.0) - P0) / D;
        float t = b.x;
        vec4 acc = vec4(0.0);
        float vmax = 0.0;
        for (int i = 0; i < 2048; i++) {
          float tn = min(min(tMax.x, tMax.y), min(tMax.z, b.y));
          float v = texelFetch(map, cell, 0).r;
          if (mode == 1) {
            vmax = max(vmax, v);
          } else if (v >= thresh) {
            vec4 s = classify(v, (tn - t) * vox);
            acc.rgb += (1.0 - acc.a) * s.a * s.rgb;
            acc.a += (1.0 - acc.a) * s.a;
            if (acc.a > 0.995) break;
          }
          t = tn;
          if (t >= b.y) break;
          if (tMax.x <= tMax.y && tMax.x <= tMax.z) { cell.x += stp.x; tMax.x += tDelta.x; }
          else if (tMax.y <= tMax.z) { cell.y += stp.y; tMax.y += tDelta.y; }
          else { cell.z += stp.z; tMax.z += tDelta.z; }
          if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, n))) break;
        }
        if (mode == 1) {
          if (vmax < thresh) discard;
          acc = vec4(texture(cmap, vec2(vmax, 0.5)).rgb, 1.0);
        }
        if (acc.a < 0.002) discard;
        outColor = acc;
        return;
      }

      float dt = 0.25 / vox;                  // quarter-voxel steps
      float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
      float t = b.x + dt * jitter;
      vec4 acc = vec4(0.0);
      float vmax = 0.0;
      for (int i = 0; i < 4096; i++) {
        if (t > b.y) break;
        float v = texture(map, vOrigin + d * t + 0.5).r;
        if (mode == 1) {
          vmax = max(vmax, v);
        } else if (v >= thresh) {
          vec4 s = classify(v, 0.25);
          acc.rgb += (1.0 - acc.a) * s.a * s.rgb;
          acc.a += (1.0 - acc.a) * s.a;
          if (acc.a > 0.99) break;
        }
        t += dt;
      }
      if (mode == 1) {
        if (vmax < thresh) discard;
        acc = vec4(texture(cmap, vec2(vmax, 0.5)).rgb, 1.0);
      }
      if (acc.a < 0.002) discard;
      outColor = acc;  // premultiplied
    }`;

  function b64ToBytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function cmapTexture(stops) {
    // stops: [[pos, "#rrggbb"], ...] -> 256x1 RGBA texture
    const data = new Uint8Array(256 * 4);
    const col = stops.map(([p, h]) => [p, parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      let k = 0;
      while (k < col.length - 2 && x > col[k + 1][0]) k++;
      const [p0, r0, g0, b0] = col[k];
      const [p1, r1, g1, b1] = col[k + 1];
      const f = Math.min(1, Math.max(0, (x - p0) / (p1 - p0 || 1)));
      data.set([r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0), 255], 4 * i);
    }
    const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  }

  function cssGradient(stops) {
    return "linear-gradient(to top, " + stops.map(([p, h]) => `${h} ${(100 * p).toFixed(1)}%`).join(", ") + ")";
  }

  function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) e.style.cssText = style;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function b64ToFloats(s) {
    const b = b64ToBytes(s);
    return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  }

  const POINT_VERT = `
    attribute vec3 color;
    uniform float size;
    varying vec3 vColor;
    void main() {
      vColor = color;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size;
    }`;
  const POINT_FRAG = `
    varying vec3 vColor;
    void main() {
      float r = length(gl_PointCoord - 0.5) * 2.0;
      if (r > 1.0) discard;
      gl_FragColor = vec4(r > 0.72 ? vec3(0.1) : vColor, 1.0);
    }`;

  function render(root, spec) {
    const W = spec.width, H = spec.height;
    root.style.cssText += `position:relative;width:100%;max-width:${W}px;height:${H}px;` +
      "font:12px system-ui,-apple-system,Segoe UI,sans-serif;color:#222;background:#fff;border:1px solid #ddd;border-radius:4px;overflow:hidden;";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.setClearColor(0xffffff, 1);
    root.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 1, 1e7);
    camera.up.set(0, 0, 1);  // must precede OrbitControls, which caches the up axis
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;

    const bounds = new THREE.Box3();
    const layers = [];  // {name, objects: [Object3D], visible, hover?}
    const style = spec.volume_style;

    // ---- volumes (shared style uniforms, one texture each) ----
    const shared = {
      thresh: { value: style.threshold },
      density: { value: style.density },
      gamma: { value: style.gamma },
      mode: { value: 0 },
      sharp: { value: !style.smooth },
    };
    const volumes = spec.volumes.map((v) => {
      const [nx, ny, nz] = v.shape;
      const raw = b64ToBytes(v.data);
      const tex = new THREE.Data3DTexture(raw, nx, ny, nz);
      tex.format = THREE.RedFormat;
      tex.type = THREE.UnsignedByteType;
      tex.unpackAlignment = 1;
      tex.minFilter = tex.magFilter = style.smooth ? THREE.LinearFilter : THREE.NearestFilter;
      tex.needsUpdate = true;
      const size = new THREE.Vector3(nx, ny, nz).multiplyScalar(v.voxel_size);
      const lo = new THREE.Vector3(...v.origin);
      const box3 = new THREE.Box3(lo.clone(), lo.clone().add(size));
      bounds.union(box3);
      const mat = new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: {
          ...shared,
          map: { value: tex },
          cmap: { value: cmapTexture(v.colorscale) },
          shape: { value: new THREE.Vector3(nx, ny, nz) },
        },
        vertexShader: VERT, fragmentShader: FRAG,
        side: THREE.BackSide, transparent: true, premultipliedAlpha: true, depthWrite: false,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
      mesh.scale.copy(size);
      box3.getCenter(mesh.position);
      mesh.renderOrder = 10;
      scene.add(mesh);
      const vol = { spec: v, raw, tex, box3, mesh, visible: v.visible };
      layers.push({ name: v.name, objects: [mesh], visible: v.visible, volume: vol });
      return vol;
    });

    // ---- polylines (one LineSegments2 per layer) ----
    const lineMats = [];
    for (const ln of spec.lines) {
      const pos = b64ToFloats(ln.segments);
      const geo = new THREE.LineSegmentsGeometry();
      geo.setPositions(pos);
      geo.setColors(b64ToFloats(ln.colors));
      const lm = new THREE.LineMaterial({ linewidth: ln.width, vertexColors: true, dashed: !!ln.dashed, dashSize: ln.dash, gapSize: 0.6 * ln.dash });
      const obj = new THREE.LineSegments2(geo, lm);
      if (ln.dashed) obj.computeLineDistances();
      scene.add(obj);
      lineMats.push(lm);
      for (let i = 0; i < pos.length; i += 3) bounds.expandByPoint(new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]));
      layers.push({ name: ln.name, objects: [obj], visible: ln.visible, hover: ln.hover && { pos: b64ToFloats(ln.hover.points), text: ln.hover.text } });
    }

    // ---- markers ----
    for (const pt of spec.points) {
      const pos = b64ToFloats(pt.positions);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(b64ToFloats(pt.colors), 3));
      const mat = new THREE.ShaderMaterial({ uniforms: { size: { value: pt.size * pixelRatio } }, vertexShader: POINT_VERT, fragmentShader: POINT_FRAG });
      const obj = new THREE.Points(geo, mat);
      scene.add(obj);
      for (let i = 0; i < pos.length; i += 3) bounds.expandByPoint(new THREE.Vector3(pos[i], pos[i + 1], pos[i + 2]));
      layers.push({ name: pt.name, objects: [obj], visible: pt.visible, hover: { pos, text: pt.hover } });
    }

    if (bounds.isEmpty()) bounds.set(new THREE.Vector3(-500, -500, -500), new THREE.Vector3(500, 500, 500));
    // Pad line/marker-only extents so content does not sit on the box edges.
    {
      const s = bounds.getSize(new THREE.Vector3());
      const pad = Math.max(s.x, s.y, s.z) * 0.04 + 1;
      if (!volumes.length) bounds.expandByScalar(pad);
    }
    const lo = bounds.min.clone(), hi = bounds.max.clone();
    const size = bounds.getSize(new THREE.Vector3());
    const centre = bounds.getCenter(new THREE.Vector3());
    scene.add(new THREE.Box3Helper(bounds.clone(), 0xc8c8c8));

    // ---- ticked axes along three edges of the bounding box ----
    // x runs along (y=lo, z=lo), y along (x=hi, z=lo), z along (x=lo, y=lo).
    const span = Math.max(size.x, size.y, size.z);
    const nice = [1, 2, 2.5, 5];
    let step = 1;
    outer: for (let e = Math.floor(Math.log10(span / 8)) - 1; e < 12; e++) {
      for (const m of nice) { step = m * 10 ** e; if (span / step <= 8) break outer; }
    }
    const tickLen = 0.02 * span;
    const fmt = (t) => String(+t.toPrecision(10));
    // Labels are placed in screen space each frame: pushed out from the box,
    // perpendicular to the projected axis, by their own size plus a margin.
    const axesInfo = [];
    const labelLayer = el("div", "position:absolute;inset:0;pointer-events:none;overflow:hidden");
    const axisDefs = [
      { a: "x", color: 0xd62728, base: [null, lo.y, lo.z], out: [0, -1, 0] },
      { a: "y", color: 0x2ca02c, base: [hi.x, null, lo.z], out: [1, 0, 0] },
      { a: "z", color: 0x1f77b4, base: [lo.x, lo.y, null], out: [-1, -1, 0] },
    ];
    for (const ax of axisDefs) {
      const k = "xyz".indexOf(ax.a);
      const out = new THREE.Vector3(...ax.out).normalize();
      const at = (val) => new THREE.Vector3(...ax.base.map((b, i) => (i === k ? val : b)));
      const info = { p0: at(lo.getComponent(k)), p1: at(hi.getComponent(k)), ticks: [] };
      const pts = [info.p0, info.p1];
      for (let i = Math.ceil(lo.getComponent(k) / step - 1e-9); i * step <= hi.getComponent(k) + 1e-9 * span; i++) {
        const t = i * step;
        const p = at(t);
        const tip = p.clone().addScaledVector(out, tickLen);
        pts.push(p, tip);
        const lab = el("div", "position:absolute;font-size:11px;color:#444;white-space:nowrap;font-variant-numeric:tabular-nums", fmt(t));
        labelLayer.appendChild(lab);
        info.ticks.push({ p, tip, lab });
      }
      info.title = el("div", "position:absolute;font-size:12px;font-weight:600;white-space:nowrap", `${ax.a} (mm)`);
      info.title.style.color = "#" + ax.color.toString(16).padStart(6, "0");
      labelLayer.appendChild(info.title);
      axesInfo.push(info);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: ax.color })));
    }
    root.appendChild(labelLayer);
    const scr = (p, w, h) => {
      const q = p.clone().project(camera);
      return { x: ((q.x + 1) / 2) * w, y: ((1 - q.y) / 2) * h, z: q.z };
    };
    // Put a label's nearest edge `gap` px from (x, y) along unit direction n.
    const put = (lab, x, y, n, gap) => {
      const bw = lab.offsetWidth, bh = lab.offsetHeight;
      const r = 0.5 * (Math.abs(n.x) * bw + Math.abs(n.y) * bh);
      lab.style.left = x + n.x * (gap + r) - bw / 2 + "px";
      lab.style.top = y + n.y * (gap + r) - bh / 2 + "px";
      return gap + 2 * r;
    };
    function placeLabels() {
      const w = root.clientWidth, h = root.clientHeight;
      const c = scr(centre, w, h);
      for (const ax of axesInfo) {
        const a = scr(ax.p0, w, h), b = scr(ax.p1, w, h);
        const dx = b.x - a.x, dy = b.y - a.y;
        const L = Math.hypot(dx, dy) || 1;
        let n = { x: -dy / L, y: dx / L };
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        if (n.x * (mx - c.x) + n.y * (my - c.y) < 0) n = { x: -n.x, y: -n.y };
        let reach = 0;
        for (const t of ax.ticks) {
          const p = scr(t.p, w, h), q = scr(t.tip, w, h);
          const hide = p.z > 1;
          t.lab.style.display = hide ? "none" : "";
          if (hide) continue;
          // start past whichever end of the tick is further out along n
          const off = Math.max(0, (q.x - p.x) * n.x + (q.y - p.y) * n.y);
          reach = Math.max(reach, off + put(t.lab, p.x, p.y, n, off + 4));
        }
        put(ax.title, mx, my, n, reach + 6);
      }
    }

    // ---- camera ----
    const radius = size.length() * 0.5;
    const resetCamera = () => {
      camera.position.copy(centre).add(new THREE.Vector3(1.0, -1.4, 0.8).normalize().multiplyScalar(radius * 3.9));
      controls.target.copy(centre);
      controls.update();
    };
    resetCamera();

    // ---- overlay UI ----
    const ui = el("div", "position:absolute;left:8px;top:8px;max-height:calc(100% - 40px);overflow:auto;background:rgba(255,255,255,.88);padding:6px 8px;border-radius:4px;line-height:1.7;box-shadow:0 1px 3px rgba(0,0,0,.15)");
    if (spec.title) ui.appendChild(el("div", "font-weight:600;max-width:260px", spec.title));
    const slider = (label, min, max, stp, val, fn) => {
      const row = el("label", "display:flex;align-items:center;gap:6px");
      row.appendChild(el("span", "width:62px", label));
      const s = el("input", "width:110px");
      Object.assign(s, { type: "range", min, max, step: stp, value: val });
      const out = el("span", "width:42px;text-align:right;font-variant-numeric:tabular-nums");
      const upd = () => { out.textContent = fn(parseFloat(s.value)); draw(); };
      s.addEventListener("input", upd);
      row.append(s, out);
      ui.appendChild(row);
      return upd;
    };
    const check = (label, val, fn) => {
      const row = el("label", "display:flex;align-items:center;gap:6px;cursor:pointer");
      const c = el("input");
      c.type = "checkbox"; c.checked = val;
      c.addEventListener("change", () => { fn(c.checked); draw(); });
      row.append(c, el("span", "", label));
      ui.appendChild(row);
      fn(val);
    };
    const inits = [];
    if (volumes.length) {
      inits.push(
        slider("threshold", 0, 0.6, 0.01, style.threshold, (x) => { shared.thresh.value = x; return x.toFixed(2); }),
        slider("density", 0.05, 6, 0.05, style.density, (x) => { shared.density.value = x; return x.toFixed(2); }),
        slider("gamma", 0.3, 3, 0.05, style.gamma, (x) => { shared.gamma.value = x; return x.toFixed(2); }),
      );
      check("smooth (trilinear)", !!style.smooth, (on) => {
        for (const v of volumes) {
          v.tex.minFilter = v.tex.magFilter = on ? THREE.LinearFilter : THREE.NearestFilter;
          v.tex.needsUpdate = true;
        }
        shared.sharp.value = !on;
      });
      check("max-intensity projection", false, (on) => { shared.mode.value = on ? 1 : 0; });
    }
    for (const L of layers) {
      check(L.name, L.visible, (on) => {
        L.visible = on;
        for (const o of L.objects) o.visible = on;
      });
    }
    const reset = el("button", "margin-top:2px;font:inherit;padding:1px 8px;cursor:pointer", "reset view");
    reset.addEventListener("click", () => { resetCamera(); draw(); });
    ui.appendChild(reset);
    root.appendChild(ui);

    // ---- colour bars, right to left ----
    const bars = spec.volumes.map((v) => ({ title: v.colorbar_title, stops: v.colorscale, min: 0, max: v.e_max }))
      .concat(spec.colorbars);
    bars.forEach((b, i) => {
      const cb = el("div", `position:absolute;right:${12 + i * 78}px;top:64px;bottom:40px;width:56px`);
      cb.appendChild(el("div", "position:absolute;bottom:calc(100% + 6px);left:0;width:70px;white-space:normal;line-height:1.15", b.title));
      cb.appendChild(el("div", `position:absolute;left:0;top:0;bottom:0;width:12px;border:1px solid #999;background:${cssGradient(b.stops)}`));
      for (let j = 0; j <= 4; j++) {
        const f = j / 4;
        cb.appendChild(el("div", `position:absolute;left:16px;bottom:calc(${100 * f}% - 7px);font-variant-numeric:tabular-nums`, (+(b.min + f * (b.max - b.min)).toPrecision(3)).toString()));
      }
      root.appendChild(cb);
    });

    const info = el("div", "position:absolute;left:8px;bottom:6px;color:#555;pointer-events:none");
    const vox = volumes.length ? `${volumes.map((v) => v.spec.voxel_size).join("/")} mm voxels · ` : "";
    const baseInfo = `${vox}ticks every ${fmt(step)} mm · drag rotate, scroll zoom, right-drag pan`;
    info.textContent = baseInfo;
    root.appendChild(info);
    const tip = el("div", "position:absolute;display:none;pointer-events:none;background:rgba(30,30,30,.9);color:#fff;padding:3px 6px;border-radius:3px;white-space:pre;font-size:11px;line-height:1.35;z-index:5");
    root.appendChild(tip);

    // ---- hover: nearest marker/step within 10 px, else brightest voxel on the ray ----
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    const tmp = new THREE.Vector3();
    function voxelUnder(o, d) {
      let best = null;
      for (const v of volumes) {
        if (!v.mesh.visible) continue;
        const p0 = new THREE.Vector3();
        if (!new THREE.Ray(o, d).intersectBox(v.box3, p0)) continue;
        const [nx, ny, nz] = v.spec.shape;
        const vs = v.spec.voxel_size, lo = v.box3.min;
        for (let s = 0; s < 4 * (nx + ny + nz); s++) {
          tmp.copy(p0).addScaledVector(d, s * vs * 0.25).sub(lo).divideScalar(vs);
          const ix = Math.floor(tmp.x), iy = Math.floor(tmp.y), iz = Math.floor(tmp.z);
          if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) { if (s > 4) break; else continue; }
          const val = v.raw[ix + nx * (iy + ny * iz)] / 255;
          if (val >= shared.thresh.value && (!best || val > best.u)) best = { u: val, v, idx: [ix, iy, iz] };
        }
      }
      return best;
    }
    let pending = null;
    function hover() {
      const ev = pending; pending = null;
      const r = renderer.domElement.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null, bd = 100;  // squared px
      for (const L of layers) {
        if (!L.visible || !L.hover) continue;
        const pos = L.hover.pos;
        for (let i = 0; i < pos.length / 3; i++) {
          tmp.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]).project(camera);
          if (tmp.z > 1) continue;
          const dx = ((tmp.x + 1) / 2) * r.width - mx, dy = ((1 - tmp.y) / 2) * r.height - my;
          const d2 = dx * dx + dy * dy;
          if (d2 < bd) { bd = d2; best = `${L.name}\n${L.hover.text[i]}`; }
        }
      }
      if (best) {
        tip.textContent = best;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, r.width - tip.offsetWidth - 4) + "px";
        tip.style.top = Math.max(my - tip.offsetHeight - 8, 4) + "px";
        info.textContent = baseInfo;
        return;
      }
      tip.style.display = "none";
      ndc.set((mx / r.width) * 2 - 1, -(my / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = voxelUnder(ray.ray.origin, ray.ray.direction);
      if (!hit) { info.textContent = baseInfo; return; }
      const c = hit.idx.map((k, a) => fmt(+(hit.v.spec.origin[a] + (k + 0.5) * hit.v.spec.voxel_size).toFixed(1)));
      info.textContent = `${hit.v.spec.name}: voxel (${c.join(", ")}) mm · ${hit.v.spec.colorbar_title} ${(hit.u * hit.v.spec.e_max).toPrecision(3)}`;
    }
    renderer.domElement.addEventListener("mousemove", (ev) => {
      if (!pending) requestAnimationFrame(hover);
      pending = ev;
    });
    renderer.domElement.addEventListener("mouseleave", () => { tip.style.display = "none"; info.textContent = baseInfo; });

    // ---- sizing / drawing (render on demand) ----
    function resize() {
      const w = root.clientWidth, h = root.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = w + "px";
      renderer.domElement.style.height = h + "px";
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      for (const lm of lineMats) lm.resolution.set(w, h);
    }
    function draw() { renderer.render(scene, camera); placeLabels(); }
    controls.addEventListener("change", () => { tip.style.display = "none"; draw(); });
    new ResizeObserver(() => { resize(); draw(); }).observe(root);
    inits.forEach((f) => f());
    resize();
    draw();
    root._trackvis = { renderer, scene, camera, controls, draw };
  }

  window.TrackVis = { render };
})();
