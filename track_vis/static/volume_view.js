// track_vis volume viewer: ray-marched voxel image + polyline tracks in three.js.
// Expects THREE (r147 classic build) with OrbitControls and Line2 on window.
// Entry point: TrackVisVolume.render(containerElement, spec). See track_vis/volume.py.
(function () {
  if (window.TrackVisVolume) return;

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

  function render(root, spec) {
    const W = spec.width, H = spec.height;
    root.style.cssText += `position:relative;width:100%;max-width:${W}px;height:${H}px;` +
      "font:12px system-ui,-apple-system,Segoe UI,sans-serif;color:#222;background:#fff;border:1px solid #ddd;border-radius:4px;overflow:hidden;";

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xffffff, 1);
    root.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 1, 1e6);
    camera.up.set(0, 0, 1);  // must precede OrbitControls, which caches the up axis
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;

    // ---- volume ----
    const v = spec.volume;
    const [nx, ny, nz] = v.shape;
    const tex = new THREE.Data3DTexture(b64ToBytes(v.data), nx, ny, nz);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.unpackAlignment = 1;
    tex.minFilter = tex.magFilter = spec.smooth ? THREE.LinearFilter : THREE.NearestFilter;
    tex.needsUpdate = true;

    const size = new THREE.Vector3(nx, ny, nz).multiplyScalar(v.voxel_size);
    const centre = new THREE.Vector3(...v.origin).addScaledVector(size, 0.5);
    const uniforms = {
      map: { value: tex },
      cmap: { value: cmapTexture(spec.colorscale) },
      shape: { value: new THREE.Vector3(nx, ny, nz) },
      thresh: { value: spec.threshold },
      density: { value: spec.density },
      gamma: { value: spec.gamma },
      mode: { value: 0 },
      sharp: { value: !spec.smooth },
    };
    const mat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, uniforms, vertexShader: VERT, fragmentShader: FRAG,
      side: THREE.BackSide, transparent: true, premultipliedAlpha: true, depthWrite: false,
    });
    const box = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
    box.scale.copy(size);
    box.position.copy(centre);
    box.renderOrder = 10;
    scene.add(box);

    const outline = new THREE.Box3Helper(new THREE.Box3().setFromCenterAndSize(centre, size), 0xbbbbbb);
    scene.add(outline);

    // ---- tracks ----
    const lineMats = [];
    const trackObjs = [];
    for (const tr of spec.tracks) {
      const geo = new THREE.LineGeometry();
      geo.setPositions(tr.points.flat());
      geo.setColors(tr.colors.flat());
      const lm = new THREE.LineMaterial({ linewidth: tr.width, vertexColors: true, dashed: !!tr.dashed, dashSize: 20, gapSize: 12 });
      const line = new THREE.Line2(geo, lm);
      if (tr.dashed) line.computeLineDistances();
      line.renderOrder = 0;
      scene.add(line);
      lineMats.push(lm);
      trackObjs.push(line);
    }

    // ---- ticked axes along three edges of the bounding box ----
    // x runs along (y=lo, z=lo), y along (x=hi, z=lo), z along (x=lo, y=lo).
    const lo = new THREE.Vector3(...v.origin);
    const hi = lo.clone().add(size);
    const span = Math.max(size.x, size.y, size.z);
    const step = [50, 100, 200, 250, 500, 1000, 2000, 5000].find((s) => span / s <= 8) || 10000;
    const tickLen = Math.max(0.02 * span, 0.3 * v.voxel_size);
    const tickLabels = [];  // [Vector3 position, element]
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
      const pts = [at(lo.getComponent(k)), at(hi.getComponent(k))];
      for (let t = Math.ceil(lo.getComponent(k) / step) * step; t <= hi.getComponent(k) + 1e-6; t += step) {
        const p = at(t);
        pts.push(p, p.clone().addScaledVector(out, tickLen));
        const lab = el("div", "position:absolute;transform:translate(-50%,-50%);font-size:11px;color:#444;white-space:nowrap;font-variant-numeric:tabular-nums", String(Math.round(t)));
        labelLayer.appendChild(lab);
        tickLabels.push([p.clone().addScaledVector(out, tickLen * 2.6), lab]);
      }
      const mid = at(0.5 * (lo.getComponent(k) + hi.getComponent(k))).addScaledVector(out, tickLen * 5.5);
      const title = el("div", "position:absolute;transform:translate(-50%,-50%);font-size:12px;font-weight:600;white-space:nowrap", `${ax.a} (mm)`);
      title.style.color = "#" + ax.color.toString(16).padStart(6, "0");
      labelLayer.appendChild(title);
      tickLabels.push([mid, title]);
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: ax.color })));
    }
    root.appendChild(labelLayer);
    const projected = new THREE.Vector3();
    function placeLabels() {
      const w = root.clientWidth, h = root.clientHeight;
      for (const [p, lab] of tickLabels) {
        projected.copy(p).project(camera);
        const behind = projected.z > 1;
        lab.style.display = behind ? "none" : "";
        lab.style.left = ((projected.x + 1) / 2) * w + "px";
        lab.style.top = ((1 - projected.y) / 2) * h + "px";
      }
    }

    // ---- camera ----
    const radius = size.length() * 0.5;
    controls.target.copy(centre);
    const resetCamera = () => {
      camera.position.copy(centre).add(new THREE.Vector3(1.0, -1.4, 0.8).normalize().multiplyScalar(radius * 3.9));
      controls.target.copy(centre);
      controls.update();
    };
    resetCamera();

    // ---- overlay UI ----
    const ui = el("div", "position:absolute;left:8px;top:8px;background:rgba(255,255,255,.88);padding:6px 8px;border-radius:4px;line-height:1.7;box-shadow:0 1px 3px rgba(0,0,0,.15)");
    if (spec.title) ui.appendChild(el("div", "font-weight:600", spec.title));
    const slider = (label, min, max, step, val, fn) => {
      const row = el("label", "display:flex;align-items:center;gap:6px");
      row.appendChild(el("span", "width:62px", label));
      const s = el("input", "width:110px");
      Object.assign(s, { type: "range", min, max, step, value: val });
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
    const inits = [
      slider("threshold", 0, 0.6, 0.01, spec.threshold, (x) => { uniforms.thresh.value = x; return x.toFixed(2); }),
      slider("density", 0.05, 6, 0.05, spec.density, (x) => { uniforms.density.value = x; return x.toFixed(2); }),
      slider("gamma", 0.3, 3, 0.05, spec.gamma, (x) => { uniforms.gamma.value = x; return x.toFixed(2); }),
    ];
    check("smooth (trilinear)", !!spec.smooth, (on) => {
      tex.minFilter = tex.magFilter = on ? THREE.LinearFilter : THREE.NearestFilter;
      tex.needsUpdate = true;
      uniforms.sharp.value = !on;
    });
    check("max-intensity projection", false, (on) => { uniforms.mode.value = on ? 1 : 0; });
    spec.tracks.forEach((tr, i) => check(tr.name, tr.visible, (on) => { trackObjs[i].visible = on; }));
    const reset = el("button", "margin-top:2px;font:inherit;padding:1px 8px;cursor:pointer", "reset view");
    reset.addEventListener("click", () => { resetCamera(); draw(); });
    ui.appendChild(reset);
    root.appendChild(ui);

    // colour bar
    const cb = el("div", "position:absolute;right:12px;top:40px;bottom:40px;width:46px;display:flex;flex-direction:column;align-items:flex-start");
    cb.appendChild(el("div", "position:absolute;top:-24px;right:0;white-space:nowrap", spec.colorbar_title));
    const bar = el("div", `position:absolute;left:0;top:0;bottom:0;width:12px;border:1px solid #999;background:${cssGradient(spec.colorscale)}`);
    cb.appendChild(bar);
    for (let i = 0; i <= 4; i++) {
      const f = i / 4;
      cb.appendChild(el("div", `position:absolute;left:16px;bottom:calc(${100 * f}% - 7px);font-variant-numeric:tabular-nums`, (f * spec.e_max).toPrecision(2)));
    }
    root.appendChild(cb);

    const info = el("div", "position:absolute;left:8px;bottom:6px;color:#555;pointer-events:none");
    const baseInfo = `${v.voxel_size} mm voxels · ticks every ${step} mm · drag rotate, scroll zoom, right-drag pan`;
    info.textContent = baseInfo;
    root.appendChild(info);

    // ---- hover: report the brightest voxel under the cursor ----
    const raw = b64ToBytes(v.data);
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    renderer.domElement.addEventListener("mousemove", (ev) => {
      const r = renderer.domElement.getBoundingClientRect();
      ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const o = ray.ray.origin, d = ray.ray.direction;
      const lo = new THREE.Vector3(...v.origin);
      const hit = new THREE.Box3(lo, lo.clone().add(size));
      const p0 = new THREE.Vector3();
      if (!ray.ray.intersectBox(hit, p0)) { info.textContent = baseInfo; return; }
      let best = -1, bi = null;
      const step = v.voxel_size * 0.25;
      for (let s = 0; s < 8 * (nx + ny + nz); s++) {
        const p = p0.clone().addScaledVector(d, s * step).sub(lo).divideScalar(v.voxel_size);
        const ix = Math.floor(p.x), iy = Math.floor(p.y), iz = Math.floor(p.z);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) { if (s > 4) break; else continue; }
        const val = raw[ix + nx * (iy + ny * iz)];
        if (val > best) { best = val; bi = [ix, iy, iz]; }
      }
      if (!bi || best / 255 < uniforms.thresh.value) { info.textContent = baseInfo; return; }
      const c = bi.map((k, a) => (v.origin[a] + (k + 0.5) * v.voxel_size).toFixed(0));
      info.textContent = `voxel (${c.join(", ")}) mm · ${spec.colorbar_title} ${(best / 255 * spec.e_max).toPrecision(3)}`;
    });

    // ---- sizing / drawing (render on demand) ----
    function resize() {
      const w = root.clientWidth, h = root.clientHeight;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = w + "px";
      renderer.domElement.style.height = h + "px";
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      for (const lm of lineMats) lm.resolution.set(w, h);
    }
    function draw() { renderer.render(scene, camera); placeLabels(); }
    controls.addEventListener("change", draw);
    new ResizeObserver(() => { resize(); draw(); }).observe(root);
    inits.forEach((f) => f());
    resize();
    draw();
    root._trackvis = { renderer, scene, camera, controls, uniforms, draw };
  }

  window.TrackVisVolume = { render };
})();
