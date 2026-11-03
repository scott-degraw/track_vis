// track_vis browser renderer: draws one scene with three.js.
//
// Input: the JSON from Scene.to_json() in track_vis/scene.py, which documents
// every field. Python calls TrackVis.mount(elementId, scene, scriptUrls), or
// TrackVis.mountMenu(elementId, scenes, labels, scriptUrls) for a drop-down of
// scenes (see track_vis/browser.py). mount waits until the element is near the viewport
// (browsers cap live WebGL contexts at ~16 per page), loads three.js r147 from
// scriptUrls if given (else it must already be on the page), then calls render().
//
// Rendering is on demand: draw() runs only when the camera or a control changes.
// See README.md in this folder for the file layout.
(function () {
  if (window.TrackVis) return;

  // ---------------------------------------------------------------- shaders

  // Volumes are drawn as the back faces of a unit box [-0.5, 0.5]^3 scaled to
  // the volume's size; each fragment marches the camera ray through the box.
  const VOLUME_VERT = `
    out vec3 vOrigin;
    out vec3 vDirection;
    void main() {
      vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
      vDirection = position - vOrigin;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  // Emission-absorption compositing, front to back, premultiplied output.
  // density is the optical depth of one voxel pitch at the top of the scale.
  const VOLUME_FRAG = `
    precision highp float;
    precision highp sampler3D;
    uniform sampler3D map;    // uint8 image, value / e_max in [0, 1]
    uniform sampler2D cmap;   // 256 x 1 colour scale
    uniform vec3 shape;       // voxels along x, y, z
    uniform float thresh;
    uniform float density;
    uniform float gamma;
    uniform bool mip;         // maximum-intensity projection instead of compositing
    uniform bool sharp;       // exact voxel traversal instead of trilinear sampling
    in vec3 vOrigin;
    in vec3 vDirection;
    out vec4 outColor;

    const int MAX_STEPS = 4096;
    const float OPAQUE = 0.995;

    // Entry and exit ray parameters of the unit box (entry > exit: miss).
    vec2 hitBox(vec3 o, vec3 d) {
      vec3 t0 = (vec3(-0.5) - o) / d;
      vec3 t1 = (vec3(0.5) - o) / d;
      vec3 tmin = min(t0, t1);
      vec3 tmax = max(t0, t1);
      return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
    }

    // Add a path of len voxel pitches through value v behind what is in acc.
    void composite(inout vec4 acc, float v, float len) {
      vec3 c = texture(cmap, vec2(v, 0.5)).rgb;
      float a = 1.0 - exp(-density * pow(v, gamma) * len);
      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a += (1.0 - acc.a) * a;
    }

    // Exact traversal (Amanatides & Woo): every voxel is a uniform cube,
    // composited with its true path length, so there is no sampling noise.
    // Returns the composited colour, or the max value in .r when mip is set.
    vec4 marchVoxels(vec3 o, vec3 d, vec2 b, float voxPerT) {
      vec3 D = d * shape;  // ray direction in voxel units
      D = mix(D, vec3(1e-6), vec3(lessThan(abs(D), vec3(1e-6))));  // avoid division by zero
      vec3 P0 = (o + d * b.x + 0.5) * shape;
      ivec3 n = ivec3(shape);
      ivec3 cell = clamp(ivec3(floor(P0)), ivec3(0), n - 1);
      ivec3 stp = ivec3(sign(D));
      vec3 tDelta = abs(1.0 / D);
      vec3 tNext = b.x + (vec3(cell) + max(vec3(stp), 0.0) - P0) / D;
      float t = b.x;
      vec4 acc = vec4(0.0);
      for (int i = 0; i < MAX_STEPS; i++) {
        float tExit = min(min(tNext.x, tNext.y), min(tNext.z, b.y));
        float v = texelFetch(map, cell, 0).r;
        if (mip) acc.r = max(acc.r, v);
        else if (v >= thresh) composite(acc, v, (tExit - t) * voxPerT);
        if (acc.a > OPAQUE) break;
        t = tExit;
        if (t >= b.y) break;
        if (tNext.x <= tNext.y && tNext.x <= tNext.z) { cell.x += stp.x; tNext.x += tDelta.x; }
        else if (tNext.y <= tNext.z) { cell.y += stp.y; tNext.y += tDelta.y; }
        else { cell.z += stp.z; tNext.z += tDelta.z; }
        if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, n))) break;
      }
      return acc;
    }

    // Trilinear sampling at quarter-voxel steps, jittered per pixel against banding.
    vec4 marchSmooth(vec3 o, vec3 d, vec2 b, float voxPerT) {
      float dt = 0.25 / voxPerT;
      float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
      vec4 acc = vec4(0.0);
      for (int i = 0; i < MAX_STEPS; i++) {
        float t = b.x + dt * (float(i) + jitter);
        if (t > b.y) break;
        float v = texture(map, o + d * t + 0.5).r;
        if (mip) acc.r = max(acc.r, v);
        else if (v >= thresh) composite(acc, v, 0.25);
        if (acc.a > OPAQUE) break;
      }
      return acc;
    }

    void main() {
      vec3 d = normalize(vDirection);
      vec2 b = hitBox(vOrigin, d);
      if (b.x > b.y) discard;
      b.x = max(b.x, 0.0);  // camera inside the box
      float voxPerT = length(d * shape);
      vec4 acc = sharp ? marchVoxels(vOrigin, d, b, voxPerT) : marchSmooth(vOrigin, d, b, voxPerT);
      if (mip) {
        if (acc.r < thresh) discard;
        acc = vec4(texture(cmap, vec2(acc.r, 0.5)).rgb, 1.0);
      }
      if (acc.a < 0.002) discard;
      outColor = acc;
    }`;

  // Round markers with a dark rim, a fixed size in pixels.
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

  const AXES = [
    // Each axis runs along one box edge; `out` points away from the box for ticks.
    { name: "x", color: 0xd62728, edge: (lo, hi) => [null, lo.y, lo.z], out: [0, -1, 0] },
    { name: "y", color: 0x2ca02c, edge: (lo, hi) => [hi.x, null, lo.z], out: [1, 0, 0] },
    { name: "z", color: 0x1f77b4, edge: (lo, hi) => [lo.x, lo.y, null], out: [-1, -1, 0] },
  ];
  const HOVER_RADIUS_PX = 10;
  const PANEL_STYLE = "position:absolute;left:8px;top:8px;max-height:calc(100% - 40px);overflow:auto;" +
    "background:rgba(255,255,255,.88);padding:6px 8px;border-radius:4px;line-height:1.7;box-shadow:0 1px 3px rgba(0,0,0,.15)";

  // ---------------------------------------------------------------- helpers

  function el(tag, style, text) {
    const e = document.createElement(tag);
    if (style) e.style.cssText = style;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function b64ToBytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  function b64ToFloats(s) {
    const b = b64ToBytes(s);
    return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  }

  function hexToRgb(h) {
    return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  }

  // [[pos, "#rrggbb"], ...] -> 256 x 1 RGBA texture, linearly interpolated.
  function colorscaleTexture(stops) {
    const col = stops.map(([p, h]) => [p, ...hexToRgb(h)]);
    const data = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const x = i / 255;
      let k = 0;
      while (k < col.length - 2 && x > col[k + 1][0]) k++;
      const [p0, ...c0] = col[k];
      const [p1, ...c1] = col[k + 1];
      const f = Math.min(1, Math.max(0, (x - p0) / (p1 - p0 || 1)));
      data.set([...c0.map((c, j) => c + f * (c1[j] - c)), 255], 4 * i);
    }
    const tex = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
    tex.magFilter = tex.minFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return tex;
  }

  function cssGradient(stops) {
    return "linear-gradient(to top, " + stops.map(([p, h]) => `${h} ${(100 * p).toFixed(1)}%`).join(", ") + ")";
  }

  function expandByPositions(box, pos) {
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.length; i += 3) box.expandByPoint(v.set(pos[i], pos[i + 1], pos[i + 2]));
  }

  // Round tick spacing (1, 2, 2.5 or 5 x 10^k) giving at most 8 ticks over span.
  function tickStep(span) {
    for (let e = Math.floor(Math.log10(span / 8)) - 1; e < 12; e++) {
      for (const m of [1, 2, 2.5, 5]) if (span / (m * 10 ** e) <= 8) return m * 10 ** e;
    }
    return 10 ** 12;
  }

  const fmt = (x) => String(+x.toPrecision(10));

  // ---------------------------------------------------------------- scene layers
  // One builder per list in the scene JSON (volumes, lines, points). Each adds
  // objects to ctx.scene, grows ctx.bounds and pushes a layer
  // {name, objects, visible, hover?: {pos: Float32Array, text: string[]}}.

  function addVolume(ctx, v) {
    const [nx, ny, nz] = v.shape;
    const raw = b64ToBytes(v.data);
    const tex = new THREE.Data3DTexture(raw, nx, ny, nz);
    tex.format = THREE.RedFormat;
    tex.type = THREE.UnsignedByteType;
    tex.unpackAlignment = 1;
    tex.minFilter = tex.magFilter = ctx.spec.volume_style.smooth ? THREE.LinearFilter : THREE.NearestFilter;
    tex.needsUpdate = true;

    const size = new THREE.Vector3(nx, ny, nz).multiplyScalar(v.voxel_size);
    const lo = new THREE.Vector3(...v.origin);
    const box = new THREE.Box3(lo.clone(), lo.clone().add(size));
    ctx.bounds.union(box);

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      // The style uniforms are shared objects, so one slider drives every volume.
      uniforms: {
        ...ctx.styleUniforms,
        map: { value: tex },
        cmap: { value: colorscaleTexture(v.colorscale) },
        shape: { value: new THREE.Vector3(nx, ny, nz) },
      },
      vertexShader: VOLUME_VERT,
      fragmentShader: VOLUME_FRAG,
      side: THREE.BackSide,
      transparent: true,
      premultipliedAlpha: true,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    mesh.scale.copy(size);
    box.getCenter(mesh.position);
    mesh.renderOrder = 10;  // after opaque lines and markers
    ctx.scene.add(mesh);

    const volume = { spec: v, raw, tex, box, mesh };
    ctx.volumes.push(volume);
    ctx.layers.push({ name: v.name, objects: [mesh], visible: v.visible });
  }

  function addLines(ctx, ln) {
    const pos = b64ToFloats(ln.segments);
    const geometry = new THREE.LineSegmentsGeometry();
    geometry.setPositions(pos);
    geometry.setColors(b64ToFloats(ln.colors));
    const material = new THREE.LineMaterial({
      linewidth: ln.width, vertexColors: true, dashed: ln.dashed, dashSize: ln.dash, gapSize: 0.6 * ln.dash,
    });
    const lines = new THREE.LineSegments2(geometry, material);
    if (ln.dashed) lines.computeLineDistances();
    ctx.scene.add(lines);
    ctx.lineMaterials.push(material);  // needs the canvas resolution on resize
    expandByPositions(ctx.bounds, pos);
    const hover = ln.hover && { pos: b64ToFloats(ln.hover.points), text: ln.hover.text };
    ctx.layers.push({ name: ln.name, objects: [lines], visible: ln.visible, hover });
  }

  function addPoints(ctx, pt) {
    const pos = b64ToFloats(pt.positions);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(b64ToFloats(pt.colors), 3));
    const material = new THREE.ShaderMaterial({
      uniforms: { size: { value: pt.size * ctx.pixelRatio } },
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
    });
    const points = new THREE.Points(geometry, material);
    ctx.scene.add(points);
    expandByPositions(ctx.bounds, pos);
    ctx.layers.push({ name: pt.name, objects: [points], visible: pt.visible, hover: { pos, text: pt.hover } });
  }

  // ---------------------------------------------------------------- axes

  // Ticked x/y/z axes along three edges of ctx.bounds. Tick labels are HTML,
  // re-placed in screen space every frame by the returned function.
  function addAxes(ctx) {
    const { bounds, scene, camera, root } = ctx;
    const lo = bounds.min, hi = bounds.max;
    const span = Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
    const step = tickStep(span);
    const tickLen = 0.02 * span;
    const layer = el("div", "position:absolute;inset:0;pointer-events:none;overflow:hidden");
    root.appendChild(layer);

    const axes = AXES.map((ax) => {
      const k = "xyz".indexOf(ax.name);
      const out = new THREE.Vector3(...ax.out).normalize();
      const at = (val) => new THREE.Vector3(...ax.edge(lo, hi).map((b, i) => (i === k ? val : b)));
      const axis = { start: at(lo.getComponent(k)), end: at(hi.getComponent(k)), ticks: [] };
      const pts = [axis.start, axis.end];
      const first = Math.ceil(lo.getComponent(k) / step - 1e-9);
      for (let i = first; i * step <= hi.getComponent(k) + 1e-9 * span; i++) {
        const base = at(i * step);
        const tip = base.clone().addScaledVector(out, tickLen);
        pts.push(base, tip);
        const label = el("div", "position:absolute;font-size:11px;color:#444;white-space:nowrap;font-variant-numeric:tabular-nums", fmt(i * step));
        layer.appendChild(label);
        axis.ticks.push({ base, tip, label });
      }
      axis.title = el("div", "position:absolute;font-size:12px;font-weight:600;white-space:nowrap", `${ax.name} (mm)`);
      axis.title.style.color = "#" + ax.color.toString(16).padStart(6, "0");
      layer.appendChild(axis.title);
      scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: ax.color })));
      return axis;
    });

    const toScreen = (p, w, h) => {
      const q = p.clone().project(camera);
      return { x: ((q.x + 1) / 2) * w, y: ((1 - q.y) / 2) * h, behind: q.z > 1 };
    };
    // Place a label so its nearest edge is `gap` px from (x, y) along unit vector n;
    // return the distance from (x, y) to its far edge.
    const place = (label, x, y, n, gap) => {
      const w = label.offsetWidth, h = label.offsetHeight;
      const r = 0.5 * (Math.abs(n.x) * w + Math.abs(n.y) * h);
      label.style.left = x + n.x * (gap + r) - w / 2 + "px";
      label.style.top = y + n.y * (gap + r) - h / 2 + "px";
      return gap + 2 * r;
    };
    const centre = bounds.getCenter(new THREE.Vector3());

    function placeLabels() {
      const w = root.clientWidth, h = root.clientHeight;
      const c = toScreen(centre, w, h);
      for (const axis of axes) {
        // Push labels out perpendicular to the projected axis, away from the box centre.
        const a = toScreen(axis.start, w, h), b = toScreen(axis.end, w, h);
        const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        let n = { x: -(b.y - a.y) / len, y: (b.x - a.x) / len };
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
        if (n.x * (mx - c.x) + n.y * (my - c.y) < 0) n = { x: -n.x, y: -n.y };
        let reach = 0;
        for (const t of axis.ticks) {
          const p = toScreen(t.base, w, h), q = toScreen(t.tip, w, h);
          t.label.style.display = p.behind ? "none" : "";
          if (p.behind) continue;
          const tickOut = Math.max(0, (q.x - p.x) * n.x + (q.y - p.y) * n.y);
          reach = Math.max(reach, tickOut + place(t.label, p.x, p.y, n, tickOut + 4));
        }
        place(axis.title, mx, my, n, reach + 6);
      }
    }
    return { step, placeLabels };
  }

  // ---------------------------------------------------------------- overlay UI

  function addPanel(ctx, resetCamera) {
    const { spec, styleUniforms: u, volumes, layers } = ctx;
    const panel = el("div", PANEL_STYLE);
    if (spec.title) panel.appendChild(el("div", "font-weight:600;max-width:260px", spec.title));

    const slider = (label, min, max, step, value, apply) => {
      const row = el("label", "display:flex;align-items:center;gap:6px");
      const input = el("input", "width:110px");
      Object.assign(input, { type: "range", min, max, step, value });
      const readout = el("span", "width:42px;text-align:right;font-variant-numeric:tabular-nums");
      const update = () => {
        const x = parseFloat(input.value);
        apply(x);
        readout.textContent = x.toFixed(2);
        ctx.draw();
      };
      input.addEventListener("input", update);
      row.append(el("span", "width:62px", label), input, readout);
      panel.appendChild(row);
      update();
    };
    const checkbox = (label, checked, apply) => {
      const row = el("label", "display:flex;align-items:center;gap:6px;cursor:pointer");
      const input = el("input");
      Object.assign(input, { type: "checkbox", checked });
      input.addEventListener("change", () => { apply(input.checked); ctx.draw(); });
      row.append(input, el("span", "", label));
      panel.appendChild(row);
      apply(checked);
    };

    if (volumes.length) {
      const s = spec.volume_style;
      slider("threshold", 0, 0.6, 0.01, s.threshold, (x) => { u.thresh.value = x; });
      slider("density", 0.05, 6, 0.05, s.density, (x) => { u.density.value = x; });
      slider("gamma", 0.3, 3, 0.05, s.gamma, (x) => { u.gamma.value = x; });
      checkbox("smooth (trilinear)", s.smooth, (on) => {
        for (const v of volumes) {
          v.tex.minFilter = v.tex.magFilter = on ? THREE.LinearFilter : THREE.NearestFilter;
          v.tex.needsUpdate = true;
        }
        u.sharp.value = !on;
      });
      checkbox("max-intensity projection", false, (on) => { u.mip.value = on; });
    }
    for (const layer of layers) {
      checkbox(layer.name, layer.visible, (on) => {
        layer.visible = on;
        for (const o of layer.objects) o.visible = on;
      });
    }
    const reset = el("button", "margin-top:2px;font:inherit;padding:1px 8px;cursor:pointer", "reset view");
    reset.addEventListener("click", () => { resetCamera(); ctx.draw(); });
    panel.appendChild(reset);
    ctx.root.appendChild(panel);
  }

  // One colour bar per volume plus spec.colorbars (the shared time bar), right to left.
  function addColorbars(ctx) {
    const bars = ctx.spec.volumes
      .map((v) => ({ title: v.colorbar_title, stops: v.colorscale, min: 0, max: v.e_max }))
      .concat(ctx.spec.colorbars);
    bars.forEach((bar, i) => {
      const box = el("div", `position:absolute;right:${12 + i * 78}px;top:64px;bottom:40px;width:56px`);
      box.appendChild(el("div", "position:absolute;bottom:calc(100% + 6px);left:0;width:70px;line-height:1.15", bar.title));
      box.appendChild(el("div", `position:absolute;left:0;top:0;bottom:0;width:12px;border:1px solid #999;background:${cssGradient(bar.stops)}`));
      for (let j = 0; j <= 4; j++) {
        const f = j / 4;
        const value = fmt(+(bar.min + f * (bar.max - bar.min)).toPrecision(3));
        box.appendChild(el("div", `position:absolute;left:16px;bottom:calc(${100 * f}% - 7px);font-variant-numeric:tabular-nums`, value));
      }
      ctx.root.appendChild(box);
    });
  }

  // ---------------------------------------------------------------- hover

  // Tooltip for the nearest visible hover point within HOVER_RADIUS_PX; otherwise
  // the status line reports the brightest above-threshold voxel under the cursor.
  function addHover(ctx, status, baseStatus) {
    const { root, renderer, camera, layers, volumes, styleUniforms } = ctx;
    const tip = el("div", "position:absolute;display:none;pointer-events:none;background:rgba(30,30,30,.9);color:#fff;" +
      "padding:3px 6px;border-radius:3px;white-space:pre;font-size:11px;line-height:1.35;z-index:5");
    root.appendChild(tip);
    const v3 = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();

    function nearestHoverPoint(mx, my, w, h) {
      let best = null, bestD2 = HOVER_RADIUS_PX ** 2;
      for (const layer of layers) {
        if (!layer.visible || !layer.hover) continue;
        const pos = layer.hover.pos;
        for (let i = 0; i < pos.length / 3; i++) {
          v3.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]).project(camera);
          if (v3.z > 1) continue;
          const dx = ((v3.x + 1) / 2) * w - mx, dy = ((1 - v3.y) / 2) * h - my;
          if (dx * dx + dy * dy < bestD2) { bestD2 = dx * dx + dy * dy; best = `${layer.name}\n${layer.hover.text[i]}`; }
        }
      }
      return best;
    }

    function brightestVoxel(ray) {
      let best = null;
      for (const vol of volumes) {
        if (!vol.mesh.visible) continue;
        const entry = new THREE.Vector3();
        if (!ray.intersectBox(vol.box, entry)) continue;
        const [nx, ny, nz] = vol.spec.shape;
        const size = vol.spec.voxel_size;
        for (let s = 0; s < 4 * (nx + ny + nz); s++) {
          v3.copy(entry).addScaledVector(ray.direction, s * size * 0.25).sub(vol.box.min).divideScalar(size);
          const [ix, iy, iz] = [Math.floor(v3.x), Math.floor(v3.y), Math.floor(v3.z)];
          if (ix < 0 || iy < 0 || iz < 0 || ix >= nx || iy >= ny || iz >= nz) {
            if (s > 4) break;  // left the box (the first steps can sit on its surface)
            continue;
          }
          const u = vol.raw[ix + nx * (iy + ny * iz)] / 255;
          if (u >= styleUniforms.thresh.value && (!best || u > best.u)) best = { u, vol, idx: [ix, iy, iz] };
        }
      }
      return best;
    }

    function update(ev) {
      const r = renderer.domElement.getBoundingClientRect();
      const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      const text = nearestHoverPoint(mx, my, r.width, r.height);
      if (text) {
        tip.textContent = text;
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 12, r.width - tip.offsetWidth - 4) + "px";
        tip.style.top = Math.max(my - tip.offsetHeight - 8, 4) + "px";
        status.textContent = baseStatus;
        return;
      }
      tip.style.display = "none";
      raycaster.setFromCamera(new THREE.Vector2((mx / r.width) * 2 - 1, -(my / r.height) * 2 + 1), camera);
      const hit = brightestVoxel(raycaster.ray);
      if (!hit) { status.textContent = baseStatus; return; }
      const { spec } = hit.vol;
      const centre = hit.idx.map((k, a) => fmt(+(spec.origin[a] + (k + 0.5) * spec.voxel_size).toFixed(1)));
      status.textContent = `${spec.name}: voxel (${centre.join(", ")}) mm · ${spec.colorbar_title} ${(hit.u * spec.e_max).toPrecision(3)}`;
    }

    let pending = null;  // coalesce mouse moves to one update per frame
    renderer.domElement.addEventListener("mousemove", (ev) => {
      if (!pending) requestAnimationFrame(() => { update(pending); pending = null; });
      pending = ev;
    });
    const hide = () => { tip.style.display = "none"; status.textContent = baseStatus; };
    renderer.domElement.addEventListener("mouseleave", hide);
    return hide;
  }

  // ---------------------------------------------------------------- render

  function render(root, spec) {
    root.style.cssText += `position:relative;width:100%;max-width:${spec.width}px;height:${spec.height}px;` +
      "font:12px system-ui,-apple-system,Segoe UI,sans-serif;color:#222;background:#fff;border:1px solid #ddd;border-radius:4px;overflow:hidden;";

    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(pixelRatio);
    renderer.setClearColor(0xffffff, 1);
    renderer.domElement.style.display = "block";
    root.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(35, 1, 1, 1e7);
    camera.up.set(0, 0, 1);  // must precede OrbitControls, which caches the up axis
    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    const style = spec.volume_style;
    const ctx = {
      root, spec, renderer, camera, pixelRatio,
      scene: new THREE.Scene(),
      bounds: new THREE.Box3(),
      layers: [],
      volumes: [],
      lineMaterials: [],
      styleUniforms: {
        thresh: { value: style.threshold },
        density: { value: style.density },
        gamma: { value: style.gamma },
        mip: { value: false },
        sharp: { value: !style.smooth },
      },
      draw: () => {},
    };
    spec.volumes.forEach((v) => addVolume(ctx, v));
    spec.lines.forEach((ln) => addLines(ctx, ln));
    spec.points.forEach((pt) => addPoints(ctx, pt));

    if (spec.bounds) {
      // A fixed view box from Python: layers outside it are still drawn.
      ctx.bounds.set(new THREE.Vector3(...spec.bounds[0]), new THREE.Vector3(...spec.bounds[1]));
    } else if (ctx.bounds.isEmpty()) {
      ctx.bounds.set(new THREE.Vector3(-500, -500, -500), new THREE.Vector3(500, 500, 500));
    } else if (!ctx.volumes.length) {
      // Lines and markers only: pad so they do not sit on the box edges.
      const size = ctx.bounds.getSize(new THREE.Vector3());
      ctx.bounds.expandByScalar(0.04 * Math.max(size.x, size.y, size.z) + 1);
    }
    ctx.scene.add(new THREE.Box3Helper(ctx.bounds.clone(), 0xc8c8c8));
    const axes = addAxes(ctx);

    const centre = ctx.bounds.getCenter(new THREE.Vector3());
    const distance = ctx.bounds.getSize(new THREE.Vector3()).length() * 0.5 * 3.9;
    const viewDir = new THREE.Vector3(1.0, -1.4, 0.8).normalize();
    const resetCamera = () => {
      camera.position.copy(centre).addScaledVector(viewDir, distance);
      controls.target.copy(centre);
      controls.update();
    };
    resetCamera();

    const voxelSizes = ctx.volumes.length ? `${ctx.volumes.map((v) => v.spec.voxel_size).join("/")} mm voxels · ` : "";
    const baseStatus = `${voxelSizes}ticks every ${fmt(axes.step)} mm · drag rotate, scroll zoom, right-drag pan`;
    const status = el("div", "position:absolute;left:8px;bottom:6px;color:#555;pointer-events:none", baseStatus);
    root.appendChild(status);
    const hideHover = addHover(ctx, status, baseStatus);
    addColorbars(ctx);

    function resize() {
      const w = root.clientWidth, h = root.clientHeight;
      if (!w || !h) return;  // hidden, e.g. in an inactive tab
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      for (const m of ctx.lineMaterials) m.resolution.set(w, h);
    }
    ctx.draw = () => {
      renderer.render(ctx.scene, camera);
      axes.placeLabels();
    };
    addPanel(ctx, resetCamera);  // after ctx.draw exists: controls apply their initial state
    controls.addEventListener("change", () => { hideHover(); ctx.draw(); });
    const resizeObserver = new ResizeObserver(() => { resize(); ctx.draw(); });
    resizeObserver.observe(root);
    // Free the GPU context and empty root, so another scene can be drawn there.
    ctx.dispose = () => {
      resizeObserver.disconnect();
      controls.dispose();
      ctx.scene.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
      ctx.volumes.forEach((v) => v.tex.dispose());
      renderer.dispose();
      renderer.forceContextLoss();
      root.replaceChildren();
      delete root._trackvis;
    };
    resize();
    ctx.draw();
    root._trackvis = ctx;  // for debugging from the console, and for dispose
  }

  // ---------------------------------------------------------------- mount

  let scriptsLoaded = null;  // one shared load per page

  function loadScripts(urls) {
    if (!urls) return Promise.resolve();
    scriptsLoaded = scriptsLoaded || urls.reduce((prev, url) => prev.then(() => new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = url;
      s.onload = resolve;
      s.onerror = () => reject(new Error(`could not load ${url}`));
      document.head.appendChild(s);
    })), Promise.resolve());
    return scriptsLoaded;
  }

  // Run draw() once three.js is loaded and root is near the viewport.
  function whenVisible(root, urls, draw) {
    let started = false;
    const start = () => {
      if (started) return;
      started = true;
      loadScripts(urls)
        .then(draw)
        .catch((e) => { root.textContent = `track_vis: ${e.message}`; });
    };
    if (!("IntersectionObserver" in window)) return start();
    new IntersectionObserver((entries, observer) => {
      if (entries.some((e) => e.isIntersecting)) { observer.disconnect(); start(); }
    }, { rootMargin: "300px" }).observe(root);
  }

  function mount(id, spec, urls) {
    const root = document.getElementById(id);
    whenVisible(root, urls, () => render(root, spec));
  }

  // Several scenes sharing one view, picked from a drop-down menu. Only the
  // shown scene holds a WebGL context; switching disposes it and draws the next.
  function mountMenu(id, specs, labels, urls) {
    const root = document.getElementById(id);
    const select = el("select", "display:block;max-width:100%;margin:0 0 6px;font:13px system-ui,-apple-system,Segoe UI,sans-serif");
    labels.forEach((label, i) => {
      const option = el("option", "", label);
      option.value = String(i);
      select.appendChild(option);
    });
    const view = el("div");
    root.replaceChildren(select, view);
    const show = (i) => {
      if (view._trackvis) view._trackvis.dispose();
      render(view, specs[i]);
    };
    select.addEventListener("change", () => show(Number(select.value)));
    whenVisible(root, urls, () => show(0));
  }

  window.TrackVis = { mount, mountMenu, render };
})();
