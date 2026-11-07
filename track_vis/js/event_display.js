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


  // Round markers a fixed size in pixels: the fill, a ring in the surface colour
  // (which separates overlapping markers) and a thin anti-aliased edge.
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
    uniform float size;   // diameter in device pixels
    uniform float dpr;    // device pixels per CSS pixel
    uniform vec3 rim;
    uniform vec3 edge;
    varying vec3 vColor;
    void main() {
      float R = 0.5 * size;
      float d = length(gl_PointCoord - 0.5) * size;
      vec3 c = mix(vColor, rim, smoothstep(R - 3.2 * dpr, R - 2.4 * dpr, d));
      c = mix(c, edge, smoothstep(R - 1.4 * dpr, R - 0.8 * dpr, d));
      float a = 1.0 - smoothstep(R - 0.8 * dpr, R, d);
      if (a <= 0.0) discard;
      gl_FragColor = vec4(c, a);
    }`;

  // ---------------------------------------------------------------- theme
  // Colours drawn by WebGL (numbers) per theme; the page elements use the CSS
  // variables in STYLE below, keyed on the same data-theme attribute.
  const THEMES = {
    light: {
      halo: 0xfafbfc, box: 0xd3d8e0, grid: 0xe9ecf1, axis: 0x9aa2af,
      rim: [1, 1, 1], edge: [0.2, 0.23, 0.29],
    },
    dark: {
      halo: 0x0e121a, box: 0x323a48, grid: 0x1a1f29, axis: 0x5b6475,
      rim: [0.055, 0.07, 0.1], edge: [0.78, 0.81, 0.87],
    },
  };
  const HOVER_RADIUS_PX = 10;
  const HALO_PX = 3;  // extra width of the background-coloured outline behind lines
  const HALO_DEPTH = 0.03;  // how far behind its line the outline sits, in scene radii

  const STYLE = `
.tv-root{--tv-ink:#1d2330;--tv-ink2:#5a6273;--tv-ink3:#8a92a2;--tv-border:#e2e5eb;--tv-bg1:#ffffff;--tv-bg2:#f2f4f7;
  --tv-card:rgba(255,255,255,.78);--tv-card-border:rgba(20,28,45,.08);--tv-accent:#3d6fe0;--tv-track:#d9dde4;
  --tv-btn:rgba(255,255,255,.9);--tv-btn-hover:#f0f2f6;--tv-tip:rgba(24,29,40,.94);--tv-tip-ink:#f3f5f8;
  --tv-shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px rgba(16,24,40,.08);
  position:relative;box-sizing:border-box;font:12px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:var(--tv-ink);background:radial-gradient(130% 100% at 50% 0%,var(--tv-bg1),var(--tv-bg2));
  border:1px solid var(--tv-border);border-radius:12px;overflow:hidden;-webkit-font-smoothing:antialiased}
.tv-root[data-theme=dark]{--tv-ink:#e6e9ef;--tv-ink2:#a3abba;--tv-ink3:#6e7788;--tv-border:#222834;--tv-bg1:#161b25;--tv-bg2:#0a0d13;
  --tv-card:rgba(20,24,33,.78);--tv-card-border:rgba(255,255,255,.07);--tv-accent:#7c9dff;--tv-track:#2c3340;
  --tv-btn:rgba(30,35,46,.9);--tv-btn-hover:#2a303c;--tv-tip:rgba(236,239,244,.96);--tv-tip-ink:#161b24;
  --tv-shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.35)}
.tv-root *{box-sizing:border-box}
.tv-root:fullscreen{max-width:none!important;width:100vw!important;height:100vh!important;border-radius:0;border:0}
.tv-card{background:var(--tv-card);border:1px solid var(--tv-card-border);border-radius:10px;box-shadow:var(--tv-shadow);
  -webkit-backdrop-filter:blur(10px) saturate(1.2);backdrop-filter:blur(10px) saturate(1.2)}
.tv-btn{font:inherit;font-weight:500;color:var(--tv-ink);background:var(--tv-btn);border:1px solid var(--tv-card-border);
  border-radius:8px;padding:4px 10px;cursor:pointer;display:inline-flex;align-items:center;gap:6px;line-height:1.3;
  box-shadow:0 1px 2px rgba(16,24,40,.06)}
.tv-btn:hover:not(:disabled){background:var(--tv-btn-hover)}
.tv-btn:disabled{opacity:.4;cursor:default}
.tv-btn:focus-visible,.tv-switch input:focus-visible+span,.tv-slider input:focus-visible{outline:2px solid var(--tv-accent);outline-offset:2px}
.tv-btn svg{width:14px;height:14px;flex:none}
.tv-icon{padding:5px}
.tv-chev{transition:transform .15s}
.tv-btn[aria-expanded=true] .tv-chev{transform:rotate(180deg)}
.tv-header{position:absolute;left:12px;top:12px;right:220px;display:flex;align-items:center;gap:10px;pointer-events:none}
.tv-header>*{pointer-events:auto}
.tv-title{font-size:13px;font-weight:600;letter-spacing:-.005em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  pointer-events:none;text-shadow:0 0 6px var(--tv-bg1)}
.tv-panel{position:absolute;left:12px;top:50px;width:264px;max-height:calc(100% - 96px);overflow:auto;padding:10px 12px}
.tv-section{font-size:10.5px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--tv-ink3);margin:2px 0 6px}
.tv-section:not(:first-child){margin-top:12px;padding-top:10px;border-top:1px solid var(--tv-card-border)}
.tv-slider{display:grid;grid-template-columns:62px 1fr 34px;align-items:center;gap:8px;margin:3px 0;color:var(--tv-ink2)}
.tv-slider input{width:100%;margin:0;accent-color:var(--tv-accent);height:16px}
.tv-slider output{text-align:right;font-variant-numeric:tabular-nums;color:var(--tv-ink)}
.tv-switch{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer;min-height:24px}
.tv-switch input{position:absolute;opacity:0;width:1px;height:1px;margin:0}
.tv-switch>span:first-of-type{flex:none;width:26px;height:15px;border-radius:8px;background:var(--tv-track);position:relative;transition:background .15s}
.tv-switch>span:first-of-type::after{content:"";position:absolute;left:2px;top:2px;width:11px;height:11px;border-radius:50%;
  background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .15s}
.tv-switch input:checked+span{background:var(--tv-accent)}
.tv-switch input:checked+span::after{transform:translateX(11px)}
.tv-switch .tv-name{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tv-switch .tv-detail{color:var(--tv-ink3);margin-left:4px}
.tv-swatch{flex:none;display:inline-block}
.tv-swatch.line{width:16px;height:4px;border-radius:2px}
.tv-swatch.point{width:10px;height:10px;border-radius:50%;box-shadow:0 0 0 1.5px var(--tv-bg1),0 0 0 2.5px var(--tv-ink3)}
.tv-swatch.volume{width:12px;height:12px;border-radius:3px}
.tv-swatch.dashed{-webkit-mask:repeating-linear-gradient(90deg,#000 0 5px,transparent 5px 8px);mask:repeating-linear-gradient(90deg,#000 0 5px,transparent 5px 8px)}
.tv-panel .tv-btn{margin-top:10px}
.tv-bars{position:absolute;right:12px;top:12px;width:188px;padding:9px 12px 7px;display:flex;flex-direction:column;gap:9px;pointer-events:none}
.tv-bar-title{font-weight:500;color:var(--tv-ink2);margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tv-bar-ramp{height:8px;border-radius:4px;box-shadow:inset 0 0 0 1px rgba(127,127,127,.18)}
.tv-bar-ticks{display:flex;justify-content:space-between;margin-top:3px;font-size:11px;color:var(--tv-ink2);font-variant-numeric:tabular-nums}
.tv-status{position:absolute;left:12px;bottom:10px;right:96px;color:var(--tv-ink3);font-size:11px;pointer-events:none;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tv-status.tv-active{color:var(--tv-ink)}
.tv-tools{position:absolute;right:12px;bottom:10px;display:flex;gap:6px}
.tv-ticks{position:absolute;inset:0;pointer-events:none;overflow:hidden}
.tv-tick{position:absolute;font-size:10.5px;color:var(--tv-ink3);white-space:nowrap;font-variant-numeric:tabular-nums}
.tv-axis-title{position:absolute;font-size:11px;font-weight:600;color:var(--tv-ink2);white-space:nowrap}
.tv-tip{position:absolute;display:none;pointer-events:none;background:var(--tv-tip);color:var(--tv-tip-ink);padding:6px 9px;
  border-radius:8px;white-space:pre;font-size:11px;line-height:1.45;z-index:5;box-shadow:0 4px 16px rgba(0,0,0,.2);
  font-variant-numeric:tabular-nums}
.tv-tip b{display:flex;align-items:center;gap:6px;font-weight:600;margin-bottom:1px}
.tv-tip .tv-dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.tv-menu{display:flex;align-items:center;gap:6px;margin:0 0 8px;font:12px/1.4 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.tv-menu .tv-btn{border-color:rgba(20,28,45,.14)}
.tv-select{font:inherit;font-size:13px;font-weight:500;color:#1d2330;background:#fff;border:1px solid rgba(20,28,45,.14);border-radius:8px;
  padding:5px 10px;max-width:100%;min-width:0;flex:0 1 auto;box-shadow:0 1px 2px rgba(16,24,40,.06);cursor:pointer}
.tv-menu[data-theme=dark] .tv-select,.tv-menu[data-theme=dark] .tv-btn{color:#e6e9ef;background:#1a1f29;border-color:#2c3340}
.tv-menu[data-theme=dark] .tv-btn:hover{background:#2a303c}
`;

  const ICONS = {
    sliders: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M2 4h7M13 4h1M2 12h1M7 12h7"/><circle cx="11" cy="4" r="1.8"/><circle cx="5" cy="12" r="1.8"/></svg>',
    chevron: '<svg class="tv-chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4.5 6.5 8 10l3.5-3.5"/></svg>',
    reset: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 8a5.5 5.5 0 1 0 1.7-4"/><path d="M2.5 2.5v3h3"/></svg>',
    expand: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10"/></svg>',
    prev: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3.5 5.5 8l4.5 4.5"/></svg>',
    next: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5"/></svg>',
  };

  // The stylesheet is added once per document.
  function ensureStyle() {
    if (document.getElementById("track-vis-style")) return;
    const s = document.createElement("style");
    s.id = "track-vis-style";
    s.textContent = STYLE;
    document.head.appendChild(s);
  }

  // ---------------------------------------------------------------- helpers

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function button(cls, html, label) {
    const b = el("button", "tv-btn " + (cls || ""));
    b.type = "button";
    b.innerHTML = html;
    if (label) { b.title = label; b.setAttribute("aria-label", label); }
    return b;
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

  function cssGradient(stops, dir = "to right") {
    return `linear-gradient(${dir}, ` + stops.map(([p, h]) => `${h} ${(100 * p).toFixed(1)}%`).join(", ") + ")";
  }

  const rgbCss = (c, i = 0) => `rgb(${[0, 1, 2].map((k) => Math.round(255 * c[3 * i + k])).join(",")})`;

  // CSS background for a legend swatch: one colour, or a few colours sampled along
  // the layer (e.g. tracks coloured by time).
  function swatchFill(colors) {
    const n = colors.length / 3;
    if (!n) return "transparent";
    const picks = [0, 0.33, 0.66, 1].map((f) => Math.round(f * (n - 1)));
    const css = [...new Set(picks.map((i) => rgbCss(colors, i)))];
    return css.length === 1 ? css[0] : `linear-gradient(to right, ${css.join(", ")})`;
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

  // "PDG 11 (3 tracks, 1.80 MeV dep.)" -> ["PDG 11", "3 tracks, 1.80 MeV dep."]
  function splitName(name) {
    const m = /^(.*\S)\s+\(([^()]*)\)$/.exec(name);
    return m ? [m[1], m[2]] : [name, ""];
  }

  // ---------------------------------------------------------------- scene layers
  // One builder per list in the scene JSON (volumes, lines, points). Each adds
  // objects to ctx.scene, grows ctx.bounds and pushes a layer
  // {name, objects, visible, swatch, hover?: {pos: Float32Array, text: string[], colors?}}.

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
    const swatch = { kind: "volume", fill: cssGradient(v.colorscale, "135deg") };
    ctx.layers.push({ name: v.name, objects: [mesh], visible: v.visible, swatch });
  }

  function addLines(ctx, ln) {
    const pos = b64ToFloats(ln.segments);
    const colors = b64ToFloats(ln.colors);
    const geometry = new THREE.LineSegmentsGeometry();
    geometry.setPositions(pos);
    geometry.setColors(colors);
    const dash = { dashed: ln.dashed, dashSize: ln.dash, gapSize: 0.6 * ln.dash };
    const material = new THREE.LineMaterial({ linewidth: ln.width, vertexColors: true, ...dash });
    const lines = new THREE.LineSegments2(geometry, material);
    // A wider outline in the background colour, drawn just behind the line, so
    // crossing tracks stay separate and nearer tracks occlude farther ones.
    // The outline is pushed a little away from the camera (see placeHalos), so it
    // never hides the next steps of its own track.
    const haloMaterial = new THREE.LineMaterial({ linewidth: ln.width + HALO_PX, color: ctx.theme.halo, ...dash });
    const halo = new THREE.LineSegments2(geometry, haloMaterial);
    ctx.halos.push(halo);
    if (ln.dashed) lines.computeLineDistances();
    ctx.scene.add(halo, lines);
    ctx.lineMaterials.push(material, haloMaterial);  // need the canvas resolution on resize
    expandByPositions(ctx.bounds, pos);
    const hover = ln.hover && { pos: b64ToFloats(ln.hover.points), text: ln.hover.text };
    const swatch = { kind: "line", dashed: ln.dashed, fill: swatchFill(colors) };
    ctx.layers.push({ name: ln.name, objects: [halo, lines], visible: ln.visible, hover, swatch });
  }

  function addPoints(ctx, pt) {
    const pos = b64ToFloats(pt.positions);
    const colors = b64ToFloats(pt.colors);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const material = new THREE.ShaderMaterial({
      uniforms: {
        size: { value: pt.size * ctx.pixelRatio },
        dpr: { value: ctx.pixelRatio },
        rim: { value: new THREE.Vector3(...ctx.theme.rim) },
        edge: { value: new THREE.Vector3(...ctx.theme.edge) },
      },
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      transparent: true,  // anti-aliased edge; drawn before the volumes (renderOrder)
    });
    const points = new THREE.Points(geometry, material);
    ctx.scene.add(points);
    expandByPositions(ctx.bounds, pos);
    const swatch = { kind: "point", fill: swatchFill(colors) };
    ctx.layers.push({ name: pt.name, objects: [points], visible: pt.visible, swatch, hover: { pos, text: pt.hover, colors } });
  }

  // ---------------------------------------------------------------- axes

  // Ticked x/y/z axes along three edges of ctx.bounds, a faint grid on the floor
  // and the box outline. Tick labels are HTML, re-placed in screen space every
  // frame by the returned function.
  const AXES = [
    // Each axis runs along one box edge; `out` points away from the box for ticks.
    { name: "x", edge: (lo, hi) => [null, lo.y, lo.z], out: [0, -1, 0] },
    { name: "y", edge: (lo, hi) => [hi.x, null, lo.z], out: [1, 0, 0] },
    { name: "z", edge: (lo, hi) => [lo.x, lo.y, null], out: [-1, -1, 0] },
  ];

  function addAxes(ctx) {
    const { bounds, scene, camera, root, theme } = ctx;
    const lo = bounds.min, hi = bounds.max;
    const span = Math.max(...bounds.getSize(new THREE.Vector3()).toArray());
    const step = tickStep(span);
    const tickLen = 0.015 * span;
    const layer = el("div", "tv-ticks");
    root.appendChild(layer);
    const ticksAlong = (k) => {
      const out = [];
      for (let i = Math.ceil(lo.getComponent(k) / step - 1e-9); i * step <= hi.getComponent(k) + 1e-9 * span; i++) out.push(i * step);
      return out;
    };

    // Floor grid at the tick positions.
    const grid = [];
    for (const x of ticksAlong(0)) grid.push(new THREE.Vector3(x, lo.y, lo.z), new THREE.Vector3(x, hi.y, lo.z));
    for (const y of ticksAlong(1)) grid.push(new THREE.Vector3(lo.x, y, lo.z), new THREE.Vector3(hi.x, y, lo.z));
    scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(grid), new THREE.LineBasicMaterial({ color: theme.grid })));
    scene.add(new THREE.Box3Helper(bounds.clone(), theme.box));

    const pts = [];
    const axes = AXES.map((ax) => {
      const k = "xyz".indexOf(ax.name);
      const out = new THREE.Vector3(...ax.out).normalize();
      const at = (val) => new THREE.Vector3(...ax.edge(lo, hi).map((b, i) => (i === k ? val : b)));
      const axis = { start: at(lo.getComponent(k)), end: at(hi.getComponent(k)), ticks: [] };
      pts.push(axis.start, axis.end);
      for (const val of ticksAlong(k)) {
        const base = at(val);
        const tip = base.clone().addScaledVector(out, tickLen);
        pts.push(base, tip);
        const label = el("div", "tv-tick", fmt(val));
        layer.appendChild(label);
        axis.ticks.push({ base, tip, label });
      }
      axis.title = el("div", "tv-axis-title", `${ax.name} (mm)`);
      layer.appendChild(axis.title);
      return axis;
    });
    scene.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: theme.axis })));

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

  // Title and "Options" button in the top-left corner; the button shows the panel
  // of volume controls and the layer list (which doubles as the legend).
  function addPanel(ctx, resetCamera) {
    const { spec, styleUniforms: u, volumes, layers, root } = ctx;
    const header = el("div", "tv-header");
    const toggle = button("", ICONS.sliders + "<span>Options</span>" + ICONS.chevron);
    header.appendChild(toggle);
    if (spec.title) {
      const title = el("div", "tv-title", spec.title);
      title.title = spec.title;
      header.appendChild(title);
    }
    const panel = el("div", "tv-panel tv-card");
    // The panel starts hidden so it doesn't cover the view on small screens. The
    // open/closed choice is kept on root, so a DisplayMenu remembers it when another
    // event is picked.
    const setOpen = (open) => {
      root._trackvisPanelOpen = open;
      panel.style.display = open ? "" : "none";
      toggle.setAttribute("aria-expanded", String(open));
      toggle.title = open ? "hide the controls" : "show the controls";
    };
    toggle.addEventListener("click", () => setOpen(!root._trackvisPanelOpen));

    const slider = (label, min, max, step, value, apply) => {
      const row = el("label", "tv-slider");
      const input = el("input");
      Object.assign(input, { type: "range", min, max, step, value });
      const readout = el("output");
      const update = () => {
        const x = parseFloat(input.value);
        apply(x);
        readout.textContent = x.toFixed(2);
        ctx.draw();
      };
      input.addEventListener("input", update);
      row.append(el("span", "", label), input, readout);
      panel.appendChild(row);
      update();
    };
    const toggleRow = (label, checked, apply, swatch) => {
      const row = el("label", "tv-switch");
      const input = el("input");
      Object.assign(input, { type: "checkbox", checked });
      input.addEventListener("change", () => { apply(input.checked); ctx.draw(); });
      row.append(input, el("span"));
      if (swatch) {
        const s = el("span", `tv-swatch ${swatch.kind}${swatch.dashed ? " dashed" : ""}`);
        s.style.background = swatch.fill;
        row.appendChild(s);
      }
      const [main, detail] = splitName(label);
      const name = el("span", "tv-name", main);
      if (detail) name.appendChild(el("span", "tv-detail", detail));
      name.title = label;
      row.appendChild(name);
      panel.appendChild(row);
      apply(checked);
    };

    if (volumes.length) {
      const s = spec.volume_style;
      panel.appendChild(el("div", "tv-section", "Volume rendering"));
      slider("threshold", 0, 0.6, 0.01, s.threshold, (x) => { u.thresh.value = x; });
      slider("density", 0.05, 6, 0.05, s.density, (x) => { u.density.value = x; });
      slider("gamma", 0.3, 3, 0.05, s.gamma, (x) => { u.gamma.value = x; });
      toggleRow("smooth (trilinear)", s.smooth, (on) => {
        for (const v of volumes) {
          v.tex.minFilter = v.tex.magFilter = on ? THREE.LinearFilter : THREE.NearestFilter;
          v.tex.needsUpdate = true;
        }
        u.sharp.value = !on;
      });
      toggleRow("max-intensity projection", false, (on) => { u.mip.value = on; });
    }
    panel.appendChild(el("div", "tv-section", "Layers"));
    for (const layer of layers) {
      toggleRow(layer.name, layer.visible, (on) => {
        layer.visible = on;
        for (const o of layer.objects) o.visible = on;
      }, layer.swatch);
    }
    const reset = button("", ICONS.reset + "<span>Reset view</span>");
    reset.addEventListener("click", () => { resetCamera(); ctx.draw(); });
    panel.appendChild(reset);
    setOpen(Boolean(root._trackvisPanelOpen));
    root.append(header, panel);
  }

  // Reset and full-screen buttons in the bottom-right corner.
  function addTools(ctx, resetCamera) {
    const tools = el("div", "tv-tools");
    const reset = button("tv-icon", ICONS.reset, "Reset the camera");
    reset.addEventListener("click", () => { resetCamera(); ctx.draw(); });
    tools.appendChild(reset);
    if (ctx.root.requestFullscreen) {
      const full = button("tv-icon", ICONS.expand, "Full screen");
      full.addEventListener("click", () => {
        if (document.fullscreenElement === ctx.root) document.exitFullscreen();
        else ctx.root.requestFullscreen().catch(() => {});
      });
      tools.appendChild(full);
    }
    ctx.root.appendChild(tools);
  }

  // One horizontal colour bar per volume plus spec.colorbars (the shared time bar),
  // stacked in a card in the top-right corner.
  function addColorbars(ctx) {
    const bars = ctx.spec.volumes
      .map((v) => ({ title: v.colorbar_title, stops: v.colorscale, min: 0, max: v.e_max }))
      .concat(ctx.spec.colorbars);
    if (!bars.length) return;
    const card = el("div", "tv-bars tv-card");
    for (const bar of bars) {
      const box = el("div");
      box.appendChild(el("div", "tv-bar-title", bar.title));
      const ramp = el("div", "tv-bar-ramp");
      ramp.style.background = cssGradient(bar.stops);
      box.appendChild(ramp);
      const ticks = el("div", "tv-bar-ticks");
      for (const f of [0, 0.5, 1]) ticks.appendChild(el("span", "", fmt(+(bar.min + f * (bar.max - bar.min)).toPrecision(3))));
      box.appendChild(ticks);
      card.appendChild(box);
    }
    ctx.root.appendChild(card);
  }

  // ---------------------------------------------------------------- hover

  // Tooltip for the nearest visible hover point within HOVER_RADIUS_PX; otherwise
  // the status line reports the brightest above-threshold voxel under the cursor.
  function addHover(ctx, status, baseStatus) {
    const { root, renderer, camera, layers, volumes, styleUniforms } = ctx;
    const tip = el("div", "tv-tip");
    root.appendChild(tip);
    const v3 = new THREE.Vector3();
    const raycaster = new THREE.Raycaster();
    const setStatus = (text, active) => {
      status.textContent = text;
      status.classList.toggle("tv-active", active);
    };

    function nearestHoverPoint(mx, my, w, h) {
      let best = null, bestD2 = HOVER_RADIUS_PX ** 2;
      for (const layer of layers) {
        if (!layer.visible || !layer.hover) continue;
        const pos = layer.hover.pos;
        for (let i = 0; i < pos.length / 3; i++) {
          v3.set(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]).project(camera);
          if (v3.z > 1) continue;
          const dx = ((v3.x + 1) / 2) * w - mx, dy = ((1 - v3.y) / 2) * h - my;
          if (dx * dx + dy * dy < bestD2) { bestD2 = dx * dx + dy * dy; best = { layer, i }; }
        }
      }
      return best;
    }

    function showTip({ layer, i }) {
      const head = el("b");
      if (layer.hover.colors) {
        const dot = el("span", "tv-dot");
        dot.style.background = rgbCss(layer.hover.colors, i);
        head.appendChild(dot);
      }
      head.appendChild(document.createTextNode(splitName(layer.name)[0]));
      tip.replaceChildren(head, document.createTextNode(layer.hover.text[i]));
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
      const near = nearestHoverPoint(mx, my, r.width, r.height);
      if (near) {
        showTip(near);
        tip.style.display = "block";
        tip.style.left = Math.min(mx + 14, r.width - tip.offsetWidth - 6) + "px";
        tip.style.top = Math.max(my - tip.offsetHeight - 10, 6) + "px";
        setStatus(baseStatus, false);
        return;
      }
      tip.style.display = "none";
      raycaster.setFromCamera(new THREE.Vector2((mx / r.width) * 2 - 1, -(my / r.height) * 2 + 1), camera);
      const hit = brightestVoxel(raycaster.ray);
      if (!hit) { setStatus(baseStatus, false); return; }
      const { spec } = hit.vol;
      const centre = hit.idx.map((k, a) => fmt(+(spec.origin[a] + (k + 0.5) * spec.voxel_size).toFixed(1)));
      setStatus(`${spec.name}: voxel (${centre.join(", ")}) mm · ${spec.colorbar_title} ${(hit.u * spec.e_max).toPrecision(3)}`, true);
    }

    let pending = null;  // coalesce mouse moves to one update per frame
    renderer.domElement.addEventListener("mousemove", (ev) => {
      if (!pending) requestAnimationFrame(() => { update(pending); pending = null; });
      pending = ev;
    });
    const hide = () => { tip.style.display = "none"; setStatus(baseStatus, false); };
    renderer.domElement.addEventListener("mouseleave", hide);
    return hide;
  }

  // ---------------------------------------------------------------- render

  function render(root, spec) {
    ensureStyle();
    const themeName = THEMES[spec.theme] ? spec.theme : "light";
    root.classList.add("tv-root");
    root.dataset.theme = themeName;
    root.style.cssText += `width:100%;max-width:${spec.width}px;height:${spec.height}px;`;

    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    // A transparent canvas over the CSS background gradient; the volumes write
    // premultiplied colour, which is what the page compositor expects.
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(pixelRatio);
    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.display = "block";
    root.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(32, 1, 1, 1e7);
    camera.up.set(0, 0, 1);  // must precede OrbitControls, which caches the up axis
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;

    const style = spec.volume_style;
    const ctx = {
      root, spec, renderer, camera, pixelRatio,
      theme: THEMES[themeName],
      scene: new THREE.Scene(),
      bounds: new THREE.Box3(),
      layers: [],
      volumes: [],
      lineMaterials: [],
      halos: [],
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
    const axes = addAxes(ctx);

    // Frame the box's bounding sphere, with room for the tick labels.
    const centre = ctx.bounds.getCenter(new THREE.Vector3());
    const radius = 0.5 * ctx.bounds.getSize(new THREE.Vector3()).length();
    const distance = (1.12 * radius) / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
    const viewDir = new THREE.Vector3(1.0, -1.4, 0.8).normalize();
    const resetCamera = () => {
      camera.position.copy(centre).addScaledVector(viewDir, distance);
      controls.target.copy(centre);
      controls.update();
    };
    resetCamera();

    const voxelSizes = ctx.volumes.length ? `${ctx.volumes.map((v) => v.spec.voxel_size).join("/")} mm voxels · ` : "";
    const baseStatus = `${voxelSizes}grid ${fmt(axes.step)} mm · drag to rotate, scroll to zoom, right-drag to pan`;
    const status = el("div", "tv-status", baseStatus);
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
    // Scale each line outline about the camera by 1 + depth / distance: it projects
    // to the same pixels but sits HALO_DEPTH of the scene size farther away.
    const haloDepth = HALO_DEPTH * radius;
    const placeHalos = () => {
      const s = 1 + haloDepth / camera.position.distanceTo(controls.target);
      for (const h of ctx.halos) {
        h.scale.setScalar(s);
        h.position.copy(camera.position).multiplyScalar(1 - s);
      }
    };
    ctx.draw = () => {
      placeHalos();
      renderer.render(ctx.scene, camera);
      axes.placeLabels();
    };
    addPanel(ctx, resetCamera);  // after ctx.draw exists: controls apply their initial state
    addTools(ctx, resetCamera);
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

  // three.min.js is a UMD build: with an AMD loader on the page (require.js, which
  // Jupyter and Quarto pages load) it would register as a module instead of setting
  // window.THREE, so the loader's define is hidden until the scripts have run.
  function loadScripts(urls) {
    if (!urls) return Promise.resolve();
    if (!scriptsLoaded) {
      const define = window.define;
      window.define = undefined;
      scriptsLoaded = urls.reduce((prev, url) => prev.then(() => new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = url;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`could not load ${url}`));
        document.head.appendChild(s);
      })), Promise.resolve());
      scriptsLoaded.finally(() => { window.define = define; });
    }
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

  // Several scenes sharing one view, picked from a drop-down menu with previous /
  // next buttons. Only the shown scene holds a WebGL context; switching disposes it
  // and draws the next.
  function mountMenu(id, specs, labels, urls) {
    ensureStyle();
    const root = document.getElementById(id);
    const bar = el("nav", "tv-menu");  // not a div: the view stays the first div inside root
    bar.dataset.theme = specs[0]?.theme === "dark" ? "dark" : "light";
    const prev = button("tv-icon", ICONS.prev, "Previous");
    const next = button("tv-icon", ICONS.next, "Next");
    const select = el("select", "tv-select");
    select.setAttribute("aria-label", "Event");
    labels.forEach((label, i) => {
      const option = el("option", "", label);
      option.value = String(i);
      select.appendChild(option);
    });
    bar.append(prev, select, next);
    const view = el("div");
    root.replaceChildren(bar, view);
    const show = (i) => {
      select.value = String(i);
      prev.disabled = i === 0;
      next.disabled = i === specs.length - 1;
      if (view._trackvis) view._trackvis.dispose();
      render(view, specs[i]);
    };
    select.addEventListener("change", () => show(Number(select.value)));
    prev.addEventListener("click", () => show(Math.max(0, Number(select.value) - 1)));
    next.addEventListener("click", () => show(Math.min(specs.length - 1, Number(select.value) + 1)));
    whenVisible(root, urls, () => show(0));
  }

  window.TrackVis = { mount, mountMenu, render };
})();
