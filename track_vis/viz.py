import awkward as ak
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import uproot

try:
    from particle import Particle

    def particle_name(pdg: int) -> str:
        try:
            return Particle.from_pdgid(pdg).name
        except Exception:
            return f"PDG {pdg}"

except ImportError:

    def particle_name(pdg: int) -> str:
        return f"PDG {pdg}"


COLORSCALE = "Plasma"
VOXEL_COLORSCALE = "Plasma"

_CUBE_VERT_OFFSETS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=float,
)

_CUBE_FACES = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 3, 7],
        [0, 7, 4],
        [1, 2, 6],
        [1, 6, 5],
    ]
)


def time_colorscale(times: np.ndarray, t_min: float, t_max: float, colorscale: str = COLORSCALE) -> list:
    """Map an array of times to a list of CSS colour strings via *colorscale*."""
    t_range = t_max - t_min or 1.0
    normalized = np.clip((times - t_min) / t_range, 0.0, 1.0).tolist()
    return pc.sample_colorscale(colorscale, normalized)


def voxel_mesh(
    centers: np.ndarray,
    energies: np.ndarray,
    voxel_size: float,
    e_max: float | None = None,
    colorscale: str = VOXEL_COLORSCALE,
    colorbar_title: str = "Energy",
    colorbar_x: float = 1.12,
    opacity: float = 0.35,
    name: str = "Voxels",
    hover_texts: np.ndarray | None = None,
) -> go.Mesh3d:
    """Return a single batched Mesh3d covering all voxel cubes.

    Parameters
    ----------
    centers:
        (N, 3) array of voxel centre positions.
    energies:
        (N,) per-voxel scalar used for colouring.
    voxel_size:
        Edge length of each cube.
    """
    n = len(centers)
    hs = voxel_size / 2
    verts = (centers[:, np.newaxis, :] + _CUBE_VERT_OFFSETS[np.newaxis] * hs).reshape(-1, 3)
    face_offsets = (np.arange(n) * 8)[:, np.newaxis, np.newaxis]
    faces = (_CUBE_FACES[np.newaxis] + face_offsets).reshape(-1, 3)
    intensity = np.repeat(energies, 8)

    kwargs = {}
    if hover_texts is not None:
        kwargs["text"] = np.repeat(hover_texts, 8)
        kwargs["hovertemplate"] = "%{text}<extra>" + name + "</extra>"

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=intensity,
        cmin=0,
        cmax=float(e_max) if e_max is not None else float(energies.max()),
        colorscale=colorscale,
        opacity=opacity,
        showscale=True,
        colorbar=dict(title=colorbar_title, x=colorbar_x),
        name=name,
        **kwargs,
    )


def voxel_cubes(
    centers: np.ndarray,
    energies: np.ndarray,
    voxel_size: float,
    e_max: float | None = None,
    threshold: float = 0.05,
    n_levels: int = 8,
    opacity_range: tuple[float, float] = (0.04, 0.45),
    gamma: float = 1.5,
    colorscale: str = VOXEL_COLORSCALE,
    colorbar_title: str = "Energy (MeV)",
    colorbar_x: float = 1.12,
    fill: float = 0.92,
    name: str = "Voxels",
    hover_texts: np.ndarray | None = None,
) -> list:
    """Return voxel cubes whose opacity and colour both encode energy.

    Plotly only supports one opacity per trace, so voxels are grouped into
    *n_levels* equal-width bins of ``energies / e_max`` and each bin becomes one
    batched ``go.Mesh3d`` with its own opacity. Opacity of level k (bin centre
    u_k in (0, 1]) is ``o_min + (o_max - o_min) * u_k ** gamma``, so faint
    voxels are near-transparent and the hottest are near-solid. Colour uses
    the same energy on a shared ``[0, e_max]`` scale, so the colourbar reads
    off directly and the two cues agree.

    Parameters
    ----------
    centers:
        (N, 3) voxel centre positions (mm).
    energies:
        (N,) per-voxel energy, e.g. reconstructed quenched energy (MeV).
    voxel_size:
        Cube edge length (mm).
    e_max:
        Top of the colour/opacity scale; defaults to ``energies.max()``. Pass
        a common value to compare several events on the same scale.
    threshold:
        Voxels below ``threshold * e_max`` are dropped (MLEM leaves a low-level
        haze over the whole grid that would otherwise fog the view).
    n_levels:
        Number of opacity levels (one trace each). 6-10 reads well.
    opacity_range:
        (min, max) opacity for the lowest and highest level.
    gamma:
        Opacity curve; >1 fades low-energy voxels more strongly.
    fill:
        Fraction of the voxel pitch each cube is drawn at, leaving small gaps
        so neighbouring cubes stay distinguishable.
    """
    centers = np.asarray(centers, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if e_max is None:
        e_max = float(energies.max()) if len(energies) else 0.0
    if e_max <= 0:
        return []

    keep = energies >= threshold * e_max
    centers, energies = centers[keep], energies[keep]
    if hover_texts is not None:
        hover_texts = np.asarray(hover_texts)[keep]

    u = np.clip(energies / e_max, 0.0, 1.0)
    level = np.minimum((u * n_levels).astype(int), n_levels - 1)
    o_min, o_max = opacity_range

    traces = []
    for k in range(n_levels):
        sel = level == k
        if not sel.any():
            continue
        u_k = (k + 0.5) / n_levels
        mesh = voxel_mesh(
            centers=centers[sel],
            energies=energies[sel],
            voxel_size=voxel_size * fill,
            e_max=e_max,
            colorscale=colorscale,
            colorbar_title=colorbar_title,
            colorbar_x=colorbar_x,
            opacity=o_min + (o_max - o_min) * u_k**gamma,
            name=name,
            hover_texts=hover_texts[sel] if hover_texts is not None else None,
        )
        mesh.update(
            showscale=False,
            flatshading=True,
            lighting={"ambient": 0.8, "diffuse": 0.3, "specular": 0.0},
            legendgroup=name,
            showlegend=False,
        )
        traces.append(mesh)

    if traces:
        # One colourbar and one legend entry for the whole group.
        traces[-1].update(showscale=True, showlegend=True)
    return traces


def plot_tracks(
    pdg: np.ndarray,
    pos_x: ak.Array,
    pos_y: ak.Array,
    pos_z: ak.Array,
    time: ak.Array,
    deposited_energy: ak.Array,
    t_min: float,
    t_max: float,
) -> list:
    traces = []
    t_range = t_max - t_min or 1.0
    for i in range(len(pdg)):
        x = ak.to_numpy(pos_x[i])
        y = ak.to_numpy(pos_y[i])
        z = ak.to_numpy(pos_z[i])
        t = ak.to_numpy(time[i])
        dep_step = ak.to_numpy(deposited_energy[i])
        dep_total = float(dep_step.sum())
        name = particle_name(int(pdg[i]))
        colors = pc.sample_colorscale(COLORSCALE, (t - t_min) / t_range)
        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                name=f"{name} ({dep_total:.2f} MeV dep.)",
                line=dict(width=6, color=colors),
                text=[f"t={tt:.1f} ns<br>dep={d:.3f} MeV" for tt, d in zip(t, dep_step)],
                hovertemplate="%{text}<extra>" + name + "</extra>",
            )
        )
    return traces


def plot_voxels(
    pos_x: ak.Array,
    pos_y: ak.Array,
    pos_z: ak.Array,
    deposited_energy: ak.Array,
    voxel_size: float = 100.0,
) -> list:
    voxel_energy: dict[tuple, float] = {}
    for i in range(len(pos_x)):
        x = ak.to_numpy(pos_x[i])
        y = ak.to_numpy(pos_y[i])
        z = ak.to_numpy(pos_z[i])
        dep = ak.to_numpy(deposited_energy[i])
        for px, py, pz, d in zip(x, y, z, dep):
            if d <= 0:
                continue
            key = (int(px // voxel_size), int(py // voxel_size), int(pz // voxel_size))
            voxel_energy[key] = voxel_energy.get(key, 0.0) + d

    if not voxel_energy:
        return []

    keys = np.array(list(voxel_energy.keys()))
    centers = (keys + 0.5) * voxel_size
    energies = np.array(list(voxel_energy.values()))
    hover_texts = np.array([f"dep={e:.4f} MeV" for e in energies])

    return voxel_cubes(
        centers=centers,
        energies=energies,
        voxel_size=voxel_size,
        threshold=0.0,
        colorbar_title="Voxel dep. energy (MeV)",
        name=f"Voxels ({voxel_size:.0f} mm)",
        hover_texts=hover_texts,
    )


def plot_truth_vertices(
    pdg: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    energy: np.ndarray,
    t_min: float,
    t_max: float,
) -> list:
    pos = np.stack([ak.to_numpy(x), ak.to_numpy(y), ak.to_numpy(z)], axis=-1)
    t = ak.to_numpy(t)
    energy = ak.to_numpy(energy)
    pdgs = ak.to_numpy(pdg)

    by_species: dict[str, list[int]] = {}
    for i, pdg in enumerate(pdgs):
        name = particle_name(int(pdg))
        by_species.setdefault(name, []).append(i)

    traces = []
    for species, indices in by_species.items():
        idx = np.array(indices)
        show_scale = species == next(iter(by_species))
        traces.append(
            go.Scatter3d(
                x=pos[idx, 0],
                y=pos[idx, 1],
                z=pos[idx, 2],
                mode="markers",
                name=f"Vertex: {species}",
                marker=dict(
                    size=10,
                    color=t[idx],
                    symbol="diamond",
                    line=dict(width=1, color="black"),
                    colorscale=COLORSCALE,
                    cmin=t_min,
                    cmax=t_max,
                    showscale=show_scale,
                    colorbar=dict(title="Time (ns)", x=1.0),
                ),
                text=[f"{species}<br>E={energy[i]:.2f} MeV<br>t={t[i]:.1f} ns" for i in idx],
                hovertemplate="%{text}<extra>Vertex</extra>",
            )
        )
    return traces


def load_event(path: str, index: int) -> tuple[dict, dict]:
    """Read one event from an imaging_reco ntuple; returns (tracks, vertices) kwargs dicts."""
    tree = uproot.open(path)["output"]
    event = tree.arrays(
        [
            "trackPDG",
            "trackPosX",
            "trackPosY",
            "trackPosZ",
            "trackTime",
            "trackDep",
            "mcpdgs",
            "mcxs",
            "mcys",
            "mczs",
            "mcts",
            "mckes",
        ],
        entry_start=index,
        entry_stop=index + 1,
    )[0]

    tracks = {
        "pdg": event["trackPDG"],
        "pos_x": event["trackPosX"],
        "pos_y": event["trackPosY"],
        "pos_z": event["trackPosZ"],
        "time": event["trackTime"],
        "deposited_energy": event["trackDep"],
    }
    vertices = {
        "pdg": event["mcpdgs"],
        "x": event["mcxs"],
        "y": event["mcys"],
        "z": event["mczs"],
        "t": event["mcts"],
        "energy": event["mckes"],
    }
    return tracks, vertices
