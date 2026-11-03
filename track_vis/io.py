"""
io.py

Read events from SNO+ ``imaging_reco`` ntuples and name PDG codes.

`load_event` returns columnar tracks and vertices that unpack straight into
`EventDisplay.add_tracks`, `add_deposits` and `add_vertices`.
`particle_name` uses the optional ``particle`` package when installed.
"""

from __future__ import annotations

import uproot

try:
    from particle import Particle

    def particle_name(pdg: int) -> str:
        """
        Return the particle name for a PDG code.

        Parameters
        ----------
        pdg : int
            PDG Monte Carlo particle code.

        Returns
        -------
        str
            Name from the ``particle`` package, or ``"PDG <code>"`` if the
            code is unknown.
        """
        try:
            return Particle.from_pdgid(pdg).name
        except Exception:  # noqa: BLE001 - any lookup failure falls back to the code
            return f"PDG {pdg}"

except ImportError:

    def particle_name(pdg: int) -> str:
        """
        Return a placeholder name for a PDG code.

        Install the ``particle`` extra (``uv sync --extra particle``) to get
        real particle names.

        Parameters
        ----------
        pdg : int
            PDG Monte Carlo particle code.

        Returns
        -------
        str
            ``"PDG <code>"``.
        """
        return f"PDG {pdg}"


def load_event(path: str, index: int) -> tuple[dict, dict]:
    """
    Read one event from an ``imaging_reco`` ntuple.

    Parameters
    ----------
    path : str
        Path to the ROOT file; events are read from its ``output`` tree.
    index : int
        Entry number of the event.

    Returns
    -------
    tracks : dict
        Keyword arguments for `EventDisplay.add_tracks`: ``pdg`` (flat) and
        jagged ``pos_x``, ``pos_y``, ``pos_z``, ``time``,
        ``deposited_energy`` (track x step).
    vertices : dict
        Keyword arguments for `EventDisplay.add_vertices`: flat ``pdg``,
        ``x``, ``y``, ``z``, ``t``, ``energy``.
    """
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
