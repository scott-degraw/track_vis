"""Reading events from SNO+ ``imaging_reco`` ntuples, and PDG code names."""

from __future__ import annotations

import uproot

try:
    from particle import Particle

    def particle_name(pdg: int) -> str:
        """PDG code to particle name (falls back to ``"PDG <code>"``)."""
        try:
            return Particle.from_pdgid(pdg).name
        except Exception:  # noqa: BLE001 - any lookup failure falls back to the code
            return f"PDG {pdg}"

except ImportError:

    def particle_name(pdg: int) -> str:
        """PDG code to particle name (install the ``particle`` extra for names)."""
        return f"PDG {pdg}"


def load_event(path: str, index: int) -> tuple[dict, dict]:
    """Read one event from an imaging_reco ntuple; returns (tracks, vertices) kwargs dicts.

    ``tracks`` unpacks into ``EventDisplay.add_tracks`` / ``add_deposits`` and
    ``vertices`` into ``EventDisplay.add_vertices``.
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
