"""EventDisplay builds the right Scene layers (no JSON or HTML involved)."""

import numpy as np
import pytest

from track_vis import DisplayMenu, EventDisplay


def test_volume_from_sparse_voxels_is_padded():
    d = EventDisplay().add_volume(centers=[[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]], energies=[1.0, 3.0], voxel_size=10.0)
    (vol,) = d.scene.volumes
    assert vol.grid.shape == (4, 3, 3)  # 2 x 1 x 1 plus one empty voxel each side
    np.testing.assert_allclose(vol.origin, [-10.0, -10.0, -10.0])
    assert vol.e_max == 3.0


def test_tracks_grouped_by_species_with_time_colorbar(tracks):
    d = EventDisplay().add_tracks(**tracks)
    electrons, gamma = d.scene.lines
    assert "2 tracks" in electrons.name and "1 track," in gamma.name
    assert len(electrons.segments) == 2 * (2 + 1)  # 2 + 1 segments, two end points each
    assert len(electrons.hover_text) == 5  # one per step
    (bar,) = d.scene.colorbars
    assert bar.title == "Time (ns)" and bar.vmax == pytest.approx(3.0)


def test_vertices_share_one_time_colorbar(tracks, vertices):
    d = EventDisplay().add_tracks(**tracks, t_min=0, t_max=3).add_vertices(**vertices, t_min=0, t_max=3)
    assert [len(p.positions) for p in d.scene.points] == [2, 1]  # e-, p
    assert len(d.scene.colorbars) == 1


def test_deposits_become_a_volume(tracks):
    d = EventDisplay().add_deposits(tracks["pos_x"], tracks["pos_y"], tracks["pos_z"], tracks["deposited_energy"], 10)
    (vol,) = d.scene.volumes
    assert vol.grid.sum() == pytest.approx(0.71, abs=1e-6)


def test_line_colours_follow_values_across_polylines():
    paths = [np.zeros((2, 3)), np.ones((3, 3))]
    d = EventDisplay().add_line(paths, values=[[0.0, 0.0], [1.0, 1.0, 1.0]], colorscale=["#000000", "#ffffff"])
    (line,) = d.scene.lines
    assert len(line.colors) == 2 * (1 + 2)
    np.testing.assert_allclose(line.colors[:2], 0.0)  # first polyline at vmin
    np.testing.assert_allclose(line.colors[2:], 1.0)  # second at vmax
    single = EventDisplay().add_line(np.zeros((4, 3)), values=np.arange(4.0))
    assert len(single.scene.lines[0].segments) == 3 * 2


def test_input_errors():
    d = EventDisplay()
    with pytest.raises(ValueError, match="voxel_size"):
        d.add_volume(np.zeros((2, 2, 2)), (0, 0, 0))
    with pytest.raises(ValueError, match="origin"):
        d.add_volume(np.zeros((2, 2, 2)), voxel_size=1.0)
    with pytest.raises(ValueError, match="3D"):
        d.add_volume(np.zeros((2, 2)), (0, 0, 0), 1.0)
    with pytest.raises(ValueError, match="hover_text"):
        d.add_line(np.zeros((3, 3)), hover_text=["a"])
    with pytest.raises(ValueError, match="hover_text"):
        d.add_points(np.zeros((2, 3)), hover_text=["a"])


def test_bounds_set_the_view_box():
    d = EventDisplay(bounds=[[-1, -2, -3], [1, 2, 3]])
    assert d.scene.to_json()["bounds"] == [[-1, -2, -3], [1, 2, 3]]
    assert EventDisplay().scene.to_json()["bounds"] is None
    with pytest.raises(ValueError):
        EventDisplay(bounds=[[0, 0, 0], [0, 1, 1]])


def test_menu_labels_default_to_titles():
    menu = DisplayMenu([EventDisplay(title="a"), EventDisplay()])
    assert menu.labels == ["a", "display 1"]
    with pytest.raises(ValueError):
        DisplayMenu([EventDisplay()], labels=["a", "b"])
    with pytest.raises(ValueError):
        DisplayMenu([])


def test_points_colorbar():
    d = EventDisplay().add_points([[0, 0, 0], [1, 1, 1]], values=[3.0, 7.0], colorbar_title="First hit (ns)")
    (bar,) = d.scene.colorbars
    assert (bar.title, bar.vmin, bar.vmax) == ("First hit (ns)", 3.0, 7.0)
    assert not EventDisplay().add_points([[0, 0, 0]], values=[1.0]).scene.colorbars
