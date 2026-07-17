#!/usr/bin/env python
"""Quadric-decimate the Vention pedestal GLB in place.

``assets/robot/pedestal.glb`` is a raw CAD export: ~3.0M triangles across 206
mesh instances, ~160x the two decimated Rizon arms combined. Rerun loads it
whole via ``rr.Asset3D`` (:mod:`dashboard.robot_view`), so it dominates the
dashboard's geometry. Two over-tessellated parts alone are ~61% of it.

The GLB has no textures or UVs — 206 geometries share 5 flat PBR materials — so
each geometry decimates safely on positions+faces with its material and the
scene-graph node transforms preserved. Geometries at or below ``--min-tris`` are
left untouched. Re-export is a standard glTF 2.0 binary that Rerun reloads.

The pedestal is background scenery, so it targets a global triangle *budget*
(``--target-total``, default 8000) rather than a uniform ratio: each mesh gets a
``target_count`` proportional to its share of the budget, floored so no part
vanishes. Quality goes low-poly/blocky, which is fine for static scenery.

Note this is a *lower bound* here: the parts are hundreds of separate solid
bodies, and single-pass quadric decimation cannot collapse a mesh below its
component count, so the whole pedestal bottoms out near ~206k triangles (~15x)
regardless of the budget. Going lower would require remeshing.

WARNING: ``*.glb`` is gitignored (see ``assets/robot/.gitignore``), so this
overwrites the ONLY copy of the mesh in place — git will NOT restore it. Back up
``pedestal.glb`` before running if you might want the original detail back.

Usage:
    python scripts/decimate_pedestal.py [--target-total 8000] [--floor 4] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fast_simplification
import numpy as np
import trimesh

_GLB = (
    Path(__file__).resolve().parent.parent
    / "src/dual_flexiv_control/dashboard/assets/robot/pedestal.glb"
)


def _decimate(mesh: trimesh.Trimesh, target_count: int) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` reduced toward ``target_count`` triangles.

    The pedestal parts have no UVs, so only positions/faces are simplified; the
    flat PBR material (base-colour factor) is re-attached so appearance is
    unchanged. The decimator may land a bit above ``target_count`` when topology
    resists further collapse.
    """
    faces = np.asarray(mesh.faces)
    material = getattr(mesh.visual, "material", None)
    if len(faces) <= target_count:
        out = mesh.copy()
    else:
        v, f = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float64), faces, target_count=target_count
        )
        out = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if material is not None:
        out.visual = trimesh.visual.TextureVisuals(material=material)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-total", type=int, default=8000, help="triangle budget for the whole pedestal")
    ap.add_argument("--floor", type=int, default=4, help="minimum triangles kept per mesh")
    ap.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = ap.parse_args()

    scene = trimesh.load(_GLB, process=False)
    sizes = {name: len(g.faces) for name, g in scene.geometry.items()}
    total = sum(sizes.values())

    before = after = 0
    for name, geom in list(scene.geometry.items()):
        b = sizes[name]
        # allocate this mesh a share of the budget proportional to its size
        target = max(args.floor, round(args.target_total * b / total))
        new = _decimate(geom, target)
        after += len(new.faces)
        before += b
        scene.geometry[name] = new  # graph references geometry by name -> transforms kept

    print(f"pedestal.glb  {before} -> {after} tris  ({before / max(after, 1):.0f}x)  over {len(scene.geometry)} meshes")
    if args.dry_run:
        print("(dry run — nothing written)")
        return
    scene.export(_GLB)
    print(f"wrote {_GLB} ({_GLB.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
