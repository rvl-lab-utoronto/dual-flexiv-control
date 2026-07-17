#!/usr/bin/env python
"""Quadric-decimate the Rizon4s visual OBJ meshes in place.

The vendor visual meshes (``assets/robot/meshes/Rizon4s/visual/*.obj``) run
~5-7k triangles each, ~44k per arm, ~88k for the two-arm dashboard scene. Rerun
logs every triangle, so the arms dominate viewer geometry. This script collapses
each mesh with a quadric-error decimator (``fast_simplification``) to a target
reduction, preserving the per-``usemtl`` group split that the dashboard's OBJ
loader (:func:`dashboard.robot_view._load_obj`) relies on to keep the
``robot``/``cover`` materials distinct. Vertex normals are recomputed after
decimation. Meshes below ``--min-tris`` are copied through untouched.

The meshes are git-tracked, so re-running against a dirty tree is safe: restore
with ``git checkout`` on the mesh directory to get the originals back.

Usage:
    python scripts/decimate_visual_meshes.py [--reduction 0.8] [--min-tris 500] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fast_simplification
import numpy as np

_VISUAL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/dual_flexiv_control/dashboard/assets/robot/meshes/Rizon4s/visual"
)


def _parse_obj(text: str):
    """Parse an OBJ into shared v/vn arrays plus per-``usemtl`` face lists.

    Mirrors the vertex-index handling of ``robot_view._load_obj``: faces are
    fan-triangulated and only the position index (``f``-token[0]) drives
    geometry. Returns ``(mtllib, v, groups)`` where ``groups`` maps material
    name -> (N, 3) int array of 0-based vertex indices.
    """
    mtllib: str | None = None
    v: list[list[float]] = []
    groups: dict[str, list[tuple[int, int, int]]] = {}
    current = ""
    for raw in text.splitlines():
        t = raw.split()
        if not t:
            continue
        k = t[0]
        if k == "v":
            v.append([float(x) for x in t[1:4]])
        elif k == "mtllib":
            mtllib = t[1]
        elif k == "usemtl":
            current = t[1]
        elif k == "f":
            corners = [int(c.split("/")[0]) - 1 for c in t[1:]]
            tris = groups.setdefault(current, [])
            for i in range(1, len(corners) - 1):
                tris.append((corners[0], corners[i], corners[i + 1]))
    return mtllib, np.asarray(v, dtype=np.float64), groups


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted smooth vertex normals (numpy only).

    Each face contributes its ``(b-a) x (c-a)`` — magnitude equals twice the
    triangle area, so accumulating the un-normalised cross products naturally
    area-weights before the final unit-normalise. Winding is preserved by the
    decimator, so the result keeps the meshes' outward orientation.
    """
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths == 0, 1.0, lengths)


def _decimate_group(verts: np.ndarray, faces: np.ndarray, reduction: float, min_tris: int):
    """Decimate one material group; recompute smooth vertex normals.

    Remaps the group's global vertex indices to a local, gap-free range before
    decimating (a group touches only some of the file's vertices). Returns
    ``(positions, normals, faces)`` local to the group.
    """
    used, inverse = np.unique(faces, return_inverse=True)
    local_v = verts[used]
    local_f = inverse.reshape(faces.shape).astype(np.int32)
    if len(local_f) > min_tris and reduction > 0:
        local_v, local_f = fast_simplification.simplify(
            local_v, local_f, target_reduction=reduction
        )
    local_v = np.asarray(local_v, dtype=np.float64)
    local_f = np.asarray(local_f, dtype=np.int64)
    return local_v, _vertex_normals(local_v, local_f), local_f


def _write_obj(path: Path, mtllib: str | None, parts) -> None:
    """Write groups back to OBJ with a global v/vn index space and ``usemtl``.

    Each group's vertices are appended sequentially; a face references the same
    index for position and normal (``f a//a b//b c//c``), matching how the
    loader pairs them.
    """
    lines: list[str] = ["# Decimated by scripts/decimate_visual_meshes.py"]
    if mtllib:
        lines.append(f"mtllib {mtllib}")
    offset = 1  # OBJ indices are 1-based
    for material, pos, nrm, faces in parts:
        for p in pos:
            lines.append(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
        for n in nrm:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        lines.append(f"usemtl {material}")
        for a, b, c in faces + offset:
            lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
        offset += len(pos)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reduction", type=float, default=0.8, help="target fraction removed (0.8 = 5x fewer tris)")
    ap.add_argument("--min-tris", type=int, default=500, help="leave groups with <= this many triangles untouched")
    ap.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = ap.parse_args()

    objs = sorted(_VISUAL_DIR.glob("*.obj"))
    if not objs:
        raise SystemExit(f"no OBJ meshes under {_VISUAL_DIR}")

    total_before = total_after = 0
    for obj in objs:
        mtllib, verts, groups = _parse_obj(obj.read_text())
        before = sum(len(f) for f in groups.values())
        parts = []
        for material, tris in groups.items():
            pos, nrm, faces = _decimate_group(
                verts, np.asarray(tris, dtype=np.int64), args.reduction, args.min_tris
            )
            parts.append((material, pos, nrm, faces))
        after = sum(len(p[3]) for p in parts)
        total_before += before
        total_after += after
        print(f"{obj.name:12s} {before:6d} -> {after:6d} tris  ({before / max(after, 1):.1f}x)")
        if not args.dry_run:
            _write_obj(obj, mtllib, parts)

    print("-" * 44)
    print(f"{'TOTAL/arm':12s} {total_before:6d} -> {total_after:6d} tris  ({total_before / max(total_after, 1):.1f}x)")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
