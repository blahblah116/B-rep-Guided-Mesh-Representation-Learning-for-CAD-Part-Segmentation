"""Convert MFCAD++ STEP files to triangle mesh with per-triangle labels.

Output format matches the Fusion360 dataset:
  <split>/<name>.obj   — triangle mesh (vertices + faces)
  <split>/<name>.seg   — per-triangle class label (one per line)
  <split>/<name>.fidx  — per-triangle B-Rep face index (one per line)

Usage:
  python mfcadpp_to_mesh.py --data_root /data2/gwlee/fovnet/data/mfcad++ \
                             --out_dir   /data2/gwlee/fovnet/data/mfcad++/meshes
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location

LIN_DEFL = 0.01
ANG_DEFL = 0.05


def extract_face_labels(step_path: Path) -> list[int]:
    """Read per-B-Rep-face class labels from ADVANCED_FACE entity names."""
    text = step_path.read_text(errors="ignore")
    # ADVANCED_FACE appears in STEP file order == face index from TopExp_Explorer
    return [int(m) for m in re.findall(r"ADVANCED_FACE\('(\d+)'", text)]


def tessellate(shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts [N,3], tris [M,3], face_ids [M]) from a tessellated OCC shape."""
    mesh = BRepMesh_IncrementalMesh(shape, LIN_DEFL, False, ANG_DEFL, True)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Tessellation failed")

    verts, tris, fids = [], [], []
    voff = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fi = 0
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nn = tri.NbNodes()
            for i in range(1, nn + 1):
                p = tri.Node(i)
                p.Transform(trsf)
                verts.append([p.X(), p.Y(), p.Z()])
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                off = voff - 1
                t = [n1 + off, n3 + off, n2 + off] if rev else [n1 + off, n2 + off, n3 + off]
                tris.append(t)
                fids.append(fi)
            voff += nn
        fi += 1
        exp.Next()

    return (
        np.array(verts, dtype=np.float64),
        np.array(tris, dtype=np.int32),
        np.array(fids, dtype=np.int32),
    )


def save_obj(path: Path, verts: np.ndarray, tris: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for t in tris:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")  # 1-indexed


def process_one(step_path: Path, out_dir: Path) -> str | None:
    """Process a single STEP file. Returns error string on failure, None on success."""
    name = step_path.stem
    out_obj = out_dir / f"{name}.obj"
    if out_obj.exists():
        return None  # already done

    try:
        face_labels = extract_face_labels(step_path)

        reader = STEPControl_Reader()
        reader.ReadFile(str(step_path))
        reader.TransferRoots()
        shape = reader.OneShape()

        verts, tris, fids = tessellate(shape)

        if len(tris) == 0:
            return f"{name}: no triangles after tessellation"

        # Map triangle → class label via face index
        n_brep_faces = len(face_labels)
        if fids.max() >= n_brep_faces:
            return f"{name}: face index {fids.max()} out of range (labels={n_brep_faces})"

        seg = np.array([face_labels[fi] for fi in fids], dtype=np.int16)

        save_obj(out_obj, verts, tris)
        np.savetxt(out_dir / f"{name}.seg", seg, fmt="%d")
        np.savetxt(out_dir / f"{name}.fidx", fids, fmt="%d")
        return None

    except Exception as e:
        return f"{name}: {e}"


def collect_step_files(data_root: Path, splits: list[str]) -> list[tuple[Path, str]]:
    files = []
    for split in splits:
        d = data_root / split
        files.extend((p, split) for p in sorted(d.glob("*.step")))
        files.extend((p, split) for p in sorted(d.glob("*.stp")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data2/gwlee/fovnet/data/mfcad++")
    parser.add_argument("--out_dir", default="/data2/gwlee/fovnet/data/mfcad++/meshes")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (for testing)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    step_files = collect_step_files(data_root, args.splits)
    if args.limit:
        step_files = step_files[: args.limit]
    print(f"Processing {len(step_files)} STEP files → {out_dir}/<split>")

    errors = []
    if args.workers <= 1:
        for f, split in tqdm(step_files, unit="file"):
            err = process_one(f, out_dir / split)
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(process_one, f, out_dir / split): (f, split)
                for f, split in step_files
            }
            for fut in tqdm(as_completed(futures), total=len(futures), unit="file"):
                err = fut.result()
                if err:
                    errors.append(err)

    print(f"\nDone. {len(step_files) - len(errors)}/{len(step_files)} succeeded.")
    if errors:
        print(f"{len(errors)} errors:")
        for e in errors[:20]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
