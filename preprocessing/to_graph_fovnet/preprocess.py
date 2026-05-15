"""FOVNet Feature Extraction — STEP → DGL graph pipeline."""

import argparse
import contextlib
import multiprocessing
import os
import pathlib
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional, Tuple

import dgl
import numpy as np
import torch
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Extend import TopologyUtils

from occwl.compound import Compound
from occwl.edge import Edge
from occwl.edge_data_extractor import EdgeConvexity, EdgeDataExtractor
from occwl.face import Face
from occwl.graph import face_adjacency
from occwl.solid import Solid

from geometry_features import (
    process_single_face, scale_solid_to_unit_box, extract_step_face_labels,
)
try:
    from ray_casting import MeshRayCaster, HAS_TRIMESH, HAS_PYEMBREE
except ImportError:
    from preprocessing.ray_casting import MeshRayCaster, HAS_TRIMESH, HAS_PYEMBREE

# ── Global settings ─────────────────────────────────────────────────────────
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_num_threads(1)
np.seterr(all="ignore")
warnings.filterwarnings("ignore")

ANGLE_TOLERANCE_RADS = 0.0872664626  # 5°

DEFAULT_FACE_ATTRIBUTES = [
    "Plane", "Cylinder", "Cone", "SphereFaceAttribute", "TorusFaceAttribute",
    "FaceAreaAttribute", "RationalNurbsFaceAttribute",
]
DEFAULT_EDGE_ATTRIBUTES = [
    "Concave edge", "Convex edge", "Smooth", "EdgeLengthAttribute",
    "CircularEdgeAttribute", "ClosedEdgeAttribute", "EllipticalEdgeAttribute",
    "NonRationalBSplineEdgeAttribute", "RationalBSplineEdgeAttribute",
    "StraightEdgeAttribute",
]

SOLIDLETTERS_INVALID_FONTS = frozenset([
    "Bokor", "Lao Muang Khong", "Lao Sans Pro", "MS Outlook", "Catamaran Black",
    "Dubai", "HoloLens MDL2 Assets", "Lao Muang Don", "Oxanium Medium",
    "Rounded Mplus 1c", "Moul Pali", "Noto Sans Tamil", "Webdings", "Armata",
    "Koulen", "Yinmar", "Ponnala", "Chenla", "Lohit Devanagari", "Metal",
    "MS Office Symbol", "Cormorant Garamond Medium", "Chiller", "Give You Glory",
    "Hind Vadodara Light", "Libre Barcode 39 Extended", "Myanmar Sans Pro",
    "Scheherazade", "Segoe MDL2 Assets", "Siemreap", "Signika SemiBold",
    "Taprom", "Times New Roman TUR", "Playfair Display SC Black", "Poppins Thin",
    "Raleway Dots", "Raleway Thin", "Spectral SC ExtraLight", "Txt", "Uchen",
    "Almarai ExtraBold", "Fasthand", "Exo", "Freckle Face", "Montserrat Light",
    "Inter", "MS Reference Specialty", "Preah Vihear", "Sitara",
    "Barkerville Old Face", "Bodoni MT", "HoloLens MDL2 Assests",
    "Libre Barcode 39", "Lohit Tamil", "Marlett", "MS outlook",
    "MS office Symbol Semilight", "MS office symbol regular",
    "Ms office symbol extralight", "Ms Reference speciality", "Symbol",
    "Wingdings", "Souliyo Unicode", "Aguafina Script", "Yantramanav Black",
])


# ── Config dataclass ────────────────────────
@dataclass
class ProcessingConfig:
    """Shared config passed through the processing pipeline."""
    surf_samples: int = 10
    curv_samples: int = 10
    vision_el: int = 6
    vision_az: int = 12
    include_vision: bool = True
    include_uv_face: bool = True
    include_uv_edge: bool = False
    include_face_attr: bool = True
    include_edge_attr: bool = False
    scale_body: bool = True
    random_rotate: bool = False
    create_rotated_files: bool = False
    segmentation: bool = True
    compress: bool = False
    use_mesh_rays: bool = False
    face_attr_list: List[str] = field(default_factory=lambda: list(DEFAULT_FACE_ATTRIBUTES))
    edge_attr_list: List[str] = field(default_factory=lambda: list(DEFAULT_EDGE_ATTRIBUTES))


# ── Helpers ──────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _suppress_fd():
    """Suppress C-level stdout/stderr."""
    old = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1); os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old[0], 1); os.dup2(old[1], 2)
        for fd in (*old, devnull):
            os.close(fd)

def _as_topods(entity):
    return entity.topods_shape() if hasattr(entity, "topods_shape") else entity


def _edge_convexity(edge_topods, faces: List[Face]) -> Optional[EdgeConvexity]:
    try:
        ed = EdgeDataExtractor(Edge(edge_topods), faces, use_arclength_params=False)
        return ed.edge_convexity(ANGLE_TOLERANCE_RADS) if ed.good else None
    except Exception:
        return None

# ── Edge attributes ─────────────────────────────────────────────────────────
def extract_aag_edge_attributes(edge, attr_list: List[str], topo_exp) -> List[float]:
    if edge is None or not attr_list or topo_exp is None:
        return [0.0] * len(attr_list)

    edge_topods = _as_topods(edge)
    try:
        faces = [Face(f) for f in topo_exp.faces_from_edge(edge_topods)]
    except Exception:
        return [0.0] * len(attr_list)

    # Lazy evaluation helpers
    curve_adaptor = curve_type = None
    edge_wrapper = edge_curve_type = None
    edge_rational = False
    geom_props = convexity = None

    need_ca = {"CircularEdgeAttribute", "EllipticalEdgeAttribute", "StraightEdgeAttribute"}
    need_ew = {"HyperbolicEdgeAttribute", "ParabolicEdgeAttribute", "BezierEdgeAttribute",
               "NonRationalBSplineEdgeAttribute", "RationalBSplineEdgeAttribute", "OffsetEdgeAttribute"}
    attr_set = set(attr_list)

    if attr_set & need_ca:
        try:
            curve_adaptor = BRepAdaptor_Curve(edge_topods)
            curve_type = curve_adaptor.GetType()
        except Exception:
            pass
    if attr_set & need_ew:
        try:
            edge_wrapper = Edge(edge_topods)
            edge_curve_type = edge_wrapper.curve_type()
            edge_rational = edge_wrapper.rational() if edge_curve_type == "bspline" else False
        except Exception:
            pass
    if "EdgeLengthAttribute" in attr_set:
        try:
            geom_props = GProp_GProps()
            brepgprop_LinearProperties(edge_topods, geom_props)
        except Exception:
            pass
    if attr_set & {"Concave edge", "Convex edge", "Smooth"}:
        try:
            convexity = _edge_convexity(edge_topods, faces)
        except Exception:
            pass

    conv_map = {"Concave edge": EdgeConvexity.CONCAVE, "Convex edge": EdgeConvexity.CONVEX, "Smooth": EdgeConvexity.SMOOTH}
    attr_fn = {
        "EdgeLengthAttribute": lambda: float(geom_props.Mass()) if geom_props else 0.0,
        "CircularEdgeAttribute": lambda: float(curve_type == GeomAbs_Circle) if curve_type is not None else 0.0,
        "ClosedEdgeAttribute": lambda: float(BRep_Tool().IsClosed(edge_topods)),
        "EllipticalEdgeAttribute": lambda: float(curve_type == GeomAbs_Ellipse) if curve_type is not None else 0.0,
        "StraightEdgeAttribute": lambda: float(curve_type == GeomAbs_Line) if curve_type is not None else 0.0,
        "HyperbolicEdgeAttribute": lambda: float(edge_curve_type == "hyperbola") if edge_curve_type else 0.0,
        "ParabolicEdgeAttribute": lambda: float(edge_curve_type == "parabola") if edge_curve_type else 0.0,
        "BezierEdgeAttribute": lambda: float(edge_curve_type == "bezier") if edge_curve_type else 0.0,
        "NonRationalBSplineEdgeAttribute": lambda: float(edge_curve_type == "bspline" and not edge_rational) if edge_curve_type else 0.0,
        "RationalBSplineEdgeAttribute": lambda: float(edge_curve_type == "bspline" and edge_rational) if edge_curve_type else 0.0,
        "OffsetEdgeAttribute": lambda: float(edge_curve_type == "offset") if edge_curve_type else 0.0,
    }

    vals = []
    for a in attr_list:
        if a in conv_map:
            vals.append(float(convexity == conv_map[a]) if convexity else 0.0)
        elif a in attr_fn:
            vals.append(attr_fn[a]())
        else:
            vals.append(0.0)
    return vals

def compute_edge_uv_grids(edge, faces: List[Face], n_samples: int) -> np.ndarray:
    """Sample edge curve: points, tangents, left/right face normals."""
    if n_samples <= 0:
        return np.zeros((0, 12), dtype=np.float32)
    try:
        ed = EdgeDataExtractor(Edge(_as_topods(edge)), faces, num_samples=n_samples, use_arclength_params=True)
    except Exception:
        ed = None
    if ed is None or not ed.good:
        return np.zeros((n_samples, 12), dtype=np.float32)
    return np.concatenate([ed.points, ed.tangents, ed.left_normals, ed.right_normals], axis=1).astype(np.float32)


# ── Graph building ──────────────────────────────────────────────────────────
def build_graph(file_path, solid: Solid, cfg: ProcessingConfig) -> Optional[dgl.DGLGraph]:
    """Build a DGL face-adjacency graph from a CAD solid."""
    try:
        graph = face_adjacency(solid)
    except Exception as e:
        print(f"face_adjacency error for {file_path}: {e}")
        return None

    solid_shape = solid.topods_shape()
    topo = TopologyUtils.TopologyExplorer(solid_shape, ignore_orientation=True)
    step_labels = extract_step_face_labels(file_path) if cfg.segmentation else None

    node_ids = sorted(graph.nodes)
    edge_keys = sorted(graph.edges)
    n_faces, n_edges = len(node_ids), len(edge_keys)
    if n_faces == 0:
        return None

    # Bounding-box diagonal for ray distance
    box = Bnd_Box()
    brepbndlib_Add(solid_shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diag = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2)
    ray_caster = MeshRayCaster(solid_shape) if cfg.use_mesh_rays else None

    ns, nv = cfg.surf_samples, cfg.surf_samples
    el, az = cfg.vision_el, cfg.vision_az

    # Pre-allocate node arrays
    face_pts = np.zeros((n_faces, ns, nv, 7), dtype=np.float32) if cfg.include_uv_face else None
    face_pts_local = np.zeros_like(face_pts) if cfg.include_uv_face else None
    vision = np.zeros((n_faces, el, az, 6), dtype=np.float32) if cfg.include_vision else None
    vision_feat = np.zeros((n_faces, 4), dtype=np.float32) if cfg.include_vision else None
    face_attr_list: Optional[List] = [] if cfg.include_face_attr else None
    face_labels: List[int] = []
    face_types, total_area = {}, 0.0

    # Pre-allocate edge arrays
    edge_uv = np.zeros((n_edges, cfg.curv_samples, 12), dtype=np.float32) if cfg.include_uv_edge else None
    edge_attr_list: Optional[List] = [] if cfg.include_edge_attr else None

    # Process faces sequentially (file-level parallelism handles concurrency)
    for fi, nid in enumerate(node_ids):
        args = (
            fi, nid, graph.nodes[nid]["face"], step_labels, cfg.segmentation,
            cfg.include_uv_face, cfg.include_face_attr, cfg.include_vision,
            ns, nv, el, az,
            cfg.face_attr_list, diag * 2, diag, solid_shape, file_path, ray_caster,
        )
        r = process_single_face(args)
        if r.get("face_type"):
            face_types[r["face_type"]] = face_types.get(r["face_type"], 0) + 1
            total_area += r.get("surface_area", 0.0)
        if cfg.segmentation and "label" in r:
            face_labels.append(r["label"])
        if cfg.include_uv_face and "face_features" in r:
            face_pts[r["face_idx"]] = r["face_features"]
            if "face_features_local" in r:
                face_pts_local[r["face_idx"]] = r["face_features_local"]
        if cfg.include_face_attr and "face_attributes" in r:
            face_attr_list.append(r["face_attributes"])
        if cfg.include_vision and "vision_grid" in r:
            vision[r["face_idx"]] = r["vision_grid"]

    # Process edges
    for ei, ek in enumerate(edge_keys):
        edge_ent = graph.edges[ek]["edge"]
        edge_td = _as_topods(edge_ent)
        efaces = [Face(f) for f in topo.faces_from_edge(edge_td)]
        if cfg.include_uv_edge and edge_uv is not None:
            g_grid = compute_edge_uv_grids(edge_ent, efaces, cfg.curv_samples)
            s = min(edge_uv.shape[1], g_grid.shape[0])
            if s > 0:
                edge_uv[ei, :s] = g_grid[:s]
        if cfg.include_edge_attr and edge_attr_list is not None:
            edge_attr_list.append(extract_aag_edge_attributes(edge_ent, cfg.edge_attr_list, topo))

    # Assemble DGL graph
    src, dst = zip(*edge_keys) if edge_keys else ([], [])
    dg = dgl.graph((list(src), list(dst)), num_nodes=n_faces)
    _set = lambda d, k, a: d.__setitem__(k, torch.from_numpy(np.asarray(a, dtype=np.float32)))
    if cfg.include_uv_face and face_pts is not None:
        _set(dg.ndata, "x", face_pts); _set(dg.ndata, "x_local", face_pts_local)
    if cfg.include_face_attr and face_attr_list:
        _set(dg.ndata, "face_feat", face_attr_list)
    if cfg.include_vision and vision is not None:
        _set(dg.ndata, "vision_grids", vision); _set(dg.ndata, "vision_features", vision_feat)
    if cfg.segmentation:
        dg.ndata["y"] = torch.from_numpy(np.asarray(face_labels, dtype=np.int64))
    if cfg.include_uv_edge and edge_uv is not None:
        _set(dg.edata, "x", edge_uv)
    if cfg.include_edge_attr and edge_attr_list:
        _set(dg.edata, "edge_feat", edge_attr_list)
    return dg


# ── File processing ─────────────────────────────────────────────────────────
def _save_graph(graph, path, compress):
    if compress:
        if "vision_grids" in graph.ndata:
            vg = graph.ndata["vision_grids"]
            graph.ndata["vision_grids"] = vg[..., :12].half() if vg.shape[-1] > 12 else vg.half()
        for k, v in graph.ndata.items():
            if k != "vision_grids" and v.dtype == torch.float32:
                graph.ndata[k] = v.half()
        for k, v in graph.edata.items():
            if v.dtype == torch.float32:
                graph.edata[k] = v.half()
    dgl.data.utils.save_graphs(str(path), [graph])


def _random_rotation(file_path):
    """Deterministic random rotation seeded by file path."""
    rng = np.random.RandomState(hash(str(file_path)) % (2**32))
    angles = rng.uniform(0, 2 * np.pi, 3)
    origin = gp_Pnt(0, 0, 0)
    axes = [(1,0,0), (0,1,0), (0,0,1)]
    trsfs = []
    for ax, ang in zip(axes, angles):
        t = gp_Trsf()
        t.SetRotation(gp_Ax1(origin, gp_Dir(*ax)), ang)
        trsfs.append(t)
    R = trsfs[0].Multiplied(trsfs[1]).Multiplied(trsfs[2])
    return R


def process_single_file(file_path, output_dir, cfg: ProcessingConfig) -> Optional[dgl.DGLGraph]:
    """Process one STEP file → DGL graph saved to disk."""
    try:
        compound = Compound.load_from_step(file_path)
        solids = list(compound.solids())
        if not solids:
            raise ValueError(f"No solids in {file_path}")
        solid = solids[0] if len(solids) == 1 else Solid(compound.topods_shape(), allow_compound=True)
        if cfg.scale_body:
            solid = scale_solid_to_unit_box(solid)
    except Exception as e:
        print(f"Load error {file_path.name}: {e}")
        return None

    try:
        if cfg.random_rotate:
            rot = _random_rotation(file_path)
            rotated = BRepBuilderAPI_Transform(solid.topods_shape(), rot, True).Shape()
            solid = Solid(rotated, allow_compound=True)
            if cfg.create_rotated_files:
                with _suppress_fd():
                    w = STEPControl_Writer()
                    w.Transfer(rotated, STEPControl_AsIs)
                    w.Write(str(output_dir.parent / (file_path.stem + "_rotated.stp")))

        graph = build_graph(file_path, solid, cfg)
        if graph is not None:
            suffix = "_rotated" if cfg.random_rotate else ""
            _save_graph(graph, output_dir / (file_path.stem + suffix + ".bin"), cfg.compress)
        return graph
    except Exception as e:
        print(f"Process error {file_path.name}: {e}")
        return None

def _process_wrapper(args):
    """Multiprocessing wrapper — unpacks (file, outdir, cfg)."""
    f, outdir, cfg = args
    try:
        process_single_file(f, outdir, cfg)
        return True, f.name, None
    except Exception as e:
        print(f"Error {f.name}: {e}")
        return False, f.name, str(e)

# ── Batch processing ────────────────────────────────────────────────────────
def _discover_step_files(input_dir, dataset):
    """Find STEP files, applying dataset-specific filtering."""
    exts = ["*.stp", "*.step", "*.STP", "*.STEP"]
    files = []
    for ext in exts:
        files.extend(input_dir.glob(ext))

    if dataset == "solidletters":
        valid = []
        for f in files:
            try:
                font = f.stem.split("_")[1]
                if font not in SOLIDLETTERS_INVALID_FONTS:
                    valid.append(f)
            except IndexError:
                continue
        files = valid
        print(f"Found {len(files)} valid SolidLetters files after filtering")
    else:
        print(f"Found {len(files)} STEP files")
    return files

def process_multiple_files(
    input_dir, output_dir, cfg: ProcessingConfig,
    num_processes=22, skip_existing=True, dataset="mfcad++",
    max_files=None,
) -> Tuple[List[Any], List[str]]:
    """Process many STEP files with file-level parallelism."""
    files = _discover_step_files(input_dir, dataset)
    outdirs = [output_dir] * len(files)

    if skip_existing:
        if cfg.random_rotate and cfg.create_rotated_files:
            pairs = [(f, output_dir) for f in files
                     if not ((output_dir / (f.stem + "_rotated.bin")).exists()
                             and (output_dir.parent / (f.stem + "_rotated.stp")).exists())]
        else:
            pairs = [(f, output_dir) for f in files
                     if not (output_dir / (f.stem + ".bin")).exists()]
        files, outdirs = (zip(*pairs) if pairs else ([], []))

    if max_files and len(files) > max_files:
        print(f"Limiting to {max_files} of {len(files)} files")
        files, outdirs = files[:max_files], outdirs[:max_files]

    if not files:
        print("No new files to process.")
        return [], []

    args = [(f, o, cfg) for f, o in zip(files, outdirs)]
    errors = []

    if num_processes is None or num_processes <= 1:
        for a in tqdm(args, desc="Processing"):
            ok, name, err = _process_wrapper(a)
            if not ok:
                errors.append(name)
    else:
        with multiprocessing.Pool(num_processes) as pool:
            for ok, name, err in tqdm(pool.imap_unordered(_process_wrapper, args),
                                       total=len(args), desc="Processing"):
                if not ok:
                    errors.append(name)
    return [], errors


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="STEP → DGL graph feature extraction")
    p.add_argument("--dataset", default="mfcad++")
    p.add_argument("--folder", default="graphs")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--all", action="store_true", default=False)
    p.add_argument("--split", default="train")
    p.add_argument("--seg", action="store_true", default=True)
    p.add_argument("--rotate", action="store_true", default=False)
    p.add_argument("--max_files", "-n", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", default=False)
    p.add_argument("--az", type=int, default=12)
    p.add_argument("--el", type=int, default=6)
    p.add_argument("--mesh_rays", action="store_true", default=False)
    p.add_argument("--uv_samples", type=int, default=10)
    p.add_argument("--curv_samples", type=int, default=10)
    p.add_argument("--num_processes", type=int, default=None)
    p.add_argument("--edge_info", action="store_true", default=False)
    p.add_argument("--no_compress", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    num_proc = args.num_processes or min(22, multiprocessing.cpu_count())
    splits = ["train", "val", "test"] if args.all else [args.split]

    cfg = ProcessingConfig(
        surf_samples=args.uv_samples, curv_samples=args.curv_samples,
        vision_el=args.el, vision_az=args.az,
        include_uv_edge=args.edge_info, include_edge_attr=args.edge_info,
        random_rotate=args.rotate, create_rotated_files=True,
        segmentation=args.seg, compress=not args.no_compress,
        use_mesh_rays=args.mesh_rays,
    )

    print(f"[INFO] {args.dataset} | splits={splits} | procs={num_proc} | mesh_rays={args.mesh_rays}")

    base = pathlib.Path(args.data_dir) if args.data_dir else (pathlib.Path(__file__).parent / "../data").resolve()
    if not base.exists():
        print(f"[ERROR] Data dir not found: {base}. Use --data_dir.")
        return

    for split in splits:
        step_path = base / args.dataset / split
        if args.rotate:
            graph_path = base / args.dataset / (split + "_rotated") / args.folder
        else:
            graph_path = base / args.dataset / split / args.folder
        graph_path.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing {split}")
        try:
            _, errors = process_multiple_files(
                step_path, graph_path, cfg,
                num_processes=num_proc, skip_existing=args.skip_existing,
                dataset=args.dataset, max_files=args.max_files,
            )
            print(f"[INFO] {split} done. Errors: {len(errors)}")
            if errors:
                print(f"[WARN] Failed: {errors[:10]}{'...' if len(errors) > 10 else ''}")
        except KeyboardInterrupt:
            print(f"[INFO] Interrupted at {split}")
            break
        except Exception as e:
            print(f"[ERROR] {split}: {e}")

    print("[INFO] Complete.")


if __name__ == "__main__":
    main()
