"""Geometric feature extraction: face attributes, UV grids, vision grids, local frames."""

import numpy as np
from typing import List, Optional, Tuple

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface, GeomAbs_BezierSurface, GeomAbs_Cone,
    GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt2d

from occwl.face import Face
from occwl.uvgrid import uvgrid

try:
    from preprocessing.ray_casting import raycast_hemisphere, MeshRayCaster, raycast_hemisphere_mesh
except ImportError:
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from preprocessing.ray_casting import raycast_hemisphere, MeshRayCaster, raycast_hemisphere_mesh

ZERO = 1e-6


def _as_topods(entity) -> TopoDS_Shape:
    return entity.topods_shape() if hasattr(entity, "topods_shape") else entity


# ── Solid scaling ───────────────────────────────────────────────────────────
def scale_solid_to_unit_box(solid):
    """Scale solid to [-1,1]³ centered at origin."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
    from occwl.solid import Solid

    bb = solid.box()
    c = bb.center()
    s = max(bb.x_length(), bb.y_length(), bb.z_length())
    t1 = gp_Trsf(); t1.SetTranslation(gp_Vec(-c[0], -c[1], -c[2]))
    shape1 = BRepBuilderAPI_Transform(solid.topods_shape(), t1, True).Shape()
    t2 = gp_Trsf(); t2.SetScale(gp_Pnt(0, 0, 0), 2.0 / s)
    return Solid(BRepBuilderAPI_Transform(shape1, t2, True).Shape(), allow_compound=True)


# ── Label extraction ────────────────────────────────────────────────────────
def extract_step_face_labels(file_path: str) -> list:
    """Extract integer face labels from STEP ADVANCED_FACE entities."""
    import re
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        matches = re.findall(r"#\d+\s*=\s*ADVANCED_FACE\('([^']*)'\s*", content)
        return [int(m) for m in matches if m.strip().isdigit()]
    except Exception as e:
        print(f"Warning: label extraction failed for {file_path}: {e}")
        return []


# ── Face attributes ─────────────────────────────────────────────────────────
def extract_face_attributes(face, attr_list: List[str], points=None, solid_bbox_diag=None) -> List[float]:
    """Extract geometric attributes (surface type, area, etc.) from a B-Rep face."""
    n_out = sum(3 if a == "FaceCentroidAttribute" else 1 for a in attr_list)
    if face is None or not attr_list:
        return [0.0] * n_out

    try:
        face_td = _as_topods(face)
        surf = BRepAdaptor_Surface(face_td)
        st = surf.GetType()
    except Exception:
        return [0.0] * n_out

    geom = None
    if {"FaceAreaAttribute", "FaceCentroidAttribute"} & set(attr_list):
        try:
            geom = GProp_GProps()
            brepgprop_SurfaceProperties(face_td, geom)
        except Exception:
            geom = None

    def _rational():
        try:
            if st == GeomAbs_BSplineSurface:
                bs = surf.BSpline()
                return 1.0 if bs.IsURational() or bs.IsVRational() else 0.0
            elif st == GeomAbs_BezierSurface:
                bz = surf.Bezier()
                return 1.0 if bz.IsURational() or bz.IsVRational() else 0.0
        except Exception:
            pass
        return 0.0

    def _area():
        if geom is None: return 0.0
        try:
            a = abs(float(geom.Mass()))
            if points is not None and solid_bbox_diag is not None:
                flat = points.reshape(-1, 3)
                if flat.shape[0] > 1:
                    lo, hi = (solid_bbox_diag * 1e-6)**2, solid_bbox_diag**2 * 10
                    if a < lo or a > hi: return 0.0
            return a
        except Exception:
            return 0.0

    def _centroid():
        if geom is None: return (0.0, 0.0, 0.0)
        try:
            c = geom.CentreOfMass()
            cen = (float(c.X()), float(c.Y()), float(c.Z()))
            if points is not None and solid_bbox_diag is not None:
                flat = points.reshape(-1, 3)
                if flat.shape[0] > 1:
                    gc = np.mean(flat, axis=0)
                    if np.linalg.norm(np.array(cen) - gc) > solid_bbox_diag * 2:
                        return tuple(gc)
            return cen
        except Exception:
            return (0.0, 0.0, 0.0)

    fn = {
        "Plane": lambda: float(st == GeomAbs_Plane),
        "Cylinder": lambda: float(st == GeomAbs_Cylinder),
        "Cone": lambda: float(st == GeomAbs_Cone),
        "SphereFaceAttribute": lambda: float(st == GeomAbs_Sphere),
        "TorusFaceAttribute": lambda: float(st == GeomAbs_Torus),
        "FaceAreaAttribute": _area,
        "RationalNurbsFaceAttribute": _rational,
        "FaceCentroidAttribute": _centroid,
    }
    result = []
    for a in attr_list:
        v = fn[a]()
        result.extend(v) if a == "FaceCentroidAttribute" else result.append(v)
    return result


# ── Vision features ─────────────────────────────────────────────────────────
def extract_face_vision_features(
    shape, num_elev=12, num_azim=12, max_dist=None,
    center=None, axes=None, ray_caster=None,
) -> np.ndarray:
    """Cast hemisphere rays → 6-channel vision grid."""
    cast = raycast_hemisphere_mesh if ray_caster is not None else raycast_hemisphere
    kw = dict(center=center, axes=axes, num_elev=num_elev, num_azim=num_azim, max_dist=max_dist, compute_dot=True)
    src = ray_caster if ray_caster is not None else shape
    _, grids = cast(src, **kw)

    z = np.zeros((num_elev, num_azim), dtype=np.float32)
    return np.stack([
        grids["occupancy_grid"], grids["distance_grid"],
        grids.get("dot_grid", z),
        grids.get("occupancy_grid_opposite", z),
        grids.get("distance_grid_opposite", z),
        grids.get("dot_grid_opposite", z),
    ], axis=-1)


# ── Local coordinate frame ──────────────────────────────────────────────────
def compute_local_frame(
    face_shape, points=None, mask=None, uv_points=None,
    num_u=10, num_v=10, file_path=None, untrimmed_center=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute face-local frame (center, 3x3 axes, original center) aligned with UV."""
    if not isinstance(face_shape, TopoDS_Shape):
        face_shape = face_shape.topods_shape()

    surf = BRepAdaptor_Surface(face_shape)
    face = Face(face_shape)
    u_mid = (surf.FirstUParameter() + surf.LastUParameter()) * 0.5
    v_mid = (surf.FirstVParameter() + surf.LastVParameter()) * 0.5

    props = BRepLProp_SLProps(surf, 1, ZERO * 1e-3)
    props.SetParameters(u_mid, v_mid)
    orig_p = props.Value()
    orig_center = np.array([orig_p.X(), orig_p.Y(), orig_p.Z()])

    # Allow option to only use face center on trimmed surface 
    if untrimmed_center and face._trimmed.Perform(gp_Pnt2d(u_mid, v_mid)) not in (0, 2):
        if points is None or mask is None or uv_points is None:
            points, uv_points = uvgrid(face, method="point", uvs=True, num_u=num_u, num_v=num_v)
            vis, _ = uvgrid(face, method="visibility_status", uvs=True, num_u=num_u, num_v=num_v)
            mask = (vis == 0) | (vis == 2)
        vp = points.reshape(-1, 3)[mask.ravel()]
        vu = uv_points.reshape(-1, 2)[mask.ravel()]
        if vp.shape[0] > 0:
            best = np.argmin(np.sum((vp - vp.mean(0))**2, axis=1))
            u_mid, v_mid = vu[best]
            props = BRepLProp_SLProps(surf, 1, ZERO * 1e-3)
            props.SetParameters(float(u_mid), float(v_mid))

    center_p = props.Value()
    n = props.Normal()
    center = np.array([center_p.X(), center_p.Y(), center_p.Z()])
    z = np.array([n.X(), n.Y(), n.Z()])
    nz = np.linalg.norm(z)
    if nz > ZERO: z /= nz
    if face_shape.Orientation() == TopAbs_REVERSED: z = -z

    # X from U-tangent projected onto tangent plane
    ud = surf.Surface().DN(float(u_mid), float(v_mid), 1, 0)
    u_deriv = np.array([ud.X(), ud.Y(), ud.Z()])
    x = u_deriv - np.dot(u_deriv, z) * z
    nx = np.linalg.norm(x)
    if nx > ZERO: x /= nx
    y = np.cross(z, x)
    return center, np.stack([x, y, z], axis=1), orig_center


# ── Single-face processor  ───────────────────────────────
def process_single_face(args):
    """Process one face: UV grid, attributes, vision grid, local frame."""
    (
        face_idx, node_id, face_shape, step_labels, segmentation,
        inc_uv, inc_attr, inc_vision,
        nu, nv, el, az,
        attr_list, max_ray_dist, diag, solid_shape, file_path, ray_caster,
    ) = args

    r = {"face_idx": face_idx, "face_type": None, "surface_area": 0.0, "is_curved": False}

    if segmentation:
        r["label"] = step_labels[face_idx] if step_labels and face_idx < len(step_labels) else -1

    # UV grid
    try:
        pts, uv_pts = uvgrid(face_shape, method="point", uvs=True, num_u=nu, num_v=nv)
    except Exception as e:
        print(f"UV error face {face_idx}: {e}")
        return r

    normals = mask = None
    if inc_uv:
        try:
            normals = uvgrid(face_shape, method="normal", num_u=nu, num_v=nv)
            vis, _ = uvgrid(face_shape, method="visibility_status", uvs=True, num_u=nu, num_v=nv)
            mask = ((vis.squeeze() if vis.ndim > 2 else vis) == 0) | (vis.squeeze() == 2) if vis.ndim > 2 else ((vis == 0) | (vis == 2))
        except Exception as e:
            print(f"UV feature error face {face_idx}: {e}")
            return r

    # Local frame
    try:
        center, axes, _ = compute_local_frame(face_shape, points=pts, mask=mask, uv_points=uv_pts, file_path=file_path)
    except Exception as e:
        print(f"Frame error face {face_idx}: {e}")
        return r

    if inc_uv and pts is not None and normals is not None and mask is not None:
        try:
            m3 = mask[..., np.newaxis].astype(np.float32) if mask.ndim == 2 else mask.astype(np.float32)
            r["face_features"] = np.concatenate((pts, normals, m3), axis=-1)
            # Local coordinates
            lp = (pts.reshape(-1, 3) - center) @ axes
            ln = normals.reshape(-1, 3) @ axes
            r["face_features_local"] = np.concatenate([
                lp.reshape(nu, nv, 3), ln.reshape(nu, nv, 3), m3
            ], axis=-1)
        except Exception as e:
            print(f"UV proc error face {face_idx}: {e}")

    if inc_attr:
        try:
            r["face_attributes"] = extract_face_attributes(face_shape, attr_list, points=pts, solid_bbox_diag=diag)
        except Exception as e:
            print(f"Attr error face {face_idx}: {e}")

    if inc_vision:
        try:
            r["vision_grid"] = extract_face_vision_features(
                solid_shape, el, az, max_dist=max_ray_dist,
                center=center, axes=axes, ray_caster=ray_caster,
            )
        except Exception as e:
            print(f"Vision error face {face_idx}: {e}")

    return r
