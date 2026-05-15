"""Ray casting: hemisphere sampling, B-Rep intersection, and mesh-based casting."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Vec
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

try:
    import trimesh
    try:
        _test = trimesh.ray.ray_pyembree
        HAS_TRIMESH = HAS_PYEMBREE = True
    except AttributeError:
        HAS_TRIMESH, HAS_PYEMBREE = True, False
except ImportError:
    HAS_TRIMESH = HAS_PYEMBREE = False

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── Utilities ───────────────────────────────────────────────────────────────
def load_step_file(filepath: str):
    reader = STEPControl_Reader()
    if reader.ReadFile(filepath) != 1:
        raise RuntimeError(f"Failed to read STEP: {filepath}")
    reader.TransferRoots()
    return reader.OneShape()


# ── Hemisphere direction sampling ───────────────────────────────────────────
def hemisphere_grid_sampling(axes: np.ndarray, num_elev=8, num_azim=16) -> np.ndarray:
    """Generate (num_elev*num_azim, 3) unit directions on a hemisphere."""
    x, y, z = axes[:, 0].astype(np.float64), axes[:, 1].astype(np.float64), axes[:, 2].astype(np.float64)

    if num_elev == 1 and num_azim == 1:
        zn = z / max(np.linalg.norm(z), 1e-12)
        return zn.reshape(1, 3)

    elev = (np.pi / 2) * (np.arange(num_elev)[:, None] + 0.5) / num_elev
    az = 2 * np.pi * np.arange(num_azim)[None, :] / num_azim

    dirs = (np.cos(elev)[..., None] * z
            + np.sin(elev)[..., None] * (np.cos(az)[..., None] * x + np.sin(az)[..., None] * y))
    nrm = np.linalg.norm(dirs, axis=2, keepdims=True)
    return np.divide(dirs, nrm, out=np.zeros_like(dirs), where=nrm != 0).reshape(-1, 3)


# ── B-Rep ray casting ──────────────────────────────────────────────────────
def raycast_hemisphere(
    shape, center, axes, num_elev=8, num_azim=16,
    compute_dot=False, max_dist=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cast hemisphere rays from *center* against a B-Rep *shape*."""
    tol_inter, tol_ray = 1e-2, 1e-4
    max_d = max_dist or 1e6
    N = num_elev * num_azim
    shape2d = (num_elev, num_azim)

    dirs = hemisphere_grid_sampling(axes, num_elev, num_azim)
    dirs_opp = hemisphere_grid_sampling(-axes, num_elev, num_azim)

    inter = IntCurvesFace_ShapeIntersector()
    try:
        inter.Load(shape, tol_inter)
    except Exception:
        z = np.zeros(shape2d, dtype=np.float32)
        grids = {"occupancy_grid": z.copy(), "distance_grid": z.copy(),
                 "occupancy_grid_opposite": z.copy(), "distance_grid_opposite": z.copy()}
        if compute_dot:
            grids["dot_grid"] = z.copy()
            grids["dot_grid_opposite"] = z.copy()
        return center, grids

    def _cast(dirs_arr):
        occ = np.zeros(shape2d, dtype=np.float32)
        dist = np.zeros(shape2d, dtype=np.float32)
        dot = np.zeros(shape2d, dtype=np.float32) if compute_dot else None
        normals = np.zeros((N, 3), dtype=np.float32) if compute_dot else None
        origin = gp_Pnt(*center)

        for idx, d in enumerate(dirs_arr):
            ei, ai = divmod(idx, num_azim)
            try:
                inter.Perform(gp_Lin(origin, gp_Dir(*d)), tol_ray, max_d)
            except RuntimeError:
                continue
            if inter.NbPnt() <= 0:
                continue
            try:
                pt = inter.Pnt(1)
                hit_face = inter.Face(1)
                dd = np.linalg.norm(np.array([pt.X(), pt.Y(), pt.Z()]) - center)
                occ[ei, ai] = 1.0
                dist[ei, ai] = dd
            except Exception:
                continue
            if compute_dot and pt is not None and hit_face is not None:
                try:
                    surf = BRepAdaptor_Surface(hit_face)
                    gs = surf.Surface().Surface()
                    proj = GeomAPI_ProjectPointOnSurf(pt, gs)
                    if proj.NbPoints() > 0:
                        u, v = proj.LowerDistanceParameters()
                        P, D1U, D1V = gp_Pnt(), gp_Vec(), gp_Vec()
                        gs.D1(u, v, P, D1U, D1V)
                        nv = D1U.Crossed(D1V)
                        if nv.Magnitude() > 0:
                            nv.Normalize()
                            normals[idx] = [nv.X(), nv.Y(), nv.Z()]
                except Exception:
                    pass

        if compute_dot and normals is not None:
            valid = np.linalg.norm(normals, axis=1) > 0
            dots = np.einsum("ij,ij->i", dirs_arr, normals)
            for i in np.where(valid)[0]:
                dot[i // num_azim, i % num_azim] = dots[i]
        return occ, dist, dot

    occ, dist, dot = _cast(dirs)
    occ_o, dist_o, dot_o = _cast(dirs_opp)

    grids = {"occupancy_grid": occ, "distance_grid": dist,
             "occupancy_grid_opposite": occ_o, "distance_grid_opposite": dist_o}
    if compute_dot:
        grids["dot_grid"] = dot
        grids["dot_grid_opposite"] = dot_o
    return center, grids


# ── Mesh tessellation ───────────────────────────────────────────────────────
def _collect_faces(shape) -> List:
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    return faces


def tessellate_shape(shape, lin_defl=0.01, ang_defl=0.05):
    """Tessellate B-Rep → (vertices, triangles, face_ids, faces)."""
    mesh = BRepMesh_IncrementalMesh(shape, lin_defl, False, ang_defl, True)
    mesh.Perform()
    if not mesh.IsDone():
        raise RuntimeError("Tessellation failed")

    verts, tris, fids = [], [], []
    voff, faces = 0, []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    fi = 0
    while exp.More():
        face = exp.Current(); faces.append(face)
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            nn = tri.NbNodes()
            for i in range(1, nn + 1):
                p = tri.Node(i); p.Transform(trsf)
                verts.append([p.X(), p.Y(), p.Z()])
            rev = face.Orientation() == TopAbs_REVERSED
            for i in range(1, tri.NbTriangles() + 1):
                n1, n2, n3 = tri.Triangle(i).Get()
                off = voff - 1
                t = [n1+off, n3+off, n2+off] if rev else [n1+off, n2+off, n3+off]
                tris.append(t); fids.append(fi)
            voff += nn
        fi += 1; exp.Next()

    return (np.array(verts, dtype=np.float64), np.array(tris, dtype=np.int32),
            np.array(fids, dtype=np.int32), faces)


# ── Mesh-based ray caster ──────────────────────────────────────────────────
class MeshRayCaster:
    """Fast ray caster: tessellates once, then uses trimesh for queries."""

    def __init__(self, shape, linear_deflection=None, angular_deflection=0.5):
        if not HAS_TRIMESH:
            raise ImportError("trimesh required for mesh rays (pip install trimesh pyembree)")

        bbox = Bnd_Box()
        brepbndlib.Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        if linear_deflection is None:
            linear_deflection = max(xmax-xmin, ymax-ymin, zmax-zmin) / 100
        self.linear_deflection = linear_deflection

        self.vertices, self.triangles, self.face_ids, self.faces = tessellate_shape(
            shape, linear_deflection, angular_deflection
        )
        self.mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.triangles)
        try:
            self.intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        except Exception:
            self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def intersect_rays(self, origins, directions):
        """Returns (locations, distances, face_ids, normals). Misses → inf / NaN / -1."""
        n = len(origins)
        locs = np.full((n, 3), np.nan, dtype=np.float64)
        dists = np.full(n, np.inf, dtype=np.float64)
        fids = np.full(n, -1, dtype=np.int32)
        nrms = np.zeros((n, 3), dtype=np.float64)

        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = np.divide(directions, norms, where=norms > 0, out=np.zeros_like(directions))

        loc, idx_ray, idx_tri = self.intersector.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False
        )
        if len(loc) > 0:
            d = np.linalg.norm(loc - origins[idx_ray], axis=1)
            order = np.lexsort((d, idx_ray))
            _, first = np.unique(idx_ray[order], return_index=True)
            sel = order[first]
            rays = idx_ray[sel]
            dists[rays] = d[sel]
            locs[rays] = loc[sel]
            fids[rays] = self.face_ids[idx_tri[sel]]
            nrms[rays] = self.mesh.face_normals[idx_tri[sel]]
        return locs, dists, fids, nrms


# ── Mesh-based hemisphere casting ──────────────────────────────────────────
def raycast_hemisphere_mesh(
    ray_caster: MeshRayCaster, center, axes,
    num_elev=8, num_azim=16, compute_dot=False, max_dist=None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Hemisphere ray casting via pre-built MeshRayCaster."""
    max_d = max_dist or 1e6
    eps = max(2.0 * ray_caster.linear_deflection, 1e-3)
    c = center.astype(np.float64)

    dirs = hemisphere_grid_sampling(axes, num_elev, num_azim)
    dirs_opp = hemisphere_grid_sampling(-axes, num_elev, num_azim)
    shape2d = (num_elev, num_azim)

    _, hd, _, hn = ray_caster.intersect_rays(c + eps * dirs, dirs)
    _, hd_o, _, hn_o = ray_caster.intersect_rays(c + eps * dirs_opp, dirs_opp)

    for d in (hd, hd_o):
        v = d < np.inf
        d[v] += eps
        d[v & (d > max_d)] = np.inf

    occ, occ_o = ~np.isinf(hd), ~np.isinf(hd_o)
    grids = {
        "occupancy_grid": occ.astype(np.float32).reshape(shape2d),
        "distance_grid": np.where(occ, hd, 0).astype(np.float32).reshape(shape2d),
        "occupancy_grid_opposite": occ_o.astype(np.float32).reshape(shape2d),
        "distance_grid_opposite": np.where(occ_o, hd_o, 0).astype(np.float32).reshape(shape2d),
    }
    if compute_dot:
        def _dot(mask, dirs_arr, normals):
            d = np.zeros(mask.size, dtype=np.float32)
            if mask.any():
                d[mask] = np.einsum("ij,ij->i", dirs_arr[mask], normals[mask]).astype(np.float32)
            return d.reshape(shape2d)
        grids["dot_grid"] = _dot(occ, dirs, hn)
        grids["dot_grid_opposite"] = _dot(occ_o, dirs_opp, hn_o)
    return center, grids
