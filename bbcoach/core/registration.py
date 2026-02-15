from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np


@dataclass
class MergeResult:
    merged_pcd: Optional[Path]
    merged_mesh: Optional[Path]
    fitness: float
    inlier_rmse: float
    ok: bool
    message: str


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception as exc:
        raise RuntimeError("Open3D not available") from exc


def _canonical_view_paths(scan_dir: Path) -> List[tuple[int, Path]]:
    out: List[tuple[int, Path]] = []
    for angle in (0, 90, 180, 270):
        p = scan_dir / "raw" / f"view_{angle:03d}" / f"view_{angle:03d}.pcd"
        if p.exists():
            out.append((angle, p))
    if len(out) >= 2:
        return out

    # Fallback for older layouts, excluding depth aliases.
    for p in sorted((scan_dir / "raw").glob("view_*/*.pcd")):
        if p.stem.endswith("_depth"):
            continue
        stem = p.stem
        if stem.startswith("view_") and len(stem) >= 8:
            try:
                angle = int(stem.split("_")[1])
            except Exception:
                angle = -1
        else:
            angle = -1
        out.append((angle, p))
    return out


def _yaw_matrix(deg: float) -> np.ndarray:
    ang = np.deg2rad(float(deg))
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def _clone_pcd(pcd, o3d):
    try:
        return pcd.clone()
    except Exception:
        try:
            return pcd.copy()
        except Exception:
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = pcd.points
            pcd2.colors = pcd.colors
            pcd2.normals = pcd.normals
            return pcd2


def _safe_center(pcd) -> Optional[np.ndarray]:
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None
    return np.mean(pts, axis=0)


def _prepare_pcd(pcd, voxel_size: float, o3d):
    p = pcd.voxel_down_sample(voxel_size)
    try:
        p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    except Exception:
        pass
    if len(p.points) > 0:
        p.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.5, max_nn=30)
        )
    return p


def _pick_init_transform(source, target, delta_deg: float, voxel_size: float, o3d) -> np.ndarray:
    src_center = _safe_center(source)
    tgt_center = _safe_center(target)
    if src_center is None or tgt_center is None:
        return np.eye(4, dtype=np.float64)

    best_init = np.eye(4, dtype=np.float64)
    best_fit = -1.0
    candidates = [delta_deg] if abs(delta_deg) < 1e-6 else [delta_deg, -delta_deg]
    for deg in candidates:
        init = _yaw_matrix(deg)
        init[:3, 3] = tgt_center - (init[:3, :3] @ src_center)
        try:
            coarse = o3d.pipelines.registration.registration_icp(
                source,
                target,
                voxel_size * 4.0,
                init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            fit = float(coarse.fitness)
        except Exception:
            fit = -1.0
        if fit > best_fit:
            best_fit = fit
            best_init = init
    return best_init


def _pairwise_registration(source, target, voxel_size: float, init: np.ndarray, o3d):
    coarse_dist = voxel_size * 8.0
    fine_dist = voxel_size * 2.0
    result_coarse = o3d.pipelines.registration.registration_icp(
        source,
        target,
        coarse_dist,
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    result_fine = o3d.pipelines.registration.registration_icp(
        source,
        target,
        fine_dist,
        result_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source,
        target,
        fine_dist,
        result_fine.transformation,
    )
    return result_fine, information


def auto_merge(scan_dir: Path, voxel_size: float = 0.008, make_mesh: bool = False) -> MergeResult:
    try:
        o3d = _require_open3d()
    except Exception as exc:
        return MergeResult(None, None, 0.0, 0.0, False, str(exc))
    try:
        view_items = _canonical_view_paths(scan_dir)
        if len(view_items) < 2:
            return MergeResult(None, None, 0.0, 0.0, False, "Not enough views to merge.")

        prepared: List[tuple[int, object]] = []
        for angle, path in view_items:
            p = o3d.io.read_point_cloud(str(path))
            p = _prepare_pcd(p, voxel_size, o3d)
            if len(p.points) > 0:
                prepared.append((angle, p))

        if len(prepared) < 2:
            return MergeResult(None, None, 0.0, 0.0, False, "No valid point clouds for merge.")

        ref_idx = 0
        for i, (angle, _) in enumerate(prepared):
            if angle == 0:
                ref_idx = i
                break
        if ref_idx != 0:
            prepared = [prepared[ref_idx]] + prepared[:ref_idx] + prepared[ref_idx + 1 :]
        angles = [a for a, _ in prepared]
        pcs = [p for _, p in prepared]

        n = len(pcs)
        pose_graph = o3d.pipelines.registration.PoseGraph()
        odometry = np.eye(4, dtype=np.float64)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.eye(4, dtype=np.float64))
        )
        edge_fitness: List[float] = []
        edge_rmse: List[float] = []
        for source_id in range(n):
            for target_id in range(source_id + 1, n):
                source = pcs[source_id]
                target = pcs[target_id]
                delta = 0.0
                if angles[source_id] >= 0 and angles[target_id] >= 0:
                    delta = float(angles[target_id] - angles[source_id])
                init = _pick_init_transform(source, target, delta, voxel_size, o3d)
                result_icp, information_icp = _pairwise_registration(
                    source, target, voxel_size, init, o3d
                )
                edge_fitness.append(float(result_icp.fitness))
                edge_rmse.append(float(result_icp.inlier_rmse))
                uncertain = target_id != source_id + 1
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        result_icp.transformation,
                        information_icp,
                        uncertain=uncertain,
                    )
                )
                if target_id == source_id + 1:
                    odometry = result_icp.transformation @ odometry
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                    )

        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=voxel_size * 2.0,
            edge_prune_threshold=0.25,
            reference_node=0,
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )

        merged = o3d.geometry.PointCloud()
        for i, src in enumerate(pcs):
            pcd_t = _clone_pcd(src, o3d)
            pcd_t.transform(np.array(pose_graph.nodes[i].pose, dtype=np.float64))
            merged += pcd_t
        merged = merged.voxel_down_sample(voxel_size)
        if len(merged.points) == 0:
            return MergeResult(None, None, 0.0, 0.0, False, "Merge failed: empty merged point cloud.")

        exports_dir = scan_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        merged_path = exports_dir / "merged.pcd"
        o3d.io.write_point_cloud(str(merged_path), merged)

        mesh_path = None
        if make_mesh:
            try:
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged, depth=8)
                mesh_path = exports_dir / "merged_mesh.ply"
                o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            except Exception:
                mesh_path = None

        fitness = float(np.mean(edge_fitness)) if edge_fitness else 0.0
        rmse = float(np.mean(edge_rmse)) if edge_rmse else 0.0
        ok = fitness > 0.08
        msg = f"Merge complete ({len(pcs)} views)." if ok else f"Merge completed with low fitness ({len(pcs)} views)."
        return MergeResult(merged_path, mesh_path, fitness, rmse, ok, msg)
    except Exception as exc:
        return MergeResult(None, None, 0.0, 0.0, False, f"Auto-merge failed: {exc}")
