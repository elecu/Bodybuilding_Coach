#!/usr/bin/env python3
import sys
import numpy as np
import open3d as o3d

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 surface.py <file.pcd> [point_size]")
        return 2

    path = sys.argv[1]
    point_size = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

    p = o3d.io.read_point_cloud(path)
    if p.is_empty():
        print(f"ERROR: empty or invalid point cloud: {path}")
        return 2

    p, _ = p.remove_statistical_outlier(30, 2.0)

    P = np.asarray(p.points)
    if P.size == 0:
        print(f"ERROR: no points after filtering: {path}")
        return 2

    # Rainbow by Z (depth)
    z = P[:, 2]
    t = (z - z.min()) / (np.ptp(z) + 1e-12)

    colors = np.c_[
        np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
        np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
        np.clip(1.5 - np.abs(4 * t - 1), 0, 1),
    ]
    p.colors = o3d.utility.Vector3dVector(colors)

    # Normals + lighting for “surface-like” look
    p.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=80)
    )
    p.orient_normals_consistent_tangent_plane(50)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="surface (rainbow)")
    vis.add_geometry(p)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.light_on = True
    vis.run()
    vis.destroy_window()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
