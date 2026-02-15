# Criteria Implementation

This system separates official judging criteria from scan-derived proxy scores.

Official judging axes
The official axes are taken directly from federation documents and stored in `data/federations/criteria_kb_v1.yaml`. These axes are used only as labels for the scorecard; internal weights are not official.

Scan-based proxy scores (non-official)
The app produces proxy scores that approximate how a scan aligns to each official axis. These are explicitly not judge scores and do not estimate bodyfat percentage. Conditioning is represented by appearance proxies only.

## Proxy Definitions

Shape proxies (geometry)
- `shoulder_width`: width at deltoid level from the front silhouette/point cloud cross-section.
- `chest_width`: width at mid-pec level.
- `waist_width`: minimal width between lower ribs and iliac crest zone.
- `hip_width`: width at greater trochanter level.
- `thigh_width_L/R`: maximal width in thigh band (left/right if available).
- `calf_width_L/R`: maximal width in calf band (left/right if available).
- `arm_girth_proxy_L/R`: upper arm cross-section circumference proxy (left/right if available).
- `v_taper_ratio`: `shoulder_width / waist_width`.
- `x_frame_ratio`: `shoulder_width / hip_width`.
- `waist_to_hips_ratio`: `waist_width / hip_width`.
- `symmetry_left_right`:
  - `thigh_symmetry = abs(L-R)/mean(L,R)`
  - `arm_symmetry = abs(L-R)/mean(L,R)`
  - `calf_symmetry = abs(L-R)/mean(L,R)`
- `leg_to_torso_balance`: `thigh_volume_proxy / upper_torso_volume_proxy` from circumference or slab-area bands.
- `posture_flags` (pose landmarks if available):
  - `shoulder_level_delta`
  - `hip_level_delta`
  - `torso_rotation_deg`
  - `stance_width_ratio`

Condition proxies (appearance)
- `shadow_contrast` zones (abs/serratus/quads/glute-ham) from RGB.
- `silhouette_edge_sharpness` zones (waist/delt_cap/quad_sweep).
- `condition_score` (0-100 proxy derived from contrast + sharpness).
- `confidence` (high/med/low) based on quality gates.

Quality gates (stored in `session_summary.json`)
- `lighting_stable`: RGB histogram similarity vs baseline.
- `distance_stable`: Z-center stability within tolerance.
- `pose_locked_ok`: stance width and arm abduction within tolerance.
- `pointcloud_quality`: num_points, coverage %, hole score, ICP residuals if available.

## Mapping to Judging Axes

- Symmetry: low L/R asymmetry increases the symmetry proxy score. Posture flags reduce confidence.
- Balance & proportions: `v_taper_ratio`, `x_frame_ratio`, `leg_to_torso_balance`, and `waist_to_hips_ratio` drive the proxy score.
- Muscularity: region volume proxies (upper torso and thigh) normalized by waist/hip widths.
- Conditioning: `condition_score` from RGB + silhouette proxies; no bodyfat percentage is estimated.
- Presentation: pose compliance plus posture/stance alignment flags.

## Output Traceability

All scorecards and recommendations include:
- `kb_version`
- `citations_used`: list of `{title, url}` entries from the KB and evidence sources.

If confidence is low, the scorecard includes a “Do a locked rescan” flag with a checklist.
