# Sessions Layout

All new outputs are stored under `sessions/` using a date-based layout.

```
sessions/<user>/<YYYY-MM-DD>/
  poses/
    <HH-MM-SS>_pose_<pose_id>[_a|_b]/
      raw/
      derived/
      media/
      reports/
      exports/
  metrics/
    <HH-MM-SS>_scan3d_<locked|free>[_a|_b]/
      raw/
        view_000/
        view_090/
        view_180/
        view_270/
      derived/
      media/
      reports/
      exports/
```

## Anti-overwrite
If a session folder already exists, a suffix is appended (`_a`, `_b`, `_c`, ...).

## Scan 3D Outputs
- Raw view data is stored under `raw/view_XXX/`.
- Session metadata lives in `derived/meta.json`.
- Scan metrics are saved to `derived/metrics.json`.
- Condition proxy is saved to `derived/condition.json`.

## Pose Outputs
- Pose capture media is stored under `media/` inside the pose session.
- Pose metadata is stored in `derived/meta.json`.

## Reports
Generate reports for a scan session with:

```bash
python3 -m scripts.make_report --session_dir sessions/<user>/<YYYY-MM-DD>/metrics/<session>
```

This writes:
- `derived/session_summary.json`
- `derived/scorecard.json`
- `reports/report_compact.pdf`
- `reports/report_full.pdf`
