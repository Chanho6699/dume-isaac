# my_so101/urdf

Official TheRobotStudio SO-101 files, byte-identical to upstream, pinned in
[../source_manifest.yaml](../source_manifest.yaml) (commit + sha256). Fetch / verify:

```bash
python scripts/script02_fetch_so101_source.py            # download missing, verify all
python scripts/script02_fetch_so101_source.py --verify-only
```

Layout keeps the upstream relative mesh paths (`assets/*.stl`), so the URDF needs no edits:

```
urdf/LICENSE                  upstream Apache-2.0 license
urdf/so101_new_calib.urdf     selected model
urdf/assets/*.stl             the 13 meshes referenced by that URDF
```

Fetched files are git-ignored (reproducible from the manifest). Meshes stay here instead of
a separate `meshes/` directory on purpose: moving them would require editing the upstream URDF.
