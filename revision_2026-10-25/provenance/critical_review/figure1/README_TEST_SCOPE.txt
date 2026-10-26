The smoke_verify.py and SMOKE_VALIDATION.json files document the historical,
isolated repair. smoke_verify.py intentionally checks that the old pre-repair
release import fails; it must NOT be described as the final-release test.

For a portable final-release test, keep smoke_release.py beside src/cache and
reference_rasters, then run:
  python smoke_release.py --release-root /path/to/release --out /path/to/output

This resolves the actual release source, hashes it, copies only lib and
part1_topology into a temporary directory, imports and runs that code on the
three retained metric CSV fixtures (about1.4MB total, no raw EEG), verifies162
heatmap cells against the original raster references, checks q13 rejection,
and independently checks the recovered mean/bootstrap helper. Output figures
and RELEASE_FIGURE1_SMOKE.json are written only under --out. Release source is
unchanged. Runtime and environment requirements follow the Figure1 generator.

The source repair changes only two files, documented in
PROPOSED_FIX_PROVENANCE.json. Supporting src/lib files here belonged to the
historical isolated proof; smoke_release.py uses lib from --release-root.
