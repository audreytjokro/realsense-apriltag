# Line-raster experiments

These sessions use a serpentine scan across a straight odor source.

The current notebook 04 mint-source retry is stored under
`background-referenced/blotter-retry-01/` and compared with the earlier clean
background `line_raster_blank_fresh_01`. It has excellent digital timing and a
strong response, but the exact oil dose is unknown and was reported excessive.
The response is positive across 96.3% of shared cells and approximately 14 cm
wide by the row-profile half-maximum measure, so the run is more useful for
detectability than edge sharpness. A height audit also excludes 21/147
in-bounds mint readings above 3.5 cm at Desk Y 31.2–43.8 cm. Do not use this
retry to validate source width or concentration.

The earlier clean-block pilot pair is stored together under
`matched-pairs/clean-block-01/`:

- `line_raster_blank_fresh_01` was recorded first on fresh backing;
- `line_raster_mint_fresh_01` was recorded second with a reported 5–6 drops
  and approximately 4 cm line width;
- the desk tag, camera, raster bounds, lane guides, starting corner, and scan
  direction were held fixed between the two runs.

`analysis/notebooks/04_mint_vs_blank_raster.ipynb` applies the independently
estimated 3.0 s lag correction, restricts both runs to common spatial support,
and directly subtracts the blank map from the mint map. Its height audit found
that 20/200 mint readings—but no blank readings—were removed for exceeding the
3.5 cm height limit, all at Desk Y 30.5–36.0 cm. This creates localized gaps at
the apparent source crossing. The pair shows a condition difference but is not
valid line-geometry evidence. Source coordinates were also not recorded, and
Desk X is strongly confounded with elapsed scan time.

The earlier mint-only pilot is stored under `pilot-usable/`. It has good
digital timing and enough desk-coordinate coverage for analysis, but lacks its
own matched blank raster and recorded source endpoints.

Analysis lives in
`analysis/notebooks/03_line_raster_mint_01.ipynb`. The notebook applies the
independently estimated 3.0 s physical response-lag correction; the raw CSV is
never shifted or rewritten.

The first dry-paper-towel control is stored under
`excluded-controls/contamination-suspected/`. The towel was placed over the
desk area that had held the mint towel, so this run is preserved as a carryover
diagnostic and must not be treated as a clean control or classifier example.

The contamination-suspected run remains excluded even though notebook 04 now
uses the reviewed clean-block-01 pilot. It must not be used as a clean control or
classifier example.
