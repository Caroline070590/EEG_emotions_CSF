# Reconstructed behavioral input provenance

Source: NEMAR dataset `nm000112`, FACED version `v1.1.3`.

- Dataset DOI: `10.82901/nemar.nm000112`
- Manifest URL: `https://data.nemar.org/nm000112/v1.1.3/manifest.json`
- Retrieved: 2026-09-04
- Event tables: 123, `sub-000` through `sub-122`
- Total event-table bytes: 1,511,524
- Manifest checksum failures: 0

`FACED_official_events_v1.1.3.json` maps the three-digit participant ID to the
verbatim corresponding official event TSV text, as required by the recovered
Appendix G generator. It was serialized deterministically with sorted keys,
UTF-8, two-space JSON indentation, and a terminal newline.

Reconstructed JSON SHA-256:

`f5552d286b702338865659148fe76f88e0bd031e369b80bb197daa215bb6176a`

The recovered historical manifest reports a different JSON SHA-256. The TSV
content is pinned and checksum-verified, but the historical JSON byte
serialization was not preserved; therefore this file is explicitly labeled
as reconstructed.
