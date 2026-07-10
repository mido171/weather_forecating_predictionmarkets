# Experiment campaigns

This is the human entry point for the HKG Tmax experiment archive. The
2026-07-10 consolidation reduced the campaign tree from 813 Markdown files to
26 canonical READMEs without deleting machine evidence.

| Campaign | Scope |
|---|---|
| [HKG Tmax](hkg-tmax/README.md) | Point forecasts, public-weather acquisition, persistence, and capacity |
| [HKG T-24](hkg-t24/README.md) | Historical T-24 acquisition and point-forecast work |
| [Probability](probability/README.md) | Weather-only bucket and distribution calibration |
| [Residual modeling](residual-modeling/README.md) | Residual-model evidence and compatibility copies |
| [Market edges](market-edges/README.md) | Model-versus-market snapshots; never order execution |

## Reading contract

1. Read this file.
2. Open the campaign README.
3. Open the experiment README.
4. Consult YAML/JSON/CSV/Parquet evidence only when the summary is insufficient.

Every campaign and top-level experiment has one human-readable `README.md`.
Nested shard, run, result, handoff, and compatibility directories contain
machine evidence only.

## Provenance and exact recovery

[`DOCUMENT_PROVENANCE.csv`](DOCUMENT_PROVENANCE.csv) records all 821 original
Markdown/text files, including original path, SHA-256, byte count, destination
README, disposition, and source commit
`b2da67aca12cdc2c2da69a691a18793ef3c35d88`.

`DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip` contains the exact pre-consolidation
working-tree bytes for all 821 records. It has 821 entries, is 834,756 bytes,
and has SHA-256
`3f82cf4abd5b3c5983aaf24f2b4fb6b8309f57eb7be38e7779f588200a515e22`.
Every archive member was independently checked against the CSV.

Recover the complete historical documentation set with:

```powershell
Expand-Archive .\DOCUMENT_ARCHIVE_PRE_CONSOLIDATION.zip -DestinationPath <recovery-directory>
```

The recorded Git commit preserves the normalized repository blobs. The ZIP is
the byte-exact recovery source because the retired Windows working-tree files
used CRLF line endings.

Historical absolute paths inside machine artifacts remain provenance, not
current operating instructions.
