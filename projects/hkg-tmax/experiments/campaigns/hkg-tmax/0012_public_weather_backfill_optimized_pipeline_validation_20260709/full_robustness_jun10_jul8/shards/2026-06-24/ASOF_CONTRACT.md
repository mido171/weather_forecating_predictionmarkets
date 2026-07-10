# As-Of Contract

Model issues use `issued_at_utc + 6h` as a conservative availability proxy unless a stronger
provider timestamp is captured. Himawari uses the later of native HSD file creation time and
observed time plus 30 minutes. ENVF radar frames use observed time plus 30 minutes and are
marked as historical display proxy, not native exact radar issue metadata.
