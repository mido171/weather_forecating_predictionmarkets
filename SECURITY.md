# Security

Do not commit API keys, private keys, cookies, authenticated headers, trading credentials, or personal data.

Report accidental secret exposure by immediately revoking the credential, removing it from active systems, and documenting the incident. Rewriting Git history is not a substitute for revocation.

Research jobs should have no live-trading permission. Execution credentials, if ever introduced after production approval, must be least privilege and isolated from Codex research environments.
