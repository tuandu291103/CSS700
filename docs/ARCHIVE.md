# Earlier work and recovery

This repository contains the revised thesis study begun on 15 October 2026. The cleanup on 17 October removed the earlier implementation, configurations, tests, protocols, experiment logs, and generated files from the active checkout. It did not erase the Git history or the preserved backups.

## Git history

The final commit before this repository cleanup is:

```text
67a42565594656890193d2feb0443ee3975bfe0b
```

[Browse the previous repository tree](https://github.com/tuandu291103/CSS700/tree/67a42565594656890193d2feb0443ee3975bfe0b).

That history contains the previously tracked files. It does not contain every ignored environment, dataset, or experiment artifact. Consult the local archive for those.

## Verified local ZIP

The backup remains outside the repository, in the adjacent `CSS700_archives` directory:

- `CSS700_before_restart_2026-10-15.zip`
- `CSS700_before_restart_2026-10-15.manifest.json`
- `CSS700_before_restart_2026-10-15.verification.json`
- `CSS700_before_restart_2026-10-15.sha256`

The ZIP covers both identified earlier workspaces, their Git histories and ignored files, plus the original proposal and locally supplied SOM2IMG manuscript. It contains 104,253 regular files and is 1,582,300,693 bytes. The archive was verified against the complete file inventory, SHA-256 hashes and unchanged source files. Its ZIP checksum was rechecked before the 17 October cleanup.

SHA-256:

```text
1642d1f3724f783f1c12dbcf7bd84caa64803512f906f50de664b0d0be1d9ae5
```

The archive is local, not hosted on GitHub, and is not encrypted. It may contain local configuration or private material. Keep it private and make an off-device copy for protection against disk failure. Symlinks preserve their targets as links, not copies of external files. Rebuild relocated virtual environments when needed.

## Additional checkout preservation

Before importing the restart files, all 18 legacy top-level checkout entries, excluding `.git`, were moved into `CSS700_archives/retired_checkout_2026-10-17/`. A `RETIREMENT_RECEIPT.json` records their locations. The move preserved each entry's inode. This includes the old environment, caches, generated artifacts, and data folders as they existed immediately before cleanup.

The separate `CSS700_restart_2026-10-15` source folder also remains unchanged. It serves as a snapshot of the restart; ongoing work belongs in this Git repository. The snapshot's README describes the state on 15 October, before the revised study was published here.

## Recovery precautions

Recover historical files into a new empty directory or separate checkout. Do not extract the ZIP over the active research repository. Verify its checksum and consult the manifest first. The cleanup preserved the existing Git history.
