# Maintainer release checklist

## Source and legal

- [ ] Confirm the official software author list, repository URL, and contact
      information in README and release metadata.
- [ ] Confirm the software copyright holder and Apache-2.0 choice.
- [ ] Review all third-party notices and current upstream model/data terms.
- [ ] Run a secret scan and `python scripts/check_release.py`.
- [ ] Verify no ignored weights/datasets are force-added.

## Reproducibility

- [ ] Build a fresh environment from `environment.yml` on a clean machine.
- [ ] Download and verify every prerequisite with `download_weights.py`.
- [ ] Run CPU tests and the 512×512 GPU smoke test.
- [ ] Run at least one clean training resume and base-plus-adapter inference.
- [ ] Publish exact split manifests, seeds, commands, configs, hardware, and logs.
- [ ] Publish baseline and claimed paper metrics with protocols.

## Release artifacts

- [ ] Export trainable-only checkpoint and complete `MODEL_CARD.md`.
- [ ] Add repository URL and real software authors to `CITATION.cff`.
- [ ] Complete all model-card fields in public artifacts.
- [ ] Create a signed/tagged semantic version and attach checksums.
- [ ] Enable GitHub branch protection, Actions, issues, discussions as desired,
      private vulnerability reporting, and release notes.
