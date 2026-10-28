# Sealed holdout record

The sealed holdouts of the paper were written down, committed and only then evaluated
once. The dates and commits below are from the development repository, from which this
repository was extracted without its history. The SHA-256 digests identify the files
exactly as published here.

| File | SHA-256 | First committed | Commit |
|---|---|---|---|
| `configs/holdout_ellipse.toml` (holdout A) | `3f795c8d90f845bee0c835b5c49348b2047e0974e25c9c21ee793f8940bf219b` | 2026-07-11 01:45 +0545 | `d0a0fae7c53536967a95f92c85f9df4efb19139d` |
| `configs/holdout_mahalanobis.toml` (Bessel holdout) | `de76e15e1d9c572daf4aaa37d4a2a218c4fe11e7ea53f6a10b180aaef8194891` | 2026-07-11 01:45 +0545 | `d0a0fae7c53536967a95f92c85f9df4efb19139d` |
| `configs/holdout_ellipse_2026b.toml` (holdout B) | `82cf5229a5e526f447ce877f37540a4269b899520c623a44c50380398128662e` | 2026-07-17 07:53 +0545 | `082888b16965ca856ef5ba99546a665cce47f02b` |
| `configs/holdout_ellipse_2026c.toml` (holdout C) | `95f6c29b3772110b1c6cd25f36997027bf3cf9227ed6bb10da0f1b308457b786` | 2026-07-17 07:53 +0545 | `082888b16965ca856ef5ba99546a665cce47f02b` |

| Evaluation record | First committed | Commit |
|---|---|---|
| `validation/holdout_evaluation.json` (holdout A, Bessel holdout) | 2026-07-12 19:43 +0545 | `5603fd0e0c6cb2e7b171b9829fe96e8bc71cf930` |
| `validation/holdout_evaluation_r4.json` (holdouts B and C) | 2026-07-17 08:13 +0545 | `c69f501db0b3baed03762a74cb23005fc3285b39` |

Holdouts B and C were drawn by a seeded rule (seed 20260716, no redraws). None of the
holdout files has been modified since it was first committed; they are published
byte for byte, comments included.
