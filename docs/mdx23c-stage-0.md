# MDX23C Stage 0: checkpoint compatibility proof

## Decision

MDX23C will be explored as a possible higher-quality buffered vocals/instrumental separator. The implementation is deliberately staged because an MDX23C architecture name alone does not prove that a particular YAML configuration and checkpoint are compatible.

Stage 0 proves only that the selected assets can be acquired reproducibly, the exact model architecture can be constructed from the selected configuration, and the checkpoint can be loaded strictly without network access.

A successful Stage 0 does **not** create a usable karaoke processor and must not be represented as one.

## Selected assets

Hugging Face repository:

```text
Politrees/UVR_resources
```

Pinned revision:

```text
0472e5cd4e0f77b95ca2df7a8162992e6799fa49
```

Files under `MDX23C_models/`:

```text
MDX23C-8KFFT-InstVoc_HQ.ckpt
model_2_stem_full_band_8k.yaml
```

Expected image destination:

```text
/models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt
/models/mdx23c/model_2_stem_full_band_8k.yaml
```

The implementation must use exact file selection rather than downloading unrelated UVR resources.

## Architecture reference

Prefer the current `ZFTurbo/Music-Source-Separation-Training` implementation as the reference for MDX23C construction and checkpoint handling.

The older `ZFTurbo/mvsep-mdx23-music-separation-model` repository may be useful historical context, but it is an ensemble/competition solution and is not the preferred dependency or runtime embedding boundary.

Stage 0 may do one of the following:

1. vendor the minimal MDX23C architecture/config-loader surface required to construct and load this checkpoint; or
2. install or copy a precisely pinned minimal upstream inference surface during the image build.

Whichever route is chosen must be documented with repository, commit, source paths and licence. Do not vendor an entire training, GUI or ensemble repository when only a small inference surface is needed.

## In scope

- Add immutable build arguments for the Hugging Face repository and revision.
- Download only the selected YAML and checkpoint into `/models/mdx23c`.
- Add the minimum compatible dependencies needed for YAML parsing, model construction and checkpoint loading.
- Add a dedicated Python smoke-test entry point or module.
- Parse the selected YAML using the same semantics intended for later inference integration.
- Construct the exact MDX23C model described by the YAML.
- Load the selected checkpoint onto CPU with `map_location="cpu"`.
- Normalise only known, documented wrapper prefixes such as `module.` when necessary.
- Load with `strict=True` after any justified normalisation.
- Put the model into evaluation mode.
- Print concise evidence: resolved asset paths, model class/type, parameter count, expected sample rate, configured target instruments/stems, and success.
- Make the Docker build fail on missing assets, parse failure, architecture mismatch, missing/unexpected keys or any other load failure.
- Provide a reproducible owner-run command that repeats the proof offline from the built image.
- Add focused automated tests for any local state-dict normalisation or validation logic that does not require downloading the large checkpoint.

## Explicitly out of scope

Do not implement any of the following in Stage 0:

- `MDX23CProcessor` or any `AudioProcessor` implementation;
- processor registry entry `mdx23c-vocals`;
- KANY packet decoding or reconstruction;
- buffering, overlap-add, resampling or paced output;
- actual audio inference;
- runtime environment settings for MDX23C;
- FastAPI runtime-settings models or endpoints for MDX23C;
- console controls for MDX23C;
- live vocal-reduction changes;
- GPU execution, latency, RTF, VRAM or subjective quality claims;
- refactoring HTDemucs or ConvTasNet into a shared separator abstraction;
- renaming `Dockerfile.demucs` or `compose.demucs.yaml`.

These belong to later, separately reviewed stages.

## Required smoke-test behaviour

The smoke test must be an executable repository-owned module rather than an opaque one-line Docker command. A representative invocation is:

```bash
python3 -m audio_trombone.tools.validate_mdx23c_checkpoint \
  --config /models/mdx23c/model_2_stem_full_band_8k.yaml \
  --checkpoint /models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt
```

The exact module path may differ, but it should remain independently runnable and testable.

Successful output should clearly identify what was proven, for example:

```text
MDX23C checkpoint validation
config: /models/mdx23c/model_2_stem_full_band_8k.yaml
checkpoint: /models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt
model: <resolved class>
parameters: <count>
sample_rate: <value from configuration>
targets: <configured instruments>
strict checkpoint load: OK
```

Do not hard-code parameter count, sample rate or stem names merely to satisfy the expected output. They must be derived from the loaded configuration/model.

## Checkpoint-loading rules

Checkpoint formats vary. The loader may encounter a raw state dictionary or a wrapper such as `state_dict` or `model`. It may also encounter a `module.` prefix from `DataParallel`.

Any handling must be narrow and visible:

- inspect and document the actual checkpoint structure;
- accept only explicitly supported wrapper keys;
- remove only explicitly recognised uniform prefixes;
- reject ambiguous or mixed-prefix state dictionaries;
- call `load_state_dict(..., strict=True)`;
- report missing and unexpected keys as failure, never warning-only success.

Do not use `strict=False` to make the proof pass.

## Reproducibility and offline requirement

The Docker build may use the network to download pinned assets. After download, the validation command must not make network calls.

The owner-run validation must work with networking disabled, for example:

```bash
docker compose -f compose.yaml -f compose.demucs.yaml run --rm --no-deps \
  karaoke-anything \
  python3 -m audio_trombone.tools.validate_mdx23c_checkpoint \
    --config /models/mdx23c/model_2_stem_full_band_8k.yaml \
    --checkpoint /models/mdx23c/MDX23C-8KFFT-InstVoc_HQ.ckpt
```

Where supported, also test with Docker networking disabled. Runtime model loading in later stages must use these baked local assets rather than Hugging Face downloads.

## Acceptance criteria

Stage 0 is complete only when all of the following are true:

1. Asset provenance and all relevant revisions/commits are documented.
2. A clean GPU-image build downloads exactly the chosen config and checkpoint.
3. The repository-owned validator constructs the model from the chosen YAML.
4. The checkpoint loads on CPU using strict state-dict matching.
5. The build fails when the checkpoint is absent or deliberately incompatible.
6. The validation can be repeated from the built image without network access.
7. Unit tests cover local validation/normalisation behaviour where practical.
8. No MDX23C processor, registry, runtime API or console surface has been added.
9. The implementation does not introduce broad training/GUI dependencies without an explicit, justified reason.
10. Documentation records any unresolved licensing or upstream compatibility concern rather than guessing.

## Evidence to return after implementation

The implementation report or pull-request description should include:

- files changed;
- exact upstream architecture repository and commit;
- exact Hugging Face revision and asset paths;
- clean-build command;
- successful validator output;
- negative/failure test performed;
- tests run and their results;
- any part not validated on the target host.

## Next stage

Only after Stage 0 is proven should Stage 1 be planned. Stage 1 will establish a small offline inference adapter against a short fixture, including input tensor shape, required sample rate, output shape, target ordering and deterministic numerical sanity checks. It still need not integrate KANY or runtime controls.