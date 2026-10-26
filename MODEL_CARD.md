# Model Card — PFrutas ASL Citizen ST-GCN Fusion v1

**Source-only public edition:** weights, source videos, processed sequences and participant-level records are not distributed here. Aggregate historical metrics are in [results/nested-loso-summary.json](results/nested-loso-summary.json). Artefact paths below identify the original experiment; they are not download links or files bundled in this edition.

## Model summary

| Field | Value |
| --- | --- |
| Model identifier | **asl_citizen_stgcn_fusion_v1** |
| Task | Closed-set recognition of 16 Ecuadorian Sign Language (EcSL) fruit signs from video landmarks |
| Release source | [andresrubiop/ecsl-sign-recognition](https://github.com/andresrubiop/ecsl-sign-recognition) |
| Training run | **fusion_asl_citizen_stgcn_v1_gpu** |
| Training hardware | Tesla V100-PCIE-16GB, CUDA 12.8, PyTorch 2.9.1+cu128, AMP FP16 |
| Validation protocol | Nested leave-one-signer-out (LOSO) cross-validation |
| Evaluation set | One outer-fold prediction for each of 72 eligible videos from 5 signers |
| Intended deployment | NVIDIA Jetson Orin NX 16 GB; MediaPipe on CPU and PyTorch CUDA/FP16 on GPU |

This card describes the target EcSL fruit classifier, not the original ASL Citizen
classifier. It is a research model for the 16 signs listed below. It is not a
general-purpose sign-language recognizer, translator, or accessibility assessment
tool.

The reproducible run is identified by:

    run fingerprint:     c4d49b0fbdddf071f20e5ba12c6c3018893ba454deee3faf8e1d6235b0ef7ce3
    dataset fingerprint: 41e16fd8dfc5b2015e6384abbe0c10439893ee725dc12c182817fda8a6f756ae

The source configuration and evaluation records are
**artifacts/fusion_asl_citizen_stgcn_v1_gpu/training_config.json**,
**cv_metrics.json**, **ensemble.json**, **splits.json**, and
**per_class_metrics.csv**. See [MODEL_RELEASE.md](MODEL_RELEASE.md) for the
publication and checkpoint-release procedure.

## Intended use

The model is intended for research on a constrained vocabulary of EcSL fruit signs
and for local live feedback in the PFrutas prototype. The live interface may show a
class prediction and a model confidence, but this confidence is not calibrated as a
measure of linguistic correctness, learning outcome, or human proficiency.

The model is not intended to:

- translate continuous EcSL or ASL;
- evaluate a signer clinically, educationally, or professionally;
- identify a person, infer demographic attributes, or make a high-stakes decision;
- generalize to unseen vocabulary, new sign languages, arbitrary hand gestures, or
  a NO_GESTURE class;
- guarantee the reported evaluation accuracy on a new camera, background, lighting
  condition, or signer.

## Inputs and outputs

### Input

The model receives a monocular RGB video stream. MediaPipe Holistic extracts pose,
hand, and face landmarks at 30 FPS. The local temporal branch uses a sequence of 48
phase-normalized steps, each with 1,037 values:

- 510 normalized geometric coordinates;
- 13 pose-visibility values and four modality-presence masks, for 527 static values;
- 510 first-order geometric differences, computed only when the corresponding
  landmarks are valid in consecutive frames.

Internal missing-landmark gaps of at most three frames are interpolated; longer gaps
remain masked. An activity detector crops the active sign interval, adds 0.25 seconds
of context when possible, and retains a minimum duration of one second before
resampling.

The transfer branch separately retains normalized image-plane pose-and-hand
coordinates with shape [T, 75, 2]. It converts them to the ASL Citizen graph contract
with shape [2, 128, 27] and float32 values.

### Output labels

The classifier returns a probability distribution over these 16 labels:

    BADEA, CAÑA, CAPULÍ, CIRUELA, COCO, GUABA, LIMÓN, MANGO,
    NARANJILLA, ORITO, SANDÍA, TAXO, TOMATE_DE_ÁRBOL, TORONJA, TUNA, UVILLA

The recommended live confidence threshold is 0.42. It is an application setting, not
a guarantee that a prediction is correct.

## Architecture

The model fuses two landmark-based branches before a 16-class softmax head.

1. **Local EcSL branch.** A one-layer bidirectional LSTM processes the
   48 × 1,037 local landmark sequence. Its projection dimension is 80, hidden size
   is 40 per direction, dropout is 0.25, and temporal attention plus max pooling
   yields a 160-dimensional representation.
2. **Transferred motion branch.** A ten-block ST-GCN initialized from ASL Citizen
   processes the 2 × 128 × 27 pose-and-hand graph and produces a 256-dimensional
   motion embedding.
3. **Fusion.** Both representations are projected to 96 dimensions and combined by
   a learned gate initialized to favour the local landmark branch. The result feeds
   the EcSL 16-class head.

The ASL Citizen classifier head for 2,731 ASL labels is discarded. No ASL-to-EcSL
word mapping or lexical equivalence is assumed. The target model contains 3,263,836
parameters, of which 3,086,298 are in the ST-GCN encoder.

During epochs 1–5, the transferred encoder is frozen. From epoch 6 onward, only its
last block is fine-tuned with learning rate 1e-5; batch-normalization statistics
remain frozen.

## Training data

The source videos come from the
[AI-Ready Ecuadorian Sign Language Corpus](https://doi.org/10.5281/zenodo.21001360)
(Zenodo v1.0.0, CC BY 4.0). The experiment uses a selected lexical subset:

| Item | Value |
| --- | ---: |
| Raw videos inventoried | 76 |
| Eligible videos used | 72 |
| Classes | 16 |
| Signers | 5 |
| Videos per signer | 10, 16, 16, 14, and 16 |
| Videos per class | 3–5 |
| Excluded by default | 4 manually reviewed clips with suffix _x |

The excluded clips are retained in the manifest for traceability:
Ciruela0_x, Tomate_de_Árbol0_x, Toronja3_x, and Uvilla0_x. Exclusion reduces the
influence of substantially different realizations but can also introduce selection
bias; results should therefore be interpreted for this selected subset.

All outer and inner partitions are signer-disjoint. Augmentation is performed only
inside each training partition, so augmented views of a held-out signer cannot enter
training.

## Training procedure and hyperparameters

| Setting | Value |
| --- | --- |
| Outer validation | 5-fold LOSO, one held-out signer per fold |
| Inner validation | LOSO within each outer-training partition |
| Epoch-selection criterion | Pooled inner out-of-fold macro F1 |
| Inner selection budget | 30 epochs |
| Final refit | 27 epochs |
| Batch size | 32 |
| Training views per video per epoch | 4 |
| Optimizer | AdamW |
| Head learning rate | 8e-4 |
| Weight decay | 1e-4 |
| Label smoothing | 0.06 |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Mixed precision | FP16 automatic mixed precision |

Training-time augmentation is deterministic from the run seed, epoch, and sample
index. It includes active-window cropping that retains 88–100% of the sequence,
phase/speed warping from 0.80 to 1.25, horizontal reflection with left/right landmark
exchange (probability 0.5), rotation up to ±7°, scale from 0.90 to 1.10, and
zero-mean Gaussian coordinate noise with standard deviation 0.004. At validation and
inference, random augmentation is disabled and probabilities from deterministic
temporal views 0.90, 1.00, and 1.10 are averaged.

## Held-out signer performance

The values below are out-of-fold (OOF) results: every one of the 72 eligible videos
was predicted by an outer model whose training data excluded that video's signer.
They are the appropriate generalization estimate for this experiment; the final
refit's training accuracy is not a held-out result and must not be reported as such.

| Metric | OOF result |
| --- | ---: |
| Accuracy | 81.94% (59/72) |
| Macro precision | 84.67% |
| Macro recall | 81.35% |
| Macro F1 | 81.15% |
| Top-3 accuracy | 90.28% |

Outer-fold accuracies for signers 0–4 were 90.00%, 81.25%, 62.50%, 85.71%, and
93.75%, respectively. Corresponding macro-F1 values were 86.67%, 75.00%, 53.54%,
77.78%, and 91.67%. The dispersion is material evidence of uncertainty with only
five signers.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| BADEA | 1.00 | 0.50 | 0.67 | 4 |
| CAÑA | 1.00 | 1.00 | 1.00 | 5 |
| CAPULÍ | 0.80 | 0.80 | 0.80 | 5 |
| CIRUELA | 1.00 | 0.67 | 0.80 | 3 |
| COCO | 1.00 | 1.00 | 1.00 | 5 |
| GUABA | 0.67 | 0.40 | 0.50 | 5 |
| LIMÓN | 0.80 | 0.80 | 0.80 | 5 |
| MANGO | 0.80 | 0.80 | 0.80 | 5 |
| NARANJILLA | 0.71 | 1.00 | 0.83 | 5 |
| ORITO | 0.80 | 1.00 | 0.89 | 4 |
| SANDÍA | 1.00 | 1.00 | 1.00 | 5 |
| TAXO | 0.80 | 0.80 | 0.80 | 5 |
| TOMATE_DE_ÁRBOL | 1.00 | 1.00 | 1.00 | 4 |
| TORONJA | 0.67 | 0.50 | 0.57 | 4 |
| TUNA | 1.00 | 0.75 | 0.86 | 4 |
| UVILLA | 0.50 | 1.00 | 0.67 | 4 |

The machine-readable authoritative files are cv_metrics.json, per_class_metrics.csv,
confusion_matrix.csv, and oof_predictions.npz in the training run.

## Known limitations and biases

- The evaluation has only 72 videos from five signers; class supports are 3–5.
  Confidence intervals and performance on a new population remain uncertain.
- There is no independent external corpus, new-camera study, or prospective live-user
  evaluation. OOF LOSO is stronger than a random video split but is not a substitute
  for external validation.
- The four _x recordings were excluded after manual review. That policy improves
  experimental consistency but can underrepresent unusual yet valid signing styles.
- The model covers only 16 fruit signs, has no NO_GESTURE class, and should not be
  used outside this closed vocabulary.
- Landmark quality can degrade with occlusion, framing, clothing/background contrast,
  lighting, camera angle, distance, motion blur, and MediaPipe tracking failures.
- The training augmentations address moderate changes in speed, amplitude, and
  left/right orientation, but do not establish invariance to all such variations.
- ASL Citizen transfer reuses visual-motion features, not language labels. It does
  not establish lexical, cultural, or grammatical equivalence between ASL and EcSL.
- Per-class recall is weakest for GUABA (0.40), BADEA (0.50), and TORONJA (0.50);
  those classes need more diverse data before stronger claims are made.
- Probabilities have not undergone a separate calibration study and must not be
  interpreted as a validated measure of signing quality.

## Privacy, ethics, and safety

The input pipeline processes face, pose, and hand landmarks. Video recordings of
signers are potentially biometric data and must be handled under the applicable
consent, ethics, retention, access-control, and deletion procedures. The public
release should not include identifiable study recordings or participant outcome logs
unless an approved de-identification and sharing process permits it.

The model provides automated feedback only. Human assessment by qualified EcSL
experts remains necessary for any educational, linguistic, or research conclusion
about a person's signing.

## Reproduction

From the repository root, the authoritative workflow is:

    python scripts/00_check_cuda.py
    python scripts/05_prepare_asl_transfer.py
    python scripts/06_download_asl_citizen_weights.py --accept-research-license
    python scripts/07_train_asl_transfer_gpu.py
    python scripts/03_evaluate_oof.py \
      --processed-dir data/processed_asl_transfer_v1 \
      --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu

For local live inference with the trained ensemble:

    python scripts/04_live_inference_gpu.py \
      --run-dir artifacts/fusion_asl_citizen_stgcn_v1_gpu

For Jetson Orin NX deployment, follow
[deploy/orin_nx16/README.md](deploy/orin_nx16/README.md). The full profile uses
the six-member ensemble; the final profile uses one checkpoint and does not inherit
the ensemble OOF metric.

## Availability and third-party terms

The training and inference code is in this repository. The target-model checkpoints
and evaluation artifacts are intentionally excluded from Git by .gitignore; they
must be distributed as versioned release assets rather than assumed to be present in
a source checkout. The exact asset set, checksums, and publication checklist are in
[MODEL_RELEASE.md](MODEL_RELEASE.md).

The original ASL Citizen checkpoint is not bundled. Its download requires explicit
acceptance of Microsoft's research terms through
scripts/06_download_asl_citizen_weights.py --accept-research-license. The adapted
ASL Citizen code is MIT-licensed, but its source weights/materials must not be
redistributed without checking and complying with their applicable terms.
