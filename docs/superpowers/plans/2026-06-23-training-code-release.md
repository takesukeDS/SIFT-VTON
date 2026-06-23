# Training Code Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_siftvton.py` and `cldm/logger.py` to the public branch and document training in README.md.

**Architecture:** File rename (yahavton → siftvton branding) with no internal edits required; README gets updated TODO checkboxes and a new Training section with an annotated example command.

**Tech Stack:** Python, PyTorch Lightning, OmegaConf, git

---

## File Map

| File | Action |
|---|---|
| `train_siftvton.py` | Create (copy of `train_yahavton.py`, filename only) |
| `cldm/logger.py` | Stage (already exists on disk, currently untracked) |
| `README.md` | Modify: TODO checkboxes (lines 7–15) + insert `## Training` before `## Citation` |

---

### Task 1: Add `train_siftvton.py` and `cldm/logger.py`

**Files:**
- Create: `train_siftvton.py`
- Stage: `cldm/logger.py`

- [ ] **Step 1: Confirm no "yahavton" content in the source file**

```bash
grep -i "yahavton\|yaha" train_yahavton.py
```

Expected: no output (zero occurrences).

- [ ] **Step 2: Copy to the public-facing filename**

```bash
cp train_yahavton.py train_siftvton.py
```

- [ ] **Step 3: Verify syntax and that no "yahavton" crept in**

```bash
python -m py_compile train_siftvton.py && echo "OK"
grep -i "yahavton\|yaha" train_siftvton.py && echo "FOUND — fix before continuing" || echo "clean"
```

Expected output:
```
OK
clean
```

- [ ] **Step 4: Stage both files**

```bash
git add train_siftvton.py cldm/logger.py
```

- [ ] **Step 5: Verify staging**

```bash
git status --short train_siftvton.py cldm/logger.py
```

Expected:
```
A  cldm/logger.py
A  train_siftvton.py
```

- [ ] **Step 6: Commit**

```bash
git commit -m "$(cat <<'EOF'
add training script and logger dependency

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Update README.md TODO checkboxes

**Files:**
- Modify: `README.md:7–15`

- [ ] **Step 1: Confirm current TODO block**

```bash
sed -n '7,15p' README.md
```

Expected:
```
## TODO
- [x] Code for preprocessing
- [ ] Filtered SIFT correspondences(json) on VITON-HD dataset
- [ ] Code for training SIFT-VTON
- [ ] Code for inference 
- [ ] Trained weights of SIFT-VTON
- [x] Instructions for preprocessing
- [ ] Instructions for training 
- [ ] Instructions for inference 
```

- [ ] **Step 2: Apply the edit**

Using the Edit tool, replace:
```
## TODO
- [x] Code for preprocessing
- [ ] Filtered SIFT correspondences(json) on VITON-HD dataset
- [ ] Code for training SIFT-VTON
- [ ] Code for inference 
- [ ] Trained weights of SIFT-VTON
- [x] Instructions for preprocessing
- [ ] Instructions for training 
- [ ] Instructions for inference 
```
with:
```
## TODO
- [x] Code for preprocessing
- [x] Filtered SIFT correspondences(json) on VITON-HD dataset
- [x] Code for training SIFT-VTON
- [x] Code for inference 
- [x] Trained weights of SIFT-VTON
- [x] Instructions for preprocessing
- [x] Instructions for training 
- [x] Instructions for inference 
```

- [ ] **Step 3: Verify**

```bash
grep "\- \[ \]" README.md
```

Expected: no output (all items checked).

---

### Task 3: Add `## Training` section to README.md

**Files:**
- Modify: `README.md` (insert after line 100's closing code fence, before `## Citation`)

- [ ] **Step 1: Confirm the insertion point**

```bash
grep -n "^## Citation" README.md
```

Expected: `102:## Citation`

- [ ] **Step 2: Apply the edit**

Using the Edit tool, replace:
```
unzip sift_matching.zip
```

(the last line of the Preprocessing section's final code block)

with:
```
unzip sift_matching.zip
```

followed immediately by inserting the Training section. The full replacement (old_string → new_string):

**old_string:**
```
unzip sift_matching.zip
```

## Citation
```

**new_string:**
```
unzip sift_matching.zip
```

## Training

Download StableVITON's pretrained checkpoint from [StableVITON](https://github.com/rlawjdghek/stableviton) and place it under `ckpts/`. <!-- TODO: fill in exact filename/URL from StableVITON releases -->

SIFT matching data (`sift_matching/` under `[VITON-HD dataset dir]/train/`) is required — see [For SIFT matching and filtering](#for-sift-matching-and-filtering-optional) above.

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_siftvton.py \
    --config_name SIFT-VTON_sift_loss_ave \
    --data_root_dir [VITON-HD dataset dir] \
    --pretrained_path ckpts/[StableVITON checkpoint filename] \
    --save_root_dir [output dir] \
    --save_name train_siftvton \
    --transform_size hflip shiftscale \
    --transform_color hsv bright_contrast \
    --use_sift_loss \
    --sift_loss_scale 0.0005 \
    --snr_gamma 5.0 \
    --accum_iter 8 \
    --batch_size 4 \
    --save_every_n_epochs 20 \
    --valid_epoch_freq 10
```

| Flag | Value | Description |
|---|---|---|
| `--use_sift_loss` | toggle | Enable SIFT correspondence supervision on cross-attention |
| `--sift_loss_scale` | `0.0005` | Weight of SIFT loss relative to diffusion loss |
| `--snr_gamma` | `5.0` | Min-SNR loss weighting |
| `--accum_iter` | `8` | Gradient accumulation steps (effective batch = `batch_size × n_gpus × accum_iter`) |

## Citation
```

- [ ] **Step 3: Verify the section is present**

```bash
grep -n "^## Training" README.md
```

Expected: `102:## Training` (line number may shift by a few).

```bash
grep "train_siftvton.py" README.md
```

Expected: one match in the Training section command.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
add training instructions and complete TODO checklist

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the implementer

- The StableVITON checkpoint placeholder (`<!-- TODO: fill in exact filename/URL -->`) must be replaced manually after checking the StableVITON release page at https://github.com/rlawjdghek/stableviton.
- `visualize_attn_yahavton.py` is intentionally excluded from this release (may be added later on reviewer request).
- WandB is already optional in `train_siftvton.py` via `--wandb_config_path` (defaults to `None`); no README mention needed.
