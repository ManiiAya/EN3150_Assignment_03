## EN3150 Assignment 03 — Resource-Constrained CNN


---

## 1. Create the Repository

1. One team member creates a **new GitHub repo** (don't fork from a template with unrelated history).
   - Name: something like `en3150-a03-cnn-edge-classification`
   - Visibility: **Private** is safer for coursework (avoid plagiarism flags), then add teammates + optionally the instructor as collaborators.
   - Initialize with a `README.md`, `.gitignore` (Python template), and a `LICENSE` (optional, MIT is fine for coursework).
2. Add all group members as **collaborators**: Settings → Collaborators → add by GitHub username or email.
3. Enable **branch protection** on `main`:
   - Settings → Branches → Add rule for `main`
   - Require pull request before merging, require at least 1 approval.
   - This forces everyone to work on branches and merge via PRs, which naturally creates a clean, reviewable history.

```bash
git clone https://github.com/<your-org-or-user>/en3150-a03-cnn-edge-classification.git
cd en3150-a03-cnn-edge-classification
```

---

## 2. Recommended Folder Structure

Structure it like a small ML project, not a pile of notebooks:

```
en3150-a03-cnn-edge-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── environment.yml              # optional, if using conda
│
├── data/
│   ├── raw/                     # original downloaded dataset (gitignored if large)
│   ├── processed/               # resized 64x64, split train/val/test (gitignored if large)
│   └── README.md                # dataset source, license, download instructions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_a_standard_cnn.ipynb
│   ├── 03_model_b_lightweight_cnn.ipynb
│   ├── 04_optimizer_comparison.ipynb
│   └── 05_sota_finetuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py             # download, resize, split logic
│   ├── models/
│   │   ├── model_a.py           # standard CNN
│   │   ├── model_b.py           # depthwise separable CNN
│   │   └── sota_models.py       # MobileNetV2 / SqueezeNet wrappers
│   ├── train.py                 # training loop, shared by all models
│   ├── evaluate.py              # accuracy, confusion matrix, precision/recall
│   └── utils.py                 # plotting, param counting, model size on disk
│
├── configs/
│   ├── model_a.yaml
│   ├── model_b.yaml
│   └── sota.yaml
│
├── results/
│   ├── figures/                 # loss curves, confusion matrices (PNG)
│   ├── logs/                    # training logs, TensorBoard runs (gitignored if huge)
│   └── tables/                  # CSV of final comparison tables
│
├── checkpoints/                 # saved model weights (gitignored, too large for git)
│
├── report/
│   └── EN3150_Assignment_03_Report.pdf   # or .tex source if using LaTeX
│
└── scripts/
    └── run_all.sh                # convenience script to reproduce everything
```

Keep notebooks for exploration/experiments, but put reusable logic (model definitions, training loop, evaluation metrics) in `src/` as `.py` modules — makes the code "complete, well-documented, and ready to run" as required, and avoids messy notebook-only code that graders dislike.

---

## 3. `.gitignore` essentials

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/

# Data & models (too large for git)
data/raw/
data/processed/
checkpoints/
*.h5
*.pt
*.pth
results/logs/

# Jupyter
.ipynb_checkpoints/

# OS/editor
.DS_Store
.vscode/
.idea/
```

Large datasets and trained weights should **not** be committed. If graders/teammates need them, use a `data/README.md` with a download script, or share via Google Drive and link it.

---

## 4. Team Collaboration Workflow

### Branching model (keep it simple — feature branching)
- `main` — always working/stable code.
- One branch per person or per task, e.g.:
  - `feature/data-preprocessing`
  - `feature/model-a-standard-cnn`
  - `feature/model-b-depthwise`
  - `feature/optimizer-comparison`
  - `feature/sota-finetuning`
  - `feature/report-writing`

```bash
git checkout -b feature/model-a-standard-cnn
# do work, commit
git push -u origin feature/model-a-standard-cnn
```

### Pull Requests
- Open a PR into `main` when a task/sub-task is done.
- At least one teammate reviews and approves before merging (even a quick skim — this is what "regular commits + collaboration" should look like).
- Use PR descriptions to summarize what changed — this is useful documentation for your report too.

### Task tracking
- Use **GitHub Issues** or a **Project board** (Kanban: To Do / In Progress / Done) mapped to the assignment's numbered tasks (Data Prep, Model A, Model B, Optimizer comparison, Training/Eval, SOTA fine-tuning, Final comparison).
- Assign each issue to a person so there's a clear paper trail of who did what.

### Commit practices (this is what the instructor checks)
- **Commit early, commit often** — don't do one giant commit at the end. Aim for small, logical commits as you build things (e.g., "Add data resizing function", "Implement Model A architecture", "Add training loop with early stopping").
- Use clear messages, ideally following a convention:
  ```
  feat: add depthwise separable conv block for Model B
  fix: correct train/val/test split ratio
  docs: update README with dataset source
  exp: run Model B with Adam lr=1e-3
  ```
- Avoid single-word commits like "update" or "fix" repeated dozens of times — spread work across the assignment period rather than committing everything the night before, since the instructor is explicitly looking for "reasonable duration" of work.
- Each team member should have **commits under their own GitHub account** (not one person pushing on behalf of everyone) — this is how individual contribution is verified.

### Code review checklist before merging
- Does it run end-to-end without errors?
- Are functions/classes documented (docstrings)?
- Are magic numbers (kernel sizes, filters, learning rates) pulled into config files, not hardcoded everywhere?
- Are results (plots, metrics) saved to `results/`, not just printed?

---

## 5. README.md — what to include

```markdown
# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

## Team Members
- [Index No.] Name
- [Index No.] Name

## Dataset
- Source: <UCI dataset link>
- Classes: N
- Resolution: 64x64 (downscaled from original)
- Split: 70% train / 15% val / 15% test

## Setup
​```bash
pip install -r requirements.txt
​```

## How to run
​```bash
python src/data_prep.py        # prepares dataset
python src/train.py --config configs/model_a.yaml
python src/train.py --config configs/model_b.yaml
python src/evaluate.py --model checkpoints/model_a.pt
​```

## Repository Structure
(link to or summarize the folder structure above)

## Results
Summary table — see report/ for full details.
```

---

## 6. Practical best practices specific to this assignment

- **Dataset choice first, and register it in the shared sheet immediately** (mentioned in the PDF) — do this in week 1 so there's no overlap with other groups and everyone can start on real data.
- Since Model B must stay **under 100,000 parameters**, write a small utility in `src/utils.py` (e.g., `count_parameters(model)`) early, and check it after every architecture change — put this check in your training script so it fails loudly if you exceed the budget.
- Save every experiment's config + result together (e.g., `results/logs/model_b_adam_lr1e-3/`) so your optimizer comparison (SGD vs SGD+Momentum vs your chosen optimizer) is reproducible and traceable to a specific commit.
- Tag key milestones with **git tags** if useful, e.g., `git tag v1-data-ready`, `git tag v2-model-a-trained` — makes it easy to point to "this is the state of the repo when Model A was finalized."
- Keep the **report** (LaTeX or Word) in the repo too (even if you submit the PDF separately to Moodle), so the whole deliverable's history is in one place.

---

## 7. Quick start checklist

1. [ ] Create private repo, add collaborators, enable branch protection
2. [ ] Set up folder structure + `.gitignore` + `requirements.txt`
3. [ ] Register dataset choice in the shared sheet
4. [ ] Create GitHub Issues for each of the 6 numbered tasks in the assignment
5. [ ] Each member creates a feature branch for their assigned task
6. [ ] Commit incrementally, open PRs, review, merge into `main`
7. [ ] Track experiments/results in `results/`
8. [ ] Assemble final report from results + discussion
9. [ ] Submit code + report separately to Moodle with correct filename format
