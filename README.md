# ConvNeXt entry — neonatal EEG grading competition
 
This repository holds the ConvNeXt submission to the machine-learning competition described in:
 
> Magarelli F, Boylan GB, Montazeri S, O'Sullivan F, Lightbody D, Ashoori M, Skoric T,
> O'Toole JM. *Machine-learning competition to grade EEG background patterns in newborns with
> hypoxic-ischaemic encephalopathy.* PLOS Digital Health (under review).
 
It contains the inference script, the helper used to read the competition data, and the trained
model checkpoint, as submitted.
 
## Authorship and hosting
 
**The code and the model in this repository were written and trained by Feargal O'Sullivan and
Dominic Lightbody** (Embedded Systems, University College Cork), who entered the competition as
participants. They hold the copyright, and the repository is released under the MIT licence they
chose (see `LICENSE.txt`).
 
The repository sits under the `fabiom91` GitHub account at the authors' request, so that the four
evaluated entries could be linked from one place in the paper. **Fabio Magarelli, who ran the
competition and hosts this repository, did not write this code and did not develop this model.**
His only commit is the initial import of the files as the authors supplied them; nothing in the
method was changed. Questions about the method should go to its authors.
 
This arrangement was not a route around the competition rules. The submission was scored by the
platform in the same way as every other entry, against labels held server-side, and this
repository was created after the competition had closed.
 
## Provenance of the results
 
The submission scored a weighted Matthews correlation coefficient of 0.677 on the public
leaderboard and 0.351 on the held-out validation dataset (Tables 4 and 5 of the paper). The
public leaderboard was produced by the competition platform:
<https://github.com/fabiom91/AI_Competition_Platform> (archived release v2.0.1,
DOI [10.5281/zenodo.22735014](https://doi.org/10.5281/zenodo.22735014)). The validation figure
comes from an offline evaluation run by the competition host after the competition closed, using
the inference script in this repository and the checkpoint in `model/`.
 
The competition data are published separately:
 
> O'Toole JM et al. *Neonatal EEG graded for severity of background abnormalities in
> hypoxic-ischaemic encephalopathy.* Scientific Data 10, 129 (2023).
> <https://doi.org/10.1038/s41597-023-02002-8>
 
The held-out validation dataset is not public: it is governed by the data-sharing agreements of
the ANSeR study.
 
## Method, in brief
 
Each EEG channel is reduced to its root mean square, segmented into windows of 384 samples and
encoded as a Gramian Angular Summation Field, giving a 384 × 384 × 3 image from the bipolar
channels F4-C4, F3-C3 and C4-T4. The images are resized to 224 × 224 × 3 for
ConvNeXt-224 (tiny), which was trained from scratch on the competition training set: the
pre-trained ImageNet weights were dropped. Full preprocessing settings are in Table 6 of the
paper.
 
## Contents
 
| Path | What it is |
|---|---|
| `generateSubmission.py` | Builds the image dataset from a set of EEG recordings, runs inference and writes a predictions CSV. |
| `get_data.py` | Helper called by `generateSubmission.py` to read and filter the EEG files. |
| `model/` | The trained checkpoint, in Hugging Face `transformers` layout. Point the script at the **folder**, not at a file inside it. |
| `requirements.txt` | The package versions the submission was run with. |
| `NAME_YOUR_DATASET/` | Placeholder for the dataset the script generates. |
 
## Model weights
 
`model/pytorch_model.bin` (about 114 MB) and `model/optimizer.pt` are stored with
[Git LFS](https://git-lfs.com). A plain `git clone` gives you pointer files of a few hundred
bytes, and inference will fail. Fetch the real files with:
 
```bash
git lfs install
git clone https://github.com/fabiom91/ConvNeXt_AI_Competition_Platform.git
cd ConvNeXt_AI_Competition_Platform
git lfs pull            # only needed if the clone predates git lfs install
ls -l model/pytorch_model.bin   # should be ~114 MB, not ~130 bytes
```
 
## Running inference
 
1. Install the dependencies: `pip install -r requirements.txt`. The pinned versions are the ones
   the submission ran with; `torch==1.10.0` and `numpy==1.19.5` need an older Python (3.9 or
   below), so use a virtual environment of that version, or relax the pins and accept that the
   library versions then differ from the ones used for the published result.
2. Open `generateSubmission.py` and set the five paths at the top:
   - `annotations_file` — the annotations CSV for the set you are scoring, in the format the
     competition supplied.
   - `path` — the folder holding the EEG recordings in CSV form. Individual `.csv.xz` files do
     not need to be extracted; the script handles that.
   - `model_checkpoint` — the path to the `model` folder.
   - `datasetName` — a name for the image dataset the script will build.
   - `predictions_csv` — a name for the predictions file.
3. Run it. The predictions are written to `copy.<predictions_csv>`, in the row order of the
   annotations file, so the column can be pasted straight into the annotations CSV.
The script writes an image dataset to disk before running inference; allow for the space and the
time that takes on a full recording set.
 
## Status
 
This is competition code, published so that the entry can be inspected and re-run. It is not
maintained, and it has not been validated for any clinical use.
 
