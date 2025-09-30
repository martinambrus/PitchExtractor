# JDC-PitchExtractor
This repo contains the training code for deep neural pitch extractor for Voice Conversion (VC) and TTS used in [StarGANv2-VC](https://github.com/yl4579/StarGANv2-VC) and [StyleTTS](https://github.com/yl4579/StyleTTS). This is the F0 network in StarGANv2-VC and pitch extractor in StyleTTS. 

## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/yl4579/PitchExtractor.git
cd PitchExtractor
```
3. Install python requirements: 
```bash
pip install SoundFile torchaudio torch pyyaml click matplotlib librosa pyworld
```
4. Prepare your own dataset and put the `train_list.txt` and `val_list.txt` in the `Data` folder (see Training section for more details).

## Training
```bash
python train.py --config_path ./Configs/config.yml
```
Please specify the training and validation data in `config.yml` file. The data list format needs to be `filename.wav|anything`, see [train_list.txt](https://github.com/yl4579/StarGANv2-VC/blob/main/Data/train_list.txt) as an example (a subset of VCTK). Note that you can put anything after the filename because the training labels are generated ad-hoc.

Checkpoints and Tensorboard logs will be saved at `log_dir`. To speed up training, you may want to make `batch_size` as large as your GPU RAM can take. 

### IMPORTANT: DATA FOLDER NEEDS WRITE PERMISSION
Since both `harvest` and `dio` are relatively slow, we do have to save the computed F0 ground truth for later use. In [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py#L77-L89), it will write the computed F0 curve `_f0.npy` for each `.wav` file. This requires write permission in your data folder. 

### F0 Computation Details
`meldataset.MelDataset` now supports a cascade of runtime-selectable F0 backends. The default configuration uses PyWorld's `harvest` followed by `dio`, mirroring the original behaviour, but you can enable neural or classical trackers such as CREPE, RMVPE, Praat/Parselmouth, and REAPER by editing `dataset_params.f0_params` in [Configs/config.yml](Configs/config.yml). Backends are evaluated in the order defined by `backend_order`, and each backend may be toggled on/off or customised individually (for example, to adjust CREPE's model size or Praat's pitch range).

Whenever the backend configuration changes the dataset automatically regenerates cached pitch files under backend-specific filenames and stores a small JSON metadata file alongside each cache to keep track of the extractor that produced it. Failed extraction attempts fall back to the next enabled backend until a valid contour with sufficient voiced frames is produced. If all backends fail the sample is logged and the stored F0 is left empty (zeros after post-processing), so you may want to audit those cases if they occur frequently.

#### Backend configuration summary

- **PyWorld (harvest/dio/stonemask)** – Controlled by the `algorithm`, optional `fallback` algorithm, and `stonemask` refinement flag.
- **CREPE** – Choose `model` (`tiny`, `small`, `medium`, `full`), override `step_size_ms`, toggle `viterbi` smoothing, set a `confidence_threshold` to zero low-confidence frames, and control TensorFlow placement with `device: auto` (default heuristic), `cpu` (always run on CPU), or `gpu` (require CUDA and surface initialisation errors).
- **RMVPE** – Provide `model_path` for custom checkpoints, select the execution `device`, and enable `is_half` to use the half-precision weights when running on CUDA.
- **Praat / Parselmouth** – Set the `method` (e.g., `ac`, `cc`), `min_pitch`/`max_pitch` bounds, and adjust `silence_threshold` and `voicing_threshold` for sensitivity.
- **REAPER** – Limit the search range with `min_pitch`/`max_pitch` and toggle `do_high_pass` to suppress low-frequency noise before analysis.

### Data Augmentation
Data augmentation is not included in this code. For better voice conversion results, please add your own data augmentation in [meldataset.py](https://github.com/yl4579/PitchExtractor/blob/main/meldataset.py) with [audiomentations](https://github.com/iver56/audiomentations).

## References
- [keums/melodyExtraction_JDC](https://github.com/keums/melodyExtraction_JDC)
- [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
