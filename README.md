# Sonia
Generation of music using autoencoders.

This repository contains all follow-up materials, such as .ipynb labs, Kaggle competitions submission notebooks, and other stuff that gives enough foundation for the project.

The project itself with everything required for its work lies in `Project` folder.

---

Quick note on usage of the project: \
Make sure that midi2audio and fluidsynth are installed (`pip install midi2audio && sudo apt install fluidsynth`) \
To download model checkpoints and samples file you will need git-lfs (check out https://git-lfs.github.com)

1. python3 inference.py
2. Enter random seed, "density rate" (float number from 0 to 1, the lower - the denser, 0.25 is optimal for most cases), output directory and path to the pretrained model (mine are stored in `States` folder).
3. Generated music (.mid and .wav) would lie in the provided folder.
