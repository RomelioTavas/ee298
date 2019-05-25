# Acoustic Music Style Transfer using CycleGAN

- [Slides](https://docs.google.com/presentation/d/19dG5vqjuzJXPAvmT2QsxvO6G5LpGiHpg1ghIl1e3tUI/edit?usp=sharing)
- [Jupyter Notebook](CycleSpecGAN.ipynb)
- [CycleSpecGAN](https://github.com/RomelioTavas/ee298/blob/master/Project2/cyclespecgan.py)
- [SoundCloud](https://soundcloud.com/user-660812907/sets/cyclespecgan-outputs/s-mqn4c)

## Training
- Download IRMAS Training Data from https://www.upf.edu/web/mtg/irmas and put it in the root directory of the project.
- Preprocess audio samples with `preprocess_audio.py`. Example:
```
# For source
python preprocess_audio.py -i IRMAS-TrainingData/pia -o source.npz
# For target
python preprocess_audio.py -i IRMAS-TrainingData/gac -o target.npz
```
- Run `python train.py`

