# coding: utf-8

import librosa
import numpy as np
from glob import glob

import joblib
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="preprocess audio and make .npz for training")
parser.add_argument("--output_file", "-o", required=False, default='training_data.npz', help="filepath of output .npz file")
parser.add_argument("--input_dir", "-i", required=True, help="root directory containing containing audio files")
args = parser.parse_args()

filepaths = glob(os.path.join(args.input_dir,"*"))

print("# of files:", len(filepaths))

if len(filepaths) == 0:
    print("Error: You need to specify a root directory containing containing audio files")
    exit()

from utils import audio_tools as audio
from utils.hparams import hparams
SR = hparams.sample_rate

IMAGE_SIZE = 128 # Generate melspectrogram image in 128 x 128



# PREPROCESS
# calculate means/standard deviation for normalization

def load_audio(path):
    try:
        y, sr = librosa.core.load(path, sr = SR)

        # sfft -> mel conversion
        db_mel = audio.melspectrogram(y)

        # means/standard deviation for each freq bin
        m_db_mel = np.mean(db_mel, axis=1)
        std_db_mel = np.std(db_mel, axis=1)
        return m_db_mel, std_db_mel
    except:
        return None, None

print("Loading all audio files and calculate the mean and the standard deviation. it may take a while...")
# parallel loading
results = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(load_audio)(path) for path in filepaths)

means = [mean for mean, _ in results if mean is not None]
stds = [std for _, std in results if std is not None]

# calcurate overall means/standard deviation
means = np.array(means)
stds = np.array(stds)

mel_means = np.mean(means, axis=0)
mel_stds = np.mean(stds, axis=0)


# scaling - based on
# Donahue, C., McAuley, J., & Puckette, M. (2018). Synthesizing Audio with Generative Adversarial Networks.
# Retrieved from http://arxiv.org/abs/1802.04208
def normalize(s):
    assert s.shape[0] == mel_means.shape[0]
    norm_Y = (s - mel_means) / (3.0 * mel_stds)
    return np.clip(norm_Y, -1.0, 1.0)

def denormalize(norm_s):
    assert norm_s.shape[0] == mel_means.shape[0]
    Y = (norm_s * (3.0 * mel_stds)) + mel_means
    return Y

def load_melspecs(path):
    filename = os.path.basename(path)

    try:
        y, sr = librosa.core.load(path, sr = SR)
    except:
        return None, None

    db_mel = audio.melspectrogram(y)
    assert  db_mel.shape[0] == IMAGE_SIZE

    dummy = np.ones((IMAGE_SIZE, IMAGE_SIZE)) * hparams.min_level_db
    db_mel = np.hstack((db_mel, dummy))
    db_mel = db_mel[:, :IMAGE_SIZE]

    norm_mel = normalize(db_mel)

    return norm_mel

# parallel loading
print("Generating normalized spectrogram images... ")
results = joblib.Parallel(n_jobs=-1)(joblib.delayed(load_melspecs)(path) for path in filepaths)

specs = [spec for spec in results if spec is not None]

# save all
print("Saving data to: ", args.output_file)
np.savez(args.output_file, specs=specs,
         mean=mel_means, std=mel_stds)
print("Done!")