import numpy as np
import librosa
from glob import glob


class DataLoader():
    def __init__(self, config):
        self.config = config
        self.dataset_name = 'IRMAS-TrainingData'

    def normalize_audio(self, data):
        max_data = np.max(data)
        min_data = np.min(data)
        data = (data-min_data)/(max_data-min_data+1e-6)
        return data-0.5

    def clip_audio(self, data):
        input_length = self.config.audio_length
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        return data

    def load_batch(self, batch_size=1):
        path_A = glob('./{}/voi/*.wav'.format(self.dataset_name))
        path_B = glob('./{}/gac/*.wav'.format(self.dataset_name))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        input_length = self.config.audio_length

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            audios_A = np.empty((batch_size, input_length, 1))
            audios_B = np.empty((batch_size, input_length, 1))

            for i, (audio_A, audio_B) in enumerate(zip(batch_A, batch_B)):
                audio_A, _ = librosa.core.load(audio_A, sr=self.config.sampling_rate,
                                               res_type='kaiser_fast')
                audio_B, _ = librosa.core.load(audio_B, sr=self.config.sampling_rate,
                                               res_type='kaiser_fast')

                audio_A = self.clip_audio(audio_A)
                audio_B = self.clip_audio(audio_B)

                audio_A = self.normalize_audio(audio_A)[:, np.newaxis]
                audio_B = self.normalize_audio(audio_B)[:, np.newaxis]

                audios_A[i, ] = audio_A
                audios_B[i, ] = audio_B

            yield audios_A, audios_B
