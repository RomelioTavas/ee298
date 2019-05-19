import librosa

from custom.dataloader import DataLoader
from cyclegan import CycleGAN
from config import Config


if __name__ == '__main__':
    
    config = Config(sampling_rate=16000, audio_duration=2)
    data_loader = DataLoader(config)

    cycleGAN = CycleGAN(config, data_loader)

    source_path = "IRMAS-TrainingData/voi"
    input_sample = data_loader.load(source_path + "/[voi][jaz_blu]2491__1.wav")

    g_AB = cycleGAN.build_generator()
    g_AB.load_weights("checkpoints/in_16khz/cyclegan_A2B_epoch_100.h5")
    output_sample = g_AB.predict(input_sample)

    librosa.output.write_wav("output.wav", output_sample[0], config.sampling_rate)
    print("Written output.wav")
