import librosa

from custom.dataloader import DataLoader
from cyclegan import CycleGAN
from config import Config


if __name__ == '__main__':
    
    config = Config(sampling_rate=16000, audio_duration=2)
    data_loader = DataLoader(config)

    cycleGAN = CycleGAN(config, data_loader)

    g_AB = cycleGAN.build_generator()
    g_AB.load_weights("checkpoints/in_16khz/cyclegan_A2B_epoch_100.h5")

    for i, (audios_A, _, audio_A_path, _) in enumerate(data_loader.load_batch2(1)):

        output_sample = g_AB.predict(audios_A)
        output_filename = audio_A_path.split('/')[-1]
        output_path = "generated/converted-{}".format(output_filename)
        librosa.output.write_wav(output_path, output_sample[0], config.sampling_rate)
        print("Written {}".format(output_path))
