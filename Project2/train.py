from custom.dataloader import DataLoader
from cyclegan import CycleGAN
from config import Config


if __name__ == '__main__':

    config = Config(sampling_rate=8000, audio_duration=2)
    data_loader = DataLoader(config)

    cycleGAN = CycleGAN(config, data_loader)
    cycleGAN.build_model()
    cycleGAN.train(200, batch_size=1)
