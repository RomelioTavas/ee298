from cyclespecgan import CycleSpecGAN


if __name__ == '__main__':

    cycleGAN = CycleSpecGAN()
    cycleGAN.train(200, batch_size=1)