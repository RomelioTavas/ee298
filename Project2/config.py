class Config(object):
    def __init__(self,
                 sampling_rate=16000, audio_duration=2):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.audio_length = self.sampling_rate * self.audio_duration

