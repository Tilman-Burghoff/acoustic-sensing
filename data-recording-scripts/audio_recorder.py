import pyaudio
import wave
from math import ceil

class AudioRecorder:
    FORMAT = pyaudio.paInt16
    CHUNK = 1024

    def __init__(self):
        self.setup_recording()
    
    def setup_recording(self):
        self.audio = pyaudio.PyAudio()
        print("Listing audio devices: ")
        print("index  name")
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"{i:>5}  {device_info["name"]}")
        
        if device_idx := input("Recording device index: "):
            self.device_idx = int(device_idx)
        else:
            self.device_idx = 0
        self.SR = int(self.audio.get_device_info_by_index(self.device_idx)["defaultSampleRate"])

        max_input_channels = self.audio.get_device_info_by_index(self.device_idx)["maxInputChannels"]
        if channels := input(f"Number of input channels (max {max_input_channels}): "):
            self.channels = max(1, min(int(channels), max_input_channels))
        else:
            self.channels = max_input_channels

        

    def create_recording(self, length, filename):
        stream = self.audio.open(
            format=self.FORMAT,
            channels=self.channels,
            rate=self.SR,
            output=False,
            input=True,
            input_device_index=self.device_idx,
            frames_per_buffer=self.CHUNK
        )

        number_of_frames = ceil(length * self.SR / self.CHUNK)
        frames = []
        for _ in range(number_of_frames):
            frames.append(stream.read(self.CHUNK, False))
        stream.close()

        output_file = wave.open(filename, 'wb')
        output_file.setnchannels(self.channels)
        output_file.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        output_file.setframerate(self.SR)
        output_file.writeframes(b''.join(frames))
        output_file.close()
        return number_of_frames * self.CHUNK