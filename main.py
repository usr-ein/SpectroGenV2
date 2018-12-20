import wave             # For WAVE file format
import struct           # For WAVE file format
import math             # For cosine
import time             # For timestamps
import numpy as np      # For manipulating matrix
from PIL import Image   # For loading images
import scipy.ndimage    # For resampling
from tqdm import tqdm   # For progress bar
from pydub import AudioSegment  # For mixing two soundtracks


def load_image(file):
    img = Image.open(file)
    img = img.convert('L')

    img_arr = np.array(img)
    img_arr = np.flip(img_arr, axis=0)

    img_arr -= np.min(img_arr)
    img_arr = img_arr/np.max(img_arr)

    return img_arr


def resize_image(img_arr, size):
    if not size[0]:
        size[0] = img_arr.shape[0]
    if not size[1]:
        size[1] = img_arr.shape[1]

    resampling_factor = size[0]/img_arr.shape[0], size[1]/img_arr.shape[1]

    # Order : 0=nearestNeighbour, 1:bilinear, 2:cubic etc...
    img_arr = scipy.ndimage.zoom(img_arr, resampling_factor, order=0)

    return img_arr


def preprocess_image(img_arr):
    # Inverse filter
    # img_arr = 1 - img_arr
    return img_arr


def generate_soundwave(file, output='sound.wav', duration=2.5, sample_rate=44100.0, min_freq=0, max_freq=22000):
    waveform = wave.open(output, 'w')
    waveform.setnchannels(1)  # mono
    waveform.setsampwidth(2)
    waveform.setframerate(sample_rate)

    total_frame_count = int(duration * sample_rate)
    max_intensity = 32767  # Defined by WAV

    step_size = 100  # Works well in most cases
    substep_size = 250  # Works well in most cases

    freq_range = max_freq - min_freq
    stepping_spectrum = int(freq_range / step_size)

    img_arr = load_image(file)
    img_arr = preprocess_image(img_arr)
    img_arr = resize_image(img_arr, size=(stepping_spectrum, total_frame_count))

    img_arr *= max_intensity

    for frame in tqdm(range(total_frame_count)):
        signal_val, count = 0, 0

        for step in range(stepping_spectrum):
            intensity = img_arr[step, frame]

            current_freq = (step * step_size) + min_freq
            next_freq = ((step+1) * step_size) + min_freq

            if next_freq - min_freq > max_freq:  # End of the spectrum
                next_freq = max_freq

            for freq in range(current_freq, next_freq, substep_size):
                signal_val += intensity * math.cos(2 * math.pi * freq * frame/sample_rate)
                count += 1

        if count == 0:
            count = 1
        signal_val /= count

        data = struct.pack('<h', int(signal_val))
        waveform.writeframesraw(data)

    waveform.writeframes(''.encode())
    waveform.close()


def mix_soundtracks(file1, file2, output_file, start=0):
    sound1 = AudioSegment.from_file(file1)
    sound2 = AudioSegment.from_file(file2)

    # mix sound2 with sound1, starting at (start)ms into sound1)
    output = sound1.overlay(sound2, position=start)

    # save the result
    output.export(output_file, format=output_file.split('.')[-1])


def main():
    # Input files
    file = 'data/image.png'
    music = 'data/music.mp3'

    # Output files
    img_sound_output = 'outputs/img_sound{}.wav'.format(str(int(time.time()))[-5:])
    final_output = 'outputs/final_sound{}.wav'.format(str(int(time.time()))[-5:])

    generate_soundwave(file,
                       output=img_sound_output,
                       duration=10.5,
                       sample_rate=44100.0,
                       min_freq=16000,
                       max_freq=22000)

    print("Image to waveform ended successfully")
    mix_soundtracks(music, img_sound_output, final_output)
    print("Mixing ended successfully")
    print("Done.")


if __name__ == "__main__":
    main()
