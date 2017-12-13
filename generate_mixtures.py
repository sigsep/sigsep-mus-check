import soundfile as sf
import numpy as np
import os
import argparse

stems = [
    "vocals",
    "drums",
    "bass",
    "other"
]

def check_length(stems):
    durations = []
    for name, tensor in stems.items():
        durations.append(tensor.shape[0])

    a = np.array(durations)
    return (a == a[0]).all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse RAW Dataset + checks')
    parser.add_argument(
        'subset',
        help='train/test'
    )

    args = parser.parse_args()

    # Load folders
    for _, track_folders, _ in os.walk(args.subset):
            for track_name in sorted(track_folders):
                print(track_name)

                track_path = os.path.join(
                    os.path.join(args.subset, track_name),
                )

                stems_audio = {}
                for stem_name in stems:
                    audio, rate = sf.read(
                        os.path.join(track_path, stem_name + ".wav")
                    )
                    stems_audio[stem_name] = audio

                assert check_length(stems_audio)

                sources = []
                for name, value in stems_audio.items():
                    sources.append(value)

                sources = np.array(sources)
                mixture = np.sum(sources, axis=0)
                max_amp = np.max(np.abs(mixture))

                print(max_amp)

                # write out mixture
                sf.write(
                    os.path.join(track_path, "mixture.wav"),
                    mixture / max_amp,
                    rate
                )

                # write normalized stems
                for name, audio in stems_audio.items():
                    sf.write(
                        os.path.join(track_path, name + ".wav"),
                        audio / max_amp,
                        rate
                    )
