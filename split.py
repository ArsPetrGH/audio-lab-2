import os
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio, Dataset, DatasetDict
from huggingface_hub import login
from tqdm import tqdm

# 1. Авторизація (твій токен)
login(token="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")

# 2. Перенесення всього кешу на диск D:
cache_path = "D:/huggingface_cache"
os.makedirs(cache_path, exist_ok=True)

os.environ["HF_HOME"] = cache_path
os.environ["HF_DATASETS_CACHE"] = cache_path
# Переносимо навіть тимчасові файли Windows для цього процесу
os.environ["TMPDIR"] = cache_path
os.environ["TEMP"] = cache_path
os.environ["TMP"] = cache_path


def gen_segments(raw_data, segment_len=3.0, step=1.5):
    for x in tqdm(raw_data):
        # Декодуємо аудіо
        audio_bytes = x['audio']['bytes']
        with io.BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f, dtype='float32')

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        duration = len(waveform) / sr

        for start_t in np.arange(0, duration - segment_len, step):
            end_t = start_t + segment_len

            chunk_labels = []
            max_speech_dur = 0
            main_spk = None

            for spk, s, e in zip(x['speakers'], x['timestamps_start'], x['timestamps_end']):
                overlap_s = max(start_t, s)
                overlap_e = min(end_t, e)

                if overlap_e > overlap_s:
                    dur = overlap_e - overlap_s
                    chunk_labels.append({
                        'spk': spk,
                        'start': float(overlap_s - start_t),
                        'end': float(overlap_e - start_t)
                    })
                    if dur > max_speech_dur:
                        max_speech_dur = dur
                        main_spk = spk

            if max_speech_dur < 0.3:
                continue

            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)

            yield {
                "audio": waveform[start_sample:end_sample].astype(np.float32),
                "main_speaker": main_spk,
                "vad_labels": chunk_labels
            }


if __name__ == "__main__":
    # Папка для готового результату
    output_dir = "D:/Oleksii_Chyzhov/AudioML/ami_segmented"

    print(f"🚀 Starting preprocessing. All cache will be on {cache_path}")

    raw_ds = load_dataset("diarizers-community/ami", "ihm")
    raw_ds = raw_ds.cast_column("audio", Audio(decode=False))

    new_splits = {}
    for split in ['train', 'validation', 'test']:
        print(f"📦 Processing {split}...")
        new_splits[split] = Dataset.from_generator(
            gen_segments,
            gen_kwargs={"raw_data": raw_ds[split]},
            cache_dir=cache_path  # Примусовий кеш для генератора
        )

    final_ds = DatasetDict(new_splits)
    final_ds.save_to_disk(output_dir)

    print(f"✅ DONE! Dataset saved to: {output_dir}")