from data_gen.tts.base_preprocess import BasePreprocessor


class VCTKPreprocess(BasePreprocessor):
    def meta_data(self):
        for l in open(f'{self.raw_data_dir}/metadata.csv').readlines():
            item_name, txt, spk = l.strip().split("|")
            item_name = item_name.replace(".wav", "")
            items = item_name.split("/")
            item_name = items[-1]
            spk_name = items[-2]
            wav_fn = f"{self.raw_data_dir}/wav48/{spk_name}/{item_name}.wav"

            yield {'item_name': item_name, 'wav_fn': wav_fn, 'txt': txt, 'spk_name': spk}
