from long_caption import LongCaptioner
from short_tag import ShortTagger
import os
from tqdm import tqdm
from PIL import Image

class Milabol:
    def __init__(self, blip_path, wd_path, device="cuda", *args, **kwargs):
        self.lc = LongCaptioner(blip_path, device, *args, **kwargs)
        self.st = ShortTagger(wd_path, *args, **kwargs)

    def predict(self, image, *args, **kwargs):
        caption = self.lc.predict(image, *args, **kwargs)
        tags = self.st.predict(image, *args, **kwargs)

        return caption, tags
    
    def run(self, img_folder_path, triggers="", *args, **kwargs):
        triggers = triggers.strip()
        if len(triggers) and triggers[-1] == ',':
            triggers = triggers[:-1]

        all_files = os.listdir(img_folder_path)
        image_files = [file for file in all_files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_name in tqdm(image_files):
            image_path = os.path.join(img_folder_path, image_name)
            image = Image.open(image_path)
            caption, tags = self.predict(image, *args, **kwargs)

            with open(f'{os.path.splitext(image_path)[0]}.txt','w') as f:
                f.write(f'{triggers}, {caption}, {tags}' if triggers else f'{caption}, {tags}')
            