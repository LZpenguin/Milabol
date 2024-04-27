from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration


class LongCaptioner:
    def __init__(self, model_name_or_path, device="cuda", version=1, *args, **kwargs):
        if version == 1:
            self.processor = BlipProcessor.from_pretrained(model_name_or_path)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        elif version == 2:
            self.processor = Blip2Processor.from_pretrained(model_name_or_path)
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        else:
            raise ValueError(f'Invalid Blip version: {version}')

    def predict(self, image: Image, max_length=100, max_sentence=2, *args, **kwargs):
        image = image.convert('RGB')
        inputs = self.processor(image, return_tensors="pt").to(self.model.device)
        pixel_values = inputs.pixel_values
        out = self.model.generate(pixel_values=pixel_values, max_length=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)

        caption = caption.replace('.',',').strip()
        if caption[-1] == '.':
            caption = caption[:-1]
        caption = ','.join(caption.split(',')[:max_sentence])

        return caption