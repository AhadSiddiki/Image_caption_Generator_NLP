import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch


class ImageCaptionGenerator:
    def __init__(self, model_name="nlpconnect/vit-gpt2-image-captioning"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Explicitly set padding token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set generation parameters
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "pad_token_id": self.tokenizer.pad_token_id  # Ensure padding is handled
        }

    @staticmethod
    def load_image(image_path):
        """Load and preprocess image for the model"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        return i_image

    def predict_caption(self, image):
        """Generate caption for an image"""
        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # Generate output with attention_mask explicitly provided
        output_ids = self.model.generate(
            pixel_values,
            attention_mask=torch.ones_like(pixel_values, dtype=torch.long, device=self.device),
            # Ensures padding tokens are masked
            **self.gen_kwargs
        )

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()

    def caption_image(self, image_path, show_image=True):
        """Load an image, generate caption, and optionally display the image"""
        image = self.load_image(image_path)
        caption = self.predict_caption(image)
        
        if show_image:
            plt.imshow(np.array(image))
            plt.axis("off")
            plt.title(caption)
            plt.show()
        
        return caption

def main():
    parser = argparse.ArgumentParser(description='Generate captions for images using NLP')
    parser.add_argument('--image', type=str, help='Path to the image file')
    parser.add_argument('--dir', type=str, help='Directory containing images to caption')
    parser.add_argument('--no-display', action='store_true', help='Do not display images')
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("Please provide either an image path or a directory of images")
    
    caption_generator = ImageCaptionGenerator()
    
    if args.image:
        caption = caption_generator.caption_image(args.image, not args.no_display)
        print(f"Caption: {caption}")
    
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} is not a valid directory")
            return
        
        image_extensions = ['.jpg', '.jpeg', '.png']
        for filename in os.listdir(args.dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(args.dir, filename)
                print(f"Processing {filename}...")
                caption = caption_generator.caption_image(image_path, not args.no_display)
                print(f"Caption: {caption}")
                print("-" * 50)

if __name__ == "__main__":
    main()
