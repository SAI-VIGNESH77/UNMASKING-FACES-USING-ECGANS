import os
import logging
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


class ESRGANModel:
    """
    A class to load and use a pretrained ESRGAN model for image enhancement.
    """

    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize the ESRGAN model.

        Args:
            model_path (str): Path to the pretrained ESRGAN model.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")

        try:
            self.model = torch.load(model_path, map_location=device)
            self.model.eval()
            self.device = device
            logger.info(f"ESRGAN model loaded successfully on device: {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load ESRGAN model: {e}")

    def enhance(self, image: Image.Image) -> Image.Image:
        """
        Enhance an image using the ESRGAN model.

        Args:
            image (PIL.Image): Input image to enhance.

        Returns:
            PIL.Image: Enhanced image.
        """
        try:
            # Convert image to tensor and move to the appropriate device
            image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)

            # Perform inference
            with torch.no_grad():
                enhanced_tensor = self.model(image_tensor)

            # Post-process the output tensor
            enhanced_tensor = enhanced_tensor.squeeze(0).clamp(0, 1).cpu().detach()
            enhanced_image = ToPILImage()(enhanced_tensor)

            return enhanced_image
        except Exception as e:
            logger.error(f"Error during image enhancement: {e}")
            raise


def enhance_images(input_dir: str, output_dir: str, model_path: str, device: torch.device):
    """
    Enhance all images in a directory using the ESRGAN model.

    Args:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save enhanced images.
        model_path (str): Path to the pretrained ESRGAN model.
        device (torch.device): Device to run the model on.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the ESRGAN model
    try:
        model = ESRGANModel(model_path, device)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # Get list of image files in the input directory
    image_files = [
        f for f in os.listdir(input_dir)
        if f.lower().endswith(SUPPORTED_IMAGE_FORMATS)
    ]

    if not image_files:
        logger.warning(f"No supported images found in '{input_dir}'.")
        return

    # Process each image
    for file_name in tqdm(image_files, desc="Enhancing images", unit="image"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            # Load and enhance the image
            image = Image.open(input_path).convert("RGB")
            enhanced_image = model.enhance(image)

            # Save the enhanced image
            enhanced_image.save(output_path)
            logger.info(f"Enhanced image saved to '{output_path}'")
        except Exception as e:
            logger.error(f"Failed to process '{input_path}': {e}")


def main():
    """Main function to parse arguments and run the image enhancement process."""
    parser = ArgumentParser(description="Enhance images using a pretrained ESRGAN model.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save enhanced images.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained ESRGAN model.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    args = parser.parse_args()

    # Set device (GPU or CPU)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available. Using CPU instead.")

    # Enhance images
    enhance_images(args.input_dir, args.output_dir, args.model_path, device)


if __name__ == "__main__":
    main()