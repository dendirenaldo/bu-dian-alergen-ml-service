import cv2
import numpy as np


class ImagePreprocessor:
    """Load and preprocess images for OCR pipeline."""

    def preprocess(self, image_path: str) -> np.ndarray:
        """Load image from path and preprocess for OCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Preprocessed binary image.

        Raises:
            ValueError: If image cannot be loaded.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self._process(img)

    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess from raw bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Preprocessed binary image.

        Raises:
            ValueError: If image cannot be decoded.
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return self._process(img)

    def _process(self, img: np.ndarray) -> np.ndarray:
        """Core preprocessing pipeline.

        Steps: grayscale -> bilateral filter -> Otsu threshold ->
        normalize foreground/background -> morphological close.

        Args:
            img: BGR image array.

        Returns:
            Preprocessed binary image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 75, 75)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Normalize foreground/background so OCR is more stable
        if np.mean(thresh) < 127:
            thresh = 255 - thresh

        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return processed
