
import sys
from pathlib import Path
from PIL import Image

# Make the project root importable
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from inference import predict_image

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: python scripts/test_inference.py "
        "REAL_IMAGE AI_IMAGE"
    )

for path_string in sys.argv[1:]:
    path = Path(path_string)

    with Image.open(path) as image:
        result = predict_image(image)

    print("\nImage:", path.name)
    print("Prediction:", result["label"])
    print("Confidence:", result["confidence"], "%")
    print("AI probability:", result["ai_probability"], "%")
    print("Model:", result["model_name"])
