from PIL import Image
import io
import torch
import torch.nn.functional as F
from .model import transform, class_names, device

def predict_image(model, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    top_index = probs.argmax()
    return {
        "diagnosis": class_names[top_index],
        "confidence": f"{probs[top_index]*100:.2f}%",
        "probabilities": {
            class_names[i]: f"{p*100:.2f}%" for i, p in enumerate(probs)
        }
    }
