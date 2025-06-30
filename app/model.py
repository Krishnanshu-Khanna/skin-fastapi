import torch
import torch.nn as nn
from torchvision import models, transforms

class_names = [
    "Healthy", "Psoriasis", "Vitiligo", "Melasma", "Impetigo", "Scabies",
    "Coldsores", "Ringworm", "Cellulitis"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(weight_path: str):
    model = models.convnext_base(pretrained=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
