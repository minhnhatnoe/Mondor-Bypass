import torch
import numpy as np
from ...build.model import models
from ... import utils

model = models.BasicCNN()
# assert torch.cuda.is_available()

device = torch.device(type="cpu")

model.load_state_dict(torch.load("blob/CNN.pt", map_location=torch.device("cpu")))
model = model.to(device)

def inference(image: bytes) -> str:
    result = []
    for img, _ in utils.split_image(image, "AAAAAA"):
        img = np.expand_dims(img, 0)
        img = torch.tensor(img).to(device, dtype=torch.float)
        idx = int(model(img).argmax(dim=1).cpu())
        result.append(utils.decode_result(idx))
    return "".join(reversed(result))    
