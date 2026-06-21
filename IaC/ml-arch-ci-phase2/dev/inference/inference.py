import os
import sys
import logging
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

JPEG_CONTENT_TYPE = 'image/jpeg'
PNG_CONTENT_TYPE = 'image/png'
def net():
    '''
          Initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_ftrs, 133))
    return model

def model_fn(model_dir):
    '''
          loads the model for inference
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    
    with open(os.path.join(model_dir, "model_test.pth"), "rb") as f:
        logger.info("Loading the dog-breed-classifier trained model")
        model.load_state_dict(torch.load(f, map_location=device))
        logger.info('model loaded!!!')
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    """Preprocesa la imagen entrante (Bytes -> Tensor)."""
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE or content_type == PNG_CONTENT_TYPE:
        img = Image.open(io.BytesIO(request_body)).convert("RGB")
        
        # Las mismas transformaciones que usaste en Validation/Test
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0) # Añade dimensión de batch (1, 3, 224, 224)
    raise Exception(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Ejecuta la inferencia."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_data = input_data.to(device)
    with torch.no_grad():
        logger.info("Running inference...")
        return model(input_data) # return logits predicted by the model