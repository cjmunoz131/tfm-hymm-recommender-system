#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import os
import logging
import sys
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
import smdebug.pytorch as smd
import time

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(model, test_loader, criterion, device):
    '''
    this function can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()

    running_corrects = 0
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, preds = torch.max(output, 1)
            running_loss += loss.item() * data.size(0)
            running_corrects += torch.sum(preds == target.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_accuracy = running_corrects / len(test_loader.dataset)
    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_accuracy
        )
    )
    return total_loss

def train(model, train_loader, val_loader, criterion, optimizer, epochs, model_dir, device, hook=None):
    '''
          this function can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss = float('inf')
    image_dataset={'train':train_loader, 'valid':val_loader}
    loss_counter=0
    

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            logger.info(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                hook.set_mode(smd.modes.EVAL)
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0
            for batch_idx, (data, target) in enumerate(image_dataset[phase]):
                data, target = data.to(device), target.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(preds == target.data).item()
                running_samples += len(data)
                if running_samples % 200  == 0:
                    accuracy = running_corrects/running_samples
                    logger.info("{} epoch: {}  [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            phase,
                            epoch,
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            # "Early Stopping" y Guardado del Mejor Modelo
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    # Guardamos el mejor modelo encontrado hasta ahora
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"Mejor modelo guardado con Loss: {best_loss:.4f}")
                    loss_counter = 0 # Reiniciamos si mejora
                else:
                    loss_counter += 1
            logger.info(f"Epoch {epoch} - Phase {phase} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Best Loss: {best_loss:.4f}")
        if loss_counter >= 3:
            logger.info("No improvement in validation loss. Stopping training.")
            break
    return model
    
def net(n_classes=133):
    '''
          Initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, n_classes))
    return model

def create_data_loaders(data_path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    # 1. Define transforms (Data Augmentation)
    # Note: transforms.ToTensor() do the rescale=1./255
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(20),             # rotation_range=20
        transforms.RandomHorizontalFlip(),         # horizontal_flip=True
        transforms.ColorJitter(brightness=0.15),   # brightness_range=[0.85, 1.15]
        transforms.ToTensor(),                     # rescale=1./255
        transforms.Normalize(                      # Normalización estándar ImageNet
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #se puede especificar dos rutas diferentes para train y test mediante los arguments en este caso solo se especifica la raiz y se asume que dentro de esa raiz hay dos carpetas train y valid con la estructura de ImageFolder
    # 2. Define datasets
    train_set = datasets.ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform) 
    valid_set = datasets.ImageFolder(root=os.path.join(data_path, 'valid'), transform=test_transform)
    test_set = datasets.ImageFolder(root=os.path.join(data_path, 'test'), transform=test_transform)
    
    # 3. Create data loaders
    return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True) , torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False), torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training script in: {device}")
    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size)

    '''
    Initialize a model by calling the net function
    '''
    model = net(args.num_classes).to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    hook.register_loss(loss_criterion)
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    start_time = time.time()
    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, args.model_dir, device, hook)
    logger.info("Training time: {} seconds.".format(round(time.time() - start_time, 2)))

    '''
    Test the model to see its accuracy
    '''
    start_time = time.time()
    test(model, test_loader, loss_criterion, device)
    logger.info("Testing time: {} seconds.".format(round(time.time() - start_time, 2)))
    '''
    Save the trained model
    '''
    logger.info("Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_test.pth"))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # Dinamic Hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_classes", type=str, default=133)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data", type=str, default=os.environ["SM_CHANNEL_DATA"])
    
    args = parser.parse_args()
    
    main(args)
