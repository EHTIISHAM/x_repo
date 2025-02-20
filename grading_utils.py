# Grading Functions
import torch
import torchvision.models as models
from torch import nn
import torchvision.transforms as transforms
import numpy as np
import PIL.Image as Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    """
    Helper function to freeze layers when fine-tuning a model.
    Sets all neurons except those in a top layer to not return gradient.

    From
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tut    orial.html
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):

    """
    Helper function to initialize pretrained AlexNet (and other vision)
    models in Pytorch.
    From https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """

    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# grader utils
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['PSA1', 'PSA10', 'PSA2', 'PSA3', 'PSA4', 'PSA5', 'PSA6', 'PSA7', 'PSA8', 'PSA9']

def grader_load_image(image):
    image = Image.open(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image
def predict(image, model):
    image = grader_load_image(image).to(device)
    _, preds = model(image)
    return preds.item()
def display_prediction(front_image,back_image, model_front,model_back, class_names=class_names):
    front_prediction = predict(front_image, model_front)
    back_prediction = predict(back_image, model_back)
    prediction = (np.argmax(front_prediction) + np.argmax(back_prediction))/2
    predicted_class = class_names[prediction.astype(np.int64)]
    return predicted_class

