#authentication functions
import cv2
from main import jap_model, japv_model, eng_model

def crop_center(img):
    y,x,z = img.shape
    startx = (x-y)//2
    starty = 0
    return img[starty:y,startx:startx+y]

def resizeImage(img):
    r = 256.0 / img.shape[0]
    dim = (256, 256)
    print (dim , r)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def pred_block (image,model_choice):
    if model_choice == "Japanese Modern":
        preds = jap_model.predict(image)
        return preds, model_choice
    elif model_choice == "Japanese Vintage":
        preds = japv_model.predict(image)
        return preds, model_choice
    else:
        preds = eng_model.predict(image)
        return preds, model_choice

