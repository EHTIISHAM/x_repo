#utils
import numpy as np
import torch
from PIL import Image
from tensorflow.keras.models import load_model

from input_module import load_images, download_image, delete_image
from auth_utils import crop_center, pred_block, resizeImage
from grading_utils import initialize_model, device ,display_prediction
from ximilar_services import ximilar_card_condition, ximilar_card_ocr_id, ximilar_grade
from comments_utils import auth_comment


def Load_model():
    print(device)
    try:
        #loading authentication models
        eng_model = load_model("eng_detect.h5")
        jap_model = load_model("jap_model.h5")
        japv_model = load_model("japv_model.h5")

        #loading grading models
        Load_dict = True
        front_grade_model , _ = initialize_model(model_name='resnet',num_classes=10)
        back_grade_model , _ = initialize_model(model_name='resnet',num_classes=10)
        if Load_dict:
            front_model_path = 'front_grading_epoch_300.pth'
            back_model_path = 'back_grading_epoch_350.pth'
            if device == 'cpu':
                front_grade_model.load_state_dict(torch.load(front_model_path, map_location=torch.device('cpu'),weights_only=True))
                back_grade_model.load_state_dict(torch.load(back_model_path, map_location=torch.device('cpu'),weights_only=True))
            front_grade_model.load_state_dict(torch.load(front_model_path))
            back_grade_model.load_state_dict(torch.load(back_model_path))
        front_grade_model.eval()
        back_grade_model.eval()

        front_grade_model = front_grade_model.to(device)
        back_grade_model = back_grade_model.to(device)
    except Exception as e:
        eng_model = None
        jap_model = None
        japv_model = None
        front_grade_model = None
        back_grade_model = None
        print('\n')
        print(e)
        print('\n')
        print('Model Loading Failed')
    return eng_model, jap_model, japv_model, front_grade_model, back_grade_model

def sub_menu():
    print("\nSelect a category:")
    print("1. English Cards")
    print("2. Japanese Cards")
    print("3. Japanese Vintage Cards")
    category = input("Enter your choice (1-3): ")
    if category == "1":
        print("Selected English Cards")
        model_choice = "English"
    elif category == "2":
        print("Selected Japanese Cards")
        model_choice = "Japanese Modern"
    elif category == "3":
        print("Selected Japanese Vintage Cards")
        model_choice == "Japanese Vintage"
    else:
        print("Invalid category. Please choose again.")
        sub_menu()
    return model_choice


def main_menu(modelsz):
    print("Enter front side image URL: ")
    front_url = input()
    front_card_path = download_image(url=front_url,folder_path="front.jpg")
    print("Enter back side image URL: ")
    back_url = input()
    back_card_path = download_image(url=back_url,folder_path="back.jpg")
    front_card_img, back_card_img = load_images(front_card_path,back_card_path)

    while True:
        print("\nMain Menu:")
        print("1. Authenticate Card and Select Category")
        print("2. Grade Cards")
        print("3. Authenticate Card and Get Review")
        print("4. Ximilar Card Grading")
        print("5. Ximilar Card Score for Surface, Corners, and Edges")
        print("6. Ximilar Card Description")
        print("0. Do All")
        print("q. Quit")

        choice = input("Enter your choice: ")

        if choice == "1":
            model_choice = sub_menu()
            authenticate_card(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        elif choice == "2":
            grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
        elif choice == "3":
            model_choice = sub_menu()
            authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        elif choice == "4":
            ximilar_grade(image_url=front_url)
        elif choice == "5":
            ximilar_card_ocr_id(image_url=front_url)
        elif choice == "6":
            ximilar_card_condition(image_url=front_url)
        elif choice == "0":
            model_choice = sub_menu()
            authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
            grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
            ximilar_grade(image_url=front_url)
            ximilar_card_ocr_id(image_url=front_url)
            ximilar_card_condition(image_url=front_url)
        elif choice.lower() == "q":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select again.")



def authenticate_card(back_card_img,model_choice,modelsz):
    back_image = crop_center(back_card_img)
    back_image = resizeImage(back_image)
    K_test = (back_image - back_image.mean()) / back_image.std()
    K_test = np.reshape(K_test, (1, 256, 256, 3))
    predictions, selection = pred_block(K_test, model_choice,modelsz[0],modelsz[1],modelsz[2])
    confi = predictions[0][0]
    if selection == "Japanese Modern" or selection == "Japanese Vintage" :
        card = selection
    else:
        card = 'English'

    if confi <= 0.5:
        confi = 1- confi
    confi = round(confi*100, 2)
    predictions = list(map(lambda x: 0 if x<0.5 else 1, predictions)) # get binary values predictions with 0.5 as thresold
    if selection == "English":
      if predictions == [1]:
          print (f"Card is {card} and This card is likely genuine. confidence: {confi}%")

          return f"Card is {card} and This card is likely genuine. confidence: {confi}%"
      else:
          print (f"Card is {card} and This card is likely counterfeit. confidence: {confi}%")

          return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%"

    else:
      if predictions == [1]:
        print (f"Card is {card} and This card is likely counterfeit. confidence: {confi}%")

        return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%"
      else:
        print (f"Card is {card} and This card is likely genuine. confidence: {confi}%")

        return f"Card is {card} and This card is likely genuine. confidence: {confi}%"


def authenticate_card_with_review(back_card_img,model_choice,modelsz):
    back_image = crop_center(back_card_img)
    back_image = resizeImage(back_image)
    K_test = (back_image - back_image.mean()) / back_image.std()
    K_test = np.reshape(K_test, (1, 256, 256, 3))
    predictions, selection = pred_block(K_test, model_choice,modelsz[0],modelsz[1],modelsz[2])
    confi = predictions[0][0]
    if selection == "Japanese Modern" or selection == "Japanese Vintage" :
        card = selection
    else:
        card = 'English'

    if confi <= 0.5:
        confi = 1- confi
    confi = round(confi*100, 2)
    predictions = list(map(lambda x: 0 if x<0.5 else 1, predictions)) # get binary values predictions with 0.5 as thresold
    comments , names = auth_comment(preds=predictions,choice=selection)
    proper_comments = f"{names[0][1]}\n{names[0][0]} : {comments[0]}\n\n{names[1][1]}\n{names[1][0]} : {comments[1]}"

    if selection == "English":
      if predictions == [1]:
          print (f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments)

          return f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments
      else:
          print (f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments)

          return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments

    else:
      if predictions == [1]:
        print (f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments)
        return f"Card is {card} and This card is likely counterfeit. confidence: {confi}%" , proper_comments

      else:
        print (f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments)

        return f"Card is {card} and This card is likely genuine. confidence: {confi}%" , proper_comments

def grade_cards(front_model, back_model, front_image, back_image):

    class_names = ['PSA1', 'PSA10', 'PSA2', 'PSA3', 'PSA4', 'PSA5', 'PSA6', 'PSA7', 'PSA8', 'PSA9']

    grade = display_prediction(front_image, back_image, front_model, back_model, class_names)
    print(grade)
    
    return grade

# dead functionsss

"""
def gradio_main(modelsz,choice,card_links,catogrey):
    front_card_path = download_image(url=card_links[0],folder_path="front.jpg")
    back_card_path = download_image(url=card_links[1],folder_path="back.jpg")
    front_card_img, back_card_img = load_images(front_card_path,back_card_path)
    if choice == "Authenticate Card":
        model_choice = catogrey
        authentic_data = authenticate_card(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        return authentic_data
    elif choice == "Grade Cards":
        grade_data = grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
        return grade_data
    elif choice == "Authenticate and Get Review":
        model_choice = catogrey
        authentic_data, comments = authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        return (authentic_data + "\n" + comments)
    elif choice == "Ximilar Card Grading":
        grade_data_ximilar= ximilar_grade(image_url=card_links[0])
    elif choice == "Ximilar Card Description":
        ocr_data = ximilar_card_ocr_id(image_url=card_links[0])
    elif choice == "Ximilar Card Score":
        card_condition = ximilar_card_condition(image_url=card_links[0])
    elif choice == "Do All":
        model_choice = sub_menu()
        authentic_data, comments = authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        grade_data = grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
        grade_data_ximilar= ximilar_grade(image_url=card_links[0])
        ocr_data = ximilar_card_ocr_id(image_url=card_links[0])
        card_condition = ximilar_card_condition(image_url=card_links[0])
    else:
        print("Invalid choice. Please select again.")
        
"""