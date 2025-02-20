import gradio as gr
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from utils import Load_model, download_image, load_images, authenticate_card,grade_cards,authenticate_card_with_review,ximilar_card_ocr_id,ximilar_card_condition,ximilar_grade
from ximilar_services import ximilar_description 
#loading models
eng_model, jap_model, japv_model, front_grade_model, back_grade_model = Load_model()

modelsz = [eng_model, jap_model, japv_model, front_grade_model, back_grade_model ]

def image_data_handler(ximilar_grading_data= None,ximilar_ocr_data= None,ximilar_card_condition= None,image = None):

    card_condition_details = " "
    finel_txt = " "
    if ximilar_grading_data is not None:
        if image is None:
            image_url = ximilar_grading_data["records"][0]["_url"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        objects = ximilar_grading_data["records"][0]["_objects"]
        corners = ximilar_grading_data["records"][0]["corners"]
        edges = ximilar_grading_data["records"][0]["edges"]
        card_info = ximilar_grading_data["records"][0]["card"][0]
        tags_data = card_info["_tags"]
        top_category = "Card"
        category_name = tags_data["Category"][0]["name"]
        which_side = tags_data["Side"][0]["name"]
        all_grade_data = ximilar_grading_data["records"][0]["grades"]
        top_finel_grade = all_grade_data["final"]
        corners_grade = all_grade_data["corners"]
        centering = all_grade_data["centering"]
        edges_data = all_grade_data["edges"]
        surface = all_grade_data["surface"]

        finel_txt = f"\nTop Category: {top_category}\nCategory: {category_name}\nSide: {which_side}\n\nAI Grade: {top_finel_grade}\nCentering: {centering}\nCorners: {corners_grade}\nEdges: {edges_data}\nSurface: {surface}"
        print("txt done")
        used_positions = []

        def get_text_position(x, y, text_size, spacing=5):
            x_end, y_end = x + text_size[0], y - text_size[1] - spacing
            # Adjust to stay within image bounds
            x = min(max(0, x), image.shape[1] - text_size[0])
            y = min(max(text_size[1] + spacing, y), image.shape[0])
            # Move downwards if overlap detected
            while any([(abs(x - ux) < text_size[0] and abs(y - uy) < text_size[1] + spacing) for ux, uy in used_positions]):
                y += text_size[1] + spacing
                y = min(y, image.shape[0])
            used_positions.append((x, y))
            return x, y
        '''
        for obj in objects:
            bbox = obj["bound_box"]
            prob = obj["prob"]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            text = f"{obj['name']} (Prob: {prob:.2f})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = get_text_position(bbox[0], bbox[1], text_size)
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0], text_y), (0, 255, 0), -1)
            cv2.putText(image, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        '''
        print(len(corners))
        for corner in corners:
            bbox = corner["bound_box"]
            grade = corner["grade"]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            text = f"{corner['name']} (Grade: {grade})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = get_text_position(bbox[0], bbox[1], text_size)
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), 
                        (text_x + text_size[0], text_y), (255, 0, 0), -1)
            cv2.putText(image, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        print("corners done")

        for edge in edges:
            polygon = np.array(edge["polygon"], np.int32).reshape((-1, 1, 2))
            grade = edge["grade"]
            cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
            text = f"{edge['name']} (Grade: {grade})"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = get_text_position(polygon[0][0][0], polygon[0][0][1], text_size)
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0], text_y), (0, 255, 255), -1)
            cv2.putText(image, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        print("edges done")
        card_polygon = np.array(card_info["polygon"], np.int32).reshape((-1, 1, 2))
        card_grade = ximilar_grading_data["records"][0]["grades"]["final"]
        #cv2.polylines(image, [card_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
        text = f"Card Grade: {card_grade}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x, text_y = get_text_position(card_polygon[0][0][0], card_polygon[0][0][1], text_size)
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 10), 
                      (text_x + text_size[0], text_y), (0, 0, 255), -1)
        cv2.putText(image, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("tsk done")

    if ximilar_ocr_data is not None:
        # Load the image from URL if not provided
        if image is None:
            image_url = ximilar_ocr_data["records"][0]["_url"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = np.array(image)

        # Convert to OpenCV format (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        used_positions = []

        def get_text_position(x, y, text_size, spacing=5):
            x = min(max(0, x), image.shape[1] - text_size[0])  # Ensure within width bounds
            y = min(max(text_size[1] + spacing, y), image.shape[0])  # Ensure within height bounds
            
            # Adjust position if there's overlap
            while any(abs(x - ux) < text_size[0] and abs(y - uy) < text_size[1] + spacing for ux, uy in used_positions):
                y += text_size[1] + spacing
                y = min(y, image.shape[0] - spacing)  # Prevent going beyond image bottom

            used_positions.append((x, y))
            return x, y

        # Process each OCR text
        for text_data in ximilar_ocr_data['records'][0]['_ocr']['texts']:
            # Extract polygon points and text
            points = np.array(text_data['polygon'], np.int32).reshape((-1, 1, 2))
            text = text_data['text']

            # Draw the polygon (bounding box)
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Determine text position
            x, y = points[0][0]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = get_text_position(x, y, text_size)

            # Draw background rectangle for text
            cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), 
                          (text_x + text_size[0], text_y), (0, 255, 0), -1)
            # Place the text on the image
            cv2.putText(image, text, (text_x, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Convert back to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    if ximilar_card_condition!= None:
        if image is None:
            # Load the image from URL
            image_url = ximilar_card_condition["records"][0]["_url"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = np.array(image)
        
        # Convert to OpenCV format (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Loop over objects to extract and draw each item
        obj = ximilar_card_condition["records"][0]["_objects"][0]
        # Extract bounding box, category, and condition details
        #x_min, y_min, x_max, y_max = obj["bound_box"]
        top_category = obj["Top Category"][0]["name"]
        category_name = obj["Category"][0]["name"]
        category_prob = obj["Category"][0]["prob"]
        condition_label = obj["Condition"][0]["label"]

        # Draw bounding box
        #cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Prepare text annotations
        text_category = f"Top Category: {top_category} \n {category_name} {category_prob*100:.1f}%"
        text_condition = f"Condition: {condition_label}"
        card_condition_details = text_category + "\n" + text_condition
        '''
        # Put category text above the bounding box
        cv2.putText(image, text_category, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 125), 1, cv2.LINE_AA)
        
        # Put condition text below the bounding box
        cv2.putText(image, text_condition, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Convert back to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)'''
        image= np.zeros((256, 256, 3), dtype=np.uint8)

    return image, card_condition_details, finel_txt


def gradio_main(choice,front_card_link, back_card_link,catogrey):
    front_card_path = download_image(url=front_card_link,folder_path="front.jpg")
    back_card_path = download_image(url=back_card_link,folder_path="back.jpg")
    front_card_img, back_card_img = load_images(front_card_path,back_card_path)

    black_img= np.zeros((256, 256, 3), dtype=np.uint8)
    if choice == "Authenticate Card":
        model_choice = catogrey
        authentic_data = authenticate_card(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        return authentic_data, cv2.cvtColor(front_card_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(back_card_img, cv2.COLOR_BGR2RGB)
    elif choice == "Grade Cards":
        grade_data = grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
        return grade_data, cv2.cvtColor(front_card_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(back_card_img, cv2.COLOR_BGR2RGB)
    elif choice == "Authenticate and Get Review":
        model_choice = catogrey
        authentic_data, comments = authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        return authentic_data + "\n" + comments, cv2.cvtColor(front_card_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(back_card_img, cv2.COLOR_BGR2RGB)
    elif choice == "Ximilar Card Grading":
        frontgrade_data_ximilar= ximilar_grade(image_url=front_card_link)
        fron_image_graded, _, front_grade_txt = image_data_handler(ximilar_grading_data=frontgrade_data_ximilar)
        back_grade_data_ximilar= ximilar_grade(image_url=back_card_link)
        back_image_graded, _, back_grade_txt = image_data_handler(ximilar_grading_data=back_grade_data_ximilar)
        print("done")
        return front_grade_txt+"\n"+back_grade_txt, fron_image_graded, back_image_graded
    elif choice == "Ximilar Card Description":
        description_data = ximilar_description(image_url=front_card_link)
        return description_data,cv2.cvtColor(front_card_img, cv2.COLOR_BGR2RGB) ,cv2.cvtColor(back_card_img, cv2.COLOR_BGR2RGB)
    elif choice == "Ximilar Card Score":
        front_card_condition = ximilar_card_condition(image_url=front_card_link)
        _,front_text_condition,_ = image_data_handler(ximilar_card_condition=front_card_condition)
        back_card_condition = ximilar_card_condition(image_url=back_card_link)
        _,back_text_condition,_ = image_data_handler(ximilar_card_condition=back_card_condition)
        return f"Front side\n {front_text_condition}+\nBack Side\n {back_text_condition}", cv2.cvtColor(front_card_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(back_card_img, cv2.COLOR_BGR2RGB)
    else:
        return "Invalid choice. Please select again.",black_img,black_img

    '''
    elif choice == "Do All":
        model_choice = sub_menu()
        authentic_data, comments = authenticate_card_with_review(back_card_img=back_card_img, model_choice=model_choice,modelsz=modelsz)
        grade_data = grade_cards(front_image=front_card_path,back_image=back_card_path,front_model=modelsz[3],back_model=modelsz[4])
        grade_data_ximilar= ximilar_grade(image_url=front_card_link)
        ocr_data = ximilar_card_ocr_id(image_url=front_card_link)
        card_condition = ximilar_card_condition(image_url=front_card_link)
        image_graded, image_ocr, text_condition = image_data_handler(grade_data_ximilar,ocr_data,card_condition)
        return ("Authentication: " +authentic_data + "\n" +"Reviews: "+ comments + "TCG Grafe: "+ grade_data, text_condition[2]), image_graded, image_ocr'''


"""# Gradio Launch #"""
choice = gr.Dropdown(choices=["Authenticate Card","Authenticate and Get Review","Grade Cards", "Ximilar Card Grading","Ximilar Card Description","Ximilar Card Score"], label="Service")
model_selector = gr.Dropdown(choices=["Japanese Vintage","Japanese Modern", "English"], label="Select Model")
title= "Pokemon Card Predictor / Authenticator / Identifier"
description=""
examples=[['00395.JPG'], ['00430.JPG']]

iface= gr.Interface(
    fn= gradio_main,
    inputs=[choice,"textbox", "textbox", model_selector],
    outputs=["text","image","image"],
    title=title,
    description=description,
    #examples=examples
)

iface.launch(debug=True,share=True)