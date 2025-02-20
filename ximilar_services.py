__APIKEY__ = "b241ef1cecb9a8a3fb2e4336ccf666e1a3c818b2"

import requests

def ximilar_grade(image_url,api_token = __APIKEY__):
    # Define the headers and the data
    headers = {
        'Authorization': f'Token {api_token}',
        'content-type': 'application/json',
    }

    data = {
        "records": [
            {"_url": image_url}
        ]
    }

    # Send the POST request
    response = requests.post(
        'https://api.ximilar.com/card-grader/v2/grade',
        headers=headers,
        json=data
    )

    # Print the response
    print(response.status_code)  # To check if the request was successful
    data = response.json()      # To see th
    for record in data['records']:
        # Print corner names and grades
        print("Corners:")
        for corner in record['corners']:
            print(f"{corner['name']}: {corner['grade']}")

        # Print edge names and grades
        print("\nEdges:")
        for edge in record['edges']:
            print(f"{edge['name']}: {edge['grade']}")

        # Print centering and surface grades
        print(f"\nCentering Grade: {record['card'][0]['centering']['grade']}")
        print(f"Surface Grade: {record['card'][0]['surface']['grade']}")

        # Print combined grades
        print("\nCombined Grades:")
        for grade_type, grade_value in record['grades'].items():
            print(f"{grade_type.capitalize()} Grade: {grade_value}")
    return data

def ximilar_card_ocr_id(image_url,api_token = __APIKEY__):
    # Define the headers and the data
    headers = {
        'Authorization': f'Token {api_token}',
        'Content-Type': 'application/json',
    }

    data = {
        "lang": "en",
        "records": [
            {"_url": image_url}
        ]
    }

    # Send the POST request
    response = requests.post(
        'https://api.ximilar.com/ocr/v2/read',
        headers=headers,
        json=data
    )

    # Print the response
    print(response.status_code)  # To check if the request was successful
    data = response.json()   # To see the OCR response data
    for text_entry in data['records'][0]['_ocr']['texts']:
        print(text_entry['text'])
    return data
def ximilar_card_condition(image_url,api_token = __APIKEY__):
    # Define the headers and the data
    headers = {
        'Authorization': f'Token {api_token}',
        'content-type': 'application/json',
    }

    data = {
        "records": [
            {"_url": image_url}
        ],
        "mode": "tcgplayer"
    }

    # Send the POST request
    response = requests.post(
        'https://api.ximilar.com/card-grader/v2/condition',
        headers=headers,
        json=data
    )

    # Print the response
    print(response.status_code)  # To check if the request was successful
    data = response.json()   
    condition_label = data['records'][0]['_objects'][0]['Condition'][0]['label']
    print(f"Condition Label: {condition_label}")

    return data

def ximilar_description(image_url, api_token = __APIKEY__):
    headers = {
        'Authorization': f'Token {api_token}',
        'content-type': 'application/json',
    }

    data = {
        "records": [
            {"_url": image_url}
        ]
    }
    response = requests.post(
        'https://api.ximilar.com/collectibles/v2/tcg_id',
        headers=headers,
        json=data
    )
    data = response.json()  
    txt = ""
    all_data = data['records'][0]['_objects'][0]["_identification"]["best_match"]
    set_data = all_data["set"]
    set_code = all_data["set_code"]
    full_name = all_data["full_name"]
    series = all_data["series"]
    year = all_data["year"]
    card_number = all_data["card_number"]
    namee = all_data["name"]
    sub_cat = all_data["subcategory"]
    links = all_data["links"]["ebay.com"]

    txt = f"Card identification\n\nSet: {set_data}\nSet Code: {set_code}\nFull Name: {full_name}\nSeries: {series}\nYear: {year}\nCard Number: {card_number}\nName: {namee}\nSubcategory: {sub_cat}\nLink: {links}"

    return txt