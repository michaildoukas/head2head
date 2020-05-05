# Copyright (C) 2019 Facesoft Ltd - All Rights Reserved

import base64
import json
import requests # needs to be installed


def reconstruct_image_web_api(filename, reconstructURL = "https://reconstruction-mobile.facesoft.io/processq/300"):
    """ Reconstruct the 3D model of a face using the web API (https://facesoft.io/docs.html)

    filename -- Path to image file with the face to reconstruct
    reconstructURL -- URL to use from the web API 

    Returns the JSON file of the reconstructed model (texture, shape weights, expression weights)
    """

    #reconstructURL = "https://reconstruction.facesoft.io/processq/300"
    #reconstructURL = "http://reconstruction-mobile.facesoft.io"
    #reconstructURL = "http://ec2-34-249-19-163.eu-west-1.compute.amazonaws.com:8080/weights/no/20"

    with open(filename, "rb") as f:
        data = f.read()
        base64_bytes = base64.b64encode(data)
        base64string = base64_bytes.decode('utf-8')

    payload = {
        'image': base64string,
    }

    response = requests.request("POST", reconstructURL, data = json.dumps(payload))
    #print(type(response))
    
    return response.json()
