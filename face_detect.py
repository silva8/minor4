import cv2
import sys
import requests
import os
import json
import time
from PIL import Image
from IPython.display import HTML
from os.path import expanduser
import pydocumentdb;
import pydocumentdb.document_client as document_client

#config = {
#    'ENDPOINT': 'https://facedata.documents.azure.com:443/',
#    'MASTERKEY': 'LB8IXwqa2Zmdzh4XeM3ERQRA3c9nucUXV2oLTTI6P8APHpfdfJUjGi4caBwWRt8lJ52s7LGIYpTtoSenPji3kw==',
#    'DOCUMENTDB_DATABASE': 'faceapi_retail',
#    'DOCUMENTDB_COLLECTION': 'advertising'
#};

# Initialize the Python DocumentDB client
#client = document_client.DocumentClient(config['ENDPOINT'], {'masterKey': config['MASTERKEY']})

 # Read databases and take first since id should not be duplicated.
#db = next((data for data in client.ReadDatabases() if data['id'] == "faceapi_retail"))

# Read collections and take first since id should not be duplicated.
#coll = next((coll for coll in client.ReadCollections(db['_self']) if coll['id'] == "advertising"))

subscription_key = "48715137ab314838989ae3066bb3dde1"
assert subscription_key
face_api_url = 'https://westus.api.cognitive.microsoft.com/face/v1.0/detect'
cwd = os.path.realpath(__file__)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

count = 0

#se inicializa el contador en cero para todas las categorias
contador=[["Male Kid","Female Kid","Young Man","Young Woman","Adult Woman","Adult Man","Old Man","Old Woman"],
              [0,0,0,0,0,0,0,0]]

def categoria(edad,sexo):
    if sexo=="male":
        if edad>0 and edad<13:
            contador[1][0]+=1
        elif edad>12 and edad<30:
            contador[1][2]+=1
        elif edad>29 and edad<65:
            contador[1][5]+=1
        elif edad>64:
            contador[1][6]+=1
    else:
        if edad>0 and edad<13:
            contador[1][1]+=1
        elif edad>12 and edad<30:
            contador[1][3]+=1
        elif edad>29 and edad<65:
            contador[1][4]+=1
        elif edad>64:
            contador[1][7]+=1

#analiza el json por persona, invocando funcion categoria. Retorna contador actualizado de la imagen
def analice(decoded):
    largo= len(decoded)
    for x in range(0, largo):
        edad=decoded[x]["faceAttributes"]["age"]
        sexo=decoded[x]["faceAttributes"]["gender"]
        categoria(edad,sexo)
        print ("Tiene "+str(edad)+" anhos y es su genero es "+sexo)

    print (contador[1])
    return contador[1]

#segun la maxima categoria de gente que haya, muestra foto
def identificadorMayoria(contador):
    indicesMaximos=[]
    if len(contador)>0:
        maximo=max(contador)
        indicesMaximos.append(contador.index(maximo))
        contador.remove(maximo)
    i=1
    while maximo==max(contador):
        indicesMaximos.append(i+contador.index(max(contador)))
        contador.remove(maximo)
        i+=1
    return indicesMaximos

def mostrarFotos(indices):
    print(len(indices))
    for categoria in indices:
        if categoria==0:
            image = 'ads/toys1.png'
        elif categoria==1:
            image = 'ads/toys2.png'
        elif categoria==2:
            image = 'ads/beer.png'
        elif categoria==3:
            image = 'ads/perfume.png'
        elif categoria==4:
            image = 'ads/cremas.png'
        elif categoria==5:
            image = 'ads/meat.png'
        elif categoria==6:
            image = 'ads/vinos.png'
        elif categoria==7:
            image = 'ads/chocolate.png'
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        cv2.resizeWindow('image', 1400,800)
        cv2.imshow('image',img)
        time.sleep(3)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    #cv2.imshow('Video', frame)
    if len(faces) != 0:
        cv2.imwrite("faces/frame%d.jpg" % count, frame)     # save frame as JPEG file
        img = open(expanduser("faces/frame%d.jpg" % count), 'rb')
        headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
        }
        params = {
            'returnFaceId': 'true',
            'returnFaceAttributes': 'age,gender'
        }
        response = requests.post(
            face_api_url, params=params, headers=headers, data= img)
        decoded = response.json()
        #print(faces)
        print(identificadorMayoria(analice(decoded)))
        print(contador[0])
        mostrarFotos(identificadorMayoria(analice(decoded)))
        count+=1
        #se inicializa el contador en cero para todas las categorias
        contador=[["Male Kid","Female Kid","Young Man","Young Woman","Adult Woman","Adult Man","Old Man","Old Woman"],
                      [0,0,0,0,0,0,0,0]]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
