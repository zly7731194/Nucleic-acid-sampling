import io
import os

import face_recognition
import uvicorn
from fastapi import FastAPI, File

app = FastAPI()

# walk images directory and get all images
def list_all_images():
    images = {}
    for root, dirs, files in os.walk('images'):
        for file in files:
            if file.endswith('.jpg'):
                images[file.strip(".jpg")] = os.path.join(root, file)
    return images


def get_all_encodings():
    result = {}
    for name, img_path in list_all_images().items():
        result[name] = face_recognition.face_encodings(face_recognition.load_image_file(img_path))[0]
    return result


@app.post("/face_recognition/")
async def create_file(file: bytes = File()):
    contents = file
    f_bytes = io.BytesIO(bytes(contents))
    encodings = face_recognition.face_encodings(face_recognition.load_image_file(f_bytes))
    if len(encodings) == 0:
        return {"message": "No face detected", "code": 400}
    for name, enc in all_encodings.items():
        compares = face_recognition.compare_faces(encodings, enc, tolerance=0.2)
        if any(compares):
            return {"message": "OK", "name": name, "code": 200}
    return {"message": "No match", "code": 404}


if __name__ == '__main__':
    print("loading encodings...")
    all_encodings = get_all_encodings()
    print("encodings loaded")
    uvicorn.run(app, host="0.0.0.0", port=7544)
