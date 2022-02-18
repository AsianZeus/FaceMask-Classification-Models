from fastapi import FastAPI, status, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import pipeline
from Classify import classify
from io import BytesIO
import numpy as np
import cv2

model = pipeline("image-classification", model='AshatSurolia/DeiT-FaceMask-Finetuned', tokenizer='AshatSurolia/DeiT-FaceMask-Finetuned')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return "Welcome to DSMATICS!"

@app.post("/image", status_code=status.HTTP_201_CREATED)
def create_image(Image: bytes = File(...)):
    if Image:
        try:
            nparr = np.frombuffer(Image, dtype=np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            count, labels, image = classify(img, model)
            _, im_png = cv2.imencode(".png", image)
            return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/png", headers={"Human-Count": str(count), "Labels":str(labels)})
        except Exception as e:
            raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in processing image...")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Image not found...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)