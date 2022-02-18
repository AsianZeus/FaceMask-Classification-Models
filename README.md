# FaceMask-Classification-Models
This repository holds the downstream task of Face Mask Classification performed on Self Currated Custom Dataset with various State of the Art deep learning models like ViT, BeIT, DeIT, ConvNeXt, VGG16, EfficientNetV2, RegNett and MobileNetV3.

---


<div align="center">
	<a href="https://huggingface.co/AkshatSurolia/ConvNeXt-FaceMask-Finetuned"><img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fhuggingface.co%2FAkshatSurolia%2FConvNeXt-FaceMask-Finetuned&label=Huggingface&color=green&message=ConvNeXt-FaceMask-Finetuned&logo=huggingface"/></a>
	<a href="https://huggingface.co/AkshatSurolia/BEiT-FaceMask-Finetuned"><img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fhuggingface.co%2FAkshatSurolia%2FBEiT-FaceMask-Finetuned&label=Huggingface&color=green&message=BEiT-FaceMask-Finetuned&logo=huggingface"/></a>
	<a href="https://huggingface.co/AkshatSurolia/ViT-FaceMask-Finetuned"><img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fhuggingface.co%2FAkshatSurolia%2FViT-FaceMask-Finetuned&label=Huggingface&color=green&message=ViT-FaceMask-Finetuned&logo=huggingface"/></a>
    <a href="https://huggingface.co/AkshatSurolia/DeiT-FaceMask-Finetuned"><img alt="Doc" src="https://img.shields.io/static/v1?url=https%3A%2F%2Fhuggingface.co%2FAkshatSurolia%2FDeiT-FaceMask-Finetuned&label=Huggingface&color=green&message=DeiT-FaceMask-Finetuned&logo=huggingface"/></a>
</div>

----
## Face Mask Detection API powered by FastAPI.

- Detects Multiple Faces from afar.
- Classify faces as Mask or No Mask.
- Sends Annotated response of an image with labelled faces.
- Sends Numbers of faces in Headers Response.
- Sends Labels as an array in Header Response.

---

## API
- Made with FASTAPI
- Validation by PyDantic

## Working
- URL: /image
- Type: Post
- Body: form-data
    - Key: "Image"
    - Value: Selected Image

    ### Response
    - Status: 201
        - Annotated Image
    - Headers:
        - "human-count": Number of faces detected
        - "labels": Labels of faces detected [MASK | NO MASK]
