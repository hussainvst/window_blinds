import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from pathlib import Path

app = FastAPI()

UPLOAD_FOLDER = Path("static/uploads/")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

def blend_curtain(window_img_path: str, curtain_img_path: str) -> str:
    """Blends the curtain image onto the window image realistically."""
    window = cv2.imread(window_img_path)
    curtain = cv2.imread(curtain_img_path, cv2.IMREAD_UNCHANGED)
    
    if window is None or curtain is None:
        raise HTTPException(status_code=400, detail="Failed to load one or both images. Check file format and path.")
    
    h, w, _ = window.shape
    curtain = cv2.resize(curtain, (w, int(h / 2)))
    
    if curtain.shape[-1] == 4:
        # Extract the alpha channel as a mask
        mask = curtain[:, :, 3]
        curtain = curtain[:, :, :3]  # Remove alpha channel
    else:
        mask = np.full((curtain.shape[0], curtain.shape[1]), 255, dtype=np.uint8)
    
    # Define the region where the curtain should be placed
    center = (w // 2, int(h / 4))
    
    # Perform seamless cloning for realistic blending
    blended = cv2.seamlessClone(curtain, window, mask, center, cv2.NORMAL_CLONE)
    
    output_path = UPLOAD_FOLDER / "output.jpg"
    cv2.imwrite(str(output_path), blended)
    return str(output_path)

@app.post("/upload/")
def upload_files(window: UploadFile = File(...), curtain: UploadFile = File(...)):
    window_path = UPLOAD_FOLDER / window.filename
    curtain_path = UPLOAD_FOLDER / curtain.filename
    
    with window_path.open("wb") as f:
        f.write(window.file.read())
    with curtain_path.open("wb") as f:
        f.write(curtain.file.read())
    
    result_path = blend_curtain(str(window_path), str(curtain_path))
    
    return {"window_img": str(window_path), "curtain_img": str(curtain_path), "result_img": result_path}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# import cv2
# import numpy as np
# from fastapi import FastAPI, File, UploadFile, HTTPException
# import os
# from pathlib import Path

# app = FastAPI()

# UPLOAD_FOLDER = Path("static/uploads/")
# UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# def overlay_curtain(window_img_path: str, curtain_img_path: str) -> str:
#     """Overlays the curtain image onto the window image."""
#     window = cv2.imread(window_img_path)
#     curtain = cv2.imread(curtain_img_path, cv2.IMREAD_UNCHANGED)
    
#     if window is None or curtain is None:
#         raise HTTPException(status_code=400, detail="Failed to load one or both images. Check file format and path.")
    
#     h, w, _ = window.shape
#     curtain = cv2.resize(curtain, (w, int(h / 2)))
    
#     if curtain.shape[-1] == 4:
#         alpha = curtain[:, :, 3] / 255.0
#         for c in range(0, 3):
#             window[: curtain.shape[0], :, c] = (1 - alpha) * window[: curtain.shape[0], :, c] + alpha * curtain[:, :, c]
#     else:
#         window[: curtain.shape[0], :, :] = curtain
    
#     output_path = UPLOAD_FOLDER / "output.jpg"
#     cv2.imwrite(str(output_path), window)
#     return str(output_path)

# @app.post("/upload/")
# def upload_files(window: UploadFile = File(...), curtain: UploadFile = File(...)):
#     window_path = UPLOAD_FOLDER / window.filename
#     curtain_path = UPLOAD_FOLDER / curtain.filename
    
#     with window_path.open("wb") as f:
#         f.write(window.file.read())
#     with curtain_path.open("wb") as f:
#         f.write(curtain.file.read())
    
#     result_path = overlay_curtain(str(window_path), str(curtain_path))
    
#     return {"window_img": str(window_path), "curtain_img": str(curtain_path), "result_img": result_path}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
