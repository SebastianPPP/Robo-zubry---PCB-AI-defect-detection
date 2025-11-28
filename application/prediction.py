from pathlib import Path
import random
from ultralytics import YOLO
from const import YOLO_V8N_160_E8, YOLO_11N_320_E10


def predict_img(
    model_name: str,
    img_path: Path = Path("./data/val/images"),
    random_val_img: bool = False
):
    print(img_path.resolve())
    options = [path for path in img_path.resolve().iterdir()]
    if random_val_img:
        img_path = random.choice(options)
    model_path = f"model/{model_name}"
    model = YOLO(model_path)
    result_ = model(
        img_path,
        imgsz=320
    )
    print(result_[0].boxes)
    return [{
        "xyxy": [res.xyxy for res in r.boxes],
        "plot": r.plot(),
        "confs": [res.conf for res in r.boxes],
        "classes": [res.cls for res in r.boxes]
    } for r in result_][0]


if __name__ == "__main__":
    result = predict_img(YOLO_11N_320_E10, random_val_img=True)
