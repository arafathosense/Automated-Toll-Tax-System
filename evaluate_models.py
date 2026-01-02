import os
import csv
from ultralytics import YOLO
from PIL import Image, ImageDraw

# Paths and Model Configuration
TRAINED_MODEL_PATH = r"C:\Users\soban\PycharmProjects\Smart-Toll-Tax-System\Merged_Dataset\runs\train\yolov8\best.pt"
PRETRAINED_MODEL_NAME = "yolov8m.pt"

TEST_IMG_DIR = r"Merged_Dataset\test\images"
TEST_LABEL_DIR = r"Merged_Dataset\test\labels"
DATA_YAML = r"Merged_Dataset\data.yaml"

# Visualization directories
VIS_DIR = "visualization"
os.makedirs(os.path.join(VIS_DIR, "trained"), exist_ok=True)
os.makedirs(os.path.join(VIS_DIR, "pretrained"), exist_ok=True)


# Load Models
trained_model = YOLO(TRAINED_MODEL_PATH)
pretrained_model = YOLO(PRETRAINED_MODEL_NAME)


# Read YOLO label file
def load_yolo_label(label_path):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            boxes.append([cls, xc, yc, w, h])
    return boxes

# Convert YOLO normalized to x1,y1,x2,y2 pixel coords
def yolo_to_xyxy(box, img_w, img_h):
    cls, xc, yc, w, h = box
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [cls, x1, y1, x2, y2]


# Compute IoU
def compute_iou(box1, box2):
    _, x1, y1, x2, y2 = box1
    _, gx1, gy1, gx2, gy2 = box2

    xi1 = max(x1, gx1)
    yi1 = max(y1, gy1)
    xi2 = min(x2, gx2)
    yi2 = min(y2, gy2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (gx2 - gx1) * (gy2 - gy1)
    union = area1 + area2 - inter

    if union <= 0:
        return 0
    return inter / union

# Draw predictions + GT on image
def visualize_boxes(img_path, gt_boxes, pred_boxes, save_path, model_name):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Ground Truth = Green
    for b in gt_boxes:
        _, x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 12), "GT", fill="green")

    # Predictions = Red
    for b in pred_boxes:
        _, x1, y1, x2, y2 = b
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y2 + 2), model_name, fill="red")

    img.save(save_path)


# Evaluation Function
def evaluate_model(model, model_name, csv_output, vis_subdir):
    print(f"\n========== Evaluating {model_name} ==========\n")

    TP = 0
    FP = 0
    FN = 0
    total_images = 0
    rows = []

    for img_name in os.listdir(TEST_IMG_DIR):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        total_images += 1
        img_path = os.path.join(TEST_IMG_DIR, img_name)

        # Prepare image once
        img = Image.open(img_path)
        img_w, img_h = img.size

        # Load ground truth boxes
        label_path = os.path.join(TEST_LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt")
        gt_boxes = [yolo_to_xyxy(b, img_w, img_h) for b in load_yolo_label(label_path)]

        # Predict
        results = model.predict(img_path, verbose=False)[0]

        pred_boxes = []
        for b in results.boxes:
            cls_id = int(b.cls.item())
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0]
            pred_boxes.append([cls_id, x1, y1, x2, y2])

        # MATCHING PRED ↔ GT
        matched_gt = set()
        matched_pred = set()

        for i, p in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = None

            for j, g in enumerate(gt_boxes):
                iou = compute_iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= 0.5:
                TP += 1
                matched_pred.add(i)
                matched_gt.add(best_gt_idx)
            else:
                FP += 1

        FN += (len(gt_boxes) - len(matched_gt))

        # Save per-image results
        rows.append({
            "image": img_name,
            "ground_truth": len(gt_boxes),
            "detections": len(pred_boxes),
            "TP": len(matched_gt),
            "FP": len(pred_boxes) - len(matched_pred),
            "FN": len(gt_boxes) - len(matched_gt)
        })

        # Visualization
        save_path = os.path.join(VIS_DIR, vis_subdir, img_name)
        visualize_boxes(img_path, gt_boxes, pred_boxes, save_path, model_name)

    # Compute metrics
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # YOLO validation metrics (Ultralytics v8+)
    val = model.val(data=DATA_YAML, imgsz=640, verbose=False)
    # mAP50 = val.results_dict["metrics/mAP50"]
    # mAP5095 = val.results_dict["metrics/mAP50-95"]
    mAP50 = val.box.map50
    mAP5095 = val.box.map

    print("\n============== Metrics ==================")
    print(f"Model: {model_name}")
    print(f"Images: {total_images}")
    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"mAP@0.5:   {mAP50:.4f}")
    print(f"mAP@0.5:0.95 = {mAP5095:.4f}")
    print("=========================================\n")

    # Save CSV
    with open(csv_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "ground_truth", "detections", "TP", "FP", "FN"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved : {csv_output}")
    print(f"Images saved : {os.path.join(VIS_DIR, vis_subdir)}\n")

    return precision, recall, f1, mAP50, mAP5095


# Run evaluations
if __name__ == '__main__':
    evaluate_model(trained_model, "Custom YOLOv8 (best.pt)", "trained_model_results.csv", "trained")
    evaluate_model(pretrained_model, "Pretrained YOLOv8m", "pretrained_model_results.csv", "pretrained")

