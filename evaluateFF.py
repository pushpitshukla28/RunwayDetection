# evaluate.py - Evaluate using mask->coords (robust OpenCV) for anchor & boolean scores
import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from main import RunwayDetector   # assumes main.RunwayDetector exists and is the segmentation model

# --- Configuration (use raw strings on Windows) ---
MODEL_PATH = "runway_detector_model_final.pth"
TEST_IMG_DIR = r"C:\Users\nandi\Downloads\640x360\640x360\test"
TEST_LABELS_JSON_PATH = r"C:\Users\nandi\Downloads\labels\labels\lines\test_labels_640x360.json"
SUBMISSION_CSV_PATH = "submission.csv"

SCORE_H, SCORE_W = 360, 640
MODEL_H, MODEL_W = 256, 512
# ----------------------------------------------------

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0.0

def create_mask_from_coords(coords, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        ledg_p1, ledg_p2 = coords[0:2], coords[2:4]
        redg_p1, redg_p2 = coords[4:6], coords[6:8]
        polygon_pts = np.array([ledg_p1, ledg_p2, redg_p2, redg_p1], dtype=np.int32)
        cv2.fillPoly(mask, [polygon_pts], 1)
    except Exception:
        pass
    return mask

def order_box_points(box):
    pts = np.array(box, dtype=np.float32)
    idx = np.argsort(pts[:, 0])
    left = pts[idx[:2]]
    right = pts[idx[2:]]
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]
    return left[0], left[1], right[0], right[1]

def mask_to_coords(mask):
    mask_u = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(12, dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 16:
        return np.zeros(12, dtype=np.float32)

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    left_top, left_bottom, right_top, right_bottom = order_box_points(box)

    H, W = mask.shape
    def clamp(pt):
        x = float(np.clip(pt[0], 0, W - 1))
        y = float(np.clip(pt[1], 0, H - 1))
        return np.array([x, y], dtype=np.float32)

    lt, lb, rt, rb = map(clamp, [left_top, left_bottom, right_top, right_bottom])
    ctl_top = (lt + rt) / 2.0
    ctl_bottom = (lb + rb) / 2.0

    coords = np.zeros(12, dtype=np.float32)
    coords[0:2], coords[2:4] = lt, lb
    coords[4:6], coords[6:8] = rt, rb
    coords[8:10], coords[10:12] = ctl_top, ctl_bottom
    return coords

def calculate_anchor_score(pred_mask, pred_coords):
    polygon_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    try:
        ledg_p1, ledg_p2 = pred_coords[0:2], pred_coords[2:4]
        redg_p1, redg_p2 = pred_coords[4:6], pred_coords[6:8]
        polygon_pts = np.array([ledg_p1, ledg_p2, redg_p2, redg_p1], dtype=np.int32)
        cv2.fillPoly(polygon_mask, [polygon_pts], 1)
    except Exception:
        return 0.0
    return calculate_iou(pred_mask, polygon_mask)

def signed_distance_to_line(point, line_p1, line_p2):
    """Calculate signed distance from point to line"""
    x0, y0 = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    # Cross product gives signed area (2x)
    cross = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    if line_length < 1e-6:
        return 0
    
    return cross / line_length

def calculate_boolean_score(pred_coords, tolerance=10):
    """
    Binary boolean score calculation that returns STRICTLY 0 or 1 ONLY
    Returns 1 if the runway configuration is valid, 0 otherwise
    """
    try:
        ledg_p1, ledg_p2 = pred_coords[0:2], pred_coords[2:4]
        redg_p1, redg_p2 = pred_coords[4:6], pred_coords[6:8]
        ctl_p1, ctl_p2 = pred_coords[8:10], pred_coords[10:12]
        
        # Check if coordinates are valid (not all zeros or very close to zero)
        if np.allclose(pred_coords, 0, atol=1e-3):
            return 0
        
        # Check if we have reasonable line segments
        left_length = np.linalg.norm(ledg_p2 - ledg_p1)
        right_length = np.linalg.norm(redg_p2 - redg_p1)
        center_length = np.linalg.norm(ctl_p2 - ctl_p1)
        
        # Lines too short indicate poor detection
        if left_length < 5 or right_length < 5 or center_length < 5:
            return 0
        
        # Check both centerline points
        valid_points = 0
        required_points = 2  # Both centerline points must be valid
        
        for ctl_point in [ctl_p1, ctl_p2]:
            # Calculate signed distances to both edges
            dist_left = signed_distance_to_line(ctl_point, ledg_p1, ledg_p2)
            dist_right = signed_distance_to_line(ctl_point, redg_p1, redg_p2)
            
            # For a valid runway, centerline should be between edges
            # This means the signs should be opposite
            if abs(dist_left) < 1 and abs(dist_right) < 1:
                # Points are very close to lines, probably degenerate case
                continue
            elif dist_left * dist_right < 0:  # Different signs = between lines
                # Check if the point is reasonably well-positioned
                total_width = abs(dist_left) + abs(dist_right)
                if total_width > tolerance:
                    # Check if reasonably centered (not too close to one edge)
                    balance = abs(abs(dist_left) - abs(dist_right)) / total_width
                    if balance < 0.8:  # Not too unbalanced (80% towards one side)
                        valid_points += 1
                else:
                    # Narrow runway but still valid if between edges
                    valid_points += 1
        
        # Additional check: centerline should be roughly parallel to runway edges
        left_vector = ledg_p2 - ledg_p1
        right_vector = redg_p2 - redg_p1
        center_vector = ctl_p2 - ctl_p1
        
        def angle_similarity(v1, v2):
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                return 0
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return abs(cos_angle)  # We want parallel or anti-parallel
        
        left_sim = angle_similarity(center_vector, left_vector)
        right_sim = angle_similarity(center_vector, right_vector)
        avg_parallelism = (left_sim + right_sim) / 2
        
        # Final decision: all points must be valid and reasonably parallel
        if valid_points >= required_points and avg_parallelism > 0.5:  # cos(60°) = 0.5
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"Error in boolean score calculation: {e}")
        return 0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = A.Compose([
        A.Resize(height=MODEL_H, width=MODEL_W),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    model = RunwayDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    with open(TEST_LABELS_JSON_PATH, "r") as f:
        true_labels = json.load(f)
    test_image_ids = sorted(list(true_labels.keys()))

    results = []
    boolean_scores_debug = []  # For debugging
    
    for img_name in tqdm(test_image_ids, desc="Evaluating model"):
        img_path = os.path.join(TEST_IMG_DIR, os.path.splitext(img_name)[0] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(TEST_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = transform(image=image_rgb)
        input_tensor = transformed['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_tensor = model(input_tensor)

        pred_mask = torch.sigmoid(pred_mask_tensor).cpu().numpy().squeeze()
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8)
        pred_mask_resized = cv2.resize(pred_mask_binary, (SCORE_W, SCORE_H), interpolation=cv2.INTER_NEAREST)

        pred_coords = mask_to_coords(pred_mask_resized)

        true_coords = np.zeros(12, dtype=np.float32)
        annotations = true_labels.get(img_name, [])
        coord_map = {'LEDG': 0, 'REDG': 4, 'CTL': 8}
        for ann in annotations:
            label = ann.get("label")
            points = np.array(ann.get("points", [])).flatten()
            if label in coord_map and len(points) == 4:
                true_coords[coord_map[label]:coord_map[label]+4] = points
        true_mask = create_mask_from_coords(true_coords, SCORE_H, SCORE_W)

        iou_score = calculate_iou(true_mask, pred_mask_resized)
        anchor_score = calculate_anchor_score(pred_mask_resized, pred_coords)
        boolean_score = calculate_boolean_score(pred_coords)
        
        # EXPLICIT SAFEGUARD: Ensure boolean score is strictly 0 or 1
        boolean_score = 1 if boolean_score > 0.5 else 0
        
        boolean_scores_debug.append(boolean_score)  # For debugging

        results.append({
            "Image Name": img_name,
            "IOU score": iou_score,
            "Anchor Score": anchor_score,
            "Boolean score": boolean_score
        })

    submission_df = pd.DataFrame(results)
    mean_scores = submission_df.mean(numeric_only=True)
    mean_row = pd.DataFrame([{"Image Name": "Mean Score", **mean_scores}])
    submission_df = pd.concat([submission_df, mean_row], ignore_index=True)
    submission_df.to_csv(SUBMISSION_CSV_PATH, index=False)

    print(f"\n✅ Evaluation complete! Submission file saved to {SUBMISSION_CSV_PATH}")
    print("\n--- Final Scores ---")
    print(mean_row.to_string(index=False))
    
    # Debug info for boolean scores
    print(f"\n--- Boolean Score Debug Info ---")
    print(f"Number of valid detections (score=1): {sum(1 for x in boolean_scores_debug if x == 1)}")
    print(f"Number of invalid detections (score=0): {sum(1 for x in boolean_scores_debug if x == 0)}")
    print(f"Total images: {len(boolean_scores_debug)}")
    print(f"Success rate: {sum(boolean_scores_debug) / len(boolean_scores_debug):.3f}")