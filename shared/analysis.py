import httpx
import io
from PIL import Image
from typing import List
from ultralytics import YOLO

def calculate_walkability_score(
    detected_labels: List[str],
    sidewalk_area_percent: float,
    tree_count: int,
    is_residential: bool,
    mode: str = "distance_walkability"
) -> float:
    score = 50.0
    if 'sidewalk' in detected_labels:
        score += 20
        dynamic_bonus = min(sidewalk_area_percent * 100, 25)
        score += dynamic_bonus
    
    if mode == 'shady_route':
        score += min(tree_count * 5, 20)

    if 'sidewalk' not in detected_labels:
        if is_residential:
            score -= 15  
        else:
            score -= 30
    final_score = max(0, min(100, score))
    return final_score

async def run_model_on_image(model: YOLO, photo_url: str) -> tuple[float, int, List[str]]:
    if model is None:
        print("Warning: AI model is not loaded. Cannot run analysis.")
        return 0.0, 0, []
    
    sidewalk_area_percent, tree_count, detected_labels = 0.0, 0, []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(photo_url, timeout=20)
            response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        results = model(image)
        
        if results and results[0].masks is not None:
            class_names = results[0].names
            sidewalk_idx = next((k for k, v in class_names.items() if v == 'sidewalk'), None)
            tree_idx = next((k for k, v in class_names.items() if v == 'tree'), None)
            total_sidewalk_pixels = 0
            
            for i, mask in enumerate(results[0].masks.data):
                class_idx = int(results[0].boxes.cls[i])
                class_name = class_names.get(class_idx)
                if class_name:
                    detected_labels.append(class_name)
                if class_idx == sidewalk_idx:
                    total_sidewalk_pixels += mask.sum()
                elif class_idx == tree_idx:
                    tree_count += 1
            
            if image.width * image.height > 0:
                sidewalk_area_percent = float(total_sidewalk_pixels / (image.width * image.height))

    except httpx.HTTPStatusError as e:
        print(f"Failed to download image: {photo_url}, Status: {e.response.status_code}")
    except Exception as e:
        print(f"Failed to process image ({photo_url}): {e}")
        
    return sidewalk_area_percent, tree_count, list(set(detected_labels))

async def is_road_residential(lat: float, lng: float) -> bool:
    return False

