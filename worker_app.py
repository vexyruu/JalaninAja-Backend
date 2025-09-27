import os
import asyncio
import httpx
import polyline
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from ultralytics import YOLO
from supabase import create_client, Client

# Your shared analysis functions must be in a `shared` directory
from shared.analysis import calculate_walkability_score, run_model_on_image, is_road_residential

# --- Load Environment Variables ---
load_dotenv(find_dotenv())
app = FastAPI(title="JalaninAja Background Worker")

# --- Service Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY]):
    raise RuntimeError("Worker environment variables are missing.")

# --- Client Initializations ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- App Lifecycle: Load Model on Startup ---
@app.on_event("startup")
def startup_event():
    local_model_path = "best.onnx"
    if os.path.exists(local_model_path):
        app.state.model = YOLO(local_model_path, task='segment')
        print("✅ Worker model loaded successfully.")
    else:
        app.state.model = None
        print("🔥 FATAL: Worker model file not found. All tasks will fail.")

# --- Pydantic Models for Task Payloads ---
class RouteTaskPayload(BaseModel):
    job_id: str
    origin_address: str
    destination_address: str
    mode: str

class PhotoTaskPayload(BaseModel):
    report_data: dict

# --- Ported Worker Logic from original worker.py ---

async def analyze_report_photo(model: YOLO, report_data: dict):
    if not model or not report_data.get('photo_url'):
        print("Skipping photo analysis: model or photo_url missing.")
        return

    photo_url = report_data['photo_url']
    print(f"Analyzing photo: {photo_url}")
    try:
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(model, photo_url)
        is_residential = await is_road_residential(report_data['latitude'], report_data['longitude'])
        new_score = calculate_walkability_score(detected_labels, sidewalk_area, tree_count, is_residential)

        cache_key_lat, cache_key_lng = round(report_data['latitude'], 5), round(report_data['longitude'], 5)
        
        cache_data = {
            "latitude": cache_key_lat, "longitude": cache_key_lng,
            "walkability_score": float(new_score), "photo_url": photo_url,
            "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
            "is_residential_road": is_residential, "heading": None,
            "detected_labels": detected_labels
        }
        
        supabase.table('RoutePointCache').upsert(cache_data, on_conflict='latitude, longitude').execute()
        print(f"Cache updated from report at ({cache_key_lat}, {cache_key_lng})")
    except Exception as e:
        print(f"ERROR: Failed to analyze report photo: {e}")

async def analyze_point(model: YOLO, lat: float, lng: float, mode: str):
    cache_key_lat, cache_key_lng = round(lat, 5), round(lng, 5)
    try:
        cached = supabase.table('RoutePointCache').select('*').eq('latitude', cache_key_lat).eq('longitude', cache_key_lng).limit(1).execute()
        if cached.data: return cached.data[0]
    except Exception as e: print(f"Cache check failed: {e}")

    MAX_ATTEMPTS = 4
    for attempt in range(MAX_ATTEMPTS):
        search_lat = lat + np.random.uniform(-0.0001, 0.0001) if attempt > 0 else lat
        search_lng = lng + np.random.uniform(-0.0001, 0.0001) if attempt > 0 else lng
        heading = np.random.randint(0, 360)
        photo_url = f"https://maps.googleapis.com/maps/api/streetview?size=400x400&location={search_lat},{search_lng}&fov=90&heading={heading}&pitch=10&key={GOOGLE_MAPS_API_KEY}"
        
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(model, photo_url)
        
        if not detected_labels and attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(1)
            continue

        is_residential = await is_road_residential(search_lat, search_lng)
        final_score = calculate_walkability_score(detected_labels, sidewalk_area, tree_count, is_residential, mode)
        
        point_data = {"latitude": cache_key_lat, "longitude": cache_key_lng, "walkability_score": float(final_score), "photo_url": photo_url, "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2), "is_residential_road": is_residential, "heading": heading, "detected_labels": detected_labels}
        try:
            supabase.table('RoutePointCache').upsert(point_data, on_conflict='latitude, longitude').execute()
        except Exception as e: print(f"Cache save failed: {e}")
        return point_data
    return None

async def analyze_single_route(model: YOLO, route_data: dict, mode: str):
    polyline_str = route_data['overview_polyline']['points']
    route_coords = polyline.decode(polyline_str)
    if not route_coords: return None
    distance_meters = route_data['legs'][0]['distance']['value']
    num_points = max(3, min(20, int(distance_meters / 250)))
    indices = np.linspace(0, len(route_coords) - 1, num_points, dtype=int)
    
    analysis_tasks = [analyze_point(model, lat, lng, mode) for lat, lng in [route_coords[i] for i in indices]]
    analysis_results = await asyncio.gather(*analysis_tasks)
    
    valid_results = [res for res in analysis_results if res is not None and res.get('detected_labels')]
    if not valid_results: return None
    
    average_score = sum(p['walkability_score'] for p in valid_results) / len(valid_results)
    return {"average_walkability_score": round(average_score, 2), "points_analyzed": valid_results, "overview_polyline": polyline_str}

async def execute_route_analysis(model: YOLO, job_id: str, origin: str, dest: str, mode: str):
    print(f"Processing job {job_id}: from {origin} to {dest}")
    try:
        supabase.table("RouteSearch").update({"status": "processing"}).eq("job_id", job_id).execute()
        async with httpx.AsyncClient() as client:
            origin_res, dest_res = await asyncio.gather(
                client.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={origin}&key={GOOGLE_MAPS_API_KEY}"),
                client.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={dest}&key={GOOGLE_MAPS_API_KEY}")
            )
            origin_coords = origin_res.json()['results'][0]['geometry']['location']
            dest_coords = dest_res.json()['results'][0]['geometry']['location']
            directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin_coords['lat']},{origin_coords['lng']}&destination={dest_coords['lat']},{dest_coords['lng']}&mode=walking&alternatives=true&key={GOOGLE_MAPS_API_KEY}"
            directions_res = await client.get(directions_url)
            routes_data = directions_res.json().get('routes', [])
            if not routes_data: raise ValueError("No walking routes found.")
            
            analysis_tasks = [analyze_single_route(model, route, mode) for route in routes_data]
            all_results = await asyncio.gather(*analysis_tasks)
            valid_results = [res for res in all_results if res is not None]
            if not valid_results: raise ValueError("Failed to analyze any valid routes.")

        supabase.table("RouteSearch").update({"status": "completed", "results": valid_results}).eq("job_id", job_id).execute()
        print(f"✅ Job {job_id} completed successfully.")
    except Exception as e:
        print(f"🔥 Job {job_id} failed: {e}")
        supabase.table("RouteSearch").update({"status": "failed", "results": {"error": str(e)}}).eq("job_id", job_id).execute()

# --- Worker Endpoints (Called by Google Cloud Tasks) ---
# Google Cloud Tasks sends a special header for authentication.
# We ensure this header is present on all incoming requests to secure the worker.
api_key_header = APIKeyHeader(name="X-CloudTasks-QueueName", auto_error=True)

@app.post("/process-route-task")
async def process_route_task(payload: RouteTaskPayload, request: Request, queue_name: str = Security(api_key_header)):
    model = request.app.state.model
    if not model:
        supabase.table("RouteSearch").update({"status": "failed", "results": {"error": "AI model not available in worker."}}).eq("job_id", payload.job_id).execute()
        raise HTTPException(status_code=503, detail="Model not loaded.")

    await execute_route_analysis(model, payload.job_id, payload.origin_address, payload.destination_address, payload.mode)
    return {"status": "acknowledged"}

@app.post("/process-photo-task")
async def process_photo_task(payload: PhotoTaskPayload, request: Request, queue_name: str = Security(api_key_header)):
    model = request.app.state.model
    if not model:
        print("Cannot process photo task, model not loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    await analyze_report_photo(model, payload.report_data)
    return {"status": "acknowledged"}

@app.get("/health")
def health_check():
    # A simple health check endpoint for Cloud Run to verify the instance is running.
    return {"status": "ok", "model_loaded": app.state.model is not None}
