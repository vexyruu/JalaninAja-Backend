import asyncio
import os
import httpx
import polyline
import numpy as np
from supabase import create_client, Client
from arq.connections import RedisSettings
from ultralytics import YOLO
from dotenv import load_dotenv, find_dotenv
from shared.analysis import calculate_walkability_score, run_model_on_image, is_road_residential
from threading import Thread
from fastapi import FastAPI
import uvicorn
from typing import List

load_dotenv(find_dotenv())
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY, REDIS_URL]):
    raise RuntimeError("Worker environment variables are missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_model():
    local_model_path = "best.pt"
    if os.path.exists(local_model_path):
        print(f"Loading model from local path: {local_model_path}")
        return YOLO(local_model_path)
    print(f"Local model not found at {local_model_path}. Analysis features will be disabled.")
    return None

model = load_model()

async def analyze_report_photo(ctx, report_data: dict):
    if not model or not report_data.get('photo_url'):
        return

    photo_url = report_data['photo_url']
    print(f"Starting background analysis for photo: {photo_url}")
    try:
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(model, photo_url)
        is_residential = await is_road_residential(report_data['latitude'], report_data['longitude'])
        new_score = calculate_walkability_score(
            detected_labels, 
            sidewalk_area, 
            tree_count, 
            is_residential
        )

        cache_key_lat = round(report_data['latitude'], 5)
        cache_key_lng = round(report_data['longitude'], 5)
        
        print(f"Updating cache with new score ({new_score}) from report at ({cache_key_lat}, {cache_key_lng})")
        cache_data = {
            "latitude": cache_key_lat, "longitude": cache_key_lng,
            "walkability_score": float(new_score), "photo_url": photo_url,
            "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
            "is_residential_road": is_residential, "heading": None,
            "detected_labels": detected_labels
        }
        
        supabase.table('RoutePointCache').upsert(
            cache_data, 
            on_conflict='latitude, longitude'
        ).execute()
        print(f"Cache updated successfully for report photo.")

    except Exception as e:
        print(f"ERROR: Failed to analyze report photo in background: {e}")

async def analyze_point(lat, lng, mode: str):
    cache_key_lat, cache_key_lng = round(lat, 5), round(lng, 5)
    try:
        cached_point_query = supabase.table('RoutePointCache').select('*').eq('latitude', cache_key_lat).eq('longitude', cache_key_lng).limit(1).execute()
        if cached_point_query.data:
            return cached_point_query.data[0]
    except Exception as e:
        print(f"Failed to check point cache: {e}")

    MAX_ATTEMPTS = 4
    for attempt in range(MAX_ATTEMPTS):
        search_lat = lat + np.random.uniform(-0.0001, 0.0001) if attempt > 0 else lat
        search_lng = lng + np.random.uniform(-0.0001, 0.0001) if attempt > 0 else lng
        heading = np.random.randint(0, 360)
        photo_url = (f"https://maps.googleapis.com/maps/api/streetview?"
                     f"size=400x400&location={search_lat},{search_lng}"
                     f"&fov=90&heading={heading}&pitch=10&key={GOOGLE_MAPS_API_KEY}")
        
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(model, photo_url)
        
        if not detected_labels and attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(1)
            continue

        is_residential = await is_road_residential(search_lat, search_lng)
        final_score = calculate_walkability_score(detected_labels, sidewalk_area, tree_count, is_residential, mode)
        
        point_data_to_cache = {
            "latitude": cache_key_lat, "longitude": cache_key_lng, "walkability_score": float(final_score),
            "photo_url": photo_url, "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
            "is_residential_road": is_residential, "heading": heading, "detected_labels": detected_labels
        }
        try:
            supabase.table('RoutePointCache').upsert(point_data_to_cache, on_conflict='latitude, longitude').execute()
        except Exception as e:
            print(f"Failed to save point to cache: {e}")
        return point_data_to_cache
    return None

async def analyze_single_route(route_data: dict, mode: str) -> dict | None:
    polyline_str = route_data['overview_polyline']['points']
    route_coords = polyline.decode(polyline_str)
    if not route_coords: return None
    distance_meters = route_data['legs'][0]['distance']['value']
    num_points = max(3, min(20, int(distance_meters / 250)))
    indices = np.linspace(0, len(route_coords) - 1, num_points, dtype=int)
    points_to_analyze = [route_coords[i] for i in indices]
    analysis_tasks = [analyze_point(lat, lng, mode) for lat, lng in points_to_analyze]
    analysis_results = await asyncio.gather(*analysis_tasks)
    valid_results = [res for res in analysis_results if res is not None and res.get('detected_labels')]
    if not valid_results: return None
    average_score = sum(p['walkability_score'] for p in valid_results) / len(valid_results)
    return {"average_walkability_score": round(average_score, 2), "points_analyzed": valid_results, "overview_polyline": polyline_str}

async def analyze_and_save_routes(ctx, job_id: str, origin: str, dest: str, mode: str):
    print(f"Processing job {job_id}: from {origin} to {dest}")
    try:
        supabase.table("RouteSearch").update({"status": "processing"}).eq("job_id", job_id).execute()
        async with httpx.AsyncClient() as client:
            geocode_url_origin = f"https://maps.googleapis.com/maps/api/geocode/json?address={origin}&key={GOOGLE_MAPS_API_KEY}"
            geocode_url_dest = f"https://maps.googleapis.com/maps/api/geocode/json?address={dest}&key={GOOGLE_MAPS_API_KEY}"
            origin_res, dest_res = await asyncio.gather(client.get(geocode_url_origin), client.get(geocode_url_dest))
            origin_coords = origin_res.json()['results'][0]['geometry']['location']
            dest_coords = dest_res.json()['results'][0]['geometry']['location']
            directions_url = (f"https://maps.googleapis.com/maps/api/directions/json?"
                              f"origin={origin_coords['lat']},{origin_coords['lng']}"
                              f"&destination={dest_coords['lat']},{dest_coords['lng']}"
                              f"&mode=walking&alternatives=true&key={GOOGLE_MAPS_API_KEY}")
            directions_res = await client.get(directions_url)
            alternative_routes_data = directions_res.json().get('routes', [])
            if not alternative_routes_data: raise ValueError("No walking routes found.")
            analysis_tasks = [analyze_single_route(route, mode) for route in alternative_routes_data]
            all_route_results = await asyncio.gather(*analysis_tasks)
            valid_results = [res for res in all_route_results if res is not None]
            if not valid_results: raise ValueError("Failed to analyze any valid routes.")
        supabase.table("RouteSearch").update({"status": "completed", "results": valid_results}).eq("job_id", job_id).execute()
        print(f"Job {job_id} completed successfully.")
    except Exception as e:
        print(f"Job {job_id} failed: {e}")
        supabase.table("RouteSearch").update({"status": "failed", "results": {"error": str(e)}}).eq("job_id", job_id).execute()

health_app = FastAPI()
@health_app.get("/healthz")
def health_check():
    return {"status": "ok"}

def run_health_server():
    uvicorn.run(health_app, host="0.0.0.0", port=8080)
redis_settings = RedisSettings.from_dsn(REDIS_URL)

class WorkerSettings:
    functions = [
        analyze_and_save_routes, 
        analyze_report_photo 
    ]
    redis_settings = redis_settings

health_server_thread = Thread(target=run_health_server)
health_server_thread.daemon = True 
health_server_thread.start()
