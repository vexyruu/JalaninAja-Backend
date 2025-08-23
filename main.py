from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
import asyncio
import os
from supabase import create_client, Client
import polyline
from typing import Optional, List
import uuid
import httpx
from dotenv import load_dotenv
import requests
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

load_dotenv()

app = FastAPI(
    title="JalaninAja API",
    description="API for calculating routes, gamification, and authentication.",
    version="27.0"
)

# --- Secret and Configuration Management ---
def get_secret(secret_name):
    return None

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") or get_secret("GMapAPIKey")
SUPABASE_URL = os.getenv("SUPABASE_URL") or get_secret("SupabaseURL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or get_secret("SupabaseKey")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or get_secret("SupabaseAnonKey")

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY, SUPABASE_ANON_KEY]):
    raise RuntimeError("One or more environment variables are missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- AI Model Loading ---
def load_model():
    local_model_path = "best.pt"
    if os.path.exists(local_model_path):
        print(f"Loading model from local path: {local_model_path}")
        return YOLO(local_model_path)
    print(f"Local model not found at {local_model_path}. Route analysis features will be disabled.")
    return None

model = load_model()

# --- Pydantic Data Models ---

class AppConfig(BaseModel):
    supabase_url: str
    supabase_anon_key: str
    google_maps_api_key: str

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class ReportRequest(BaseModel):
    user_id: str
    category: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    photo_url: Optional[str] = None

class UserProfileResponse(BaseModel):
    user_name: str
    user_email: str
    user_avatar_url: Optional[str] = None
    user_level: str
    user_points: int
    reports_made: int
    badges: List[str]

class PointDetail(BaseModel):
    point_cache_id: Optional[int] = None
    latitude: float
    longitude: float
    walkability_score: float
    photo_url: str
    tree_count: Optional[int] = None
    sidewalk_area: Optional[float] = None
    is_residential_road: Optional[bool] = None
    heading: Optional[int] = None
    detected_labels: Optional[List[str]] = None

class RouteAlternative(BaseModel):
    route_id: int
    average_walkability_score: float
    points_analyzed: list[PointDetail]
    overview_polyline: str

class RouteComparisonResponse(BaseModel):
    status: str
    alternatives: list[RouteAlternative]
    source: str
    
class RouteRequest(BaseModel):
    origin_address: str
    destination_address: str
    mode: str = "distance_walkability"

class AutocompleteResponse(BaseModel):
    predictions: list[dict]

# --- Configuration Endpoint ---
@app.get("/config", response_model=AppConfig)
def get_app_configuration():
    return AppConfig(
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
    )

# --- User and Profile Endpoints ---

@app.patch("/users/me")
async def update_current_user(request: UserUpdateRequest, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")

        update_data_auth = {}
        update_data_public = {}

        if request.name:
            update_data_auth['full_name'] = request.name
        if request.avatar_url is not None:
            update_data_auth['avatar_url'] = request.avatar_url
        
        if request.name:
            update_data_public['name'] = request.name
        if request.avatar_url is not None:
            update_data_public['avatar_url'] = request.avatar_url

        if update_data_auth:
            supabase.auth.admin.update_user_by_id(
                user.id, {'data': update_data_auth}
            )

        if update_data_public:
            supabase.table("Users").update(update_data_public).eq("user_id_auth", user.id).execute()

        return {"status": "success", "message": "Profile updated successfully."}
    except Exception as e:
        print(f"Failed to update user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user profile.")

@app.get("/users/{user_id_auth}", response_model=UserProfileResponse)
async def get_user_profile(user_id_auth: str):
    try:
        user_query = supabase.table('Users').select('user_id, name, points, email, avatar_url').eq('user_id_auth', user_id_auth).single().execute()
        
        user_data = user_query.data
        internal_user_id = user_data['user_id']
        points = user_data.get('points', 0)
        avatar_url = user_data.get('avatar_url')

        report_count_query = supabase.table('Report').select('report_id', count='exact').eq('user_id', internal_user_id).execute()
        reports_made = report_count_query.count if report_count_query.count is not None else 0

        user_level = "Pejalan Kaki"
        if points >= 1500: user_level = "Legenda Gotong Royong"
        elif points >= 750: user_level = "Jawara Jalan"
        elif points >= 300: user_level = "Pelopor Trotoar"
        elif points >= 100: user_level = "Penjelajah Kota"


        badges = []
        if reports_made >= 10: badges.append("Pelapor Rajin")
        if reports_made >= 5: badges.append("Pembantu Sesama")

        return UserProfileResponse(
            user_name=user_data.get('name', 'N/A'),
            user_email=user_data.get('email', 'N/A'),
            user_avatar_url=avatar_url,
            user_level=user_level,
            user_points=points,
            reports_made=reports_made,
            badges=badges
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve profile data: {e}")

# --- Helper Functions for AI and Scoring ---

def calculate_walkability_score(detected_labels: List[str], tree_count: int, is_residential: bool, mode: str = "distance_walkability") -> float:
    score = 50.0
    if 'sidewalk' in detected_labels:
        score += 40
    elif is_residential:
        score -= 10
    else:
        score -= 30
    if mode == 'shady_route':
        score += tree_count * 5
    final_score = max(0, min(100, score))
    return final_score

async def run_model_on_image(photo_url: str) -> tuple[float, int, List[str]]:
    if model is None:
        return 0.0, 0, []
    
    sidewalk_area_percent, tree_count, detected_labels = 0.0, 0, []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(photo_url, timeout=15)
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
                if class_name: detected_labels.append(class_name)
                if class_idx == sidewalk_idx: total_sidewalk_pixels += mask.sum()
                elif class_idx == tree_idx: tree_count += 1
            
            if image.width * image.height > 0:
                sidewalk_area_percent = float(total_sidewalk_pixels / (image.width * image.height))
    except Exception as e:
        print(f"Failed to process image: {e}")
    return sidewalk_area_percent, tree_count, list(set(detected_labels))

async def is_road_residential(lat: float, lng: float) -> bool:
    return False

async def process_and_cache_report_photo(report: ReportRequest):
    if not model or not report.photo_url:
        return

    print(f"Starting background analysis for photo: {report.photo_url}")
    try:
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(report.photo_url)
        
        is_residential = await is_road_residential(report.latitude, report.longitude)
        new_score = calculate_walkability_score(detected_labels, tree_count, is_residential, mode="distance_walkability")

        cache_key_lat = round(report.latitude, 5)
        cache_key_lng = round(report.longitude, 5)
        
        existing_cache_query = supabase.table('RoutePointCache').select('walkability_score').eq('latitude', cache_key_lat).eq('longitude', cache_key_lng).limit(1).execute()
        
        existing_score = -1.0
        if existing_cache_query.data:
            existing_score = existing_cache_query.data[0].get('walkability_score', -1.0)

        if new_score > existing_score:
            print(f"New score ({new_score}) is better than old score ({existing_score}). Updating cache.")
            cache_data = {
                "latitude": cache_key_lat, "longitude": cache_key_lng,
                "walkability_score": float(new_score), "photo_url": report.photo_url,
                "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
                "is_residential_road": is_residential, "heading": None,
                "detected_labels": detected_labels
            }
            
            supabase.table('RoutePointCache').upsert(
                cache_data, 
                on_conflict='latitude, longitude'
            ).execute()
            print(f"Cache updated from report at location ({report.latitude}, {report.longitude})")
        else:
            print(f"New score ({new_score}) is not better than old score ({existing_score}). Cache not changed.")

    except Exception as e:
        print(f"Failed to analyze report photo in background: {e}")

# --- Report and Community Endpoints ---

@app.post("/upload-photo")
async def upload_photo(file: UploadFile = File(...)):
    try:
        file_extension = os.path.splitext(file.filename)[1]
        file_name = f"report_{uuid.uuid4()}{file_extension}"
        contents = await file.read()
        
        supabase.storage.from_("report-photos").upload(
            file=contents,
            path=file_name,
            file_options={"content-type": file.content_type}
        )
        
        public_url = supabase.storage.from_("report-photos").get_public_url(file_name)
        return {"status": "success", "photo_url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")

@app.get("/reports")
async def get_all_reports():
    try:
        reports_query = supabase.table('Report').select('*, Users(name, avatar_url, points)').order('created_at', desc=True).execute()
        return reports_query.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve reports.")


@app.get("/leaderboard")
async def get_leaderboard():
    try:
        leaderboard_query = supabase.table('Users').select('name, points, avatar_url').order('points', desc=True).limit(100).execute()
        return leaderboard_query.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard.")

@app.get("/reports/user/{user_id_auth}")
async def get_user_top_reports(user_id_auth: str):
    try:
        user_query = supabase.table('Users').select('user_id').eq('user_id_auth', user_id_auth).single().execute()
        internal_user_id = user_query.data['user_id']
        reports_query = supabase.table('Report').select('*').eq('user_id', internal_user_id).order('upvote_count', desc=True).limit(3).execute()
        return reports_query.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user reports: {e}")

async def get_address_from_coords(lat: float, lng: float) -> str:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'OK' and len(data['results']) > 0:
                return data['results'][0]['formatted_address']
        except Exception as e:
            print(f"Reverse geocode failed: {e}")
    return f"Lat: {lat:.5f}, Lng: {lng:.5f}"

@app.post("/reports")
async def create_report(report: ReportRequest):
    try:
        user_profile = supabase.table('Users').select('user_id, points').eq('user_id_auth', report.user_id).single().execute()
        internal_user_id = user_profile.data['user_id']
        current_points = user_profile.data['points']

        address = await get_address_from_coords(report.latitude, report.longitude)

        report_data = {
            "user_id": internal_user_id, "description": report.description,
            "latitude": report.latitude, "longitude": report.longitude,
            "category": report.category, "photo_url": report.photo_url,
            "address": address
        }
        supabase.table('Report').insert(report_data).execute()

        new_points = current_points + 10
        supabase.table('Users').update({'points': new_points}).eq('user_id', internal_user_id).execute()
        
        if report.photo_url:
            asyncio.create_task(process_and_cache_report_photo(report))

        return {"status": "success", "message": "Report submitted! +10 Points."}
    except Exception as e:
        print(f"Failed to save report: {e}")
        raise HTTPException(status_code=500, detail="Failed to save report.")

@app.post("/reports/{report_id}/upvote")
async def upvote_report(report_id: int, token: str = Depends(oauth2_scheme)):
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid Token")
        
        user_profile = supabase.table('Users').select('user_id').eq('user_id_auth', user.id).single().execute()
        internal_user_id = user_profile.data['user_id']

        report_owner_profile = supabase.table('Report').select('Users(user_id, points)').eq('report_id', report_id).single().execute()
        owner_internal_id = report_owner_profile.data['Users']['user_id']
        owner_current_points = report_owner_profile.data['Users']['points']

        if internal_user_id == owner_internal_id:
            raise HTTPException(status_code=403, detail="You cannot upvote your own report.")

        supabase.table('Upvotes').insert({'report_id': report_id, 'user_id': internal_user_id}).execute()

        report = supabase.table('Report').select('upvote_count').eq('report_id', report_id).single().execute()
        new_upvote_count = report.data['upvote_count'] + 1
        supabase.table('Report').update({'upvote_count': new_upvote_count}).eq('report_id', report_id).execute()

        new_points_for_owner = owner_current_points + 5
        supabase.table('Users').update({'points': new_points_for_owner}).eq('user_id', owner_internal_id).execute()

        return {"status": "success", "message": "Upvote successful!"}
    except Exception as e:
        if 'unique_user_report_upvote' in str(e):
            raise HTTPException(status_code=409, detail="You have already upvoted this report.")
        raise HTTPException(status_code=500, detail="Failed to process upvote.")

# --- Route Calculation and AI Analysis Endpoints ---

def geocode_address(address: str) -> dict | None:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200 and response.json()['status'] == 'OK':
        return response.json()['results'][0]['geometry']['location']
    return None

def get_alternative_routes_from_google(origin_coords, dest_coords):
    url = (f"https://maps.googleapis.com/maps/api/directions/json?"
           f"origin={origin_coords['lat']},{origin_coords['lng']}"
           f"&destination={dest_coords['lat']},{dest_coords['lng']}"
           f"&mode=walking&alternatives=true&key={GOOGLE_MAPS_API_KEY}")
    response = requests.get(url)
    if response.status_code == 200 and response.json().get('routes'):
        return response.json()['routes']
    return []

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

        sidewalk_area, tree_count, detected_labels = await run_model_on_image(photo_url)
        
        if not detected_labels and attempt < MAX_ATTEMPTS - 1:
            continue

        is_residential = await is_road_residential(search_lat, search_lng)
        final_score = calculate_walkability_score(detected_labels, tree_count, is_residential, mode)

        point_data_to_cache = {
            "latitude": cache_key_lat, "longitude": cache_key_lng,
            "walkability_score": float(final_score), "photo_url": photo_url,
            "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
            "is_residential_road": is_residential, 
            "heading": heading, "detected_labels": detected_labels
        }
        
        try:
            saved_point = supabase.table('RoutePointCache').upsert(point_data_to_cache, on_conflict='latitude, longitude').execute()
            point_data_to_cache['point_cache_id'] = saved_point.data[0]['point_cache_id']
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
    
    return {
        "average_walkability_score": round(average_score, 2),
        "points_analyzed": valid_results,
        "overview_polyline": polyline_str
    }

async def save_analysis_to_db(origin: str, dest: str, mode: str, alternatives: list[dict]):
    try:
        search_entry = supabase.table('RouteSearch').insert({
            'start_point': origin, 'end_point': dest, 'mode': mode
        }).execute()
        search_id = search_entry.data[0]['search_id']

        for alt in alternatives:
            point_ids = [p['point_cache_id'] for p in alt['points_analyzed'] if p.get('point_cache_id')]
            supabase.table('RouteAlternative').insert({
                'search_id': search_id,
                'average_walkability_score': alt['average_walkability_score'],
                'overview_polyline': alt['overview_polyline'],
                'point_ids': point_ids
            }).execute()
    except Exception as e:
        print(f"Failed to save route analysis to DB: {e}")

async def get_cached_analysis(origin: str, dest: str, mode: str) -> list[RouteAlternative] | None:
    try:
        search_query = supabase.table('RouteSearch').select('search_id').eq('start_point', origin).eq('end_point', dest).eq('mode', mode).order('created_at', desc=True).limit(1).execute()
        if not search_query.data: return None
        search_id = search_query.data[0]['search_id']
        
        alt_query = supabase.table('RouteAlternative').select('*').eq('search_id', search_id).execute()
        if not alt_query.data: return None

        final_alternatives = []
        for alt_data in alt_query.data:
            point_ids = alt_data.get('point_ids', [])
            if not point_ids: continue
            points_query = supabase.table('RoutePointCache').select('*').in_('point_cache_id', point_ids).execute()
            points_map = {p['point_cache_id']: p for p in points_query.data}
            ordered_points = [PointDetail(**points_map[pid]) for pid in point_ids if pid in points_map]
            final_alternatives.append(RouteAlternative(
                route_id=alt_data['route_alt_id'],
                average_walkability_score=alt_data['average_walkability_score'],
                points_analyzed=ordered_points,
                overview_polyline=alt_data['overview_polyline']
            ))
        return final_alternatives
    except Exception as e:
        print(f"Failed to retrieve from cache: {e}")
        return None

@app.post("/calculate-routes", response_model=RouteComparisonResponse)
async def calculate_route_alternatives(request: RouteRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="AI model is not available.")
    
    cached_results = await get_cached_analysis(request.origin_address, request.destination_address, request.mode)
    if cached_results:
        return RouteComparisonResponse(status="success", alternatives=cached_results, source="cache")

    origin_coords = geocode_address(request.origin_address)
    dest_coords = geocode_address(request.destination_address)
    if not origin_coords or not dest_coords:
        raise HTTPException(status_code=404, detail="One or both addresses could not be found.")
    
    alternative_routes_data = get_alternative_routes_from_google(origin_coords, dest_coords)
    if not alternative_routes_data:
        raise HTTPException(status_code=404, detail="No walking routes found.")
    
    analysis_tasks = [analyze_single_route(route, request.mode) for route in alternative_routes_data]
    all_route_results = await asyncio.gather(*analysis_tasks)
    
    valid_results = [res for res in all_route_results if res is not None]
    if not valid_results:
        raise HTTPException(status_code=500, detail="Failed to analyze routes.")

    asyncio.create_task(save_analysis_to_db(request.origin_address, request.destination_address, request.mode, valid_results))

    final_alternatives = [RouteAlternative(
        route_id=i + 1,
        average_walkability_score=result['average_walkability_score'],
        points_analyzed=[PointDetail(**p) for p in result['points_analyzed']],
        overview_polyline=result['overview_polyline']
    ) for i, result in enumerate(valid_results)]

    return RouteComparisonResponse(status="success", alternatives=final_alternatives, source="live")

@app.get("/autocomplete-address", response_model=AutocompleteResponse)
async def autocomplete_address(query: str):
    url = (f"https://maps.googleapis.com/maps/api/place/autocomplete/json?"
           f"input={query}&key={GOOGLE_MAPS_API_KEY}&location=-7.2575,112.7521&radius=50000&strictbounds=true")
    response = requests.get(url)
    if response.status_code == 200:
        return AutocompleteResponse(predictions=response.json().get('predictions', []))
    raise HTTPException(status_code=500, detail="Failed to fetch address suggestions.")

@app.get("/reverse-geocode")
async def get_readable_address(lat: float, lng: float):
    address = await get_address_from_coords(lat, lng)
    return {"address": address}

@app.get("/")
def read_root():
    return {"message": "Welcome to the JalaninAja API!"}