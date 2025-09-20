from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import os
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest import APIError
import uuid
import httpx
from dotenv import load_dotenv
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO
import json
from google.cloud import secretmanager
import torch

load_dotenv()
from dotenv import load_dotenv, find_dotenv
from arq import create_pool
from arq.connections import RedisSettings
from shared.analysis import calculate_walkability_score, run_model_on_image, is_road_residential
load_dotenv(find_dotenv())

app = FastAPI(
    title="JalaninAja API",
    description="API for calculating routes, gamification, and authentication.",
    version="2.0"
    version="3"
)

# --- Secret and Configuration Management ---
def get_secret_from_manager(project_id, secret_name):
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"[ERROR] Failed to access secret '{secret_name}': {e}")
        return None

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

if GOOGLE_CLOUD_PROJECT:
    GOOGLE_MAPS_API_KEY = get_secret_from_manager(GOOGLE_CLOUD_PROJECT, "GMapAPIKey")
    SUPABASE_URL        = get_secret_from_manager(GOOGLE_CLOUD_PROJECT, "SupabaseURL")
    SUPABASE_KEY        = get_secret_from_manager(GOOGLE_CLOUD_PROJECT, "SupabaseKey")
    SUPABASE_ANON_KEY   = get_secret_from_manager(GOOGLE_CLOUD_PROJECT, "SupabaseAnonKey")
else:
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    SUPABASE_URL        = os.getenv("SUPABASE_URL")
    SUPABASE_KEY        = os.getenv("SUPABASE_KEY")
    SUPABASE_ANON_KEY   = os.getenv("SUPABASE_ANON_KEY")

print("DEBUG STARTUP SECRETS:")
print(f"  GOOGLE_CLOUD_PROJECT: {GOOGLE_CLOUD_PROJECT}")
print(f"  GOOGLE_MAPS_API_KEY: {'OK' if GOOGLE_MAPS_API_KEY else 'MISSING'}")
print(f"  SUPABASE_URL:        {'OK' if SUPABASE_URL else 'MISSING'}")
print(f"  SUPABASE_KEY:        {'OK' if SUPABASE_KEY else 'MISSING'}")
print(f"  SUPABASE_ANON_KEY:   {'OK' if SUPABASE_ANON_KEY else 'MISSING'}")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
REDIS_URL = os.getenv("REDIS_URL")

if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY, SUPABASE_ANON_KEY]):
    raise RuntimeError("❌ One or more critical environment variables or secrets are missing.")
if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY, SUPABASE_ANON_KEY, REDIS_URL]):
    raise RuntimeError("One or more critical environment variables are missing.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        user = supabase.auth.get_user(token).user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

redis_settings = RedisSettings.from_dsn(REDIS_URL)

@app.on_event("startup")
async def startup():
    app.state.redis = await create_pool(redis_settings)
    app.state.model = load_model()

@app.on_event("shutdown")
async def shutdown():
    await app.state.redis.close()

def load_model():
    local_model_path = "best.pt"
    if os.path.exists(local_model_path):
        print(f"Loading model from local path: {local_model_path}")
        model = YOLO(local_model_path)
        model.to('cpu')
        return model
    print(f"Local model not found at {local_model_path}. Route analysis features will be disabled.")
        from ultralytics import YOLO
        return YOLO(local_model_path)
    return None

model = load_model()

class ConfigResponse(BaseModel):
    supabase_url: str
    supabase_anon_key: str
    google_maps_api_key: str

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None

class ReportResponse(BaseModel):
    report_id: int
    user_id: str
    category: str
    description: Optional[str] = None
    latitude: float
    longitude: float
    photo_url: Optional[str] = None
    upvote_count: int
    address: Optional[str] = None
    Users: Optional[dict] = None
    
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
class ConfigResponse(BaseModel): supabase_url: str; supabase_anon_key: str; google_maps_api_key: str
class UserUpdateRequest(BaseModel): name: str | None = None; avatar_url: str | None = None
class UserProfileResponse(BaseModel): user_name: str; user_email: str; user_avatar_url: str | None = None; user_level: str; user_points: int; reports_made: int; badges: list[str]
class RouteRequest(BaseModel): origin_address: str; destination_address: str; mode: str = "distance_walkability"
class RouteCalculationStartedResponse(BaseModel): message: str; job_id: str
class RouteStatusResponse(BaseModel): status: str; data: list[dict] | None = None; error: str | None = None
class AutocompleteResponse(BaseModel): predictions: list[dict]

async def get_http_client():
    """Dependency to provide an httpx.AsyncClient session."""
    async with httpx.AsyncClient() as client:
        yield client

@app.get("/config", response_model=ConfigResponse)
def get_app_configuration():
    return ConfigResponse(
        supabase_url=SUPABASE_URL,
        supabase_anon_key=SUPABASE_ANON_KEY,
        google_maps_api_key=GOOGLE_MAPS_API_KEY,
    )

async def get_http_client():
    async with httpx.AsyncClient() as client: yield client

# --- Endpoints ---
@app.get("/config", response_model=ConfigResponse)
def get_app_configuration(): return ConfigResponse(supabase_url=SUPABASE_URL, supabase_anon_key=SUPABASE_ANON_KEY, google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.patch("/users/me")
async def update_current_user(request: UserUpdateRequest, user: dict = Depends(get_current_user)):
    try:
        update_data_public = {}
        if request.name: update_data_public['name'] = request.name
        if request.avatar_url is not None: update_data_public['avatar_url'] = request.avatar_url
        if update_data_public:
            supabase.table("Users").update(update_data_public).eq("user_id_auth", user.id).execute()
            supabase.auth.admin.update_user_by_id(user.id, {'data': {'full_name': request.name, 'avatar_url': request.avatar_url}})
        return {"status": "success", "message": "Profile updated successfully."}
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to update user profile.")

@app.get("/users/me", response_model=UserProfileResponse)
async def get_current_user_profile(user: dict = Depends(get_current_user)):
    try:
        user_query = supabase.table('Users').select('user_id, name, points, email, avatar_url').eq('user_id_auth', user.id).single().execute()
        user_data = user_query.data
        if not user_data: raise HTTPException(status_code=404, detail="User profile not found.")
        internal_user_id = user_data['user_id']
        points = user_data.get('points', 0)
        report_count_query = supabase.table('Report').select('report_id', count='exact').eq('user_id', internal_user_id).execute()
        reports_made = report_count_query.count or 0
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

async def process_and_cache_report_photo(report_data: dict):
    if not model or not report_data.get('photo_url'):
        return

    print(f"Starting background analysis for photo: {report_data['photo_url']}")
    try:
        sidewalk_area, tree_count, detected_labels = await run_model_on_image(report_data['photo_url'])
        
        is_residential = await is_road_residential(report_data['latitude'], report_data['longitude'])
        new_score = calculate_walkability_score(detected_labels, tree_count, is_residential, mode="distance_walkability")

        cache_key_lat = round(report_data['latitude'], 5)
        cache_key_lng = round(report_data['longitude'], 5)
        
        existing_cache_query = supabase.table('RoutePointCache').select('walkability_score').eq('latitude', cache_key_lat).eq('longitude', cache_key_lng).limit(1).execute()
        
        existing_score = -1.0
        if existing_cache_query.data:
            existing_score = existing_cache_query.data[0].get('walkability_score', -1.0)

        if new_score > existing_score:
            print(f"New score ({new_score}) is better than old score ({existing_score}). Updating cache.")
            cache_data = {
                "latitude": cache_key_lat, "longitude": cache_key_lng,
                "walkability_score": float(new_score), "photo_url": report_data['photo_url'],
                "tree_count": tree_count, "sidewalk_area": round(sidewalk_area * 100, 2),
                "is_residential_road": is_residential, "heading": None,
                "detected_labels": detected_labels
            }
            
            supabase.table('RoutePointCache').upsert(
                cache_data, 
                on_conflict='latitude, longitude'
            ).execute()
            print(f"Cache updated from report at location ({report_data['latitude']}, {report_data['longitude']})")
        else:
            print(f"New score ({new_score}) is not better than old score ({existing_score}). Cache not changed.")

    except Exception as e:
        print(f"Failed to analyze report photo in background: {e}")

        return UserProfileResponse(user_name=user_data.get('name', 'N/A'), user_email=user_data.get('email', 'N/A'), user_avatar_url=user_data.get('avatar_url'), user_level=user_level, user_points=points, reports_made=reports_made, badges=badges)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to retrieve profile data: {e}")

@app.get("/reports")
async def get_all_reports(offset: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
async def get_all_reports(offset: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
    try:
        reports_query = supabase.table('Report').select('*, Users(name, avatar_url, points)').order('created_at', desc=True).range(offset, offset + limit - 1).execute()
        return reports_query.data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve reports.")

        return supabase.table('Report').select('*, Users(name, avatar_url, points)').order('created_at', desc=True).range(offset, offset + limit - 1).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to retrieve reports.")

@app.get("/leaderboard")
async def get_leaderboard():
    try: return supabase.table('Users').select('name, points, avatar_url').order('points', desc=True).limit(100).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard.")

@app.get("/reports/me/top")
async def get_my_top_reports(user: dict = Depends(get_current_user)):
    try:
        user_query = supabase.table('Users').select('user_id').eq('user_id_auth', user_id_auth).single().execute()
        internal_user_id = user_query.data['user_id']
        reports_query = supabase.table('Report').select('*').eq('user_id', internal_user_id).order('upvote_count', desc=True).limit(3).execute()
        return reports_query.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve user reports: {e}")

@app.post("/reports", response_model=ReportResponse)
async def create_report(
    user_id: str = Form(...),
    category: str = Form(...),
    description: Optional[str] = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: Optional[UploadFile] = File(None),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    photo_url = None
    if file:
        try:
            file_content = await file.read()
            file_extension = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
            file_name = f"reports/{user_id}-{uuid.uuid4()}.{file_extension}"
            
            supabase.storage.from_("report-photos").upload(
                path=file_name,
                file=file_content,
                file_options={"content-type": file.content_type}
            )
            photo_url = supabase.storage.from_("report-photos").get_public_url(file_name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload photo: {str(e)}")

    address = "Address not found"
    try:
        reverse_geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GOOGLE_MAPS_API_KEY}"
        response = await client.get(reverse_geocode_url)
        if response.status_code == 200:
            results = response.json().get('results', [])
            if results:
                address = results[0]['formatted_address']
    except httpx.RequestError as e:
        print(f"Could not fetch address: {e}")

    try:
        user_profile = supabase.table('Users').select('user_id, points').eq('user_id_auth', user_id).single().execute()
        internal_user_id = user_profile.data['user_id']
        current_points = user_profile.data['points']

        report_data = {
            "user_id": internal_user_id, "description": description,
            "latitude": latitude, "longitude": longitude,
            "category": category, "photo_url": photo_url,
            "address": address
        user_id_query = supabase.table('Users').select('user_id').eq('user_id_auth', user.id).single().execute()
        internal_user_id = user_id_query.data['user_id']
        return supabase.table('Report').select('*').eq('user_id', internal_user_id).order('upvote_count', desc=True).limit(3).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to retrieve your reports: {e}")

@app.post("/reports")
async def create_report(
    fastapi_req: Request,
    category: str = Form(...),
    description: str | None = Form(None),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: UploadFile | None = File(None),
    client: httpx.AsyncClient = Depends(get_http_client),
    user: dict = Depends(get_current_user)
):
    try:
        user_id_auth = user.id
        photo_url = None
        if file:
            file_content = await file.read()
            file_name = f"reports/{user_id_auth}-{uuid.uuid4()}.{file.filename.split('.')[-1]}"
            supabase.storage.from_("report-photos").upload(file_name, file_content, {"contentType": file.content_type})
            photo_url = supabase.storage.from_("report-photos").get_public_url(file_name)
        
        address = "Address not found"
        geo_response = await client.get(f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GOOGLE_MAPS_API_KEY}")
        if geo_response.status_code == 200 and geo_response.json().get('results'):
            address = geo_response.json()['results'][0]['formatted_address']

        params = {
            "p_user_id_auth": user_id_auth, "p_category": category, "p_description": description,
            "p_latitude": latitude, "p_longitude": longitude, "p_photo_url": photo_url, "p_address": address
        }
        
        insert_response = supabase.table('Report').insert(report_data, returning="representation").execute()
        created_report = insert_response.data[0]

        new_points = current_points + 10
        supabase.table('Users').update({'points': new_points}).eq('user_id', internal_user_id).execute()
        
        if photo_url:
            asyncio.create_task(process_and_cache_report_photo(created_report))
        
        created_report['user_id'] = user_id
        if 'upvote_count' not in created_report:
            created_report['upvote_count'] = 0

        return ReportResponse(**created_report)
        
        response = supabase.rpc("create_report_atomic", params).execute()
        print(f"DEBUG: RPC response data: {response.data}")
        if response.data and isinstance(response.data, list) and len(response.data) > 0:
            created_report_object = response.data[0]
            if photo_url:
                redis = fastapi_req.app.state.redis
                await redis.enqueue_job("analyze_report_photo", created_report_object)
            return created_report_object
        elif response.data and isinstance(response.data, dict):
             created_report_object = response.data
             if photo_url:
                redis = fastapi_req.app.state.redis
                await redis.enqueue_job("analyze_report_photo", created_report_object)
             return created_report_object
        else:
            error_detail = f"Failed to create report. Invalid data returned from DB: {response.data}"
            print(error_detail)
            raise HTTPException(status_code=500, detail=error_detail)

    except APIError as e:
        raise HTTPException(status_code=400, detail=f"Failed to create report: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create report in database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/reports/{report_id}/upvote")
async def upvote_report(report_id: int, user: dict = Depends(get_current_user)):
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
async def geocode_address_async(address: str, client: httpx.AsyncClient) -> dict | None:
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = await client.get(url)
        if response.status_code == 200 and response.json()['status'] == 'OK':
            return response.json()['results'][0]['geometry']['location']
    except httpx.RequestError as e:
        print(f"Geocoding failed for {address}: {e}")
    return None

async def get_alternative_routes_from_google_async(origin_coords, dest_coords, client: httpx.AsyncClient):
    url = (f"https://maps.googleapis.com/maps/api/directions/json?"
           f"origin={origin_coords['lat']},{origin_coords['lng']}"
           f"&destination={dest_coords['lat']},{dest_coords['lng']}"
           f"&mode=walking&alternatives=true&key={GOOGLE_MAPS_API_KEY}")
    try:
        response = await client.get(url)
        if response.status_code == 200 and response.json().get('routes'):
            return response.json()['routes']
    except httpx.RequestError as e:
        print(f"Directions API call failed: {e}")
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
        params = {"p_report_id": report_id, "p_user_id_auth": user.id}
        result = supabase.rpc("upvote_report_atomic", params).execute().data
        return result
    except APIError as e:
        if 'User cannot upvote their own report' in e.message: raise HTTPException(status_code=403, detail="You cannot upvote your own report.")
        if 'unique constraint' in e.message: raise HTTPException(status_code=409, detail="You have already upvoted this report.")
        raise HTTPException(status_code=500, detail=f"Failed to process upvote: {e.message}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/calculate-routes", response_model=RouteCalculationStartedResponse, status_code=202)
async def create_route_calculation_job(request: RouteRequest, fastapi_req: Request):
    if fastapi_req.app.state.model is None: raise HTTPException(status_code=503, detail="AI model is not available.")
    job_id = str(uuid.uuid4())
    try:
        supabase.table("RouteSearch").insert({
            "job_id": job_id, "start_point": request.origin_address, "end_point": request.destination_address,
            "mode": request.mode, "status": "pending"
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
async def calculate_route_alternatives(request: RouteRequest, client: httpx.AsyncClient = Depends(get_http_client)):
    if model is None:
        raise HTTPException(status_code=503, detail="AI model is not available.")
    
    cached_results = await get_cached_analysis(request.origin_address, request.destination_address, request.mode)
    if cached_results:
        return RouteComparisonResponse(status="success", alternatives=cached_results, source="cache")

    origin_coords = await geocode_address_async(request.origin_address, client)
    dest_coords = await geocode_address_async(request.destination_address, client)
    if not origin_coords or not dest_coords:
        raise HTTPException(status_code=404, detail="One or both addresses could not be found.")
    
    alternative_routes_data = await get_alternative_routes_from_google_async(origin_coords, dest_coords, client)
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
        redis = fastapi_req.app.state.redis
        await redis.enqueue_job("analyze_and_save_routes", job_id, request.origin_address, request.destination_address, request.mode)
        return {"message": "Route calculation has been started.", "job_id": job_id}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {str(e)}")

@app.get("/routes/status/{job_id}", response_model=RouteStatusResponse)
async def get_route_calculation_status(job_id: str):
    try:
        query = supabase.table("RouteSearch").select("status, results").eq("job_id", job_id).single().execute()
        if not query.data: raise HTTPException(status_code=404, detail="Job not found.")
        data = query.data
        if data['status'] == 'completed': return {"status": "completed", "data": data['results']}
        elif data['status'] == 'failed':
            error_details = data.get('results', {}).get('error', 'An unknown error occurred during processing.')
            return {"status": "failed", "error": error_details}
        else: return {"status": data['status']}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

@app.get("/autocomplete-address", response_model=AutocompleteResponse)
async def autocomplete_address(query: str, client: httpx.AsyncClient = Depends(get_http_client)):
    url = (f"https://maps.googleapis.com/maps/api/place/autocomplete/json?"
           f"input={query}&key={GOOGLE_MAPS_API_KEY}&location=-7.2575,112.7521&radius=50000&strictbounds=true")
    try:
        response = await client.get(url)
        response.raise_for_status()
        return AutocompleteResponse(predictions=response.json().get('predictions', []))
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error contacting address service: {e}")
async def autocomplete_address(query: str, client: httpx.AsyncClient = Depends(get_http_client)):
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={query}&key={GOOGLE_MAPS_API_KEY}&location=-7.2575,112.7521&radius=50000&strictbounds=true"
    try: response = await client.get(url); response.raise_for_status(); return AutocompleteResponse(predictions=response.json().get('predictions', []))
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error contacting address service: {e}")

@app.get("/reverse-geocode")
async def get_readable_address(lat: float, lng: float, client: httpx.AsyncClient = Depends(get_http_client)):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = await client.get(url)
        response.raise_for_status()
        results = response.json().get('results', [])
        if results:
            return {"address": results[0]['formatted_address']}
        else:
            raise HTTPException(status_code=404, detail="Address not found for the given coordinates.")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Error during reverse geocoding: {e}")
async def get_readable_address(lat: float, lng: float, client: httpx.AsyncClient = Depends(get_http_client)):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = await client.get(url); response.raise_for_status()
        results = response.json().get('results', [])
        if results: return {"address": results[0]['formatted_address']}
        else: raise HTTPException(status_code=404, detail="Address not found.")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error during reverse geocoding: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the JalaninAja API!"}

def read_root(): return {"message": "Welcome to the JalaninAja API!"}

