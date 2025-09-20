from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import os
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest import APIError
import uuid
import httpx
from dotenv import load_dotenv, find_dotenv
from arq import create_pool
from arq.connections import RedisSettings
from shared.analysis import calculate_walkability_score, run_model_on_image, is_road_residential
load_dotenv(find_dotenv())

app = FastAPI(
    title="JalaninAja API",
    description="API for calculating routes, gamification, and authentication.",
    version="3"
)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
REDIS_URL = os.getenv("REDIS_URL")

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
        from ultralytics import YOLO
        return YOLO(local_model_path)
    return None

class ConfigResponse(BaseModel): supabase_url: str; supabase_anon_key: str; google_maps_api_key: str
class UserUpdateRequest(BaseModel): name: str | None = None; avatar_url: str | None = None
class UserProfileResponse(BaseModel): user_name: str; user_email: str; user_avatar_url: str | None = None; user_level: str; user_points: int; reports_made: int; badges: list[str]
class RouteRequest(BaseModel): origin_address: str; destination_address: str; mode: str = "distance_walkability"
class RouteCalculationStartedResponse(BaseModel): message: str; job_id: str
class RouteStatusResponse(BaseModel): status: str; data: list[dict] | None = None; error: str | None = None
class AutocompleteResponse(BaseModel): predictions: list[dict]

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
        return UserProfileResponse(user_name=user_data.get('name', 'N/A'), user_email=user_data.get('email', 'N/A'), user_avatar_url=user_data.get('avatar_url'), user_level=user_level, user_points=points, reports_made=reports_made, badges=badges)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to retrieve profile data: {e}")

@app.get("/reports")
async def get_all_reports(offset: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
    try:
        return supabase.table('Report').select('*, Users(name, avatar_url, points)').order('created_at', desc=True).range(offset, offset + limit - 1).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to retrieve reports.")

@app.get("/leaderboard")
async def get_leaderboard():
    try: return supabase.table('Users').select('name, points, avatar_url').order('points', desc=True).limit(100).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard.")

@app.get("/reports/me/top")
async def get_my_top_reports(user: dict = Depends(get_current_user)):
    try:
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
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/reports/{report_id}/upvote")
async def upvote_report(report_id: int, user: dict = Depends(get_current_user)):
    try:
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
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={query}&key={GOOGLE_MAPS_API_KEY}&location=-7.2575,112.7521&radius=50000&strictbounds=true"
    try: response = await client.get(url); response.raise_for_status(); return AutocompleteResponse(predictions=response.json().get('predictions', []))
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error contacting address service: {e}")

@app.get("/reverse-geocode")
async def get_readable_address(lat: float, lng: float, client: httpx.AsyncClient = Depends(get_http_client)):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = await client.get(url); response.raise_for_status()
        results = response.json().get('results', [])
        if results: return {"address": results[0]['formatted_address']}
        else: raise HTTPException(status_code=404, detail="Address not found.")
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error during reverse geocoding: {e}")

@app.get("/")
def read_root(): return {"message": "Welcome to the JalaninAja API!"}
