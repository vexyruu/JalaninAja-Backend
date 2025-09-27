from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import os
import uuid
import httpx
import json
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from postgrest import APIError
from google.cloud import tasks_v2
from google.api_core.client_options import ClientOptions as GCloudClientOptions

# --- Load Environment Variables ---
load_dotenv(find_dotenv())
app = FastAPI(
    title="JalaninAja API",
    description="API for calculating routes, gamification, and authentication.",
    version="3.7-final"
)

# --- Service Configuration ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# --- Google Cloud Tasks Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION")
TASK_QUEUE_NAME = os.getenv("TASK_QUEUE_NAME")
WORKER_URL = os.getenv("WORKER_URL")
WORKER_SERVICE_ACCOUNT_EMAIL = os.getenv("WORKER_SERVICE_ACCOUNT_EMAIL")

# --- Runtime Checks ---
if not all([SUPABASE_URL, SUPABASE_KEY, GOOGLE_MAPS_API_KEY, SUPABASE_ANON_KEY, GCP_PROJECT_ID, GCP_LOCATION, TASK_QUEUE_NAME, WORKER_URL, WORKER_SERVICE_ACCOUNT_EMAIL]):
    raise RuntimeError("One or more critical environment variables are missing.")

# --- Client Initializations ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
tasks_client = tasks_v2.CloudTasksClient(
    client_options=GCloudClientOptions(api_endpoint="cloudtasks.googleapis.com")
)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Pydantic Models ---
class ConfigResponse(BaseModel): supabase_url: str; supabase_anon_key: str; google_maps_api_key: str
class UserUpdateRequest(BaseModel): name: str | None = None; avatar_url: str | None = None
class UserProfileResponse(BaseModel): user_name: str; user_email: str; user_avatar_url: str | None = None; user_level: str; user_points: int; reports_made: int; badges: list[str]
class RouteRequest(BaseModel): origin_address: str; destination_address: str; mode: str = "distance_walkability"
class RouteCalculationStartedResponse(BaseModel): message: str; job_id: str
class RouteStatusResponse(BaseModel): status: str; data: list[dict] | None = None; error: str | None = None
class AutocompleteResponse(BaseModel): predictions: list[dict]

# --- Dependencies ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        user = supabase.auth.get_user(token).user
        if not user: raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

async def get_http_client():
    async with httpx.AsyncClient() as client: yield client

# --- Helper Function for Creating Tasks ---
def create_gcp_task(payload: dict, endpoint: str):
    parent = tasks_client.queue_path(GCP_PROJECT_ID, GCP_LOCATION, TASK_QUEUE_NAME)
    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{WORKER_URL}{endpoint}",
            "headers": {"Content-type": "application/json"},
            "body": json.dumps(payload).encode(),
            "oidc_token": {"service_account_email": WORKER_SERVICE_ACCOUNT_EMAIL},
        },
    }
    tasks_client.create_task(parent=parent, task=task)

# --- Endpoints ---
@app.get("/config", response_model=ConfigResponse)
def get_app_configuration(): return ConfigResponse(supabase_url=SUPABASE_URL, supabase_anon_key=SUPABASE_ANON_KEY, google_maps_api_key=GOOGLE_MAPS_API_KEY)

@app.patch("/users/me")
async def update_current_user(request: UserUpdateRequest, user: dict = Depends(get_current_user)):
    try:
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        if update_data:
            supabase.table("Users").update(update_data).eq("user_id_auth", user.id).execute()
            auth_update = {}
            if 'name' in update_data: auth_update['full_name'] = update_data['name']
            if 'avatar_url' in update_data: auth_update['avatar_url'] = update_data['avatar_url']
            if auth_update: supabase.auth.admin.update_user_by_id(user.id, {'data': auth_update})
        return {"status": "success", "message": "Profile updated successfully."}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to update user profile: {e}")


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
        badges = [b for p, b in [(10, "Pelapor Rajin"), (5, "Pembantu Sesama")] if reports_made >= p]
        return UserProfileResponse(user_name=user_data.get('name', 'N/A'), user_email=user_data.get('email', 'N/A'), user_avatar_url=user_data.get('avatar_url'), user_level=user_level, user_points=points, reports_made=reports_made, badges=badges)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to retrieve profile data: {e}")

@app.get("/reports/nearby")
async def get_nearby_reports(lat: float, lng: float, user: dict = Depends(get_current_user)):
    try:
        nearby_reports = supabase.rpc('get_reports_nearby', {'p_lat': lat, 'p_lng': lng}).execute().data
        report_ids = [report['report_id'] for report in nearby_reports]
        if not report_ids: return []
        full_reports_data = supabase.table('Report').select('*, Users(name, avatar_url, points)').in_('report_id', report_ids).order('created_at', desc=True).execute().data
        return full_reports_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve nearby reports: {e}")

# FIX: Added authentication to the search endpoint.
@app.get("/reports/search")
async def search_reports_by_location(query: str, user: dict = Depends(get_current_user), client: httpx.AsyncClient = Depends(get_http_client)):
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    try:
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={query}&key={GOOGLE_MAPS_API_KEY}&region=id"
        response = await client.get(geocode_url)
        response.raise_for_status()
        geocode_data = response.json()
        if not geocode_data.get('results'): return [] 
        location = geocode_data['results'][0]['geometry']['location']
        lat, lng = location['lat'], location['lng']
        nearby_reports = supabase.rpc('get_reports_nearby', {'p_lat': lat, 'p_lng': lng}).execute().data
        report_ids = [report['report_id'] for report in nearby_reports]
        if not report_ids: return []
        full_reports_data = supabase.table('Report').select('*, Users(name, avatar_url, points)').in_('report_id', report_ids).order('created_at', desc=True).execute().data
        return full_reports_data
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Failed to contact geolocation service: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {e}")

@app.get("/reports/{report_id}")
async def get_report_details(report_id: int, user: dict = Depends(get_current_user)):
    try:
        params = {"p_report_id": report_id, "p_user_id_auth": user.id}
        result = supabase.rpc("get_report_details", params).execute().data
        if not result:
            raise HTTPException(status_code=404, detail="Report not found.")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/reports")
async def get_all_reports(user: dict = Depends(get_current_user), offset: int = Query(0, ge=0), limit: int = Query(10, ge=1)):
    try: 
        return supabase.table('Report').select('*, Users(name, avatar_url, points)').order('created_at', desc=True).range(offset, offset + limit - 1).execute().data
    except Exception as e: 
        raise HTTPException(status_code=500, detail="Failed to retrieve reports.")

@app.get("/leaderboard")
async def get_leaderboard(user: dict = Depends(get_current_user), period: str = Query("week", enum=["week", "month", "year"])):
    try:
        return supabase.rpc('get_leaderboard_by_period', {'p_period': period}).execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve leaderboard: {e}")

@app.get("/reports/me/top")
async def get_my_top_reports(user: dict = Depends(get_current_user)):
    try:
        user_id_query = supabase.table('Users').select('user_id').eq('user_id_auth', user.id).single().execute()
        internal_user_id = user_id_query.data['user_id']
        return supabase.table('Report').select('*').eq('user_id', internal_user_id).order('upvote_count', desc=True).limit(3).execute().data
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to retrieve your reports: {e}")

@app.post("/reports")
async def create_report(category: str = Form(...), description: str | None = Form(None), latitude: float = Form(...), longitude: float = Form(...), file: UploadFile | None = File(None), client: httpx.AsyncClient = Depends(get_http_client), user: dict = Depends(get_current_user)):
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
        params = {"p_user_id_auth": user_id_auth, "p_category": category, "p_description": description, "p_latitude": latitude, "p_longitude": longitude, "p_photo_url": photo_url, "p_address": address}
        response = supabase.rpc("create_report_atomic", params).execute()
        created_report_object = (response.data[0] if response.data and isinstance(response.data, list) else None)
        if created_report_object and photo_url:
            task_payload = {"report_data": created_report_object}
            create_gcp_task(task_payload, "/process-photo-task")
        return created_report_object or {"error": "Failed to create report"}
    except APIError as e: raise HTTPException(status_code=400, detail=f"Failed to create report: {e.message}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/reports/{report_id}/upvote")
async def upvote_report(report_id: int, user: dict = Depends(get_current_user)):
    try:
        params = {"p_report_id": report_id, "p_user_id_auth": user.id}
        result = supabase.rpc("upvote_report_atomic", params).execute().data
        return result
    except APIError as e:
        if 'User cannot vote on their own report' in e.message: raise HTTPException(status_code=403, detail="You cannot vote on your own report.")
        if 'unique constraint' in e.message: raise HTTPException(status_code=409, detail="You have already upvoted this report.")
        raise HTTPException(status_code=500, detail=f"Failed to process upvote: {e.message}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.delete("/reports/{report_id}/vote")
async def remove_vote(report_id: int, user: dict = Depends(get_current_user)):
    try:
        params = {"p_report_id": report_id, "p_user_id_auth": user.id}
        result = supabase.rpc("remove_vote_atomic", params).execute().data
        return result
    except APIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove vote: {e.message}")
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/calculate-routes", response_model=RouteCalculationStartedResponse, status_code=202)
async def create_route_calculation_job(request: RouteRequest):
    if not os.path.exists("best.onnx"):
        raise HTTPException(status_code=503, detail="AI model file is not available in the service container.")
    job_id = str(uuid.uuid4())
    try:
        supabase.table("RouteSearch").insert({"job_id": job_id, "start_point": request.origin_address, "end_point": request.destination_address, "mode": request.mode, "status": "pending"}).execute()
        task_payload = {
            "job_id": job_id,
            "origin_address": request.origin_address,
            "destination_address": request.destination_address,
            "mode": request.mode
        }
        create_gcp_task(task_payload, "/process-route-task")
        return {"message": "Route calculation has been started.", "job_id": job_id}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/routes/status/{job_id}", response_model=RouteStatusResponse)
async def get_route_calculation_status(job_id: str):
    try:
        query = supabase.table("RouteSearch").select("status, results").eq("job_id", job_id).single().execute()
        if not query.data: raise HTTPException(status_code=404, detail="Job not found.")
        data = query.data
        if data['status'] == 'completed': return {"status": "completed", "data": data['results']}
        elif data['status'] == 'failed':
            error_details = data.get('results', {}).get('error', 'An unknown error occurred.')
            return {"status": "failed", "error": error_details}
        else: return {"status": data['status']}
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error retrieving job status: {str(e)}")

@app.get("/autocomplete-address", response_model=AutocompleteResponse)
async def autocomplete_address(
    query: str, 
    lat: float | None = Query(None),
    lng: float | None = Query(None),
    client: httpx.AsyncClient = Depends(get_http_client)
):
    url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={query}&key={GOOGLE_MAPS_API_KEY}&region=id"
    if lat is not None and lng is not None:
        url += f"&location={lat},{lng}&radius=50000&strictbounds=true"
    try: 
        response = await client.get(url)
        response.raise_for_status()
        return AutocompleteResponse(predictions=response.json().get('predictions', []))
    except httpx.RequestError as e: 
        raise HTTPException(status_code=503, detail=f"Error contacting address service: {e}")

@app.get("/reverse-geocode")
async def get_readable_address(lat: float, lng: float, client: httpx.AsyncClient = Depends(get_http_client)):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={GOOGLE_MAPS_API_KEY}"
    try:
        response = await client.get(url); response.raise_for_status()
        results = response.json().get('results', [])
        return {"address": results[0]['formatted_address']} if results else {"address": "Address not found"}
    except httpx.RequestError as e: raise HTTPException(status_code=503, detail=f"Error during reverse geocoding: {e}")

@app.get("/")
def read_root(): return {"message": "Welcome to the JalaninAja API!"}

