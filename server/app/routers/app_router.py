from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from app.utils.workflow_helper import get_file_upload_url_helper
from app.utils.workflow_helper import PUBLIC_API_BASE_URL, UPLOADS_DIR

router = APIRouter()

@router.get("/get_file_upload_url")
async def get_file_upload_url(request: Request):
    try:
        # FastAPI's request.query_params returns an immutable dict-like object
        params = dict(request.query_params)
        return await get_file_upload_url_helper(params)
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload")
async def upload_file(key: str = Form(None), file: UploadFile = File(...)):
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        filename = Path(key or file.filename or "upload.bin").name
        if not filename:
            filename = f"{uuid4()}.bin"
        target = UPLOADS_DIR / filename
        content = await file.read()
        target.write_bytes(content)
        return {
            "key": filename,
            "url": f"{PUBLIC_API_BASE_URL.rstrip('/')}/uploads/{filename}",
        }
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/calculate_dynamic_cost")
async def calculate_dynamic_cost():
    return {"cost": 0}
