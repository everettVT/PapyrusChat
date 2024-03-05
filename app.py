from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

# Route for uploading files
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # You can now save the file, process it, etc. Here's a simple placeholder response.
        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route for handling queries
@app.get("/query/")
async def query_item(q: Optional[str] = None):
    if q:
        # Perform some operations based on query 'q'. Here's a simple placeholder response.
        return {"query": q}
    else:
        return {"error": "Query parameter 'q' is missing"}



