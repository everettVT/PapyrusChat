from pymilvus import connections
import json
import typing


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional



app = FastAPI()

# Route for uploading files
@app.post("/run/")
async def run(config):
    try:

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

def main():
    import sys
    # collection_name = sys.argv[1]
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    collection_name = f"demo_collection"
    query = sys.argv[1]
    print(query_db(collection_name, query, embeddings))



if __name__ == '__main__':
    main()
