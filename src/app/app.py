import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rag import predict_label

class Query(BaseModel):
    text: str
    label: str

app = FastAPI()

@app.post("/query", response_class=JSONResponse)
def query_rag(query_data: Query):
    text = query_data.text
    label = query_data.label
    try:
        top_labels, label = predict_label(text, label)

        return {"top_labels": top_labels, "predicted_label": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host='127.0.0.1',
        port=4830,
        log_level="info",
    )

