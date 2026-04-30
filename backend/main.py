from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from exploration.routes import ExplorationRouter
import pandas as pd
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "EDA API is running"}
exploration_router = ExplorationRouter()
app.include_router(exploration_router.router, prefix="/exploration", tags=["Exploration"])

@app.post("/test/csv")
async def impute(file: UploadFile = File(...), method: str = Form(...), columns: str = Form(None)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    if "Embarked" in df.columns:
        df = df[df["Embarked"] == "Q"]
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")