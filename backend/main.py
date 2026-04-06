from fastapi import FastAPI, UploadFile, File, Form

import io
import pandas as pd
from fastapi import HTTPException, status
from regresje import train_linear_regression

app = FastAPI()


@app.post("/regressions/linear")
async def run_linear_regression(
    file: UploadFile = File(...),
    mode: str = Form(...),
    columns: str | None = Form(None),
    target_column: str = Form(...),
):
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="COULD_NOT_READ_CSV",
        )

    return train_linear_regression(
        df=df, mode=mode, columns=columns, target_column=target_column
    )
