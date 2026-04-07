from fastapi import FastAPI, UploadFile, File, Form

import io
import pandas as pd
from fastapi import HTTPException, status
from typing import Optional
from regresje import train_linear_regression

app = FastAPI()


@app.post("/regressions/linear")
async def run_linear_regression(
    train_file: UploadFile = File(...),
    test_file: UploadFile | None = File(None),
    mode: str = Form(...),
    columns: Optional[str] = Form(None),
    transformations: Optional[str] = Form(None),
    target_column: str = Form(...),
    method: str = Form("ols"),
    alpha: Optional[float] = Form(None),
):
    contents = await train_file.read()
    try:
        train_df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="COULD_NOT_READ_CSV",
        )

    test_df = None
    if test_file is not None:
        test_contents = await test_file.read()
        try:
            test_df = pd.read_csv(io.StringIO(test_contents.decode("utf-8")))
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="COULD_NOT_READ_TEST_CSV",
            )

    return train_linear_regression(
        train_df=train_df,
        test_df=test_df,
        mode=mode,
        columns=columns,
        target_column=target_column,
        transformations=transformations,
        method=method,
        alpha=alpha,
    )
