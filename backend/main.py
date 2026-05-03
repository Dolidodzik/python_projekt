from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi import HTTPException

import pandas as pd
import io

from cleaning_and_preprocessing.utils import read_dataframe, parse_columns
from cleaning_and_preprocessing.impute import impute_data
from cleaning_and_preprocessing.normalize import normalize_data
from cleaning_and_preprocessing.split import split_data
from cleaning_and_preprocessing.encode import encode_data
from cleaning_and_preprocessing.drop import drop_column_data
from cleaning_and_preprocessing.fillna import fillna_constant_data
from cleaning_and_preprocessing.outliers import remove_outliers_data

from statistic_tests.chi_square import chi_square
from statistic_tests.correlation import correlation
from statistic_tests.t_test import t_test
from statistic_tests.anova import anova
from statistic_tests.normality import normality_test
from statistic_tests.mann_whitney import mann_whitney_test
from statistic_tests.covariance import covariance
from typing import List

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Data preprocessing API running"}

@app.post("/cleaning_and_preprocessing/impute")
async def impute(
    file: UploadFile = File(...),
    method: str = Form(...),
    columns: str = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    cols = parse_columns(columns, df)
    df = impute_data(df, method, cols)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")


@app.post("/cleaning_and_preprocessing/normalize")
async def normalize(
    file: UploadFile = File(...),
    method: str = Form(...),
    columns: str = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    cols = parse_columns(columns, df)
    df = normalize_data(df, method, cols)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")


@app.post("/cleaning_and_preprocessing/split")
async def split(
    file: UploadFile = File(...),
    test_size: float = Form(...),
    random_state: int = Form(None),
    stratify: str = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    train, test = split_data(df, test_size, random_state, stratify)
    return {
        "train": train.to_csv(index=False),
        "test": test.to_csv(index=False)
    }


@app.post("/cleaning_and_preprocessing/encode_categorical")
async def encode_categorical(
    file: UploadFile = File(...),
    method: str = Form(...),
    columns: str = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    cols = parse_columns(columns, df)
    df = encode_data(df, method, cols)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")


@app.post("/cleaning_and_preprocessing/drop_column")
async def drop_column(
    file: UploadFile = File(...),
    column: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)
    df = drop_column_data(df, column)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")


@app.post("/cleaning_and_preprocessing/fillna_constant")
async def fillna_constant(
    file: UploadFile = File(...),
    value: str = Form(...),
    columns: str = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    cols = parse_columns(columns, df)
    df = fillna_constant_data(df, value, cols)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")


@app.post("/cleaning_and_preprocessing/remove_outliers")
async def remove_outliers(
    file: UploadFile = File(...),
    method: str = Form(...),
    columns: str = Form(None),
    threshold: float = Form(None)
):
    contents = await file.read()
    df = read_dataframe(contents)
    cols = parse_columns(columns, df)
    df = remove_outliers_data(df, method, cols, threshold)
    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")

@app.post("/statistic-tests/chi_square")
async def chi_square_endpoint(
    file: UploadFile = File(...),
    col1: str = Form(...),
    col2: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        result = chi_square(df, col1, col2)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/correlation")
async def correlation_endpoint(
    file: UploadFile = File(...),
    method: str = Form(...),
    columns: List[str] = Form(...)
):

    contents = await file.read()
    df = read_dataframe(contents)

    try:
        cols = parse_columns(columns, df)
        result_df = correlation(df, method, cols)
        output = result_df.to_csv(index=False)
        return Response(content=output, media_type="text/csv")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/t_test")
async def t_test_endpoint(
    file: UploadFile = File(...),
    group_col: str = Form(...),
    value_col: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        result = t_test(df, group_col, value_col)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/anova")
async def anova_endpoint(
    file: UploadFile = File(...),
    group_col: str = Form(...),
    value_col: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        result = anova(df, group_col, value_col)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/normality")
async def normality_endpoint(
    file: UploadFile = File(...),
    column: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        result = normality_test(df, column)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/mann_whitney")
async def mann_whitney_endpoint(
    file: UploadFile = File(...),
    group_col: str = Form(...),
    value_col: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        result = mann_whitney_test(df, group_col, value_col)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/statistic-tests/covariance")
async def covariance_endpoint(
    file: UploadFile = File(...),
    columns: str = Form(...)
):
    contents = await file.read()
    df = read_dataframe(contents)

    try:
        cols = parse_columns(columns, df)
        result_df = covariance(df, cols)
        output = result_df.to_csv(index=False)
        return Response(content=output, media_type="text/csv")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/test/csv")
async def test_csv(
    file: UploadFile = File(...), method: str = Form(...), columns: str = Form(None)
):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # tescik z repo
    if "Embarked" in df.columns:
        df = df[df["Embarked"] == "Q"]

    output = df.to_csv(index=False)
    return Response(content=output, media_type="text/csv")