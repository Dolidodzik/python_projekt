from fastapi import APIRouter, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
from .utils import FileHandler
from .core import DataExplorer

class ExplorationRouter:
    def __init__(self, explorer_class=DataExplorer):
        self.explorer_class = explorer_class
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/basic-stats", self.basic_stats, methods=["POST"])
        self.router.add_api_route("/column-types", self.column_types, methods=["POST"])
        self.router.add_api_route("/correlation", self.correlation, methods=["POST"])
        self.router.add_api_route("/pca", self.pca, methods=["POST"])
        self.router.add_api_route("/describe-all", self.describe_all, methods=["POST"])
        self.router.add_api_route("/distribution", self.distribution, methods=["POST"])

    async def basic_stats(self, file: UploadFile, columns: Optional[str] = Form(None)):
        df = FileHandler.read_csv_safe(file)
        col_list = FileHandler.validate_columns(df, columns, numeric_only=True)
        if not col_list:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Brak kolumn numerycznych.")
        explorer = self.explorer_class(df)
        result = explorer.basic_stats(col_list)
        return JSONResponse(content=result)

    async def column_types(self, file: UploadFile):
        df = FileHandler.read_csv_safe(file)
        explorer = self.explorer_class(df)
        return JSONResponse(content=explorer.column_types_report())

    async def correlation(self, file: UploadFile, method: str = Form("pearson"), columns: Optional[str] = Form(None)):
        if method not in ["pearson", "spearman", "kendall"]:
            raise HTTPException(status_code=400, detail="Metoda musi być 'pearson', 'spearman' lub 'kendall'.")
        df = FileHandler.read_csv_safe(file)
        col_list = FileHandler.validate_columns(df, columns, numeric_only=True)
        if len(col_list) < 2:
            raise HTTPException(status_code=400, detail="Potrzeba co najmniej dwóch kolumn numerycznych.")
        explorer = self.explorer_class(df)
        result = explorer.correlation_matrix(col_list, method)
        return JSONResponse(content=result)

    async def pca(self, file: UploadFile, columns: Optional[str] = Form(None), n_components: Optional[int] = Form(None), standardize: bool = Form(True)):
        df = FileHandler.read_csv_safe(file)
        col_list = FileHandler.validate_columns(df, columns, numeric_only=True)
        if len(col_list) < 2:
            raise HTTPException(status_code=400, detail="PCA wymaga co najmniej dwóch kolumn numerycznych.")
        explorer = self.explorer_class(df)
        try:
            result = explorer.pca(col_list, n_components, standardize)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(content=result)

    async def describe_all(self, file: UploadFile):
        df = FileHandler.read_csv_safe(file)
        explorer = self.explorer_class(df)
        return JSONResponse(content=explorer.describe_all())

    async def distribution(self, file: UploadFile, column: str = Form(...)):
        df = FileHandler.read_csv_safe(file)
        explorer = self.explorer_class(df)
        try:
            result = explorer.value_distribution(column)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(content=result)