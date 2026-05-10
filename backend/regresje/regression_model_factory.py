from fastapi import HTTPException, status
from typing import Optional

from .elastic_net import ElasticNetRegressionModel
from .lasso import LassoRegressionModel
from .ridge import RidgeRegressionModel


class RegressionModelFactory:
    _registry = {
        "ridge": RidgeRegressionModel,
        "lasso": LassoRegressionModel,
        "elastic_net": ElasticNetRegressionModel,
    }

    @classmethod
    def create(
        cls, method: str, alpha: Optional[float] = None, l1_ratio: Optional[float] = None
    ):
        model_class = cls._registry.get(method)
        if model_class is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="UNKNOWN_METHOD",
            )
        return model_class(alpha=alpha, l1_ratio=l1_ratio)
