from pathlib import Path

from fastapi import APIRouter, Depends

from app.core.settings import Settings, get_settings
from app.schemas.experiment import DatasetStatus

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("/status", response_model=DatasetStatus)
def dataset_status(settings: Settings = Depends(get_settings)):
    data = settings.project_root / "amazon-book"
    try:
        rel = str(data.relative_to(settings.project_root))
    except ValueError:
        rel = str(data)
    return DatasetStatus(
        data_dir=rel,
        exists=data.is_dir(),
        train_txt=(data / "train.txt").is_file(),
        test_txt=(data / "test.txt").is_file(),
    )
