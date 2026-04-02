from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends

from app.core.settings import Settings, get_settings
from app.schemas.experiment import DatasetStatus, DatasetsOverview

router = APIRouter(prefix="/datasets", tags=["datasets"])

# Under project-root `dataset/`, same layout (train.txt, test.txt) for each benchmark.
KNOWN_DATASET_DIRS: List[str] = ["dataset/amazon-book", "dataset/movielens-100k"]


def _status_for_dir(root: Path, name: str) -> DatasetStatus:
    data = root / name
    try:
        rel = str(data.relative_to(root))
    except ValueError:
        rel = name
    return DatasetStatus(
        data_dir=rel,
        exists=data.is_dir(),
        train_txt=(data / "train.txt").is_file(),
        test_txt=(data / "test.txt").is_file(),
    )


@router.get("/status", response_model=DatasetsOverview)
def dataset_status(settings: Settings = Depends(get_settings)):
    root = settings.project_root
    entries = [_status_for_dir(root, name) for name in KNOWN_DATASET_DIRS]
    return DatasetsOverview(datasets=entries)
