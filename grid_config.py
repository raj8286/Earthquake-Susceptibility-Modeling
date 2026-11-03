# grid_config.py


from dataclasses import dataclass
from pathlib import Path


# ============================================================
# COMMON PROJECTED DOMAIN
#
# This is the finalized global EPSG:6933 domain used by the
# corrected 30 km pipeline:
#
#   30 km: 1158 x 488
#
# The domain dimensions are deliberately divisible by
# 15 km, 30 km and 60 km, so all three experiments cover
# EXACTLY the same projected area.
# ============================================================

X_MIN = -17367530.45
X_MAX =  17372469.55

Y_MIN = -7312230.14
Y_MAX =  7327769.86

SUPPORTED_CELL_KM = (15, 30, 60)


@dataclass(frozen=True)
class GridConfig:
    cell_km: int
    cell_m: float

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    nx: int
    ny: int

    dataset_dir: Path
    earthquake_dir: Path
    feature_dir: Path
    model_dir: Path
    intermediate_dir: Path


def get_grid_config(cell_km: int) -> GridConfig:
    if cell_km not in SUPPORTED_CELL_KM:
        raise ValueError(
            f"Unsupported cell size: {cell_km} km. "
            f"Supported values: {SUPPORTED_CELL_KM}"
        )

    cell_m = float(cell_km * 1000)

    width_m = X_MAX - X_MIN
    height_m = Y_MAX - Y_MIN

    nx_float = width_m / cell_m
    ny_float = height_m / cell_m

    nx = int(round(nx_float))
    ny = int(round(ny_float))

    # Require exact alignment of the shared study domain.
    if abs(nx * cell_m - width_m) > 1e-6:
        raise RuntimeError(
            f"X extent is not exactly divisible by {cell_km} km."
        )

    if abs(ny * cell_m - height_m) > 1e-6:
        raise RuntimeError(
            f"Y extent is not exactly divisible by {cell_km} km."
        )

    dataset_dir = Path("data") / f"grid_{cell_km}km"

    earthquake_dir = dataset_dir / "earthquakes"
    feature_dir = dataset_dir / "features"
    model_dir = dataset_dir / "model"

    intermediate_dir = (
        Path("intermediate")
        / f"grid_{cell_km}km"
    )

    return GridConfig(
        cell_km=cell_km,
        cell_m=cell_m,

        x_min=X_MIN,
        x_max=X_MAX,
        y_min=Y_MIN,
        y_max=Y_MAX,

        nx=nx,
        ny=ny,

        dataset_dir=dataset_dir,
        earthquake_dir=earthquake_dir,
        feature_dir=feature_dir,
        model_dir=model_dir,
        intermediate_dir=intermediate_dir,
    )


def ensure_grid_directories(config: GridConfig) -> None:
    for path in [
        config.dataset_dir,
        config.earthquake_dir,
        config.feature_dir,
        config.model_dir,
        config.intermediate_dir,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )