from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "hourly.parquet"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "forecaster.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BACKTEST_PATH = REPORTS_DIR / "backtest_predictions.parquet"

TZ = "Europe/Paris"
FIRST_YEAR = 2015          # weather data starts in 2015
TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
RANDOM_STATE = 42

# Approximate metropolitan-area populations (millions), used to weight city temperatures.
# Demand follows people, not area: Paris weighs far more than Bordeaux.
CITIES = {
    "Paris": (48.85, 2.35, 13.0),
    "Lyon": (45.76, 4.84, 2.3),
    "Marseille": (43.30, 5.37, 1.9),
    "Lille": (50.63, 3.06, 1.5),
    "Bordeaux": (44.84, -0.58, 1.0),
}
