"""Shared loaders and helpers for the Nutrition Explorer app."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    ROOT / "DATA" / "processed",
    ROOT / "data" / "processed",
]
MODELS_DIR = ROOT / "models"
NUTRIENT_BASES = [
    "energy_kcal",
    "protein_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "added_sugar_g",
    "sodium_mg",
    "sat_fat_g",
]
NAV_ITEMS = [
    ("Home", "Home.py"),
    ("Search", "Search.py"),
    # ("Detail", "Detail.py"),
    ("Compare", "Compare.py"),
    ("Substitute", "Substitute.py"),
]

if TYPE_CHECKING:  # pragma: no cover
    from src.service.predictor import PredictorService

LOGGER = logging.getLogger(__name__)
_PREDICTOR_FAILED = object()


def _resolve_processed_path(filename: str) -> Path:
    for base in DATA_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return candidate
    # fall back to first candidate even if missing to surface helpful error upstream
    return DATA_CANDIDATES[0] / filename


@st.cache_data(show_spinner=False)
def load_foods() -> pd.DataFrame:
    """Load the foods dataset with expected alias columns."""
    path = _resolve_processed_path("foods_nutrients.parquet")
    if not path.exists():
        raise FileNotFoundError(
            "foods_nutrients.parquet not found. Run make build to generate processed data."
        )
    try:
        df = pd.read_parquet(path).copy()
    except OSError as exc:
        raise RuntimeError(
            "Failed to load foods_nutrients.parquet. This is often caused by "
            "an incompatible pyarrow build; try updating pyarrow or rebuilding the dataset."
        ) from exc

    # Ensure both score columns exist for downstream references
    if "score" not in df.columns and "nutrition_score" in df.columns:
        df["score"] = df["nutrition_score"]
    if "nutrition_score" not in df.columns and "score" in df.columns:
        df["nutrition_score"] = df["score"]

    for base in NUTRIENT_BASES:
        per100_col = f"{base}_per100g"
        perserving_col = f"{base}_perserving"
        has_per100 = per100_col in df.columns
        has_perserving = perserving_col in df.columns
        if not has_per100 and has_perserving:
            df[per100_col] = pd.NA
        if not has_perserving and has_per100:
            df[perserving_col] = pd.NA

    # Guarantee optional text columns exist to avoid KeyErrors
    for column in ("brand_owner", "food_category", "wweia_category", "grade"):
        if column not in df.columns:
            df[column] = pd.NA

    return df

@st.cache_data(show_spinner=False)
def load_sample_foods() -> pd.DataFrame:
    """Load the 50k sample dataset for fast interactive features."""
    sample_path = _resolve_processed_path("foods_sample_50k.parquet")
    if not sample_path.exists():
        # Fallback to full dataset if sample doesn't exist
        return load_foods()
    
    try:
        df = pd.read_parquet(sample_path).copy()
    except OSError as exc:
        LOGGER.warning("Failed to load sample dataset: %s", exc)
        return load_foods()

    # Apply same transformations as full dataset
    if "score" not in df.columns and "nutrition_score" in df.columns:
        df["score"] = df["nutrition_score"]
    if "nutrition_score" not in df.columns and "score" in df.columns:
        df["nutrition_score"] = df["score"]

    for base in NUTRIENT_BASES:
        per100_col = f"{base}_per100g"
        perserving_col = f"{base}_perserving"
        has_per100 = per100_col in df.columns
        has_perserving = perserving_col in df.columns
        if not has_per100 and has_perserving:
            df[per100_col] = pd.NA
        if not has_perserving and has_per100:
            df[perserving_col] = pd.NA

    for column in ("brand_owner", "food_category", "wweia_category", "grade"):
        if column not in df.columns:
            df[column] = pd.NA

    return df


@st.cache_data(show_spinner=False)
def load_nn() -> pd.DataFrame:
    """Load nearest-neighbor dataframe if available."""
    path = _resolve_processed_path("nn_index.parquet")
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted files
        LOGGER.warning("Failed to read nn_index.parquet: %s", exc)
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_importances() -> Dict[str, Any]:
    """Load importance coefficients for tooltip/context if present."""
    path = MODELS_DIR / "coef.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        LOGGER.warning("Invalid coef.json: %s", exc)
        return {}


def _format_number(value: Any, decimals: int = 1, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    fmt = f"{{:.{decimals}f}}".format(number)
    return f"{fmt}{suffix}"


def safe_serving_value(row: pd.Series, column: str, unit: str = "", decimals: int = 0) -> str:
    value = row.get(column)
    suffix = f" {unit}" if unit else ""
    return _format_number(value, decimals=decimals, suffix=suffix)


def find_items(df: pd.DataFrame, q: str, max_rows: int = 500) -> pd.DataFrame:
    """Case-insensitive substring search over name and brand owner."""
    if not q:
        return df.head(0)
    needles = [token.strip() for token in q.lower().split() if token.strip()]
    if not needles:
        return df.head(0)

    def contains(series: pd.Series, needle: str) -> pd.Series:
        return series.fillna("").str.lower().str.contains(needle, na=False)

    mask = pd.Series(True, index=df.index)
    for needle in needles:
        name_match = contains(df["description"], needle)
        brand_match = contains(df["brand_owner"], needle)
        mask &= name_match | brand_match

    results = df.loc[mask].copy()
    if results.empty:
        # fallback to first token match to keep UX forgiving
        primary = needles[0]
        results = df[contains(df["description"], primary) | contains(df["brand_owner"], primary)].copy()
    return results.head(max_rows)


def link_to(page: str, **params: Any) -> str:
    """Return a URL pointing at a registered Streamlit page with optional query params."""
    return page_url(page, **params)


def _resolve_page(page: str) -> Path:
    return (APP_DIR / page).resolve()


def _find_page_info(page: str) -> Dict[str, Any]:
    ctx = get_script_run_ctx()
    if not ctx:
        return {}
    target = str(_resolve_page(page))
    for info in ctx.pages_manager.get_pages().values():
        if info.get("script_path") == target:
            return info
    return {}


def page_path(page: str) -> str:
    """Return the path portion for a registered page (e.g. '/detail')."""
    info = _find_page_info(page)
    url_path = info.get("url_pathname")
    if url_path is None:
        return ""
    return "/" if url_path == "" else f"/{url_path}"


def page_url(page: str, **params: Any) -> str:
    """Return a URL for a page, including optional query parameters."""
    base = page_path(page)
    query = {key: value for key, value in params.items() if value is not None}
    if not base:
        fallback = {"page": page}
        fallback.update(query)
        return "?" + urlencode(fallback, doseq=True)
    if not query:
        return base
    return f"{base}?{urlencode(query, doseq=True)}"


def page_link_button(
    page: str,
    label: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    disabled: bool = False,
    width: str = "content",
    button_type: str = "secondary",
    icon: Optional[str] = None,
) -> None:
    """Render a link-style button that navigates to another Streamlit page."""
    url = page_url(page, **(params or {}))
    st.link_button(
        label,
        url=url,
        disabled=disabled,
        width=width,
        type=button_type,
        icon=icon,
    )

def _switch_page_fallback(page: str) -> bool:
    """Attempt to switch to a page using several candidate script paths."""
    candidates = []
    if page.endswith(".py"):
        candidates.append(page)
        candidates.append(f"app/{page}")
    else:
        script = f"{page}.py"
        candidates.extend([script, f"app/{script}"])
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            st.switch_page(candidate)
            return True
        except Exception:
            continue
    return False


def quick_nav_button(label: str, page: str) -> None:
    """
    Render a navigation button that prefers Streamlit's native page_link
    but gracefully falls back to switch_page for direct script runs.
    """
    script = page if page.endswith(".py") else f"{page}.py"
    try:
        st.page_link(script, label=label, width="stretch")
        return
    except Exception:
        pass
    # Fallback for contexts where navigation APIs aren't initialised (e.g. direct script run)
    if st.button(label, width="stretch"):
        if not _switch_page_fallback(script):
            st.warning("Unable to navigate to the requested page. Use the sidebar navigation instead.")

def render_nav(active: str) -> None:
    """Display a top navigation bar using Streamlit page links."""
    cols = st.columns(len(NAV_ITEMS))
    for col, (label, page) in zip(cols, NAV_ITEMS):
        with col:
            nav_label = f"· {label} ·" if label == active else label
            page_info = _find_page_info(page)
            if page_info.get("url_pathname") is not None:
                st.page_link(
                    page,
                    label=nav_label,
                    disabled=label == active,
                    width="stretch",
                )
            else:
                page_link_button(
                    page,
                    nav_label,
                    disabled=label == active,
                    width="stretch",
                    button_type="primary" if label == active else "secondary",
                )


def load_service_or_error() -> "AppService":
    """Return the cached service or surface a user-friendly error."""
    try:
        return get_service()
    except Exception as exc:  # pragma: no cover - Streamlit should show message
        st.error(
            "We couldn't load the processed nutrition dataset. "
            "Please rerun the data pipeline (`make build`) or check your local PyArrow installation."
        )
        st.exception(exc)
        st.stop()

class AppService:
    """Service for managing nutrition data with lazy loading and dual dataset support."""
    
    def __init__(self, dataset: pd.DataFrame = None, neighbors: pd.DataFrame = None,
                 importances: Dict[str, Any] = None, processed_dir: Path = None):
        # Fix DataFrame boolean evaluation issue
        self.__dict__["_dataset"] = dataset if dataset is not None else pd.DataFrame()
        self.neighbors = neighbors if neighbors is not None else pd.DataFrame()
        self.importances = importances if importances is not None else {}
        self.processed_dir = processed_dir or _resolve_processed_path("foods_nutrients.parquet").parent
        self._predictor = None
        self._score_column_cache = None
        self._dataset_loaded = False
        self._sample_loaded = False
        self._empty_dataset = pd.DataFrame()

    @property
    def dataset(self) -> pd.DataFrame:
        """Lazy load the full dataset only when first accessed."""
        if not self._dataset_loaded or self.__dict__["_dataset"].empty:
            with st.spinner("Loading nutrition data..."):
                self._dataset_loaded = True
                # Use optimized data loader for better performance
                loaded_dataset = load_foods_optimized()
                # Ensure score columns exist
                if "score" not in loaded_dataset.columns and "nutrition_score" in loaded_dataset.columns:
                    loaded_dataset["score"] = loaded_dataset["nutrition_score"]
                if "nutrition_score" not in loaded_dataset.columns and "score" in loaded_dataset.columns:
                    loaded_dataset["nutrition_score"] = loaded_dataset["score"]
                # Ensure nutrient columns exist
                for base in NUTRIENT_BASES:
                    per100_col = f"{base}_per100g"
                    perserving_col = f"{base}_perserving"
                    has_per100 = per100_col in loaded_dataset.columns
                    has_perserving = perserving_col in loaded_dataset.columns
                    if not has_per100 and has_perserving:
                        loaded_dataset[per100_col] = pd.NA
                    if not has_perserving and has_per100:
                        loaded_dataset[perserving_col] = pd.NA
                # Ensure optional text columns exist
                for column in ("brand_owner", "food_category", "wweia_category", "grade"):
                    if column not in loaded_dataset.columns:
                        loaded_dataset[column] = pd.NA
                self.__dict__["_dataset"] = loaded_dataset
        return self.__dict__["_dataset"]
    
    @property
    def sample_dataset(self) -> pd.DataFrame:
        """Lazy load the 50k sample dataset for fast interactive features."""
        if not self._sample_loaded or "_sample_dataset" not in self.__dict__:
            with st.spinner("Loading sample dataset..."):
                self._sample_loaded = True
                # Use optimized data loader for better performance
                loaded_sample = load_foods_optimized_sample()
                # Ensure score columns exist
                if "score" not in loaded_sample.columns and "nutrition_score" in loaded_sample.columns:
                    loaded_sample["score"] = loaded_sample["nutrition_score"]
                if "nutrition_score" not in loaded_sample.columns and "score" in loaded_sample.columns:
                    loaded_sample["nutrition_score"] = loaded_sample["score"]
                # Ensure nutrient columns exist
                for base in NUTRIENT_BASES:
                    per100_col = f"{base}_per100g"
                    perserving_col = f"{base}_perserving"
                    has_per100 = per100_col in loaded_sample.columns
                    has_perserving = perserving_col in loaded_sample.columns
                    if not has_per100 and has_perserving:
                        loaded_sample[per100_col] = pd.NA
                    if not has_perserving and has_per100:
                        loaded_sample[perserving_col] = pd.NA
                # Ensure optional text columns exist
                for column in ("brand_owner", "food_category", "wweia_category", "grade"):
                    if column not in loaded_sample.columns:
                        loaded_sample[column] = pd.NA
                self.__dict__["_sample_dataset"] = loaded_sample
        return self.__dict__["_sample_dataset"]

    @dataset.setter
    def dataset(self, value: pd.DataFrame) -> None:
        """Allow setting dataset with lazy loading support."""
        self.__dict__["_dataset"] = value
        if not value.empty:
            self._dataset_loaded = True

    @property
    def score_column(self) -> str:
        value = self._score_column_cache
        if value is None:
            # Access the dataset property to trigger lazy loading
            current_dataset = self.dataset
            value = "score" if "score" in current_dataset.columns else "nutrition_score"
            self._score_column_cache = value
        return value

    def nutrient_cols(self, per: str) -> List[str]:
        suffix = "_per100g" if per == "per100g" else "_perserving"
        columns: List[str] = []
        for base in NUTRIENT_BASES:
            column_name = f"{base}{suffix}"
            if column_name in self.dataset.columns:
                columns.append(column_name)
        return columns

    def predict_probs(self, fdc_id: int) -> Dict[str, Any]:
        predictor = self._get_predictor()
        if predictor is None:
            return {}
        try:
            return predictor.predict(fdc_id)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Prediction failed for %s: %s", fdc_id, exc)
            return {}

    def format_badges(self, row: pd.Series) -> Dict[str, str]:
        # Use normalized score if available, otherwise use original score
        normalized_score_col = f"{self.score_column}_normalized"
        if normalized_score_col in row.index and pd.notna(row.get(normalized_score_col)):
            score = _format_number(row.get(normalized_score_col), decimals=0)  # Use 0 decimals for normalized scores
        else:
            score = _format_number(row.get(self.score_column), decimals=1)
        grade = row.get("grade") if pd.notna(row.get("grade")) else "—"
        kcal = safe_serving_value(row, "energy_kcal_perserving", unit="kcal", decimals=0)
        sodium = safe_serving_value(row, "sodium_mg_perserving", unit="mg", decimals=0)
        sugar = safe_serving_value(row, "sugar_g_perserving", unit="g", decimals=1)
        return {
            "score": score,
            "grade": grade,
            "kcal_serv": kcal,
            "sodium_serv": sodium,
            "sugar_serv": sugar,
        }

    def _get_predictor(self) -> Optional["PredictorService"]:
        if self._predictor is _PREDICTOR_FAILED:
            return None
        if self._predictor is None:
            try:
                from src.service.predictor import PredictorService  # local import for small apps

                self._predictor = PredictorService(processed_dir=self.processed_dir)
            except Exception as exc:
                LOGGER.warning("PredictorService unavailable: %s", exc)
                self._predictor = _PREDICTOR_FAILED
        return None if self._predictor is _PREDICTOR_FAILED else self._predictor


@st.cache_data(show_spinner=False)
def load_foods_optimized() -> pd.DataFrame:
    """Load foods with optimized data types for better performance."""
    df = load_foods()
    
    # Optimize data types for memory and speed
    if "fdc_id" in df.columns:
        df["fdc_id"] = df["fdc_id"].astype("int32")
    
    # Optimize string columns
    for col in ["description", "brand_owner", "food_category", "grade"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("")
    
    # Optimize numeric columns
    numeric_cols = [col for col in df.columns if any(suffix in col for suffix in ["_g", "_mg", "_kcal", "_per"])]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

@st.cache_data(show_spinner=False)
def load_foods_optimized_sample() -> pd.DataFrame:
    """Load sample foods with optimized data types for better performance."""
    df = load_sample_foods()
    
    # Optimize data types for memory and speed
    if "fdc_id" in df.columns:
        df["fdc_id"] = df["fdc_id"].astype("int32")
    
    # Optimize string columns
    for col in ["description", "brand_owner", "food_category", "grade"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("")
    
    # Optimize numeric columns
    numeric_cols = [col for col in df.columns if any(suffix in col for suffix in ["_g", "_mg", "_kcal", "_per"])]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

@st.cache_data(show_spinner=False)
def _create_search_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create pre-processed search index for faster searching."""
    if df.empty:
        return df
    
    # Create searchable combined field
    df_indexed = df.copy()
    df_indexed["_search_text"] = (
        df["description"].fillna("").astype(str).str.lower() + " " +
        df["brand_owner"].fillna("").astype(str).str.lower()
    )
    return df_indexed

@st.cache_data(show_spinner=False)
def find_items_optimized(df: pd.DataFrame, q: str, max_rows: int = 500) -> pd.DataFrame:
    """Optimized case-insensitive substring search using pre-processed index."""
    if not q or df.empty:
        return pd.DataFrame()
    
    # Use pre-processed search index
    if "_search_text" not in df.columns:
        df = _create_search_index(df)
    
    # Split query into tokens
    needles = [token.strip().lower() for token in q.split() if token.strip()]
    if not needles:
        return pd.DataFrame()
    
    # Create combined search condition
    mask = pd.Series(True, index=df.index)
    for needle in needles:
        mask &= df["_search_text"].str.contains(needle, na=False, regex=False)
    
    results = df.loc[mask].copy()
    
    # Fallback to first token if no results
    if results.empty and needles:
        primary = needles[0]
        mask = df["_search_text"].str.contains(primary, na=False, regex=False)
        results = df.loc[mask].copy()
    
    # Remove helper column before returning
    if "_search_text" in results.columns:
        results = results.drop("_search_text", axis=1)
    
    # Only apply limit if max_rows is reasonable (to avoid memory issues)
    if max_rows and max_rows < len(results):
        return results.head(max_rows)
    else:
        return results


@st.cache_resource(show_spinner=False)
def get_service() -> AppService:
    # Load only metadata initially, full dataset when needed
    processed_dir = _resolve_processed_path("foods_nutrients.parquet").parent
    return AppService(
        dataset=pd.DataFrame(),  # Empty initially
        neighbors=load_nn(),
        importances=load_importances(),
        processed_dir=processed_dir,
    )
