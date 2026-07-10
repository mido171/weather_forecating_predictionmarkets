from __future__ import annotations

from app.extract.herbie_worker import HerbieModelConfig, extract_with_herbie
from app.extract.point_extract import VariableSpec


CONFIG = HerbieModelConfig(
    model="rap",
    product="awp130pgrb",
    default_forecast_hours=tuple(range(0, 22, 1)),
)

VARIABLE_SPECS = (
    VariableSpec("temp_2m_f", (":TMP:2 m",), conversion="kelvin_to_f"),
    VariableSpec("dewpoint_2m_f", (":DPT:2 m",), conversion="kelvin_to_f"),
    VariableSpec("wind_u_10m_ms", (":UGRD:10 m",)),
    VariableSpec("wind_v_10m_ms", (":VGRD:10 m",)),
    VariableSpec("cloud_cover_pct", (":TCDC:entire atmosphere", ":TCDC:")),
    VariableSpec("qpf_in", (":APCP:surface", ":APCP:"), conversion="mm_to_in"),
    VariableSpec("pressure_hpa", (":PRES:surface", ":PRMSL:mean sea level"), conversion="pa_to_hpa"),
)


def run(request: dict) -> dict:
    request = dict(request)
    request.setdefault("model_name", "rap")
    request.setdefault("max_forecast_hours", 21)
    return extract_with_herbie(request, "rap", CONFIG, VARIABLE_SPECS)
