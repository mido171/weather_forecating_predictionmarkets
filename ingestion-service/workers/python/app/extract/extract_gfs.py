from __future__ import annotations

from app.extract.herbie_worker import HerbieModelConfig, extract_with_herbie
from app.extract.point_extract import VariableSpec


CONFIG = HerbieModelConfig(
    model="gfs",
    product="pgrb2.0p25",
    default_forecast_hours=tuple(range(0, 49, 3)),
)

VARIABLE_SPECS = (
    VariableSpec("temp_2m_f", (":TMP:2 m",), conversion="kelvin_to_f"),
    VariableSpec("dewpoint_2m_f", (":DPT:2 m",), conversion="kelvin_to_f"),
    VariableSpec("wind_u_10m_ms", (":UGRD:10 m",)),
    VariableSpec("wind_v_10m_ms", (":VGRD:10 m",)),
    VariableSpec("cloud_cover_pct", (":TCDC:entire atmosphere", ":TCDC:")),
    VariableSpec("qpf_in", (":APCP:surface", ":APCP:"), conversion="mm_to_in"),
    VariableSpec("pressure_hpa", (":PRMSL:mean sea level", ":PRES:surface", ":MSLET:mean sea level"), conversion="pa_to_hpa"),
    VariableSpec("tmax_2m_f", (":TMAX:2 m",), conversion="kelvin_to_f"),
)


def run(request: dict) -> dict:
    request = dict(request)
    request.setdefault("model_name", "gfs")
    request.setdefault("max_forecast_hours", 48)
    return extract_with_herbie(request, "gfs", CONFIG, VARIABLE_SPECS)
