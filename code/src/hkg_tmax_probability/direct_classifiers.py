"""Direct bucket classifier wrappers."""

from hkg_tmax_probability.models import multinomial_predict, ordinal_cdf_predict

__all__ = ["multinomial_predict", "ordinal_cdf_predict"]
