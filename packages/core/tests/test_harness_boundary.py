"""Alias for harness boundary test.

P1-PY-002: harness must never import neuroagent_app.
"""

from test_harness_standalone import test_no_imports_from_neuroagent_app as test_boundary_no_app_import  # noqa: F811
