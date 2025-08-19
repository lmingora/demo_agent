# DEPRECATED: Compatibilidad transitoria. Reemplazado por HybridRouter.
import warnings
from .router_hybrid import HybridRouter as KeywordRouter

warnings.warn(
    "KeywordRouter está deprecado. Usá HybridRouter "
    "desde src.orchestrator.router_hybrid import HybridRouter",
    DeprecationWarning,
    stacklevel=2,
)
