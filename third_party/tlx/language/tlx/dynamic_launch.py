import triton.language.core as tl

from . import types as tlx

# Blackwell-only
@tl.builtin
def clc_issue(
    clc_response: tlx.clc_response,
    barrier: tlx.mbarrier,
    _semantic=None,
):
    """
    Issue async `clusterlaunchcontrol.try_cancel` request for
    CTA ID of available cluster
    """
    return _semantic.builder.clc_issue(clc_response, barrier)


# Blackwell-only
@tl.builtin
def clc_query(
    clc_response: tlx.clc_response,
    _semantic=None,
) -> tl.base_value:
    """
    (blocking) Wait for barrier to arrive and parse CTA ID from CLC response
    """
    return _semantic.builder.clc_query(clc_response)
