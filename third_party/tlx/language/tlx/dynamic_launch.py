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
    return _semantic.builder.create_clc_try_cancel(clc_response, barrier)


# Blackwell-only
# @tl.builtin
# def clc_query_cancel(
#     clc_response: tlx.clc_response,
#     _semantic=None,
# ):
#     """
#     (blocking) Wait for barrier to arrive and parse CTA ID from CLC response
#     """
#     return _semantic.builder.create_clc_try_cancel(clc_response, barrier)
