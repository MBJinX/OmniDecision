"""OmniVQA tool wrappers for grounding and verification."""

__all__ = [
    "audio_scout_tool",
    "clip_grounding_tool",
    "frame_confirm_tool",
    "subtitle_grounding_tool",
    "web_search_tool",
]


def audio_event_locator_tool(*args, **kwargs):
    from .audio_event_locator_tool import audio_event_locator_tool as impl

    return impl(*args, **kwargs)


def audio_scout_tool(*args, **kwargs):
    from .audio_scout_tool import audio_scout_tool as impl

    return impl(*args, **kwargs)


def clip_grounding_tool(*args, **kwargs):
    from .clip_grounding_tool import clip_grounding_tool as impl

    return impl(*args, **kwargs)


def frame_confirm_tool(*args, **kwargs):
    from .frame_confirm_tool import frame_confirm_tool as impl

    return impl(*args, **kwargs)


def subtitle_grounding_tool(*args, **kwargs):
    from .subtitle_grounding_tool import subtitle_grounding_tool as impl

    return impl(*args, **kwargs)


def web_search_tool(*args, **kwargs):
    from .web_search_tool import web_search_tool as impl

    return impl(*args, **kwargs)
