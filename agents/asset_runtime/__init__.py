"""OmniVQA asset runtime exports."""

__all__ = [
    "build_grounding_cache_key",
    "clear_runtime_cache_entry",
    "consume_runtime_cache_summary",
    "get_frame_caption_index",
    "get_video_metadata_index",
    "load_grounding_cache",
    "load_asr_segments",
    "load_dialogue_segments",
    "load_subtitle_segments",
    "reset_runtime_cache_summary",
    "resolve_video_asset_runtime",
    "save_grounding_cache",
    "normalize_grounding_query",
]


def resolve_video_asset_runtime(*args, **kwargs):
    from .resolver import resolve_video_asset_runtime as impl

    return impl(*args, **kwargs)


def clear_runtime_cache_entry(*args, **kwargs):
    from .resolver import clear_runtime_cache_entry as impl

    return impl(*args, **kwargs)


def reset_runtime_cache_summary(*args, **kwargs):
    from .resolver import reset_runtime_cache_summary as impl

    return impl(*args, **kwargs)


def consume_runtime_cache_summary(*args, **kwargs):
    from .resolver import consume_runtime_cache_summary as impl

    return impl(*args, **kwargs)


def load_dialogue_segments(*args, **kwargs):
    from .transcript_store import load_dialogue_segments as impl

    return impl(*args, **kwargs)


def load_subtitle_segments(*args, **kwargs):
    from .transcript_store import load_subtitle_segments as impl

    return impl(*args, **kwargs)


def load_asr_segments(*args, **kwargs):
    from .transcript_store import load_asr_segments as impl

    return impl(*args, **kwargs)


def get_frame_caption_index(*args, **kwargs):
    from .inference_store import get_frame_caption_index as impl

    return impl(*args, **kwargs)


def get_video_metadata_index(*args, **kwargs):
    from .metadata_store import get_video_metadata_index as impl

    return impl(*args, **kwargs)


def load_grounding_cache(*args, **kwargs):
    from .grounding_store import load_grounding_cache as impl

    return impl(*args, **kwargs)


def save_grounding_cache(*args, **kwargs):
    from .grounding_store import save_grounding_cache as impl

    return impl(*args, **kwargs)


def build_grounding_cache_key(*args, **kwargs):
    from .grounding_store import build_grounding_cache_key as impl

    return impl(*args, **kwargs)


def normalize_grounding_query(*args, **kwargs):
    from .grounding_store import normalize_grounding_query as impl

    return impl(*args, **kwargs)
