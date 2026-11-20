import json
import os
from typing import Annotated as A

import numpy as np
from nano_vectordb import NanoVectorDB
from tqdm import tqdm

import agents.config as config
from agents.shared.func_call_schema import doc as D
from agents.shared.perception import perception_call
from agents.shared.utils import (
    AzureOpenAIEmbeddingService,
    LLMCallFailure,
    call_openai_model_with_tools,
)


def _get_single_query_embedding(query_text: str) -> list[float]:
    embs = AzureOpenAIEmbeddingService.get_embeddings(
        endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
        model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
        input_text=[query_text],
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
        dimensions=config.AOAI_EMBEDDING_LARGE_DIM,
    )
    if not embs or "embedding" not in embs[0]:
        raise RuntimeError(
            "Embedding request failed. Check EMBEDDING_BASE_URL / EMBEDDING_API_KEY / "
            "AOAI_EMBEDDING_LARGE_MODEL_NAME, or retry if the provider/network is unstable."
        )
    return embs[0]["embedding"]


def frame_inspect_tool(
    database: A[NanoVectorDB, D("The database containing video metadata. Must be an instance of NanoVectorDB.")], 
    question: A[str, D("The specific detailed question to ask about the video content during the specified time ranges. No need to add time ranges in the question.")], 
    time_ranges_hhmmss: A[list[tuple], D("A list of tuples containing start and end times in HH:MM:SS format. If the time range is longer than 50 seconds, the function samples 50 evenly distributed frames.  Otherwise, it uses all frames within the specified range.")], 
) -> str:
    """
    Crop the video frames based on the time ranges and ask the model a detailed question about the cropped video clips.
    Returns:
        str: The model's response to the question. If no relevant content is found within the time range,
             returns an error message: "Error: Cannot find corresponding result in the given time range."
    """

    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    video_length_secs = convert_hhmmss_to_seconds(database.get_additional_data()['video_length'])
    video_meta = database.get_additional_data()
    video_file_root = video_meta['video_file_root']
    fps = video_meta.get('fps', config.VIDEO_FPS)
    time_ranges_secs = []
    for time_range in time_ranges_hhmmss:
        start_secs = convert_hhmmss_to_seconds(time_range[0])
        end_secs = convert_hhmmss_to_seconds(time_range[1])
        if start_secs > video_length_secs:
            raise ValueError(f"One of start time {time_range[0]} exceeds video length {video_length_secs}")
        end_secs = min(end_secs, video_length_secs)
        time_ranges_secs.append((start_secs, end_secs))

    time_ranges_secs.sort(key=lambda x: x[0])  # Sort by start time
    # Calculate total time across all ranges
    total_time = sum(end - start for start, end in time_ranges_secs)
    
    # Maximum number of timepoints to sample
    max_timepoints = config.AOAI_TOOL_VLM_MAX_FRAME_NUM
    timepoints = []

    assert total_time > 0 and max_timepoints > 0 

    # ① Uniformly sample on the flattened timeline
    #    endpoint=False ensures the last sample point < total_time
    offsets = np.linspace(
        0, total_time,
        num=max_timepoints,
        endpoint=False,
        dtype=float
    )

    # ② Calculate prefix sums for each segment, used to map offsets back to actual timestamps
    prefix_len = []          # (cumulative length, segment start, segment length)
    acc = 0
    for start, end in time_ranges_secs:
        seg_len   = end - start
        prefix_len.append((acc, start, seg_len))
        acc += seg_len

    # ③ Complete the mapping
    for off in offsets:
        # off = int(round(off))          # Ensure it is an integer
        for base, seg_start, seg_len in prefix_len:
            if off < base + seg_len:   # Find the corresponding segment
                timepoints.append(seg_start + (off - base))
                break
    # ④ Convert the sampled timestamps (seconds) to frame indices according to VIDEO_FPS
    max_frame_idx = int(video_length_secs * fps) - 1   # last valid frame index

    framepoints = [
        min(max(int(round(ts * fps)), 0), max_frame_idx)  # clamp to [0, max_frame_idx]
        for ts in timepoints
    ]
    framepoints = sorted(set(framepoints))[:max_timepoints]


    input_msgs = [
        {
            "role": "system",
            "content": (
                "You are a visual-only assistant. You can inspect the provided video frames, "
                "but you cannot hear audio, recover missing speech, or infer off-screen facts "
                "that are not visibly shown."
            )
        },
        {
            "role": "user",
            "content": (
                "Carefully watch the video frames. Pay attention to the cause and sequence of "
                "events, the detail and movement of objects, visible on-screen text, and the "
                "action and pose of persons.\n\n"
                "Important constraints:\n"
                "- Use only what is visibly present in the frames.\n"
                "- Do not claim to hear dialogue, narration, music, or other audio.\n"
                "- Do not infer exact spoken wording, dates, or timing unless they are visibly "
                "shown on screen.\n"
                "- If the question depends on speech/audio content or a fact that is not visibly "
                "shown in the provided frames, return: `Error: Cannot find corresponding result "
                "in the given time range.`.\n\n"
                "Based on your observations, if you find content that can answer the question, "
                "answer concisely. If no relevant visible content is found within the given time "
                "range, return: `Error: Cannot find corresponding result in the given time "
                "range.`.\nQuestion: {question}\n"
            ),
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=question)

    files = [
        os.path.join(video_file_root, "frames", f"frame_n{fn:06d}.jpg") for fn in framepoints
    ]
    from agents.tools.frame_confirm_tool import _trim_frames_to_payload_budget
    files = _trim_frames_to_payload_budget(files)
    if config.PERCEPTION_MODE == "unified":
        _pc_result = perception_call(
            "vision",
            prompt=input_msgs[1]["content"],
            image_paths=files,
            system_prompt=input_msgs[0]["content"] if len(input_msgs) > 1 else None,
            temperature=0,
            max_tokens=512,
            caller="build_database",
        )
        if _pc_result["error"]:
            msgs = None
        else:
            msgs = {
                "content": _pc_result["content"],
                "tool_calls": _pc_result.get("tool_calls"),
                "role": "assistant",
            }
    else:
        try:
            msgs = call_openai_model_with_tools(
                messages=input_msgs,
                endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
                model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
                api_key=config.AOAI_TOOL_VLM_API_KEY,
                base_url=config.AOAI_TOOL_VLM_BASE_URL,
                image_paths=files,
                temperature=0,
                max_tokens=512,
                caller="build_database",
            )
        except LLMCallFailure:
            msgs = None
    if msgs is None:
        return (
            "Error: Visual model backend unavailable. "
            "No response from the model after retry attempts."
        )
    return msgs["content"]

def clip_search_tool(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        event_description: A[str, D("A textual description of the event to search for.")],
        top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16
) -> str:
    """
    Searches for events in a video clip database based on a given event description and retrieves the top-k most relevant video clip captions.

    Returns:
        str: A formatted string containing the concatenated captions of the searched video clip scripts.

    Notes:
        - This function utilizes the OpenAI Embedding Service to generate embeddings for the input text.
        - Use default values for `top_k` to limit the number of results returned.
    """
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = _get_single_query_embedding(event_description)
    results = database.query(
        query_emb,
        top_k=top_k,
    )
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    return "Here is the searched video clip scripts:\n\n" + captions


def global_browse_tool(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        query: A[str, D("A textual description which will be used to search for relevant video clips in the database.")],
) -> str:
    """
    Analyzes a video database to answer a detailed question by first searching for relevant video clips and then generating a comprehensive answer based on their captions.

    Args:
        database (NanoVectorDB): The database object that supports querying with embeddings.
        query (str): A textual description or question to search for in the video.

    Returns:
        str: A JSON string containing the subject registry and the model's answer to the query based on the most relevant video clips.
    """
    # search related clips
    assert isinstance(database, NanoVectorDB), "Database must be an instance of NanoVectorDB"
    query_emb = _get_single_query_embedding(query)
    results = database.query(
        query_emb,
        top_k=config.GLOBAL_BROWSE_TOPK,
    )
    captions = [
        (data['time_start_secs'], data['caption'])
        for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])

    clip_captions = "Here is the searched video clip scripts:\n\n" + captions

    # ask the question

    input_msgs = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant specializing in analyzing video content and providing detailed, insightful answers."
        },
        {
            "role": "user",
            "content": (
                "Below are descriptions of video clips, each with its corresponding timestamp. "
                "Carefully review the sequence of events, the details and movements of objects, and the actions and poses of people. "
                "Based on these observations, provide a thorough and specific answer to the following question, referencing key events and timestamps as appropriate.\n"
                "Question: {question}\n\n{clip_captions}"
            ),
        },
    ]

    input_msgs[1]['content'] = input_msgs[1]['content'].format(question=query, clip_captions=clip_captions)

    try:
        msgs = call_openai_model_with_tools(
            messages=input_msgs,
            endpoints=config.AOAI_TOOL_VLM_ENDPOINT_LIST,
            model_name=config.AOAI_TOOL_VLM_MODEL_NAME,
            api_key=config.AOAI_TOOL_VLM_API_KEY,
            base_url=config.AOAI_TOOL_VLM_BASE_URL,
            temperature=0,
            max_tokens=512,
            caller="build_database",
        )
    except LLMCallFailure:
        msgs = None
    if msgs is None:
        raise ValueError("No response from the model")

    subject_registry = database.get_additional_data()['subject_registry']
    return json.dumps({'subject_registry': subject_registry, 'query_related_event': msgs["content"]})


def convert_seconds_to_hhmmss(seconds):
    total_seconds = max(0, int(round(float(seconds))))
    hours = int(total_seconds // 3600)
    total_seconds %= 3600
    minutes = int(total_seconds // 60)
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def convert_hhmmss_to_seconds(hhmmss):
    hhmmss = hhmmss.split('.')[0]
    parts = hhmmss.split(":")
    if len(parts) < 2:
        raise ValueError("Invalid time format. Expected HH:MM:SS.")
    elif len(parts) == 2:
        parts = ["00"] + parts
    hours, minutes, seconds = map(int, parts)
    return hours * 3600 + minutes * 60 + seconds

def is_covered(d,N):
    i=sorted((int(a),int(b))for a,b in(map(lambda x:x.split('_'),d)));c=0
    return all(s==c and not (c:=e) for s,e in i) and c==N

def init_single_video_db(video_caption_json_path, output_video_db_path, emb_dim):
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    video_file_root = os.path.dirname(os.path.abspath(output_video_db_path))
    # with open(video_caption_json_path, "r") as f:
    #     captions = json.load(f)
    # subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))            
    # video_length = max([float(k.split("_")[1]) for k in captions.keys()])
    # if not is_covered(captions.keys(), video_length):
    #     error_msg = (f"Fail to build video database for video {video_caption_json_path.split("/")[-3]}. Get None video clip captions for some clips in the video.")
    #     raise ValueError(error_msg)
    # video_length = convert_seconds_to_hhmmss(video_length)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
        additional_data = vdb.get_additional_data()
        if additional_data.get("video_file_root") != video_file_root:
            additional_data["video_file_root"] = video_file_root
            vdb.store_additional_data(**additional_data)
            vdb.save()
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path)
        data = []
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        if data:
            _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))          
        video_length = max(
            (
                float(k.split("_")[1])
                for k in captions.keys()
                if isinstance(k, str) and "_" in k
            ),
            default=0.0,
        )
        video_length = convert_seconds_to_hhmmss(video_length)
        addtional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length,
            'video_file_root': video_file_root,
            'fps': getattr(config, "VIDEO_FPS", 2),
        }
        vdb.store_additional_data(**addtional_data)
        vdb.save()
    return vdb

def init_audio_only_db(
    audio_path: str,
    output_db_path: str,
    emb_dim: int,
    t_start: float | None = None,
    t_end: float | None = None,
) -> NanoVectorDB:
    """Build NanoVectorDB for audio-only asset from whisper transcription."""
    vdb = NanoVectorDB(emb_dim, storage_file=output_db_path)
    audio_file_root = os.path.dirname(os.path.abspath(output_db_path))

    if os.path.exists(output_db_path):
        print(f"Audio-only database {output_db_path} already exists.")
        additional_data = vdb.get_additional_data()
        if additional_data.get("video_file_root") != audio_file_root:
            additional_data["video_file_root"] = audio_file_root
            vdb.store_additional_data(**additional_data)
            vdb.save()
        return vdb

    # Determine effective audio path (handle t_start/t_end slicing)
    effective_audio_path = audio_path
    if t_start is not None and t_end is not None:
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)
            start_frame = int(t_start * sr)
            end_frame = int(t_end * sr)
            waveform = waveform[:, start_frame:end_frame]
            trimmed_dir = os.path.join(audio_file_root, "raw")
            os.makedirs(trimmed_dir, exist_ok=True)
            trimmed_path = os.path.join(trimmed_dir, "trimmed_audio.wav")
            torchaudio.save(trimmed_path, waveform, sr)
            effective_audio_path = trimmed_path
        except Exception as exc:
            print(f"Warning: audio slicing failed ({exc}), using full audio.")

    # Get audio duration
    try:
        import soundfile
        info = soundfile.info(effective_audio_path)
        duration = info.duration
    except Exception:
        duration = 0.0

    from agents.tools.transcript_utils import transcribe_audio_segments

    asr_cache_dir = os.path.join(audio_file_root, "audio_cache")
    os.makedirs(asr_cache_dir, exist_ok=True)
    asr_cache_path = os.path.join(asr_cache_dir, "asr_segments.json")

    segments = transcribe_audio_segments(
        str(effective_audio_path),
        str(effective_audio_path),
        str(asr_cache_path),
    )

    # Embed transcript segments and store in vdb
    data = []
    for seg in segments:
        text = seg.get("clean_text") or seg.get("spoken_text") or seg.get("raw_text", "")
        if not text.strip():
            continue
        emb = _get_single_query_embedding(text)
        t1 = convert_seconds_to_hhmmss(seg["start_secs"])
        t2 = convert_seconds_to_hhmmss(seg["end_secs"])
        prefix = f"[From {t1} to {t2} seconds]\n"
        data.append(
            {
                "__vector__": np.array(emb),
                "time_start_secs": seg["start_secs"],
                "time_end_secs": seg["end_secs"],
                "caption": prefix + text,
            }
        )
    if data:
        vdb.upsert(data)

    vdb.store_additional_data(
        video_file_root=audio_file_root,
        video_length=convert_seconds_to_hhmmss(duration),
        fps=0,
        asset_type="audio",
        raw_audio_path=str(effective_audio_path),
    )
    vdb.save()
    return vdb


def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            print(f"Empty caption information for {timestamp} in {caption_json_path}")
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            print(f"Invalid caption type for {cap_info['caption']}")
            cap_info['caption'] = str(cap_info['caption'])
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

    if not scripts:
        print(f"No usable captions found in {caption_json_path}")
        return []

    # batchify
    batch_size = max(1, config.AOAI_EMBEDDING_BATCH_SIZE)
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    for batch in tqdm(batched_scripts, desc="Embedding captions"):
        cap2emb_list.extend(single_batch_embedding_task(batch))
    return cap2emb_list

def single_batch_embedding_task(data):
    timestamps, captions, cap_infos = map(list, (zip(*data)))

    def _embed():
        return AzureOpenAIEmbeddingService.get_embeddings(
            endpoints=config.AOAI_EMBEDDING_RESOURCE_LIST,
            model_name=config.AOAI_EMBEDDING_LARGE_MODEL_NAME,
            input_text=captions,
            api_key=config.EMBEDDING_API_KEY,
            base_url=config.EMBEDDING_BASE_URL,
            dimensions=config.AOAI_EMBEDDING_LARGE_DIM,
        )

    # The retry loop below is for a provider that returned nothing this time.
    # A refusal the provider will reproduce for the identical batch is not
    # worth three more identical requests, so it fails immediately, keeping the
    # ValueError contract this function already has but naming the category.
    try:
        embs = _embed()
        max_tries = 3
        while embs is None or len(embs) != len(captions):
            max_tries -= 1
            if max_tries < 0:
                raise ValueError(f"Failed to get embeddings for {timestamps} {captions}")
            print(f"Error in embedding {timestamps} {captions} retrying...")
            embs = _embed()
    except LLMCallFailure as failure:
        raise ValueError(
            f"Embedding provider refused the batch ({failure.category}, "
            f"status={failure.status_code}) for {timestamps}"
        ) from failure
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))
