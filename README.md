<p align="center">
  <a href="https://mbjinx.github.io/omnidecision_web/"><img src="docs/assets/omnidecision.svg" width="90" height="90" alt="Omni-Decision project website"></a>
</p>

<h1 align="center">Omni-Decision</h1>
<p align="center"><strong>Evidence-Ledger Planning for Omni-Modal Agents</strong></p>
<p align="center">
  Ming Ma · Yi Zhu · Yiran Zhong · Feida Zhu · Yuhao Wang<br>
  Junhan Shi · Lingrui Mei · Tianming Yang · Steven Hoi
</p>

<p align="center">
  <a href="https://mbjinx.github.io/omnidecision_web/"><strong>Project website</strong></a> ·
  <a href="https://arxiv.org/abs/2607.11433"><strong>arXiv</strong></a> ·
  <a href="https://arxiv.org/pdf/2607.11433">PDF</a> ·
  <a href="https://mbjinx.github.io/omnidecision_web/#demo">Interactive walkthrough</a> ·
  <a href="https://mbjinx.github.io/omnidecision_web/#results">Results</a>
</p>

Omni-Decision answers questions about long videos and images by keeping an
**evidence ledger** for each question. The planner reads what is still missing
and chooses a tool. The critic checks the observation, and the reducer records
which needs it resolves and which conflicts remain. The next decision starts
from that updated ledger.

<p align="center">
  <a href="https://arxiv.org/html/2607.11433v3#S2.F2"><img src="docs/assets/control-loop.png" width="761" alt="The planner reads open needs from the ledger and calls a tool. The critic validates its observation, and the reducer updates the ledger before the agent continues or answers."></a>
</p>

*Figure 2 from the paper.* This release includes video and audio grounding,
frame verification, and optional web search. The walkthrough on the project
website lets you inspect the ledger after each step.

[Installation](#install-and-configure) · [Run a question](#run-your-own-task) · [Local trace viewer](#local-trace-viewer) · [Source guide](#source-guide) · [Citation](#citation)

## Install and configure

From the repository root, use Python 3.11 and install FFmpeg (including
`ffprobe`) on your PATH for audio extraction. For example, on Ubuntu run
`sudo apt-get install ffmpeg`.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your own credentials and the model IDs supported by your
provider. Shell variables take precedence. The template includes example model
IDs, and you can point each service at a separate OpenAI-compatible API.

| Setting | Required capability |
| --- | --- |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PLANNER_MODEL` | Chat completions, tool calling and JSON responses. Use an API root ending in `/v1`. |
| `PERCEPTION_MODEL` | Image input for frame captioning and inspection. Audio tasks also require a model accepting `input_audio`. Blank `PERCEPTION_BASE_URL` and `PERCEPTION_API_KEY` reuse the main connection. |
| `OMNIVQA_EMBEDDING_MODEL`, `AOAI_EMBEDDING_LARGE_DIM` | Video indexing and transcript retrieval use the embeddings API with the configured output dimension. Blank `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` reuse the main connection. Image-only tasks can run without embeddings. |

## Run your own task

Use a local video file. Place an optional same-name subtitle file beside the video
(e.g. `inputs/video.srt`) to use its dialogue during captioning and retrieval.

```bash
python scripts/run_single.py --video inputs/video.mp4 \
  --question "What happens immediately after the person opens the door?"
```

For an image-only task, or a question comparing multiple assets, repeat
`--image` or `--video` as needed.

```bash
python scripts/run_single.py --image inputs/photo.jpg --question "What is on the table?"
python scripts/run_single.py --video inputs/video.mp4 --image inputs/photo.jpg \
  --question "When does the object in the photo first appear in the video?"
```

<details>
<summary>Download a video from your own URL</summary>

Point the script at a local file. For a remote video, download it from a source
you are permitted to use. This shell example prompts for your media URL.

```bash
mkdir -p inputs
read -r -p "Video download URL: " VIDEO_URL
curl --fail --location "$VIDEO_URL" --output inputs/video.mp4
```

</details>

The runner prints the answer and its run directory. Each invocation creates an
independent directory under `outputs/` with `result.json`, `step_trace.jsonl`
and, for videos, sampled frames, captions and a vector index. Use
`--run-dir PATH` to select a new or empty directory and `--max-iterations N`
to set the decision-loop limit. A result with kind `insufficient_evidence`
records the agent's abstention.

## Local trace viewer

```bash
python -m gui.server
```

Open **http://127.0.0.1:13000** on the same machine. In **Run**, enter the path
to your video or image, type a question and launch the task. The viewer streams
the planner, tool and critic steps and displays the final result. Use **Load
Log** to replay a local `step_trace.jsonl` under `outputs/`. To replay traces
from another directory, set the shell variable `OMNIVQA_RUNS_ROOT` to that
directory before launching the viewer.

## Data and optional web evidence

Supply your own media. Questions, sampled frames, captions and relevant
transcripts are sent to the model endpoints you configure. Audio tools can
also send audio to the perception endpoint. Run traces can contain prompts,
responses and local file paths. Keep your `.env`, `inputs/` and `outputs/`
private. The viewer binds to loopback by default and is intended for local use.

To enable external fact lookup, install `tavily-python` in the same environment
(`python -m pip install tavily-python`) and set `OMNIVQA_TAVILY_API_KEY`.
This sends search queries to Tavily and selected page URLs to Jina Reader.
The tool is disabled when the search key is blank.

## Source guide

| Component | Source |
| --- | --- |
| Planner, tool dispatch and stopping | [agents/core/agent.py](agents/core/agent.py) |
| Evidence reducer and state schemas | [agents/state/state_machine.py](agents/state/state_machine.py), [agents/shared/types.py](agents/shared/types.py) |
| Question profiling and critics | [agents/planning/](agents/planning/), [prompts/agent.yaml](prompts/agent.yaml) |
| Media preprocessing, retrieval and verification | [agents/frame_caption.py](agents/frame_caption.py), [agents/tools/](agents/tools/) |
| Task runner and local UI | [scripts/run_single.py](scripts/run_single.py), [gui/](gui/) |

## Citation

```bibtex
@article{ma2026omnidecision,
  title   = {Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents},
  author  = {Ma, Ming and Zhu, Yi and Zhong, Yiran and Zhu, Feida and Wang, Yuhao and
             Shi, Junhan and Mei, Lingrui and Yang, Tianming and Hoi, Steven},
  journal = {arXiv preprint arXiv:2607.11433},
  year    = {2026}
}
```

## License

Original project code is released under the [MIT License](LICENSE). The schema
helper in `agents/shared/func_call_schema.py` includes an adaptation of
Peter Roelants' annotated-docs and retains its [Apache 2.0 license](licenses/annotated-docs-Apache-2.0.txt).
