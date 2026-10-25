# VoiceOps 2026 - ZentroMart Support Agent

This project builds a customer-support agent in four stages: an LLM-only
baseline, policy-grounded RAG, a stateful tool-aware conversation agent, and a
Gnani STT/TTS voice layer. The reference company and customer/order records are
synthetic.

Public repository: https://github.com/Ansh13579/zentromart_RAG

## Project layout

The supplied dataset must be extracted beside this repository as
`../Capstone_Project_Dataset_FTTT_2026/`, with `kb/`, `data/`, `practice/`, and
`tools/` inside it. The dataset is not copied into this repository.

Important files:

- `run.py`: evaluator batch entry point; each audio file is an independent call.
- `run_voice_chat.py`: records and plays a multi-turn microphone conversation
  while preserving conversation state between turns.
- `run_voice.py`: processes one audio file.
- `src/`: RAG, data access, agent, conversation, voice, and evaluation logic.
- `indexes/`: generated knowledge-base chunks and FAISS index.
- `evaluations/`: metrics, traces, and failure-analysis output.
- `sample_outputs/`: example text/audio replies and retrieval traces.

## Setup on Windows PowerShell

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally and add your own provider credentials. Choose one LLM
provider with `LLM_PROVIDER` (`ifm`, `gemini`, or `nvidia`) and set its API key.
Set `GNANI_API_KEY` for speech input/output. Never commit `.env` or paste its
contents into a report, issue, or chat.

## Validate data and build the knowledge index

```powershell
.\.venv\Scripts\python.exe ..\Capstone_Project_Dataset_FTTT_2026\validate_dataset.py
.\.venv\Scripts\python.exe .\src\knowledge_base.py
.\.venv\Scripts\python.exe .\src\retriever.py
```

The index is generated locally from the supplied Markdown policy documents.
Retrieval uses FAISS plus BM25, then reranks up to 25 candidates with a local
multilingual MiniLM cross-encoder before selecting the top five evidence
sections. The reranker loads on CPU and is downloaded from Hugging Face on the
first RAG request (about 471 MB). It reuses the existing
`sentence-transformers` dependency, makes no model API calls, and does not need
the FAISS index rebuilt. Set `RERANKER_ENABLED=false` in `.env` to use the
semantic-plus-BM25 ranking without the cross-encoder. Rebuild the FAISS index
after changing the knowledge-base documents or embedding model.

## Run the evaluator batch command

Put supported audio files (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`,
`.aac`, or `.aiff`) in an input directory, then run:

```powershell
.\.venv\Scripts\python.exe .\run.py --input_folder .\input_folder --output_folder .\sample_outputs
```

The command creates one output directory per input file. Each successful call
contains `transcript.txt`, `response.txt`, `response.wav`, `sources.json`, and
`trace.json`. Sources contain complete evidence text and mark which sections
the model cited; supporting calendar rules are identified separately from
ranked retrieval results. Confirmed handovers include the ticket ID and actual
tool invocation in the trace. The batch summary records successes, failures, empty-transcript
recovery, and speech-output creation. Files longer than the configured Gnani
STT limit are reported as failures. To choose another destination, change the
`--output_folder` value.

## Have a live multi-turn voice conversation

The batch evaluator intentionally resets conversation state for each input
file. For a spoken follow-up conversation, use the interactive client:

```powershell
.\.venv\Scripts\python.exe .\run_voice_chat.py --seconds 8
```

Press Enter to record each turn; enter `q` to quit. The agent keeps the same
conversation session and plays each Gnani-generated reply. Turn artifacts are
saved under `sample_outputs/live_sessions/`. Microphone permissions and an
available input/output audio device are required.

For English mixed with Hindi, set `GNANI_STT_LANGUAGE=en-IN`. For Hindi speech,
set it to `hi-IN`. Set `GNANI_TTS_LANGUAGE` to a Gnani-supported output
language such as `en-IN`, `hi-IN`, or `hi-en` for Hinglish. Replies remain
grounded in retrieved policy sections and trusted tool results.

## Run the text evaluations

```powershell
.\.venv\Scripts\python.exe .\src\evaluation.py
```

The evaluation runs all 52 practice questions for Step 1 and Step 2, followed
by all 23 conversations (96 turns). It writes per-step metrics and failures
under `evaluations/`, including `failure_analysis.json` with the input,
expected behavior, actual behavior, failure category, likely root cause, and
planned fix. LLM evaluation makes provider requests and can consume quota.

To build the failure analysis from already-saved evaluation JSON without making
LLM requests, run:

```powershell
.\.venv\Scripts\python.exe .\src\evaluation.py --failures-only
```

If only evaluator text-normalization rules changed, recalculate Step 2 scores
from the saved responses without sending new LLM requests:

```powershell
.\.venv\Scripts\python.exe .\src\evaluation.py --recheck-saved-step2
```

After the final evaluation, refresh the four-page report from the saved metrics:

```powershell
.\.venv\Scripts\python.exe .\src\build_report.py
```

Individual runs:

```powershell
.\.venv\Scripts\python.exe .\src\evaluation.py --step1-only
.\.venv\Scripts\python.exe .\src\evaluation.py --step2-only
.\.venv\Scripts\python.exe .\src\evaluation.py --conversations-only
```

## Evaluation snapshot and remaining work

The saved evaluation snapshot passed 4/52 Step 1 questions, 32/52
Step 2 questions, and 74/96 conversation turns (6/23 complete conversations).
The previous batch run processed 8 audio files with 8 successes and 0 failures.
See `evaluations/*_metrics.json` for exact scores and evaluation timestamps.
The saved Step 1 run is from 24 October 2026; Step 2 and conversation runs
are from 25 October 2026. Local regression checks cover the implementation
fixes. Rerun the hosted-model evaluations after code changes and use the
resulting artifacts in the final report.

Refund deadlines use payment-method ranges parsed from the current refund
policy, excluding the initiation day, Sundays and `holidays.csv` dates. Return
dates use calendar-day windows parsed from the return policy, including its
historical electronics and festive fashion exceptions. Calculated dates do
not establish item-condition eligibility by themselves.

To run the local regression checks without provider or speech API requests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The remaining quality work is to review the new failure analysis, improve
answerable-question retrieval and required-fact coverage, and test clear,
empty/unclear, English, Hindi, Hinglish, noisy, and follow-up speech. The
project brief permits your own recorded voice for testing; do not submit real
customer audio or third-party recordings as project customer data.

## Submission hygiene

The brief requires a public repository, `.env.example`, requirements, a run
command, a report of at most four pages, a demo video of at most two minutes,
and sample outputs. It says not to include `.env`, API keys, virtual
environments, or a dataset copy. Do not include raw personal recordings in a
submission. Regenerate `report.pdf` after the final evaluation. Record the live
demo using your own voice, showing a grounded answer, a contextual follow-up,
and one handover or an honest "I do not know" response.
