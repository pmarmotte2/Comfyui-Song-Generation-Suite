"""
Song Generation Suite (ComfyUI custom nodes)

This extension merges:
1) Song Writer (Standalone) - generates an English music prompt (tags) + lyrics in the selected language via a local HF LLM.
2) Ace Step helper nodes (1.0 / 1.5) - encoders + latent audio + reference timbre.

Additionally, it provides a convenience node:
- SongGenerationSuite: runs Song Writer, then immediately encodes conditioning for AceStep (with bpm/duration/timesignature/language/keyscale when supported).

Install:
- Put this folder into: ComfyUI/custom_nodes/SongGenerationSuite/
- Restart ComfyUI.

Notes:
- First run may download LLM weights from HuggingFace.
- Models cached under: ComfyUI/models/LLM/SongWriter/<model_id_sanitized>

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Tuple
import gc
import re

import torch

import node_helpers
import comfy.model_management
import folder_paths

# Optional deps (download + transformers)
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None  # type: ignore

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore


# -----------------------------
# Utilities (Song Writer)
# -----------------------------

def _norm(x) -> str:
    return "" if x is None else str(x).strip()

def _compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _safe_model_dirname(model_id: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip())
    return t[:180] if len(t) > 180 else t

def _strip_music_prompt_prefix(s: str) -> str:
    if not s:
        return ""
    t = s.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    header_patterns = [
        r"(?im)^\s*\[\s*music\s*prompt\s*\]\s*$\n?",
        r"(?im)^\s*<\s*music\s*prompt\s*>\s*$\n?",
        r"(?im)^\s*music\s*prompt\s*:?\s*$\n?",
        r"(?im)^\s*\[\s*detailed\s+music\s+prompt\s*\]\s*$\n?",
        r"(?im)^\s*<\s*detailed\s+music\s+prompt\s*>\s*$\n?",
        r"(?im)^\s*detailed\s+music\s+prompt\s*:?\s*$\n?",
    ]
    for pat in header_patterns:
        t2 = re.sub(pat, "", t, count=1)
        if t2 != t:
            t = t2.lstrip()
            break
    return t.strip()

def _split_music_prompt_and_lyrics(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    marker = re.search(
        r"(?im)^\s*(?:\[\s*Lyrics\s*\]|Lyrics)\s*:?\s*$\n",
        t,
        flags=re.MULTILINE,
    )
    if marker:
        music_prompt = t[:marker.start()].strip()
        lyrics = t[marker.end():].strip()
        return music_prompt, lyrics

    section = re.search(
        r"(?im)^\s*\[(intro|verse\s*1|verse\s*2|chorus|solo\s*or\s*bridge|bridge|outro)\]\s*$",
        t,
        flags=re.MULTILINE,
    )
    if section:
        music_prompt = t[:section.start()].strip()
        lyrics = t[section.start():].strip()
        return music_prompt, lyrics

    return t, ""

def _cleanup_lyrics_text(lyrics: str) -> str:
    if not lyrics:
        return ""
    t = lyrics.replace("\r\n", "\n").replace("\r", "\n")

    t = re.sub(r"(?m)^\s*[-–—]{3,}\s*$", "", t)
    t = re.sub(r"(?im)^\s*\[\s*lyrics\s*\]\s*$", "", t)

    t = re.sub(r"(?im)^\s*\[\s*vers\s*1\s*\]\s*$", "[Verse 1]", t)
    t = re.sub(r"(?im)^\s*\[\s*vers\s*2\s*\]\s*$", "[Verse 2]", t)
    t = re.sub(r"(?im)^\s*\[\s*refrain\s*\]\s*$", "[Chorus]", t)
    t = re.sub(r"(?im)^\s*\[\s*chorus\s*\]\s*$", "[Chorus]", t)
    t = re.sub(r"(?im)^\s*\[\s*bridge\s*\]\s*$", "[Solo or Bridge]", t)

    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def _enforce_lyrics_tag_order(lyrics: str) -> str:
    if not lyrics:
        return ""

    allowed = [
        "[Intro]",
        "[Verse 1]",
        "[Chorus]",
        "[Verse 2]",
        "[Solo or Bridge]",
        "[Outro]",
    ]

    lines = lyrics.splitlines()
    kept = []
    for line in lines:
        l = line.strip()
        if l.startswith("[") and l.endswith("]"):
            if l in allowed:
                kept.append(l)
            continue
        kept.append(line)

    out = "\n".join(kept)

    mm = re.search(r"(?m)^\[Outro\]\s*$", out)
    if mm:
        before = out[: mm.start()].rstrip()
        after_lines = out[mm.start():].strip().splitlines()
        outro = after_lines[: 1 + 6]
        out = (before + "\n\n" + "\n".join(outro)).strip()

    return out.strip()


def _looks_like_english(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    t = re.sub(r"\[[^\]]+\]", " ", t)
    markers = [" the ", " you ", " your ", " and ", " under ", " with ", " this ", " that ", " was ", " were ", " but "]
    hits = sum(t.count(m) for m in markers)
    return hits >= 6



def _force_instrumental_solo_bridge(lyrics: str) -> str:
    """
    Ensures [Solo or Bridge] contains only a short placeholder line.
    If it contains multiple lines or looks like lyrical/descriptive sentences, replace content with "(Instrumental)".
    """
    if not lyrics:
        return ""
    t = lyrics.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "[Solo or Bridge]":
            out.append(line)
            i += 1
            content = []
            while i < len(lines) and not (lines[i].strip().startswith("[") and lines[i].strip().endswith("]")):
                content.append(lines[i])
                i += 1

            joined = "\n".join([c for c in content if c.strip()])
            if not joined.strip():
                out.append("(Instrumental)")
            else:
                non_empty = [c for c in content if c.strip()]
                looks_sentence = bool(re.search(r"[.!?;:]", joined)) or len(non_empty) > 1 or len(joined.split()) > 6
                if looks_sentence:
                    out.append("(Instrumental)")
                else:
                    allowed = joined.strip()
                    if allowed.lower() in ("(instrumental)", "(solo instrumental)"):
                        out.append(allowed)
                    else:
                        out.append("(Instrumental)")
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip()



def _lyrics_is_effectively_empty(lyrics: str) -> bool:
    if not lyrics:
        return True
    t = lyrics.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not t:
        return True
    t2 = re.sub(r"\[[^\]]+\]", " ", t)
    t2 = re.sub(r"\(instrumental\)|\(solo instrumental\)", " ", t2, flags=re.IGNORECASE)
    t2 = re.sub(r"\s+", " ", t2).strip()
    return len(t2) == 0

def _make_lyrics_only_retry_prompt(lyrics_language_name: str, base_topic: str, base_style: str, base_sex: str) -> str:
    return f"""Write complete song LYRICS in {lyrics_language_name}.

STRICT RULES:
- Output ONLY the lyrics (tags + lines). Do NOT output any music prompt. Do NOT output a 'Lyrics' separator line.
- Use EXACTLY these tags and EXACT order:
  [Intro]
  [Verse 1]
  [Chorus]
  [Verse 2]
  [Chorus]
  [Solo or Bridge]
  [Chorus]
  [Outro]
- [Solo or Bridge] MUST be instrumental only: output exactly '(Instrumental)' as the only line under that tag.
- [Outro] max 6 lines. End immediately after [Outro].
- Lyrics MUST be written EXCLUSIVELY in {lyrics_language_name}.

CONTEXT (do NOT repeat verbatim):
- Topic: {base_topic if base_topic else "(none)"}
- Music style/genre: {base_style if base_style else "(none)"}
- Lead singer sex: {base_sex if base_sex else "(not specified)"}
"""


def _make_song_prompt(
    lyrics_language_name: str,
    lead_singer_sex: str,
    song_topic: str,
    music_style: str,
    lyrics: str,
    strict_user_lyrics: bool,
) -> str:
    """
    IMPORTANT REQUIREMENT:
    - MUSIC PROMPT: ALWAYS IN ENGLISH
    - LYRICS: in lyrics_language_name (only lyrics are affected by language)
    """
    lyrics_language_name = _norm(lyrics_language_name) or "English"
    lead_singer_sex = _norm(lead_singer_sex) or "Not specified"
    song_topic = _norm(song_topic)
    music_style = _compact_spaces(music_style) or "Not specified"
    lyrics = _norm(lyrics)

    user_lyrics_is_empty = "YES" if not lyrics else "NO"
    lyrics_block = lyrics if lyrics else "[EMPTY]"

    return f"""You are a professional songwriter and music producer.

LANGUAGE RULES (VERY IMPORTANT):
- STRICT LANGUAGE ENFORCEMENT (CRITICAL):
  - The LYRICS must be written EXCLUSIVELY in the requested language. If the requested language is not English, you MUST NOT output any English lines.
  - Do NOT keep the user's input language if it differs from the requested language. Always comply with the requested language.
  - If you accidentally produce lyrics in the wrong language, rewrite them into the requested language before finalizing.
- The MUSIC PROMPT must ALWAYS be written in ENGLISH, regardless of the requested lyrics language.
- The LYRICS must be written in: {lyrics_language_name}.
- Do NOT mix languages inside the music prompt. Do NOT translate the music prompt.

INPUTS:
- Lead singer sex: {lead_singer_sex}
- Music style / genre: {music_style}
- Topic: {song_topic if song_topic else "(No topic provided.)"}
- User lyrics is empty: {user_lyrics_is_empty}
- strict_user_lyrics: {"TRUE" if strict_user_lyrics else "FALSE"}
- User lyrics: {lyrics_block}

TASK:
Generate:
1) A detailed music generation prompt (production-level, vivid, precise) -> ENGLISH ONLY
2) Complete song lyrics -> in {lyrics_language_name}

RULES:
- If User lyrics is EMPTY, write original lyrics from scratch in {lyrics_language_name}.
- If User lyrics is provided:
  - If strict_user_lyrics is TRUE: you MUST reproduce the user's lyrics VERBATIM (character-for-character), preserving all line breaks, spacing, punctuation, and wording. Do NOT rewrite, translate, or "polish" anything. Only add the required bracket section tags if they are missing, without modifying the lyric lines.
  - If strict_user_lyrics is FALSE: keep the meaning and overall structure, but you may lightly polish for flow/rhyme. Do NOT change the topic. Do NOT add unrelated sections.
- Lyrics MUST use exactly these bracket tags and order:
  [Intro]
  [Verse 1]
  [Pre-Chorus]
  [Chorus]
  [Verse 2]
  [Pre-Chorus]
  [Chorus]
  [Verse 3]
  [Chorus]
  [Solo or Bridge]
  [Chorus]
  [Outro]
- The [Solo or Bridge] section must be INSTRUMENTAL ONLY: output a single line '(Instrumental)' (or '(Solo instrumental)') and nothing else under that tag.
- Do not repeat the same line more than 3 times.
- Each Verse should be 8–12 lines.
- Each Pre-Chorus should be 4–6 lines.
- Each Chorus should be 6–8 lines.
- The [Outro] must be 6–10 lines.
- End immediately after the [Outro].
MUSIC PROMPT must describe (IN ENGLISH):
- energy and tempo feel (no BPM number needed)
- core instrumentation and arrangement
- vocal characteristics matching the singer sex (range, texture, delivery)
- structure and key moments (drops, build, solo/bridge, outro)
- production / mix (clean, punchy, saturated, warm, airy, etc.)
- mood and artistic intent (no copyrighted lyric references)

OUTPUT RULES (STRICT):
+CRITICAL LANGUAGE ENFORCEMENT:
+- Lyrics MUST be written EXCLUSIVELY in {lyrics_language_name}.
+- If you output lyrics in English while {lyrics_language_name} is selected, the output is INVALID.
+- Do NOT use any English words, expressions, or grammar in the lyrics unless the selected language is English.
+- This rule has priority over style, rhyme, or musical constraints.
- Output MUST contain exactly 2 parts:
  PART A: MUSIC PROMPT (English only) as plain text. Do NOT write any header like "MUSIC PROMPT" or "**MUSIC PROMPT**".
PART B: LYRICS in {lyrics_language_name}.
+Start the lyrics thinking and writing as a native {lyrics_language_name} songwriter.

- Between Part A and Part B, write EXACTLY this single line (no markdown, no extra spaces):
Lyrics

LYRICS TAGS (STRICT):
- SOLO/BRIDGE RULE (CRITICAL): The [Solo or Bridge] section must be INSTRUMENTAL ONLY.
  - Output ONLY one of these exact lines under [Solo or Bridge]:
    (Instrumental)
    (Solo instrumental)
  - Do NOT write descriptive sentences, narration, or any sung lyrics in [Solo or Bridge].
- You MUST use ONLY these tags, exactly spelled and in this exact order:
  [Intro]
  [Verse 1]
  [Chorus]
  [Verse 2]
  [Chorus]
  [Solo or Bridge]
  [Chorus]
  [Outro]

- Do NOT invent any other tags. Forbidden examples: [Bridge], [Verse], [Vers 2], [Refrain], [Lyrics], or any other bracket tag not listed above.
- NEVER output a line that is exactly "[Lyrics]" (this is forbidden).
- Do not use markdown separators like "---" anywhere.
- End immediately after the [Outro] section (max 6 lines).
""".strip()


# -----------------------------
# Language dropdown
# -----------------------------
LANGUAGE_NAME_TO_CODE = {
    "English": "en",
    "Japanese": "ja",
    "Chinese": "zh",
    "Spanish": "es",
    "German": "de",
    "French": "fr",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Czech": "cs",
    "Persian": "fa",
    "Indonesian": "id",
    "Korean": "ko",
    "Ukrainian": "uk",
    "Hungarian": "hu",
    "Arabic": "ar",
    "Swedish": "sv",
    "Romanian": "ro",
    "Greek": "el",
}
LANGUAGE_DROPDOWN = list(LANGUAGE_NAME_TO_CODE.keys())


# -----------------------------
# Standalone model configs
# -----------------------------
DEFAULT_MODELS: Dict[str, str] = {
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}

class Quantization(str, Enum):
    FP16 = "None (FP16)"
    Q8 = "8-bit (Balanced)"
    Q4 = "4-bit (VRAM-friendly)"

    @classmethod
    def get_values(cls):
        return [item.value for item in cls]

    @classmethod
    def from_value(cls, value: str) -> "Quantization":
        for item in cls:
            if item.value == value:
                return item
        return cls.FP16

ATTENTION_MODES = ["auto", "flash_attention_2", "sdpa"]

def _flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8

def _resolve_attention_mode(mode: str) -> str:
    mode = _norm(mode) or "auto"
    if mode == "sdpa":
        return "sdpa"
    if mode == "flash_attention_2":
        return "flash_attention_2" if _flash_attn_available() else "sdpa"
    return "flash_attention_2" if _flash_attn_available() else "sdpa"

def _ensure_model_local(repo_id: str) -> str:
    if snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed, cannot download models. "
            "Install it in your ComfyUI python env."
        )
    models_dir = Path(folder_paths.models_dir) / "LLM" / "SongWriter"
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / _safe_model_dirname(repo_id)

    if (target / "config.json").exists() or any(target.glob("*.safetensors")):
        return str(target)

    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", ".git*"],
    )
    return str(target)

def _bnb_config(quant: Quantization):
    if BitsAndBytesConfig is None:
        if quant in (Quantization.Q4, Quantization.Q8):
            raise RuntimeError(
                "bitsandbytes/transformers BitsAndBytesConfig not available. "
                "Install bitsandbytes for 4-bit/8-bit quantization, or use FP16."
            )
        return None
    if quant == Quantization.Q4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quant == Quantization.Q8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

@dataclass
class _LoadedModel:
    model_id: str
    quant: str
    attn: str
    device: str
    compiled: bool

class _TextLLMEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.signature: _LoadedModel | None = None

    def clear(self):
        self.model = None
        self.tokenizer = None
        self.signature = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(
        self,
        repo_id: str,
        quantization: str,
        attention_mode: str,
        use_torch_compile: bool,
        device_choice: str,
        keep_model_loaded: bool,
    ):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed (AutoModelForCausalLM/AutoTokenizer missing). "
                "Install transformers in your ComfyUI python env."
            )

        quant = Quantization.from_value(quantization)
        attn_impl = _resolve_attention_mode(attention_mode)

        device_choice = _norm(device_choice) or "auto"
        if device_choice == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_choice

        sig = _LoadedModel(
            model_id=repo_id,
            quant=quant.value,
            attn=attn_impl,
            device=device,
            compiled=bool(use_torch_compile),
        )

        if keep_model_loaded and self.model is not None and self.signature == sig:
            return

        self.clear()
        local_path = _ensure_model_local(repo_id)

        load_kwargs = {
            "device_map": {"": 0} if device == "cuda" and torch.cuda.is_available() else device,
            "attn_implementation": attn_impl,
            "use_safetensors": True,
        }

        bnb = _bnb_config(quant)
        if bnb is not None:
            load_kwargs["quantization_config"] = bnb
        else:
            load_kwargs["torch_dtype"] = torch.float16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True, **load_kwargs).eval()

        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.use_cache = True

        if use_torch_compile and device == "cuda" and torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass

        self.signature = sig

    @torch.no_grad()
    def generate(
        self,
        prompt_text,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
    ):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        torch.manual_seed(int(seed))

        messages = [{"role": "user", "content": prompt_text}]
        attention_mask = None

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            attention_mask = torch.ones_like(input_ids)
        except Exception:
            enc = self.tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask

        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "repetition_penalty": float(repetition_penalty),
            "num_beams": int(num_beams),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if int(num_beams) == 1:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
        else:
            gen_kwargs["do_sample"] = False

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        new_tokens = outputs[0, input_ids.shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return (text or "").strip()

_ENGINE = _TextLLMEngine()

def write(
        song_topic,
        lead_singer_sex,
        music_style,
        music_style_custom,
        lyrics,
        language,
        strict_user_lyrics,
        model,
        custom_model_repo_id,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        keep_model_loaded,
    ) -> Tuple[str, str, str, str, str]:


        model = _norm(model)
        if model == "Custom (HF repo id)":
            repo_id = _norm(custom_model_repo_id)
            if not repo_id:
                raise RuntimeError("Custom model selected but custom_model_repo_id is empty.")
        else:
            repo_id = DEFAULT_MODELS.get(model, "")
            if not repo_id:
                raise RuntimeError(f"Unknown model label: {model}")

        topic = _norm(song_topic)
        sex = _norm(lead_singer_sex)
        lyr = _norm(lyrics)
        style = _compact_spaces(_norm(music_style_custom)) if _norm(music_style_custom) else _compact_spaces(_norm(music_style))

        lyrics_language_name = _norm(language) or "English"
        lyrics_language_code = LANGUAGE_NAME_TO_CODE.get(lyrics_language_name, "en")

        prompt = _make_song_prompt(
            lyrics_language_name=lyrics_language_name,
            lead_singer_sex=sex,
            song_topic=topic,
            music_style=style,
            lyrics=lyr,
            strict_user_lyrics=bool(strict_user_lyrics),
        )

        _ENGINE.load(
            repo_id=repo_id,
            quantization=_norm(quantization),
            attention_mode=_norm(attention_mode),
            use_torch_compile=bool(use_torch_compile),
            device_choice=_norm(device),
            keep_model_loaded=bool(keep_model_loaded),
        )

        try:
            full_output = _ENGINE.generate(
                prompt_text=prompt,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                num_beams=int(num_beams),
                repetition_penalty=float(repetition_penalty),
                seed=int(seed),
            )
            full_output = (full_output or "").strip()

            music_prompt_out, lyrics_out = _split_music_prompt_and_lyrics(full_output)
            music_prompt_out = _strip_music_prompt_prefix(music_prompt_out)
            lyrics_out = _cleanup_lyrics_text(lyrics_out)
            lyrics_out = _enforce_lyrics_tag_order(lyrics_out)

            return (
                lyrics_language_code,
                (music_prompt_out or "").strip(),
                (lyrics_out or "").strip(),
                full_output,
                prompt,
            )
        finally:
            if not keep_model_loaded:
                _ENGINE.clear()
class SongGenerationSuite:
    """
    Convenience node:
    - Generates (tags/music_prompt + lyrics) w
    - Immediately encodes conditioning through the provided AceStep CLIP
    """
    @classmethod
    def INPUT_TYPES(cls):
        # Reuse the same UI lists as SongWriter
        sexes = ["Not specified", "Male", "Female", "Non-binary", "Other"]
        music_styles = [
            "Heavy Metal","Hard Rock","Pop","Synth Pop","Electro","EDM","Techno","House",
            "Hip-Hop / Rap","Trap","R&B","Soul","Jazz","Blues","Reggae","Funk","Country",
            "Folk","Acoustic Ballad","Orchestral / Cinematic","Lo-fi","K-Pop","J-Pop","Drill","Punk",
        ]
        model_labels = list(DEFAULT_MODELS.keys()) + ["Custom (HF repo id)"]
        return {
            "required": {
                "clip": ("CLIP",),

                "song_topic": ("STRING", {"default": "", "multiline": True}),
                "lead_singer_sex": (sexes, {"default": "Male"}),
                "music_style": (music_styles, {"default": "Country"}),
                "music_style_custom": ("STRING", {"default": "", "multiline": False}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
                "strict_user_lyrics": ("BOOLEAN", {"default": True}),
                "lyrics_language": (LANGUAGE_DROPDOWN, {"default": "English"}),

                # AceStep 1.5-ish controls
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "bpm": ("INT", {"default": 120, "min": 10, "max": 300}),
                "duration": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 2000.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "timesignature": (["2", "3", "4", "6"], {"default": "4"}),
                "keyscale": ("STRING", {"multiline": False, "default": "C major"}),
                "lyrics_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),

                # LLM controls
                "model": (model_labels, {"default": model_labels[0]}),
                "custom_model_repo_id": ("STRING", {"default": "", "multiline": False}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto"}),
                "use_torch_compile": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "max_tokens": ("INT", {"default": 1100, "min": 64, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.2}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 0.5, "max": 2.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","LATENT","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("conditioning","latent_audio","ACE_LANGUAGE","TAGS","LYRICS","FULL_OUTPUT","PROMPT_USED")
    FUNCTION = "run"
    CATEGORY = "Song Generation Suite"

    @staticmethod
    def _sanitize_keyscale(keyscale: str) -> str:
        return " ".join((keyscale or "").strip().split())

    def run(
        self,
        clip,
        song_topic,
        lead_singer_sex,
        music_style,
        music_style_custom,
        lyrics,
        lyrics_language,
        strict_user_lyrics,
        seed,
        bpm,
        duration,
        batch_size,
        timesignature,
        keyscale,
        lyrics_strength,
        model,
        custom_model_repo_id,
        quantization,
        attention_mode,
        use_torch_compile,
        device,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        keep_model_loaded,
    ):
        # 1) SongWriter generation
        ace_language, tags, lyr, full_output, prompt_used = write(
            song_topic=song_topic,
            lead_singer_sex=lead_singer_sex,
            music_style=music_style,
            music_style_custom=music_style_custom,
            lyrics=lyrics,
            language=lyrics_language,
            strict_user_lyrics=strict_user_lyrics,
            model=model,
            custom_model_repo_id=custom_model_repo_id,
            quantization=quantization,
            attention_mode=attention_mode,
            use_torch_compile=use_torch_compile,
            device=device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            seed=seed if int(seed) > 0 else 1,
            keep_model_loaded=keep_model_loaded,
        )

        # 2) Encode conditioning via provided CLIP
        keyscale = self._sanitize_keyscale(keyscale)

        # Try AceStep 1.5 signature first, fallback to minimal signature.
        try:
            tokens = clip.tokenize(
                tags,
                lyrics=lyr,
                bpm=int(bpm),
                duration=float(duration),
                timesignature=int(timesignature),
                language=str(ace_language),
                keyscale=str(keyscale),
                seed=int(seed),
            )
        except Exception:
            tokens = clip.tokenize(tags, lyrics=lyr)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": float(lyrics_strength)})
        # 3) Create Empty Ace Step 1.5 latent audio directly (uses duration)
        try:
            seconds = float(duration)
        except Exception:
            seconds = 0.0
        if seconds <= 0.0:
            seconds = 120.0

        latent_length = round((seconds * 48000 / 1920))
        latent_tensor = torch.zeros([int(batch_size), 64, latent_length], device=comfy.model_management.intermediate_device())
        latent_audio = {"samples": latent_tensor, "type": "audio"}

        return (conditioning, latent_audio, str(ace_language), str(tags), str(lyr), str(full_output), str(prompt_used))



# -----------------------------
# Node registration
# -----------------------------

NODE_CLASS_MAPPINGS = {
    # Suite
    "SongGenerationSuite": SongGenerationSuite,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SongGenerationSuite": "Song Generation Suite",
}
