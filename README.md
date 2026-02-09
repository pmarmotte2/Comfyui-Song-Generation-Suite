# 🎵 SongGenerationSuite – AI Song Writer for ComfyUI (ACE 1.5)

**SongGenerationSuite** is a ComfyUI custom node designed to generate **complete, structured songs** (lyrics + music prompt) and convert them into **conditioning** compatible with **ACE 1.5** audio generation workflows.

It allows you to go from a **simple idea** to a **fully conditioned song** ready to be sampled with a KSampler, without writing complex prompts or lyrics manually.

---

## ✨ What this node does

From a single node, SongGenerationSuite generates:

- 🎼 **A detailed music prompt** (style, instruments, mood, structure)
- ✍️ **Fully structured lyrics** with strict bracket tags  
  (`[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`, etc.)
- 🧠 **Text conditioning** compatible with ACE 1.5
- 🧪 Debug outputs (full LLM output & prompt used)

All outputs are designed to plug directly into an **ACE 1.5 + KSampler** pipeline.

---

## 🧩 Workflow overview

![SongGenerationSuite Workflow](./workflow.png)

Minimal graph:

1. ACE 1.5 Checkpoint Loader  
2. SongGenerationSuite  
3. KSampler  
4. ConditioningZeroOut  
5. VAE Decode Audio / Save Audio  

---

## 🔌 Inputs & Wiring (ACE 1.5)

### CLIP input (required)

SongGenerationSuite **does not load ACE by itself**.

You must:
- Load **ACE 1.5** using a **Checkpoint Loader**
- Connect the loader’s **CLIP output** to **SongGenerationSuite → clip**

This ensures:
- The generated conditioning uses the **same text encoder**
- Full compatibility with the ACE pipeline

```
ACE 1.5 Checkpoint Loader
 ├─ MODEL → KSampler (model)
 └─ CLIP  → SongGenerationSuite (clip)
```

---

### Conditioning output

The **conditioning output** from SongGenerationSuite must be connected to:

- **KSampler → positive**
- **ConditioningZeroOut → conditioning**

This is the recommended ACE wiring.

---

## 🎛 User parameters

### Core songwriting

- **song_topic** – Main theme or story of the song  
- **lead_singer_sex** – Vocal profile used in the prompt  
- **music_style** – Base genre  
- **music_style_custom** – Optional detailed or hybrid style  
- **lyrics** – Optional user lyrics  
- **strict_user_lyrics** – Enforce lyrics exactly if enabled  
- **lyrics_language** – Language of the lyrics  

### Musical structure

- **bpm** – Target tempo  
- **duration** – Target song length (seconds)  
- **timesignature** – Musical time signature  
- **keyscale** – Musical key  
- **lyrics_strength** – Conditioning strength  

### Generation

- **seed** – Reproducibility  
- **batch_size** – Number of generations  

### LLM model & performance

- **model** – Built-in or custom HF model  
- **custom_model_repo_id** – HuggingFace repo ID  
- **quantization** – FP16 / 8-bit / 4-bit  
- **attention_mode** – Attention backend  
- **use_torch_compile** – Optional optimization  
- **device** – auto / cuda / cpu  
- **keep_model_loaded** – Keep model in VRAM  

### Decoding controls

- **max_tokens** – Maximum generated tokens  
- **temperature** – Creativity level  
- **top_p** – Nucleus sampling  
- **num_beams** – Beam search  
- **repetition_penalty** – Reduce repetition  

---

## 📦 Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOURNAME/SongGenerationSuite.git
pip install -r requirements.txt
```

Restart ComfyUI.

---

## 📝 Notes

- Prompts are written in **English** for LLM stability
- Lyrics language is controlled independently
- Designed for **ACE 1.5**
- Ideal for music generation and creative pipelines

---

## 🚀 Use cases

- AI song prototyping
- Music prompt generation
- ACE 1.5 audio experiments
- Rapid creative iteration
