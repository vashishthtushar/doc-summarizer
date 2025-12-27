# summarizer.py
# Core summarization module using Hugging Face API
# Implements text summarization with support for multiple styles

import os
import time
import json
import requests
from typing import List
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
ENV_PATH = (Path(__file__).resolve().parent / ".env").resolve()
if not ENV_PATH.exists():
    print(f"Warning: .env file not found at {ENV_PATH}")
load_dotenv(dotenv_path=str(ENV_PATH))

# Configuration settings
HF_MODEL = os.getenv("HF_MODEL", "facebook/bart-large-cnn")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 3000))
MAX_RETRIES = int(os.getenv("HF_MAX_RETRIES", 3))
RETRY_BASE_SECONDS = float(os.getenv("HF_RETRY_BASE_SECONDS", 1.5))
TEMPERATURE = float(os.getenv("HF_TEMPERATURE", 0.1))

# Setup logging
logger = logging.getLogger("summarizer")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


class SummarizationError(Exception):
    pass


class Summarizer:
    """
    Main summarization class that handles text summarization using Hugging Face API.
    
    Supports multiple summary styles: brief, detailed, and bullets.
    Handles long documents through chunking and synthesis.
    """
    
    def __init__(self, model_id: str = None):
        """
        Initialize the Summarizer instance.
        
        Args:
            model_id: Optional model ID to override default. Defaults to HF_MODEL from environment.
        
        Raises:
            SummarizationError: If HF_API_KEY is not found in environment variables
        """
        self.model_id = model_id or HF_MODEL
        if not HF_API_KEY:
            raise SummarizationError("HF_API_KEY missing in .env")
        self.headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

    # Text chunking methods
    def _chunk_text(self, text: str, chunk_chars: int = CHUNK_SIZE) -> List[str]:
        if not text:
            return []
        text = text.replace("\r\n", "\n").strip()
        if not text:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) + 2 <= chunk_chars:
                current += ("\n\n" if current else "") + p
            else:
                if current:
                    chunks.append(current)
                    current = ""
                if len(p) > chunk_chars:
                    for i in range(0, len(p), chunk_chars):
                        chunks.append(p[i:i + chunk_chars])
                else:
                    current = p
        if current:
            chunks.append(current)
        return chunks

    # Prompt generation and parameter setup
    def _prompt_and_params(self, text: str, style: str):
        style = (style or "brief").lower()
        
        # Check if using BART model (requires different parameter format)
        is_bart = "bart" in self.model_id.lower()
        
        if is_bart:
            # BART models: just pass the text directly
            prompt = text
            # Adjust max_new_tokens based on style
            # BART-Large-CNN is trained for CNN/DailyMail style summaries (concise)
            if style == "brief":
                max_tokens = 50  # Short, concise summary
            elif style == "detailed":
                max_tokens = 120  # More detailed but still reasonable
            elif style == "bullets":
                max_tokens = 80  # Medium length for bullets
            else:
                max_tokens = 60
        else:
            # For instruction-tuned models (like T5/Flan), use instruction prompts
            if style == "brief":
                instr = (
                    "You are a helpful assistant. Provide a concise summary in 2–4 sentences "
                    "that captures only the main points. Do NOT repeat the input verbatim. "
                    "Keep it neutral and do not add new facts."
                )
                max_tokens = 140
                examples = None
            elif style == "detailed":
                instr = (
                    "You are a helpful assistant. Provide a clear, detailed summary covering key ideas, "
                    "important details, and conclusions. Use paragraphs. Do NOT invent new facts."
                )
                max_tokens = 400
                examples = None
            elif style == "bullets":
                instr = (
                    "You are a helpful assistant. Summarize the text as 4–8 short bullet points, each 8–20 words. "
                    "Do NOT repeat the input verbatim. Use hyphen '-' at start of each bullet."
                )
                max_tokens = 220
                examples = (
                    "Example:\n"
                    "Text: The city will invest in schools and public transport to improve education and reduce traffic.\n"
                    "Summary:\n- Investment in schools announced to improve education.\n- Upgrades to public transport planned to reduce traffic.\n\n"
                )
            else:
                instr = "Summarize the following text."
                max_tokens = 200
                examples = None

            # assemble prompt for instruction-tuned models
            prompt = f"{instr}\n\nText:\n{text}\n\nSummary:"
            if examples:
                prompt = examples + prompt

        # Build params based on model type
        if is_bart:
            # BART uses max_length and min_length
            if style == "brief":
                max_length = 50
                min_length = 10
            elif style == "detailed":
                max_length = 120
                min_length = 20
            elif style == "bullets":
                max_length = 80
                min_length = 15
            else:
                max_length = 60
                min_length = 10
            
            params = {
                "max_length": max_length,
                "min_length": min_length,
                "temperature": TEMPERATURE
            }
        else:
            # Other models use max_new_tokens
            params = {
                "max_new_tokens": max_tokens,
                "temperature": TEMPERATURE
            }
        
        return prompt, params

    # Hugging Face API call implementation
    def _call_hf(self, prompt: str, params: dict):
        payload = {"inputs": prompt, "parameters": params}
        model = self.model_id

        urls = [
            f"https://api-inference.huggingface.co/models/{model}",
            # f"https://api-inference.huggingface.co/models/{model}"
        ]

        last_exc = None
        for url in urls:
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"Calling Hugging Face API: {url} (attempt {attempt})")
                    resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
                    status = resp.status_code
                    txt = resp.text
                    logger.info(f"API response status: {status}")
                    if status == 410:
                        # legacy endpoint removed — do not retry this
                        raise SummarizationError("HF 410: legacy inference endpoint removed; use Router.")
                    if status in (401, 403):
                        raise SummarizationError(f"HF auth error {status}: {txt[:500]}")
                    if status == 503:
                        # transient
                        raise RuntimeError("model loading (503)")
                    if status == 404:
                        # try next url
                        break
                    if status >= 400:
                        raise SummarizationError(f"HF API error {status}: {txt[:500]}")

                    # parse JSON
                    try:
                        data = resp.json()
                    except Exception:
                        data = None

                    # common shapes
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict):
                            # Check for various response formats (BART uses "summary_text", others use "generated_text")
                            for key in ("summary_text", "generated_text", "text", "result", "output"):
                                if key in first:
                                    return first[key] if isinstance(first[key], str) else json.dumps(first[key])
                        if isinstance(first, str):
                            return first
                        return json.dumps(first)

                    if isinstance(data, dict):
                        # Check for various response formats
                        for k in ("summary_text", "generated_text", "text", "result", "output"):
                            if k in data:
                                return data[k] if isinstance(data[k], str) else json.dumps(data[k])
                        # otherwise stringify
                        return json.dumps(data)

                    # fallback to text
                    return txt
                except RuntimeError as re:
                    last_exc = re
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_SECONDS * attempt)
                        continue
                    raise SummarizationError(f"Model loading / transient error: {re}")
                except SummarizationError:
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BASE_SECONDS * attempt)
                        continue
                    raise SummarizationError(f"Hugging Face request failed: {e}")
            # next url
        raise SummarizationError(f"LLM call failed; last error: {last_exc}")

    # Detect if output is just echoing the input (not a real summary)
    def _looks_like_echo(self, input_text: str, output_text: str) -> bool:
        if not output_text:
            return True
        in_s = " ".join(input_text.split())
        out_s = " ".join(output_text.split())
        # if exactly the same or output contains the full input
        if out_s == in_s or in_s in out_s or out_s in in_s:
            return True
        # if output length is > 90% of input length and more than 40 chars -> likely echo
        if len(out_s) >= 0.9 * len(in_s) and len(in_s) > 40:
            return True
        return False

    # Main summarization method with echo detection and retry logic
    def summarize(self, text: str, style: str = "brief") -> str:
        """
        Generate a summary of the input text.
        
        Args:
            text: Input text to summarize
            style: Summary style - 'brief', 'detailed', or 'bullets' (default: 'brief')
        
        Returns:
            str: Generated summary text
        
        Raises:
            SummarizationError: If input is empty or API calls fail after retries
        """
        if not text or not text.strip():
            raise SummarizationError("Empty input.")

        chunks = self._chunk_text(text)
        partials = []

        for i, ch in enumerate(chunks):
            prompt, params = self._prompt_and_params(ch, style)
            # 1st attempt
            out = self._call_hf(prompt, params)
            out = out.strip() if isinstance(out, str) else str(out).strip()

            # if output looks like an echo, retry with adjusted parameters
            is_bart = "bart" in self.model_id.lower()
            if self._looks_like_echo(ch, out):
                logger.info("Detected possible echo; retrying with adjusted parameters for chunk %d", i+1)
                
                if is_bart:
                    # For BART: try with shorter max_length to force more concise summary
                    params2 = params.copy()
                    if "max_length" in params2:
                        params2["max_length"] = min(params2["max_length"], 40)
                    out2 = self._call_hf(prompt, params2)  # Keep prompt same (just text)
                else:
                    # For instruction-tuned models: add stronger instructions
                    stronger_instr = (
                        "Important: Do NOT repeat the original text verbatim. Summarize only the main points. "
                        "If the text is short, condense to a single short sentence. "
                    )
                    prompt2 = stronger_instr + "\n\n" + prompt
                    out2 = self._call_hf(prompt2, params)
                
                out2 = out2.strip() if isinstance(out2, str) else str(out2).strip()
                # if second try is better (not echo), use it, else keep first but trim
                if not self._looks_like_echo(ch, out2):
                    out = out2
                else:
                    # as a last resort, produce a short synthetic summary:
                    words = ch.split()
                    out = " ".join(words[:min(30, max(10, len(words)//6))]) + ("..." if len(words) > 30 else "")

            partials.append(out)

        # if only one chunk, return it cleaned
        if len(partials) == 1:
            return partials[0].strip()

        # synthesize partials
        synth_prompt = (
            "Combine the following partial summaries into a single coherent summary. Remove duplicates and be concise.\n\n"
        )
        for idx, p in enumerate(partials):
            synth_prompt += f"--- PART {idx+1} ---\n{p}\n\n"
        params = {"max_new_tokens": 220, "temperature": TEMPERATURE}
        final = self._call_hf(synth_prompt, params).strip()
        if self._looks_like_echo(text, final):
            # fallback: join partials
            return "\n".join(partials)
        return final
