import sys
import os
import base64
import io
from pathlib import Path

# Add project directory to path
# Add project directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.join(base_dir, "project")
sys.path.append(project_path)

from core.rag_system import RAGSystem
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import config
import csv
import re
import json
from pydantic import BaseModel
from typing import Any, List, Dict, Optional

# Merged into visual_prompt_json when the model omits POD keys (print-on-demand safety).
POD_BACKGROUND_DEFAULT = (
    "Transparent background / alpha intent: isolated subject for POD cutout; empty canvas must not be filled with "
    "black, charcoal, or dark gray. If the raster API cannot return true transparency, use clean white (#FFFFFF) "
    "only outside the artwork. Avoid black letterboxing unless the art style intentionally uses a poster/mat frame."
)
POD_ALPHA_NOTE_DEFAULT = (
    "Raster image APIs often cannot return true PNG alpha; if you need transparency, treat it as design intent "
    "and plan background removal or masking after generation."
)
POD_LETTERBOXING_DEFAULT = "avoid"

# Max suggestions per field for `suggest_field_inputs` when the client does not override.
SUGGEST_FIELD_DEFAULT_LIMITS: Dict[str, int] = {
    "subject": 3,
    "action": 3,
    "mood": 3,
    "style": 3,
    "colors": 3,
    "context": 3,
}


class CreativeAgent:
    def __init__(self):
        print("Initializing Creative Agent...")
        self.rag = RAGSystem()
        self.rag.initialize()
        self.concept_timeout_seconds = int(os.getenv("FAST_TRACK_CONCEPT_TIMEOUT_SECONDS", "120"))
        self.concept_max_retries = int(os.getenv("FAST_TRACK_CONCEPT_MAX_RETRIES", "1"))

        from langchain_google_genai import ChatGoogleGenerativeAI
        self.default_vision_model = "gemini-3-pro-preview"
        self.default_concept_model = "gemini-3-pro-preview"
        self._creative_model_aliases = {
            "gemini-3-pro-preview": "gemini-3-pro-preview",
            "gemini 3 pro preview": "gemini-3-pro-preview",
            "gemini-3-flash-preview": "gemini-3-flash-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
            "gemini 3 flash": "gemini-3-flash-preview",
            "gpt-5.2": "gpt-5.2",
            "gpt 5.2": "gpt-5.2",
            "gpt-5-mini": "gpt-5-mini",
            "gpt 5 mini": "gpt-5-mini",
        }
        self._allowed_concept_models = {
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gpt-5.2",
            "gpt-5-mini",
        }
        self._creative_llm_cache = {}

        # Vision analysis remains fixed on Gemini Pro.
        self.vision_llm = ChatGoogleGenerativeAI(model=self.default_vision_model, max_tokens=10000)
        self.creative_llm, _ = self._get_creative_llm(self.default_concept_model)

    def _normalize_concept_model(self, concept_model: Optional[str]) -> str:
        raw = str(concept_model or "").strip()
        if not raw:
            return self.default_concept_model
        canonical = self._creative_model_aliases.get(raw.lower(), raw)
        if canonical not in self._allowed_concept_models:
            allowed = ", ".join(sorted(self._allowed_concept_models))
            raise ValueError(f"Unsupported concept model '{raw}'. Allowed: {allowed}")
        return canonical

    def _get_creative_llm(self, concept_model: Optional[str] = None):
        model_name = self._normalize_concept_model(concept_model)
        if model_name in self._creative_llm_cache:
            return self._creative_llm_cache[model_name], model_name

        common_kwargs: Dict[str, Any] = {"temperature": 0.3}
        if self.concept_timeout_seconds > 0:
            common_kwargs["timeout"] = self.concept_timeout_seconds
        if self.concept_max_retries >= 0:
            common_kwargs["max_retries"] = self.concept_max_retries

        if model_name.startswith("gemini-"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, **common_kwargs)
            except TypeError:
                # Compatibility fallback for older provider wrappers.
                llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        elif model_name.startswith("gpt-"):
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for GPT concept models.")
            try:
                llm = ChatOpenAI(model=model_name, **common_kwargs)
            except TypeError:
                llm = ChatOpenAI(model=model_name, temperature=0.3)
        else:
            raise ValueError(f"Cannot resolve provider for concept model '{model_name}'.")

        self._creative_llm_cache[model_name] = llm
        return llm, model_name

    def encode_image(self, image_path):
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"

        # Resize/compress before sending to vision model to reduce payload latency.
        try:
            from PIL import Image, ImageOps

            max_dim = int(os.getenv("VISION_MAX_DIM", "1536"))
            jpeg_quality = int(os.getenv("VISION_JPEG_QUALITY", "85"))

            with Image.open(image_path) as img:
                img = ImageOps.exif_transpose(img)
                original_size = img.size

                if max(original_size) > max_dim:
                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                    img.thumbnail((max_dim, max_dim), resampling)

                has_alpha = (
                    img.mode in ("RGBA", "LA")
                    or (img.mode == "P" and "transparency" in img.info)
                )

                output = io.BytesIO()
                if has_alpha:
                    if img.mode not in ("RGBA", "LA"):
                        img = img.convert("RGBA")
                    img.save(output, format="PNG", optimize=True)
                    out_mime = "image/png"
                else:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(
                        output,
                        format="JPEG",
                        quality=jpeg_quality,
                        optimize=True,
                        progressive=True,
                    )
                    out_mime = "image/jpeg"

                encoded = base64.b64encode(output.getvalue()).decode("utf-8")
                print(
                    f"Vision image preprocessed: {original_size} -> {img.size}, "
                    f"mime={out_mime}, quality={jpeg_quality if out_mime == 'image/jpeg' else 'lossless'}"
                )
                return f"data:{out_mime};base64,{encoded}"
        except Exception as e:
            print(f"Vision preprocess skipped, using original file bytes: {e}")

        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"

    def _normalize_palette_list(self, colors_value: Any) -> List[str]:
        if isinstance(colors_value, list):
            return [str(v).strip() for v in colors_value if str(v).strip()]
        text = str(colors_value or "").strip()
        if not text:
            return []
        return [c.strip() for c in re.split(r",|\||/|;", text) if c.strip()]

    def _normalize_text_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value or "").strip()
        if not text:
            return []
        return [v.strip() for v in re.split(r",|\||/|;|\n", text) if v.strip()]



    def _coerce_prompt_json_object(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _merge_pod_visual_prompt_json(self, prompt_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fill missing POD keys so downstream image gen and operators get stable defaults."""
        out: Dict[str, Any] = dict(prompt_json) if isinstance(prompt_json, dict) else {}
        if not str(out.get("pod_background", "")).strip():
            out["pod_background"] = POD_BACKGROUND_DEFAULT
        if not str(out.get("pod_letterboxing", "")).strip():
            out["pod_letterboxing"] = POD_LETTERBOXING_DEFAULT
        if not str(out.get("pod_alpha_note", "")).strip():
            out["pod_alpha_note"] = POD_ALPHA_NOTE_DEFAULT
        return out

    def _count_words(self, text: Any) -> int:
        content = str(text or "").strip()
        if not content:
            return 0
        return len(re.findall(r"\b[\w'-]+\b", content))

    def _expand_prompt_to_min_words(
        self,
        visual_prompt: str,
        concept_fields: Dict[str, Any],
        min_words: int = 800,
    ) -> str:
        text = str(visual_prompt or "").strip()
        if not text:
            text = (
                f"Create a print-ready apparel illustration featuring {concept_fields.get('subject', 'the main subject')} "
                f"{concept_fields.get('action', '')} in {concept_fields.get('context', 'a clear visual setting')}, "
                f"with mood {concept_fields.get('mood', 'balanced and coherent')}, in {concept_fields.get('art_style', 'a strong commercial style')} "
                f"and palette {concept_fields.get('colors', 'cohesive high-contrast colors')}."
            )

        subject = str(concept_fields.get("subject", "")).strip()
        action = str(concept_fields.get("action", "")).strip()
        context = str(concept_fields.get("context", "")).strip()
        mood = str(concept_fields.get("mood", "")).strip()
        art_style = str(concept_fields.get("art_style", "")).strip()
        colors = str(concept_fields.get("colors", "")).strip()

        enrichment = (
            "Production directives: keep the character silhouette clear at thumbnail and shirt-print size, maintain strong separation between foreground subject and background elements, and avoid clutter near the outer contour. "
            f"Subject fidelity: preserve {subject or 'the main subject'} as the hero element with unmistakable anatomy and visual identity. "
            f"Action fidelity: stage {action or 'the core action'} with readable gesture arcs, clear limb placement, and convincing weight distribution so the motion reads instantly. "
            f"Context fidelity: build {context or 'the environment'} with layered depth cues, controlled spacing, and secondary elements that support the story without stealing focus. "
            f"Mood fidelity: sustain {mood or 'the intended emotional tone'} through pose, expression, color contrast, line rhythm, and visual pacing. "
            f"Style fidelity: execute in {art_style or 'the requested style'} with disciplined linework, intentional shape language, and consistent rendering logic from head to toe. "
            f"Color fidelity: use {colors or 'the specified palette'} with high print contrast, preserve dominant accent hierarchy, and avoid muddy blends. "
            "Lighting directives: define key light direction, edge highlights, form shadows, and reflected light zones; keep values organized for high readability on textile prints. "
            "Composition directives: center the focal hierarchy, lock primary action near the visual anchor, and distribute supporting motifs to balance left-right and top-bottom weight. "
            "Detail directives: include material definition for fabric, metal, fur, skin, or props as relevant; ensure micro-details are crisp but not noisy. "
            "Print optimization directives: avoid tiny unreadable details, avoid low-contrast color-on-color collisions, and maintain strong negative space around critical edges. "
            "Output constraints: keep composition self-contained, no accidental crop of key anatomy, no visual tangents that break readability, and no conflicting style artifacts."
        )

        expanded = text
        while self._count_words(expanded) < min_words:
            expanded = f"{expanded}\n\n{enrichment}"

        return expanded.strip()

    def _build_prompt_json_from_concept_fields(
        self,
        concept_fields: Dict[str, Any],
        visual_prompt_text: str = "",
    ) -> Dict[str, Any]:
        prompt_json: Dict[str, Any] = {}

        subject = str(concept_fields.get("subject", "")).strip()
        action = str(concept_fields.get("action", "")).strip()
        context = str(concept_fields.get("context", "")).strip()
        mood = str(concept_fields.get("mood", "")).strip()
        art_style = str(concept_fields.get("art_style", "")).strip()
        colors = str(concept_fields.get("colors", "")).strip()

        if subject:
            prompt_json["subject"] = subject
        if action:
            prompt_json["action"] = action
        if context:
            prompt_json["context"] = context
        if mood:
            prompt_json["mood"] = mood
        if art_style:
            prompt_json["style"] = art_style

        palette = self._normalize_palette_list(colors)
        if palette:
            prompt_json["color_palette"] = palette
        elif colors:
            prompt_json["colors"] = colors

        if visual_prompt_text.strip():
            prompt_json["notes"] = visual_prompt_text.strip()

        return self._merge_pod_visual_prompt_json(prompt_json)

    def _get_vision_field_value(self, source: Optional[Dict[str, Any]], field: str) -> str:
        source = source if isinstance(source, dict) else {}
        def _flatten(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (int, float, bool)):
                return str(value).strip()
            if isinstance(value, list):
                parts = []
                for item in value:
                    text = _flatten(item)
                    if text:
                        parts.append(text)
                dedup = list(dict.fromkeys(parts))
                return ", ".join(dedup)
            if isinstance(value, dict):
                preferred_order = [
                    "subject", "type", "name", "primary_focus",
                    "action", "pose",
                    "context", "background", "setting",
                    "style", "visual_family", "genre",
                    "mood",
                    "colors", "color_palette", "dominant_colors", "supporting_colors",
                ]
                parts = []
                for key in preferred_order:
                    if key in value:
                        text = _flatten(value.get(key))
                        if text:
                            parts.append(text)
                if not parts:
                    for nested in value.values():
                        text = _flatten(nested)
                        if text:
                            parts.append(text)
                dedup = list(dict.fromkeys(parts))
                return ", ".join(dedup)
            return str(value).strip()

        def _get_path(obj: Dict[str, Any], path: str) -> Any:
            cur: Any = obj
            for part in path.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return None
                cur = cur[part]
            return cur

        path_map = {
            "subject": [
                "subject",
                "prompt_json.subject",
                "prompt_json.composition.subject",
                "prompt_json.meta.primary_focus",
            ],
            "action": [
                "action",
                "prompt_json.action",
                "prompt_json.composition.action",
                "prompt_json.composition.pose",
                "prompt_json.subject.pose",
            ],
            "context": [
                "context",
                "prompt_json.context",
                "prompt_json.composition.background",
                "prompt_json.background.setting",
                "prompt_json.background.elements",
                "prompt_json.background.main_graphic",
            ],
            "mood": [
                "mood",
                "prompt_json.mood",
                "prompt_json.style.mood",
            ],
            "art_style": [
                "art_style",
                "style",
                "prompt_json.art_style",
                "prompt_json.style",
                "prompt_json.style.visual_family",
                "prompt_json.style.genre",
            ],
            "colors": [
                "colors",
                "prompt_json.colors",
                "prompt_json.color_palette",
                "prompt_json.color_palette.dominant_colors",
                "prompt_json.color_palette.supporting_colors",
            ],
        }

        for path in path_map.get(field, [field]):
            raw = _get_path(source, path)
            text = _flatten(raw)
            if text:
                return text

        return ""

    def analyze_image(self, image_path):
        print(f"Analyzing image: {image_path}...")

        image_data_url = self.encode_image(image_path)
        
        prompt_instruction = """Analyze this image in very high detail and create a rich JSON prompt package for image generation.

Return ONE valid JSON object only (no markdown code fences, no extra text).

Primary goal:
- Capture as much useful visual information as possible while staying faithful to the image.

Required fields:
1. "visual_prompt":
   - A long, detailed, cohesive text-to-image prompt.
   - Describe subject, action, scene, style, materials, lighting, composition, textures, colors, mood, and rendering details.
2. "prompt_json":
   - A deeply structured JSON object for generation.
   - Include nested keys where useful: subject, environment, composition, style, color_palette, lighting, camera, materials, texture, typography, effects, constraints, print_notes.

Detail policy:
- Be exhaustive and include fine-grained details (shape, proportions, pose, line quality, brush/ink behavior, surface feel, distress patterns, highlights/shadows, depth layering, focal hierarchy).
- If text/typography exists, describe content, placement, style, readability, and visual role.
- If ambiguity exists, include an "assumptions" field with best-effort interpretations.
- Include "negative_constraints" (what to avoid) and "confidence" scores (0-1) for major inferred groups when possible.

Output format rules:
- Output must be valid JSON.
- Do not use markdown.
- Do not omit required fields.
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ]
        )
        
        # Use simple invoke for freeform JSON instead of rigid Pydantic output
        response = self.vision_llm.invoke([message])
        
        # Extract text content safely because Gemini might return a list of dicts
        raw_content = response.content
        if isinstance(raw_content, list):
            texts = [part.get("text", "") for part in raw_content if isinstance(part, dict) and "text" in part]
            text_content = "".join(texts).strip()
            # In some cases parts could be string items
            if not text_content:
                text_content = "".join([str(p) for p in raw_content if isinstance(p, str)]).strip()
        else:
            text_content = str(raw_content).strip()
        
        # Clean markdown code block formatting if present
        if text_content.startswith("```json"):
            text_content = text_content[7:]
        elif text_content.startswith("```"):
            text_content = text_content[3:]
        if text_content.endswith("```"):
            text_content = text_content[:-3]
        text_content = text_content.strip()
        
        print("\n\n[DEBUG] --- RAW TEXT CONTENT EXTRACTED FROM LLM ---")
        print(text_content)
        print("[DEBUG] -------------------------------------------\n")

        try:
            payload = json.loads(text_content)
            print("[DEBUG] Successfully parsed unstructured JSON.")
        except Exception as e:
            print(f"[DEBUG] Failed to parse LLM JSON: {e}")
            payload = {
                "raw_text": text_content, 
                "visual_prompt": text_content,
                "prompt_json": {}
            }

        # Provide raw debugging text to frontend
        payload["__raw_debug_json"] = text_content

        # Create a dynamic dictionary object to maintain model_dump compatibility with existing code
        class DynamicAnalysis(dict):
            def model_dump(self):
                return dict(self)
            def model_dump_json(self):
                return json.dumps(dict(self), indent=4)
            def __getattr__(self, item):
                return self.get(item)

        return DynamicAnalysis(payload)

    def retrieve_ideas(self, description, user_instruction=None):
        print("Retrieving related concepts from RAG...")
        # Formulate a query for the RAG agent
        query = f"Based on this visual description: '{description}', suggest relevant funny T-shirt keywords, puns, or style concepts from the database."
        
        if user_instruction:
            query += f"\n\nUSER COMMAND: {user_instruction}\n(Prioritize keywords related to this command)"
            
        # Invoke the Agentic RAG graph
        # The graph state expects 'messages'
        inputs = {"messages": [HumanMessage(content=query)]}
        
        # We need to run the graph. 
        # The graph returns a dict with 'messages' and 'agent_answers'
        result = self.rag.agent_graph.invoke(inputs, config=self.rag.get_config())
        
        # Extract the final answer (usually the last AIMessage)
        final_message = result["messages"][-1]
        return final_message.content

    def _normalize_field_inputs(self, field_inputs: Optional[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        normalized = {k: [] for k in ["subject", "action", "mood", "style", "colors", "context"]}
        if not isinstance(field_inputs, dict):
            return normalized

        for k in normalized.keys():
            values = field_inputs.get(k, [])
            if not isinstance(values, list):
                continue
            clean_vals = [str(v).strip() for v in values if str(v).strip()]
            normalized[k] = clean_vals[:3]
        return normalized

    def _classify_fields(
        self,
        normalized_fields: Dict[str, List[str]],
        vision_analysis: Optional[Dict],
    ) -> tuple:
        """Compare user field inputs vs vision baseline.
        Returns (changed: dict[field->list[str]], locked: dict[field->str]).

        A field is 'changed' ONLY if the user provided >1 value for it,
        meaning they explicitly want VARIATION in that dimension.
        A field with 0 or 1 value is 'locked' (baseline or user override).
        """
        source = vision_analysis if isinstance(vision_analysis, dict) else {}
        baseline_map = {
            "subject": "subject",
            "action": "action",
            "mood": "mood",
            "style": "art_style",
            "colors": "colors",
            "context": "context",
        }
        changed: Dict[str, List[str]] = {}
        locked: Dict[str, str] = {}

        for field in ["subject", "action", "mood", "style", "colors", "context"]:
            user_vals = normalized_fields.get(field, [])
            baseline_key = baseline_map.get(field, field)
            baseline_val = self._get_vision_field_value(source, baseline_key)

            if len(user_vals) > 1:
                # Multiple values → user wants variation in this field
                changed[field] = user_vals
            else:
                # 0 or 1 value → locked (use user's value or fall back to baseline)
                locked[field] = user_vals[0].strip() if user_vals else baseline_val

        return changed, locked

    def _build_smart_combos(
        self,
        changed: Dict[str, List[str]],
        locked: Dict[str, str],
    ) -> tuple:
        """Build Cartesian-product combos from changed fields, grouped by
        the primary changed field (the one with the most values).

        Returns (groups: list[dict], primary_field: str, total_cards: int).
        Each group dict: { 'group_value': str, 'combos': list[dict] }
        Each combo dict has all 6 fields with concrete single values.
        No cap — all combos are generated.
        """
        import itertools

        if not changed:
            return [], "", 0

        # Determine primary field (most values → used for grouping)
        primary_field = max(changed, key=lambda f: len(changed[f]))

        # Build ordered list of (field_name, values_list)
        field_order = []
        for f in ["subject", "action", "mood", "style", "colors", "context"]:
            if f in changed:
                field_order.append((f, changed[f]))

        # Cartesian product — generate ALL combos
        all_combos = list(itertools.product(*[vals for _, vals in field_order]))

        # Convert tuples to full field dicts
        combo_dicts = []
        for combo_tuple in all_combos:
            d = dict(locked)  # start with locked fields
            for i, (field_name, _) in enumerate(field_order):
                d[field_name] = combo_tuple[i]
            combo_dicts.append(d)

        # Group by primary field
        from collections import OrderedDict
        groups_map = OrderedDict()
        for cd in combo_dicts:
            key = cd[primary_field]
            if key not in groups_map:
                groups_map[key] = []
            groups_map[key].append(cd)

        groups = [
            {"group_value": gv, "combos": combos}
            for gv, combos in groups_map.items()
        ]

        total_cards = sum(len(g["combos"]) for g in groups)
        return groups, primary_field, total_cards

    def _subject_identity_key(self, value: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", str(value or "").lower())
        if not tokens:
            return ""

        stop_tokens = {
            "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "from",
            "anthropomorphic", "cute", "funny", "happy", "sad", "retro", "vintage", "fluffy", "tiny",
            "big", "small", "brown", "black", "white", "red", "blue", "green", "yellow", "pink",
            "wearing", "holding", "with", "oversized", "winter", "summer", "ski", "snow", "gear",
            "goggles", "jacket", "coat", "hat", "scarf", "hoodie", "puffer", "outfit", "costume",
            "style", "look", "vibe", "character"
        }

        for token in reversed(tokens):
            if token in stop_tokens:
                continue
            if token.isdigit():
                continue
            return token

        return tokens[-1]

    def _dedupe_subject_candidates(self, candidates: List[str], existing: Optional[List[str]] = None, limit: int = 3) -> List[str]:
        result: List[str] = []
        seen_text = set()
        seen_identity = set()

        for value in existing or []:
            text_key = str(value).strip().lower()
            if not text_key:
                continue
            seen_text.add(text_key)
            ident = self._subject_identity_key(value)
            if ident:
                seen_identity.add(ident)

        for value in candidates:
            v = str(value).strip()
            if not v:
                continue
            text_key = v.lower()
            if text_key in seen_text:
                continue
            ident = self._subject_identity_key(v)
            if ident and ident in seen_identity:
                continue

            result.append(v)
            seen_text.add(text_key)
            if ident:
                seen_identity.add(ident)
            if len(result) >= limit:
                break

        return result

    def _load_style_rows(self) -> List[dict]:
        rows = []
        csv_path = os.path.join(base_dir, "..", "ArtStyles_v2.csv")
        try:
            if not os.path.exists(csv_path):
                return []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Style"):
                        rows.append(row)
        except Exception as e:
            print(f"Error loading styles CSV: {e}")
        return rows

    def suggest_field_inputs(
        self,
        vision_analysis: dict,
        current_inputs: Optional[Dict[str, List[str]]] = None,
        target_field: Optional[str] = None,
        force_regenerate: bool = False,
        field_limits: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[str]]:
        from pydantic import BaseModel, Field

        class SuggestedFields(BaseModel):
            subject: List[str] = Field(default_factory=list)
            action: List[str] = Field(default_factory=list)
            mood: List[str] = Field(default_factory=list)
            style: List[str] = Field(default_factory=list)
            colors: List[str] = Field(default_factory=list)
            context: List[str] = Field(default_factory=list)

        normalized = self._normalize_field_inputs(current_inputs)
        caps: Dict[str, int] = {
            field: int(SUGGEST_FIELD_DEFAULT_LIMITS.get(field, 3)) for field in normalized.keys()
        }
        if isinstance(field_limits, dict):
            for key, raw in field_limits.items():
                if key not in caps:
                    continue
                try:
                    n = int(raw)
                except (TypeError, ValueError):
                    continue
                if n >= 1:
                    caps[key] = min(n, 30)

        styles = self._load_style_rows()
        style_names = [row.get("Style", "").strip() for row in styles if row.get("Style", "").strip()]
        style_categories = [row.get("Category", "").strip() for row in styles if row.get("Category", "").strip()]
        style_usages = [row.get("Usage", "").strip() for row in styles if row.get("Usage", "").strip()]

        topic_keywords = self.get_preset_keywords()
        valid_target = target_field if target_field in normalized else ""
        caps_json = json.dumps(caps, ensure_ascii=False)
        subject_cap = caps.get("subject", 3)

        prompt = f"""
You are a POD concept input assistant.

Goal:
- Return suggestions for 6 fields: subject, action, mood, style, colors, context.
- Per-field maximum counts (never exceed these list lengths): {caps_json}
- Keep suggestions coherent (avoid contradictions like mood=dark while colors=pastel cute unless intentionally stylistic).

Rules:
- If target_field is set:
  - Operate ONLY on that field and keep all other fields unchanged.
  - If force_regenerate is FALSE: keep existing values for target field, then top up to that field's max count in the caps above.
  - If force_regenerate is TRUE: regenerate target field values from scratch (up to that field's max).
- If target_field is empty:
  - If force_regenerate is FALSE: keep existing values and top up each field toward its max in the caps.
  - If force_regenerate is TRUE: regenerate all fields from scratch (respect each field's max).
- Prefer using these data sources:
  1) Topic keywords from 50Topics_Keyword.csv
  2) Styles metadata from ArtStyles_v2.csv
- SUBJECT-SPECIFIC DIVERSITY RULE:
  - Subject suggestions must be different character identities (different species/archetypes/professions), not outfit/accessory variants of the same character.
  - Bad: "brown sloth", "sloth with goggles", "sloth in jacket".
  - Good: "red panda snowboarder", "retro robot courier", "samurai frog drummer".
  - When the subject cap is {subject_cap}, return up to {subject_cap} distinct subjects (fewer only if you cannot find enough diverse options).
- Do not copy Vision analysis values verbatim as suggestions. Prefer alternatives or paraphrased variants.

Current field inputs:
{json.dumps(normalized, ensure_ascii=False)}

Target field:
{valid_target or "all"}

force_regenerate:
{str(force_regenerate).lower()}

Vision analysis:
{json.dumps(vision_analysis or {}, ensure_ascii=False)}

Data source snippets:
- Topic keywords sample: {json.dumps(topic_keywords[:120], ensure_ascii=False)}
- Style names sample: {json.dumps(style_names[:180], ensure_ascii=False)}
- Style categories sample: {json.dumps(style_categories[:120], ensure_ascii=False)}
- Style usages sample: {json.dumps(style_usages[:120], ensure_ascii=False)}

Output:
- Return JSON only for fields: subject, action, mood, style, colors, context.
- Each field must be a list with at most the length given in the caps (concise values).
"""

        structured_llm = self.creative_llm.with_structured_output(SuggestedFields)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        suggested = response.model_dump()

        vision_key_map = {
            "subject": "subject",
            "action": "action",
            "mood": "mood",
            "style": "art_style",
            "colors": "colors",
            "context": "context",
        }

        def _filter_verbatim_vision(field: str, values: List[str]) -> List[str]:
            if not values:
                return values
            vision_key = vision_key_map.get(field, field)
            vision_value = self._get_vision_field_value(vision_analysis or {}, vision_key).lower()
            if not vision_value:
                return values
            filtered = [v for v in values if str(v).strip().lower() != vision_value]
            return filtered or values

        result = {}
        for field in normalized.keys():
            cap = int(caps.get(field, 3))
            old_vals = normalized[field]
            new_vals = [str(v).strip() for v in suggested.get(field, []) if str(v).strip()][:cap]
            new_vals = _filter_verbatim_vision(field, new_vals)
            if field == "subject":
                new_vals = self._dedupe_subject_candidates(
                    new_vals,
                    existing=old_vals if not force_regenerate else None,
                    limit=cap,
                )

            if valid_target and field != valid_target:
                result[field] = old_vals
                continue

            if not force_regenerate:
                # Preserve user/current values first, then append unique AI suggestions.
                merged = list(old_vals)
                if field == "subject":
                    appended = self._dedupe_subject_candidates(new_vals, existing=merged, limit=max(0, cap - len(merged)))
                    merged.extend(appended)
                else:
                    seen = {v.lower() for v in merged}
                    for candidate in new_vals:
                        key = candidate.lower()
                        if key in seen:
                            continue
                        merged.append(candidate)
                        seen.add(key)
                        if len(merged) >= cap:
                            break
                result[field] = merged[:cap]
            else:
                result[field] = new_vals[:cap]
        return result

    def mix_and_create(self, description, rag_ideas, user_instruction=None, selected_keywords=None, field_inputs=None, concept_model: Optional[str] = None):
        print("Mixing ideas...")
        from typing import List

        class DynamicConcept(dict):
            def model_dump(self):
                return dict(self)
            def model_dump_json(self):
                return json.dumps(dict(self), indent=4)
            def __getattr__(self, item):
                return self.get(item)

        normalized_fields = self._normalize_field_inputs(field_inputs)
        field_context = ""
        if any(len(v) > 0 for v in normalized_fields.values()):
            field_context = f"""

**MANDATORY USER FIELD INPUTS**:
User-provided fields (each can have 1-3 values):
- Subject: {normalized_fields['subject']}
- Action: {normalized_fields['action']}
- Mood: {normalized_fields['mood']}
- Style: {normalized_fields['style']}
- Colors: {normalized_fields['colors']}
- Context: {normalized_fields['context']}

Use these as hard constraints:
- If a field has values, you MUST prioritize them.
- When a concept's focus field needs 3 variants, use user values first, then extend creatively if fewer than 3.
- Keep cross-field coherence; avoid contradictory mood/style/color combinations unless explicitly implied by user input.
- If Subject has user-provided values, treat them as the source of truth and DO NOT revert to the original image subject unless that original subject is explicitly included by the user.
"""

        keywords_context = ""
        if selected_keywords and len(selected_keywords) > 0:
            keywords_context = f"\n\n**MANDATORY USER SELECTED KEYWORDS/NICHES**:\nThe user has explicitly selected these keywords to focus on: {', '.join(selected_keywords)}.\nENSURE these keywords are heavily integrated into the concepts (either as subjects, themes, or puns)."

        prompt = f"""
        You are a Creative Director for a POD (Print on Demand) T-shirt business.
        
        Original Image Analysis:
        {description}
        
        Inspiration from our Database (Trending Keywords/Styles):
        {rag_ideas}
        {field_context}
        {keywords_context}
        
        Task:
        Create EXACTLY 6 DISTINCT T-shirt design concept groups based on these strategies.
        
        **CRITICAL RULE FOR ALL 6 CONCEPTS:**
        Each concept has a single "focus" field that gets **3 creative variants**.
        ALL OTHER FIELDS must contain **exactly 1 value each** (kept consistent within the group).
        
        1. **Concept 1: The Subject Twist** (focus = "Subject")
           - Keep the original Action, Context, Mood, Art Style, Colors FIXED (1 value each).
           - Provide EXACTLY 3 creative DIFFERENT subjects (e.g. ["T-Rex", "Corgi", "Grizzly Bear"]).
           - Pick the strongest subject as the basis for the visual_prompt and title.
           
        2. **Concept 2: The Action Switch** (focus = "Action")
           - Keep the original Subject, Context, Mood, Art Style, Colors FIXED (1 value each).
           - Provide EXACTLY 3 creative DIFFERENT actions (e.g. ["skateboarding", "gaming", "drinking boba"]).
           
        3. **Concept 3: The Context Shift** (focus = "Context")
           - Keep the original Subject, Action, Mood, Art Style, Colors FIXED (1 value each).
           - Provide EXACTLY 3 creative DIFFERENT contexts (e.g. ["Outer Space", "Cyberpunk City", "Medieval Tavern"]).
           
        4. **Concept 4: The Mood Shift** (focus = "Mood")
           - Keep the original Subject, Action, Context, Art Style, Colors FIXED (1 value each).
           - Provide EXACTLY 3 creative DIFFERENT moods (e.g. ["epic & dramatic", "funny & playful", "dark & mysterious"]).

        5. **Concept 5: The Style Remix** (focus = "Style")
           - Keep the original Subject, Action, Context, Mood FIXED (1 value each).
           - Keep the original Colors FIXED (1 value) — do NOT change the color palette.
           - Provide EXACTLY 3 creative DIFFERENT art_style values (e.g. ["Kawaii", "Vaporwave", "Glitch Art"]).
           
        6. **Concept 6: The Color Shift** (focus = "Color")
           - Keep the original Subject, Action, Context, Mood, Art Style ALL FIXED (1 value each).
           - Provide EXACTLY 3 creative DIFFERENT color palettes (e.g. ["Neon Cyberpunk", "Pastel Watercolor", "Monochrome Black & White"]).
        """
        
        if user_instruction:
            prompt += f"""
            
            IMMEDIATE USER INSTRUCTION:
            "{user_instruction}"
            
            CRITICAL OVERRIDE: 
            If the user instruction starts with "Refining concept...", it means they want to IMPROVE a specific concept they already saw.
            In this case:
            1. **FOCUS ONLY ON THAT CONCEPT'S CORE IDEA**.
            2. **APPLY THE USER'S REQUESTED CHANGE AS THE NEW TRUTH**.
               - If the user asks to "change action to dancing", the new Action IS "dancing". **DO NOT** keep the old action.
               - If the user asks to "change subject to a cat", the new Subject IS "cat". **DO NOT** keep the old subject.
            3. Generate 3 VARIATIONS of this *refined* concept (e.g. 3 different ways to show the T-Rex dancing).
            4. You can ignore the strict split if it doesn't make sense for a refinement, BUT try to keep diversity in the details.
            """
            
        prompt += """
        
        CRITICAL REQUIREMENT: Output ONLY valid JSON matching the structure below. Do not wrap it in markdown blocks. Just return the raw JSON object.
        {
          "concepts": [
            {
              "title": "...",
              "visual_prompt": "...",
              "visual_prompt_json": { ... },
              "subject": ["...", "...", "..."],
              "action": ["..."],
              "context": ["..."],
              "mood": ["..."],
              "art_style": ["..."],
              "colors": ["..."],
              "caption": "...",
              "logic": "...",
              "focus": "..."
            }
          ]
        }
        """
        
        creative_llm, resolved_model = self._get_creative_llm(concept_model)
        print(f"Concept generation model: {resolved_model}")
        response = creative_llm.invoke([HumanMessage(content=prompt)])
        
        raw_text = response.content
        if isinstance(raw_text, list):
            texts = [part.get("text", "") for part in raw_text if isinstance(part, dict) and "text" in part]
            raw_text = "".join(texts).strip()
            if not raw_text:
                raw_text = "".join([str(p) for p in response.content if isinstance(p, str)]).strip()
        else:
            raw_text = str(raw_text).strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        
        try:
            payload = json.loads(raw_text)
            concept_dicts = payload.get("concepts", [])
            if isinstance(concept_dicts, dict):
                concept_dicts = list(concept_dicts.values())
        except Exception as e:
            print(f"[mix_and_create] Failed to parse JSON: {e}")
            concept_dicts = []
            
        concepts_out = []
        for cd in concept_dicts:
            vj = self._coerce_prompt_json_object(cd.get("visual_prompt_json"))
            cd["visual_prompt_json"] = self._merge_pod_visual_prompt_json(vj)
            cd["__raw_debug_json"] = raw_text
            concepts_out.append(DynamicConcept(cd))
            
        return concepts_out

    def _extract_baseline_fields(self, description, vision_analysis: Optional[Dict], normalized_fields: Dict[str, List[str]]) -> Dict[str, str]:
        baseline = {
            "subject": "",
            "action": "",
            "context": "",
            "mood": "",
            "art_style": "",
            "colors": "",
        }

        source = vision_analysis if isinstance(vision_analysis, dict) else {}
        if not source and isinstance(description, str):
            try:
                parsed = json.loads(description)
                if isinstance(parsed, dict):
                    source = parsed
            except Exception:
                source = {}

        key_map = {
            "subject": "subject",
            "action": "action",
            "context": "context",
            "mood": "mood",
            "art_style": "art_style",
            "colors": "colors",
        }
        for out_key, src_key in key_map.items():
            baseline[out_key] = self._get_vision_field_value(source, src_key)

        user_override_map = {
            "subject": "subject",
            "action": "action",
            "context": "context",
            "mood": "mood",
            "art_style": "style",
            "colors": "colors",
        }
        for out_key, field_key in user_override_map.items():
            vals = normalized_fields.get(field_key, [])
            if vals:
                baseline[out_key] = str(vals[0]).strip()

        return baseline

    def _coerce_concept_shape(self, concept: Dict, focus_name: str, baseline: Dict[str, str]) -> Dict:
        focus_to_field = {
            "Subject": "subject",
            "Action": "action",
            "Context": "context",
            "Mood": "mood",
            "Style": "art_style",
            "Color": "colors",
        }
        focused_field = focus_to_field.get(focus_name, "subject")
        concept["focus"] = focus_name

        for field in ["subject", "action", "context", "mood", "art_style", "colors"]:
            val = concept.get(field, [])
            if not isinstance(val, list):
                val = [str(val).strip()] if str(val).strip() else []
            else:
                val = [str(v).strip() for v in val if str(v).strip()]

            if field == focused_field:
                if not val:
                    fallback = baseline.get(field, "") or "Creative variation"
                    val = [fallback]
                while len(val) < 3:
                    val.append(val[-1])
                concept[field] = val[:3]
            else:
                if not val:
                    fallback = baseline.get(field, "") or "Keep baseline"
                    val = [fallback]
                concept[field] = [val[0]]

        concept["title"] = str(concept.get("title", "")).strip() or f"{focus_name} Concept"
        concept["caption"] = str(concept.get("caption", "")).strip() or ""
        concept["logic"] = str(concept.get("logic", "")).strip() or f"Explores {focus_name.lower()} variations while preserving coherence."

        if not str(concept.get("visual_prompt", "")).strip():
            concept["visual_prompt"] = (
                f"{concept['art_style'][0]} T-shirt design featuring {concept['subject'][0]} "
                f"{concept['action'][0]} in {concept['context'][0]}, mood {concept['mood'][0]}, "
                f"colors {concept['colors'][0]}, clean white background, POD ready."
            )
        else:
            concept["visual_prompt"] = str(concept["visual_prompt"]).strip()

        prompt_json = self._coerce_prompt_json_object(concept.get("visual_prompt_json", {}))
        if not prompt_json:
            prompt_json = self._build_prompt_json_from_concept_fields(
                {
                    "subject": concept["subject"][0] if concept.get("subject") else "",
                    "action": concept["action"][0] if concept.get("action") else "",
                    "context": concept["context"][0] if concept.get("context") else "",
                    "mood": concept["mood"][0] if concept.get("mood") else "",
                    "art_style": concept["art_style"][0] if concept.get("art_style") else "",
                    "colors": concept["colors"][0] if concept.get("colors") else "",
                },
                visual_prompt_text=concept.get("visual_prompt", ""),
            )
        concept["visual_prompt_json"] = self._merge_pod_visual_prompt_json(prompt_json)

        return concept

    def iter_concepts(
        self,
        description: str,
        rag_ideas: str,
        user_instruction: Optional[str] = None,
        selected_keywords: Optional[List[str]] = None,
        field_inputs: Optional[Dict[str, List[str]]] = None,
        vision_analysis: Optional[Dict] = None,
        concept_model: Optional[str] = None,
    ):
        pass

        normalized_fields = self._normalize_field_inputs(field_inputs)
        baseline = self._extract_baseline_fields(description, vision_analysis, normalized_fields)

        # ── Smart classification ──
        changed, locked = self._classify_fields(normalized_fields, vision_analysis)
        num_changed = len(changed)
        use_full_exploration = (num_changed == 0)

        print(f"[iter_concepts] changed={list(changed.keys())} locked={list(locked.keys())} → mode={'full' if use_full_exploration else 'smart'}")

        creative_llm, resolved_model = self._get_creative_llm(concept_model)
        print(f"Streaming concept model: {resolved_model}")

        keywords_context = ""
        if selected_keywords and len(selected_keywords) > 0:
            keywords_context = f"\nUser-selected keywords to integrate strongly: {', '.join(selected_keywords)}."

        # ════════════════════════════════════════════════════════════
        #  MODE A: Full exploration (6 groups, original behavior)
        # ════════════════════════════════════════════════════════════
        if use_full_exploration:
            field_context = ""
            if any(len(v) > 0 for v in normalized_fields.values()):
                field_context = f"""

User provided field values (prioritize these):
- Subject: {normalized_fields['subject']}
- Action: {normalized_fields['action']}
- Mood: {normalized_fields['mood']}
- Style: {normalized_fields['style']}
- Colors: {normalized_fields['colors']}
- Context: {normalized_fields['context']}
"""

            # Fields the user explicitly provided (even with 1 value) → lock them,
            # don't use them as a focus dimension so we don't explore what's already fixed.
            _user_provided = {k for k, v in normalized_fields.items() if v}
            # Map focus_key → normalized_fields key (art_style ↔ style)
            _focus_key_to_norm = {
                "subject": "subject",
                "action": "action",
                "context": "context",
                "mood": "mood",
                "art_style": "style",
                "colors": "colors",
            }
            all_focus_specs = [
                ("Subject", "subject"),
                ("Action", "action"),
                ("Context", "context"),
                ("Mood", "mood"),
                ("Style", "art_style"),
                ("Color", "colors"),
            ]
            focus_specs = [
                (fname, fkey)
                for fname, fkey in all_focus_specs
                if _focus_key_to_norm.get(fkey, fkey) not in _user_provided
            ]
            # Always explore at least 1 dimension
            if not focus_specs:
                focus_specs = all_focus_specs[:1]

            print(f"[iter_concepts][full] user_provided={_user_provided} → {len(focus_specs)} focus groups")

            # Yield metadata first
            yield {
                "_meta": True,
                "mode": "full",
                "total": len(focus_specs),
                "groups_info": [{"focus": f[0], "cards": 3} for f in focus_specs],
            }

            for focus_name, focus_key in focus_specs:
                prompt = f"""
You are a Creative Director for POD T-shirt concepts.

Original image analysis:
{description}

RAG inspiration:
{rag_ideas}
{field_context}
{keywords_context}

Generate EXACTLY ONE concept focused on "{focus_name}".

BASELINE LOCKED VALUES:
- subject: {baseline['subject']}
- action: {baseline['action']}
- context: {baseline['context']}
- mood: {baseline['mood']}
- art_style: {baseline['art_style']}
- colors: {baseline['colors']}

Rules:
- Focus field is "{focus_key}".
- For focus field: return EXACTLY 3 distinct values.
- For all other fields: return EXACTLY 1 value each.
- Keep non-focus fields aligned to baseline/user constraints.
- The title, visual_prompt, and visual_prompt_json must use the first value of each field.
- visual_prompt must be detailed and print-on-demand ready.
- visual_prompt_json must be a dynamic JSON prompt object with any creative fields useful for image generation; keep it coherent with visual_prompt.
- focus must be exactly "{focus_name}".
"""
                if user_instruction:
                    prompt += f'\nUser instruction to honor: "{user_instruction}"\n'

                prompt += """
Output ONLY valid JSON. Do not wrap it in markdown blocks. Just return the raw JSON object.
{
  "concept": {
    "title": "...",
    "visual_prompt": "...",
    "visual_prompt_json": { ... },
    "subject": ["..."],
    "action": ["..."],
    "context": ["..."],
    "mood": ["..."],
    "art_style": ["..."],
    "colors": ["..."],
    "caption": "...",
    "logic": "...",
    "focus": "..."
  }
}
"""
                print(
                    f"[iter_concepts][full] requesting focus={focus_name} "
                    f"model={resolved_model} timeout={self.concept_timeout_seconds}s"
                )
                response = creative_llm.invoke([HumanMessage(content=prompt)])
                
                raw_text = response.content
                if isinstance(raw_text, list):
                    texts = [part.get("text", "") for part in raw_text if isinstance(part, dict) and "text" in part]
                    raw_text = "".join(texts).strip()
                    if not raw_text:
                        raw_text = "".join([str(p) for p in response.content if isinstance(p, str)]).strip()
                else:
                    raw_text = str(raw_text).strip()

                if raw_text.startswith("```json"):
                    raw_text = raw_text[7:]
                elif raw_text.startswith("```"):
                    raw_text = raw_text[3:]
                if raw_text.endswith("```"):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()
                
                try:
                    payload = json.loads(raw_text)
                    concept_dict = payload.get("concept", payload)
                except Exception as e:
                    print(f"[iter_concepts] Failed to parse JSON: {e}")
                    concept_dict = {
                        "title": "Failed to parse",
                        "visual_prompt": raw_text,
                        "visual_prompt_json": {},
                        "subject": [baseline.get("subject", "")],
                        "action": [baseline.get("action", "")],
                        "context": [baseline.get("context", "")],
                        "mood": [baseline.get("mood", "")],
                        "art_style": [baseline.get("art_style", "")],
                        "colors": [baseline.get("colors", "")],
                        "caption": "",
                        "logic": "",
                        "focus": focus_name
                    }

                concept_dict = self._coerce_concept_shape(concept_dict, focus_name, baseline)
                concept_dict["__raw_debug_json"] = raw_text
                concept_dict["mode"] = "full"
                yield concept_dict

        # ════════════════════════════════════════════════════════════
        #  MODE B: Smart mode (Cartesian combos, grouped)
        # ════════════════════════════════════════════════════════════
        else:
            groups, primary_field, total_cards = self._build_smart_combos(changed, locked)

            # Build groups_info for metadata
            field_to_focus = {
                "subject": "Subject", "action": "Action", "mood": "Mood",
                "style": "Style", "colors": "Color", "context": "Context",
            }
            primary_focus_name = field_to_focus.get(primary_field, primary_field.capitalize())

            groups_info = []
            for g in groups:
                groups_info.append({
                    "focus": primary_focus_name,
                    "group_value": g["group_value"],
                    "cards": len(g["combos"]),
                })

            # Yield metadata first
            yield {
                "_meta": True,
                "mode": "smart",
                "total": len(groups),
                "total_cards": total_cards,
                "primary_field": primary_field,
                "changed_fields": list(changed.keys()),
                "groups_info": groups_info,
            }

            # Map field names for prompt building
            field_key_to_concept = {
                "subject": "subject", "action": "action", "mood": "mood",
                "style": "art_style", "colors": "colors", "context": "context",
            }

            pass

            for gi, group in enumerate(groups):
                # Build sub-cards for this group
                sub_cards = []
                for si, combo in enumerate(group["combos"]):
                    # Map field_inputs keys to concept keys
                    concept_fields = {}
                    for fk, ck in field_key_to_concept.items():
                        concept_fields[ck] = combo.get(fk, locked.get(fk, ""))

                    # AI generates title + visual_prompt for this combo
                    combo_desc = ", ".join([
                        f"{k}: {v}" for k, v in concept_fields.items() if v
                    ])
                    prompt = f"""
You are a Creative Director for POD T-shirt concepts.

Original image analysis:
{description}

RAG inspiration:
{rag_ideas}
{keywords_context}

Generate a catchy title, a highly detailed visual_prompt, and a dynamic visual_prompt_json for this specific T-shirt concept:

FIELD VALUES:
- subject: {concept_fields.get('subject', '')}
- action: {concept_fields.get('action', '')}
- context: {concept_fields.get('context', '')}
- mood: {concept_fields.get('mood', '')}
- art_style: {concept_fields.get('art_style', '')}
- colors: {concept_fields.get('colors', '')}

Rules:
- visual_prompt must be detailed, cohesive, and print-on-demand ready.
- Include composition, lighting, texture hints.
- STRICT minimum length: visual_prompt MUST be at least 800 words.
- visual_prompt_json must be a dynamic JSON prompt object tailored to this combo; include any creative structured fields that help image generation; keep it coherent with visual_prompt.
- Title should be catchy and commercial.
- Explain briefly why this combo works in 'logic'.
"""
                    if user_instruction:
                        prompt += f'\nUser instruction to honor: "{user_instruction}"\n'

                    min_words = 800
                    max_attempts = 3
                    smart_result: Dict[str, Any] = {}
                    visual_prompt_text = ""
                    visual_prompt_words = 0
                    prompt_with_feedback = prompt

                    for attempt in range(1, max_attempts + 1):
                        prompt_with_feedback_call = prompt_with_feedback + '\nCRITICAL REQUIREMENT: Output ONLY valid JSON representing the response. Do not wrap it in markdown blocks. Just return the raw JSON object: { "title": "...", "visual_prompt": "...", "visual_prompt_json": {...}, "caption": "...", "logic": "..." }'
                        print(
                            f"[iter_concepts][smart] requesting {gi + 1}.{si + 1} "
                            f"attempt={attempt}/{max_attempts} model={resolved_model} "
                            f"timeout={self.concept_timeout_seconds}s"
                        )
                        response = creative_llm.invoke([HumanMessage(content=prompt_with_feedback_call)])
                        
                        raw_text = response.content
                        if isinstance(raw_text, list):
                            texts = [part.get("text", "") for part in raw_text if isinstance(part, dict) and "text" in part]
                            raw_text = "".join(texts).strip()
                            if not raw_text:
                                raw_text = "".join([str(p) for p in response.content if isinstance(p, str)]).strip()
                        else:
                            raw_text = str(raw_text).strip()

                        if raw_text.startswith("```json"):
                            raw_text = raw_text[7:]
                        elif raw_text.startswith("```"):
                            raw_text = raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        raw_text = raw_text.strip()
                        
                        try:
                            smart_result = json.loads(raw_text)
                        except Exception as e:
                            print(f"[iter_concepts][smart] JSON parsing failed: {e}")
                            smart_result = {"visual_prompt": raw_text, "visual_prompt_json": {}}

                        smart_result["__raw_debug_json"] = raw_text
                        visual_prompt_text = str(smart_result.get("visual_prompt", "") or "").strip()
                        visual_prompt_words = self._count_words(visual_prompt_text)
                        print(
                            f"[iter_concepts][smart] {gi + 1}.{si + 1} "
                            f"words={visual_prompt_words} attempt={attempt}/{max_attempts}"
                        )
                        if visual_prompt_words >= min_words:
                            break

                        prompt_with_feedback = (
                            f"{prompt}\n\n"
                            f"LENGTH REVISION (attempt {attempt}):\n"
                            f"- Your previous visual_prompt had {visual_prompt_words} words.\n"
                            f"- Rewrite ONLY visual_prompt so it has at least {min_words} words.\n"
                            f"- Keep all field values and stylistic intent unchanged.\n"
                            f"- Return the full JSON object again with title, visual_prompt, visual_prompt_json, caption, logic."
                        )

                    if visual_prompt_words < min_words:
                        smart_result["visual_prompt"] = self._expand_prompt_to_min_words(
                            visual_prompt_text,
                            concept_fields=concept_fields,
                            min_words=min_words,
                        )
                        visual_prompt_words = self._count_words(smart_result["visual_prompt"])
                        print(
                            f"[iter_concepts][smart] {gi + 1}.{si + 1} "
                            f"fallback_expand_applied words={visual_prompt_words}"
                        )

                    prompt_json = self._coerce_prompt_json_object(smart_result.get("visual_prompt_json", {}))
                    if not prompt_json:
                        prompt_json = self._build_prompt_json_from_concept_fields(
                            concept_fields,
                            visual_prompt_text=str(smart_result.get("visual_prompt", "") or ""),
                        )
                    else:
                        prompt_json = self._merge_pod_visual_prompt_json(prompt_json)

                    sub_card = {
                        "sub_label": f"{gi + 1}.{si + 1}",
                        "title": smart_result.get("title", ""),
                        "visual_prompt": smart_result.get("visual_prompt", ""),
                        "visual_prompt_json": prompt_json,
                        "caption": smart_result.get("caption", ""),
                        "logic": smart_result.get("logic", ""),
                        "__raw_debug_json": smart_result.get("__raw_debug_json", ""),
                        **concept_fields,
                    }
                    sub_cards.append(sub_card)

                    # Progressive streaming: emit each sub-card as soon as it is ready so the UI can render 1.1, then 1.2, etc.
                    yield {
                        "_streaming_partial": True,
                        "mode": "smart",
                        "group_index": gi + 1,
                        "group_value": group["group_value"],
                        "primary_focus_name": primary_focus_name,
                        "sub_index": si + 1,
                        "subs_in_group": len(group["combos"]),
                        "sub_card": sub_card,
                    }

                # Build concept dict compatible with frontend
                # Focus field gets all values from this group's combos
                # Other fields get single locked value
                focus_key = field_key_to_concept.get(primary_field, primary_field)
                concept_dict = {
                    "title": f"{group['group_value']} Variations",
                    "visual_prompt": sub_cards[0]["visual_prompt"] if sub_cards else "",
                    "visual_prompt_json": sub_cards[0].get("visual_prompt_json", {}) if sub_cards else {},
                    "caption": sub_cards[0].get("caption", "") if sub_cards else "",
                    "logic": sub_cards[0].get("logic", "") if sub_cards else "",
                    "focus": primary_focus_name,
                    "mode": "smart",
                    "sub_cards": sub_cards,
                    "group_value": group["group_value"],
                }

                # Fill field arrays
                for fk, ck in field_key_to_concept.items():
                    if fk == primary_field:
                        concept_dict[ck] = [group["group_value"]]
                    elif fk in changed:
                        # Secondary changed fields: collect unique values across combos
                        vals = list(dict.fromkeys(sc[ck] for sc in sub_cards if sc.get(ck)))
                        concept_dict[ck] = vals if vals else [locked.get(fk, "")]
                    else:
                        concept_dict[ck] = [locked.get(fk, "")]

                yield concept_dict

    def generate_mixed_subcards(self, concepts: list, vision_analysis: dict = None, strict_subject_swap: bool = False) -> list:
        """
        Second AI pass: given the 5 raw concept objects, the AI autonomously decides
        which fields to combine from which group to create 9 interesting sub-cards
        per group, and writes a creative visual_prompt for each combination.

        Returns a list of 5 group dicts, each with 9 sub-card dicts.
        """
        from pydantic import BaseModel, Field
        from typing import List, Dict

        class SubCard(BaseModel):
            sub_label: str = Field(description="Short label e.g. '1.1', '1.2', ..., '1.9'")
            subject: str   = Field(description="Subject used in this sub-card")
            action: str    = Field(description="Action used in this sub-card")
            context: str   = Field(description="Context/setting used in this sub-card")
            mood: str      = Field(description="Mood used in this sub-card")
            art_style: str = Field(description="Art style used in this sub-card")
            colors: str    = Field(description="Color palette used in this sub-card")
            visual_prompt: str = Field(description="Full creative T-shirt visual_prompt built from the above fields")
            mixed_fields: List[str] = Field(description="List of field names that were taken from a concept other than the primary group (e.g. ['action', 'mood'])")

        class ConceptGroup(BaseModel):
            focus: str           = Field(description="The primary focus of this group: Subject, Action, Style, Context, or Mood")
            group_index: int     = Field(description="1-based index of this group (1-5)")
            cross_label: str     = Field(description="Short label for the cross dimension e.g. 'Subject × Action'")
            logic: str           = Field(description="One sentence explaining what this group explores")
            sub_cards: List[SubCard] = Field(description="Exactly 9 sub-cards for this group")

        class MixedGroups(BaseModel):
            groups: List[ConceptGroup] = Field(description="Exactly 5 concept groups")

        # Serialize the 5 concepts into a readable summary for the AI
        concept_summaries = []
        for c in concepts:
            focus = c.get("focus", "?")
            summary = f"- Group [{focus}]:"
            for field in ["subject", "action", "context", "mood", "art_style", "colors"]:
                val = c.get(field, [])
                vals = val if isinstance(val, list) else [val]
                summary += f"\n    {field}: {' | '.join(vals)}"
            concept_summaries.append(summary)
        concepts_text = "\n".join(concept_summaries)

        vision_analysis = vision_analysis or {}
        vision_context_lines = []
        if vision_analysis:
            visual_prompt = str(vision_analysis.get("visual_prompt", "")).strip()
            if visual_prompt:
                vision_context_lines.append(f"- Baseline visual prompt: {visual_prompt}")
            key_elements = vision_analysis.get("key_elements", [])
            if isinstance(key_elements, list) and key_elements:
                vision_context_lines.append(f"- Key visual anchors: {', '.join([str(v).strip() for v in key_elements if str(v).strip()])}")
            for key, label in [
                ("composition", "Composition"),
                ("lighting", "Lighting"),
                ("camera_angle", "Camera angle"),
                ("linework", "Linework"),
                ("texture_details", "Texture details"),
                ("negative_constraints", "Negative constraints"),
            ]:
                val = str(vision_analysis.get(key, "")).strip()
                if val:
                    vision_context_lines.append(f"- {label}: {val}")

        vision_context = "\n".join(vision_context_lines) if vision_context_lines else "- No additional baseline detail."
        strict_subject_rule = """
- STRICT SUBJECT SWAP MODE IS ON:
  - For group_index=1 (Subject group), ONLY the subject may vary.
  - Action, context, mood, art_style, and colors must stay exactly the same as the primary Subject group base values.
  - Do not mix non-subject fields for Subject sub-cards.
""" if strict_subject_swap else ""

        prompt = f"""
You are a Creative Director for a T-shirt POD business.

Below are 5 concept groups generated for a T-shirt design.
Each group has a "focus" field with 3 variants, and other fields with 1 value each.

{concepts_text}

TASK:
For each of the 5 groups, create EXACTLY 9 sub-cards.
Each sub-card must have all 6 fields: subject, action, context, mood, art_style, colors.

MIXING RULES (follow intelligently – the goal is maximum creative diversity):
- For the primary group's focus field: cycle through its 3 variants (each variant appears in 3 sub-cards).
- For the other 5 fields: CREATIVELY mix values from OTHER groups wherever it improves diversity and aesthetics.
  - You can fix 2-3 fields from the primary group, and swap 2-3 fields from other groups.
  - Make sure each of the 9 sub-cards feels like a genuinely DIFFERENT and interesting T-shirt design.
  - Avoid repeating the exact same combination twice.
{strict_subject_rule}

BASELINE DETAIL CONTEXT (preserve quality/style cues when relevant):
{vision_context}

For EACH sub-card:
1. Choose the field values (clearly state which group each field came from if it's NOT from the primary group).
2. Write a CREATIVE, COHESIVE visual_prompt using all 6 chosen field values.
   The prompt should flow naturally and be detailed enough for high-fidelity generation.
   - Include composition, lighting, texture/material hints, and print constraints if available from baseline context.
   - MUST write a minimum of 200 words for the visual_prompt. There is no maximum limit. Make it as highly detailed and comprehensive as the original vision analysis prompt in length and depth.
   - Ensure print-on-demand readiness: prefer solid white/off-white garment backdrop unless the concept explicitly needs another treatment; avoid accidental black letterboxing unless intentional to the style.

OUTPUT:
Return a JSON with EXACTLY 5 groups.
Each group has:
- focus: name of the primary group (Subject/Action/Style/Context/Mood)
- group_index: 1-5 in this fixed order: Subject=1, Action=2, Style=3, Context=4, Mood=5
- cross_label: short string like "Subject × Action" describing what's mixed
- logic: 1 sentence explaining the group's creative strategy
- sub_cards: list of EXACTLY 9 sub-cards, labeled 1.1 through 1.9 (or 2.1-2.9, etc.)
  Each sub_card: sub_label, subject, action, context, mood, art_style, colors, visual_prompt, mixed_fields
"""

        structured_llm = self.creative_llm.with_structured_output(MixedGroups)
        response = structured_llm.invoke([HumanMessage(content=prompt)])

        # Convert Pydantic objects to plain dicts
        groups = []
        for g in response.groups:
            group_dict = g.model_dump()
            groups.append(group_dict)
        return groups

    def refine_prompt_with_instruction(self, current_prompt: str, instruction: str, current_concept: dict = None) -> dict:
        from pydantic import BaseModel, Field

        class RefinedConcept(BaseModel):
            visual_prompt: str = Field(description="The rewritten, highly detailed, flowing text-to-image prompt.")
            visual_prompt_json: Dict[str, Any] = Field(default_factory=dict, description="Structured JSON prompt aligned with the rewritten concept.")
            subject: str = Field(description="Updated subject based on the instruction, or original if unchanged")
            action: str = Field(description="Updated action based on the instruction, or original if unchanged")
            context: str = Field(description="Updated context based on the instruction, or original if unchanged")
            mood: str = Field(description="Updated mood based on the instruction, or original if unchanged")
            art_style: str = Field(description="Updated art style based on the instruction, or original if unchanged")
            colors: str = Field(description="Updated colors based on the instruction, or original if unchanged")

        current_concept = current_concept or {}
        concept_context = ""
        if current_concept:
            concept_context = f"\nCURRENT FIELDS:\n"
            for k, v in current_concept.items():
                concept_context += f"- {k.capitalize()}: {v}\n"

        prompt = f"""
        You are an expert at writing highly detailed cohesive text-to-image prompts (for Midjourney/Dall-E).
        
        The user wants to refine an existing image generation concept based on a specific instruction.
        
        CURRENT PROMPT:
        {current_prompt}
        {concept_context}
        
        USER INSTRUCTION:
        {instruction}
        
        Task:
        1. Rewrite the CURRENT PROMPT to carefully incorporate the USER INSTRUCTION. Ensure the new prompt remains cohesive, highly detailed, and flowing.
        2. Update the CURRENT FIELDS to reflect any changes caused by the user instruction. If a field is not affected by the instruction, keep its original value.
        3. Return visual_prompt_json that is consistent with the rewritten prompt and updated fields.

        CRITICAL OVERRIDE RULE:
        - If the user explicitly requests a subject replacement (example: "change subject to cat"), you MUST set subject to the requested new subject and remove references to the old subject in both visual_prompt and visual_prompt_json.
        """
        
        structured_llm = self.creative_llm.with_structured_output(RefinedConcept)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        result = response.model_dump()

        # Ensure missing fields are backfilled from current concept for stability.
        for field in ("subject", "action", "context", "mood", "art_style", "colors"):
            if not str(result.get(field, "")).strip():
                result[field] = str(current_concept.get(field, "")).strip()

        # Deterministic subject override when user instruction is explicit.
        explicit_subject = ""
        subject_patterns = [
            r"change\s+subject\s+to\s+([^\n,.;]+)",
            r"thay\s*doi\s+subject\s*(?:to|thanh)\s+([^\n,.;]+)",
            r"thay\s*đổi\s+subject\s*(?:to|thành)\s+([^\n,.;]+)",
            r"subject[^\n]{0,80}?(?:to|thanh|thành)\s+([^\n,.;]+)",
        ]
        for pattern in subject_patterns:
            m = re.search(pattern, str(instruction or ""), flags=re.IGNORECASE)
            if m:
                explicit_subject = str(m.group(1) or "").strip().strip("'\"")
                if explicit_subject:
                    break
        if explicit_subject:
            old_subject = str(result.get("subject", "")).strip() or str(current_concept.get("subject", "")).strip()
            result["subject"] = explicit_subject
            vp = str(result.get("visual_prompt", "") or "")
            if vp and old_subject and old_subject.lower() != explicit_subject.lower():
                result["visual_prompt"] = re.sub(re.escape(old_subject), explicit_subject, vp, flags=re.IGNORECASE)

        concept_fields = {
            "subject": str(result.get("subject", "")).strip(),
            "action": str(result.get("action", "")).strip(),
            "context": str(result.get("context", "")).strip(),
            "mood": str(result.get("mood", "")).strip(),
            "art_style": str(result.get("art_style", "")).strip(),
            "colors": str(result.get("colors", "")).strip(),
        }

        llm_json = self._coerce_prompt_json_object(result.get("visual_prompt_json", {}))
        fallback_json = self._build_prompt_json_from_concept_fields(
            concept_fields,
            visual_prompt_text=str(result.get("visual_prompt", "") or ""),
        )
        # Prefer LLM JSON when available, but guarantee key concept fields exist.
        prompt_json = dict(fallback_json)
        prompt_json.update(llm_json)
        if concept_fields["subject"] and not str(prompt_json.get("subject", "")).strip():
            prompt_json["subject"] = concept_fields["subject"]
        if concept_fields["action"] and not str(prompt_json.get("action", "")).strip():
            prompt_json["action"] = concept_fields["action"]
        if concept_fields["context"] and not str(prompt_json.get("context", "")).strip():
            prompt_json["context"] = concept_fields["context"]
        if concept_fields["mood"] and not str(prompt_json.get("mood", "")).strip():
            prompt_json["mood"] = concept_fields["mood"]

        result["visual_prompt_json"] = self._merge_pod_visual_prompt_json(prompt_json)
        return result

    def run(self, image_path, user_instruction=None):
        # 1. Vision Analysis (Structured)
        vision_analysis = self.analyze_image(image_path)
        
        # Convert to string for downstream tasks
        if hasattr(vision_analysis, 'model_dump_json'):
             description_str = vision_analysis.model_dump_json()
             analysis_dict = vision_analysis.model_dump()
        else:
             description_str = str(vision_analysis)
             analysis_dict = {"raw_text": description_str} # Fallback
             
        # 2. RAG Retrieval
        rag_ideas = self.retrieve_ideas(vision_analysis, user_instruction)
        
        # 3. Creative Generation
        concepts = self.mix_and_create(description_str, rag_ideas, user_instruction)
        
        return {
            "analysis": description_str, # Keep string for backward compat if needed, or update frontend to use 'vision_analysis'
            "vision_analysis": analysis_dict, # Pass structured data
            "rag_context": rag_ideas,
            "concepts": [c.model_dump() for c in concepts]
        }

    def remix_concept(self, original_concept: dict, allowed_keywords: List[str] = None):
        print(f"Remixing concept: {original_concept.get('title')}...")
        from pydantic import BaseModel, Field
        from typing import List

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Detailed prompt for AI image generator")
            subject: List[str] = Field(description="3 alternative subjects (first is primary)")
            action: List[str] = Field(description="3 alternative actions (first is primary)")
            context: List[str] = Field(description="3 alternative environments (first is primary)")
            mood: List[str] = Field(description="3 alternative moods (first is primary)")
            art_style: List[str] = Field(description="3 alternative art styles (first is primary)")
            colors: List[str] = Field(description="3 alternative color palettes (first is primary)")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Why this variation works")
            focus: str = Field(description="The specific element that was changed")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

        focus_area = original_concept.get('focus', 'General')
        
        # tailoring the prompt based on what kind of concept this is
        specific_instruction = ""
        if focus_area == 'Subject':
            keyword_constraint = ""
            if allowed_keywords and len(allowed_keywords) > 0:
                 keyword_constraint = f" The NEW SUBJECT MUST be chosen from this list: {', '.join(allowed_keywords)}."
            
            specific_instruction = f"This concept was about a Subject Twist. Create 1 NEW VARIATION with a **DIFFERENT SUBJECT** performing the SAME Action.{keyword_constraint} **CRITICAL: You MUST KEEP the original Action, Context, Art Style, and Colors EXACTLY AS IS.**"
        elif focus_area == 'Action':
            specific_instruction = "This concept was about an Action Switch. Create 1 NEW VARIATION with the SAME Subject performing a **DIFFERENT ACTION**. **CRITICAL: You MUST KEEP the original Subject, Context, Art Style, and Colors EXACTLY AS IS.**"
        elif focus_area == 'Style' or focus_area == 'Visual':
            specific_instruction = "This concept was about a Visual Remix. Create 1 NEW VARIATION with the SAME Subject/Action/Context but a **DIFFERENT ART STYLE AND COLOR PALETTE**. **CRITICAL: You MUST KEEP the original Subject, Action, and Context EXACTLY AS IS.**"
        elif focus_area == 'Context':
            specific_instruction = "This concept was about a Context Shift. Create 1 NEW VARIATION with the SAME Subject/Action but in a **DIFFERENT CONTEXT**. **CRITICAL: You MUST KEEP the original Subject, Action, Art Style, and Colors EXACTLY AS IS.**"
        else:
             specific_instruction = "Create 1 BOLD variation of this concept using SCAMPER techniques."

        prompt = f"""
        You are a Creative Director for a POD T-shirt business.
        
        Original Concept:
        Title: {original_concept.get('title')}
        Visual: {original_concept.get('visual_prompt')}
        Caption: {original_concept.get('caption')}
        Subject: {original_concept.get('subject')}
        Action: {original_concept.get('action')}
        Context: {original_concept.get('context')}
        Mood: {original_concept.get('mood')}
        Art Style: {original_concept.get('art_style')}
        Colors: {original_concept.get('colors')}
        Focus: {focus_area}
        
        Task:
        {specific_instruction}
        
        CRITICAL RULES:
        1. **Generate ONLY 1 Concept**.
        2. **STRICTLY Maintain Non-Focus Elements**: If the focus is Context, DO NOT change the Art Style or Subject. If the focus is Action, DO NOT change the Context.
        3. **New Caption**: Update the caption to match the new variation.
        4. **Detailed Visual Prompt**: Write a full prompt for image generation that explicitly describes the elements to keep.
        """
        
        structured_llm = self.creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return [c.model_dump() for c in response.concepts]

    def get_preset_keywords(self):
        """Extracts keywords from the local CSV file."""
        csv_path = os.path.join(base_dir, "..", "50Topics_Keyword.csv")
        keywords = []
        try:
            if not os.path.exists(csv_path):
                print(f"CSV not found at {csv_path}")
                return []
                
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line: continue
                    
                    # Exclude headers
                    if line.startswith("Dưới đây là") or line.startswith("Các Từ khóa"): continue
                    if re.match(r'^\d+\.', line) and " – " not in line: continue
                    
                    keyword = None
                    if " – " in line:
                        keyword = line.split(" – ")[0]
                    elif " - " in line:
                        keyword = line.split(" - ")[0]
                    elif ":" in line:
                        parts = line.split(":")
                        if len(parts[0]) < 30: keyword = parts[0]
                    
                    if keyword:
                        clean_kw = re.sub(r'^\d+\.\s*', '', keyword).strip()
                        if clean_kw and len(clean_kw) < 50:
                            keywords.append(clean_kw)
        except Exception as e:
            print(f"Error reading preset keywords: {e}")
            return []
            
        # Preserve order and deduplicate while returning full set for downstream suggestion logic.
        deduped = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                deduped.append(kw)
                seen.add(kw)
        return deduped

    def get_viral_keywords(self, vision_analysis_str):
        """Uses RAG to find viral keywords and returns them as a structured list."""
        # 1. Get loose ideas from RAG
        rag_text = self.retrieve_ideas(vision_analysis_str)
        
        # 2. Use LLM to extract/refine into a clean list
        # 2. Use LLM to extract/refine into a clean list
        class KeywordList(BaseModel):
            keywords: List[str]
            
        extraction_prompt = f"""
        Based on the following RAG research text, extract the top 5-10 most relevant, viral, and funny T-shirt keywords or short phrases.
        
        RAG Research:
        {rag_text}
        
        Return ONLY the list of keywords.
        """
        
        structured_llm = self.vision_llm.with_structured_output(KeywordList)
        try:
            response = structured_llm.invoke([HumanMessage(content=extraction_prompt)])
            return {"rag_text": rag_text, "keywords": response.keywords}
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return {"rag_text": rag_text, "keywords": []}

    def suggest_matrix_criteria(self, vision_analysis_dict: dict) -> dict:
        print("Suggesting Auto Matrix Criteria...")
        from pydantic import BaseModel, Field
        from typing import List

        class MatrixCriteria(BaseModel):
            subjects: List[str] = Field(description="5 highly creative alternative subjects/characters for the design")
            actions: List[str] = Field(description="5 totally different, funny, or trending actions/poses")
            moods: List[str] = Field(description="5 distinct emotional expressions or moods")

        prompt = f"""
        You are an expert Print-on-Demand Creative Director.
        
        We have analyzed a base image to extract its core attributes:
        Base Analysis: {json.dumps(vision_analysis_dict)}
        
        We are preparing an "Auto Matrix" generation where we swap out the Subject, Action, and Mood, while keeping the original Art Style and Context.
        
        Task: 
        Suggest exactly 5 creative, highly marketable alternative Subjects, 5 alternative Actions, and 5 alternative Moods that would fit perfectly within the original image's Art Style.
        """

        structured_llm = self.creative_llm.with_structured_output(MatrixCriteria)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.model_dump()

    def matrix_create_concept(self, vision_analysis_dict: dict, subject: str, action: str, mood: str) -> dict:
        print(f"Creating Matrix Concept: Subject={subject}, Action={action}, Mood={mood}")
        from pydantic import BaseModel, Field

        class MatrixConcept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Full prompt for AI image generator")
            subject: str = Field(description="The main character/subject")
            action: str = Field(description="What the subject is doing")
            context: str = Field(description="Environment or background elements")
            mood: str = Field(description="The emotional tone or vibe of the concept")
            art_style: str = Field(description="The specific art style used")
            colors: str = Field(description="The color palette used")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Why this Matrix combination works")
            focus: str = Field(description="Matrix Generation")

        prompt = f"""
        You are a Creative Director for a POD (Print on Demand) T-shirt business.
        We are doing a "Matrix Generation" where we force-swap elements of a base image.
        
        BASE IMAGE ANALYSIS (Keep its Art Style, Colors, and Context):
        {json.dumps(vision_analysis_dict)}
        
        NEW MANDATORY ELEMENTS TO USE:
        - SUBJECT: {subject}
        - ACTION/POSE: {action}
        - MOOD/VIBE: {mood}
        
        Task:
        Create a single, highly detailed cohesive text-to-image prompt (for Midjourney/Dall-E) that seamlessly combines the NEW SUBJECT, NEW ACTION, and NEW MOOD, but explicitly uses the Art Style, Lines, Coloring style, and Environment/Context from the BASE IMAGE ANALYSIS.
        """
        
        structured_llm = self.creative_llm.with_structured_output(MatrixConcept)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.model_dump()

if __name__ == "__main__":
    # Test with a dummy image path or ask user
    # Ideally checking if an image exists or creating a dummy
    agent = CreativeAgent()
    
    # Check for test image
    test_image = "test_image.jpg"
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    if os.path.exists(test_image):
        agent.run(test_image)
    else:
        print(f"Please provide an image path. Usage: python creative_agent.py <path_to_image>")
