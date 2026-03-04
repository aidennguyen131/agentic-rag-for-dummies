import sys
import os
import base64
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
from typing import List, Dict, Optional

class CreativeAgent:
    def __init__(self):
        print("Initializing Creative Agent...")
        self.rag = RAGSystem()
        self.rag.initialize()

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
        self.vision_llm = ChatGoogleGenerativeAI(model=self.default_vision_model, max_tokens=4000)
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

        if model_name.startswith("gemini-"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        elif model_name.startswith("gpt-"):
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for GPT concept models.")
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
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{b64}"

    def analyze_image(self, image_path):
        print(f"Analyzing image: {image_path}...")
        
        from pydantic import BaseModel, Field
        from typing import List

        class VisionAnalysis(BaseModel):
            subject: str = Field(description="The main character or subject")
            action: str = Field(description="What the subject is doing")
            context: str = Field(description="Environment, background, or setting")
            art_style: str = Field(description="The visual style (e.g. vintage, vector, photo-real)")
            colors: str = Field(description="Dominant colors and palette description")
            mood: str = Field(description="The emotional tone (e.g. funny, scary, serious)")
            key_elements: List[str] = Field(description="List of other important visual elements")
            composition: str = Field(default="", description="Composition breakdown: foreground, midground, background, focal point, and framing")
            lighting: str = Field(default="", description="Lighting direction, intensity, contrast, and shadows/highlights")
            camera_angle: str = Field(default="", description="Viewpoint and lens feeling (e.g. eye-level, low-angle, close-up, wide shot)")
            linework: str = Field(default="", description="Line quality details: thickness, cleanliness, sketchiness, edge treatment")
            texture_details: str = Field(default="", description="Surface/material textures and print-relevant details (grain, halftone, distress, brush marks)")
            typography: str = Field(default="", description="Text treatment if present: wording area, font vibe, layout, readability")
            negative_constraints: str = Field(default="", description="What should be avoided in generation to preserve style fidelity and print usability")
            visual_prompt: str = Field(description="A highly detailed cohesive text-to-image prompt")

        image_data_url = self.encode_image(image_path)
        
        prompt_instruction = """Analyze this image for a POD T-shirt design and return JSON strictly matching the schema.

Your analysis must be detailed, concrete, and production-oriented for print design.

Requirements:
1. SUBJECT/ACTION/CONTEXT/MOOD/STYLE/COLORS:
   - Be specific, avoid vague labels.
   - Preserve what is actually in the image (do not "clean up" rough art unless it is truly clean).
2. KEY_ELEMENTS:
   - List 6-12 important visual elements (objects, symbols, motifs, accessories, background items).
3. COMPOSITION:
   - Describe foreground/midground/background, focal point, subject scale, spacing, and framing.
4. LIGHTING:
   - Describe key light direction, contrast, shadows, highlights, color temperature.
5. CAMERA_ANGLE:
   - Describe perspective and shot type (eye-level/low-angle/top-down, close/medium/wide).
6. LINEWORK:
   - Describe line thickness, edge quality, sketchiness/cleanliness, contour behavior.
7. TEXTURE_DETAILS:
   - Describe textures and print-relevant effects (grain, halftone, noise, distress, brush strokes, ink bleed, gradients).
8. TYPOGRAPHY:
   - If text exists, describe placement, hierarchy, style, and readability; otherwise explicitly say "No visible typography".
9. NEGATIVE_CONSTRAINTS:
   - List what to avoid in generation (wrong style shifts, clutter, unreadable details, trademark/copyrighted characters, NSFW).
10. VISUAL_PROMPT:
   - Write one cohesive, high-detail, generation-ready prompt that faithfully preserves the original style and composition.
   - Minimum length: 250 words. Target range: 250-350 words.
   - Include explicit details for subject anatomy/pose, props, foreground/midground/background, lighting setup, texture treatment, line quality, palette behavior, and print composition.
   - Include medium/style/texture/lighting/composition details and clear print-friendly constraints.
   - Keep it safe and free of trademarked character names.

Do not output markdown. Output only valid JSON.
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
        # Use structured output
        structured_llm = self.vision_llm.with_structured_output(VisionAnalysis)
        response = structured_llm.invoke([message])
        
        # Return the dict, or handle it in run()
        # For compatibility with existing string-based flow, we might need to adjust,
        # but let's return the dict object so we can pass it to frontend.
        # However, mix_and_create expects a description string.
        # We should return the dict, and let caller handle it.
        # Wait, if we change return type, we break mix_and_create which expects a string description?
        # Let's check mix_and_create usage. It uses 'description' in prompt. 
        # We can accept dict in mix_and_create or convert dict to string there.
        # Let's return the object (Pydantic model) and convert to string where needed.
        return response

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
            baseline_val = str(source.get(baseline_key, "")).strip()

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
        force_regenerate: bool = False
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
        styles = self._load_style_rows()
        style_names = [row.get("Style", "").strip() for row in styles if row.get("Style", "").strip()]
        style_categories = [row.get("Category", "").strip() for row in styles if row.get("Category", "").strip()]
        style_usages = [row.get("Usage", "").strip() for row in styles if row.get("Usage", "").strip()]

        topic_keywords = self.get_preset_keywords()
        valid_target = target_field if target_field in normalized else ""

        prompt = f"""
You are a POD concept input assistant.

Goal:
- Return suggestions for 6 fields: subject, action, mood, style, colors, context.
- Each field can contain 1 to 3 values.
- Keep suggestions coherent (avoid contradictions like mood=dark while colors=pastel cute unless intentionally stylistic).

Rules:
- If target_field is set:
  - Operate ONLY on that field and keep all other fields unchanged.
  - If force_regenerate is FALSE: keep existing values for target field, then top up to max 3.
  - If force_regenerate is TRUE: regenerate target field values from scratch.
- If target_field is empty:
  - If force_regenerate is FALSE: keep existing values and top up to max 3 for all fields.
  - If force_regenerate is TRUE: regenerate all fields from scratch.
- Maximum 3 values per field.
- Prefer using these data sources:
  1) Topic keywords from 50Topics_Keyword.csv
  2) Styles metadata from ArtStyles_v2.csv
- SUBJECT-SPECIFIC DIVERSITY RULE:
  - Subject suggestions must be different character identities (different species/archetypes/professions), not outfit/accessory variants of the same character.
  - Bad: "brown sloth", "sloth with goggles", "sloth in jacket".
  - Good: "red panda snowboarder", "retro robot courier", "samurai frog drummer".
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
- Each field must be a list with up to 3 concise values.
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
            vision_value = str((vision_analysis or {}).get(vision_key, "")).strip().lower()
            if not vision_value:
                return values
            filtered = [v for v in values if str(v).strip().lower() != vision_value]
            return filtered or values

        result = {}
        for field in normalized.keys():
            old_vals = normalized[field]
            new_vals = [str(v).strip() for v in suggested.get(field, []) if str(v).strip()][:3]
            new_vals = _filter_verbatim_vision(field, new_vals)
            if field == "subject":
                new_vals = self._dedupe_subject_candidates(
                    new_vals,
                    existing=old_vals if not force_regenerate else None,
                    limit=3,
                )

            if valid_target and field != valid_target:
                result[field] = old_vals
                continue

            if not force_regenerate:
                # Preserve user/current values first, then append unique AI suggestions.
                merged = list(old_vals)
                if field == "subject":
                    appended = self._dedupe_subject_candidates(new_vals, existing=merged, limit=max(0, 3 - len(merged)))
                    merged.extend(appended)
                else:
                    seen = {v.lower() for v in merged}
                    for candidate in new_vals:
                        key = candidate.lower()
                        if key in seen:
                            continue
                        merged.append(candidate)
                        seen.add(key)
                        if len(merged) >= 3:
                            break
                result[field] = merged[:3]
            else:
                result[field] = new_vals
        return result

    def mix_and_create(self, description, rag_ideas, user_instruction=None, selected_keywords=None, field_inputs=None, concept_model: Optional[str] = None):
        print("Mixing ideas...")
        from pydantic import BaseModel, Field
        from typing import List

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Full prompt for AI image generator")
            # Structured breakdown – each field is a list of 3 AI-suggested alternatives
            subject: List[str] = Field(description="3 alternative main characters/subjects (the first one is the primary choice)")
            action: List[str] = Field(description="3 alternative actions the subject could perform (the first one is the primary choice)")
            context: List[str] = Field(description="3 alternative environments or background elements (the first one is the primary choice)")
            mood: List[str] = Field(description="3 alternative emotional tones or vibes (the first one is the primary choice)")
            art_style: List[str] = Field(description="3 alternative art styles (the first one is the primary choice)")
            colors: List[str] = Field(description="3 alternative color palettes (the first one is the primary choice)")

            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Business logic or why this design works")
            focus: str = Field(description="The specific element that was changed, e.g., 'Subject', 'Action', 'Context', 'Mood', 'Style', 'Color'")

        class DesignConcepts(BaseModel):
            concepts: List[Concept]

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
        
        Output Format:
        Return a JSON with EXACTLY 6 concepts (one per strategy above).
        For each concept, provide:
        - title: Catchy name (based on the primary/first variant).
        - visual_prompt: Full detailed prompt using the PRIMARY (first) value of each field.
        - subject: LIST with EXACTLY 3 values if focus="Subject", else LIST with EXACTLY 1 value.
        - action: LIST with EXACTLY 3 values if focus="Action", else LIST with EXACTLY 1 value.
        - context: LIST with EXACTLY 3 values if focus="Context", else LIST with EXACTLY 1 value.
        - mood: LIST with EXACTLY 3 values if focus="Mood", else LIST with EXACTLY 1 value.
        - art_style: LIST with EXACTLY 3 values if focus="Style", else LIST with EXACTLY 1 value.
        - colors: LIST with EXACTLY 3 values if focus="Color", else LIST with EXACTLY 1 value.
        - caption: Shirt slogan text.
        - logic: Explain the strategy used and why it works.
        - focus: ONE word – "Subject", "Action", "Context", "Mood", "Style", or "Color".
        """
        
        creative_llm, resolved_model = self._get_creative_llm(concept_model)
        print(f"Concept generation model: {resolved_model}")
        structured_llm = creative_llm.with_structured_output(DesignConcepts)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.concepts

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
            baseline[out_key] = str(source.get(src_key, "")).strip()

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
        from pydantic import BaseModel, Field

        class Concept(BaseModel):
            title: str = Field(description="Catchy title for the T-shirt design")
            visual_prompt: str = Field(description="Full prompt for AI image generator")
            subject: List[str] = Field(description="Subject values")
            action: List[str] = Field(description="Action values")
            context: List[str] = Field(description="Context values")
            mood: List[str] = Field(description="Mood values")
            art_style: List[str] = Field(description="Art style values")
            colors: List[str] = Field(description="Color palette values")
            caption: str = Field(description="Text or slogan on the shirt")
            logic: str = Field(description="Business logic or why this design works")
            focus: str = Field(description="Focus field name")

        class SingleConcept(BaseModel):
            concept: Concept

        normalized_fields = self._normalize_field_inputs(field_inputs)
        baseline = self._extract_baseline_fields(description, vision_analysis, normalized_fields)

        # ── Smart classification ──
        changed, locked = self._classify_fields(normalized_fields, vision_analysis)
        num_changed = len(changed)
        use_full_exploration = (num_changed == 0)

        print(f"[iter_concepts] changed={list(changed.keys())} locked={list(locked.keys())} → mode={'full' if use_full_exploration else 'smart'}")

        creative_llm, resolved_model = self._get_creative_llm(concept_model)
        print(f"Streaming concept model: {resolved_model}")
        structured_llm = creative_llm.with_structured_output(SingleConcept)

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

            focus_specs = [
                ("Subject", "subject"),
                ("Action", "action"),
                ("Context", "context"),
                ("Mood", "mood"),
                ("Style", "art_style"),
                ("Color", "colors"),
            ]

            # Yield metadata first
            yield {
                "_meta": True,
                "mode": "full",
                "total": 6,
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
- The title and visual_prompt must use the first value of each field.
- visual_prompt must be detailed and print-on-demand ready.
- focus must be exactly "{focus_name}".
"""
                if user_instruction:
                    prompt += f'\nUser instruction to honor: "{user_instruction}"\n'

                prompt += """
Output JSON object:
{
  "concept": {
    "title": "...",
    "visual_prompt": "...",
    "subject": [...],
    "action": [...],
    "context": [...],
    "mood": [...],
    "art_style": [...],
    "colors": [...],
    "caption": "...",
    "logic": "...",
    "focus": "..."
  }
}
"""
                response = structured_llm.invoke([HumanMessage(content=prompt)])
                concept_dict = response.concept.model_dump()
                concept_dict = self._coerce_concept_shape(concept_dict, focus_name, baseline)
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

            class SmartConcept(BaseModel):
                title: str = Field(description="Catchy title for the T-shirt design")
                visual_prompt: str = Field(description="Full detailed prompt for AI image generator")
                caption: str = Field(description="Text or slogan on the shirt")
                logic: str = Field(description="Why this design works commercially")

            smart_llm = creative_llm.with_structured_output(SmartConcept)

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

Generate a catchy title and a highly detailed visual_prompt for this specific T-shirt concept:

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
- Target 80-150 words for visual_prompt.
- Title should be catchy and commercial.
- Explain briefly why this combo works in 'logic'.
"""
                    if user_instruction:
                        prompt += f'\nUser instruction to honor: "{user_instruction}"\n'

                    response = smart_llm.invoke([HumanMessage(content=prompt)])
                    smart_result = response.model_dump()

                    sub_card = {
                        "sub_label": f"{gi + 1}.{si + 1}",
                        "title": smart_result.get("title", ""),
                        "visual_prompt": smart_result.get("visual_prompt", ""),
                        "caption": smart_result.get("caption", ""),
                        "logic": smart_result.get("logic", ""),
                        **concept_fields,
                    }
                    sub_cards.append(sub_card)

                # Build concept dict compatible with frontend
                # Focus field gets all values from this group's combos
                # Other fields get single locked value
                focus_key = field_key_to_concept.get(primary_field, primary_field)
                concept_dict = {
                    "title": f"{group['group_value']} Variations",
                    "visual_prompt": sub_cards[0]["visual_prompt"] if sub_cards else "",
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
   - Target depth: around 90-160 words (not a short one-liner).
   - Ensure white/clean background and print-on-demand readiness.

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
            subject: str = Field(description="Updated subject based on the instruction, or original if unchanged")
            action: str = Field(description="Updated action based on the instruction, or original if unchanged")
            context: str = Field(description="Updated context based on the instruction, or original if unchanged")
            mood: str = Field(description="Updated mood based on the instruction, or original if unchanged")
            art_style: str = Field(description="Updated art style based on the instruction, or original if unchanged")
            colors: str = Field(description="Updated colors based on the instruction, or original if unchanged")

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
        """
        
        structured_llm = self.creative_llm.with_structured_output(RefinedConcept)
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.model_dump()

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
