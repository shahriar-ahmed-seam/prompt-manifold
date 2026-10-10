"""Syntactic and semantic intent extraction from sparse user prompts."""

import re
from typing import List, Optional, Tuple

from prompt_manifold.schemas import IntentSlots, MediumType


class IntentExtractor:
    """Decomposes sparse user queries into structured semantic slots."""

    MEDIUM_PATTERNS = [
        (r"\b(photo|photograph|photorealistic|dslr|film|portrait)\b", MediumType.PHOTOGRAPHY),
        (r"\b(cinematic|movie scene|film still|anamorphic)\b", MediumType.CINEMATIC),
        (r"\b(digital art|illustration|matte painting|artstation)\b", MediumType.DIGITAL_ART),
        (r"\b(3d|render|octane|unreal engine|cgi)\b", MediumType.THREE_D_RENDER),
        (r"\b(anime|manga|studio ghibli|makoto shinkai)\b", MediumType.ANIME_MANGA),
        (r"\b(oil painting|canvas|impasto|watercolor)\b", MediumType.OIL_PAINTING),
        (r"\b(vector|flat design|icon|svg|minimalist graphic)\b", MediumType.VECTOR_FLAT),
        (r"\b(concept art|sketch|speedpaint)\b", MediumType.CONCEPT_ART),
    ]

    LIGHTING_PATTERNS = [
        (
            r"\b(golden hour|sunset light|warm light)\b",
            "warm golden hour directional sunlight",
        ),
        (
            r"\b(neon|cyberpunk|fluorescent)\b",
            "vibrant volumetric neon lighting with soft cyan and magenta reflections",
        ),
        (
            r"\b(studio light|softbox|rim light)\b",
            "balanced three-point studio lighting with subtle rim separation",
        ),
        (
            r"\b(dark|moody|chiaroscuro|dramatic light)\b",
            "deep chiaroscuro lighting with dramatic volumetric shadows",
        ),
        (
            r"\b(overcast|diffused|soft light)\b",
            "softly diffused overcast ambient daylight",
        ),
        (
            r"\b(natural light|window light)\b",
            "soft directional morning sunlight streaming through a nearby window",
        ),
    ]

    CAMERA_PATTERNS = [
        (
            r"\b(macro|extreme close-up|close up)\b",
            "macro close-up shot, shallow depth of field, sharp focal plane",
        ),
        (
            r"\b(wide angle|panoramic|landscape view)\b",
            "wide-angle panoramic composition, deep depth of field, expansive perspective",
        ),
        (
            r"\b(portrait|bokeh|telephoto)\b",
            "medium portrait framing, 85mm f/1.4 lens, creamy optical bokeh",
        ),
        (
            r"\b(drone|aerial|birds eye)\b",
            "high-angle aerial drone photography, geometric top-down composition",
        ),
        (
            r"\b(isometric)\b",
            "orthographic isometric perspective, clean geometric alignment",
        ),
    ]

    NEGATION_REGEX = re.compile(
        r"\b(?:no|without|avoid|exclude|never)\s+([a-zA-Z0-9_\-\s]+?)(?:,|\.|$)",
        re.IGNORECASE,
    )

    TRIGGER_REGEX = re.compile(r'"([^"]+)"|\'([^\']+)\'')

    def extract(self, raw_input: str) -> IntentSlots:
        """Parse raw query into deterministic intent slots."""
        cleaned = self._clean_string(raw_input)
        if not cleaned:
            return IntentSlots(
                subject="conceptual subject",
                subject_attributes=["finely detailed", "balanced composition"],
                environment="neutral atmospheric studio background",
                lighting="soft ambient directional lighting",
                medium=MediumType.PHOTOGRAPHY,
                camera_framing="centered medium shot, sharp focus",
            )

        # 1. Extract preserved trigger tokens (quoted phrases)
        preserved_tokens = self._extract_preserved_tokens(cleaned)

        # 2. Extract explicit negation requests
        explicit_negatives, text_stripped_negatives = self._extract_negations(cleaned)

        # 3. Detect medium
        medium = self._detect_medium(text_stripped_negatives)

        # 4. Detect lighting hints
        lighting = self._detect_lighting(text_stripped_negatives)

        # 5. Detect camera framing hints
        camera_framing = self._detect_camera(text_stripped_negatives)

        # 6. Extract core subject anchor and attributes
        subject, attributes, style_keywords = self._decompose_subject_and_style(
            text_stripped_negatives
        )

        return IntentSlots(
            subject=subject,
            subject_attributes=attributes,
            environment=None,  # Populated downstream by manifold engine
            lighting=lighting,
            medium=medium,
            camera_framing=camera_framing,
            color_palette=None,
            atmosphere=None,
            explicit_negatives=explicit_negatives,
            style_keywords=style_keywords,
            preserved_tokens=preserved_tokens,
        )

    def _clean_string(self, text: str) -> str:
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _extract_preserved_tokens(self, text: str) -> List[str]:
        tokens = []
        for match in self.TRIGGER_REGEX.finditer(text):
            tok = match.group(1) or match.group(2)
            if tok and tok.strip():
                tokens.append(tok.strip())
        return tokens

    def _extract_negations(self, text: str) -> Tuple[List[str], str]:
        negatives: List[str] = []
        matches = self.NEGATION_REGEX.findall(text)
        for m in matches:
            phrase = m.strip()
            if phrase and len(phrase) < 40:
                negatives.append(phrase)

        cleaned = self.NEGATION_REGEX.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return negatives, cleaned

    def _detect_medium(self, text: str) -> MediumType:
        lower = text.lower()
        for pattern, med in self.MEDIUM_PATTERNS:
            if re.search(pattern, lower):
                return med
        return MediumType.PHOTOGRAPHY

    def _detect_lighting(self, text: str) -> Optional[str]:
        lower = text.lower()
        for pattern, desc in self.LIGHTING_PATTERNS:
            if re.search(pattern, lower):
                return desc
        return None

    def _detect_camera(self, text: str) -> Optional[str]:
        lower = text.lower()
        for pattern, desc in self.CAMERA_PATTERNS:
            if re.search(pattern, lower):
                return desc
        return None

    def _decompose_subject_and_style(self, text: str) -> Tuple[str, List[str], List[str]]:
        clean = text.replace('"', "").replace("'", "").strip()

        known_styles = {
            "cyberpunk", "steampunk", "minimalist", "surrealist", "baroque",
            "renaissance", "gothic", "art nouveau", "retro", "vintage",
            "vaporwave", "futuristic", "hyperrealistic", "noir"
        }
        words = clean.split()
        found_styles: List[str] = []
        remaining_words: List[str] = []

        for w in words:
            w_lower = w.lower().strip(".,;:!")
            if w_lower in known_styles:
                found_styles.append(w_lower)
            else:
                remaining_words.append(w)

        core_phrase = " ".join(remaining_words).strip(".,;: ")
        if not core_phrase:
            core_phrase = clean or "subject"

        attributes = self._derive_attributes_for_subject(core_phrase)
        return core_phrase, attributes, found_styles

    def _derive_attributes_for_subject(self, subject: str) -> List[str]:
        sub_lower = subject.lower()
        attributes = []
        if "bear" in sub_lower or "plush" in sub_lower or "toy" in sub_lower:
            attributes.extend(
                ["intricate plush fur texture", "visible artisan stitching", "warm glassy eyes"]
            )
        elif any(k in sub_lower for k in ("portrait", "person", "woman", "man")):
            attributes.extend(
                [
                    "natural skin pores and texture",
                    "subtle micro-expressions",
                    "lifelike specular catchlights",
                ]
            )
        elif any(k in sub_lower for k in ("car", "vehicle", "cyber")):
            attributes.extend(
                [
                    "aerodynamic metallic contours",
                    "glossy clearcoat reflections",
                    "fine mechanical precision",
                ]
            )
        elif any(k in sub_lower for k in ("shoe", "sneaker")):
            attributes.extend(
                [
                    "breathable engineered mesh weave",
                    "cushioned ergonomic midsole",
                    "crisp stitching detail",
                ]
            )
        elif any(k in sub_lower for k in ("city", "street", "architecture")):
            attributes.extend(
                [
                    "intricate architectural masonry",
                    "wet asphalt ground reflections",
                    "layered urban depth",
                ]
            )
        else:
            attributes.extend(
                [
                    "tactile surface micro-details",
                    "three-dimensional volume",
                    "authentic material physical properties",
                ]
            )
        return attributes
