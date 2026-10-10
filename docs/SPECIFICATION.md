# PromptManifold Specification (RFC-PM-001)

## Status: Final Standard
**Version**: 0.1.0  
**License**: Apache-2.0  

---

## 1. Scope

This document specifies the canonical protocols, data schemas, validation constraints, and API contracts for the PromptManifold inference-time distribution alignment system.

---

## 2. Terminology

- **Sparse Prompt**: Raw conversational user query consisting of fewer than 15 natural language words (for example, "a teddy bear reading a book").
- **Target Manifold**: The high-dimensional prompt representation distribution upon which a foundation generative model was trained (~80 to 150 words containing explicit lighting, camera, spatial, and stylistic parameters).
- **In-Distribution Proximity Index ($S_{\text{ID}}$)**: A scalar metric in $[0.0, 1.0]$ quantifying the structural alignment between an enhanced prompt and the target model training caption distribution.
- **Semantic Drift Penalty ($\Delta_{\text{drift}}$)**: A scalar in $[0.0, 1.0]$ quantifying the loss or corruption of user intent during projection.

---

## 3. Data Schemas

### 3.1 ManifoldTarget Enum
The foundation model distribution to which the prompt is projected:
- `flux`: Black Forest Labs Flux architecture (natural descriptive prose).
- `sdxl`: Stable Diffusion XL dual-text-encoder format (syntax weighting, comma tokens).
- `sd3`: Stable Diffusion 3 / 3.5 multi-encoder format (T5-XXL hybrid).
- `midjourney`: Midjourney v6+ syntax (flags `--v 6.1`, `--style raw`, `--ar`).
- `dalle3`: DALL-E 3 narrative scene representation.
- `universal_vision`: Universal fallback multimodal representation.

### 3.2 IntentSlots Schema
| Field | Type | Required | Description |
|---|---|---|---|
| `subject` | `string` | Yes | Primary semantic entity or anchor. |
| `subject_attributes` | `list[string]` | No | Morphological, textural, and descriptive adjectives. |
| `environment` | `string` | No | Background setting and spatial context. |
| `lighting` | `string` | No | Directional, volumetric, or ambient illumination schema. |
| `medium` | `string` | Yes | Visual medium classification (e.g., `photography`, `digital_art`). |
| `camera_framing` | `string` | No | Focal length, camera angle, optical depth parameters. |
| `color_palette` | `string` | No | Tonal balance and chromatic harmony. |
| `atmosphere` | `string` | No | Environmental and emotional mood. |
| `explicit_negatives` | `list[string]` | No | Unwanted concepts explicitly negated in raw input. |
| `style_keywords` | `list[string]` | No | Movement or style identifiers (e.g., `cyberpunk`, `baroque`). |
| `preserved_tokens` | `list[string]` | No | Literal trigger tokens requiring exact character preservation. |

### 3.3 EnhancedPromptResult Schema
| Field | Type | Constraint | Description |
|---|---|---|---|
| `raw_input` | `string` | Non-empty | Original sparse user query. |
| `enhanced_prompt` | `string` | Non-empty | In-distribution projected prompt. |
| `negative_prompt` | `string` | Optional | Paired negative steering vector. |
| `target` | `string` | Valid `ManifoldTarget` | Target foundation model. |
| `slots` | `IntentSlots` | Valid object | Semantic slots used during projection. |
| `in_distribution_score` | `float` | $[0.0, 1.0]$ | Estimated in-distribution proximity index. |
| `expansion_ratio` | `float` | $\ge 1.0$ | Length ratio: enhanced tokens / raw tokens. |
| `semantic_drift_score` | `float` | $[0.0, 1.0]$ | Intent deviation penalty. |
| `slot_coverage_score` | `float` | $[0.0, 1.0]$ | Syntactic slot completeness. |
| `latency_ms` | `float` | $\ge 0.0$ | End-to-end processing latency in milliseconds. |
| `adapter_metadata` | `object` | Key-value pairs | Model-specific execution details. |

---

## 4. HTTP API Contract

### 4.1 Health Check
- **Endpoint**: `GET /v1/health`
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "service": "prompt-manifold",
  "version": "0.1.0",
  "engine": "PromptManifoldEngine-v1"
}
```

### 4.2 Synchronous Enhancement
- **Endpoint**: `POST /v1/enhance`
- **Request Body**:
```json
{
  "prompt": "a teddy bear reading a book",
  "target": "flux",
  "detail_level": 3,
  "generate_negative": true
}
```
- **Response**: `200 OK` (returns `EnhancedPromptResult`)

### 4.3 Streaming Projection (Server-Sent Events)
- **Endpoint**: `POST /v1/enhance/stream`
- **Events**:
  - `start`: Initial metadata payload.
  - `token`: Sequential token chunk payload (`{"token": "...", "index": n}`).
  - `complete`: Final telemetry and negative prompt payload.
