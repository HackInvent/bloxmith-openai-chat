# OPENAI CHAT artwork

This image is a concept illustration of the `openai_chat` BloxSmith block.
The implemented function is: Calls an OpenAI-compatible Responses API and returns assistant text and response JSON.

- `cover.png`: selected original square PNG, copied byte for byte.
- `thumbnail.webp`: 320 × 320 WebP export for the README and catalog cards.
- Accent: deep blue.
- Geometry reference: the zcode pilot cover. It is a visual reference only;
  its block-specific content was replaced.
- Creation: built-in image generation tool, 2026-09-28.
- License: [Apache-2.0](../LICENSE), like the rest of this block repository.

The image is not a Studio screenshot or a description of additional controls.
The artwork is documentation; its presence does not register it automatically
with Studio or the website. The block version and runtime contract are unchanged.

Export command (ImageMagick):

```sh
convert media/cover.png -thumbnail 320x320 -strip -quality 88 -define webp:method=6 media/thumbnail.webp
```

## Refinement prompt for the selected cover

The first render was edited with the following exact instruction to remove
provider marks while preserving the block's function and illustration style.

```text
Edit only the supplied image. Remove the OpenAI knot/swirl logo from the reply speech bubble. Replace it with a generic assistant sparkle (simple four-point star) or plain message lines. Keep the title exactly OPENAI CHAT, input-to-reply layout, cube geometry, color, size, margins and shadows. No official provider logos. Preserve the same polished industrial pixel-art style. Do not introduce any other text, icons, objects or modifications.
```

## Exact generation prompt

```text
Use case: stylized-concept.
Asset type: one square BloxSmith block cover illustration, to be read as a small catalog thumbnail.
Input image: the zcode pilot cover is a STYLE AND GEOMETRY REFERENCE ONLY. Use its centered graphite and steel industrial cube, exact front/top/right three-quarter perspective, reinforced corners, neat vents and ports, plain near-white background, subtle grounded shadow, upper-left light, scale and generous margins. Replace every subject-specific icon, display and accent from the reference; create a new original cover for openai_chat.
Primary request: visualize the real software block role accurately: Calls an OpenAI-compatible Responses API and returns assistant text and response JSON.
Main roof symbol: one large conversation-response symbol.
Front panel: a prominent upper title plate, and directly below it one user instruction/message card becomes one assistant reply card, with a small structured-result tile. Keep the concept legible at 320 pixels. Only a few large functional details; no microtext.
Color palette: one dominant restrained deep blue accent, otherwise charcoal housing and brushed steel structural edges. The palette identifies a family but the symbols must distinguish the block without color.
Style/medium: polished, crisp industrial pixel-art product illustration with stepped edges and deliberate pixel clusters. Restrained glow, tactile surfaces, no photorealistic rendering.
Composition: one isolated complete near-cubical module on a 1:1 square canvas, about 84 percent of frame, no crop, no extra scene or props; match the pilot's geometry and consistent clean framing.
Text (verbatim): "OPENAI CHAT" exactly once in bold white uppercase pixel lettering on the front title plate. This is the ONLY writing in the picture. Spell it exactly; never add a subtitle, version, badge, copied text, fake small labels, watermark or slogan. The README carries the full technical name.
Accuracy constraints: coding terminal, microphone audio, official OpenAI logo. The picture should communicate function through simple symbols and one primary flow. Do not copy the pilot's icon, title, faders, waveform, code screen or scanning page unless actually relevant. No cats, people, desks, mascots, official third-party logos, detached modules or floating objects.
```
