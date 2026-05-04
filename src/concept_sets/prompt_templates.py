from __future__ import annotations

# Shared template bank for direct / neighbor / control prompts.
# Keep these structurally aligned to reduce prompt-family confounds.
CONCEPT_PROMPT_TEMPLATES = [
    "a photo of a {concept} in a natural setting",
    "a close-up of a {concept} in soft daylight",
    "a realistic {concept} standing outdoors",
    "a {concept} near water at sunset",
    "a {concept} moving through its environment",
    "a detailed image of a {concept}",
    "a high-resolution photograph of a {concept} in the wild",
    "a {concept} under warm afternoon light",
    "a documentary-style photo of a {concept}",
    "a sharply focused {concept} with a blurred background",
]


# Template bank for indirect prompts generated from context words.
# The generator picks one or more context terms and composes them naturally.
INDIRECT_PROMPT_TEMPLATES = [
    "a cinematic scene featuring {w1} and {w2} at sunrise",
    "a realistic outdoor moment with {w1}, {w2}, and {w3}",
    "a candid photo of {w1} beside {w2}",
    "a detailed scene showing {w1} near {w2} in soft light",
    "an action shot with {w1} moving through {w2}",
    "a naturalistic image of {w1} with {w2} in the background",
    "a warm daylight scene centered on {w1} and {w2}",
    "a documentary-style moment with {w1}, {w2}, and {w3}",
]


# Words that often introduce text-heavy confounds in generated images.
TEXT_HEAVY_CUES = {
    "poster",
    "logo",
    "advertisement",
    "advertising",
    "magazine cover",
    "book cover",
    "subtitle",
    "subtitles",
    "text overlay",
    "watermark",
    "sign",
    "signboard",
    "banner",
    "scoreboard",
    "chalkboard",
    "speech bubble",
    "caption text",
}

