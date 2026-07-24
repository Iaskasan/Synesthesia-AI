from src.generation.prompt_builder import build_prompt


def test_prompt_includes_audio_and_user_choices():
    prompt = build_prompt(
        "energetic",
        tempo_bpm=140,
        style="watercolor",
        subject="a mountain city",
    )

    assert "a mountain city" in prompt
    assert "watercolor" in prompt
    assert "fast visual rhythm" in prompt
    assert "vibrant colors" in prompt

