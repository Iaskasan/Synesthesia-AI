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


def test_prompt_supports_multiple_moods_and_control_levels():
    prompt = build_prompt(
        ["dream", "sad"], abstraction=0.2, music_influence=0.4
    )
    assert "dream atmosphere" in prompt
    assert "sad atmosphere" in prompt
    assert "recognizable subjects" in prompt
    assert "gently informed" in prompt


def test_prompt_rejects_out_of_range_controls():
    import pytest

    with pytest.raises(ValueError, match="between 0 and 1"):
        build_prompt("calm", abstraction=1.1)
