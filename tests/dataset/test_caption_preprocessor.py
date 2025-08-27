from src.dataset.caption import (
    CaptionPrefix,
    CaptionSuffix,
    CaptionRandomPrefix,
    CaptionRandomSuffix,
    CaptionDrop,
    CaptionTagDrop,
    CaptionShuffle,
    CaptionShuffleInGroup,
    CaptionPassthrough,
)


def test_caption_prefix():
    processor = CaptionPrefix(prefix="sks style, ")
    input_caption = "1girl, solo, looking at viewer"

    output_caption = processor.process(input_caption)
    assert output_caption == "sks style, 1girl, solo, looking at viewer"


def test_caption_suffix():
    processor = CaptionSuffix(suffix=", masterpiece")
    input_caption = "1girl, solo, looking at viewer"

    output_caption = processor.process(input_caption)
    assert output_caption == "1girl, solo, looking at viewer, masterpiece"


def test_caption_random_prefix():
    processor = CaptionRandomPrefix(
        prefix=[
            "sks style, ",
            "original, ",
            "",
        ]
    )
    input_caption = "1girl, solo, looking at viewer"

    results = [processor.process(input_caption) for _ in range(100)]
    assert "sks style, 1girl, solo, looking at viewer" in results
    assert "original, 1girl, solo, looking at viewer" in results
    assert "1girl, solo, looking at viewer" in results


def test_caption_random_suffix():
    processor = CaptionRandomSuffix(
        suffix=[
            ", masterpiece",
            ", best quality",
            "",
        ]
    )
    input_caption = "1girl, solo, looking at viewer"

    results = [processor.process(input_caption) for _ in range(100)]
    assert "1girl, solo, looking at viewer, masterpiece" in results
    assert "1girl, solo, looking at viewer, best quality" in results
    assert "1girl, solo, looking at viewer" in results


def test_caption_drop():
    processor = CaptionDrop(
        drop_rate=0.0,
    )
    input_caption = "1girl, solo, looking at viewer"

    output_caption = processor.process(input_caption)
    assert output_caption == input_caption

    processor = CaptionDrop(
        drop_rate=1.0,
    )
    output_caption = processor.process(input_caption)
    assert output_caption == ""


def test_caption_tag_drop():
    processor = CaptionTagDrop(
        drop_rate=0.0,
        separator=",",
    )
    input_caption = "1girl, solo, looking at viewer"

    output_caption = processor.process(input_caption)
    assert output_caption == input_caption

    processor = CaptionTagDrop(
        drop_rate=0.5,
        separator=",",
    )
    results = [processor.process(input_caption) for _ in range(1000)]
    assert "1girl, looking at viewer" in results

    processor = CaptionTagDrop(
        drop_rate=1.0,
        separator=",",
    )
    output_caption = processor.process(input_caption)
    assert output_caption == ""


def test_caption_shuffle():
    processor = CaptionShuffle(
        split_separator=",",
        trim=True,
        concat_separator=", ",
    )
    input_caption = "1girl,  solo,          looking at viewer"

    results = [processor.process(input_caption) for _ in range(1000)]

    assert "looking at viewer, solo, 1girl" in results
    assert "1girl, solo, looking at viewer" in results


def test_caption_shuffle_in_group():
    processor = CaptionShuffleInGroup(
        group_separator="|||",
        split_separator=",",
        trim=True,
        concat_separator=", ",
    )
    input_caption = "1girl, solo, looking at viewer|||blue hair, cat ears"

    results = [processor.process(input_caption) for _ in range(1000)]

    assert "1girl, solo, looking at viewer, blue hair, cat ears" in results
    assert "looking at viewer, solo, 1girl, cat ears, blue hair" in results


def test_caption_passthrough():
    processor = CaptionPassthrough()
    input_caption = "1girl, solo, looking at viewer"

    output_caption = processor.process(input_caption)
    assert output_caption == input_caption
