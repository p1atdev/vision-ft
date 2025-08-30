def _num_object(num: int, noun: str) -> str:
    # 1girl, 2girls, 3girls, ..., 6+girls
    return f"{num}{'+' if num == 6 else ''}{noun}{'s' if num > 1 else ''}"


PEOPLE_TAGS = [
    *[
        _num_object(i, "girl")
        for i in range(1, 7)  # 1~6
    ],
    *[
        _num_object(i, "boy")
        for i in range(1, 7)  # 1~6
    ],
    *[
        _num_object(i, "other")
        for i in range(1, 7)  # 1~6
    ],
]


def format_general_character_tags(
    general: list[str],
    character: list[str],
    rating: str,
    separator: str = ", ",
    group_separator: str = "|||",
):
    people_tags = []
    general_tags = []

    for tag in general:
        if tag in PEOPLE_TAGS:
            people_tags.append(tag)
        else:
            general_tags.append(tag)

    # Animagine-like
    rating_tags = []
    if rating in ["explicit", "e", "questionable", "q"]:
        rating_tags.append("nsfw")

        if rating in ["explicit", "e"]:
            rating_tags.append("explicit")
    else:
        rating_tags.append("safe")

    return group_separator.join(
        [
            part
            for part in [
                separator.join(people_tags),
                separator.join(character),
                separator.join(general_tags),
            ]
            if part.strip() != ""  # skip empty parts
        ]
    )


KAOMOJI = [
    ">_<",
    ">_o",
    "0_0",
    "o_o",
    "3_3",
    "6_9",
    "@_@",
    "u_u",
    "x_x",
    "^_^",
    "|_|",
    "=_=",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    # ↓ deprecated
    "||_||",
    "(o)_(o)",
]


def replace_underscore(tag: str) -> str:
    if tag in KAOMOJI:
        return tag

    return tag.replace("_", " ")


def map_replace_underscore(tags: list[str]) -> list[str]:
    return [replace_underscore(tag) for tag in tags]
