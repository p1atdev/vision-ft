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
