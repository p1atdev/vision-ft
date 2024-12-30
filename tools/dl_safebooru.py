import os
import click
from pathlib import Path
import json

from cheesechaser.datapool import Danbooru2024WebpDataPool
from cheesechaser.query import DanbooruIdQuery
from cheesechaser.pipe import SimpleImagePipe, PipeItem


@click.command()
@click.option("--output", type=str, required=True)
@click.option("--limit", type=int, default=1000)
@click.option("--start-date", type=str, default="2021-01-01")
@click.option("--end-date", type=str, default="2024-8-31")
def main(
    output: str,
    limit: int,
    start_date: str,
    end_date: str,
):
    output_dir = Path(output)

    pool = Danbooru2024WebpDataPool()
    post_ids = DanbooruIdQuery(
        [
            "-duplicate",
            "score:>4",
            "filetype:png,jpg,webp",
            "rating:g",
            f"date:{start_date}..{end_date}",
        ],
        username=os.getenv("DANBOORU_USERNAME"),
        api_key=os.getenv("DANBOORU_API_KEY"),
    )
    pipe = SimpleImagePipe(pool)

    posts = {}
    for post in post_ids._iter_items():
        posts[post["id"]] = post

        if len(posts) >= limit:
            break

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    with pipe.batch_retrieve(posts.keys()) as session:
        for i, item in enumerate(session):
            item.data.save(output_dir / "images" / f"{item.id}.webp")
            with open(output_dir / "images" / f"{item.id}.json", "w") as f:
                json.dump(posts[item.id], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
