import polars as pl
from pathlib import Path
import click
from cheesechaser.datapool import (
    Danbooru2024WebpDataPool,
)


@click.command()
@click.option(
    "--metadata",
    type=str,
)
@click.option(
    "--images_dir",
    type=str,
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
)
def main(
    metadata: str,
    images_dir: str,
    debug: bool = False,
):
    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(metadata)

    # get image ids
    ids = (
        lf.select("another_id")
        .explode("another_id")
        .unique()
        .filter(
            # filter out already downloaded
            pl.col("another_id")
            .map_elements(
                lambda x: (images_path / f"{x}.webp").exists(),
                return_dtype=pl.Boolean,
                strategy="threading",
            )
            .not_()
        )
        .collect()
        .to_dict()["another_id"]
    )

    if debug:
        ids = ids[:20]
    print(f"Found {len(ids)} posts to download.")

    # filter out already downloaded
    print(f"Downloading {len(ids)} posts...")

    pool = Danbooru2024WebpDataPool()
    pool.batch_download_to_directory(
        resource_ids=ids,
        dst_dir=str(images_dir),
    )


if __name__ == "__main__":
    main()
