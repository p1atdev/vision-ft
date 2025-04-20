import polars as pl
import click

PEOPLE_PATTERN = r"(\d\+?(girl|boy|other)s?|no humans)"


@click.command()
@click.option(
    "--metadata",
    default="hf://datasets/deepghs/danbooru2024-webp-4Mpixel/metadata.parquet",
)
@click.option(
    "--output",
    default="./metadata_styled_images.parquet",
)
@click.option(
    "--has_more_than",
    default=4,
    help="Minimum number of images for an artist to be included.",
)
@click.option(
    "--each_count",
    default=4,
    help="Number of images to select for each artist.",
)
@click.option(
    "--min_count",
    default=2,
    help="Minimum number of images to select for each artist.",
)
@click.option(
    "--total_count",
    default=1000,
    help="Total number of images to select.",
)
def main(
    metadata: str,
    output: str,
    has_more_than: int = 4,
    each_count: int = 2,
    min_count: int = 2,
    total_count: int = 1000,
):
    lf = pl.scan_parquet(metadata)

    lf = (
        lf.select(
            [
                "id",
                "created_at",
                "score",
                "rating",
                "image_width",
                "image_height",
                "tag_string_artist",
                #
                "tag_string_copyright",
                "tag_string_character",
                "tag_string_general",
                "tag_string_meta",
                #
                "parent_id",
            ]
        )
        # tag filtering
        .filter(
            # meta tag filtering
            (
                pl.col("tag_string_meta").str.contains("duplicate").not_()
            )  # just duplicate
            & (
                pl.col("tag_string_meta").str.contains("artist_collaboration").not_()
            )  # not consistent style
            & pl.col("tag_string_meta").str.contains("revision").not_()  # almost same
            & (
                pl.col("tag_string_meta").str.contains("variant_set").not_()
            )  # too similar
            & (
                pl.col("tag_string_meta").str.contains("animated").not_()
            )  # not a static image
            # general tag filtering
            ## unconsistent style
            & pl.col("tag_string_general").str.split(" ").list.contains("meme").not_()
            & pl.col("tag_string_general").str.contains("_challenge").not_()
            & pl.col("tag_string_general").str.contains("comic").not_()
            & pl.col("tag_string_general").str.contains("(style)").not_()
            & pl.col("parent_id").is_null()  # too similar because related
        )
        .drop("parent_id")
        .filter(pl.col("tag_string_artist").is_not_null())
        # no artist collaboration (mixed-style)
        .filter(pl.col("tag_string_artist").str.split(" ").list.len() == 1)
        .with_columns(
            pl.col("tag_string_artist").str.split(" ").list.get(0).alias("artist")
        )
    )
    lf = lf.filter(pl.col("id") < 8000000)

    # split "tag_string_artist" into list
    artist_lf = (
        lf.select("artist")
        .group_by("artist")
        .agg(pl.col("artist").count().alias("count"))
    )

    artist_series = (
        artist_lf.filter((pl.col("count") > has_more_than))
        .select("artist")
        .collect(engine="streaming")
        .to_series()
    )

    latest_ids = (
        lf.filter(pl.col("artist").is_in(artist_series))
        .sort(pl.col("id"), descending=True)
        .group_by("artist")
        .agg(
            [
                (
                    pl.col("tag_string_character").is_unique()
                    | pl.col("tag_string_character").is_null()
                    | (pl.col("tag_string_character").str.len_chars() == 0)
                ).alias("is_unique"),
                pl.col("id"),
                pl.col("tag_string_character"),
                pl.col("tag_string_copyright"),
                pl.col("tag_string_general"),
                pl.col("tag_string_meta"),
            ]
        )
        .explode(pl.all().exclude("artist"))
        .filter(pl.col("is_unique"))
        .drop("is_unique")
        .group_by("artist")
        .agg(pl.all().head(each_count).drop_nulls())
        .filter(pl.col("id").count() >= min_count)
        .explode(pl.all().exclude("artist"))
    )
    artist2id = latest_ids.select("artist").unique().sort("artist").with_row_index()

    post_and_artist = latest_ids.join(
        artist2id.rename({"index": "artist_id"}),
        left_on="artist",
        right_on="artist",
        how="inner",
    ).drop("artist")

    tag_prepared = post_and_artist.with_columns(
        character=pl.col("tag_string_character")
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element() != "").str.replace_all("_", " "),
        ),
        copyright=pl.col("tag_string_copyright")
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element() != "").str.replace_all("_", " "),
        ),
        general=pl.col("tag_string_general")
        .str.split(" ")
        .list.eval(
            pl.element()
            .filter(
                (pl.element() != "") & pl.element().str.contains(PEOPLE_PATTERN).not_()
            )
            .str.replace_all("_", " ")
        ),
        people=pl.col("tag_string_general")
        .str.split(" ")
        .list.eval(
            pl.element()
            .filter((pl.element() != "") & pl.element().str.contains(PEOPLE_PATTERN))
            .str.replace_all("_", " ")
        ),
        meta=pl.col("tag_string_meta")
        .str.split(" ")
        .list.eval(
            pl.element().filter(pl.element() != "").str.replace_all("_", " "),
        ),
    ).select("id", "artist_id", "character", "copyright", "general", "meta", "people")

    artist_to_images = (
        tag_prepared.select("artist_id", "id")
        .group_by("artist_id")
        .agg(
            [
                pl.col("id"),
            ]
        )
    )

    metadata_lf = (
        tag_prepared.join(
            artist_to_images.rename({"id": "another_id"}),
            left_on="artist_id",
            right_on="artist_id",
            how="inner",
        )
        .filter(pl.col("another_id").list.len() >= 2)
        .sort("id", descending=True)
        .head(total_count)
    )

    metadata_lf.sink_parquet(output)


if __name__ == "__main__":
    main()
