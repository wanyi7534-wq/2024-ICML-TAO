"""Create the four-by-four TAO dehazing reproduction comparison figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_IDS = ("1381.jpg", "3146.jpg", "4561.jpg", "5920.jpg")
COLUMNS = (
    ("Hazy Input", "test_samples/HSTS_256x256/synthetic"),
    ("Our TAO", "results_batch_ours"),
    ("Author TAO", "test_samples/HSTS_256x256/results"),
    ("Ground Truth", "test_samples/HSTS_256x256/original"),
)
OUTPUT_NAME = "tao_dehazing_comparison_4x4.png"

CELL_SIZE = 256
LEFT_MARGIN = 126
RIGHT_MARGIN = 30
TOP_MARGIN = 28
BOTTOM_MARGIN = 30
HEADER_HEIGHT = 52
COLUMN_GAP = 18
ROW_GAP = 28


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_without_stretching(image: Image.Image) -> Image.Image:
    """Fit an image in a square cell while preserving its aspect ratio."""
    contained = image.copy()
    contained.thumbnail((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
    cell = Image.new("RGB", (CELL_SIZE, CELL_SIZE), "white")
    x = (CELL_SIZE - contained.width) // 2
    y = (CELL_SIZE - contained.height) // 2
    cell.paste(contained, (x, y))
    return cell


def centered_text_x(draw: ImageDraw.ImageDraw, text: str, center: int, font) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return center - (box[2] - box[0]) // 2


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = Path(__file__).resolve().parent / OUTPUT_NAME
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    inputs = {
        (title, image_id): repo_root / relative_dir / image_id
        for title, relative_dir in COLUMNS
        for image_id in IMAGE_IDS
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    width = (
        LEFT_MARGIN
        + len(COLUMNS) * CELL_SIZE
        + (len(COLUMNS) - 1) * COLUMN_GAP
        + RIGHT_MARGIN
    )
    height = (
        TOP_MARGIN
        + HEADER_HEIGHT
        + len(IMAGE_IDS) * CELL_SIZE
        + (len(IMAGE_IDS) - 1) * ROW_GAP
        + BOTTOM_MARGIN
    )
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(25, bold=True)
    row_font = load_font(23, bold=True)
    border_color = (185, 185, 185)

    for column_index, (title, _) in enumerate(COLUMNS):
        x = LEFT_MARGIN + column_index * (CELL_SIZE + COLUMN_GAP)
        title_x = centered_text_x(draw, title, x + CELL_SIZE // 2, header_font)
        draw.text((title_x, TOP_MARGIN), title, fill="black", font=header_font)

    for row_index, image_id in enumerate(IMAGE_IDS):
        y = TOP_MARGIN + HEADER_HEIGHT + row_index * (CELL_SIZE + ROW_GAP)
        label = Path(image_id).stem
        label_box = draw.textbbox((0, 0), label, font=row_font)
        label_height = label_box[3] - label_box[1]
        draw.text(
            (24, y + (CELL_SIZE - label_height) // 2),
            label,
            fill="black",
            font=row_font,
        )

        for column_index, (title, _) in enumerate(COLUMNS):
            x = LEFT_MARGIN + column_index * (CELL_SIZE + COLUMN_GAP)
            with Image.open(inputs[(title, image_id)]) as source:
                cell = fit_without_stretching(source.convert("RGB"))
            canvas.paste(cell, (x, y))
            draw.rectangle(
                (x, y, x + CELL_SIZE - 1, y + CELL_SIZE - 1),
                outline=border_color,
                width=1,
            )

    canvas.save(output_path, format="PNG", optimize=True)
    print(output_path)


if __name__ == "__main__":
    main()
