"""
Real whole-slide image (WSI) loading via OpenSlide.

This is the concrete implementation of the extensibility point documented
in ARCHITECTURE.md: everything downstream (tissue_detection.py,
tiling_inference.py, quantification.py) operates on a plain HxWx3 uint8
RGB numpy array and doesn't care where it came from. This module's only
job is to produce that array from a real .svs/.tiff slide file instead of
the synthetic mosaic.

Requires the OpenSlide system library plus the openslide-python bindings:
    apt-get install -y openslide-tools     (or libopenslide-dev on newer distros)
    pip install openslide-python

WSI files are multi-resolution ("pyramidal") - level 0 is full resolution
(often 40,000+ pixels per side), with each subsequent level downsampled
2x. Reading a whole slide at level 0 into memory is impractical (and
exactly the problem this whole project's tiling pipeline exists to avoid).
For this project's scope, we read a single region at a chosen level -
demonstrating that the pipeline genuinely works against a real WSI file
format, without requiring a full slide-scanning strategy on top of it.
"""

import numpy as np


def load_wsi_region(path, level=0, location=(0, 0), size=None):
    """
    path: path to a .svs/.tiff/other OpenSlide-supported file
    level: pyramid level to read from (0 = highest resolution available)
    location: (x, y) top-left corner, in level-0 coordinates (OpenSlide's
              convention - always specified relative to level 0, regardless
              of which level you're actually reading from)
    size: (width, height) to read, in the target level's own pixel
          coordinates. If None, reads the entire selected level - only
          safe for small test slides or a already-small level, never for
          level 0 of a real production slide.

    Returns: HxWx3 uint8 RGB numpy array - the same interface every other
             image-source in this project already produces (see mosaic.py
             for the synthetic equivalent).
    """
    try:
        import openslide
    except ImportError as e:
        raise ImportError(
            "openslide-python is not installed. Run: "
            "apt-get install -y openslide-tools && pip install openslide-python"
        ) from e

    slide = openslide.OpenSlide(path)

    if size is None:
        size = slide.level_dimensions[level]

    # OpenSlide returns RGBA PIL Image; drop alpha, convert to numpy RGB
    region = slide.read_region(location, level, size)
    region_rgb = region.convert("RGB")
    array = np.array(region_rgb)

    slide.close()
    return array


def describe_slide(path):
    """Print basic slide metadata - useful for picking a sensible level/
    region before attempting to read one, since level 0 of a real slide
    can be tens of thousands of pixels per side."""
    try:
        import openslide
    except ImportError as e:
        raise ImportError(
            "openslide-python is not installed. Run: "
            "apt-get install -y openslide-tools && pip install openslide-python"
        ) from e

    slide = openslide.OpenSlide(path)
    print(f"Slide: {path}")
    print(f"Levels: {slide.level_count}")
    for i, (dims, downsample) in enumerate(zip(slide.level_dimensions, slide.level_downsamples)):
        print(f"  Level {i}: {dims[0]}x{dims[1]} px (downsample {downsample:.1f}x)")
    slide.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True)
    args = parser.parse_args()
    describe_slide(args.slide)