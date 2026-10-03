# Project Notes: Tanzil Quran Text Usage

- Source files in `data/tanzil/` (`quran-uthmani.txt`, `quran-simple-clean.txt`, etc.) are downloaded directly from Tanzil.net and kept verbatim and unmodified.
- Per Tanzil's Terms of Use, changing the source text is not allowed.
- All downstream transformations and normalizations (e.g. splitting Basmalah, stripping non-vocab symbols, creating derived datasets for OCR/HTR training) are strictly performed in derived files under `data/derived/`.
- Attribution notice: "Tanzil Quran Text - Copyright (C) 2007-2021 Tanzil Project - CC BY 3.0 (http://tanzil.net)".
