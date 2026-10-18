# GEMS-3D project page

The project page is a buildless static website in this directory, served at
https://bilybill.github.io/GEMS-3D/ through GitHub Pages. It has no external
JavaScript, font, or CSS dependencies.

Preview from the repository root:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open http://localhost:8000. The main files are `index.html`, `styles.css`, and
`site.js`. The original scientific teaser and the favicon live in `assets/`.

## Content and release links

Data counts, split sizes, and payload sizes come from these public release cards:

- [Full](https://huggingface.co/datasets/anomy4pub/GEMS-3D)
- [Standard](https://huggingface.co/datasets/anomy4pub/GEMS-3D-Standard)
- [Tiny](https://huggingface.co/datasets/anomy4pub/GEMS-3D-Tiny)

Recheck the cards when changing release numbers. Benchmark code and schema links
point to the curated public [GMESDataset repository](https://github.com/Bilybill/GMESDataset).
The page does not link to private model-generation or numerical-solver workflows.

The BibTeX is a resource reference to the dataset, not an accompanying paper.
Replace or extend it only when verified paper metadata becomes available. Do not
add placeholder authors, conference badges, paper links, or performance numbers.

The design takes inspiration from the clear resource navigation and visual
research explanations on [NeRF](https://www.matthewtancik.com/nerf) and
[Nerfies](https://nerfies.github.io/). The implementation is original.

## Verification

Check desktop and mobile layouts, keyboard tab navigation, mobile-menu closing,
clipboard success and fallback, relative assets, navigation anchors, structured
metadata, and the external resource links. Respect reduced-motion preferences.

Publish only the intended `docs/` changes. Keep unrelated research edits out of
the website commit, and preserve the existing GitHub Pages publishing source.
