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

## Manuscript figures and desktop inspection

The research-figure section includes three original figure assets supplied with
the NeurIPS manuscript, with figure-specific captions:

| Website asset                | Manuscript source        | Content                                              |
| ---------------------------- | ------------------------ | ---------------------------------------------------- |
| `paper/aligned-samples.pdf`  | `Fig_family_gallery.pdf` | Nine families, aligned properties and observations   |
| `paper/dataset-coverage.pdf` | `Fig2.pdf`               | Family/background coverage and property fingerprints |
| `paper/joint-inversion.pdf`  | `jointInvArch.pdf`       | Joint-inversion baseline architecture                |

PDFs are preserved unchanged. Lossless WebP derivatives retain the entire figure,
including original labels, without cropping, recoloring, or redrawing data.
The page uses 1200-pixel previews; 3000-pixel figures load when the image viewer
opens. The complete manuscript and model-generation implementation are not part
of these website assets.

The native dialog supports fit-to-window, original pixel size, zoom buttons,
plus/minus keys, Escape, and focus restoration. Modified link clicks still open
the image directly. Section navigation indicates the current reading location;
the reading-progress indicator and return-to-top link update on scrolling.

The CSS and JavaScript URLs include a release query version. Bump it when
changing those assets so returning visitors receive a coherent new version.
