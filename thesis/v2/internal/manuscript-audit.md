# OARS-GS bilingual manuscript Humanizer audit

## Scope

- Base commit: `65b9e73793afa5d182ef7a74e56f2ef84d57885b`
- Reviewed files: `thesis/v2/manuscript-en.md` and `thesis/v2/manuscript-zh.md`
- Mode: Humanizer file mode; only final prose was written to the manuscript files.
- Protected material: title and headings, citation order, display formulas and tags, algorithm block, inline code, commands and paths, numeric/percentage/unit tokens, dataset and strategy names, figure/table/equation/section numbers, anchors, local link targets, Markdown tables, and image captions/links.

## Snapshot method and exact extraction rules

Both manuscripts were read as UTF-8. Every category below was extracted as an ordered list. Before/after verification used exact list equality, not set or multiset equality. Each category hash is SHA-256 over compact UTF-8 JSON with `ensure_ascii=False`. The combined frozen snapshot hash is SHA-256 over compact, key-sorted JSON mapping each manuscript path to all category lists.

~~~text
citations_first_order:
  @[A-Za-z0-9_:.+-]+

display_formula_blocks_and_tags:
  (?ms)^\\\[[ \t]*\n.*?^\\\][ \t]*$

code_blocks_and_algorithm:
  (?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=fence)[ \t]*$

inline_code:
  (?<!`)`[^`\n]+`(?!`)

commands_and_paths:
  (?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.-]+)|[A-Za-z_][A-Za-z0-9_.]*\(\)

numeric_percentage_tokens:
  [+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?%?

unit_tokens:
  (?<![A-Za-z])(?:dB|cm|mm|km|m|ms|s|rad|degrees?|degree|GiB|MiB|GB|MB|Hz|fps|pixels?)(?![A-Za-z])

dataset_and_strategy_names:
  GI-KF\+Pyramid|OARS-GS|LoopSplat|TUM RGB-D|FMDataset|Replica|Baseline|GI-KF|Pyramid|IMU|ALL|Azure Kinect|SplaTAM|Photo-SLAM|ORB-SLAM3|TUM|FM|A[1-5]|R[1-8]|C[1-3]

figure_table_equation_section_numbers:
  (?:Figure|Table|Algorithm|Eq\.|Equation|Section|图|表|算法|式|第)\s*\d+(?:\.\d+)?

anchors:
  \{#[^}\n]+\}

local_link_targets:
  target capture from !?\[[^\]\n]*\]\(([^)\n]+)\), excluding ^[a-z]+://

headings_and_title:
  (?m)^#{1,6} .+$

markdown_tables:
  (?m)(?:^\|.*\|[ \t]*\n?)+

image_captions_links:
  (?m)^!\[[^\n]+$
~~~

The local-link existence check removed any fragment after `#`, resolved each remaining target against the manuscript's parent directory, and required `Path.exists()`.

## Snapshot hashes and counts

### Whole files and text counts

| File | State | SHA-256 | Lines | Whitespace words | Unicode characters | Han characters | UTF-8 bytes |
|---|---|---|---:|---:|---:|---:|---:|
| `manuscript-en.md` | Before | `f512dcd62a95ed957518211132b2cbf53fc34bc8f4b1c85fadbe621b524434da` | 473 | 6033 | 48796 | 0 | 48812 |
| `manuscript-en.md` | After | `2f2256455961354b14d7c8c5aa8d6fc3bd7147dd60dacdcd317675a2a9d98f06` | 473 | 5987 | 48484 | 0 | 48500 |
| `manuscript-zh.md` | Before | `f40188e06d12facd50c84ceb7746a5d8952408453dbba5570bd8abbbc2b47f9c` | 473 | 1737 | 23992 | 8346 | 42198 |
| `manuscript-zh.md` | After | `2aee1d9c4e32763aa340b111695c1291849dd6d1c081b4b6ca787270a805747e` | 473 | 1734 | 23918 | 8289 | 41986 |

### Frozen ordered-list categories

All hashes in this table are identical before and after. Counts are ordered-list lengths.

| Category | English count | English SHA-256 | Chinese count | Chinese SHA-256 |
|---|---:|---|---:|---|
| Citations, first order | 83 | `4b63ea173a82c1f56e18923ce4f5628658fefd0f6e3a5b83e96a638d70f563bc` | 83 | `4b63ea173a82c1f56e18923ce4f5628658fefd0f6e3a5b83e96a638d70f563bc` |
| Display formula blocks and tags | 20 | `c8f411a4f51100001f9b04de80a272d27a1d83df704ce8439181073fa0e2f526` | 20 | `c8f411a4f51100001f9b04de80a272d27a1d83df704ce8439181073fa0e2f526` |
| Code blocks and algorithm | 1 | `97cc8ef7d9acb4855f85d493342fb56bd274a81a7e23f44cd1d1f5d3feb96567` | 1 | `97cc8ef7d9acb4855f85d493342fb56bd274a81a7e23f44cd1d1f5d3feb96567` |
| Inline code | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 0 | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| Commands and paths | 28 | `70c0276602d2acaed72651ce6f6fd77565cc70f6669cf33386c9d37ea4f4a3c8` | 28 | `70c0276602d2acaed72651ce6f6fd77565cc70f6669cf33386c9d37ea4f4a3c8` |
| Numeric and percentage tokens | 474 | `2ee623f2814f002f9206aed6fb0852fc335491a1a78553806eb8d8e0b7de4f9e` | 473 | `86efa476d3f573f085b63b26d3b9a4a48c8ad7080721e4be0b80a02d62250227` |
| Unit tokens | 62 | `2863f487ff565d8fd2bae34d70beb0c60add4f8b7a80e52f8430752721595e77` | 54 | `ee19c6fc7636e262e7f39bf71ebf0055a6933662e76b58d99996ec95956eb98d` |
| Dataset and strategy names | 237 | `b08680e560d8b45cdfc6bb459c87b2e67524585768aefdf7199a688d20ad0fdb` | 236 | `727d7f11be0fae4b37e356d1b5a1f85ec8149d13c9580b8e9957334088470187` |
| Figure/table/equation/section numbers | 12 | `ffb125548ae63ed6bc52cc9484028438f9e3b36a9f934d725b2a40407cf345d4` | 14 | `7da02203d5b53fffa7aceeb7280a04f23e77b87ba32dc685efe83284a0b67a1e` |
| Anchors | 8 | `f203f05a11a7bf5c0c30c4eda0674d60908e5cc8f07d9c3d0837a768319a856f` | 8 | `f203f05a11a7bf5c0c30c4eda0674d60908e5cc8f07d9c3d0837a768319a856f` |
| Local link targets | 12 | `e69585faa4d3608fe6c33ea477a65f181cfdfc26a1025ebb4a4075448656c335` | 12 | `e69585faa4d3608fe6c33ea477a65f181cfdfc26a1025ebb4a4075448656c335` |
| Headings and title | 23 | `512a7752de7de4252260f1ad453b0bd9b4571edee832664007f7f5bcaa63999d` | 23 | `d7401d43bc42edbad033415b68932a1e790f41c486346b1259fc86d3ad415db9` |
| Markdown tables | 3 | `70d955b76934d450e93bdf5325b5d850be52d22a5b6c9c343920607a68a9cc76` | 3 | `e9e5a2bf2bf5c0b2bc2256f17d2be1e1d0920df5b6a132ce0319adb41b0b9e77` |
| Image captions and links | 8 | `4d2a31d7e89d34211f7bc48620188d6816a16f66dff5fa1cba94d87af68dbc14` | 8 | `55c75026c7d6ade51952729a93f74960ac7a98e814ab26241d70c4bdbeeb4404` |

Combined frozen snapshot SHA-256:

- Before: `a614fd398da0567ce20e396035bcfc3903b55c7b38383fd32e1f8c87c5547916`
- After: `a614fd398da0567ce20e396035bcfc3903b55c7b38383fd32e1f8c87c5547916`
- Result: `PASS`

## Edit categories

### English

- Replaced staged research-question and result-summary lead-ins with direct statements.
- Replaced inflated or generic reporting verbs such as “exposes” and “demonstrates” where a plain verb carried the same claim.
- Removed filler transitions such as unnecessary “therefore,” “instead,” “however,” and “consequently” when the logical relationship remained explicit.
- Rewrote the prose dash construction `quality--budget` as a grammatical phrase.
- Tightened ablation and conclusion sentences without changing their caveats or conclusion strength.

### Chinese

- Removed the template construction “不仅……也……” and replaced it with a direct statement.
- Removed staged phrases such as “由此产生一个研究问题” and repeated “结果表明/结果显示” framing.
- Reduced mechanical connectors including unnecessary “因此”“因而”“尤其是” while preserving causal meaning.
- Rewrote `质量--预算` into natural Chinese syntax.
- Tightened ablation and conclusion phrasing without changing evidence scope.

No title, heading, formula, table, image caption/link, citation, algorithm line, bibliography item, number, unit, dataset/strategy name, anchor, or local target was changed.

## Intentionally retained Humanizer patterns

- Scientific negative contrasts remain where they correct plausible misconceptions: rotation regularization is not a full visual-inertial estimator; the formal translation weight is zero; a transient support observation is not a persistent keyframe; the supervision pyramid does not create a Gaussian hierarchy; observed-view rendering does not establish trajectory accuracy; and the TUM A3 failure rules out a general tracking-improvement or divergence-prevention claim.
- The problem and contribution enumerations remain because their items correspond to the actual method decisions and evaluation contribution, rather than decorative triads.
- The short paper roadmap remains because it supplies conventional navigation rather than repeating a scientific claim, and its protected method-name occurrence is part of the frozen name sequence.
- Structured bold labels in the experimental setup and ablation section remain because they identify technical categories; they are not decorative emphasis.

## Verification record

Required Humanizer pattern scan:

~~~text
$ rg -n "—|–|not only|not merely|it is worth noting|delve|pivotal|crucial|landscape|值得注意的是|不仅.*而且|具有重要意义|综上所述" thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md
(no output)
~~~

Strict validators:

~~~text
$ python3 thesis/scripts/validate_bibliography.py thesis/v2/references.bib thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --published-only --require-doi --require-order
bibliography validation passed

$ python3 thesis/scripts/validate_manuscripts.py thesis/v2/manuscript-en.md thesis/v2/manuscript-zh.md --bib thesis/v2/references.bib --root thesis/v2 --require-citation-order
manuscript validation passed
~~~

Frozen-token comparator:

~~~text
before_combined=a614fd398da0567ce20e396035bcfc3903b55c7b38383fd32e1f8c87c5547916
after_combined=a614fd398da0567ce20e396035bcfc3903b55c7b38383fd32e1f8c87c5547916
FROZEN_TOKEN_COMPARATOR=PASS
~~~

Local links and counts:

~~~text
LOCAL_LINK_CHECK: 24 targets checked, 0 missing
thesis/v2/manuscript-en.md: lines=473 whitespace_words=5987 unicode_characters=48484 han_characters=0 utf8_bytes=48500
thesis/v2/manuscript-zh.md: lines=473 whitespace_words=1734 unicode_characters=23918 han_characters=8289 utf8_bytes=41986
~~~

Placeholder and prohibited-claim scan:

~~~text
PLACEHOLDER_SCAN: 0 matches
PROHIBITED_CLAIM_SCAN: 0 positive matches
NEGATED_BOUNDARY_MENTIONS: 1 allowed
  ALLOWED thesis/v2/manuscript-zh.md:417: explicit denial of a divergence-prevention claim
~~~

Whitespace check:

~~~text
$ git diff --check
(no output)
~~~

## Self-review

- The inherited LoopSplat backend boundary remains explicit.
- The formal inertial translation weight remains zero, and the method is not recast as full visual-inertial estimation.
- The fixed selected-seed protocol, observed-view limit, missing FMDataset trajectory truth, TUM A3 failure, and compatible-hardware resource restrictions remain unchanged.
- No SOTA, first-of-kind, universal, or general tracking-improvement claim was introduced.
- The support-observation versus persistent-keyframe distinction and supervision-pyramid versus Gaussian-hierarchy distinction remain explicit.
