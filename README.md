# MiniMax H3-as-Suno: Empirical Notes on Local Music Generation

> These are **empirical observations and reasonable inferences**, not official or authoritative conclusions. They come from repeated A/B listening tests on **one machine, one model configuration**. They are **not guaranteed to generalize** to other GPUs, other quantization builds, other CLIPs, or other versions of the model.

**Acknowledgement**: This work is built entirely on [Deveraux-Parker/minimax-h3-voice-api](https://github.com/Deveraux-Parker/minimax-h3-voice-api) — its `song_prompt()` / `instrumental_prompt()` assemblers and its 99 genre presets are the foundation of every test. No code from that project was modified; this repository is a standalone write-up.

**Test environment**: local ComfyUI + MiniMax H3 (quantized FL2VA build), NVIDIA 8 GB VRAM, bf16 CLIP.

---

## ⭐ The headline finding: canvas size is the style knob, step count is the smoothness knob

The most reproducible observation of this whole investigation:

| | 32×32 canvas | 64×64 canvas |
|---|---|---|
| **Any step count (5–70 tested)** | keeps the requested style (e.g. horror-funk stays horror-funk) | drifts toward a gentle, "modern" short-video-BGM sound, even at 5 steps |
| Effect of raising steps | rough → smooth → **flat** (see below) | smooth → **flat** |

- **32×32 (1024 spatial tokens) = condition-dominated** → the output is "pinned" by the prompt, so a distinctive style survives.
- **64×64 (4096 spatial tokens) = prior-dominated** → there is more room for the model's trained prior to take over, and the prior's *mode* is the most common audio in its training data — which, for an audiovisual model whose main job is short-form video, is gentle, unobtrusive, background-music-ish audio. Edge styles (horror-funk, metal, etc.) get averaged toward that center.
- **There are two different "over-smoothing" mechanisms**:
  - **Canvas over-smoothing (64×64)** = the *style identity* is lost (horror-funk becomes gentle; you get a different song).
  - **Step over-smoothing (60+ steps)** = *dynamic variation* is lost — aggressive denoising treats instrument/vocal expression (swells, attacks, bends) as noise and averages them out, so the same song goes "flat" while its style survives.
  - The two are independent: canvas decides *which* song you get, steps decide *how much life* that song keeps.
- **Optimal step count ≈ 25–60**: below ~25 the audio is rough/noisy; 25–60 is the sweet spot (smooth enough, still alive); above ~60 it flattens.
- **32×32 is the hard minimum** for the canvas (the node validates `min=32`, `step=32`), so it cannot be shrunk further to get even stronger prompt control.

**Practical takeaway**:
- Want a *specific style* with life → lock 32×32, steps in the 25–60 range.
- Want *short-video-style BGM* → use 64×64; you barely need to describe anything, the prior gives it to you.

---

## 1. The steps × canvas grid (core experiment)

Same prompt (a "Thriller-era" horror-funk song), same seed, same bf16 CLIP; only canvas and steps varied:

| | 5 steps | 10 | 15 | 25 | 50–70 |
|---|---|---|---|---|---|
| 32×32 | horror ✓ | horror ✓ | horror ✓ | horror ✓ | smooth + horror (still under test) |
| 64×64 | modern/gentle | modern/gentle | modern/gentle | modern/gentle | modern/gentle |

The 64×64 result holds **even at 5 steps**, which is the cleanest evidence that the canvas — not the steps — is what pulls the audio toward the short-video centroid.

---

## 2. CLIP and base-model differences

Different text encoders and diffusion base models measurably change the music output:

| Variable | Observed effect |
|---|---|
| **bf16 CLIP vs int8 CLIP** | bf16 gives clearly better instrument separation; int8 tends to smear instruments into one blob. Same UNET/prompt/seed, only the encoder precision changes. |
| **pruned int8 UNET vs full int8 UNET** | full int8 (≈34 GB) is impractical on 8 GB VRAM (constant pagefile swapping, per-step disk reads). The difference this test could isolate was mostly I/O-bound, not quality-bound. |
| **w4a8 base** | audio quality is degraded vs the int8 line (this project's earlier A/B). |
| **Different video base variants (pruned/full/w4a8)** | each shifts the audio flavor; on this rig, pruned int8 + bf16 CLIP was the practical sweet spot. |

If you run a different CLIP or a different base build, expect the "how much the prompt sticks" balance to shift.

---

## 3. Prompting discipline (13 empirical lessons)

1. **Keep music prompts ≤ ~4000 chars**: style anchor + arrangement anchors + one vocal line. Every extra instruction competes for attention; patch-on descriptions have negative marginal returns.
2. **Single name beats double name**: naming two artists/genres causes cluster flicker — more varied singing but more breakage.
3. **Don't over-specify vocal techniques** (`hiccuping ad-libs`, `trembling high notes`, `staccato`) — the model renders them as strange voices. Use "in X's unmistakable style, smooth and powerful" level of simplicity.
4. **Negatives backfire** (`never over-decorated` summons over-decoration). Use structure, not bans.
5. **Change, don't add**: when adding a new instruction, delete the old one it contradicts. `explosive` + `authentically early-1980s` = contradiction → mud.
6. **Vintage/analog production language invites mud**: tape saturation / gated-reverb / analog character → the audio VAE renders it blurry. Anchor era lightly ("Thriller-era") without production fingerprints.
7. **Verse-led Japanese ballads = strong "vocal-first" prior**: adding backing, locking a groove, or restructuring can't override it; accept it rather than fight it. Groove-based genres (funk) don't have this.
8. **Per-section directing does not work**: hand-written timestamp storyboards (`From X to Y + intensity`) are only partially followed and make the whole piece bizarre. Use the prose-and-section-tags approach.
9. **IP name + precise rhythmic instructions risks regurgitating the original song**: naming a specific artist/song plus per-section rhythm instructions drifted into reproducing a real musical passage (deformed). Name IPs in prose style only.
10. **Modern-word clusters override era anchors**: `stadium/explosive/slam` is effectively a "nu-funk spell" that beats a "Thriller-era 80s" anchor. For era-authentic sound, use era-authentic words (80s synth-funk: LinnDrum, synth bass, boogie syncopation).
11. **The whole prompt must be one coherent cluster**: style words / lyrics / scene / vocal must all point the same direction. Cross-cluster mixing (e.g. "midnight-horror" lyrics on a smooth-R&B groove) is reliably disharmonious.
12. **The naming model** — see section 4.
13. **Canvas/steps split** — see the headline finding above.

---

## 4. The naming model (what an artist/song name does)

| Dimension | Finding |
|---|---|
| Vocal identity | **Determined by technique words, but only inside a matching cluster.** MJ technique words in a Thriller cluster → MJ voice. Put them in a metalcore cluster → voice is not MJ *and* the metalcore is dragged toward generic rock. |
| Name strength | Weak style nudge; cannot beat a competing word cluster. |
| Name presence | Cohesion anchor; the whole track is more coherent with it. |
| Name identity (which song/album) | ≈ no effect; direction nearly identical, only details differ. |
| Content words | The real driver; delete the name and keep the content words and the style mostly survives. |

Practical: to get a specific singer's voice, write that singer's technique words — inside a matching cluster. Include the name for cohesion, but don't expect a stronger name to override other clusters.

---

## 5. Capability boundaries (why H3 music "has a ceiling")

1. **Training-distribution bias**: H3's musical priors were shaped by music *in video*. Mainstream, formulaic genres generalize well; niche edge genres (e.g. breakcore) come out but miss the genre's soul.
2. **Audio-VAE synthesis ceiling**: dense-harmonic timbres (distorted electric guitar) render blurry no matter the prompt.
3. **Operational limits**: 60 s is ~4× the training domain (5–15 s); attention budget; canvas coupling.
4. **Weak cross-cluster mixing**: clusters must match; mixing clusters dilutes both sides.

---

## 6. Caveats / disclaimer

- **Empirical only.** Based on one machine (8 GB VRAM) and one quantized H3 build. Not guaranteed to hold on other hardware, quantization variants, CLIPs, or model versions.
- **Subjective judging.** Every "better/muddier/gentler" came from human listening, not quantitative metrics.
- **Not an authoritative statement about MiniMax H3's capabilities.** Different builds may behave differently.
- **No private or authorized information included**: no API keys, no machine paths, no personal data.

---

## License / note

This is a research write-up, not affiliated with MiniMax. Reuse at your own discretion; cite the original `minimax-h3-voice-api` if you build on its tooling.
