# Clipper

Paste a YouTube or Twitch link and get the 5 best moments as vertical 1080x1920 clips, with captions, a hook banner, and a title, description and hashtags for each.

## One-time setup (Windows)
1. Install **Python 3.11** from python.org and tick **"Add python.exe to PATH"**.
2. Double-click **`setup_windows.bat`**. It installs FFmpeg, Deno (yt-dlp needs it for YouTube) and the Python packages, including the GPU libraries for your RTX 3050.
3. Close that window when it finishes.
4. Optional, but recommended: get a free Gemini API key at https://aistudio.google.com/apikey and paste it into `config.json` → `"gemini_api_key"`. With a key, Gemini picks the final moments and writes proper titles and hook banners. Without one, the pipeline still runs on its own scoring with simple titles.

## Daily use
Double-click **`run_clipper.bat`** and paste a link. It then asks:
- **Campaign** (`skiimasktimmy` or Enter = none)
- **Language** (`en` or `hi`, Enter = `en`; only asked without a campaign, since a campaign sets it)
- **Section** (Twitch VOD links only, e.g. `0:30:00-2:00:00`; Enter = whole VOD)

Paste the next link, or press Enter on an empty line to quit.
The first run asks where to save clips and remembers the folder.

Other ways to run it (from the folder, after `.venv\Scripts\activate`):
```
python clipper.py "LINK"                 one video
python clipper.py LINK1 LINK2 LINK3      several videos
python clipper.py --list links.txt       one link per line
python clipper.py "LINK" --clips 3       a different number of clips
python clipper.py "LINK" --lang hi       Hindi / Hinglish video (en | hi | auto)
python clipper.py --file game.mp4 --credit "CreatorName"   a local file
python clipper.py "TWITCH_VOD" --section 0:30:00-2:00:00   only that part of a VOD
python clipper.py "LINK" --campaign skiimasktimmy           a campaign preset (campaigns.json)
python clipper.py "LINK" --overlay-image overlays\logo.png  burn a PNG onto every clip
```

## Twitch
Twitch VODs (`twitch.tv/videos/<id>`) and clips (`twitch.tv/<channel>/clip/<slug>`) work like YouTube links. VODs are hours long, so they're downloaded at most 720p, and `--section START-END` downloads and processes only that part (timestamps in the `.txt` files still point to the right moment in the full VOD). Twitch has no "Most replayed" heatmap, so moments are scored on loudness, the transcript and hype words. The YouTube-only parts (PO Token provider, browser cookies) are skipped for Twitch.

## Campaign: SkiiMaskTimmy Streamer Campaign
```
python clipper.py "https://www.twitch.tv/videos/<id>" --campaign skiimasktimmy --section 0:30:00-2:00:00
```
The `skiimasktimmy` preset in `campaigns.json` enforces the campaign rules:
- **Source check:** only Twitch footage from the `skiimasktimmy` channel is accepted; any other link is refused before downloading.
- **Facecam layout:** Timmy's facecam is cropped and shown large on top, the official overlay `overlays/skiimasktimmy.png` sits in a band below it, the gameplay fills the rest, and the captions sit over the lower gameplay. If a clip's facecam box is empty (cam off or a different scene layout), that clip falls back to the `blur` layout, where the full frame keeps the facecam visible.
- **Audio:** only his stream audio, loudness-normalised. Nothing is ever added.
- **Clips:** 20-45 s, 5 per video, English. Gemini only picks the approved formats (clutch plays opening on the payoff, extraction tension, funny moments/fails/rage, first impressions/hot takes, genuine reactions/jump scares, chat banter/challenges) and writes 6-word-max hook banners.
- **Each clip's `.txt`:** TITLE; CAPTION with `twitch.tv/skiimasktimmy` and `#skiimasktimmy #twitchclips #fps #gaming #<game>`; SOURCE (VOD link at the clip's timestamp); and "Post at US prime time: 5:30-9:30 AM IST".
- **Output:** `E:\GamingClips\DD-MM-YYYY\NN_SkiiMaskTimmy_<vod title>\`.

**If his stream layout changes** (the facecam moves): run
`python clipper.py --find-facecam "<VOD link>"` (add `--facecam-at 0:45:00` to pick a gameplay moment). It saves `facecam_check.png` with a labelled pixel grid; read the camera's x, y, width and height and update `facecam_box` (and `facecam_ref`, the frame size printed by the command) in `campaigns.json`.

**New campaign:** copy the `skiimasktimmy` block in `campaigns.json`, rename it, and change the overlay, channel, facecam box, hashtags and formats.

## Output
```
<your folder>\
  12-11-2026\
    01_CreatorName_Video_Title\
      01_<title>.mp4   <- strongest clip
      01_<title>.txt   <- title, description with credit + timestamped source link, hashtags
      01_<title>.srt   <- captions file (optional upload)
      02_...  03_...  04_...  05_...
      _summary.txt     <- all clips with their source timestamps
      _transcript.txt
    02_NextCreator_Next_Video\
```

## How it picks the "best moments"
1. **YouTube "Most replayed" heatmap.** These are the parts real viewers rewatch, and this is the strongest signal. It's only available on videos with enough views, which famous videos usually have.
2. **Audio loudness and hype.** Screams, explosions and hype words ("no way", "let's go", "clutch" and so on) raise a moment's score.
3. **Strong opening.** The first 5 seconds of a clip are weighted extra.
4. **Skipped:** the intro, the outro, and sponsor or "subscribe" talk.
5. **Clean cuts:** clips start and end on sentence boundaries and don't overlap each other.
6. **Gemini** (if you added a key) reviews the top 12 candidates, picks the 5 best, tightens the cuts and writes the text.

## Hindi / Hinglish videos
Use `--lang hi` (or answer `hi` in `run_clipper.bat`). Whisper transcribes in Hindi, captions use the Nirmala UI font, Hindi hype words ("bhai", "arey", "bachao"…) count towards the score, and with a Gemini key the captions, title, banner and description are written in Roman-script Hinglish.

## Settings (`config.json`)
| key | default | meaning |
|---|---|---|
| `min_clip_seconds` / `max_clip_seconds` | 30 / 59 | clip length; match each Whop campaign's rules |
| `layout` | `blur` | `blur` = full gameplay on a blurred background; `crop` = full-screen center crop; `facecam` = facecam on top + gameplay below (needs `facecam_box`) |
| `overlay_image` / `overlay_width_pct` | "" / 85 | PNG burned on every clip, kept clear of the facecam, gameplay and captions |
| `twitch_max_height` | 720 | download cap for Twitch |
| `zoom` | 1.25 | how much to enlarge the gameplay in the blur layout |
| `show_title_banner` | true | white hook box at the top (needs a Gemini key) |
| `words_per_caption` | 3 | words shown at once |
| `whisper_model` | large-v3-turbo | speech model (GPU); falls back to `small` on CPU automatically |
| `whisper_batch_size` | 8 | GPU batched transcription (several chunks at once, much faster on long videos); lower to 4, or 1 = off, if the GPU runs out of memory |
| `render_workers` | 2 | clips rendered at the same time |
| `language` | en | default for `--lang`: `en`, `hi`, or `""` = auto-detect |
| `hindi_caption_font` | Nirmala UI | caption font for Hindi (supports Devanagari); Hindi captions are not uppercased |
| `hinglish_roman_captions` | true | Hindi only: Gemini rewrites Devanagari caption words in Roman Hinglish ("bhai ye kya scene hai"), keeping English words and word timings. Without a key, or if the call fails, captions stay in Devanagari |
| `cookies_from_browser` | "" | set to `"chrome"` if a video needs sign-in (age-restricted) |
| `pot_provider` | true | start the bgutil PO Token server before YouTube downloads (set false when using `cookies_from_browser`; the two are never combined) |
| `gemini_model` | gemini-3.5-flash-lite | free tier: 500 requests/day, 15/min |
| `gemini_fallback_models` | 3.6-flash, 3.8-flash | free tier: 20/day, 5/min each. A 429 (quota used up) jumps straight to the next model; a 503 (overloaded) waits and retries. All Gemini calls are queued at most 14 per minute. The log shows which model answered |
| `keep_source` | false | keep the full downloaded video |

## If something breaks
- **YouTube download errors:** `run_clipper.bat` updates yt-dlp on every start. If it still fails, run `pip install -U "yt-dlp[default]"` and check that `deno --version` works.
- **"cudnn…dll not found" or GPU errors:** the pipeline falls back to CPU by itself, which is slower but works. To fix the GPU path, run `pip install -U nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"`.
- **"Driver does not support the required nvenc API version":** your NVIDIA driver is older than the FFmpeg build needs, so clips are encoded on the CPU (libx264) instead. Update the NVIDIA driver to use the GPU encoder.
- **No captions on a clip:** that moment had no speech (pure gameplay).

Only clip creators who allow it (a Whop campaign or explicit clip permission), and keep the credit line in the description.
