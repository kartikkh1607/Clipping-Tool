# Clipper

Paste a YouTube or Twitch link and get the 5 best moments as vertical 1080x1920 clips, with captions, a hook banner, and a title, description and hashtags for each.

Clips are rendered from the best footage available (the chosen moments are re-downloaded at up to 4K / 60 fps), keep the source frame rate, follow the speaker on talking-head videos, and are checked automatically before they're marked done. Works on Windows (NVIDIA GPU used when present) and macOS.

## One-time setup (Windows)
1. Install **Python 3.11** from python.org and tick **"Add python.exe to PATH"**.
2. Double-click **`setup_windows.bat`**. It installs FFmpeg, Deno (yt-dlp needs it for YouTube) and the Python packages, including the GPU libraries for your RTX 3050.
3. Close that window when it finishes.
4. Optional, but recommended: get a free Gemini API key at https://aistudio.google.com/apikey and paste it in the app's **Settings** page (or into `config.json` → `"gemini_api_key"`). With a key, Gemini picks the final moments and writes proper titles and hook banners. Without one, the pipeline still runs on its own scoring with simple titles.

## One-time setup (macOS)
1. Install [Homebrew](https://brew.sh) if you don't have it.
2. In Terminal, in this folder: `bash setup_mac.sh` (installs FFmpeg, Deno, Python 3.11 and the packages).
3. Optional: add a Gemini key in the app's **Settings** page, same as on Windows.

Start it with `./start_app.sh` (web app) or `./run_clipper.sh` (terminal version). Macs have no NVIDIA GPU, so speech recognition runs on the CPU with the `small` model and clips are encoded with x264; both work, just slower than on an RTX PC.

## Daily use
Daily use: double-click the **Clipper** shortcut on the desktop (Windows) or run `./start_app.sh` (macOS).

It opens Clipper in your browser (http://127.0.0.1:7860, only reachable from this PC):
- **Make Clips**: paste one or more links (one per line) and press **Make clips** or Enter. Twitch links from a campaign channel select that campaign automatically and show the **Section** box (e.g. `0:30:00-2:00:00`, empty = whole VOD). Campaign, language and clips per video are under **Options**. A progress bar per video and the live log show what's happening; finished clips appear as cards with **Copy title**, **Copy caption** and **Open folder** buttons.
- **History**: every day's clips, newest first.
- **Settings**: Gemini API key (**Save key**, **Test key**, **Get a free key**), Gemini model, output folder, clip length, default clips per video, layout and cookies browser. Every save keeps the previous settings in `config.json.bak`. Changes apply to the next run without restarting.

Keep the black Clipper window open (it's minimized) while clips are being made. Closing the browser tab doesn't stop a job: open the shortcut again and the page picks up where it was. Links pasted while a job runs wait their turn (one GPU job at a time).
If the shortcut is missing, double-click `start_app.bat` in this folder.

### Terminal version
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
Twitch VODs (`twitch.tv/videos/<id>`) and clips (`twitch.tv/<channel>/clip/<slug>`) work like YouTube links. VODs are hours long, so the analysis copy is at most 720p (the chosen moments are then re-downloaded at source quality, usually 1080p60), and `--section START-END` downloads and processes only that part (timestamps in the `.txt` files still point to the right moment in the full VOD). Twitch has no "Most replayed" heatmap, so moments are scored on loudness, the transcript and hype words. The YouTube-only parts (PO Token provider, browser cookies) are skipped for Twitch.

## Campaign: SkiiMaskTimmy Streamer Campaign
```
python clipper.py "https://www.twitch.tv/videos/<id>" --campaign skiimasktimmy --section 0:30:00-2:00:00
```
The `skiimasktimmy` preset in `campaigns.json` enforces the campaign rules:
- **Source check:** only Twitch footage from the `skiimasktimmy` channel is accepted; any other link is refused before downloading.
- **Facecam layout:** Timmy's facecam is cropped and shown large on top, the official overlay `overlays/skiimasktimmy.png` sits in a band below it, the gameplay fills the rest, and the captions sit over the lower gameplay. Each clip checks that his face is actually in the facecam box. If the cam moved, the clip uses the cam where it was found instead; if there's no cam at all (cam off or a different scene layout), that clip falls back to the `blur` layout, where the full frame keeps the facecam visible.
- **Audio:** only his stream audio, loudness-normalised. Nothing is ever added.
- **Clips:** 20-45 s, 5 per video, English. Gemini only picks the approved formats (clutch plays opening on the payoff, extraction tension, funny moments/fails/rage, first impressions/hot takes, genuine reactions/jump scares, chat banter/challenges) and writes 6-word-max hook banners.
- **Each clip's `.txt`:** TITLE; CAPTION with `twitch.tv/skiimasktimmy` and `#skiimasktimmy #twitchclips #fps #gaming #<game>`; SOURCE (VOD link at the clip's timestamp); and "Post at US prime time: 5:30-9:30 AM IST".
- **Output:** `E:\GamingClips\DD-MM-YYYY\NN_SkiiMaskTimmy_<vod title>\`.

**If his stream layout changes** (the facecam moves): clips find the moved cam on their own (see the log), but for the best framing run
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
2. **Sudden jumps in sound, not just loudness.** Game audio is loud all the time, so each second is compared with the 30 seconds before it and with the video's normal level. A scream after silence, a sudden explosion or chat going off scores high; steady gunfire doesn't. Hype words ("no way", "let's go", "clutch" and so on) add to it.
3. **Strong opening.** The first 5 seconds of a clip are weighted extra.
4. **Skipped:** the intro, the outro, and sponsor or "subscribe" talk.
5. **Clean cuts:** clips start and end on sentence boundaries and don't overlap each other.
6. **Gemini watches the moments** (if you added a key). The top 24 candidates are cut into small 360p previews and Gemini watches them, picture and sound, in batches of 6, scoring each 0-10 as a standalone Short and noting when the payoff happens. Silent clutches and visual fails get found this way. The previews are deleted from Google right after.
7. **Gemini picks the final clips** from those scores, starts clutch plays a few seconds before the payoff, and writes the text. If the watching step fails, it falls back to text-only picking, and if Gemini is unavailable, to the automatic scores.
8. **Every cut is checked again** after Gemini moves it: it never splits a word, it runs on to the end of the sentence when that's within 3 s, the payoff Gemini saw stays inside the clip, and the length stays within your min/max.

## Pro edit (streams with a webcam)
Streamer clips are edited the way pro clippers do it, automatically:
1. **Split layout.** In the `auto` layout, a stream with a webcam in a corner (a small face that stays put and moves like a live cam) gets the facecam layout: the cam large on top, the gameplay below. Before, these got the blur layout, where the streamer was a tiny face in a thin strip. Turn off with `"auto_facecam": false`.
2. **Face zooms on the big reactions.** The clip cuts to the streamer's face full-screen (a 9:16 crop of the cam) for 2 s on the loudest sudden jumps in sound: a scream, "NO WAY", chat going off. Up to 2 per clip, at least 6 s apart (`react_max`, `react_seconds`, `react_min_score`; off with `"react_zoom": false`). Skipped for campaigns with an overlay image, since the overlay would sit on the face.
3. **A word behind the head.** During the first face zoom, one big word that sums up the moment ("SPIDER", "CLUTCH") pops in *behind* the streamer: each frame's person cut-out comes from OpenCV Zoo's PP-HumanSeg model (`models/`), and the word is drawn everywhere except on the person. Gemini picks the word (`punch_word` in its answer); without a key it's the strongest word spoken around the reaction. Swear words are never used. Off with `"punch_word": false`; colour with `punch_color`.
4. **The hook box only for the hook.** The white banner shows for the first 3 s, then fades (`title_banner_seconds`, 0 = whole clip).
5. **Clean captions.** Slurs and hard swears are masked in the captions (`N****`, `F******`); the audio is untouched. Off with `"censor_captions": false`.

These are also switches in the app's **Settings → Pro edit**.

## How it keeps quality high
1. **Two downloads.** The whole video is downloaded once for analysis (up to 1080p; Twitch 720p). After the clips are chosen, only those moments are downloaded again at the best quality the source has (YouTube up to 4K, Twitch "source" 1080p60). Downscaling 4K to 1080x1920 is what makes vertical crops sharp; before, a 1080p frame was enlarged 1.8x (Twitch 720p: 2.7x, facecam: 4.9x). The new footage is lined up with the first download by matching their audio, so cuts are exact to the frame. If the second download fails or can't be lined up, that clip is made from the first download, so nothing is lost.
2. **SDR only.** HDR versions are skipped: converted to normal video they look washed out.
3. **Source frame rate.** 60 fps gameplay stays 60 fps (capped by `max_fps`); 30 fps sources stay 30.
4. **Encoder settings for a clean master.** YouTube re-encodes every upload, so the file it gets should be clean, but past ~20 Mbps the extra size only costs upload time: NVIDIA: preset p7, constant quality 22, adaptive quantisation and lookahead; CPU: x264 `slow`, CRF 20; both capped at 20 Mbps (`max_bitrate`). Lanczos scaling, BT.709 colour tags, a keyframe every 2 s.
5. **Audio.** Each clip's loudness is measured first, then one steady gain brings it to -14 LUFS (a fast limiter only catches peaks), so the volume doesn't pump. Tiny fades at the cut points stop clicks. AAC 192k, 48 kHz.
6. **Layouts that follow people** (`layout`):
   - `auto` (default): on talking-head clips (podcasts, interviews, reactions) a 9:16 crop follows the speaker: it holds still while they stay near the middle, pans when they drift, and cuts on the same frame as the video's own camera switches. Everything else gets `blur`.
   - `blur`: the full frame on a blurred background. `crop`: always the 9:16 crop that follows faces (centre crop if there are none). `facecam`: facecam on top, gameplay below.
   - With `facecam`, every clip checks that a face is actually inside the facecam box. If the streamer moved their cam, Clipper finds it (a face that stays in one place) and uses that; set `"facecam_box": "auto"` to always find it. Faces are found with OpenCV's YuNet model (bundled in `models/`).
7. **Quality check.** Every finished clip is opened again and checked: 1080x1920, right length and frame rate, audio present, not mostly black, loudness on target. A broken file is re-rendered once on the CPU; anything odd is written in the log and the clip's `.txt` (`Quality check:` line), and broken clips are marked in `_summary.txt`.

## Hindi / Hinglish videos
Use `--lang hi` (or answer `hi` in `run_clipper.bat`). Whisper transcribes in Hindi, captions use the Nirmala UI font, Hindi hype words ("bhai", "arey", "bachao"…) count towards the score, and with a Gemini key the captions, title, banner and description are written in Roman-script Hinglish.

## Settings (`config.json`)
| key | default | meaning |
|---|---|---|
| `min_clip_seconds` / `max_clip_seconds` | 20 / 40 | clip length; gaming clips do best around 20-35 s; match each campaign's rules |
| `ai_pool_size` | 24 | how many candidate moments Gemini reviews |
| `vision_review` | true | Gemini watches each candidate before picking; set false to save quota (text-only picks) |
| `vision_batch` | 6 | candidate previews per Gemini request (4 requests for 24 candidates) |
| `vision_preview_height` | 360 | preview resolution sent to Gemini |
| `layout` | `auto` | `auto` = speaker-tracked crop on talking-head clips, `blur` otherwise; `blur` = full gameplay on a blurred background; `crop` = 9:16 crop that follows faces; `facecam` = facecam on top + gameplay below (`facecam_box` as `[x, y, w, h]`, or `"auto"` to find it) |
| `face_tracking` | true | use face detection for `auto`/`crop` and to check/find the facecam |
| `overlay_image` / `overlay_width_pct` | "" / 85 | PNG burned on every clip, kept clear of the facecam, gameplay and captions |
| `max_download_height` | 1080 | resolution of the first (analysis) download |
| `twitch_max_height` | 720 | analysis download cap for Twitch |
| `hq_refetch` | true | re-download the chosen moments at the best quality before rendering |
| `hq_max_height` / `twitch_hq_max_height` | 2160 / 1080 | cap for that high-quality download |
| `max_fps` | 60 | clips keep the source frame rate up to this |
| `encoder` | auto | `auto` (NVIDIA GPU if it works, else CPU), `nvenc` or `x264` |
| `nvenc_cq` / `x264_crf` / `x264_preset` | 22 / 20 / slow | encoder quality (lower = better and bigger); `x264_preset` `medium` is faster |
| `max_bitrate` | 20M | video bitrate cap |
| `audio_bitrate` / `target_lufs` | 192k / -14 | audio quality and loudness target |
| `qa_check` | true | check every clip after rendering (and re-render broken ones) |
| `zoom` | 1.25 | how much to enlarge the gameplay in the blur layout |
| `show_title_banner` | true | white hook box at the top (needs a Gemini key) |
| `title_banner_seconds` | 3 | how long the hook box stays up (0 = whole clip) |
| `censor_captions` | true | mask slurs and hard swears in captions |
| `auto_facecam` / `react_zoom` / `punch_word` | true | Pro edit for streams with a webcam (see above) |
| `react_max` / `react_seconds` / `react_min_score` | 2 / 2.0 / 0.35 | face zooms per clip, their length, how sudden the reaction must be |
| `words_per_caption` | 3 | words shown at once |
| `whisper_model` | large-v3-turbo | speech model (GPU); falls back to `whisper_cpu_model` (`small`) on CPU automatically, and always on a Mac |
| `whisper_batch_size` | 8 | GPU batched transcription (several chunks at once, much faster on long videos); lower to 4, or 1 = off, if the GPU runs out of memory |
| `render_workers` | 2 | clips rendered at the same time |
| `language` | en | default for `--lang`: `en`, `hi`, or `""` = auto-detect |
| `caption_font` | Arial Black (Windows, Mac) | caption font; DejaVu Sans on Linux |
| `hindi_caption_font` | Nirmala UI (Windows), Kohinoor Devanagari (Mac) | caption font for Hindi (supports Devanagari); Hindi captions are not uppercased |
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
- **"High-quality download failed":** the clips were still made, from the first download. Usually YouTube's bot check; the fixes for download errors above apply. Set `"hq_refetch": false` to skip the second download entirely.
- **Clips render slowly on the CPU:** set `"x264_preset": "medium"` (a little less quality per MB) or lower `render_workers` if the PC gets too hot.
- **The speaker-following crop picks the wrong person:** it follows the biggest face (or both, when they fit). Use `"layout": "blur"` for that video.

## Tests
`pip install pytest` then `python -m pytest tests` (needs FFmpeg). The tests render real clips from generated footage and check frame rate, size, colour tags, loudness, the quality check, audio alignment, cut snapping and face-tracking decisions. They also run on GitHub for every push (`.github/workflows/tests.yml`).

Only clip creators who allow it (a Whop campaign or explicit clip permission), and keep the credit line in the description.
