# The blink sample clip

`web/samples/blink_sos_help.mp4` is **synthetic**: a computer-generated face
blinking "SOS HELP" in Morse. No real person was filmed.

- Character: [MakeHuman](http://www.makehumancommunity.org/) via its Blender
  add-on MPFB 2. The system assets (skin, eyes, lashes, hair, clothes) are CC0.
- The eyelids close with MakeHuman's own `eye-left-closure` / `eye-right-closure`
  expression targets, loaded as shape keys named `eyeBlinkLeft` / `eyeBlinkRight`
  so MPFB copies them onto the eyelashes and eyebrows too.
- Timing: dot 0.30 s, dash 0.90 s, pauses 0.35 s inside a letter, 0.95 s
  between letters, 2.4 s between words, every length randomly within ±10%;
  eyelids take 100 ms to close and to open. The true timings are in
  `truth.json`.
- 640x480, 30 fps, Blender EEVEE, a slight head sway.

Rebuild (Blender 5.2 with MPFB 2 and the MakeHuman system assets installed):

```powershell
blender -b --addons bl_ext.user_default.mpfb --python build.py -- face.blend
blender -b face.blend --python animate.py -- frames "SOS HELP"
ffmpeg -framerate 30 -i frames/f_%04d.png -c:v libx264 -crf 26 -pix_fmt yuv420p -movflags +faststart blink_sos_help.mp4
```

(The published clip also has a "SYNTHETIC SAMPLE" label burned in.)
