# The sign language sample clip

`web/samples/sign_be_bold.mp4` is **synthetic**: a computer-generated person
fingerspelling "BE BOLD" in ASL with the right hand. No real person was filmed.

- Character: [MakeHuman](http://www.makehumancommunity.org/) via its Blender
  add-on MPFB 2, with its "default" skeleton (finger bones included). The
  system assets (skin, eyes, hair, clothes) are CC0.
- `pose.py` raises the hand by aiming the arm bones in world space and turning
  the hand rigidly so the fingers point up and the palm faces the camera, then
  curls each finger for the letter. Every letter shape was checked by rendering
  a still and running the real pipeline on it (MediaPipe HandLandmarker + the
  letter network): B D E F I L O V W Y are read correctly; A and U are not
  (see the comment in `pose.py`), so the phrase avoids them.
- `animate_sign.py`: each letter is held 0.9 s with a slight drift, 0.3 s to
  change shape, and the hand drops out of view for 1.6 s between words (read
  as a space). All poses are worked out before any keyframe is set, and
  quaternion keys are kept on one hemisphere so the arm never swings the long
  way round. The true timings are in `truth.json`.
- 640x480, 30 fps, Blender EEVEE, "SYNTHETIC SAMPLE" label burned in.
- Encoded at CRF 18, not the CRF 26 used for the blink clip: compression blurs
  the fingers enough to matter. O scored 0.93-0.99 on the rendered frames,
  0.59-0.80 after CRF 26 (too close to the 0.6 cut-off), 0.88-0.98 at CRF 18.

Rebuild (Blender 5.2 with MPFB 2 and the MakeHuman system assets installed):

```powershell
blender -b --addons bl_ext.user_default.mpfb --python build.py -- signer.blend
blender -b signer.blend --python animate_sign.py -- frames "BE BOLD"
ffmpeg -framerate 30 -i frames/f_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart sign_be_bold.mp4
```
