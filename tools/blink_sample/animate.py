"""Keyframe the eyelids to blink a Morse message, add a little head sway, render PNG frames."""
import bpy, sys, json, random, math
args = sys.argv[sys.argv.index("--") + 1:]
OUT, TEXT = args[0], args[1]
MORSE = {"S": "...", "O": "---", "H": "....", "E": ".", "L": ".-..", "P": ".--."}
DOT, DASH, IN_GAP, LETTER_GAP, WORD_GAP, LEAD = 0.30, 0.90, 0.35, 0.95, 2.40, 1.5
FPS, RAMP = 30, 3          # eyelids take 3 frames (100 ms) to close and to open
r = random.Random(7)
j = lambda d: d * r.uniform(0.9, 1.1)    # people are not metronomes

closures, t = [], LEAD
for wi, word in enumerate(TEXT.split()):
    if wi:
        t += j(WORD_GAP) - j(LETTER_GAP)
    for li, ch in enumerate(word):
        if li or wi:
            t += j(LETTER_GAP) - j(IN_GAP)
        for m in MORSE[ch]:
            d = j(DOT if m == "." else DASH)
            closures.append((round(t, 3), round(t + d, 3), m))
            t += d + j(IN_GAP)
end = t + 1.5

scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 1, int(end * FPS)
keys = [o.data.shape_keys.key_blocks[k] for o in scene.objects if o.type == "MESH" and o.data.shape_keys
        for k in ("eyeBlinkLeft", "eyeBlinkRight") if k in o.data.shape_keys.key_blocks]

def key(frame, value):
    for kb in keys:
        kb.value = value
        kb.keyframe_insert("value", frame=frame)

key(1, 0.0)
for a, b, _ in closures:
    fa, fb = 1 + round(a * FPS), 1 + round(b * FPS)
    key(fa - 1, 0.0); key(fa - 1 + RAMP, 1.0)        # fully closed RAMP frames after the start
    key(fb - RAMP + 1, 1.0); key(fb + 1, 0.0)        # starts reopening shortly before the end
human = bpy.data.objects["Human"]
for i, amp, period in ((0, 0.012, 7.3), (2, 0.02, 9.1)):   # small nod (x) and turn (z), in radians
    for f in range(1, scene.frame_end + 1, 10):
        human.rotation_euler[i] = amp * math.sin(2 * math.pi * f / FPS / period + i)
        human.keyframe_insert("rotation_euler", index=i, frame=f)

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = OUT + "/f_"
json.dump({"text": TEXT, "fps": FPS, "closures": closures, "duration_s": round(end, 2)},
          open(OUT + "/truth.json", "w"), indent=1)
print("FRAMES", scene.frame_end, "MARKS", len(closures))
bpy.ops.render.render(animation=True)
