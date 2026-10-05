"""Keyframe the signer fingerspelling a phrase, render PNG frames.

Each letter: the hand moves into the shape (0.3 s; 0.5 s when raising the hand), holds it
for 0.9 s with a slight drift, then the next letter. Between words the hand drops out of
the picture for 1.6 s, which the app reads as a space.
"""
import bpy, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose

args = sys.argv[sys.argv.index("--") + 1:]
OUT, TEXT = args[0], args[1]
FPS, LEAD, RAISE, MOVE, HOLD, DOWN, TAIL = 30, 1.0, 0.5, 0.3, 0.9, 1.6, 1.2

scene = bpy.context.scene
scene.render.fps = FPS
pose.setup_camera()
bones = [pb for pb in pose.P if pb.name.endswith(".R") or pb.name.endswith(".L")]


keys = []   # (time, snapshot): poses are all worked out first, keyframed afterwards, so
            # existing animation can never overwrite a pose while it is being computed


def key(t):
    keys.append((t, {pb.name: (pb.rotation_mode, pb.location.copy(), pb.rotation_quaternion.copy(), pb.rotation_euler.copy())
                     for pb in bones}))


t, truth = 0.0, []
pose.hands_down(); key(t)
t += LEAD; key(t)
for wi, word in enumerate(TEXT.split()):
    if wi:
        pose.halfway(); key(t + RAISE / 2)
        pose.hands_down(); t += RAISE; key(t)
        t += DOWN; key(t)
    for li, L in enumerate(word):
        if li == 0:
            pose.halfway(); key(t + RAISE / 2)
        t += RAISE if li == 0 else MOVE
        pose.letter(L, sway=-0.03); key(t)
        truth.append({"letter": L, "from_s": round(t, 2), "to_s": round(t + HOLD, 2)})
        t += HOLD
        pose.letter(L, sway=0.03); key(t)
pose.halfway(); key(t + RAISE / 2)
pose.hands_down(); t += RAISE; key(t)
t += TAIL; key(t)

# q and -q are the same rotation, but keys interpolate channel by channel: keep each bone's
# consecutive quaternions on the same side, or the arm swings the long way round
for (_, prev), (_, cur) in zip(keys, keys[1:]):
    for name, (mode, loc, q, e) in cur.items():
        if q.dot(prev[name][2]) < 0:
            cur[name] = (mode, loc, -q, e)

for kt, snap in keys:
    f = 1 + round(kt * FPS)
    for pb in bones:
        mode, loc, q, e = snap[pb.name]
        pb.rotation_mode = mode
        pb.location, pb.rotation_quaternion, pb.rotation_euler = loc, q, e
        pb.keyframe_insert("location", frame=f)
        pb.keyframe_insert("rotation_euler" if mode == "XYZ" else "rotation_quaternion", frame=f)

scene.frame_start, scene.frame_end = 1, 1 + round(t * FPS)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = OUT + "/f_"
json.dump({"text": TEXT, "fps": FPS, "letters": truth, "duration_s": round(t, 2)}, open(OUT + "/truth.json", "w"), indent=1)
print("FRAMES", scene.frame_end, "LETTERS", len(truth))
if "--still-only" not in args:
    bpy.ops.render.render(animation=True)
