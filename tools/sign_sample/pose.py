"""Pose the MakeHuman signer: raise the right hand, palm to camera, and shape ASL letters."""
import bpy, math, sys, os
from mathutils import Matrix, Vector

rig = bpy.data.objects["signer"]
P, D = rig.pose.bones, rig.data.bones
R = ".R"


def upd():
    bpy.context.view_layer.update()


def rest_vec(a, b):
    return (D[b].head_local - D[a].head_local).normalized()


# the hand's own axes at rest (armature space = world here): fingers and thumb side
F_REST = (D["finger3-1" + R].head_local - D["wrist" + R].head_local).normalized()   # wrist -> middle knuckle
S_REST = (D["finger2-1" + R].head_local - D["finger5-1" + R].head_local).normalized()  # pinky -> index knuckle


def frame(f, s):
    f = f.normalized()
    s = (s - f * s.dot(f)).normalized()
    return Matrix((s, f, s.cross(f))).transposed()          # columns: thumb side, fingers, palm normal


def orient(name, rot):
    """Rotate bone `name` rigidly by world rotation `rot` from its rest orientation, keeping its posed head."""
    pb = P[name]
    m = (rot @ D[name].matrix_local.to_3x3()).to_4x4()
    m.translation = pb.head.copy() if pb.parent else D[name].head_local
    pb.matrix = m
    upd()


def aim(name, direction):
    """Point bone `name` along `direction` with the smallest turn from rest."""
    rest_dir = (D[name].tail_local - D[name].head_local).normalized()
    orient(name, rest_dir.rotation_difference(direction.normalized()).to_matrix())


def raise_hand(upper=Vector((-0.28, -0.10, -1.0)), fore=Vector((0.12, -0.45, 1.0)),
               fingers=Vector((0.05, -0.12, 1.0)), thumb_side=Vector((1.0, 0.0, 0.0))):
    """Upper arm down by the side, forearm up, hand fingers-up with the palm toward the camera (-Y)."""
    for pb in P:
        pb.matrix_basis = Matrix.Identity(4)
    upd()
    aim("upperarm01" + R, upper); aim("upperarm02" + R, upper)
    hand = frame(fingers, thumb_side) @ frame(F_REST, S_REST).inverted()
    fore_rot = frame(fore, thumb_side) @ frame(rest_vec("lowerarm01" + R, "wrist" + R), S_REST).inverted()
    orient("lowerarm01" + R, fore_rot); orient("lowerarm02" + R, fore_rot)
    orient("wrist" + R, hand)
    aim("upperarm01.L", Vector((0.22, -0.02, -1.0))); aim("upperarm02.L", Vector((0.22, -0.02, -1.0)))   # other arm relaxed
    aim("lowerarm01.L", Vector((0.08, -0.35, -1.0))); aim("lowerarm02.L", Vector((0.08, -0.35, -1.0)))


def bend(name, deg, axis="X"):
    pb = P[name]
    pb.rotation_mode = "XYZ"
    e = list(pb.rotation_euler)
    e["XYZ".index(axis)] = math.radians(deg)
    pb.rotation_euler = e


STRAIGHT, CURL = (0, 0, 0), (80, 95, 60)
FINGERS = {"index": 2, "middle": 3, "ring": 4, "pinky": 5}


def finger(which, angles, spread=0):
    i = FINGERS[which]
    for j, a in zip((1, 2, 3), angles):
        bend(f"finger{i}-{j}{R}", a)
    if spread:
        bend(f"finger{i}-1{R}", spread, "Z")


THUMB = {   # finger1-1, -2, -3 as (x, z) per joint; tuned against stills
    "side": [(0, 0), (0, 0), (0, 0)],
    "across": [(35, -30), (40, 0), (35, 0)],
    "out": [(-20, 35), (0, 0), (0, 0)],
    "touch": [(30, -15), (25, 0), (20, 0)],
    "side1": [(5, -20), (0, 0), (0, 0)],
}


def thumb(state):
    for j, (x, z) in zip((1, 2, 3), THUMB[state]):
        n = f"finger1-{j}{R}"
        bend(n, x, "X"); bend(n, z, "Z")


# Each shape was checked by rendering a still and running the real pipeline on it
# (MediaPipe HandLandmarker + the letter network). B D E F I L O V W Y read correctly.
# A (read as Y/E/X: the thumb would not sit up along the fist) and U (fingers together,
# still read as V) do not, so the sample phrase avoids them.
LETTERS = {   # index, middle, ring, pinky, thumb
    "A": (CURL, CURL, CURL, CURL, "side"),
    "B": (STRAIGHT, STRAIGHT, STRAIGHT, STRAIGHT, "across"),
    "D": (STRAIGHT, CURL, CURL, CURL, "touch"),
    "I": (CURL, CURL, CURL, STRAIGHT, "across"),
    "L": (STRAIGHT, CURL, CURL, CURL, "out"),
    "U": (STRAIGHT, STRAIGHT, CURL, CURL, "across"),
    "V": ((STRAIGHT, -9), (STRAIGHT, 9), CURL, CURL, "across"),
    "W": ((STRAIGHT, -10), STRAIGHT, (STRAIGHT, 10), CURL, "across"),
    "Y": (CURL, CURL, CURL, STRAIGHT, "out"),
    "F": ((85, 60, 40), STRAIGHT, STRAIGHT, STRAIGHT, "touch"),
    "E": ((40, 100, 70), (40, 100, 70), (40, 100, 70), (40, 100, 70), "side1"),
    "O": ((25, 60, 45), (25, 60, 45), (25, 60, 45), (25, 60, 45), "touch"),
}


def hands_down():
    """Both arms relaxed by the sides: the hand is out of the picture."""
    for pb in P:
        pb.matrix_basis = Matrix.Identity(4)
    upd()
    for side, x in ((".R", -0.22), (".L", 0.22)):
        aim("upperarm01" + side, Vector((x, -0.02, -1.0))); aim("upperarm02" + side, Vector((x, -0.02, -1.0)))
        aim("lowerarm01" + side, Vector((x * 0.4, -0.35, -1.0))); aim("lowerarm02" + side, Vector((x * 0.4, -0.35, -1.0)))
    for which in FINGERS:
        finger(which, (15, 20, 10))
    upd()


def halfway():
    """Midway between hands down and signing: forearm forward, so the hand comes up in front."""
    raise_hand(fore=Vector((0.1, -1.0, 0.2)), fingers=Vector((0.05, -0.8, 0.6)))
    for which in FINGERS:
        finger(which, (15, 20, 10))
    upd()


def letter(L, sway=0.0):
    raise_hand(fingers=Vector((0.05 + sway, -0.12, 1.0)))
    for which, spec in zip(FINGERS, LETTERS[L][:4]):
        angles, spread = (spec if isinstance(spec[0], tuple) else (spec, 0))
        finger(which, angles, spread)
    thumb(LETTERS[L][4])
    upd()


def setup_camera():
    cam = bpy.data.objects["cam"]
    cam.data.lens = 50
    target = Vector((-0.12, 0.0, 1.42))
    cam.location = target + Vector((0.0, -1.55, 0.0))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


if __name__ == "__main__" and "--" in sys.argv:
    setup_camera()
    out, letters = sys.argv[sys.argv.index("--") + 1], sys.argv[sys.argv.index("--") + 2]
    scene = bpy.context.scene
    sway = float(os.environ.get("SWAY", "0"))
    for L in letters:
        letter(L, sway)
        scene.render.filepath = f"{out}/{L}.png"
        bpy.ops.render.render(write_still=True)
