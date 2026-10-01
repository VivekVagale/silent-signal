"""Build a CC0 MakeHuman character (MPFB) with eye-closure shape keys, camera and lights."""
import bpy, sys, math, os
from mathutils import Vector
from bl_ext.user_default.mpfb.services.humanservice import HumanService
from bl_ext.user_default.mpfb.services.targetservice import TargetService
from bl_ext.user_default.mpfb.services.faceservice import FaceService
from bl_ext.user_default.mpfb.services.locationservice import LocationService

OUT = sys.argv[sys.argv.index("--") + 1]
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

info = HumanService._create_default_human_info_dict()
info["phenotype"]["gender"] = 1.0
info["eyes"] = "high-poly/high-poly.mhclo"
info["eyebrows"] = "eyebrow001/eyebrow001.mhclo"
info["eyelashes"] = "eyelashes01/eyelashes01.mhclo"
info["hair"] = "short02/short02.mhclo"
info["clothes"] = ["male_casualsuit01/male_casualsuit01.mhclo"]
info["skin_mhmat"] = "young_asian_male/young_asian_male.mhmat"
info["skin_material_type"] = "ENHANCED_SSS"
info["alternative_materials"] = {}
info["eyes_material_type"] = "MAKESKIN"
settings = HumanService.get_default_deserialization_settings()
settings["subdiv_levels"] = 0
basemesh = HumanService.deserialize_from_dict(info, settings)
print("BASEMESH", basemesh.name, [c.name for c in basemesh.children])

units = os.path.join(LocationService.get_mpfb_data("targets"), "expression", "units", "asian")
for side, name in (("left", "eyeBlinkLeft"), ("right", "eyeBlinkRight")):
    TargetService.load_target(basemesh, os.path.join(units, f"eye-{side}-closure.target.gz"), weight=0.0, name=name)
FaceService.interpolate_targets(basemesh)
for o in [basemesh, *basemesh.children]:
    if o.type == "MESH" and o.data.shape_keys:
        print("KEYS", o.name, [k.name for k in o.data.shape_keys.key_blocks])
bpy.context.view_layer.update()

# head position: the eye vertex groups or the top of the body
coords = [basemesh.matrix_world @ v.co for v in basemesh.data.vertices]
top = max(c.z for c in coords)
eye = Vector((0, 0, 0))
eyes = [o for o in basemesh.children if "high-poly" in o.name]
if eyes:
    vs = [eyes[0].matrix_world @ v.co for v in eyes[0].data.vertices]
    eye = sum(vs, Vector()) / len(vs)
print("TOP", top, "EYE", eye)

scene = bpy.context.scene
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
cam.data.lens = 85
# MakeHuman faces -Y; camera in front, slightly above eye level, framing head and shoulders
cam.location = (eye.x, eye.y - 0.95, eye.z + 0.02)
cam.rotation_euler = (math.radians(89), 0, 0)
scene.camera = cam

def area(name, loc, energy, size, color=(1, 1, 1)):
    l = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
    l.data.energy, l.data.size, l.data.color = energy, size, color
    l.location = loc
    d = eye - Vector(loc)
    l.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(l)
area("key", (eye.x - 0.6, eye.y - 0.8, eye.z + 0.4), 30, 0.8, (1, 0.96, 0.9))
area("fill", (eye.x + 0.7, eye.y - 0.6, eye.z + 0.1), 8, 1.0, (0.9, 0.95, 1))
area("rim", (eye.x + 0.3, eye.y + 0.6, eye.z + 0.4), 30, 0.6)
world = bpy.data.worlds.new("room") if not scene.world else scene.world
scene.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs[0].default_value = (0.32, 0.30, 0.28, 1)
bg.inputs[1].default_value = 0.35

for eng in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
    try:
        scene.render.engine = eng
        break
    except TypeError:
        pass
scene.render.resolution_x, scene.render.resolution_y = 640, 480
scene.render.fps = 30
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("SAVED", OUT, scene.render.engine)
scene.view_settings.view_transform = "AgX"
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass
bpy.ops.wm.save_as_mainfile(filepath=OUT)
