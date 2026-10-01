from silent_signal.analyze import calibrated_states


def test_calibration_handles_eyes_that_look_half_closed():
    # old low-res film: open eyes already read 0.44, blinks reach 0.75
    values = [0.44] * 40 + [0.75] * 4 + [0.44] * 40 + [0.74] * 3 + [0.45] * 20
    found = [True] * len(values)
    states = calibrated_states(values, found)
    assert sum(states) == 7                     # exactly the 7 closed frames
    fixed = [v >= 0.5 for v in values]          # the old fixed threshold also gets these right here...
    assert sum(fixed) == 7
    values_dim = [v + 0.1 for v in values]      # ...but not when the whole clip reads 0.1 higher
    assert sum(v >= 0.5 for v in values_dim) > 90
    assert sum(calibrated_states(values_dim, found)) == 7


def test_frames_without_a_face_keep_the_previous_state():
    values = [0.1] * 20 + [0.9, 0.9, 0.0, 0.9] + [0.1] * 20
    found = [True] * 20 + [True, True, False, True] + [True] * 20
    s = calibrated_states(values, found)
    assert s[20:24] == [True, True, True, True]
