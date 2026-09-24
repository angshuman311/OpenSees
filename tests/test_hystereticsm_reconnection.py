try:
    import opensees as ops
except ModuleNotFoundError:
    import openseespy.opensees as ops


POSITIVE_ENVELOPE = (
    170.0908879721198, 0.0008256150522648083,
    806.5664389032247, 0.3299875444832722,
    68.03635518884792, 0.33779606261927597,
    0.0, 0.33794004766395014,
)
NEGATIVE_ENVELOPE = (
    -170.0908879721198, -0.0008256150522648083,
    -806.5664389032247, -0.3299875444832722,
    -68.03635518884792, -0.33779606261927597,
    0.0, -0.33794004766395014,
)
YIELD_ROTATION = POSITIVE_ENVELOPE[1]
RETAINED_MAXIMUM = 0.0009391377421725002
COMMITTED_DEFORMATIONS = (
    RETAINED_MAXIMUM,
    0.0008689933708300856,
    -0.0004230444166321932,
)
EPSILON = 1.0e-12


def advance(increment):
    ops.integrator("DisplacementControl", 2, 2, increment)
    assert ops.analyze(1) == 0


def deformation():
    return ops.nodeDisp(2, 2)


def force():
    return ops.eleResponse(1, "material", 1, "stress")[0]


def tangent():
    return ops.eleResponse(1, "material", 1, "tangent")[0]


def build_model():
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 0, 1)
    ops.uniaxialMaterial(
        "HystereticSM", 1,
        "-posEnv", *POSITIVE_ENVELOPE,
        "-negEnv", *NEGATIVE_ENVELOPE,
        "-pinch", 0.08, 0.98,
        "-damage", 0.0, 0.0,
        "-beta", 0.15,
        "-degEnv", 0.0, 0.0,
    )
    ops.element("zeroLength", 1, 1, 2, "-mat", 1, "-dir", 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, 1.0, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormUnbalance", 1.0e-8, 100)
    ops.algorithm("Newton")
    ops.integrator("DisplacementControl", 2, 2, EPSILON)
    ops.analysis("Static")


def reconnection_residual(sign):
    build_model()

    current = 0.0
    for target in COMMITTED_DEFORMATIONS:
        target = sign * target
        advance(target - current)
        current = target

    retained_extreme = sign * RETAINED_MAXIMUM
    inside = retained_extreme - sign * EPSILON
    advance(inside - current)
    force_inside = force()
    tangent_inside = tangent()

    advance(retained_extreme - inside)
    force_at_extreme = force()
    ops.wipe()

    return (
        force_at_extreme
        - force_inside
        - tangent_inside * (retained_extreme - inside)
    )


def partial_unloading_reload_residual(sign):
    """Load past yield, unload a little (the force keeps its sign), then reload
    by a very small increment. The force must continue from the committed
    state along the committed tangent; a reload line anchored at zero force
    jumps instead."""
    build_model()

    advance(sign * 2.0 * YIELD_ROTATION)
    advance(-sign * 0.1 * YIELD_ROTATION)
    force_before = force()
    tangent_before = tangent()

    increment = sign * 1.0e-7 * YIELD_ROTATION
    advance(increment)
    force_after = force()
    ops.wipe()

    return force_after - force_before - tangent_before * increment


def zero_crossing_split_residual(sign):
    """Load past yield, reverse past yield on the other side (the force changes
    sign), then reload towards the retained extreme: once with one increment
    that crosses the zero-force point, once with a committed step exactly at
    that point, and once with a committed step before it (the force still of
    the reversed sign). The final force must be the same on the three paths,
    so the reload line must be anchored at the zero-force point of the
    reversal, not at the last committed state."""
    def excursion():
        build_model()
        advance(sign * 2.0 * YIELD_ROTATION)
        advance(-sign * 3.5 * YIELD_ROTATION)

    # the zero-force point of the reload, projected from the reversal state
    # along the unloading stiffness (read from a probe step)
    excursion()
    strain_reversal = deformation()
    force_reversal = force()
    advance(sign * 1.0e-9 * YIELD_ROTATION)
    zero = strain_reversal - force_reversal / tangent()
    target = zero + sign * 0.5 * YIELD_ROTATION      # on the reload line, before the pinching branch

    forces = []
    for committed_steps in ((), (zero,), (zero - sign * 0.3 * YIELD_ROTATION,)):
        excursion()
        for step in committed_steps:
            advance(step - deformation())
        advance(target - deformation())
        forces.append(force())
    ops.wipe()

    return max(forces) - min(forces)


def test_positive_reload_reconnects_continuously():
    assert abs(reconnection_residual(1.0)) < 1.0e-9


def test_negative_reload_reconnects_continuously():
    assert abs(reconnection_residual(-1.0)) < 1.0e-9


def test_positive_reload_after_partial_unloading_is_continuous():
    assert abs(partial_unloading_reload_residual(1.0)) < 1.0e-9


def test_negative_reload_after_partial_unloading_is_continuous():
    assert abs(partial_unloading_reload_residual(-1.0)) < 1.0e-9


def test_positive_reload_across_zero_force_is_step_size_independent():
    assert abs(zero_crossing_split_residual(1.0)) < 1.0e-9


def test_negative_reload_across_zero_force_is_step_size_independent():
    assert abs(zero_crossing_split_residual(-1.0)) < 1.0e-9
