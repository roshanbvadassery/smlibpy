"""STEP (ISO 10303-21, AP214) writer for exact B-rep geometry.

Planes, cylinders, cones, spheres, tori, lines, circles and ellipses are written as
their analytic STEP entities, everything else as (rational) B-spline curves and
surfaces with the kernel's exact control points and knots. Solids become
MANIFOLD_SOLID_BREPs (BREP_WITH_VOIDS for hollow ones), sheets become open shells.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np

from ._kernel import ext, k
from .geometry import Vector

__all__ = ["write_step"]

_UNITS = {
    "mm": "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))",
    "m": "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.))",
    "cm": "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.CENTI.,.METRE.))",
    "um": "(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MICRO.,.METRE.))",
}
_INCH = 25.4
#: millimeters per declared unit
_MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "um": 0.001, "inch": _INCH, "in": _INCH}


def _r(v: float) -> str:
    """A STEP REAL literal."""
    v = float(v)
    if not math.isfinite(v):
        raise ValueError("non-finite number in STEP output")
    if v == 0.0:
        return "0."
    s = repr(v)
    if "e" in s or "E" in s:
        mant, exp = re.split("[eE]", s)
        if "." not in mant:
            mant += "."
        return f"{mant}E{int(exp)}"
    if "." not in s:
        s += "."
    return s


def _s(text: str | None) -> str:
    """A STEP STRING literal (ASCII; quotes doubled)."""
    t = (text or "").encode("ascii", "replace").decode("ascii").replace("'", "''")
    return f"'{t}'"


def _color(c: Any) -> tuple[float, float, float] | None:
    if c is None:
        return None
    if isinstance(c, str):
        from .colors import to_rgb

        return to_rgb(c)
    r, g, b = (float(x) for x in list(c)[:3])
    if max(r, g, b) > 1.0:
        r, g, b = r / 255, g / 255, b / 255
    return (r, g, b)


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._points: dict[tuple[float, float, float], int] = {}
        self._dirs: dict[tuple[float, float, float], int] = {}

    def add(self, text: str) -> int:
        self.lines.append(text)
        return len(self.lines)

    def point(self, p: Sequence[float]) -> int:
        key = (float(p[0]), float(p[1]), float(p[2]))
        if key not in self._points:
            self._points[key] = self.add(f"CARTESIAN_POINT('',({_r(key[0])},{_r(key[1])},{_r(key[2])}))")
        return self._points[key]

    def direction(self, d: Sequence[float]) -> int:
        v = np.asarray(d, dtype=float)
        v = v / np.linalg.norm(v)
        key = (float(v[0]), float(v[1]), float(v[2]))
        if key not in self._dirs:
            self._dirs[key] = self.add(f"DIRECTION('',({_r(key[0])},{_r(key[1])},{_r(key[2])}))")
        return self._dirs[key]

    def axis2(self, origin: Sequence[float], z: Sequence[float], x: Sequence[float]) -> int:
        return self.add(f"AXIS2_PLACEMENT_3D('',#{self.point(origin)},#{self.direction(z)},#{self.direction(x)})")

    def text(self, header: list[str]) -> str:
        out = ["ISO-10303-21;", "HEADER;", *header, "ENDSEC;", "DATA;"]
        out.extend(f"#{i}={line};" for i, line in enumerate(self.lines, 1))
        out += ["ENDSEC;", "END-ISO-10303-21;", ""]
        return "\n".join(out)


def _mults(knots: Sequence[float]) -> tuple[list[float], list[int]]:
    uniq: list[float] = []
    mult: list[int] = []
    for kv in knots:
        if uniq and abs(kv - uniq[-1]) <= 1e-14 * max(1.0, abs(kv)):
            mult[-1] += 1
        else:
            uniq.append(float(kv))
            mult.append(1)
    return uniq, mult


def _ints(v: Iterable[int]) -> str:
    return "(" + ",".join(str(int(x)) for x in v) + ")"


def _reals(v: Iterable[float]) -> str:
    return "(" + ",".join(_r(x) for x in v) + ")"


def _orthonormal(z: Sequence[float], x: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    zz = np.asarray(z, dtype=float)
    zz /= np.linalg.norm(zz)
    xx = np.asarray(x, dtype=float)
    xx = xx - zz * xx.dot(zz)
    xx /= np.linalg.norm(xx)
    return zz, xx


class _BodyWriter:
    """Writes the topology of one kernel Brep."""

    def __init__(self, w: _Writer, brep: Any):
        self.w = w
        self.brep = brep
        self.dump = ext.brep_dump(brep, True)
        for f in self.dump["faces"]:
            srf = f["surface"]
            if srf["kind"] not in ("plane", "cone", "sphere", "torus") and "poles" not in srf:
                raise _NeedsNurbs()
        self.kfaces = brep.faces()
        self.kedges = brep.edges()
        self.vertex_ids: dict[int, int] = {}
        self.edge_ids: dict[int, int] = {}

    # -- geometry -------------------------------------------------------------
    def bspline_curve(self, c: dict[str, Any]) -> int:
        w = self.w
        pts = ",".join(f"#{w.point(p)}" for p in c["poles"])
        uniq, mult = _mults(c["knots"])
        deg = c["degree"]
        if c["weights"] is None:
            return w.add(f"B_SPLINE_CURVE_WITH_KNOTS('',{deg},({pts}),.UNSPECIFIED.,.F.,.F.,{_ints(mult)},{_reals(uniq)},.UNSPECIFIED.)")
        return w.add(
            "(BOUNDED_CURVE() B_SPLINE_CURVE({deg},({pts}),.UNSPECIFIED.,.F.,.F.) "
            "B_SPLINE_CURVE_WITH_KNOTS({m},{kn},.UNSPECIFIED.) CURVE() GEOMETRIC_REPRESENTATION_ITEM() "
            "RATIONAL_B_SPLINE_CURVE({ws}) REPRESENTATION_ITEM(''))".format(
                deg=deg, pts=pts, m=_ints(mult), kn=_reals(uniq), ws=_reals(c["weights"])))

    def curve(self, idx: int) -> tuple[int, bool]:
        """STEP curve for edge ``idx`` and whether the edge runs along its parameterisation."""
        e = self.dump["edges"][idx]
        c = e["curve"]
        w = self.w
        kind = c["kind"]
        t0, t1 = e["interval"]
        if kind == "line":
            ke = self.kedges[idx]
            a = np.array(ke.curve().evaluate(t0))
            b = np.array(ke.curve().evaluate(t1))
            d = b - a
            L = float(np.linalg.norm(d))
            vec = w.add(f"VECTOR('',#{w.direction(d / L)},{_r(L)})")
            return w.add(f"LINE('',#{w.point(a)},#{vec})"), True
        if kind in ("circle", "ellipse"):
            pl = c["placement"]
            z, x = _orthonormal(pl["z_dir"], pl["x_dir"])
            ax = w.axis2(pl["origin"], z, x)
            if kind == "circle":
                ent = w.add(f"CIRCLE('',#{ax},{_r(c['radius'])})")
            else:
                r1, r2 = c["radius1"], c["radius2"]
                if r1 >= r2:
                    ent = w.add(f"ELLIPSE('',#{ax},{_r(r1)},{_r(r2)})")
                else:  # STEP wants semi_axis_1 >= semi_axis_2: rotate the frame by 90 degrees
                    y = np.cross(z, x)
                    ax = w.axis2(pl["origin"], z, y)
                    ent = w.add(f"ELLIPSE('',#{ax},{_r(r2)},{_r(r1)})")
            # sense: STEP conics run counterclockwise about their axis
            ke = self.kedges[idx]
            tm = 0.5 * (t0 + t1)
            p, dp = (np.array(v) for v in ext.curve_eval(ke.curve(), tm, 1))
            ccw = np.cross(z, p - np.array(pl["origin"]))
            return ent, bool(np.dot(dp, ccw) > 0)
        return self.bspline_curve(c), True

    def surface(self, idx: int) -> tuple[int, np.ndarray | None]:
        """STEP surface for face ``idx`` and, for analytic surfaces, a function of the STEP normal."""
        s = self.dump["faces"][idx]["surface"]
        w = self.w
        kind = s["kind"]
        pl = s.get("placement")
        if kind == "plane":
            z, x = _orthonormal(pl["z_dir"], pl["x_dir"])
            return w.add(f"PLANE('',#{w.axis2(pl['origin'], z, x)})"), ("plane", z)
        if kind == "cone":
            z, x = _orthonormal(pl["z_dir"], pl["x_dir"])
            semi = float(s.get("semi_angle_deg", 0.0))
            o = np.array(pl["origin"], dtype=float)
            r = float(s["radius"])
            if abs(semi) < 1e-12:
                return w.add(f"CYLINDRICAL_SURFACE('',#{w.axis2(o, z, x)},{_r(r)})"), ("cylinder", o, z)
            if semi < 0:
                z, semi = -z, -semi
            ang = math.radians(semi)
            return (w.add(f"CONICAL_SURFACE('',#{w.axis2(o, z, x)},{_r(r)},{_r(ang)})"),
                    ("cone", o, z, r, ang))
        if kind == "sphere":
            z, x = _orthonormal(pl["z_dir"], pl["x_dir"])
            o = np.array(pl["origin"], dtype=float)
            return w.add(f"SPHERICAL_SURFACE('',#{w.axis2(o, z, x)},{_r(s['radius'])})"), ("sphere", o)
        if kind == "torus":
            z, x = _orthonormal(pl["z_dir"], pl["x_dir"])
            o = np.array(pl["origin"], dtype=float)
            R, r = float(s["major_radius"]), float(s["minor_radius"])
            return (w.add(f"TOROIDAL_SURFACE('',#{w.axis2(o, z, x)},{_r(R)},{_r(r)})"), ("torus", o, z, R))
        return self.bspline_surface(s), None

    def bspline_surface(self, s: dict[str, Any]) -> int:
        w = self.w
        uu, um = _mults(s["u_knots"])
        vu, vm = _mults(s["v_knots"])
        nu = len(s["u_knots"]) - s["u_degree"] - 1
        nv = len(s["v_knots"]) - s["v_degree"] - 1
        poles = s["poles"]
        rows = []
        for i in range(nu):
            rows.append("(" + ",".join(f"#{w.point(poles[i * nv + j])}" for j in range(nv)) + ")")
        net = "(" + ",".join(rows) + ")"
        if s["weights"] is None:
            return w.add(f"B_SPLINE_SURFACE_WITH_KNOTS('',{s['u_degree']},{s['v_degree']},{net},.UNSPECIFIED.,"
                         f".F.,.F.,.F.,{_ints(um)},{_ints(vm)},{_reals(uu)},{_reals(vu)},.UNSPECIFIED.)")
        wts = s["weights"]
        wrows = "(" + ",".join("(" + ",".join(_r(wts[i * nv + j]) for j in range(nv)) + ")" for i in range(nu)) + ")"
        return w.add(
            f"(BOUNDED_SURFACE() B_SPLINE_SURFACE({s['u_degree']},{s['v_degree']},{net},.UNSPECIFIED.,.F.,.F.,.F.) "
            f"B_SPLINE_SURFACE_WITH_KNOTS({_ints(um)},{_ints(vm)},{_reals(uu)},{_reals(vu)},.UNSPECIFIED.) "
            f"GEOMETRIC_REPRESENTATION_ITEM() RATIONAL_B_SPLINE_SURFACE({wrows}) REPRESENTATION_ITEM('') SURFACE())")

    # -- topology -------------------------------------------------------------
    def vertex(self, idx: int) -> int:
        if idx not in self.vertex_ids:
            self.vertex_ids[idx] = self.w.add(f"VERTEX_POINT('',#{self.w.point(self.dump['vertices'][idx])})")
        return self.vertex_ids[idx]

    def edge(self, idx: int) -> int:
        if idx not in self.edge_ids:
            e = self.dump["edges"][idx]
            crv, sense = self.curve(idx)
            v0, v1 = self.vertex(e["v0"]), self.vertex(e["v1"])
            if sense:
                self.edge_ids[idx] = self.w.add(f"EDGE_CURVE('',#{v0},#{v1},#{crv},.T.)")
            else:
                self.edge_ids[idx] = self.w.add(f"EDGE_CURVE('',#{v0},#{v1},#{crv},.F.)")
        return self.edge_ids[idx]

    def face(self, idx: int, use: dict[str, Any]) -> int:
        """ADVANCED_FACE for face ``idx`` seen from faceuse ``use`` (its normal is the face normal)."""
        w = self.w
        srf, normal_fn = self.surface(idx)
        bounds = []
        for loop in use["loops"]:
            if loop["vertex"] is not None and not loop["edges"]:
                lp = w.add(f"VERTEX_LOOP('',#{self.vertex(loop['vertex'])})")
            else:
                oes = []
                for eidx, forward in loop["edges"]:
                    flag = ".T." if forward else ".F."
                    oe = w.add(f"ORIENTED_EDGE('',*,*,#{self.edge(eidx)},{flag})")
                    oes.append(f"#{oe}")
                lp = w.add(f"EDGE_LOOP('',({','.join(oes)}))")
            kind = "FACE_OUTER_BOUND" if loop["outer"] else "FACE_BOUND"
            bound = w.add(f"{kind}('',#{lp},.T.)")
            bounds.append(f"#{bound}")
        same = self._same_sense(idx, use, normal_fn)
        return w.add(f"ADVANCED_FACE('',({','.join(bounds)}),#{srf},{'.T.' if same else '.F.'})")

    def _same_sense(self, idx: int, use: dict[str, Any], normal_fn: Any) -> bool:
        # The faceuse normal is the kernel's natural normal when the faceuse is SAME.
        if normal_fn is None:  # exported with the kernel's own parameterisation
            return bool(use["same"])
        kf = self.kfaces[idx]
        kind = normal_fn[0]
        try:
            p = np.array(kf.interior_point())
            _, uvs, _ = k.surface_closest_point(kf.surface(), tuple(p))
            u, v = uvs[0].x, uvs[0].y
        except Exception:
            # The kernel finds no interior point on some faces (slivers, faces with many holes).
            # These analytic surfaces have one sense everywhere, so any point of the surface will do.
            (u0, v0), (u1, v1) = kf.uv_domain()
            u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            p = np.array(kf.point_at_uv(u, v))
        if kind in ("cone", "torus"):
            o, z = normal_fn[1], normal_fn[2]
            d = p - o
            if np.linalg.norm(d - z * d.dot(z)) < 1e-9 * max(1.0, np.linalg.norm(d)):
                # The point is on the axis (the apex of a cone): no radial direction there.
                # The sense is a property of the whole surface, so any other point will do.
                (u0, v0), (u1, v1) = kf.uv_domain()
                u, v = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
                p = np.array(kf.point_at_uv(u, v))
        n_kernel = np.array(kf.normal_at_uv(u, v))
        n_face = n_kernel if use["same"] else -n_kernel
        if kind == "plane":
            n_step = normal_fn[1]
        elif kind == "cylinder":
            o, z = normal_fn[1], normal_fn[2]
            d = p - o
            n_step = d - z * d.dot(z)
        elif kind == "cone":
            o, z, r, ang = normal_fn[1:]
            d = p - o
            radial = d - z * d.dot(z)
            radial /= np.linalg.norm(radial)
            n_step = radial * math.cos(ang) - z * math.sin(ang)
        elif kind == "sphere":
            n_step = p - normal_fn[1]
        else:  # torus
            o, z, R = normal_fn[1:]
            d = p - o
            radial = d - z * d.dot(z)
            radial /= np.linalg.norm(radial)
            n_step = p - (o + radial * R)
        return bool(np.dot(n_face, n_step) > 0)

    def solids(self, name: str) -> list[int]:
        """MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS items, one per material region."""
        d = self.dump
        items = []
        shell_faces: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for fi, f in enumerate(d["faces"]):
            for ui, use in enumerate(f["uses"]):
                if use is None or use["shell"] is None:
                    continue
                # The faceuse in a region's shell faces into that region; its mate faces out.
                mate = f["uses"][1 - ui]
                shell_faces.setdefault(use["shell"], []).append((fi, mate))
        for region in d["regions"]:
            if region["infinite"] or region["void"]:
                continue
            shells = []
            for si in region["shells"]:
                faces = [self.face(fi, mate) for fi, mate in shell_faces.get(si, [])]
                shells.append(self.w.add(f"CLOSED_SHELL('',({','.join(f'#{x}' for x in faces)}))"))
            if not shells:
                continue
            if len(shells) == 1:
                items.append(self.w.add(f"MANIFOLD_SOLID_BREP({_s(name)},#{shells[0]})"))
            else:
                voids = [self.w.add(f"ORIENTED_CLOSED_SHELL('',*,#{s},.F.)") for s in shells[1:]]
                items.append(self.w.add(f"BREP_WITH_VOIDS({_s(name)},#{shells[0]},({','.join(f'#{v}' for v in voids)}))"))
        return items

    def sheets(self, name: str) -> list[int]:
        """An OPEN_SHELL of all faces (seen from their natural side)."""
        faces = []
        for fi, f in enumerate(self.dump["faces"]):
            faces.append(self.face(fi, f["uses"][0]))
        if not faces:
            return []
        shell = self.w.add(f"OPEN_SHELL('',({','.join(f'#{x}' for x in faces)}))")
        return [self.w.add(f"SHELL_BASED_SURFACE_MODEL({_s(name)},(#{shell}))")]


class _NeedsNurbs(Exception):
    pass


def write_step(bodies: Sequence[Any], path: str, *, unit: str = "mm", name: str | None = None,
               author: str = "", organization: str = "") -> None:
    """Write bodies (parts and sheets) to a STEP file.

    The model is in millimeters; with another ``unit`` the coordinates are converted to
    it, so the file holds the same part measured in that unit.
    """
    from .shapes import Part, Shell, _Body

    if unit not in _MM_PER_UNIT:
        raise ValueError(f"unknown unit {unit!r}; use mm, cm, m, um or inch")
    if _MM_PER_UNIT[unit] != 1.0:
        f = 1.0 / _MM_PER_UNIT[unit]
        bodies = [b.scale(f) if isinstance(b, _Body) else b for b in bodies]
    w = _Writer()
    product = name or _product_name(path)
    app = w.add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    w.add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pctx = w.add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    prod = w.add(f"PRODUCT({_s(product)},{_s(product)},'',(#{pctx}))")
    w.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod}))")
    form = w.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod})")
    dctx = w.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pdef = w.add(f"PRODUCT_DEFINITION('design','',#{form},#{dctx})")
    pshape = w.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pdef})")

    if unit == "inch" or unit == "in":
        mm = w.add("(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.))")
        lm = w.add(f"LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE({_r(_INCH)}),#{mm})")
        dim = w.add("DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.)")
        length = w.add(f"(CONVERSION_BASED_UNIT('INCH',#{lm}) LENGTH_UNIT() NAMED_UNIT(#{dim}))")
    elif unit in _UNITS:
        length = w.add(_UNITS[unit])
    else:
        raise ValueError(f"unknown unit {unit!r}; use mm, cm, m, um or inch")
    angle = w.add("(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.))")
    solid_angle = w.add("(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT())")
    unc = w.add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{length},'distance_accuracy_value','confusion accuracy')")
    ctx = w.add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc})) "
                f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) "
                f"REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY'))")

    origin = w.axis2((0, 0, 0), (0, 0, 1), (1, 0, 0))
    solid_items: list[int] = []
    sheet_items: list[int] = []
    styled: list[tuple[int, tuple[float, float, float]]] = []
    for i, body in enumerate(bodies):
        if not isinstance(body, _Body):
            raise TypeError(f"cannot write a {type(body).__name__} to STEP")
        label = body.name or (product if len(bodies) == 1 else f"{product}_{i + 1}")
        brep = body._brep
        try:
            bw = _BodyWriter(w, brep)
            items = bw.solids(label) if isinstance(body, Part) else bw.sheets(label)
        except _NeedsNurbs:
            nb = body.to_nurbs()
            bw = _BodyWriter(w, nb._brep)
            items = bw.solids(label) if isinstance(body, Part) else bw.sheets(label)
        if isinstance(body, Part):
            solid_items += items
        else:
            sheet_items += items
        rgb = _color(body.color)
        if rgb is not None:
            styled += [(it, rgb) for it in items]

    reps = []
    if solid_items:
        reps.append(w.add(f"ADVANCED_BREP_SHAPE_REPRESENTATION({_s(product)},(#{origin},{','.join(f'#{x}' for x in solid_items)}),#{ctx})"))
    if sheet_items:
        reps.append(w.add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION({_s(product)},(#{origin},{','.join(f'#{x}' for x in sheet_items)}),#{ctx})"))
    if not reps:
        raise ValueError("nothing to write")
    w.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pshape},#{reps[0]})")
    for extra in reps[1:]:
        w.add(f"SHAPE_REPRESENTATION_RELATIONSHIP('','',#{reps[0]},#{extra})")

    if styled:
        style_ids = []
        for item, (r, g, b) in styled:
            col = w.add(f"COLOUR_RGB('',{_r(r)},{_r(g)},{_r(b)})")
            fill = w.add(f"FILL_AREA_STYLE_COLOUR('',#{col})")
            fas = w.add(f"FILL_AREA_STYLE('',(#{fill}))")
            ssfa = w.add(f"SURFACE_STYLE_FILL_AREA(#{fas})")
            sss = w.add(f"SURFACE_SIDE_STYLE('',(#{ssfa}))")
            ssu = w.add(f"SURFACE_STYLE_USAGE(.BOTH.,#{sss})")
            psa = w.add(f"PRESENTATION_STYLE_ASSIGNMENT((#{ssu}))")
            style_ids.append(w.add(f"STYLED_ITEM('color',(#{psa}),#{item})"))
        w.add(f"MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',({','.join(f'#{s}' for s in style_ids)}),#{ctx})")

    stamp = _dt.datetime.now().replace(microsecond=0).isoformat()
    header = [
        "FILE_DESCRIPTION(('smlibpy model'),'2;1');",
        f"FILE_NAME({_s(path.split('/')[-1])},{_s(stamp)},({_s(author)}),({_s(organization)}),"
        f"'smlibpy','smlibpy (SMLib kernel)','');",
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));",
    ]
    with open(path, "w", encoding="ascii") as fh:
        fh.write(w.text(header))


def _product_name(path: str) -> str:
    base = path.replace("\\", "/").split("/")[-1]
    return base.rsplit(".", 1)[0] or "part"
