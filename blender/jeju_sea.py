# -*- coding: utf-8 -*-
"""
제주도 DEM + 물결치는 바다(Ocean 모디파이어) + 하늘을 가로지르는 태양  (Blender 스크립트)

만들어지는 장면
  * 제주도 DEM(고도 자료)으로 만든 지형 (한라산 포함)
  * Ocean 모디파이어로 실제처럼 일렁이는 바다 (시간에 따라 파도가 움직임, 거품 포함)
  * 실제 태양 위치 계산(날짜·시각·제주 위도/경도)으로 하늘을 이동하는 태양
  * 태양이 바다 표면에 반사되어 반짝이는 "윤슬" (Cycles 렌더)

사용법
  1) Blender(4.2 이상, 5.2에서 확인)를 열고 [Scripting] 탭에서 이 파일을 열어 [Run Script]
     또는 명령줄 (Windows 예시):
        blender -b -P jeju_sea.py -- --dem C:/data/N33E126.hgt --still 150
        blender -b -P jeju_sea.py -- --dem C:/data/N33E126.hgt --view sunrise --anim --device gpu
     옵션: --view sunset|sunrise  --still 프레임  --anim  --save 파일.blend  --res 1280x720
           --samples 64  --device cpu|gpu  --out 결과폴더
     (렌더 시간 참고: 1920x1080, 96샘플 한 장에 CPU로 약 20초 / 가짜 DEM 기준)
  2) DEM 파일
     - SRTM:  N33E126.hgt   (제주도 전체가 이 타일 한 장에 들어갑니다)
     - ESRI ASCII 격자(.asc), 16비트 PNG/TIFF 높이 이미지(.png/.tif, --bbox 와 --zrange 필요)
     - 파일을 지정하지 않으면 제주도 모양을 흉내 낸 "가짜 DEM"으로 동작을 확인합니다.
       (실제 지형이 아니므로 반드시 진짜 DEM 을 넣어 주세요)
  3) F12(한 장) 또는 Ctrl+F12(애니메이션) 로 렌더. 타임라인을 재생하면 태양과 파도가 움직입니다.

주의: API 키·비밀번호 같은 값은 이 파일에 쓰지 않습니다. (DEM 다운로드 계정 정보 등)
"""
import argparse
import datetime
import math
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Vector

# =====================================================================
# 설정 (여기만 고치면 됩니다)
# =====================================================================
DEM_PATH = ""                                  # 비워 두면 가짜 제주도 DEM 사용
DEM_IMAGE_BBOX = (126.10, 33.10, 127.00, 33.60)  # (경도최소, 위도최소, 경도최대, 위도최대): 이미지형 DEM 일 때만
DEM_IMAGE_ZRANGE = None                        # 예: (0, 1950) 16비트 이미지의 0~1 값을 미터로 바꿀 범위
JEJU_BBOX = (126.10, 33.10, 127.00, 33.60)     # 잘라 쓸 범위(제주도 전체)
CENTER_LONLAT = (126.55, 33.38)                # 장면 원점(제주도 중심)

MAX_TERRAIN_VERTS = 650_000                    # 지형 점 개수 상한 (넘으면 자동으로 성김)
Z_EXAG = 1.5                                   # 지형 높이 과장 배수

# --- 바다 ---
OCEAN_TILE_M = 800.0       # 파도 무늬가 반복되는 한 변 길이(m). 이음매 없이 이어집니다.
OCEAN_RESOLUTION = 20      # 파도 계산 해상도
OCEAN_WIND = 18.0          # 바람 세기 (클수록 큰 파도)
OCEAN_WAVE_SCALE = 1.3     # 파도 높이 배수
OCEAN_CHOPPINESS = 1.3     # 뾰족함(파도 마루가 날카로울수록 큼)
OCEAN_FOAM = 0.12          # 거품 양 (0이면 거품 없음)
OCEAN_SPEED = 1.0          # 파도 진행 속도 배수
OCEAN_SEGMENTS = 540       # 바다 메시 둘레 분할 수
OCEAN_MIN_CELL_M = 2.5     # 카메라 가까운 곳 격자 크기(m)
OCEAN_RADIUS_M = 120_000   # 바다 반지름(m)

# --- 태양 (실제 태양 위치 계산) ---
SUN_LATLON = (33.35, 126.40)                   # 위도, 경도
SUN_DATE = (2026, 6, 21)                       # 년, 월, 일 (하지: 해가 길고 서쪽 바다로 집니다)
UTC_OFFSET = 9.0                               # 한국 표준시
SUN_HOUR_START, SUN_HOUR_END = 16.0, 19.95     # 시각(시). 16:00 → 일몰 직전
SUN_ANGLE_DEG = 0.9                            # 태양 지름(작을수록 윤슬이 날카로움, 실제는 0.53)
SUN_DISC_DEG = 1.4                             # 하늘에 보이는 태양 원반 지름(도)
SUN_STRENGTH = 4.5
EXPOSURE_DROP = 2.8                            # 해가 높을 때 노출을 얼마나 낮출지(클수록 어둡게)

# --- 시간 / 카메라 ---
FRAME_START, FRAME_END, FPS = 1, 240, 24
CAM_LAT = 33.33            # 이 위도에서 서쪽 해안을 찾아 카메라를 둡니다
CAM_INLAND_M = 1200.0      # 해안선에서 안쪽으로 떨어진 거리
CAM_HEIGHT_M = 220.0       # 지면 위 높이
CAM_PITCH_DEG = 3.0        # 위(+)아래(-)
CAM_YAW_CENTER = 270.0     # 바라보는 방위(0=북, 90=동, 180=남, 270=서)
CAM_FOLLOW = 0.55          # 카메라가 태양 방위를 따라 천천히 도는 정도(0=고정)
CAM_LENS_MM = 22.0

# --- 렌더 ---
ENGINE = "CYCLES"
SAMPLES = 96
RES = (1920, 1080)
OUT_DIR = os.path.join(os.path.expanduser("~"), "jeju_sea_render")
# =====================================================================

# --- 구도 프리셋 (--view 로 선택) ---
#   sunset : 서쪽 해안에서 바다로 지는 해 (기본)
#   sunrise: 서쪽 바다 위에서 한라산을 보며, 한라산 뒤로 해가 떠오르는 장면
VIEW_PRESETS = {
    "sunset": {},
    "sunrise": dict(SUN_HOUR_START=5.0, SUN_HOUR_END=8.6, CAM_FIXED_LONLAT=(126.02, 33.36),
                    CAM_HEIGHT_M=650.0, CAM_PITCH_DEG=-0.5, CAM_YAW_CENTER=82.0, CAM_FOLLOW=0.5,
                    CAM_LENS_MM=26.0, EXPOSURE_DROP=2.4),
}
CAM_FIXED_LONLAT = None        # 지정하면 해안 탐색 대신 이 위치(바다 위)에 카메라를 둡니다

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(CENTER_LONLAT[1]))


# ---------------------------------------------------------------------
# DEM 읽기
# ---------------------------------------------------------------------
class DEM:
    """lons: 서→동(오름차순), lats: 북→남(내림차순), z[행=lats, 열=lons] (m)."""

    def __init__(self, lons, lats, z, metric=False, label=""):
        self.lons, self.lats = np.asarray(lons, np.float64), np.asarray(lats, np.float64)
        self.z = np.asarray(z, np.float32)
        self.metric = metric      # True 면 lons/lats 가 이미 미터(투영 좌표)
        self.label = label


def load_hgt(path):
    m = re.search(r"([NS])(\d{2})([EW])(\d{3})", os.path.basename(path).upper())
    if not m:
        raise ValueError("HGT 파일 이름이 N33E126.hgt 형식이어야 합니다: " + path)
    lat0 = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
    lon0 = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
    n = int(round(math.sqrt(os.path.getsize(path) / 2)))
    z = np.fromfile(path, dtype=">i2").reshape(n, n).astype(np.float32)
    z[z < -1000] = 0.0                                   # 자료 없음(-32768)
    lats = lat0 + 1.0 - np.arange(n) / (n - 1)
    lons = lon0 + np.arange(n) / (n - 1)
    return DEM(lons, lats, z, label="SRTM " + os.path.basename(path))


def load_asc(path):
    head, rows = {}, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0].lower() in (
                    "ncols", "nrows", "xllcorner", "yllcorner", "xllcenter", "yllcenter", "cellsize", "nodata_value"):
                head[parts[0].lower()] = float(parts[1])
            elif parts:
                rows.append(np.array(parts, dtype=np.float32))
    z = np.vstack(rows)
    nrows, ncols = z.shape
    cs = head["cellsize"]
    # xllcorner/yllcorner 는 왼쪽 아래 "모서리"라서 셀 중심은 반 칸 안쪽
    x0 = head["xllcorner"] + cs / 2 if "xllcorner" in head else head.get("xllcenter", 0.0)
    y0 = head["yllcorner"] + cs / 2 if "yllcorner" in head else head.get("yllcenter", 0.0)
    nod = head.get("nodata_value", -9999.0)
    z[z <= nod + 0.5] = 0.0
    lons = x0 + np.arange(ncols) * cs
    lats = y0 + (nrows - 1 - np.arange(nrows)) * cs
    return DEM(lons, lats, z, metric=cs > 0.1, label="ASC " + os.path.basename(path))


def load_image_dem(path, bbox, zrange):
    img = bpy.data.images.load(path)
    # 8/16비트 PNG·TIFF 는 색 보정 없이 원본 값으로 읽도록 Non-Color 로 지정.
    # 실수형(EXR, 32비트 TIFF)은 색공간을 바꾸면 값이 변하므로 그대로 읽는다.
    if not (path.lower().endswith(".exr") or img.depth >= 96):
        img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    a = px.reshape(h, w, 4)[:, :, 0][::-1].copy()        # 위쪽이 북쪽이 되도록 뒤집기
    if zrange:
        a = zrange[0] + a * (zrange[1] - zrange[0])
    elif a.max() <= 1.0001:
        raise ValueError("0~1 범위 이미지입니다. DEM_IMAGE_ZRANGE=(최저, 최고) 를 지정해 주세요.")
    lon_min, lat_min, lon_max, lat_max = bbox
    lons = np.linspace(lon_min, lon_max, w)
    lats = np.linspace(lat_max, lat_min, h)
    return DEM(lons, lats, a, label="IMAGE " + os.path.basename(path))


def synthetic_dem():
    """제주도 모양(동서 약 73km, 남북 약 31km, 한라산 1947m)을 흉내 낸 가짜 DEM."""
    lons = np.linspace(JEJU_BBOX[0], JEJU_BBOX[2], 1000)
    lats = np.linspace(JEJU_BBOX[3], JEJU_BBOX[1], 667)
    LON, LAT = np.meshgrid(lons, lats)
    xk = (LON - CENTER_LONLAT[0]) * M_PER_DEG_LON / 1000.0
    yk = (LAT - CENTER_LONLAT[1]) * M_PER_DEG_LAT / 1000.0
    rng = np.random.default_rng(7)
    th = np.arctan2(yk, xk)
    ex, ey = xk / 36.5, (yk - 0.12 * xk) / 15.2
    r = np.hypot(ex, ey) * (1 + 0.05 * np.sin(3 * th + 1.0) + 0.03 * np.sin(5 * th + 2.0) + 0.02 * np.sin(9 * th))
    base = 760.0 * np.clip(1 - r, 0, 1) ** 1.45
    d2 = (xk + 1.6) ** 2 + (yk + 1.9) ** 2
    hal = 1250.0 * np.exp(-(d2 / 4.6 ** 2) ** 1.25) - 90.0 * np.exp(-d2 / 0.45 ** 2)
    z = base + hal
    for _ in range(320):                                  # 오름(작은 화산) 흩뿌리기
        cx, cy = rng.uniform(-30, 30), rng.uniform(-11, 11)
        if math.hypot(cx / 36.5, cy / 15.2) > 0.82:
            continue
        rad, hgt = rng.uniform(0.35, 0.8), rng.uniform(40, 190)
        z += hgt * np.exp(-(((xk - cx) ** 2 + (yk - cy) ** 2) / rad ** 2))
    z += 6.0 * np.sin(xk * 2.3) * np.cos(yk * 1.9) + 4.0 * np.sin(xk * 5.1 + yk * 3.7)   # 잔물결
    land = np.clip((1.0 - r) * 14.0, 0, 1)                # 해안선 부근 부드럽게
    z = z * land
    sx, sy = (126.94 - CENTER_LONLAT[0]) * M_PER_DEG_LON / 1000.0, (33.458 - CENTER_LONLAT[1]) * M_PER_DEG_LAT / 1000.0
    cone = 185.0 * np.exp(-(((xk - sx) ** 2 + (yk - sy) ** 2) / 0.55 ** 2))   # 성산일출봉 자리
    z = np.maximum(z, cone)
    z[z < 1.0] = 0.0
    z *= 1947.0 / z.max()
    return DEM(lons, lats, z, label="가짜 제주도 DEM (실제 지형 아님)")


def load_dem(path, bbox=DEM_IMAGE_BBOX, zrange=DEM_IMAGE_ZRANGE):
    if not path:
        return synthetic_dem()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hgt":
        return load_hgt(path)
    if ext == ".asc":
        return load_asc(path)
    if ext in (".png", ".tif", ".tiff", ".exr"):
        return load_image_dem(path, bbox, zrange)
    raise ValueError("지원하지 않는 DEM 형식입니다: " + ext)


def local_axes(dem):
    """격자 가로(X)·세로(Y) 좌표를 장면 원점 기준 미터로."""
    if dem.metric:
        return dem.lons - dem.lons.mean(), dem.lats - dem.lats.mean()
    return ((dem.lons - CENTER_LONLAT[0]) * M_PER_DEG_LON,
            (dem.lats - CENTER_LONLAT[1]) * M_PER_DEG_LAT)


def crop_and_reduce(dem, bbox, max_verts):
    if not dem.metric:
        cj = np.where((dem.lons >= bbox[0]) & (dem.lons <= bbox[2]))[0]
        ci = np.where((dem.lats >= bbox[1]) & (dem.lats <= bbox[3]))[0]
        if len(cj) < 4 or len(ci) < 4:
            raise ValueError("DEM 이 제주도 범위(JEJU_BBOX)와 겹치지 않습니다.")
        dem = DEM(dem.lons[cj[0]:cj[-1] + 1], dem.lats[ci[0]:ci[-1] + 1],
                  dem.z[ci[0]:ci[-1] + 1, cj[0]:cj[-1] + 1], False, dem.label)
    rows, cols = dem.z.shape
    step = max(1, int(math.ceil(math.sqrt(rows * cols / max_verts))))
    if step > 1:
        r, c = rows // step, cols // step
        z = dem.z[:r * step, :c * step].reshape(r, step, c, step).mean(axis=(1, 3))
        lons = dem.lons[:c * step].reshape(c, step).mean(axis=1)
        lats = dem.lats[:r * step].reshape(r, step).mean(axis=1)
        dem = DEM(lons, lats, z, dem.metric, dem.label)
    return dem, step


def box_blur(a, radius, iters=3):
    for _ in range(iters):
        for axis in (0, 1):
            k = 2 * radius + 1
            pad = [(0, 0), (0, 0)]
            pad[axis] = (radius + 1, radius)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
            n = a.shape[axis]
            hi = np.take(c, np.arange(k, k + n), axis=axis)
            lo = np.take(c, np.arange(0, n), axis=axis)
            a = ((hi - lo) / k).astype(np.float32)
    return a


def add_seafloor(z, cell_m):
    """DEM 의 바다는 0m 로 비어 있으므로, 해안에서 멀어질수록 깊어지는 바닥을 만들어 줍니다."""
    land = (z > 0.5).astype(np.float32)
    f = box_blur(land, max(2, int(1200.0 / cell_m) // 2))
    depth = -60.0 * np.clip((0.5 - f) * 2.0, 0, 1) ** 0.8 - 0.4
    return np.where(land > 0, np.maximum(z, 0.4), depth).astype(np.float32)


# ---------------------------------------------------------------------
# 메시 만들기
# ---------------------------------------------------------------------
def make_mesh(name, verts, quads):
    me = bpy.data.meshes.new(name)
    n, m = len(verts), len(quads)
    me.vertices.add(n)
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, np.float32).ravel())
    me.loops.add(4 * m)
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(quads, np.int32).ravel())
    me.polygons.add(m)
    me.polygons.foreach_set("loop_start", np.arange(m, dtype=np.int32) * 4)
    try:
        me.polygons.foreach_set("loop_total", np.full(m, 4, np.int32))
    except Exception:
        pass
    me.update()
    me.validate()
    try:
        me.shade_smooth()
    except Exception:
        me.polygons.foreach_set("use_smooth", np.ones(m, bool))
    return me


def build_terrain(X, Y, zf):
    rows, cols = zf.shape
    GX, GY = np.meshgrid(X, Y)
    verts = np.stack([GX.ravel(), GY.ravel(), (zf * Z_EXAG).ravel()], axis=1)
    idx = np.arange(rows * cols).reshape(rows, cols)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], axis=-1).reshape(-1, 4)
    ob = bpy.data.objects.new("Jeju_Terrain", make_mesh("Jeju_Terrain", verts, quads))
    bpy.context.scene.collection.objects.link(ob)
    return ob


def ring_radii(r_in, r_out, nseg, min_cell):
    k = 2.0 * math.pi / nseg
    rs = [r_in]
    while rs[-1] < r_out:
        rs.append(rs[-1] + max(min_cell, rs[-1] * k))
    return np.array(rs)


def build_ocean(center, X, Y, zf):
    nseg = OCEAN_SEGMENTS
    rs = ring_radii(6.0, OCEAN_RADIUS_M, nseg, OCEAN_MIN_CELL_M)
    th = np.linspace(0, 2 * math.pi, nseg, endpoint=False)
    px = rs[:, None] * np.cos(th)[None, :]
    py = rs[:, None] * np.sin(th)[None, :]
    verts = np.stack([px.ravel(), py.ravel(), np.zeros(px.size)], axis=1)
    nr = len(rs)
    idx = np.arange(nr * nseg).reshape(nr, nseg)
    nxt = np.roll(idx, -1, axis=1)
    quads = np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], axis=-1).reshape(-1, 4)
    me = make_mesh("Ocean", verts, quads)

    # 점마다 "얕은 정도"를 계산해 해안 가까이는 옥색, 멀리는 짙은 파랑으로 칠하기 위한 값
    wx, wy = px.ravel() + center[0], py.ravel() + center[1]
    dx, dy = X[1] - X[0], Y[0] - Y[1]
    j = np.rint((wx - X[0]) / dx).astype(np.int64)
    i = np.rint((Y[0] - wy) / dy).astype(np.int64)
    inside = (i >= 0) & (i < zf.shape[0]) & (j >= 0) & (j < zf.shape[1])
    depth = np.full(wx.shape, 100.0, np.float32)
    depth[inside] = np.maximum(0.0, -zf[i[inside], j[inside]])
    shallow = np.exp(-depth / 10.0).astype(np.float32)
    attr = me.attributes.new("shallow", "FLOAT", "POINT")
    attr.data.foreach_set("value", shallow)

    ob = bpy.data.objects.new("Ocean", me)
    ob.location = (center[0], center[1], 0.0)
    bpy.context.scene.collection.objects.link(ob)

    mod = ob.modifiers.new("Ocean", "OCEAN")
    mod.geometry_mode = "DISPLACE"
    mod.spatial_size = int(OCEAN_TILE_M)
    mod.resolution = OCEAN_RESOLUTION
    mod.viewport_resolution = max(8, OCEAN_RESOLUTION // 2)
    mod.wind_velocity = OCEAN_WIND
    mod.wave_scale = OCEAN_WAVE_SCALE
    mod.choppiness = OCEAN_CHOPPINESS
    mod.use_foam = True
    mod.foam_layer_name = "foam"
    mod.foam_coverage = OCEAN_FOAM
    mod.random_seed = 3
    # 파도가 시간에 따라 움직이도록 (프레임 → 초)
    drv = mod.driver_add("time").driver
    drv.type = "SCRIPTED"
    drv.expression = "frame / %d * %.3f" % (FPS, OCEAN_SPEED)
    return ob


# ---------------------------------------------------------------------
# 재질
# ---------------------------------------------------------------------
def _node(nodes, kind, x, y, **props):
    n = nodes.new(kind)
    n.location = (x, y)
    for k, v in props.items():
        setattr(n, k, v)
    return n


def _set(node, name, value):
    s = node.inputs.get(name)
    if s is not None:
        s.default_value = value


def _new_material(name):
    mat = bpy.data.materials.new(name)
    if mat.node_tree is None:                                  # 5.x 는 처음부터 노드가 있음
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


def make_water_material():
    mat = _new_material("Water")
    nt = mat.node_tree
    nd = nt.nodes
    out = _node(nd, "ShaderNodeOutputMaterial", 1000, 0)
    bsdf = _node(nd, "ShaderNodeBsdfPrincipled", 700, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _set(bsdf, "IOR", 1.333)
    _set(bsdf, "Metallic", 0.0)

    # 색: 깊은 곳 짙은 청록 ↔ 얕은 곳 옥색
    shallow = _node(nd, "ShaderNodeAttribute", -400, 300, attribute_name="shallow")
    mix_depth = _node(nd, "ShaderNodeMix", -100, 250, data_type="RGBA")
    mix_depth.inputs[6].default_value = (0.004, 0.03, 0.055, 1)
    mix_depth.inputs[7].default_value = (0.02, 0.32, 0.36, 1)
    nt.links.new(shallow.outputs["Fac"], mix_depth.inputs["Factor"])

    # 거품: Ocean 모디파이어가 만든 foam 값 → 하얀색
    foam = _node(nd, "ShaderNodeAttribute", -400, -50, attribute_name="foam")
    foam_gain = _node(nd, "ShaderNodeMath", -200, -50, operation="MULTIPLY", use_clamp=True)
    foam_gain.inputs[1].default_value = 1.2
    nt.links.new(foam.outputs["Fac"], foam_gain.inputs[0])
    mix_foam = _node(nd, "ShaderNodeMix", 150, 150, data_type="RGBA")
    mix_foam.inputs[7].default_value = (0.9, 0.93, 0.95, 1)
    nt.links.new(foam_gain.outputs[0], mix_foam.inputs["Factor"])
    nt.links.new(mix_depth.outputs[2], mix_foam.inputs[6])
    nt.links.new(mix_foam.outputs[2], bsdf.inputs["Base Color"])

    rough = _node(nd, "ShaderNodeMath", 400, -150, operation="MULTIPLY_ADD")
    rough.inputs[1].default_value = 0.55
    rough.inputs[2].default_value = 0.035
    nt.links.new(foam_gain.outputs[0], rough.inputs[0])
    nt.links.new(rough.outputs[0], bsdf.inputs["Roughness"])

    # 잔물결(작은 요철) — 움직이는 4D 노이즈를 범프로 → 햇빛이 반짝이는 윤슬
    tc = _node(nd, "ShaderNodeTexCoord", -900, -400)
    mp = _node(nd, "ShaderNodeMapping", -700, -400)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    noise = _node(nd, "ShaderNodeTexNoise", -450, -400, noise_dimensions="4D")
    _set(noise, "Scale", 1.6)
    _set(noise, "Detail", 6.0)
    _set(noise, "Roughness", 0.6)
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    d = noise.inputs["W"].driver_add("default_value").driver
    d.type = "SCRIPTED"
    d.expression = "frame * 0.045"
    bump = _node(nd, "ShaderNodeBump", 300, -450)
    _set(bump, "Strength", 0.55)
    _set(bump, "Distance", 0.35)
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_terrain_material():
    mat = _new_material("Terrain")
    nt = mat.node_tree
    nd = nt.nodes
    out = _node(nd, "ShaderNodeOutputMaterial", 900, 0)
    bsdf = _node(nd, "ShaderNodeBsdfPrincipled", 650, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _set(bsdf, "Roughness", 0.92)

    geo = _node(nd, "ShaderNodeNewGeometry", -900, 100)
    sep = _node(nd, "ShaderNodeSeparateXYZ", -700, 200)
    nt.links.new(geo.outputs["Position"], sep.inputs[0])
    hnorm = _node(nd, "ShaderNodeMath", -500, 200, operation="DIVIDE", use_clamp=True)
    hnorm.inputs[1].default_value = 2000.0 * Z_EXAG          # 0~2000m → 0~1
    nt.links.new(sep.outputs["Z"], hnorm.inputs[0])

    ramp = _node(nd, "ShaderNodeValToRGB", -250, 250)
    stops = [(0.000, (0.09, 0.085, 0.08, 1)),     # 해안 현무암 (해발 0~2m)
             (0.0007, (0.12, 0.11, 0.09, 1)),
             (0.0025, (0.30, 0.30, 0.12, 1)),     # 해안 초지
             (0.020, (0.17, 0.27, 0.08, 1)),      # 들판·초지
             (0.200, (0.10, 0.22, 0.08, 1)),      # 숲
             (0.450, (0.07, 0.15, 0.07, 1)),      # 짙은 숲
             (0.750, (0.26, 0.23, 0.19, 1)),      # 한라산 윗부분
             (1.000, (0.45, 0.43, 0.41, 1))]      # 정상부 바위
    el = ramp.color_ramp.elements                          # 기본으로 2개가 있음
    el[0].position, el[0].color = stops[0]
    el[1].position, el[1].color = stops[1]
    for pos, col in stops[2:]:
        el.new(pos).color = col
    nt.links.new(hnorm.outputs[0], ramp.inputs["Fac"])

    # 색 얼룩
    tc = _node(nd, "ShaderNodeTexCoord", -900, -200)
    noise = _node(nd, "ShaderNodeTexNoise", -650, -200, noise_dimensions="3D")
    _set(noise, "Scale", 0.0009)
    _set(noise, "Detail", 8.0)
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    tint = _node(nd, "ShaderNodeMix", 150, 150, data_type="RGBA", blend_type="MULTIPLY")
    tint.inputs["Factor"].default_value = 0.55
    tint.inputs[7].default_value = (1.2, 1.15, 1.05, 1)
    nt.links.new(ramp.outputs["Color"], tint.inputs[6])
    mix_n = _node(nd, "ShaderNodeMix", -100, -150, data_type="RGBA")
    mix_n.inputs[6].default_value = (0.7, 0.7, 0.7, 1)
    mix_n.inputs[7].default_value = (1.3, 1.3, 1.3, 1)
    nt.links.new(noise.outputs["Fac"], mix_n.inputs["Factor"])
    nt.links.new(mix_n.outputs[2], tint.inputs[7])
    nt.links.new(tint.outputs[2], bsdf.inputs["Base Color"])
    return mat


# ---------------------------------------------------------------------
# 태양 / 하늘 / 카메라
# ---------------------------------------------------------------------
def solar_position(lat, lon, when, utc_offset):
    """(방위각°, 고도°) — 방위각은 북쪽 0°, 시계방향. (NOAA 간이식)"""
    n = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60.0 + when.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (n - 1 + (hour - 12.0) / 24.0)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = hour * 60.0 + eqt + 4.0 * lon - 60.0 * utc_offset
    ha = math.radians(tst / 4.0 - 180.0)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)
    elev = 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cz))))
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(dec) * math.cos(la)))
    return (180.0 + az) % 360.0, elev


def sun_vector(az_deg, el_deg):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return Vector((math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)))


def sun_color(el_deg):
    t = max(0.0, min(1.0, el_deg / 22.0))
    lo, hi = (1.0, 0.45, 0.18), (1.0, 0.95, 0.88)
    return tuple(lo[i] + (hi[i] - lo[i]) * t for i in range(3))


def smooth(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def make_world():
    world = bpy.data.worlds.new("Sky")
    if world.node_tree is None:
        world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    nd = nt.nodes
    out = _node(nd, "ShaderNodeOutputWorld", 900, 0)
    add = _node(nd, "ShaderNodeAddShader", 700, 0)
    nt.links.new(add.outputs[0], out.inputs["Surface"])

    # 하늘 색(대기 산란). 태양 원반은 아래에서 직접 그리므로 여기서는 끕니다.
    sky = _node(nd, "ShaderNodeTexSky", -300, 200)
    for kind in ("MULTIPLE_SCATTERING", "NISHITA"):         # Blender 5.x / 4.x 이름
        try:
            sky.sky_type = kind
            break
        except TypeError:
            continue
    sky.sun_disc = False
    sky.air_density = 1.3
    for name in ("aerosol_density", "dust_density"):          # Blender 5.x / 4.x 이름
        if hasattr(sky, name):
            setattr(sky, name, 0.35)                           # 작을수록 맑고 푸른 하늘
    bg = _node(nd, "ShaderNodeBackground", 400, 200)
    nt.links.new(sky.outputs[0], bg.inputs["Color"])
    nt.links.new(bg.outputs[0], add.inputs[0])

    # 태양 원반: 시선 방향과 태양 방향이 이루는 각으로 그립니다 (카메라에만 보이게)
    tc = _node(nd, "ShaderNodeTexCoord", -900, -250)
    sunv = _node(nd, "ShaderNodeCombineXYZ", -900, -450)
    dot = _node(nd, "ShaderNodeVectorMath", -650, -300, operation="DOT_PRODUCT")
    nt.links.new(tc.outputs["Object"], dot.inputs[0])
    nt.links.new(sunv.outputs[0], dot.inputs[1])
    r = math.radians(SUN_DISC_DEG / 2.0)
    mrange = _node(nd, "ShaderNodeMapRange", -400, -300, clamp=True)
    mrange.inputs["From Min"].default_value = math.cos(r * 1.7)
    mrange.inputs["From Max"].default_value = math.cos(r)
    nt.links.new(dot.outputs["Value"], mrange.inputs["Value"])
    lp = _node(nd, "ShaderNodeLightPath", -400, -550)
    cam_only = _node(nd, "ShaderNodeMath", -150, -350, operation="MULTIPLY")
    nt.links.new(mrange.outputs[0], cam_only.inputs[0])
    nt.links.new(lp.outputs["Is Camera Ray"], cam_only.inputs[1])
    strength = _node(nd, "ShaderNodeMath", 100, -350, operation="MULTIPLY")
    strength.inputs[1].default_value = 6000.0
    nt.links.new(cam_only.outputs[0], strength.inputs[0])
    disc = _node(nd, "ShaderNodeBackground", 400, -350)
    nt.links.new(strength.outputs[0], disc.inputs["Strength"])
    nt.links.new(disc.outputs[0], add.inputs[1])
    bpy.context.scene.world = world
    return world, sky, sunv, disc


def find_west_coast(X, Y, zf, lat):
    y = (lat - CENTER_LONLAT[1]) * M_PER_DEG_LAT
    i = int(np.argmin(np.abs(Y - y)))
    land = np.where(zf[i] > 1.0)[0]
    if len(land) == 0:
        raise ValueError("카메라를 둘 해안을 찾지 못했습니다. CAM_LAT 를 바꿔 보세요.")
    return float(X[land[0]]), float(Y[i])


def setup_sun_camera_animation(zf, X, Y, sky, sunv, disc):
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = FRAME_START, FRAME_END
    scene.render.fps = FPS

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.angle = math.radians(SUN_ANGLE_DEG)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_mode = "QUATERNION"

    if CAM_FIXED_LONLAT:
        cx = (CAM_FIXED_LONLAT[0] - CENTER_LONLAT[0]) * M_PER_DEG_LON
        cy = (CAM_FIXED_LONLAT[1] - CENTER_LONLAT[1]) * M_PER_DEG_LAT
    else:
        cx, cy = find_west_coast(X, Y, zf, CAM_LAT)
        cx += CAM_INLAND_M
    j = int(np.clip(np.argmin(np.abs(X - cx)), 0, len(X) - 1))
    i = int(np.clip(np.argmin(np.abs(Y - cy)), 0, len(Y) - 1))
    inside = X[0] <= cx <= X[-1] and Y[-1] <= cy <= Y[0]
    cz = (max(float(zf[i, j]), 0.0) * Z_EXAG if inside else 0.0) + CAM_HEIGHT_M
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = CAM_LENS_MM
    cam_data.clip_start, cam_data.clip_end = 1.0, 400_000.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (cx, cy, cz)
    scene.camera = cam

    when0 = datetime.datetime(*SUN_DATE, 12, 0, 0)
    mid_az = solar_position(SUN_LATLON[0], SUN_LATLON[1], when0.replace(
        hour=int((SUN_HOUR_START + SUN_HOUR_END) / 2), minute=int(((SUN_HOUR_START + SUN_HOUR_END) / 2 % 1) * 60)),
        UTC_OFFSET)[0]
    log = []
    for f in range(FRAME_START, FRAME_END + 1):
        t = (f - FRAME_START) / max(1, FRAME_END - FRAME_START)
        hour = SUN_HOUR_START + (SUN_HOUR_END - SUN_HOUR_START) * t
        when = when0.replace(hour=int(hour), minute=int((hour % 1) * 60), second=int(((hour * 60) % 1) * 60))
        az, el = solar_position(SUN_LATLON[0], SUN_LATLON[1], when, UTC_OFFSET)
        v = sun_vector(az, max(el, -3.0))
        col = sun_color(el)

        sun.rotation_quaternion = v.to_track_quat("Z", "Y")          # 조명의 -Z 가 태양에서 땅으로 향함
        sun_data.energy = SUN_STRENGTH * smooth(-1.0, 5.0, el)
        sun_data.color = col
        sky.sun_elevation = math.radians(max(el, -3.0))
        sky.sun_rotation = math.radians(az)                           # 방위각 = sun_rotation (0=북, 90=동)
        for k in range(3):
            sunv.inputs[k].default_value = v[k]
        disc.inputs["Color"].default_value = (col[0], col[1], col[2], 1.0)

        # 자동 노출: 해가 높을수록 노출을 낮춰 하늘이 하얗게 날아가지 않게
        scene.view_settings.exposure = -EXPOSURE_DROP * smooth(-1.0, 18.0, el)
        scene.view_settings.keyframe_insert("exposure", frame=f)

        yaw = CAM_YAW_CENTER + CAM_FOLLOW * (az - mid_az)
        cam.rotation_euler = (math.radians(90.0 + CAM_PITCH_DEG), 0.0, math.radians(-yaw))

        sun.keyframe_insert("rotation_quaternion", frame=f)
        sun_data.keyframe_insert("energy", frame=f)
        sun_data.keyframe_insert("color", frame=f)
        sky.keyframe_insert("sun_elevation", frame=f)
        sky.keyframe_insert("sun_rotation", frame=f)
        for k in range(3):
            sunv.inputs[k].keyframe_insert("default_value", frame=f)
        disc.inputs["Color"].keyframe_insert("default_value", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)
        if f in (FRAME_START, (FRAME_START + FRAME_END) // 2, FRAME_END):
            log.append("  프레임 %3d  %02d:%02d  태양 방위 %5.1f°  고도 %5.1f°" % (f, int(hour), int((hour % 1) * 60), az, el))
    print("태양 이동:\n" + "\n".join(log))
    return cam, sun


def setup_render(args):
    scene = bpy.context.scene
    try:
        scene.render.engine = args.engine
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 6
        scene.cycles.glossy_bounces = 4
        scene.cycles.device = "CPU"
        if args.device == "gpu":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = kind
                        prefs.get_devices()
                        gpus = [d for d in prefs.devices if d.type != "CPU"]
                        if gpus:
                            for d in prefs.devices:
                                d.use = d.type != "CPU"
                            scene.cycles.device = "GPU"
                            print("GPU 사용:", kind)
                            break
                    except TypeError:
                        continue
            except Exception as e:
                print("GPU 설정 실패, CPU 사용:", e)


# ---------------------------------------------------------------------
# 전체 조립
# ---------------------------------------------------------------------
def reset_scene():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights, bpy.data.worlds):
        for item in list(coll):
            coll.remove(item)


def apply_preset(name):
    for key, value in VIEW_PRESETS[name].items():
        globals()[key] = value


def build(args):
    apply_preset(args.view)
    reset_scene()
    dem = load_dem(args.dem, args.bbox or DEM_IMAGE_BBOX, args.zrange or DEM_IMAGE_ZRANGE)
    print("DEM:", dem.label, dem.z.shape)
    dem, step = crop_and_reduce(dem, JEJU_BBOX, MAX_TERRAIN_VERTS)
    X, Y = local_axes(dem)
    cell = float(abs(X[1] - X[0]))
    print("지형 격자 %d x %d (간격 약 %.0f m, 다운샘플 %d배), 최고 %.0f m" % (dem.z.shape[1], dem.z.shape[0], cell, step, dem.z.max()))
    zf = add_seafloor(dem.z, cell)

    terrain = build_terrain(X, Y, zf)
    terrain.data.materials.append(make_terrain_material())

    world, sky, sunv, disc = make_world()
    cam, sun = setup_sun_camera_animation(zf, X, Y, sky, sunv, disc)

    ocean = build_ocean((float(cam.location.x), float(cam.location.y)), X, Y, zf)
    ocean.data.materials.append(make_water_material())
    setup_render(args)
    bpy.context.scene.frame_set(FRAME_START)
    print("카메라 위치: x=%.0f y=%.0f z=%.0f (m)" % tuple(cam.location))
    if "가짜" in dem.label:
        print("※ 지금은 실제 DEM 이 아니라 모양만 흉내 낸 가짜 지형입니다. --dem N33E126.hgt 처럼 실제 파일을 지정하세요.")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="제주 바다 장면 만들기")
    p.add_argument("--dem", default=DEM_PATH, help=".hgt / .asc / .png / .tif 파일")
    p.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--zrange", type=float, nargs=2, default=None, metavar=("ZMIN", "ZMAX"))
    p.add_argument("--view", default="sunset", choices=sorted(VIEW_PRESETS), help="구도: sunset | sunrise")
    p.add_argument("--still", type=int, default=None, help="이 프레임 한 장만 렌더")
    p.add_argument("--anim", action="store_true", help="전체 애니메이션 렌더")
    p.add_argument("--save", default=None, help="완성된 장면을 .blend 로 저장")
    p.add_argument("--out", default=OUT_DIR, help="렌더 결과 폴더")
    p.add_argument("--res", default="%dx%d" % RES, help="예: 1280x720")
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--engine", default=ENGINE)
    p.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    a = p.parse_args(argv)
    a.res = tuple(int(v) for v in a.res.lower().split("x"))
    return a


def main():
    args = parse_args()
    build(args)
    scene = bpy.context.scene
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        print("저장:", args.save)
    if args.still is not None or args.anim:
        os.makedirs(args.out, exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        if args.still is not None:
            scene.frame_set(args.still)
            scene.render.filepath = os.path.join(args.out, "jeju_sea_%04d.png" % args.still)
            bpy.ops.render.render(write_still=True)
        else:
            scene.render.filepath = os.path.join(args.out, "jeju_sea_")
            bpy.ops.render.render(animation=True)
        print("렌더 완료:", args.out)
    else:
        print("장면이 만들어졌습니다. F12 로 한 장, Ctrl+F12 로 애니메이션을 렌더하세요. (스페이스바: 재생)")


if __name__ == "__main__":
    main()
