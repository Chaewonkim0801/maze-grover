import json
import math
import random
import time
from pathlib import Path
from collections import deque, defaultdict
import tkinter as tk
from tkinter import ttk

try:
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicProvider
except Exception as e:
    raise SystemExit(
        "請先安裝 qiskit：\n"
        "python -m pip install -U qiskit\n\n"
        f"原始錯誤：{e}"
    ) from e


# ============================================================
# 基本設定
# ============================================================
W = 20
H = 20
D = 20

BRAID_DEADEND_PROB = 0.18
EXTRA_HOLES_RATIO = 0.03

MAX_PATH_FACTOR = 8

# heuristic
MANHATTAN_WEIGHT = 3.0
TURN_WEIGHT = 0.8
VISIT_WEIGHT = 2.8
BACKTRACK_WEIGHT = 2.5
VERTICAL_PENALTY = 0.5
RECENT_PENALTY = 7.0
RECENT_WINDOW = 12

# good state 判定鬆弛量
GOOD_COST_MARGIN = 1.2

# 2D 多層顯示設定
CELL_2D = 18
LAYER_PAD = 28
LAYER_TITLE_H = 24
LAYER_GAP_X = 34
LAYER_GAP_Y = 40
LAYERS_PER_ROW = 3

# 3D 投影顯示設定
INITIAL_YAW = 0.85
INITIAL_PITCH = -0.60
INITIAL_SCALE = 32.0
MIN_SCALE = 8.0
MAX_SCALE = 120.0
DRAG_SENSITIVITY = 0.01
ZOOM_IN_FACTOR = 1.10
ZOOM_OUT_FACTOR = 0.90
CLICK_RADIUS = 16

# 顏色
BG_COLOR = "#f7f8fc"
TEXT_COLOR = "#1f2937"

WALL_2D_COLOR = "#111827"
GRID_2D_COLOR = "#e5e7eb"
LAYER_BORDER_COLOR = "#94a3b8"
LABEL_FG = "#334155"

UP_COLOR = "#8b5cf6"
DOWN_COLOR = "#ec4899"
BOTH_UD_COLOR = "#7c3aed"

PRED_COLOR = "#22c55e"
ACT_COLOR = "#2563eb"

START_BOX_COLOR = "#22c55e"
GOAL_BOX_COLOR = "#ef4444"
SELECT_BOX_COLOR = "#f59e0b"

START_FILL = "#dcfce7"
GOAL_FILL = "#fee2e2"

# 記憶迷宮
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
MEM_DIR = APP_DIR / "maze_memory"
MEM_DIR.mkdir(exist_ok=True)
MAZE_FILE = MEM_DIR / "maze_state_quantum_grover_fused_2d_3d_optimized.json"

# 方向
DIRS = {
    "N": (0, -1, 0),
    "S": (0,  1, 0),
    "W": (-1, 0, 0),
    "E": (1,  0, 0),
    "U": (0,  0, 1),
    "D": (0,  0, -1),
}
OPP = {
    "N": "S", "S": "N",
    "W": "E", "E": "W",
    "U": "D", "D": "U"
}
DIR_ORDER = ["N", "E", "S", "W", "U", "D"]

DIR_TO_QBITS = {
    "N": "000",
    "E": "001",
    "S": "010",
    "W": "011",
    "U": "100",
    "D": "101",
}


# ============================================================
# Maze3D
# ============================================================
class Maze3D:
    def __init__(self, w, h, d, seed=None):
        self.w = w
        self.h = h
        self.d = d
        self.seed = seed if seed is not None else random.randrange(1, 10**9)
        self.rng = random.Random(self.seed)

        self.walls = {}
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    self.walls[(x, y, z)] = {
                        "N": True, "S": True, "W": True, "E": True, "U": True, "D": True
                    }

        self.start = (0, 0, 0)
        self.goal = (w - 1, h - 1, d - 1)

    def in_bounds(self, x, y, z):
        return 0 <= x < self.w and 0 <= y < self.h and 0 <= z < self.d

    def neighbors(self, x, y, z):
        for d, (dx, dy, dz) in DIRS.items():
            nx, ny, nz = x + dx, y + dy, z + dz
            if self.in_bounds(nx, ny, nz):
                yield d, nx, ny, nz

    def remove_wall(self, x, y, z, d):
        dx, dy, dz = DIRS[d]
        nx, ny, nz = x + dx, y + dy, z + dz
        if not self.in_bounds(nx, ny, nz):
            return
        self.walls[(x, y, z)][d] = False
        self.walls[(nx, ny, nz)][OPP[d]] = False

    def is_open(self, x, y, z, d):
        return not self.walls[(x, y, z)][d]

    def adj_list(self):
        adj = {(x, y, z): [] for z in range(self.d) for y in range(self.h) for x in range(self.w)}
        for z in range(self.d):
            for y in range(self.h):
                for x in range(self.w):
                    for d, nx, ny, nz in self.neighbors(x, y, z):
                        if self.is_open(x, y, z, d):
                            adj[(x, y, z)].append((nx, ny, nz))
        return adj

    def _degree(self, x, y, z):
        deg = 0
        for d, nx, ny, nz in self.neighbors(x, y, z):
            if not self.walls[(x, y, z)][d]:
                deg += 1
        return deg

    def _braid_dead_ends(self, prob=0.18):
        dead_ends = []
        for z in range(self.d):
            for y in range(self.h):
                for x in range(self.w):
                    if (x, y, z) in (self.start, self.goal):
                        continue
                    if self._degree(x, y, z) == 1:
                        dead_ends.append((x, y, z))

        self.rng.shuffle(dead_ends)
        for x, y, z in dead_ends:
            if self.rng.random() > prob:
                continue

            candidates = []
            for d, nx, ny, nz in self.neighbors(x, y, z):
                if self.walls[(x, y, z)][d]:
                    candidates.append(d)
            if not candidates:
                continue

            scored = []
            for d in candidates:
                dx, dy, dz = DIRS[d]
                nx, ny, nz = x + dx, y + dy, z + dz
                scored.append((self._degree(nx, ny, nz), d))
            scored.sort(reverse=True)
            self.remove_wall(x, y, z, scored[0][1])

    def _punch_random_closed_walls(self, k=10):
        for _ in range(k):
            x = self.rng.randrange(self.w)
            y = self.rng.randrange(self.h)
            z = self.rng.randrange(self.d)
            cands = []
            for d, nx, ny, nz in self.neighbors(x, y, z):
                if self.walls[(x, y, z)][d]:
                    cands.append(d)
            if cands:
                self.remove_wall(x, y, z, self.rng.choice(cands))

    def generate(self, braid_p=BRAID_DEADEND_PROB, extra_holes_ratio=EXTRA_HOLES_RATIO):
        self.rng.seed(self.seed)

        visited = {self.start}
        stack = [self.start]

        while stack:
            x, y, z = stack[-1]
            unvisited = []
            for d, nx, ny, nz in self.neighbors(x, y, z):
                if (nx, ny, nz) not in visited:
                    unvisited.append((d, nx, ny, nz))

            if not unvisited:
                stack.pop()
                continue

            d, nx, ny, nz = self.rng.choice(unvisited)
            self.remove_wall(x, y, z, d)
            visited.add((nx, ny, nz))
            stack.append((nx, ny, nz))

        self._braid_dead_ends(prob=braid_p)
        extra_holes = max(1, int(self.w * self.h * self.d * extra_holes_ratio))
        self._punch_random_closed_walls(k=extra_holes)

    def to_dict(self):
        return {
            "w": self.w,
            "h": self.h,
            "d": self.d,
            "seed": self.seed,
            "start": list(self.start),
            "goal": list(self.goal),
            "walls": {
                f"{x},{y},{z}": self.walls[(x, y, z)]
                for z in range(self.d)
                for y in range(self.h)
                for x in range(self.w)
            },
        }

    @staticmethod
    def from_dict(d):
        m = Maze3D(d["w"], d["h"], d["d"], d["seed"])
        m.start = tuple(d["start"])
        m.goal = tuple(d["goal"])
        m.walls = {}
        for key, wdict in d["walls"].items():
            x, y, z = map(int, key.split(","))
            m.walls[(x, y, z)] = dict(wdict)
        return m


def save_maze(maze: Maze3D):
    with open(MAZE_FILE, "w", encoding="utf-8") as f:
        json.dump(maze.to_dict(), f, ensure_ascii=False, indent=2)


def load_or_create_maze(w=W, h=H, d=D):
    if MAZE_FILE.exists():
        try:
            with open(MAZE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = Maze3D.from_dict(data)
            if m.w == w and m.h == h and m.d == d:
                return m
        except Exception:
            pass

    m = Maze3D(w, h, d)
    m.generate()
    save_maze(m)
    return m


# ============================================================
# BFS
# ============================================================
def bfs_from_goal(maze: Maze3D):
    adj = maze.adj_list()
    goal = maze.goal

    dist = {goal: 0}
    q = deque([goal])

    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist, adj


def shortest_path_by_dist(start, goal, dist, adj):
    if start not in dist:
        return None
    if start == goal:
        return [start]

    path = [start]
    cur = start
    seen = {start}

    while cur != goal:
        opts = [nb for nb in adj[cur] if nb in dist and dist[nb] == dist[cur] - 1]
        if not opts:
            return None
        opts.sort(key=lambda p: (p[2], p[1], p[0]))
        nxt = opts[0]
        if nxt in seen and nxt != goal:
            return None
        path.append(nxt)
        seen.add(nxt)
        cur = nxt

    return path


# ============================================================
# heuristic
# ============================================================
def turn_penalty_3d(prev, cur, nxt):
    if prev is None:
        return 0.0
    dx1 = cur[0] - prev[0]
    dy1 = cur[1] - prev[1]
    dz1 = cur[2] - prev[2]
    dx2 = nxt[0] - cur[0]
    dy2 = nxt[1] - cur[1]
    dz2 = nxt[2] - cur[2]
    return 0.0 if (dx1, dy1, dz1) == (dx2, dy2, dz2) else 1.0


# ============================================================
# 3-qubit Local Grover chooser（優化版）
# ============================================================
class LocalGroverChooser3D:
    def __init__(self, seed: int):
        self.seed = int(seed) & 0xFFFFFFFF
        self.backend = BasicProvider().get_backend("basic_simulator")
        self.backend_name = "Qiskit BasicSimulator / 3-qubit local Grover (optimized)"
        self.run_counter = 0

        # key: tuple(sorted(good_qbits_list))
        self.circuit_cache = {}

        # 統計資訊
        self.cache_hits = 0
        self.cache_misses = 0

    def _oracle_mark_state(self, qc: QuantumCircuit, qbits012: str):
        for q, b in enumerate(qbits012):
            if b == "0":
                qc.x(q)

        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)

        for q, b in enumerate(qbits012):
            if b == "0":
                qc.x(q)

    def _apply_diffusion(self, qc: QuantumCircuit):
        qc.h([0, 1, 2])
        qc.x([0, 1, 2])

        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)

        qc.x([0, 1, 2])
        qc.h([0, 1, 2])

    def _optimal_iterations_small_space(self, m, n_states=8):
        if m <= 0 or m >= n_states:
            return 0
        theta = math.asin(math.sqrt(m / n_states))
        iters = round(math.pi / (4 * theta) - 0.5)
        return max(1, iters)

    def _shots_for_m(self, m):
        if m <= 1:
            return 192
        elif m <= 3:
            return 96
        else:
            return 48

    def _build_grover_circuit(self, good_qbits_list):
        key = tuple(sorted(good_qbits_list))

        cached = self.circuit_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        self.cache_misses += 1

        qc = QuantumCircuit(3, 3)
        qc.h([0, 1, 2])

        m = len(good_qbits_list)
        if 0 < m < 8:
            iters = self._optimal_iterations_small_space(m, 8)
            for _ in range(iters):
                for state in good_qbits_list:
                    self._oracle_mark_state(qc, state)
                self._apply_diffusion(qc)

        qc.measure([0, 1, 2], [0, 1, 2])

        self.circuit_cache[key] = qc
        return qc

    def _measure_counts(self, good_qbits_list):
        qc = self._build_grover_circuit(good_qbits_list)
        seed = (self.seed ^ (self.run_counter * 0x9E3779B1)) & 0xFFFFFFFF
        self.run_counter += 1

        shots = self._shots_for_m(len(good_qbits_list))
        job = self.backend.run(qc, shots=shots, seed_simulator=seed)
        result = job.result()
        return result.get_counts()

    @staticmethod
    def _counts_key_to_qbits012(counts_key: str):
        return counts_key[::-1]

    def choose_direction(self, prev, cur, maze: Maze3D, visit_count, recent_path):
        scored = []
        legal_dirs = []

        for d in DIR_ORDER:
            dx, dy, dz = DIRS[d]
            nx, ny, nz = cur[0] + dx, cur[1] + dy, cur[2] + dz

            legal = maze.in_bounds(nx, ny, nz) and maze.is_open(cur[0], cur[1], cur[2], d)
            if not legal:
                continue

            legal_dirs.append(d)
            nxt = (nx, ny, nz)

            m_cost = MANHATTAN_WEIGHT * (
                abs(nx - maze.goal[0]) +
                abs(ny - maze.goal[1]) +
                abs(nz - maze.goal[2])
            )
            t_cost = TURN_WEIGHT * turn_penalty_3d(prev, cur, nxt)
            v_cost = VISIT_WEIGHT * visit_count[nxt]
            b_cost = BACKTRACK_WEIGHT if (prev is not None and nxt == prev) else 0.0
            z_cost = VERTICAL_PENALTY if nz != cur[2] else 0.0
            r_cost = RECENT_PENALTY if nxt in recent_path else 0.0

            total = m_cost + t_cost + v_cost + b_cost + z_cost + r_cost
            scored.append((d, total, nxt))

        if not legal_dirs:
            return None

        scored.sort(key=lambda t: (t[1], t[2][2], t[2][1], t[2][0]))
        best_cost = scored[0][1]

        # 取接近最佳的一群作為 good states
        good_dirs = [d for d, cost, _ in scored if cost <= best_cost + GOOD_COST_MARGIN]
        if not good_dirs:
            good_dirs = [scored[0][0]]

        good_qbits = [DIR_TO_QBITS[d] for d in good_dirs]
        counts = self._measure_counts(good_qbits)

        best_dir = None
        best_count = -1

        for d in legal_dirs:
            qbits = DIR_TO_QBITS[d]
            c = 0
            for key, val in counts.items():
                if self._counts_key_to_qbits012(key) == qbits:
                    c += int(val)
            if c > best_count:
                best_count = c
                best_dir = d

        # 若量測結果沒有有效偏向，退回最低成本方向
        if best_dir is None:
            return scored[0][2]

        for d, _, nxt in scored:
            if d == best_dir:
                return nxt

        return scored[0][2]


def path_quantum_only_3d(start, goal, maze: Maze3D, qchooser: LocalGroverChooser3D):
    if start == goal:
        return [start]

    path = [start]
    prev = None
    cur = start
    visit_count = defaultdict(int)
    visit_count[start] = 1

    max_steps = maze.w * maze.h * maze.d * MAX_PATH_FACTOR

    while cur != goal and len(path) <= max_steps:
        recent_path = set(path[-RECENT_WINDOW:])
        nxt = qchooser.choose_direction(prev, cur, maze, visit_count, recent_path)
        if nxt is None:
            return None

        path.append(nxt)
        visit_count[nxt] += 1
        prev, cur = cur, nxt

    if cur != goal:
        return None

    return path


# ============================================================
# UI
# ============================================================
class App:
    def __init__(self, root):
        self.root = root
        self.view_mode = "layers"

        self.maze = load_or_create_maze(W, H, D)
        self.dist, self.adj = bfs_from_goal(self.maze)

        self.fixed_start = self.maze.start
        self.goal = self.maze.goal
        self.click_start = self.fixed_start

        self.pred_path = None
        self.actual_path = None

        self.qchooser = LocalGroverChooser3D(seed=(self.maze.seed ^ 0xA5A5A5A5))
        self.screen_centers = []

        # 3D camera
        self.yaw = INITIAL_YAW
        self.pitch = INITIAL_PITCH
        self.scale = INITIAL_SCALE
        self.dragging = False
        self.last_mouse = None
        self.view_cx = 0.0
        self.view_cy = 0.0

        self.build_ui()
        self.update_title()
        self.draw_scene()

    # --------------------------------------------------------
    # UI layout
    # --------------------------------------------------------
    def build_ui(self):
        self.root.title(f"{W}x{H}x{D} 多層 2D / 可旋轉 3D 量子迷宮（Grover 優化版）")

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        ttk.Button(top, text="Grover", command=self.on_quantum).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="清除", command=self.on_clear).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="新迷宮", command=self.on_new).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="切換 2D / 3D", command=self.on_toggle_view).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="重設視角", command=self.on_reset_view).pack(side=tk.LEFT, padx=6)

        self.title_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.title_var, justify=tk.LEFT).pack(
            side=tk.TOP, fill=tk.X, padx=10
        )

        self.info_var = tk.StringVar(value="預測路徑：—\n實際路徑：—\n處理時間：—\n快取：—")
        info = ttk.LabelFrame(self.root, text="結果")
        info.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        ttk.Label(
            info,
            textvariable=self.info_var,
            justify=tk.LEFT,
            anchor="w"
        ).pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.legend_frame = ttk.LabelFrame(self.root, text="操作說明")
        self.legend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        self.legend_var = tk.StringVar()
        ttk.Label(
            self.legend_frame,
            textvariable=self.legend_var,
            justify=tk.LEFT,
            anchor="w"
        ).pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        canvas_wrap = ttk.Frame(self.root)
        canvas_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(
            canvas_wrap,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.ysb = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.canvas.yview)
        self.ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.ysb.set)

        self.bind_events_for_mode()
        self.update_legend()

    def bind_events_for_mode(self):
        for seq in [
            "<Button-1>", "<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>",
            "<Double-Button-1>", "<MouseWheel>", "<Button-4>", "<Button-5>", "<Configure>"
        ]:
            self.canvas.unbind(seq)

        if self.view_mode == "layers":
            self.canvas.bind("<Button-1>", self.on_canvas_click_2d)
            self.canvas.bind("<MouseWheel>", self.on_mousewheel_2d)
            self.canvas.bind("<Button-4>", self.on_mousewheel_linux_up_2d)
            self.canvas.bind("<Button-5>", self.on_mousewheel_linux_down_2d)
        else:
            self.canvas.bind("<ButtonPress-1>", self.on_drag_start_3d)
            self.canvas.bind("<B1-Motion>", self.on_drag_move_3d)
            self.canvas.bind("<ButtonRelease-1>", self.on_drag_end_3d)
            self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click_3d)
            self.canvas.bind("<MouseWheel>", self.on_mousewheel_3d)
            self.canvas.bind("<Button-4>", self.on_mousewheel_linux_up_3d)
            self.canvas.bind("<Button-5>", self.on_mousewheel_linux_down_3d)
            self.canvas.bind("<Configure>", self.on_canvas_resize_3d)

    # --------------------------------------------------------
    # text helpers
    # --------------------------------------------------------
    def update_title(self):
        mode_text = "多層 2D" if self.view_mode == "layers" else "可旋轉 3D"
        self.title_var.set(
            f"顯示模式：{mode_text}    固定起點：{self.fixed_start}    目前起點：{self.click_start}    終點：{self.goal}"
        )

    def update_legend(self):
        if self.view_mode == "layers":
            self.legend_var.set(
                "目前模式：多層 2D\n"
                "單擊任一層任一格：設為起點並直接計算到終點\n"
                "滑鼠滾輪：上下捲動畫面\n"
                "黑線 = 牆，綠色虛線 = Grover 預測路徑，藍色實線 = BFS 最短路\n"
                "↑ = 可往上層，↓ = 可往下層，紫色 = 同時可上下\n"
                "綠框 = 固定起點，紅框 = 終點，橘框 = 目前點選起點"
            )
        else:
            self.legend_var.set(
                "目前模式：3D\n"
                "雙擊任一格：設為起點並直接計算到終點\n"
                "滑鼠拖曳：旋轉視角\n"
                "滑鼠滾輪：縮放\n"
                "黑線 = 牆，綠色虛線 = Grover 路徑，藍色實線 = BFS 最短路\n"
                "↑ = 可往上層，↓ = 可往下層，↕ = 同時可上下\n"
                "綠圈 = 固定起點，紅圈 = 終點，橘圈 = 目前點選起點"
            )

    def set_result(self, pred_path, actual_path, ms):
        pred_text = f"{len(pred_path) - 1} 步" if pred_path else "找不到"
        actual_text = f"{len(actual_path) - 1} 步" if actual_path else "不可達"
        cache_text = f"hit={self.qchooser.cache_hits} / miss={self.qchooser.cache_misses} / cached={len(self.qchooser.circuit_cache)}"
        self.info_var.set(
            f"預測路徑：{pred_text}（綠色虛線）\n"
            f"實際路徑：{actual_text}（藍色實線）\n"
            f"處理時間：{ms:.3f} ms\n"
            f"快取：{cache_text}"
        )

    # --------------------------------------------------------
    # 幾何 helpers
    # --------------------------------------------------------
    def normalize_angle(self, a):
        return ((a + math.pi) % (2 * math.pi)) - math.pi

    def layer_origin(self, z):
        layer_w = self.maze.w * CELL_2D
        layer_h = self.maze.h * CELL_2D

        col = z % LAYERS_PER_ROW
        row = z // LAYERS_PER_ROW

        ox = LAYER_PAD + col * (layer_w + LAYER_GAP_X)
        oy = LAYER_PAD + row * (layer_h + LAYER_TITLE_H + LAYER_GAP_Y) + LAYER_TITLE_H
        return ox, oy

    def cell_rect(self, x, y, z):
        ox, oy = self.layer_origin(z)
        x1 = ox + x * CELL_2D
        y1 = oy + y * CELL_2D
        x2 = x1 + CELL_2D
        y2 = y1 + CELL_2D
        return x1, y1, x2, y2

    def world_center(self):
        return (
            (self.maze.w - 1) / 2.0,
            (self.maze.h - 1) / 2.0,
            (self.maze.d - 1) / 2.0,
        )

    def rotate_point(self, x, y, z):
        cx, cy, cz = self.world_center()
        x -= cx
        y -= cy
        z -= cz

        cyaw = math.cos(self.yaw)
        syaw = math.sin(self.yaw)
        x, y = x * cyaw - y * syaw, x * syaw + y * cyaw

        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        y, z = y * cp - z * sp, y * sp + z * cp

        return x, y, z

    def project_point(self, x, y, z):
        rx, ry, rz = self.rotate_point(x, y, z)

        scene_size = max(self.maze.w, self.maze.h, self.maze.d)
        cam_dist = scene_size * 3.0
        denom = cam_dist - rz
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6
        factor = cam_dist / denom

        sx = self.view_cx + rx * self.scale * factor
        sy = self.view_cy + ry * self.scale * factor
        return sx, sy, rz

    def cell_corners_3d(self, x, y, z):
        return [
            (x - 0.5, y - 0.5, z),
            (x + 0.5, y - 0.5, z),
            (x + 0.5, y + 0.5, z),
            (x - 0.5, y + 0.5, z),
        ]

    def cell_center_projected(self, x, y, z):
        return self.project_point(x, y, z)

    # --------------------------------------------------------
    # draw
    # --------------------------------------------------------
    def draw_scene(self):
        if self.view_mode == "layers":
            self.draw_layers_2d()
        else:
            self.draw_rotatable_3d()

    def draw_layers_2d(self):
        self.canvas.delete("all")
        self.screen_centers = []

        layer_w = self.maze.w * CELL_2D
        layer_h = self.maze.h * CELL_2D

        for z in range(self.maze.d):
            ox, oy = self.layer_origin(z)

            self.canvas.create_text(
                ox, oy - 12,
                text=f"z = {z}",
                anchor="w",
                fill=LABEL_FG,
                font=("Arial", 11, "bold")
            )

            self.canvas.create_rectangle(
                ox, oy, ox + layer_w, oy + layer_h,
                outline=LAYER_BORDER_COLOR,
                width=2
            )

            for y in range(self.maze.h):
                for x in range(self.maze.w):
                    x1, y1, x2, y2 = self.cell_rect(x, y, z)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    self.screen_centers.append(((x, y, z), cx, cy, 0))

                    if (x, y, z) == self.fixed_start:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=START_FILL, outline="")
                    elif (x, y, z) == self.goal:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill=GOAL_FILL, outline="")
                    else:
                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="")

                    self.canvas.create_rectangle(
                        x1, y1, x2, y2,
                        outline=GRID_2D_COLOR,
                        width=1
                    )

                    walls = self.maze.walls[(x, y, z)]

                    if walls["N"]:
                        self.canvas.create_line(x1, y1, x2, y1, fill=WALL_2D_COLOR, width=2)
                    if walls["S"]:
                        self.canvas.create_line(x1, y2, x2, y2, fill=WALL_2D_COLOR, width=2)
                    if walls["W"]:
                        self.canvas.create_line(x1, y1, x1, y2, fill=WALL_2D_COLOR, width=2)
                    if walls["E"]:
                        self.canvas.create_line(x2, y1, x2, y2, fill=WALL_2D_COLOR, width=2)

                    up_open = not walls["U"]
                    down_open = not walls["D"]

                    if up_open and down_open:
                        self.canvas.create_oval(
                            cx - 4, cy - 4, cx + 4, cy + 4,
                            fill=BOTH_UD_COLOR, outline=""
                        )
                        self.canvas.create_text(
                            cx, cy - 7,
                            text="↑",
                            fill=BOTH_UD_COLOR,
                            font=("Arial", 7, "bold")
                        )
                        self.canvas.create_text(
                            cx, cy + 7,
                            text="↓",
                            fill=BOTH_UD_COLOR,
                            font=("Arial", 7, "bold")
                        )
                    elif up_open:
                        self.canvas.create_text(
                            cx, cy,
                            text="↑",
                            fill=UP_COLOR,
                            font=("Arial", 9, "bold")
                        )
                    elif down_open:
                        self.canvas.create_text(
                            cx, cy,
                            text="↓",
                            fill=DOWN_COLOR,
                            font=("Arial", 9, "bold")
                        )

        if self.actual_path:
            for a, b in zip(self.actual_path[:-1], self.actual_path[1:]):
                ax, ay, az = a
                bx, by, bz = b

                if az != bz:
                    continue

                x1, y1, x2, y2 = self.cell_rect(ax, ay, az)
                sx1 = (x1 + x2) / 2
                sy1 = (y1 + y2) / 2

                x1, y1, x2, y2 = self.cell_rect(bx, by, bz)
                sx2 = (x1 + x2) / 2
                sy2 = (y1 + y2) / 2

                self.canvas.create_line(
                    sx1, sy1, sx2, sy2,
                    fill=ACT_COLOR,
                    width=4,
                    capstyle=tk.ROUND
                )

        if self.pred_path:
            for a, b in zip(self.pred_path[:-1], self.pred_path[1:]):
                ax, ay, az = a
                bx, by, bz = b

                if az != bz:
                    continue

                x1, y1, x2, y2 = self.cell_rect(ax, ay, az)
                sx1 = (x1 + x2) / 2
                sy1 = (y1 + y2) / 2

                x1, y1, x2, y2 = self.cell_rect(bx, by, bz)
                sx2 = (x1 + x2) / 2
                sy2 = (y1 + y2) / 2

                self.canvas.create_line(
                    sx1, sy1, sx2, sy2,
                    fill=PRED_COLOR,
                    width=3,
                    dash=(4, 3),
                    capstyle=tk.ROUND
                )

        if self.actual_path:
            for a, b in zip(self.actual_path[:-1], self.actual_path[1:]):
                ax, ay, az = a
                bx, by, bz = b
                if az == bz:
                    continue

                x1, y1, x2, y2 = self.cell_rect(ax, ay, az)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                text = "↑" if bz > az else "↓"
                self.canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=ACT_COLOR, outline="")
                self.canvas.create_text(cx, cy, text=text, fill="white", font=("Arial", 9, "bold"))

        if self.pred_path:
            for a, b in zip(self.pred_path[:-1], self.pred_path[1:]):
                ax, ay, az = a
                bx, by, bz = b
                if az == bz:
                    continue

                x1, y1, x2, y2 = self.cell_rect(ax, ay, az)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                text = "↑" if bz > az else "↓"
                self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, outline=PRED_COLOR, width=2)
                self.canvas.create_text(cx, cy, text=text, fill=PRED_COLOR, font=("Arial", 8, "bold"))

        for cell, color, text, pad in [
            (self.fixed_start, START_BOX_COLOR, "起", 2),
            (self.goal, GOAL_BOX_COLOR, "終", 2),
            (self.click_start, SELECT_BOX_COLOR, "", 5),
        ]:
            x, y, z = cell
            x1, y1, x2, y2 = self.cell_rect(x, y, z)
            self.canvas.create_rectangle(
                x1 + pad, y1 + pad, x2 - pad, y2 - pad,
                outline=color, width=3
            )
            if text:
                self.canvas.create_text(
                    (x1 + x2) / 2, (y1 + y2) / 2,
                    text=text,
                    fill=color,
                    font=("Arial", 9, "bold")
                )

        total_rows = (self.maze.d + LAYERS_PER_ROW - 1) // LAYERS_PER_ROW
        total_h = (
            LAYER_PAD * 2
            + total_rows * (self.maze.h * CELL_2D + LAYER_TITLE_H)
            + (total_rows - 1) * LAYER_GAP_Y
        )
        total_cols = min(LAYERS_PER_ROW, self.maze.d)
        total_w = (
            LAYER_PAD * 2
            + total_cols * (self.maze.w * CELL_2D)
            + (total_cols - 1) * LAYER_GAP_X
        )
        self.canvas.configure(scrollregion=(0, 0, total_w + 20, total_h + 20))

    def draw_rotatable_3d(self):
        self.canvas.delete("all")
        self.screen_centers = []

        cw = max(self.canvas.winfo_width(), 1200)
        ch = max(self.canvas.winfo_height(), 800)
        self.view_cx = cw * 0.50
        self.view_cy = ch * 0.56

        draw_items = []

        for z in range(self.maze.d):
            for y in range(self.maze.h):
                for x in range(self.maze.w):
                    corners3d = self.cell_corners_3d(x, y, z)
                    proj = [self.project_point(px, py, pz) for px, py, pz in corners3d]

                    pts2d = []
                    avg_depth = 0.0
                    for sx, sy, rz in proj:
                        pts2d.extend([sx, sy])
                        avg_depth += rz
                    avg_depth /= 4.0

                    if (x, y, z) == self.fixed_start:
                        fill = START_FILL
                    elif (x, y, z) == self.goal:
                        fill = GOAL_FILL
                    else:
                        fill = "white"

                    draw_items.append(("poly", avg_depth, {
                        "pts": pts2d,
                        "fill": fill,
                        "outline": GRID_2D_COLOR,
                        "width": 1
                    }))

                    cx, cy, crz = self.cell_center_projected(x, y, z)
                    self.screen_centers.append(((x, y, z), cx, cy, crz))

                    walls = self.maze.walls[(x, y, z)]

                    def add_wall_line(p1_3d, p2_3d):
                        p1 = self.project_point(*p1_3d)
                        p2 = self.project_point(*p2_3d)
                        draw_items.append(("line", (p1[2] + p2[2]) / 2.0, {
                            "coords": (p1[0], p1[1], p2[0], p2[1]),
                            "fill": WALL_2D_COLOR,
                            "width": 2
                        }))

                    if walls["N"]:
                        add_wall_line((x - 0.5, y - 0.5, z), (x + 0.5, y - 0.5, z))
                    if walls["S"]:
                        add_wall_line((x - 0.5, y + 0.5, z), (x + 0.5, y + 0.5, z))
                    if walls["W"]:
                        add_wall_line((x - 0.5, y - 0.5, z), (x - 0.5, y + 0.5, z))
                    if walls["E"]:
                        add_wall_line((x + 0.5, y - 0.5, z), (x + 0.5, y + 0.5, z))

                    up_open = not walls["U"]
                    down_open = not walls["D"]
                    if up_open or down_open:
                        color = BOTH_UD_COLOR if (up_open and down_open) else (UP_COLOR if up_open else DOWN_COLOR)
                        text = "↕" if (up_open and down_open) else ("↑" if up_open else "↓")
                        draw_items.append(("text", crz + 0.01, {
                            "x": cx,
                            "y": cy,
                            "text": text,
                            "fill": color,
                            "font": ("Arial", 10, "bold")
                        }))

        if self.actual_path:
            for a, b in zip(self.actual_path[:-1], self.actual_path[1:]):
                p1 = self.project_point(*a)
                p2 = self.project_point(*b)
                depth = (p1[2] + p2[2]) / 2.0

                draw_items.append(("line", depth + 0.20, {
                    "coords": (p1[0], p1[1], p2[0], p2[1]),
                    "fill": ACT_COLOR,
                    "width": 4
                }))

                if a[2] != b[2]:
                    mx = (p1[0] + p2[0]) / 2.0
                    my = (p1[1] + p2[1]) / 2.0
                    text = "↑" if b[2] > a[2] else "↓"
                    draw_items.append(("solid_oval", depth + 0.22, {
                        "bbox": (mx - 8, my - 8, mx + 8, my + 8),
                        "fill": ACT_COLOR,
                        "outline": ACT_COLOR,
                        "width": 1
                    }))
                    draw_items.append(("text", depth + 0.23, {
                        "x": mx,
                        "y": my,
                        "text": text,
                        "fill": "white",
                        "font": ("Arial", 9, "bold")
                    }))

        if self.pred_path:
            for a, b in zip(self.pred_path[:-1], self.pred_path[1:]):
                p1 = self.project_point(*a)
                p2 = self.project_point(*b)
                depth = (p1[2] + p2[2]) / 2.0

                draw_items.append(("line_dash", depth + 0.30, {
                    "coords": (p1[0], p1[1], p2[0], p2[1]),
                    "fill": PRED_COLOR,
                    "width": 3,
                    "dash": (4, 3)
                }))

                if a[2] != b[2]:
                    mx = (p1[0] + p2[0]) / 2.0
                    my = (p1[1] + p2[1]) / 2.0
                    text = "↑" if b[2] > a[2] else "↓"
                    draw_items.append(("oval", depth + 0.31, {
                        "bbox": (mx - 7, my - 7, mx + 7, my + 7),
                        "outline": PRED_COLOR,
                        "width": 2
                    }))
                    draw_items.append(("text", depth + 0.32, {
                        "x": mx,
                        "y": my,
                        "text": text,
                        "fill": PRED_COLOR,
                        "font": ("Arial", 8, "bold")
                    }))

        for cell, color, text, r in [
            (self.fixed_start, START_BOX_COLOR, "起", 10),
            (self.goal, GOAL_BOX_COLOR, "終", 10),
            (self.click_start, SELECT_BOX_COLOR, "", 14),
        ]:
            cx, cy, depth = self.cell_center_projected(*cell)

            draw_items.append(("oval", depth + 0.40, {
                "bbox": (cx - r, cy - r, cx + r, cy + r),
                "outline": color,
                "width": 3
            }))

            if text:
                draw_items.append(("text", depth + 0.41, {
                    "x": cx,
                    "y": cy,
                    "text": text,
                    "fill": color,
                    "font": ("Arial", 10, "bold")
                }))

        draw_items.sort(key=lambda item: item[1])

        for kind, _, data in draw_items:
            if kind == "poly":
                self.canvas.create_polygon(
                    *data["pts"],
                    fill=data["fill"],
                    outline=data["outline"],
                    width=data["width"]
                )
            elif kind == "line":
                self.canvas.create_line(
                    *data["coords"],
                    fill=data["fill"],
                    width=data["width"],
                    capstyle=tk.ROUND
                )
            elif kind == "line_dash":
                self.canvas.create_line(
                    *data["coords"],
                    fill=data["fill"],
                    width=data["width"],
                    dash=data["dash"],
                    capstyle=tk.ROUND
                )
            elif kind == "oval":
                self.canvas.create_oval(
                    *data["bbox"],
                    outline=data["outline"],
                    width=data["width"]
                )
            elif kind == "solid_oval":
                self.canvas.create_oval(
                    *data["bbox"],
                    fill=data["fill"],
                    outline=data["outline"],
                    width=data["width"]
                )
            elif kind == "text":
                self.canvas.create_text(
                    data["x"], data["y"],
                    text=data["text"],
                    fill=data["fill"],
                    font=data["font"]
                )

        self.canvas.configure(scrollregion=(0, 0, cw, ch))

    # --------------------------------------------------------
    # picking
    # --------------------------------------------------------
    def find_clicked_cell_2d(self, px, py):
        if not self.screen_centers:
            return None

        best = None
        best_d2 = float("inf")

        for cell, sx, sy, _ in self.screen_centers:
            d2 = (sx - px) ** 2 + (sy - py) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = cell

        if best_d2 <= (CELL_2D * 0.9) ** 2:
            return best
        return None

    def find_clicked_cell_3d(self, px, py):
        if not self.screen_centers:
            return None

        candidates = []
        for cell, sx, sy, depth in self.screen_centers:
            d2 = (sx - px) ** 2 + (sy - py) ** 2
            if d2 <= CLICK_RADIUS ** 2:
                candidates.append((d2, -depth, cell))

        if not candidates:
            return None

        candidates.sort()
        return candidates[0][2]

    # --------------------------------------------------------
    # 2D events
    # --------------------------------------------------------
    def on_canvas_click_2d(self, event):
        px = self.canvas.canvasx(event.x)
        py = self.canvas.canvasy(event.y)

        cell = self.find_clicked_cell_2d(px, py)
        if cell is not None:
            self.click_start = cell
            self.update_title()
            self.run_quantum(self.click_start)

    def on_mousewheel_2d(self, event):
        if event.delta < 0:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(-1, "units")

    def on_mousewheel_linux_up_2d(self, event):
        self.canvas.yview_scroll(-1, "units")

    def on_mousewheel_linux_down_2d(self, event):
        self.canvas.yview_scroll(1, "units")

    # --------------------------------------------------------
    # 3D events
    # --------------------------------------------------------
    def on_drag_start_3d(self, event):
        self.dragging = True
        self.last_mouse = (event.x, event.y)

    def on_drag_move_3d(self, event):
        if not self.dragging or self.last_mouse is None:
            return

        lx, ly = self.last_mouse
        dx = event.x - lx
        dy = event.y - ly
        self.last_mouse = (event.x, event.y)

        self.yaw -= dx * DRAG_SENSITIVITY
        self.pitch -= dy * DRAG_SENSITIVITY

        self.yaw = self.normalize_angle(self.yaw)
        self.pitch = self.normalize_angle(self.pitch)

        self.draw_scene()

    def on_drag_end_3d(self, event):
        self.dragging = False
        self.last_mouse = None

    def on_canvas_double_click_3d(self, event):
        px = self.canvas.canvasx(event.x)
        py = self.canvas.canvasy(event.y)

        cell = self.find_clicked_cell_3d(px, py)
        if cell is not None:
            self.click_start = cell
            self.update_title()
            self.run_quantum(self.click_start)

    def on_mousewheel_3d(self, event):
        if event.delta < 0:
            self.scale *= ZOOM_OUT_FACTOR
        else:
            self.scale *= ZOOM_IN_FACTOR
        self.scale = max(MIN_SCALE, min(MAX_SCALE, self.scale))
        self.draw_scene()

    def on_mousewheel_linux_up_3d(self, event):
        self.scale *= ZOOM_IN_FACTOR
        self.scale = min(MAX_SCALE, self.scale)
        self.draw_scene()

    def on_mousewheel_linux_down_3d(self, event):
        self.scale *= ZOOM_OUT_FACTOR
        self.scale = max(MIN_SCALE, self.scale)
        self.draw_scene()

    def on_canvas_resize_3d(self, event):
        self.draw_scene()

    # --------------------------------------------------------
    # actions
    # --------------------------------------------------------
    def run_quantum(self, start):
        if start not in self.dist:
            self.pred_path = None
            self.actual_path = None
            self.draw_scene()
            self.info_var.set("預測路徑：找不到\n實際路徑：不可達\n處理時間：0.000 ms\n快取：—")
            return

        t0 = time.perf_counter()
        self.pred_path = path_quantum_only_3d(start, self.goal, self.maze, self.qchooser)
        self.actual_path = shortest_path_by_dist(start, self.goal, self.dist, self.adj)
        ms = (time.perf_counter() - t0) * 1000.0

        self.update_title()
        self.draw_scene()
        self.set_result(self.pred_path, self.actual_path, ms)

    def on_quantum(self):
        self.run_quantum(self.click_start)

    def on_clear(self):
        self.pred_path = None
        self.actual_path = None
        self.info_var.set("預測路徑：—\n實際路徑：—\n處理時間：—\n快取：—")
        self.draw_scene()

    def on_new(self):
        self.maze = Maze3D(W, H, D)
        self.maze.generate()
        save_maze(self.maze)

        self.fixed_start = self.maze.start
        self.goal = self.maze.goal
        self.click_start = self.fixed_start

        self.pred_path = None
        self.actual_path = None

        self.dist, self.adj = bfs_from_goal(self.maze)
        self.qchooser = LocalGroverChooser3D(seed=(self.maze.seed ^ 0xA5A5A5A5))

        self.update_title()
        self.draw_scene()
        self.info_var.set("預測路徑：—\n實際路徑：—\n處理時間：—\n快取：—")

    def on_toggle_view(self):
        self.view_mode = "3d" if self.view_mode == "layers" else "layers"
        self.bind_events_for_mode()
        self.update_title()
        self.update_legend()

        if self.view_mode == "layers":
            self.canvas.yview_moveto(0.0)

        self.draw_scene()

    def on_reset_view(self):
        self.yaw = INITIAL_YAW
        self.pitch = INITIAL_PITCH
        self.scale = INITIAL_SCALE
        if self.view_mode == "3d":
            self.draw_scene()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    App(root)
    root.geometry("1500x1050")
    root.minsize(1100, 800)
    root.mainloop()


if __name__ == "__main__":
    main()