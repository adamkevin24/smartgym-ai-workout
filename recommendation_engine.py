"""
Smart Gym - AI Workout Recommendation Engine
==============================================
Rule-based recommendation engine (bukan ML) yang menghasilkan jadwal workout
mingguan dari profile user (age, gender, weight, height, fitness_level, fitness_goal).

Alur:
    User Profile (level, goal)
        -> LEVEL menentukan STRUKTUR SPLIT (berapa hari latihan, muscle group per hari)
        -> GOAL menentukan PARAMETER latihan (sets, reps, rest)
        -> Dataset exercise (Kaggle) di-filter berdasarkan `target` muscle
        -> Exercise dipilih dengan aturan: minimal 1 compound + sisanya isolation
        -> Output: workout plan siap disimpan BE ke tabel `schedules`

CATATAN KETERBATASAN DATA:
    Tabel `workout_histories` di ERD saat ini hanya menyimpan `focus_muscle`,
    BUKAN nama exercise spesifik. Artinya sistem tidak bisa tahu persis exercise
    apa saja yang sudah pernah dilakukan user di minggu-minggu sebelumnya.
    Anti-repeat exercise di sini hanya berlaku DALAM SATU BATCH generate
    (1 minggu penuh), bukan lintas minggu. Kalau mau anti-repeat lintas minggu,
    ERD perlu tabel tambahan untuk mencatat exercise per sesi (misal
    `workout_history_exercises`) -> ini didiskusikan dulu ke tim BE.
"""

import random
from datetime import date, timedelta, datetime
import pandas as pd


# ---------------------------------------------------------------------------
# 1. KONFIGURASI RULE (hasil desain kita)
# ---------------------------------------------------------------------------

# Level -> struktur split mingguan. Setiap hari berisi list "muscle label" kita
# sendiri (bukan target dataset langsung). List kosong [] = hari rest.
LEVEL_SPLITS = {
    "easy": [
        ["chest", "triceps"],
        [],
        ["back", "biceps"],
        [],
        ["legs", "shoulders", "abs"],
        [],
        [],
    ],
    "medium": [
        ["chest", "triceps", "shoulders"],
        ["legs", "abs"],
        [],
        ["back", "biceps"],
        ["legs", "calves"],
        [],
        [],
    ],
    "intermediate": [
        ["chest"],
        ["back"],
        ["legs"],
        ["shoulders", "abs"],
        ["biceps", "triceps"],
        [],
        [],
    ],
}

# Mapping "muscle label" kita -> nilai `target` yang ada di dataset Kaggle.
MUSCLE_TARGET_MAP = {
    "chest": ["pectorals"],
    "back": ["lats", "upper back"],
    "shoulders": ["delts"],
    "biceps": ["biceps"],
    "triceps": ["triceps"],
    "legs": ["quads", "hamstrings", "glutes"],
    "calves": ["calves"],
    "abs": ["abs"],
    "traps": ["traps"],
}

# Goal -> parameter latihan (sets, reps, rest dalam detik). Berupa range (min, max).
GOAL_PARAMS = {
    "gain": {"sets": (3, 4), "reps": (6, 10), "rest": (90, 120)},
    "lose": {"sets": (3, 3), "reps": (12, 15), "rest": (30, 45)},
    "healthy": {"sets": (3, 3), "reps": (10, 12), "rest": (60, 60)},
}

# Level -> jumlah exercise per muscle group per sesi.
N_EXERCISES_PER_LEVEL = {"easy": 3, "medium": 4, "intermediate": 5}

# Keyword buat nge-tag compound vs isolation dari nama exercise.
COMPOUND_KEYWORDS = [
    "squat", "deadlift", "press", "row", "pull-up", "pulldown",
    "snatch", "clean", "lunge", "dip",
]
ISOLATION_KEYWORDS = ["curl", "extension", "raise", "fly", "shrug", "crusher"]


# ---------------------------------------------------------------------------
# 2. DATASET LOADING + TAGGING
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset Kaggle dan tambahkan kolom `tag` (compound/isolation/other)."""
    df = pd.read_csv(csv_path)

    def tag_exercise(name: str) -> str:
        n = name.lower()
        if any(k in n for k in COMPOUND_KEYWORDS):
            return "compound"
        if any(k in n for k in ISOLATION_KEYWORDS):
            return "isolation"
        return "other"

    df["tag"] = df["name"].apply(tag_exercise)
    return df


# ---------------------------------------------------------------------------
# 3. EXERCISE SELECTION
# ---------------------------------------------------------------------------

def select_exercises(df: pd.DataFrame, muscle_label: str, n: int,
                      already_used: set) -> list[dict]:
    """
    Pilih n exercise untuk satu muscle label.

    Aturan (v2 - anti-monoton):
      - Minimal 1 compound (kalau tersedia).
      - Sisanya dipilih dengan SELANG-SELING antara pool isolation/other dan
        pool compound, bukan "habiskan isolation dulu baru compound".
        Ini mencegah kasus seperti target `lats` yang pool isolation-nya
        tipis (cuma varian pullover) -> tanpa selang-seling, 3-4 slot
        tersisa semua kepaksa ambil pullover yang mirip-mirip. Dengan
        selang-seling, begitu pool isolation abis, otomatis diisi compound
        lain (row/pulldown) yang jauh lebih variatif.
      - Exercise yang sudah dipakai di `already_used` diprioritaskan untuk
        DIHINDARI dulu; baru kalau pool unik sudah habis, boleh repeat.
    """
    targets = MUSCLE_TARGET_MAP.get(muscle_label, [])
    pool = df[df["target"].isin(targets)].copy()
    if pool.empty:
        return []

    def ordered_records(sub_df: pd.DataFrame) -> list[dict]:
        """Shuffle, lalu taruh yang belum pernah dipakai (`already_used`) di depan."""
        shuffled = sub_df.sample(frac=1) if not sub_df.empty else sub_df
        unused = shuffled[~shuffled["name"].isin(already_used)]
        used = shuffled[shuffled["name"].isin(already_used)]
        return list(unused.to_dict("records")) + list(used.to_dict("records"))

    compound_list = ordered_records(pool[pool["tag"] == "compound"])
    other_list = ordered_records(pool[pool["tag"] != "compound"])

    result: list[dict] = []
    picked_names: set[str] = set()

    def take_from(lst: list[dict], pos: list[int]) -> bool:
        """Ambil 1 item unik dari lst mulai index pos[0]. Return True kalau berhasil."""
        while pos[0] < len(lst):
            item = lst[pos[0]]
            pos[0] += 1
            if item["name"] not in picked_names:
                result.append(item)
                picked_names.add(item["name"])
                return True
        return False

    c_pos, o_pos = [0], [0]

    # 1 compound wajib duluan (kalau ada)
    if compound_list:
        take_from(compound_list, c_pos)

    # selang-seling: isolation/other dulu, lalu compound, bergantian
    lists_cycle = [other_list, compound_list]
    positions_cycle = [o_pos, c_pos]
    i = 0
    while len(result) < n:
        li = i % 2
        got = take_from(lists_cycle[li], positions_cycle[li])
        if not got:
            got = take_from(lists_cycle[1 - li], positions_cycle[1 - li])
        if not got:
            break  # kedua pool sudah habis total (lebih kecil dari n)
        i += 1

    return result[:n]


# ---------------------------------------------------------------------------
# 4. PERSONALISASI DARI HISTORY (recovery-based reordering)
# ---------------------------------------------------------------------------

DEFAULT_RECENCY_DAYS = 999  # dianggap "sudah lama sekali" kalau belum pernah dilatih


def _parse_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)).date()


def compute_muscle_recency(history: list[dict], today: date) -> dict[str, int]:
    """
    Dari list workout_histories (dict dengan keys `focus_muscle` dan
    `completed_at`), hitung "sudah berapa hari sejak muscle ini terakhir
    dilatih". Muscle yang belum pernah muncul di history tidak ada di dict
    hasil (caller anggap sebagai DEFAULT_RECENCY_DAYS -- "belum pernah").

    `focus_muscle` diasumsikan berformat sama seperti yang kita tulis ke
    `schedules.focus_muscle`, yaitu muscle label dipisah koma, misal:
    "chest,triceps,shoulders".
    """
    last_trained: dict[str, date] = {}
    for entry in history or []:
        raw_muscle = entry.get("focus_muscle")
        raw_date = entry.get("completed_at")
        if not raw_muscle or not raw_date:
            continue
        completed = _parse_date(raw_date)
        for muscle in str(raw_muscle).split(","):
            muscle = muscle.strip().lower()
            if not muscle:
                continue
            if muscle not in last_trained or completed > last_trained[muscle]:
                last_trained[muscle] = completed

    return {m: (today - d).days for m, d in last_trained.items()}


def reorder_split_by_recovery(split: list[list[str]],
                               recency: dict[str, int]) -> list[list[str]]:
    """
    Susun ulang URUTAN hari latihan (bukan isi/kombinasinya) berdasarkan
    seberapa "pulih" tiap kombinasi muscle. Posisi hari REST tidak berubah
    -- yang berubah cuma kombinasi muscle mana yang ditaruh di hari mana.

    Readiness sebuah hari = recency muscle yang PALING BARU dilatih di
    kombinasi itu (bottleneck). Kombinasi paling "siap" (readiness paling
    besar / paling lama nggak dilatih) ditaruh di slot paling awal.

    Kalau `recency` kosong (user baru, belum ada history), urutan tidak
    berubah -- karena semua kombinasi dapat readiness default yang sama,
    dan Python sort stabil (urutan asli dipertahankan kalau nilainya sama).
    """
    workout_slots = [(i, muscles) for i, muscles in enumerate(split) if muscles]

    def readiness(muscles: list[str]) -> int:
        return min(recency.get(m, DEFAULT_RECENCY_DAYS) for m in muscles)

    sorted_slots = sorted(workout_slots, key=lambda x: readiness(x[1]), reverse=True)

    new_split = list(split)
    original_positions = [pos for pos, _ in workout_slots]
    for pos, (_, muscles) in zip(original_positions, sorted_slots):
        new_split[pos] = muscles
    return new_split


# ---------------------------------------------------------------------------
# 5. MAIN: GENERATE WORKOUT PLAN
# ---------------------------------------------------------------------------

def generate_workout_plan(df: pd.DataFrame, user_profile: dict,
                           history: list[dict] | None = None,
                           start_date: date | None = None) -> list[dict]:
    """
    user_profile: dict dengan keys minimal:
        - fitness_level: "easy" | "medium" | "intermediate"
        - fitness_goal:  "gain" | "lose" | "healthy"
    (age, gender, weight, height belum dipakai di versi ini -- disimpan
    untuk pengembangan V2, misal buat menentukan berat angkatan/kalori)

    history: list of dict dari `workout_histories`, tiap dict minimal punya
        `focus_muscle` (string, dipisah koma) dan `completed_at` (date/str).
        None atau [] -> user baru, urutan split pakai default (cold start).

    Return: list of dict, 1 dict per hari (7 hari), siap dipetakan ke tabel `schedules`.
    """
    level = user_profile["fitness_level"]
    goal = user_profile["fitness_goal"]

    if level not in LEVEL_SPLITS:
        raise ValueError(f"fitness_level tidak dikenal: {level}")
    if goal not in GOAL_PARAMS:
        raise ValueError(f"fitness_goal tidak dikenal: {goal}")

    if start_date is None:
        start_date = date.today()

    n_ex = N_EXERCISES_PER_LEVEL[level]
    params = GOAL_PARAMS[goal]

    # personalisasi: susun ulang urutan hari berdasarkan recovery dari history
    recency = compute_muscle_recency(history, today=start_date)
    split = reorder_split_by_recovery(LEVEL_SPLITS[level], recency)

    already_used: set[str] = set()  # anti-repeat dalam 1 minggu generate ini
    schedule = []

    for i, day_muscles in enumerate(split):
        current_date = start_date + timedelta(days=i)

        if not day_muscles:
            schedule.append({
                "date": current_date.isoformat(),
                "title": "Rest Day",
                "focus_muscle": None,
                "status": "pending",
                "exercises": [],
            })
            continue

        day_exercises = []
        for muscle in day_muscles:
            picked = select_exercises(df, muscle, n_ex, already_used)
            for ex in picked:
                sets = random.randint(*params["sets"])
                reps = random.randint(*params["reps"])
                rest = random.randint(*params["rest"])
                day_exercises.append({
                    "name": ex["name"],
                    "target": ex["target"],
                    "muscle_group": muscle,
                    "equipment": ex["equipment"],
                    "sets": sets,
                    "reps": reps,
                    "rest_seconds": rest,
                })
                already_used.add(ex["name"])

        schedule.append({
            "date": current_date.isoformat(),
            "title": " + ".join(m.capitalize() for m in day_muscles),
            "focus_muscle": ",".join(day_muscles),
            "status": "pending",
            "exercises": day_exercises,
        })

    return schedule


# ---------------------------------------------------------------------------
# 6. DEMO
# ---------------------------------------------------------------------------

def print_plan(plan: list[dict], label: str) -> None:
    print(f"--- {label} ---")
    for day in plan:
        exercises_summary = (
            ", ".join(ex["name"] for ex in day["exercises"])
            if day["exercises"] else "(rest)"
        )
        print(f"{day['date']} | {day['title']:<28} -> {exercises_summary}")
    print()


if __name__ == "__main__":
    dataset = load_dataset("exercise_dataset.csv")

    sample_user = {
        "age": 22,
        "gender": "male",
        "weight": 70,
        "height": 175,
        "fitness_level": "medium",
        "fitness_goal": "gain",
    }
    week_start = date(2026, 9, 21)

    # Skenario 1: user baru, belum ada history -> urutan default
    plan_new_user = generate_workout_plan(dataset, sample_user, history=None,
                                           start_date=week_start)
    print_plan(plan_new_user, "User BARU (tanpa history) - urutan default")

    # Skenario 2: user lama, baru aja latihan Chest+Triceps+Shoulders 1 hari
    # sebelum tanggal generate -> harusnya digeser ke belakang, Legs+Abs
    # (belum pernah dilatih) yang maju ke depan.
    fake_history = [
        {
            "focus_muscle": "chest,triceps,shoulders",
            "completed_at": (week_start - timedelta(days=1)).isoformat(),
        },
    ]
    plan_with_history = generate_workout_plan(dataset, sample_user,
                                               history=fake_history,
                                               start_date=week_start)
    print_plan(plan_with_history,
               "User LAMA (Chest+Triceps+Shoulders baru dilatih kemarin)")

    # Contoh detail 1 hari lengkap (sets/reps/rest/equipment)
    print("--- Detail hari pertama (skenario user lama) ---")
    first_day = plan_with_history[0]
    print(f"{first_day['date']} - {first_day['title']} [{first_day['status']}]")
    for ex in first_day["exercises"]:
        print(f"   - {ex['name']} ({ex['muscle_group']}) "
              f"| {ex['sets']}x{ex['reps']} rest {ex['rest_seconds']}s "
              f"| equipment: {ex['equipment']}")
