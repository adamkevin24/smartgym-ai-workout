"""
Smart Gym - AI Workout Recommendation API
==========================================
Bungkus recommendation_engine.py jadi REST API pakai FastAPI, supaya BE
(bahasa/framework apa pun) bisa manggil lewat HTTP biasa.

Cara jalanin:
    pip install fastapi uvicorn pandas
    uvicorn api:app --reload --port 8000

Setelah jalan, buka http://localhost:8000/docs untuk lihat dokumentasi
interaktif (Swagger UI) -- bisa langsung dicoba dari situ tanpa Postman.

Endpoint utama: POST /generate-workout-plan
"""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommendation_engine import load_dataset, generate_workout_plan


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Gym - AI Workout Recommendation API",
    description="Rule-based recommendation engine untuk workout plan mingguan.",
    version="1.0.0",
)

# CORS: biar FE/BE dari domain lain boleh manggil API ini.
# Untuk production, ganti allow_origins jadi domain spesifik, jangan "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dataset di-load SEKALI aja waktu server nyala, bukan tiap ada request
# (biar cepat -- baca CSV tiap request itu boros).
dataset = load_dataset("exercise_dataset.csv")


# ---------------------------------------------------------------------------
# Schema request & response (pakai Pydantic, otomatis divalidasi FastAPI)
# ---------------------------------------------------------------------------

class HistoryEntry(BaseModel):
    """Satu baris dari tabel workout_histories."""
    focus_muscle: str = Field(..., examples=["chest,triceps,shoulders"])
    completed_at: str = Field(..., examples=["2026-09-20"])


class UserProfile(BaseModel):
    """Sesuai kolom di tabel user_profiles."""
    age: Optional[int] = None
    gender: Optional[str] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    fitness_level: str = Field(..., examples=["medium"],
                                description="easy | medium | intermediate")
    fitness_goal: str = Field(..., examples=["gain"],
                               description="gain | lose | healthy")


class GenerateRequest(BaseModel):
    user_profile: UserProfile
    history: list[HistoryEntry] = []
    start_date: Optional[str] = Field(
        None, description="Format YYYY-MM-DD. Kosongkan untuk pakai hari ini.")


class ExerciseItem(BaseModel):
    name: str
    target: str
    muscle_group: str
    equipment: str
    sets: int
    reps: int
    rest_seconds: int


class DayPlan(BaseModel):
    date: str
    title: str
    focus_muscle: Optional[str]
    status: str
    exercises: list[ExerciseItem]


class GenerateResponse(BaseModel):
    week_start: str
    days: list[DayPlan]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Cek server hidup dan dataset ke-load. Panggil ini dulu buat tes koneksi."""
    return {"status": "ok", "total_exercises_in_dataset": len(dataset)}


@app.post("/generate-workout-plan", response_model=GenerateResponse)
def generate_plan(request: GenerateRequest):
    """
    Input: profile user + (opsional) history + (opsional) tanggal mulai.
    Output: jadwal 7 hari, siap disimpan BE ke tabel `schedules`
    (dan detail exercise-nya bisa ditampilkan langsung ke FE atau
    disimpan sebagian ke kolom `notes` di workout_histories nanti).
    """
    try:
        parsed_start_date = (
            date.fromisoformat(request.start_date) if request.start_date else None
        )
    except ValueError:
        raise HTTPException(status_code=400,
                             detail="start_date harus format YYYY-MM-DD")

    history_as_dicts = [h.model_dump() for h in request.history]

    try:
        plan = generate_workout_plan(
            dataset,
            user_profile=request.user_profile.model_dump(),
            history=history_as_dicts,
            start_date=parsed_start_date,
        )
    except ValueError as e:
        # fitness_level / fitness_goal nggak dikenal -> 400, bukan 500
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "week_start": plan[0]["date"],
        "days": plan,
    }
