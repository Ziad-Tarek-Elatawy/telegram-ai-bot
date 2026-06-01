"""
Database Layer — Async SQLite via aiosqlite.
Manages users and generation history for the AI Image Bot.
"""

from __future__ import annotations

import aiosqlite
from config import DB_PATH, logger

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,
    username        TEXT,
    first_name      TEXT,
    total_generations INTEGER NOT NULL DEFAULT 0,
    total_edits     INTEGER NOT NULL DEFAULT 0,
    joined_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    message_id      INTEGER NOT NULL,
    chat_id         INTEGER NOT NULL,
    original_prompt TEXT,
    translated_prompt TEXT,
    overlay_text    TEXT,
    image_path      TEXT,
    generation_type TEXT NOT NULL DEFAULT 'text2img',
    source_message_id INTEGER,
    strength        REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_generations_user_id ON generations(user_id);
CREATE INDEX IF NOT EXISTS idx_generations_message_id ON generations(chat_id, message_id);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

async def get_db() -> aiosqlite.Connection:
    """Return a new database connection (caller must close it)."""
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    db = await get_db()
    try:
        await db.executescript(SCHEMA_SQL)
        await db.commit()
        logger.info("Database initialized successfully.")
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

async def get_or_create_user(
    user_id: str, username: str | None = None, first_name: str | None = None
) -> dict:
    """Fetch existing user or insert a new row. Returns user dict."""
    db = await get_db()
    try:
        cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        if row is None:
            await db.execute(
                "INSERT INTO users (user_id, username, first_name) VALUES (?, ?, ?)",
                (user_id, username or "", first_name or ""),
            )
            await db.commit()
            cursor = await db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
        else:
            # Touch last_active
            await db.execute(
                "UPDATE users SET last_active = CURRENT_TIMESTAMP, username = ?, first_name = ? WHERE user_id = ?",
                (username or "", first_name or "", user_id),
            )
            await db.commit()
        return dict(row)
    finally:
        await db.close()


async def increment_generation_count(user_id: str) -> None:
    """Bump the user's generation counter."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET total_generations = total_generations + 1, last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
    finally:
        await db.close()


async def increment_edit_count(user_id: str) -> None:
    """Bump the user's edit counter."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE users SET total_edits = total_edits + 1, last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Generation operations
# ---------------------------------------------------------------------------

async def save_generation(
    user_id: str,
    message_id: int,
    chat_id: int,
    original_prompt: str | None = None,
    translated_prompt: str | None = None,
    overlay_text: str | None = None,
    image_path: str | None = None,
    generation_type: str = "text2img",
    source_message_id: int | None = None,
    strength: float | None = None,
) -> int:
    """Save a generation record. Returns the new record ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            INSERT INTO generations
                (user_id, message_id, chat_id, original_prompt, translated_prompt,
                 overlay_text, image_path, generation_type, source_message_id, strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, message_id, chat_id, original_prompt, translated_prompt,
                overlay_text, image_path, generation_type, source_message_id, strength,
            ),
        )
        await db.commit()
        return cursor.lastrowid
    finally:
        await db.close()


async def get_generation_by_message(chat_id: int, message_id: int) -> dict | None:
    """Retrieve a generation record by chat + message ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT * FROM generations WHERE chat_id = ? AND message_id = ?",
            (chat_id, message_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()


async def get_user_stats(user_id: str) -> dict:
    """Return aggregated stats for a user."""
    db = await get_db()
    try:
        cursor = await db.execute(
            """
            SELECT
                u.total_generations,
                u.total_edits,
                u.joined_at,
                u.last_active,
                COUNT(g.id) AS total_records
            FROM users u
            LEFT JOIN generations g ON g.user_id = u.user_id
            WHERE u.user_id = ?
            GROUP BY u.user_id
            """,
            (user_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return {"total_generations": 0, "total_edits": 0, "total_records": 0}
        return dict(row)
    finally:
        await db.close()


async def get_global_stats() -> dict:
    """Return platform-wide stats."""
    db = await get_db()
    try:
        total_users = await db.execute("SELECT COUNT(*) FROM users")
        total_users = (await total_users.fetchone())[0]

        total_gens = await db.execute("SELECT COUNT(*) FROM generations")
        total_gens = (await total_gens.fetchone())[0]

        total_text2img = await db.execute(
            "SELECT COUNT(*) FROM generations WHERE generation_type = 'text2img'"
        )
        total_text2img = (await total_text2img.fetchone())[0]

        total_img2img = await db.execute(
            "SELECT COUNT(*) FROM generations WHERE generation_type = 'img2img'"
        )
        total_img2img = (await total_img2img.fetchone())[0]

        return {
            "total_users": total_users,
            "total_generations": total_gens,
            "text2img": total_text2img,
            "img2img": total_img2img,
        }
    finally:
        await db.close()