"""
Core Bot Handlers — /start, menu buttons, image generation, and callback routing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.constants import ParseMode

from config import logger, TEMP_DIR, _RUNPOD_READY
from bot.services.database import (
    get_or_create_user,
    get_user_stats,
    get_global_stats,
    increment_generation_count,
    increment_edit_count,
    save_generation,
    get_generation_by_message,
)
from bot.services.gemini_handler import translate_and_expand
from bot.services.runpod_handler import call_runpod_text2img, call_runpod_img2img
from bot.services.image_editor import add_arabic_text_to_image

# ---------------------------------------------------------------------------
# Load messages from JSON
# ---------------------------------------------------------------------------

_MESSAGES_PATH = Path(__file__).resolve().parent.parent / "messages.json"

with open(_MESSAGES_PATH, "r", encoding="utf-8") as _f:
    MSG = json.load(_f)

# ---------------------------------------------------------------------------
# Keyboard setup
# ---------------------------------------------------------------------------

BTN_GENERATE = MSG["buttons"]["generate"]
BTN_GENERATE_DIRECT = MSG["buttons"]["generate_direct"]
BTN_STATS = MSG["buttons"]["stats"]

REPLY_KEYBOARD = ReplyKeyboardMarkup(
    [[KeyboardButton(BTN_GENERATE), KeyboardButton(BTN_GENERATE_DIRECT)], [KeyboardButton(BTN_STATS)]],
    resize_keyboard=True,
    one_time_keyboard=False,
)

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
# Simple in-memory dict to track user mode ("gemini" or "direct")
USER_MODE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# /start command
# ---------------------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — register user and show welcome + menu."""
    user = update.effective_user
    if user is None or update.message is None:
        return

    await get_or_create_user(
        user_id=str(user.id),
        username=user.username,
        first_name=user.first_name,
    )

    logger.info(f"User {user.id} (@{user.username}) started the bot.")

    await update.message.reply_text(
        MSG["welcome"],
        reply_markup=REPLY_KEYBOARD,
    )


# ---------------------------------------------------------------------------
# /help command
# ---------------------------------------------------------------------------

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show usage guide."""
    if update.message is None:
        return
    await update.message.reply_text(MSG["help"], reply_markup=REPLY_KEYBOARD)


# ---------------------------------------------------------------------------
# Core Generation Flow
# ---------------------------------------------------------------------------

async def _process_generation(
    update: Update,
    text: str,
    user_id: str,
    chat_id: str,
    is_edit: bool = False,
    source_message_id: int | None = None,
    source_image_path: str | None = None,
    mode: str = "gemini",
) -> None:
    """Process both new generations and edits."""
    base_msg = update.message if update.message else update.callback_query.message

    # Phase 1: Translate & expand (if mode is gemini)
    if mode == "gemini":
        status_msg = await base_msg.reply_text(MSG["status_translating"])
        try:
            english_prompt, overlay_texts = await translate_and_expand(text)
        except Exception as exc:
            logger.error(f"Gemini translation failed: {exc}")
            await status_msg.edit_text(MSG["errors"]["translation_failed"])
            return
    else:
        # Direct mode: bypass Gemini completely
        from bot.services.gemini_handler import extract_overlay_text
        status_msg = await base_msg.reply_text("⏳ جاري تجهيز وصفك المباشر...")
        english_prompt, overlay_texts = extract_overlay_text(text)
        if not english_prompt.strip():
            english_prompt = text.replace("[", "").replace("]", "").strip()

    if not _RUNPOD_READY:
        await status_msg.edit_text(MSG["errors"]["runpod_not_ready"])
        return

    status_text = MSG["status_editing"] if is_edit else MSG["status_generating"]
    await status_msg.edit_text(status_text)

    # Phase 2: Generate via RunPod
    image_path: Path | None = None
    try:
        image_filename = f"gen_{user_id}_{int(time.time())}.png"
        save_to = TEMP_DIR / image_filename

        if is_edit and source_image_path:
            img_bytes = Path(source_image_path).read_bytes()
            image_path = await call_runpod_img2img(
                prompt=english_prompt,
                image_bytes=img_bytes,
                save_to=save_to,
            )
        else:
            image_path = await call_runpod_text2img(
                prompt=english_prompt,
                save_to=save_to,
            )
    except Exception as exc:
        logger.error(f"RunPod generation failed: {exc}")
        error_detail = str(exc)[:200]
        await base_msg.reply_text(
            f"{MSG['errors']['generation_failed']}\n{error_detail}"
        )
        try:
            await status_msg.delete()
        except Exception:
            pass
        return

    # Phase 3: Add Arabic Text Overlay if requested
    if overlay_texts:
        try:
            full_text = " | ".join(overlay_texts)
            image_path = add_arabic_text_to_image(image_path, full_text)
        except Exception as e:
            logger.error(f"Failed to add text overlay: {e}")

    # Phase 4: Send Image
    inline_keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton(MSG["buttons"]["regen"], callback_data="regen"),
            InlineKeyboardButton(MSG["buttons"]["edit"], callback_data="magic_edit"),
        ]
    ])

    try:
        success_text = MSG["success_edited"] if is_edit else MSG["success_generated"]
        caption = f"{success_text}\n\n{text[:200]}"

        image_bytes = image_path.read_bytes()
        sent_message = await base_msg.reply_photo(
            photo=image_bytes,
            caption=caption,
            reply_markup=inline_keyboard,
            write_timeout=60,
            read_timeout=60,
            connect_timeout=60,
        )
    except Exception as exc:
        logger.error(f"Failed to send photo: {exc}")
        await base_msg.reply_text(
            f"{MSG['errors']['send_failed']}\nالسبب: {str(exc)}"
        )
        try:
            await status_msg.delete()
        except Exception:
            pass
        return

    # Phase 5: DB Save
    try:
        if is_edit:
            await increment_edit_count(user_id)
        else:
            await increment_generation_count(user_id)

        await save_generation(
            user_id=user_id,
            message_id=sent_message.message_id,
            chat_id=int(chat_id),
            original_prompt=text,
            translated_prompt=english_prompt,
            overlay_text=" | ".join(overlay_texts) if overlay_texts else None,
            image_path=str(image_path),
            generation_type="img2img" if is_edit else "text2img",
            source_message_id=source_message_id,
        )
    except Exception as exc:
        logger.error(f"Failed to save generation record: {exc}")

    try:
        await status_msg.delete()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Text message handler (menu buttons + prompt input + Reply to modify)
# ---------------------------------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Route text messages: menu buttons, prompt input, or edits."""
    if update.message is None:
        return

    text = update.message.text.strip() if update.message.text else ""
    user = update.effective_user
    if user is None:
        return

    user_id_str = str(user.id)
    await get_or_create_user(
        user_id=user_id_str,
        username=user.username,
        first_name=user.first_name,
    )

    if text == BTN_GENERATE:
        USER_MODE[user_id_str] = "gemini"
        await update.message.reply_text(MSG["prompt_ask"])
        return
        
    if text == BTN_GENERATE_DIRECT:
        USER_MODE[user_id_str] = "direct"
        await update.message.reply_text(MSG["prompt_ask_direct"])
        return

    if text == BTN_STATS:
        await show_stats(update, context)
        return

    # Check for Reply-to-Modify
    is_edit = False
    source_msg_id = None
    source_img_path = None

    if update.message.reply_to_message and update.message.reply_to_message.photo:
        reply_msg = update.message.reply_to_message
        if reply_msg.from_user and reply_msg.from_user.id == context.bot.id:
            gen_record = await get_generation_by_message(
                chat_id=reply_msg.chat_id,
                message_id=reply_msg.message_id,
            )
            if gen_record and gen_record["image_path"]:
                img_p = Path(gen_record["image_path"])
                if img_p.exists():
                    is_edit = True
                    source_msg_id = reply_msg.message_id
                    source_img_path = str(img_p)
                else:
                    await update.message.reply_text(MSG["errors"]["image_not_found"])
                    return

    # Fetch current user mode
    current_mode = USER_MODE.get(user_id_str, "gemini")

    await _process_generation(
        update=update,
        text=text,
        user_id=user_id_str,
        chat_id=str(update.effective_chat.id),
        is_edit=is_edit,
        source_message_id=source_msg_id,
        source_image_path=source_img_path,
        mode=current_mode,
    )


# ---------------------------------------------------------------------------
# /stats command
# ---------------------------------------------------------------------------

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_stats(update, context)


async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    user = update.effective_user
    if user is None:
        return

    stats = await get_user_stats(str(user.id))
    global_stats = await get_global_stats()

    joined_date = stats.get("joined_at")
    joined_str = joined_date[:10] if isinstance(joined_date, str) else "غير معروف"

    s = MSG["stats"]
    message = (
        f"{s['title']}\n\n"
        f"{s['generated']} {stats['total_generations']}\n"
        f"{s['edited']} {stats['total_edits']}\n"
        f"{s['joined']} {joined_str}\n\n"
        f"{s['global_title']}\n"
        f"{s['users']} {global_stats['total_users']}\n"
        f"{s['total_images']} {global_stats['total_generations']}\n"
        f"{s['generations']} {global_stats['text2img']} | {s['edits']} {global_stats['img2img']}"
    )

    await update.message.reply_text(message, reply_markup=REPLY_KEYBOARD)


# ---------------------------------------------------------------------------
# Callback query handler (for inline buttons)
# ---------------------------------------------------------------------------

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    user = update.effective_user

    if data == "regen" or data.startswith("regen|"):
        gen_record = await get_generation_by_message(
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
        )

        if gen_record and gen_record["original_prompt"]:
            original_prompt = gen_record["original_prompt"]
        else:
            original_prompt = data.split("|", 1)[1] if "|" in data else "صورة جديدة"

        await _process_generation(
            update=update,
            text=original_prompt,
            user_id=str(user.id),
            chat_id=str(update.effective_chat.id),
        )

    elif data == "magic_edit":
        await query.message.reply_text(MSG["edit_instructions"])

    else:
        logger.warning(f"Unknown callback data: {data}")


# ---------------------------------------------------------------------------
# Register all handlers on the Application
# ---------------------------------------------------------------------------

def register_handlers(app) -> None:
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_callback))
    logger.info("All handlers registered.")