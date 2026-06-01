"""
AI Image Bot — Main Entry Point
Launches the Telegram bot with all configured handlers.
"""

import sys

from config import logger, BOT_TOKEN
from telegram.ext import Application, ApplicationBuilder

# Import handlers
from bot.handlers.commands import register_handlers
from bot.services.database import init_db


async def post_init(application: Application) -> None:
    """Initialize resources after the application starts but before it fetches updates."""
    await init_db()
    logger.info("Database initialized successfully.")

def main() -> None:
    """Build and run the Telegram bot application."""
    logger.info("Starting AI Image Bot...")

    # Build the application with post_init
    app: Application = ApplicationBuilder().token(BOT_TOKEN).post_init(post_init).build()

    # Register all handlers
    register_handlers(app)

    # Start polling (this function blocks and manages the event loop itself)
    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        sys.exit(0)