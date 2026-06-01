"""
AI Image Bot — Main Entry Point
Launches the Telegram bot with all configured handlers.
"""

import sys
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from config import logger, BOT_TOKEN
from telegram.ext import Application, ApplicationBuilder

# Import handlers
from bot.handlers.commands import register_handlers
from bot.services.database import init_db

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Bot is running smoothly! \xf0\x9f\x9a\x80")
    # Suppress logging to keep console clean
    def log_message(self, format, *args):
        pass

def start_dummy_server():
    """Starts a simple web server in a background thread to satisfy Render's port binding requirement."""
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    logger.info(f"Dummy web server started on port {port}")

async def post_init(application: Application) -> None:
    """Initialize resources after the application starts but before it fetches updates."""
    await init_db()
    logger.info("Database initialized successfully.")

def main() -> None:
    """Build and run the Telegram bot application."""
    logger.info("Starting AI Image Bot...")

    # Start dummy web server for Render Free Tier (Web Service)
    start_dummy_server()

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