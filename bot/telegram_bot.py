import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_core.rag import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("telegram_bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OWNER_ID = os.getenv("TELEGRAM_OWNER_ID", "")
SHOW_SOURCES = os.getenv("TELEGRAM_SHOW_SOURCES", "0") == "1"

MAX_TELEGRAM_LEN = 3900


def is_allowed(user_id: Optional[int]) -> bool:
    if OWNER_ID is None or OWNER_ID == "":
        return True
    try:
        return user_id is not None and int(OWNER_ID) == int(user_id)
    except ValueError:
        return True


async def send_long(reply_fn, text: str):
    if not text:
        return
    for i in range(0, len(text), MAX_TELEGRAM_LEN):
        await reply_fn(text[i : i + MAX_TELEGRAM_LEN])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! Send me a question and I'll answer using the knowledge base."
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Just send a question in plain text. For best results, be specific."
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if msg is None or msg.text is None:
        return

    user_id = msg.from_user.id if msg.from_user else None
    if not is_allowed(user_id):
        await msg.reply_text("Sorry, this bot is in private prototype mode.")
        return

    question = msg.text.strip()
    if not question:
        return

    await msg.chat.send_action(action=ChatAction.TYPING)

    def run_rag():
        res = answer_question(question)
        answer = res.get("answer_text", "") if isinstance(res, dict) else str(res)
        sources = res.get("sources", []) if isinstance(res, dict) else []
        return answer, sources

    try:
        answer, sources = await asyncio.to_thread(run_rag)
    except Exception as e:
        log.exception("RAG error")
        await msg.reply_text(f"Error while answering: {e}")
        return

    if SHOW_SOURCES and sources:
        src_line = "Sources: " + ", ".join(str(s) for s in sources[:8])
        answer = answer.rstrip() + "\n\n" + src_line

    await send_long(msg.reply_text, answer or "I couldn't find an answer in the corpus.")


def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    log.info("Bot started (polling).")
    app.run_polling(close_loop=False)


if __name__ == "__main__":
    main()
