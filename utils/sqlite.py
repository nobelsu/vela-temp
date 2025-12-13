import aiosqlite
from datetime import datetime, timezone, timedelta

async def getLatest(table_name, database_name):
    async with aiosqlite.connect(database_name) as db:
        async with db.execute(f"""
        SELECT content FROM {table_name}
        ORDER BY created_at DESC
        LIMIT 1
        """) as cursor:
            row = await cursor.fetchone()

    return row[0] if row else ""

async def upload(content, table_name, database_name):
    async with aiosqlite.connect(database_name) as db:
        await db.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        async with db.execute(f"SELECT COUNT(*) FROM {table_name}") as cur:
            (count,) = await cur.fetchone()
        if count == 0:
            await db.execute(f"DELETE FROM sqlite_sequence WHERE name='{table_name}'")
            await db.commit()

        await db.execute(f"""
            INSERT INTO {table_name} (content, created_at)
            VALUES (?, ?)
        """, (content, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()