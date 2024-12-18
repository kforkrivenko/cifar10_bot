import sqlite3
from scripts import drop_table_script, create_table_script, update_userid_info_script, insert_userid_info_script, get_userid_info, get_info

def init_db():
    conn = sqlite3.connect('data/botdatabase.db')
    cursor = conn.cursor()
    # cursor.execute(drop_table_script())
    cursor.execute(create_table_script())
    conn.commit()
    conn.close()

async def update_images(user_id, username):
    conn = sqlite3.connect('data/botdatabase.db')
    cursor = conn.cursor()
    
    # смотрим есть ли userid в БД
    cursor.execute(get_userid_info(user_id))
    result = cursor.fetchone()

    if result:
        image_count = result[1] + 1
        cursor.execute(update_userid_info_script(user_id, image_count, result[2]))
    else:
        image_count = 1
        cursor.execute(insert_userid_info_script(user_id, username, image_count, 0))

    conn.commit()
    conn.close()

async def update_images_correct(user_id, username):
    conn = sqlite3.connect('data/botdatabase.db')
    cursor = conn.cursor()
    
    # смотрим есть ли userid в БД
    cursor.execute(get_userid_info(user_id))
    result = cursor.fetchone()


    if result:
        image_count = result[2] + 1
        cursor.execute(update_userid_info_script(user_id, result[1], image_count))
    else:
        image_count = 1
        cursor.execute(insert_userid_info_script(user_id, username, 0, image_count))

    conn.commit()
    conn.close()

async def get_stats():
    conn = sqlite3.connect('data/botdatabase.db')
    cursor = conn.cursor()
    
    # смотрим есть ли userid в БД
    cursor.execute(get_info())
    result = cursor.fetchall()

    if result:
        return result
    else:
        return None

    conn.commit()
    conn.close()