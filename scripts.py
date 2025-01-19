def drop_table_script(): 
    return '''
        DROP TABLE messages
    '''

def create_table_script(): 
    return '''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            images INTEGER,
            images_correct INTEGER
        )
    '''

def update_userid_info_script(user_id, images, images_correct):
    return f'''
        UPDATE messages
        SET images = {images}, images_correct = {images_correct}
        WHERE user_id = {user_id}
    '''

def insert_userid_info_script(user_id, username, images, images_correct):
    return f'''
        INSERT INTO messages
        (user_id, username, images, images_correct)
        VALUES ({user_id}, '{username}', {images}, {images_correct})
    '''

def get_userid_info(user_id):
    return f'''
        SELECT username, images, images_correct
        FROM messages
        WHERE user_id = {user_id}
    '''

def get_info():
    return f'''
        SELECT username, images, images_correct
        FROM messages
        ORDER BY images DESC
    '''

