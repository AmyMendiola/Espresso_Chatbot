import os
import json

DB_NAME = 'user_db.json'

def init():
    empty_db = []

    if not os.path.exists(DB_NAME):
        with open(DB_NAME, 'w') as file:
            json.dump(empty_db, file, indent=4)
        print('Database created.')
    else:
        print('Database already exists.')

def find(name):

    isReturning = False

    empty_user = {
        "name": name,
        "interested topics": [],
        "prior knowledge score": "",
        "favorite drink": "",
        "queries": []
    }

    with open(DB_NAME, 'r') as db_file:
        db = json.load(db_file)

    for user in db:
        if user["name"] == name:
            isReturning = True
            return user, db, isReturning
        
    db.append(empty_user)

    with open(DB_NAME, 'w') as file:
        json.dump(db, file, indent=4)

    print("New user added.")
    return empty_user, db, isReturning

def update(name, type, info):
    user, db, _ = find(name)

    #update
    for user in db:
        if (user.get('name') == name):
            if (type == "favorite drink" or type == "prior knowledge score"):
                user[type] = info
            elif (type == "interested topics" or type == "queries"):
                user[type].append(info)
    
    with open(DB_NAME, 'w') as file:
        json.dump(db, file, indent=4)

def get(name, type):
    user, db, _ = find(name)
    for user in db:
        if (user.get('name') == name):
            return user[type]

