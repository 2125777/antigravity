import json
import os
from datetime import datetime

DB_FILE = "parking_db.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

def record_entry(plate_number):
    db = load_db()
    # Check if car is already in (entry without exit)
    if plate_number in db and db[plate_number]["exit_time"] is None:
        return f"Car {plate_number} is already inside."
    
    db[plate_number] = {
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time": None,
        "is_paid": False
    }
    save_db(db)
    return f"Entry recorded for {plate_number}"

def mark_as_paid(plate_number):
    db = load_db()
    if plate_number in db:
        db[plate_number]["is_paid"] = True
        save_db(db)
        return True
    return False

def record_exit(plate_number):
    db = load_db()
    if plate_number not in db:
        return False, "Vehicle record not found!"
    
    if db[plate_number]["exit_time"] is not None:
        return False, f"Vehicle {plate_number} has already exited."

    if not db[plate_number]["is_paid"]:
        return False, "Please pay your ticket!"
    
    db[plate_number]["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_db(db)
    return True, f"Exit recorded for {plate_number}. Gate opening..."

def get_all_records():
    db = load_db()
    # Convert to list for easier display in tables
    records = []
    for plate, info in db.items():
        records.append({
            "Plate Number": plate,
            "Entry Time": info["entry_time"],
            "Exit Time": info["exit_time"],
            "Paid": "Yes" if info["is_paid"] else "No"
        })
    return records

def clear_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    return True
