"""
seed_data.py - Generates a reproducible synthetic dataset for LifeLink:
6 hospitals, 40 donors, 150 donations, and 25 blood requests.
Uses a fixed random seed (42) so results are reproducible.
"""
import random
from datetime import datetime, timedelta, timezone

try:
    from pymongo import UpdateOne
except ImportError:
    UpdateOne = None

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
BG_WEIGHTS = [0.22, 0.04, 0.30, 0.04, 0.08, 0.02, 0.26, 0.04]

HOSPITALS_INFO = [
    ("Sanjeevani Multispeciality Hospital", "PB-BB-1001", "Amritsar", "9999900001"),
    ("Jeevan Jyoti Blood Bank", "PB-BB-1002", "Ludhiana", "9999900002"),
    ("Raksha Care Hospital", "PB-BB-1003", "Patiala", "9999900003"),
    ("Asha Kiran Medical Centre", "PB-BB-1004", "Jalandhar", "9999900004"),
    ("Prem Nagar Civil Blood Centre", "PB-BB-1005", "Bathinda", "9999900005"),
    ("Navjeevan Health Institute", "PB-BB-1006", "Mohali", "9999900006"),
]

PINCODES = {
    "Amritsar": "143001",
    "Ludhiana": "141001",
    "Patiala": "147001",
    "Jalandhar": "144001",
    "Bathinda": "151001",
    "Mohali": "160055",
}

FIRST_NAMES = [
    "Rohit", "Pooja", "Gurmeet", "Kiran", "Vikram", "Gurpreet", "Rupinder",
    "Manpreet", "Amandeep", "Tanvi", "Sukhdeep", "Navdeep", "Arjun", "Karan",
    "Simran", "Harpreet", "Jaspreet", "Rajवीर", "Anjali", "Neha"
]
FIRST_NAMES[17] = "Rajveer"

LAST_NAMES = [
    "Sandhu", "Kaur", "Dhillon", "Chopra", "Singh", "Sidhu", "Verma", "Gill", "Mehta", "Sharma"
]

CONDITIONS = ["Diabetes", "Hypertension"]


def generate_dataset(now=None):
    """Return synthetic (hospitals, donors, donations, requests) lists."""
    rng = random.Random(42)
    if now is None:
        now = datetime.now(timezone.utc)

    hospitals = []
    for idx, (name, lic, city, phone) in enumerate(HOSPITALS_INFO):
        stock = {
            "A+": rng.randint(6, 16),
            "A-": rng.randint(2, 6),
            "B+": rng.randint(8, 18),
            "B-": rng.randint(5, 11),
            "AB+": rng.randint(3, 9),
            "AB-": rng.randint(3, 8),
            "O+": rng.randint(10, 22),
            "O-": rng.randint(2, 6),
        }
        hospitals.append({
            "_id": f"hosp_{idx+1}",
            "name": name,
            "license_no": lic,
            "city": city,
            "phone": phone,
            "blood_stock": stock,
        })

    cities = [h[2] for h in HOSPITALS_INFO]
    donors = []
    for i in range(40):
        fn = rng.choice(FIRST_NAMES)
        ln = rng.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        bg = rng.choices(BLOOD_GROUPS, weights=BG_WEIGHTS, k=1)[0]
        age = rng.randint(19, 62)
        gender = rng.choice(["Male", "Female"])
        city = rng.choice(cities)
        phone = f"98140{10000 + i:05d}"
        med = [rng.choice(CONDITIONS)] if rng.random() < 0.15 else []
        reg_on = now - timedelta(days=rng.randint(180, 720))

        donors.append({
            "_id": f"donor_{i+1}",
            "name": name,
            "blood_group": bg,
            "age": age,
            "gender": gender,
            "phone": phone,
            "email": f"{fn.lower()}.{ln.lower()}{i+1}@example.com",
            "city": city,
            "address": {"district": city, "pincode": PINCODES[city]},
            "is_available": True,
            "total_donations": 0,
            "last_donation_date": None,
            "medical_conditions": med,
            "registered_on": reg_on,
        })

    donations = []
    for j in range(150):
        d = rng.choice(donors)
        h = rng.choice(hospitals)
        days_ago = rng.randint(2, 360)
        don_date = now - timedelta(days=days_ago)
        status = "completed" if rng.random() < 0.94 else "deferred"
        hb = round(rng.uniform(12.6, 17.2), 1)
        units = 1

        donations.append({
            "donor_id": d["_id"],
            "hospital_id": h["_id"],
            "blood_group": d["blood_group"],
            "units": units,
            "hemoglobin": hb,
            "donation_date": don_date,
            "status": status,
        })
        if status == "completed":
            d["total_donations"] += 1
            if d["last_donation_date"] is None or don_date > d["last_donation_date"]:
                d["last_donation_date"] = don_date
            h["blood_stock"][d["blood_group"]] += units

    cutoff = now - timedelta(days=90)
    for d in donors:
        if d["last_donation_date"] is not None and d["last_donation_date"] > cutoff:
            d["is_available"] = False
        else:
            d["is_available"] = True

    requests = []
    urgencies = ["critical", "high", "normal"]
    statuses = ["open", "fulfilled", "cancelled"]
    for k in range(25):
        h = rng.choice(hospitals)
        fn = rng.choice(FIRST_NAMES)
        ln = rng.choice(LAST_NAMES)
        bg = rng.choices(BLOOD_GROUPS, weights=BG_WEIGHTS, k=1)[0]
        units_needed = rng.randint(1, 5)
        urgency = rng.choices(urgencies, weights=[0.25, 0.35, 0.40], k=1)[0]
        status = rng.choices(statuses, weights=[0.48, 0.36, 0.16], k=1)[0]
        req_on = now - timedelta(days=rng.randint(1, 45), hours=rng.randint(1, 20))
        doc = {
            "hospital_id": h["_id"],
            "patient_name": f"{fn} {ln}",
            "blood_group": bg,
            "units_needed": units_needed,
            "urgency": urgency,
            "status": status,
            "requested_on": req_on,
        }
        if status == "fulfilled":
            base_hrs = {"critical": 12, "high": 32, "normal": 48}[urgency]
            hrs = max(2, int(rng.gauss(base_hrs, 8)))
            doc["fulfilled_on"] = req_on + timedelta(hours=hrs)
        requests.append(doc)

    return hospitals, donors, donations, requests


def seed(db):
    """Populate MongoDB collections (hospitals, donors, donations, requests)."""
    hospitals, donors, donations, requests = generate_dataset()

    # Strip temporary string _id before inserting into MongoDB so ObjectIds are created
    hosp_docs = [{k: v for k, v in h.items() if k != "_id"} for h in hospitals]
    hosp_res = db.hospitals.insert_many(hosp_docs)
    hosp_id_map = {hospitals[i]["_id"]: hosp_res.inserted_ids[i] for i in range(len(hospitals))}

    donor_docs = []
    for d in donors:
        doc = {k: v for k, v in d.items() if k != "_id"}
        doc["total_donations"] = 0
        doc["last_donation_date"] = None
        doc["is_available"] = True
        donor_docs.append(doc)
    donor_res = db.donors.insert_many(donor_docs)
    donor_id_map = {donors[i]["_id"]: donor_res.inserted_ids[i] for i in range(len(donors))}

    don_docs = []
    for don in donations:
        don_docs.append({
            "donor_id": donor_id_map[don["donor_id"]],
            "hospital_id": hosp_id_map[don["hospital_id"]],
            "blood_group": don["blood_group"],
            "units": don["units"],
            "hemoglobin": float(don["hemoglobin"]),
            "donation_date": don["donation_date"],
            "status": don["status"],
        })
    db.donations.insert_many(don_docs)

    # Bulk update donors with computed total_donations, last_donation_date, is_available
    if UpdateOne is not None:
        donor_ops = [
            UpdateOne(
                {"_id": donor_id_map[d["_id"]]},
                {"$set": {
                    "total_donations": d["total_donations"],
                    "last_donation_date": d["last_donation_date"],
                    "is_available": d["is_available"],
                }}
            )
            for d in donors
        ]
        db.donors.bulk_write(donor_ops)

    req_docs = []
    for r in requests:
        rdoc = {k: v for k, v in r.items() if k != "hospital_id"}
        rdoc["hospital_id"] = hosp_id_map[r["hospital_id"]]
        req_docs.append(rdoc)
    db.requests.insert_many(req_docs)

    counts = {
        "hospitals": len(hosp_docs),
        "donors": len(donor_docs),
        "donations": len(don_docs),
        "requests": len(req_docs),
    }
    print(f"Seeded: {counts}")
    return counts
