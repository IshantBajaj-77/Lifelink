
import argparse
import sys

from pymongo.errors import ServerSelectionTimeoutError

import aggregations
import crud_operations as crud
from db import DB_NAME, MONGO_URI, get_db
import ml_models
from schema import create_collections, create_indexes
from seed_data import seed


def main():
    parser = argparse.ArgumentParser(description="LifeLink - MongoDB + PyMongo + ML project")
    parser.add_argument("--keep", action="store_true",
                        help="do not drop/reseed the database")
    parser.add_argument("--only", choices=["validation", "crud", "agg", "ml"],
                        help="run a single section")
    args = parser.parse_args()

    try:
        db = get_db()
    except ServerSelectionTimeoutError:
        sys.exit(f"Could not connect to MongoDB at {MONGO_URI}. "
                 "Start mongod (or set MONGO_URI) and try again.")
    print(f"Connected to MongoDB, database: {DB_NAME}")

    if not args.keep:
        print("\n=== SETUP ===")
        create_collections(db)
        create_indexes(db)
        seed(db)

    if args.only in (None, "validation"):
        crud.demo_validation(db)
    if args.only in (None, "crud"):
        crud.demo_create(db)
        crud.demo_read(db)
        crud.demo_update(db)
        crud.demo_delete(db)
    if args.only in (None, "agg"):
        aggregations.run_all(db)
    if args.only in (None, "ml"):
        ml_models.run_ml_pipeline(db)

    print("\nDone.")


if __name__ == "__main__":
    main()
