# LifeLink – Blood Donation & Request Management System (MongoDB + PyMongo + Machine Learning)

CSE494 – Intelligent NoSQL Databases, Academic Task 2.

**Problem:** Hospitals often struggle to find eligible donors of the right blood group quickly and estimate blood request fulfilment times accurately.
LifeLink stores donors, hospitals, donations and urgent requests in MongoDB, enforces data quality with `$jsonSchema` validation rules, answers operational questions using 10 aggregation pipelines, and integrates a **Scikit-Learn Machine Learning pipeline** (`ml_models.py`) to predict donor eligibility and request fulfilment time.

## Run it

```bash
pip install -r requirements.txt
# start MongoDB (local: `mongod`, or set MONGO_URI to an Atlas connection string)
python main.py                 # reset + seed + run validation, CRUD, aggregations, and ML pipeline
python main.py --only agg      # just aggregations (add --keep to reuse data)
python main.py --only ml       # just data preprocessing + ML models + evaluation metrics
python ml_models.py            # run ML pipeline directly (with automatic fallback if mongod is offline)
python menu.py                 # interactive CLI menu (includes CRUD, requests, reports, and ML evaluation)
mongosh lifelink_db mongo_queries.js   # same queries in mongosh
```

## Files

| File | Purpose |
|---|---|
| `schema.py` | Document design, `$jsonSchema` validators, indexes |
| `db.py` | PyMongo connection + output helper |
| `seed_data.py` | Synthetic dataset (6 hospitals, 40 donors, 150 donations, 25 requests) |
| `crud_operations.py` | Validation tests + Create/Read/Update/Delete demos |
| `aggregations.py` | 10 aggregation pipelines |
| `ml_models.py` | Data preprocessing (`pandas`/`scikit-learn`), ML models (Linear Regression, Logistic Regression, Decision Tree, Random Forest) & evaluation metrics (Accuracy, Precision, Recall, F1, R² Score, MAE) |
| `main.py` | Runs the complete demonstration end to end |
| `menu.py` | Interactive CLI menu for user-friendly CRUD, requests, reports, and ML predictions |
| `mongo_queries.js` | mongosh equivalents of all queries |

## Machine Learning Pipeline & Evaluation Metrics

- **Data Preprocessing:** Extracts documents from `donors`, `donations`, `hospitals`, and `requests` collections via PyMongo into `pandas.DataFrame`s; imputes missing `last_donation_date` values; engineers `days_since_last_donation`, `has_medical_condition`, `avg_hemoglobin`, and `hospital_stock_for_bg`; encodes categorical variables (`blood_group`, `gender`, `city`, `urgency`) using `OneHotEncoder`; scales numerical features via `StandardScaler`; and performs an 80/20 `train_test_split`.
- **Models Implemented:**
  - **Classification (Donor Eligibility & Response Prediction):** Logistic Regression, Decision Tree Classifier, Random Forest Classifier.
  - **Regression (Blood Request Fulfilment Time Prediction in Hours):** Linear Regression, Decision Tree Regressor, Random Forest Regressor.
- **Evaluation Metrics Computed:**
  - **Classification:** Accuracy, Precision, Recall, F1-Score.
  - **Regression:** R² Score (Coefficient of Determination), MAE (Mean Absolute Error), and SLA (<24h) Accuracy.

## Design decisions (for the viva)

- **Embedded:** `address` in donors, `blood_stock` in hospitals – read together, small, bounded.
- **Referenced:** `donations` link to `donor_id` and `hospital_id` – they grow without limit, so embedding would risk the 16 MB document cap. Joined with `$lookup`.
- **Denormalised:** `total_donations`, `last_donation_date`, `is_available` live on the donor so eligibility is one indexed query instead of a join.
- **Validation:** `$jsonSchema` with `validationLevel: strict`, `validationAction: error` (applies to inserts and updates), `additionalProperties: false`, enums, regex patterns, ranges. A unique index on `phone` / `license_no` adds a second integrity layer.
- **Indexes:** `(blood_group, city)` for donor search, `last_donation_date` for eligibility, `(donor_id, donation_date)` for history, `(status, urgency)` for the request queue. `explain()` is demonstrated in the Read section.
