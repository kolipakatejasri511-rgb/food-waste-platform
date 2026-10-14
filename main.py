from datetime import datetime
from pathlib import Path
import joblib

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Food Waste Donation Platform")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Database configuration
DATABASE_URL = (
    "sqlite:///" + str(BASE_DIR / "food_donations.db")
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()


class FoodListing(Base):
    __tablename__ = "food_listings"

    id = Column(Integer, primary_key=True, index=True)
    donor_name = Column(String, nullable=False)
    food_name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    quantity_kg = Column(Float, nullable=False)
    location = Column(String, nullable=False)
    pickup_deadline = Column(String, nullable=False)
    status = Column(String, default="Available")


class DonationRequest(Base):
    __tablename__ = "donation_requests"

    id = Column(Integer, primary_key=True, index=True)
    listing_id = Column(
        Integer,
        ForeignKey("food_listings.id"),
        nullable=False
    )
    ngo_name = Column(String, nullable=False)
    requested_kg = Column(Float, nullable=False)
    status = Column(String, default="Pending")


Base.metadata.create_all(bind=engine)

model_path = BASE_DIR / "food_surplus_model.joblib"
ml_model = (
    joblib.load(model_path)
    if model_path.exists()
    else None
)


@app.get("/")
def home(request: Request):
    db = SessionLocal()
    try:
        listings = db.query(FoodListing).filter(
            FoodListing.status == "Available"
        ).order_by(FoodListing.id.desc()).limit(6).all()

        total_donated = db.query(DonationRequest).filter(
            DonationRequest.status == "Completed"
        ).count()

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "listings": listings,
                "total_donated": total_donated
            }
        )
    finally:
        db.close()


@app.get("/donor")
def donor_page(request: Request):
    db = SessionLocal()
    try:
        listings = db.query(FoodListing).order_by(
            FoodListing.id.desc()
        ).all()

        return templates.TemplateResponse(
            request=request,
            name="donor.html",
            context={"listings": listings}
        )
    finally:
        db.close()


@app.post("/donor/add")
def add_food(
    donor_name: str = Form(...),
    food_name: str = Form(...),
    category: str = Form(...),
    quantity_kg: float = Form(...),
    location: str = Form(...),
    pickup_deadline: str = Form(...)
):
    if quantity_kg <= 0 or quantity_kg > 10000:
        return RedirectResponse(
            "/donor?error=quantity",
            status_code=303
        )

    deadline = pickup_deadline.strip()
    try:
        deadline_dt = datetime.fromisoformat(deadline)
        if deadline_dt <= datetime.now():
            return RedirectResponse(
                "/donor?error=deadline",
                status_code=303
            )
    except ValueError:
        return RedirectResponse(
            "/donor?error=deadline",
            status_code=303
        )

    db = SessionLocal()
    try:
        listing = FoodListing(
            donor_name=donor_name.strip(),
            food_name=food_name.strip(),
            category=category.strip(),
            quantity_kg=quantity_kg,
            location=location.strip(),
            pickup_deadline=deadline,
            status="Available"
        )
        db.add(listing)
        db.commit()
    finally:
        db.close()

    return RedirectResponse("/donor", status_code=303)


@app.get("/ngo")
def ngo_page(request: Request, location: str = ""):
    db = SessionLocal()
    try:
        query = db.query(FoodListing).filter(
            FoodListing.status == "Available"
        )

        if location.strip():
            query = query.filter(
                FoodListing.location.ilike(
                    "%" + location.strip() + "%"
                )
            )

        listings = query.order_by(
            FoodListing.pickup_deadline.asc()
        ).all()

        return templates.TemplateResponse(
            request=request,
            name="ngo.html",
            context={
                "listings": listings,
                "location": location
            }
        )
    finally:
        db.close()


@app.post("/ngo/request/{listing_id}")
def request_donation(
    listing_id: int,
    ngo_name: str = Form(...),
    requested_kg: float = Form(...)
):
    db = SessionLocal()
    try:
        listing = db.get(FoodListing, listing_id)

        if not listing or listing.status != "Available":
            return RedirectResponse("/ngo", status_code=303)

        existing = db.query(DonationRequest).filter(
            DonationRequest.listing_id == listing_id,
            DonationRequest.status.in_(["Pending", "Accepted"])
        ).all()

        reserved_kg = sum(r.requested_kg for r in existing)
        remaining = listing.quantity_kg - reserved_kg

        if requested_kg <= 0 or requested_kg > remaining:
            return RedirectResponse(
                "/ngo?error=quantity",
                status_code=303
            )

        donation = DonationRequest(
            listing_id=listing_id,
            ngo_name=ngo_name.strip(),
            requested_kg=requested_kg,
            status="Pending"
        )
        db.add(donation)
        db.commit()

        return RedirectResponse("/requests", status_code=303)
    finally:
        db.close()


@app.get("/requests")
def requests_page(request: Request):
    db = SessionLocal()
    try:
        records = db.query(
            DonationRequest, FoodListing
        ).join(
            FoodListing,
            DonationRequest.listing_id == FoodListing.id
        ).order_by(DonationRequest.id.desc()).all()

        return templates.TemplateResponse(
            request=request,
            name="requests.html",
            context={"records": records}
        )
    finally:
        db.close()


@app.post("/requests/{request_id}/update")
def update_request(
    request_id: int,
    status: str = Form(...)
):
    allowed = {"Accepted", "Rejected", "Completed"}

    if status not in allowed:
        return RedirectResponse("/requests", status_code=303)

    db = SessionLocal()
    try:
        donation = db.get(DonationRequest, request_id)

        if not donation or donation.status not in {
            "Pending", "Accepted"
        }:
            return RedirectResponse("/requests", status_code=303)

        listing = db.get(FoodListing, donation.listing_id)

        if status == "Completed":
            if donation.status != "Accepted":
                return RedirectResponse("/requests", status_code=303)

            if listing is None:
                return RedirectResponse("/requests", status_code=303)

            # Calculate completed quantity before this update.
            completed_kg = sum(
                r.requested_kg
                for r in db.query(DonationRequest).filter(
                    DonationRequest.listing_id == listing.id,
                    DonationRequest.status == "Completed"
                ).all()
            )

            donation.status = "Completed"
            completed_kg += donation.requested_kg

            if completed_kg >= listing.quantity_kg:
                listing.status = "Donated"
            if donation.status != "Accepted":
                return RedirectResponse("/requests", status_code=303)

            if listing is None:
                return RedirectResponse("/requests", status_code=303)

            donation.status = "Completed"

            completed_kg = sum(
                r.requested_kg
                for r in db.query(DonationRequest).filter(
                    DonationRequest.listing_id == listing.id,
                    DonationRequest.status == "Completed"
                ).all()
            ) + donation.requested_kg

            if completed_kg >= listing.quantity_kg:
                listing.status = "Donated"

        elif status == "Accepted":
            if donation.status != "Pending" or listing is None:
                return RedirectResponse("/requests", status_code=303)

            other_reservations = db.query(DonationRequest).filter(
                DonationRequest.listing_id == listing.id,
                DonationRequest.id != donation.id,
                DonationRequest.status.in_(["Accepted", "Completed"])
            ).all()

            used = sum(r.requested_kg for r in other_reservations)

            if used + donation.requested_kg > listing.quantity_kg:
                return RedirectResponse("/requests", status_code=303)

            donation.status = "Accepted"

        else:
            donation.status = "Rejected"

        db.commit()
        return RedirectResponse("/requests", status_code=303)
    finally:
        db.close()


@app.get("/predict")
def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={"prediction": None}
    )


@app.post("/predict")
def predict_food(
    request: Request,
    day: str = Form(...),
    category: str = Form(...),
    customers: int = Form(...),
    prepared_kg: float = Form(...),
    event: int = Form(...)
):
    if (
        ml_model is None
        or day not in [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]
        or category not in [
            "Rice", "Meals", "Snacks", "Bread", "Vegetables"
        ]
        or customers < 1
        or customers > 10000
        or prepared_kg <= 0
        or prepared_kg > 10000
        or event not in [0, 1]
    ):
        return templates.TemplateResponse(
            request=request,
            name="predict.html",
            context={
                "prediction": None,
                "error": "Check the inputs and train the model first."
            },
            status_code=400
        )

    import pandas as pd

    sample = pd.DataFrame([{
        "day": day,
        "category": category,
        "customers": customers,
        "prepared_kg": prepared_kg,
        "event": event
    }])

    prediction = float(ml_model.predict(sample)[0])
    prediction = max(0.0, min(prediction, prepared_kg))

    return templates.TemplateResponse(
        request=request,
        name="predict.html",
        context={
            "prediction": round(prediction, 2),
            "error": None
        }
    )
