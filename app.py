from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.car_price.pipeline.predection_pipeline import PredictionPipeline, CustomData  # your corrected pipeline

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # put your HTML here

# Initialize pipeline once
pipeline = PredictionPipeline()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    model: str = Form(...),
    year: int = Form(...),
    transmission: str = Form(...),
    mileage: int = Form(...),
    fuelType: str = Form(...),
    tax: int = Form(...),
    mpg: float = Form(...),
    engineSize: float = Form(...)
):
    try:
        # Strip trailing spaces
        car = CustomData(
            model=model.strip(),
            year=year,
            transmission=transmission.strip(),
            mileage=mileage,
            fuelType=fuelType.strip(),
            tax=tax,
            mpg=mpg,
            engineSize=engineSize
        )

        # Prediction
        result = pipeline.predict_with_details(car.get_data_as_dict())

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction_formatted": f"£{result['predicted_price']:,.2f}",
                "lower_bound_formatted": f"£{result['confidence_interval']['lower']:,.2f}",
                "upper_bound_formatted": f"£{result['confidence_interval']['upper']:,.2f}",
                "input_data": car,  # pre-fill form
                "error": None
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction_formatted": None,
                "lower_bound_formatted": None,
                "upper_bound_formatted": None,
                "input_data": car if 'car' in locals() else None,
                "error": str(e)
            }
        )
