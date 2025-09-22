from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

class WineResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[wine_features.fixed_acidity,
                    wine_features.volatile_acidity,
                    wine_features.citric_acid,
                    wine_features.residual_sugar,
                    wine_features.chlorides,
                    wine_features.free_sulfur_dioxide,
                    wine_features.total_sulfur_dioxide,
                    wine_features.density,
                    wine_features.pH,
                    wine_features.sulphates,
                    wine_features.alcohol]]

        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))