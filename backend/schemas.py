from pydantic import BaseModel, Field


class PropertyFeatures(BaseModel):
    """Raw input schema for a single rent prediction request."""

    beds: int = Field(..., ge=0, le=20, description="Number of bedrooms (0 = studio)")
    baths: int = Field(..., ge=0, le=20, description="Number of bathrooms")
    type: str = Field(..., description="Property type, e.g. 'Apartment', 'Villa'")
    area_in_sqft: float = Field(..., gt=0, description="Total area in square feet")
    city: str = Field(..., description="City, e.g. 'Dubai', 'Abu Dhabi'")
    furnishing: str = Field(..., description="'Furnished' or 'Unfurnished'")
    location: str = Field(..., description="Neighborhood/community, e.g. 'Dubai Marina'")

    class Config:
        json_schema_extra = {
            "example": {
                "beds": 2,
                "baths": 2,
                "type": "Apartment",
                "area_in_sqft": 1200,
                "city": "Dubai",
                "furnishing": "Unfurnished",
                "location": "Dubai Marina",
            }
        }


class PredictionResponse(BaseModel):
    predicted_rent_aed: float
    currency: str = "AED"
    frequency: str = "Yearly"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_engineer_loaded: bool