import { config } from "@/lib/config";

export interface PropertyFeatures {
  beds: number;
  baths: number;
  type: string;
  area_in_sqft: number;
  city: string;
  furnishing: string;
  location: string;
}

export interface PredictionResponse {
  predicted_rent_aed: number;
  currency: string;
  frequency: string;
}

export async function predictRent(features: PropertyFeatures): Promise<PredictionResponse> {
  const response = await fetch(`${config.apiUrl}/predict`, {
    method: "POST",
    headers: { 
      "Content-Type": "application/json",
    },
    body: JSON.stringify(features),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    
    switch (response.status) {
      case 400:
        throw new Error(`Invalid request: ${errorBody}`);
      case 500:
        throw new Error('Server error. Please try again later.');
      default:
        throw new Error(`Prediction request failed (${response.status})`);
    }
  }

  const data = await response.json();
  
  if (typeof data.predicted_rent_aed !== 'number' || !Number.isFinite(data.predicted_rent_aed)) {
    throw new Error('Invalid prediction value received from server');
  }

  return data as PredictionResponse;
}