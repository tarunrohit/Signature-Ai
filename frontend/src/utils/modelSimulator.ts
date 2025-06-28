// utils/modelSimulator.ts (Now a real API client)
import { getModelType } from './modelUtils'; // We'll create this small utility

// --- The backend URL. Change this if your backend is running elsewhere. ---
const API_BASE_URL = "http://127.0.0.1:8000";

// --- Type Definitions ---
// This is the data structure your UI components expect
export interface VerificationResult {
  isOriginal: boolean;
  confidence: number;
  model: string; // User-friendly model name
  processingTime: number;
  modelType: 'single' | 'dual';
  additionalMetrics?: {
    similarity?: number;
    risk_level?: 'low' | 'medium' | 'high' | 'very high';
  };
}

// This is the raw data structure returned from the backend
interface ApiVerificationResponse {
  isOriginal: boolean;
  confidence: number;
  model: string; // model_id like 'siamesenet'
  processingTime: number;
  distance?: number; // Only for siamese model
}

const modelDisplayNames: { [key: string]: string } = {
  simplecnn: 'SimpleCNN',
  mobilenetv2: 'MobileNetV2',
  siamesenet: 'Siamese Networks'
};

// --- Main API Function ---
export async function verifySignature(
  modelId: string,
  image: File,
  referenceImage?: File
): Promise<VerificationResult> {
  const formData = new FormData();
  formData.append("model_id", modelId);
  formData.append("image", image, image.name);

  if (modelId === "siamesenet" && referenceImage) {
    formData.append("reference_image", referenceImage, referenceImage.name);
  } else if (modelId === "siamesenet" && !referenceImage) {
    throw new Error("Reference image is required for the Siamese model.");
  }

  try {
    const response = await fetch(`${API_BASE_URL}/verify/`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "An error occurred during verification.");
    }

    const apiResult: ApiVerificationResponse = await response.json();
    
    // --- Enrich the backend response for the frontend UI ---
    const modelType = getModelType(apiResult.model);
    const riskConfidence = apiResult.isOriginal ? apiResult.confidence : 100 - apiResult.confidence;
    let riskLevel: VerificationResult['additionalMetrics']['risk_level'] = 'very high';
    if (riskConfidence > 50) riskLevel = 'high';
    if (riskConfidence > 75) riskLevel = 'medium';
    if (riskConfidence > 90) riskLevel = 'low';

    const verificationResult: VerificationResult = {
      ...apiResult,
      model: modelDisplayNames[apiResult.model] || apiResult.model,
      modelType: modelType,
      additionalMetrics: {
        risk_level: riskLevel,
        // Calculate similarity score from distance for Siamese network
        similarity: modelType === 'dual' && apiResult.distance !== undefined 
          ? Math.max(0, (1 - apiResult.distance / 1.5)) * 100 
          : undefined,
      }
    };

    return verificationResult;

  } catch (error) {
    console.error("API call failed:", error);
    // Re-throw the error so the component can catch it
    throw error;
  }
}