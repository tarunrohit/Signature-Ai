// The backend URL. This MUST be your deployed Render URL.
const API_BASE_URL = 'https://signature-ai-1.onrender.com';

// The type for the response data from the backend.
export interface VerificationResult {
  isOriginal: boolean;
  confidence: number;
  model: string;
  processingTime: number;
  distance?: number; 
  modelType: 'single' | 'dual';
  additionalMetrics?: {
    similarity?: number;
  };
}

export async function verifySignature(
  modelId: string,
  image: File,
  referenceImage?: File
): Promise<VerificationResult> {
  const formData = new FormData();
  formData.append('model_id', modelId);
  formData.append('image', image);

  if (referenceImage) {
    formData.append('reference_image', referenceImage);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/verify/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      const detail = errorData?.detail || `Server responded with status: ${response.status}`;
      throw new Error(detail);
    }

    const result = await response.json();
    
    // Augment the result for the frontend UI components
    if (result.model === 'siamesenet') {
      result.modelType = 'dual';
      if (result.distance !== undefined) {
         const similarity = Math.max(0, (1 - (result.distance / 1.5))) * 100;
         result.additionalMetrics = { similarity: similarity };
      }
      result.model = 'Siamese Networks'; 
    } else {
      result.modelType = 'single';
      if (result.model === 'simplecnn') result.model = 'SimpleCNN';
      if (result.model === 'mobilenetv2') result.model = 'MobileNetV2';
    }

    return result as VerificationResult;

  } catch (error: any) {
    console.error("API call failed:", error);
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running.');
    }
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}