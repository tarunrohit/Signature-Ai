// The backend URL. This MUST be your deployed Hugging Face Space URL.
const API_BASE_URL = 'https://tarun5098-signature-ai.hf.space';

// The type for the response data from the backend.
// It's augmented with frontend-specific fields for the ResultDisplay component.
export interface VerificationResult {
  isOriginal: boolean;
  confidence: number;
  model: string;
  processingTime: number;
  distance?: number; // Raw distance from the Siamese model
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

  // Define the specific endpoint for the verification API
  const endpoint = '/verify/';

  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
      // Note: Do not set a 'Content-Type' header when using FormData with fetch.
      // The browser automatically sets it to 'multipart/form-data' with the correct boundary.
    });

    if (!response.ok) {
      // Try to get a more specific error message from the backend response
      const errorData = await response.json().catch(() => ({})); // Use empty object as fallback
      const detail = errorData?.detail || `Server responded with status: ${response.status}`;
      throw new Error(detail);
    }

    const result = await response.json();
    
    // --- Augment the result with frontend-specific data for the UI components ---
    if (result.model === 'siamesenet') {
      result.modelType = 'dual';
      if (result.distance !== undefined) {
         // Convert distance (where lower is better) to a more intuitive similarity percentage for display
         const similarity = Math.max(0, (1 - (result.distance / 1.5))) * 100;
         result.additionalMetrics = { similarity: similarity };
      }
      result.model = 'Siamese Networks'; // Use a more display-friendly name
    } else {
      result.modelType = 'single';
      if (result.model === 'simplecnn') result.model = 'SimpleCNN';
      if (result.model === 'mobilenetv2') result.model = 'MobileNetV2';
    }

    return result as VerificationResult;

  } catch (error: any) {
    console.error("API call failed:", error);
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running and accessible.');
    }
    // Re-throw the more specific error message from the backend if available
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}