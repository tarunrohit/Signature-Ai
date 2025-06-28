// The backend URL. Change this if your backend is running elsewhere.
const API_BASE_URL = 'http://localhost:8000';

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

  // Add reference image to the form data if it exists (for Siamese model)
  if (referenceImage) {
    formData.append('reference_image', referenceImage);
  }

  try {
    const response = await fetch(`${API_BASE_URL}/verify/`, {
      method: 'POST',
      body: formData,
      // Note: Do not set 'Content-Type' when using FormData.
      // The browser automatically sets it with the correct boundary.
    });

    if (!response.ok) {
      // Get a more specific error message from the backend response
      const errorData = await response.json().catch(() => null);
      const detail = errorData?.detail || `Server responded with status: ${response.status}`;
      throw new Error(detail);
    }

    const result = await response.json();
    
    // --- Augment the result for the frontend UI components ---
    if (result.model === 'siamesenet') {
      result.modelType = 'dual';
      if (result.distance !== undefined) {
         // Convert distance (lower is better) to a similarity percentage for display
         const similarity = Math.max(0, (1 - (result.distance / 1.5))) * 100;
         result.additionalMetrics = { similarity: similarity };
      }
      result.model = 'Siamese Networks'; // Use display-friendly name
    } else {
      result.modelType = 'single';
      if (result.model === 'simplecnn') result.model = 'SimpleCNN';
      if (result.model === 'mobilenetv2') result.model = 'MobileNetV2';
    }

    return result as VerificationResult;

  } catch (error: any) {
    // This handles network errors (like the original "Failed to fetch")
    console.error("API call failed:", error);
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running.');
    }
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}