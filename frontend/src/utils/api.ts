/**
 * A map of model IDs to their unique deployed backend URLs.
 * IMPORTANT: You must replace these placeholder URLs with the actual URLs
 * from your deployed services on Render or Railway.
 */
const API_URL_MAP: { [key: string]: string } = {
  simplecnn: 'https://your-simple-cnn-backend.onrender.com',    // <-- REPLACE WITH YOUR SIMPLECNN URL
  mobilenetv2: 'https://your-mobilenet-backend.onrender.com',  // <-- REPLACE WITH YOUR MOBILENET URL
  siamesenet: 'https://your-siamese-backend.onrender.com',   // <-- REPLACE WITH YOUR SIAMESE URL
};

/**
 * The structure of the data expected back from the API, augmented with
 * frontend-specific fields for the ResultDisplay component.
 */
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

/**
 * Makes a POST request to the appropriate backend service to verify a signature.
 *
 * @param modelId The ID of the model to use ('simplecnn', 'mobilenetv2', 'siamesenet').
 * @param image The signature image file to be tested.
 * @param referenceImage An optional reference image file, required for the Siamese model.
 * @returns A Promise that resolves to a VerificationResult object.
 */
export async function verifySignature(
  modelId: string,
  image: File,
  referenceImage?: File
): Promise<VerificationResult> {
  // 1. Select the correct backend URL based on the model ID.
  const baseUrl = API_URL_MAP[modelId];
  if (!baseUrl || baseUrl.includes('your-backend-name')) {
    // This provides a helpful error if the URL hasn't been configured.
    throw new Error(`Backend URL for model '${modelId}' is not configured. Please update api.ts.`);
  }

  // 2. Prepare the form data for the request.
  const formData = new FormData();
  formData.append('image', image);

  // The Siamese network backend is the only one that needs a reference image.
  if (modelId === 'siamesenet' && referenceImage) {
    formData.append('reference_image', referenceImage);
  }

  try {
    // 3. Make the network request to the selected backend.
    const response = await fetch(`${baseUrl}/verify/`, {
      method: 'POST',
      body: formData,
      // Note: Do not set 'Content-Type' when using FormData.
      // The browser automatically sets it with the correct multipart boundary.
    });

    // 4. Handle non-successful responses (e.g., 404, 500).
    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      const detail = errorData?.detail || `Server for ${modelId} responded with status: ${response.status}`;
      throw new Error(detail);
    }

    // 5. Process the successful JSON response.
    const result = await response.json();
    
    // Add a placeholder for processingTime if the backend doesn't provide it.
    if (result.processingTime === undefined) {
      result.processingTime = 0; // Default value
    }
    
    // 6. Augment the result with frontend-specific data for the UI components.
    if (modelId === 'siamesenet') {
      result.modelType = 'dual';
      if (result.distance !== undefined) {
         // Convert distance (lower is better) to a user-friendly similarity percentage.
         const similarity = Math.max(0, (1 - (result.distance / 1.5))) * 100;
         result.additionalMetrics = { similarity: similarity };
      }
      result.model = 'Siamese Networks'; // Use a display-friendly name
    } else {
      result.modelType = 'single';
      if (modelId === 'simplecnn') result.model = 'SimpleCNN';
      if (modelId === 'mobilenetv2') result.model = 'MobileNetV2';
    }

    return result as VerificationResult;

  } catch (error: any) {
    // 7. Handle network errors (e.g., backend is down, CORS issue).
    console.error(`API call failed for model ${modelId}:`, error);
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running and reachable.');
    }
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}