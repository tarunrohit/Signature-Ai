/**
 * A map of model IDs to their unique deployed backend URLs.
 * IMPORTANT: You must replace these with your actual Render URLs.
 */
const API_URL_MAP: { [key: string]: string } = {
  simplecnn: 'https://signature-ai-2.onrender.com',      // <-- REPLACE
  mobilenetv2: 'https://signature-ai-3.onrender.com',  // <-- REPLACE
  siamesenet: 'https://signature-ai-4.onrender.com',     // <-- REPLACE
};

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

/**
 * NEW: Sends a quick, lightweight request to wake up the server.
 * This prevents the main /verify request from timing out on a cold start.
 * @param baseUrl The base URL of the backend service to wake up.
 */
async function wakeUpServer(baseUrl: string): Promise<void> {
  try {
    console.log(`Pinging server at ${baseUrl} to wake it up...`);
    // We use the healthz endpoint as it's the fastest possible response.
    // We don't care about the response, just that the request completes.
    await fetch(`${baseUrl}/`, { method: 'GET' });
    console.log(`Server at ${baseUrl} is awake.`);
  } catch (error) {
    // We can ignore errors here, as the main fetch will handle them.
    console.warn(`Wake-up ping failed for ${baseUrl}, proceeding anyway.`);
  }
}

export async function verifySignature(
  modelId: string,
  image: File,
  referenceImage?: File
): Promise<VerificationResult> {
  const baseUrl = API_URL_MAP[modelId as keyof typeof API_URL_MAP];
  if (!baseUrl || baseUrl.includes('your-')) {
    throw new Error(`Backend URL for model '${modelId}' is not configured in api.ts.`);
  }
  
  // --- MODIFICATION: Wake up the server before sending the heavy request ---
  await wakeUpServer(baseUrl);
  // --- END OF MODIFICATION ---

  const formData = new FormData();
  formData.append('image', image);

  if (modelId === 'siamesenet' && referenceImage) {
    formData.append('reference_image', referenceImage);
  }
  
  const startTime = Date.now();
  try {
    const response = await fetch(`${baseUrl}/verify/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => null);
      const detail = errorData?.detail || `Server for ${modelId} responded with status: ${response.status}`;
      throw new Error(detail);
    }

    const result = await response.json();
    const endTime = Date.now();
    
    // Augment the result with frontend-specific data
    result.processingTime = result.processingTime || (endTime - startTime);
    
    if (modelId === 'siamesenet') {
      result.modelType = 'dual';
      if (result.distance !== undefined) {
         const similarity = Math.max(0, (1 - (result.distance / 1.5))) * 100;
         result.additionalMetrics = { similarity: similarity };
      }
      result.model = 'Siamese Networks';
    } else {
      result.modelType = 'single';
      if (modelId === 'simplecnn') result.model = 'SimpleCNN';
      if (modelId === 'mobilenetv2') result.model = 'MobileNetV2';
    }

    return result as VerificationResult;

  } catch (error: any) {
    console.error(`API call failed for model ${modelId}:`, error);
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running and reachable.');
    }
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}