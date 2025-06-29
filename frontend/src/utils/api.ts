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
 * Sends a quick, lightweight request to "wake up" a sleeping server on a free tier.
 * @param baseUrl The base URL of the backend service to wake up.
 */
async function wakeUpServer(baseUrl: string): Promise<void> {
  try {
    console.log(`Pinging server at ${baseUrl} to wake it up...`);
    // We use the root endpoint as it's the fastest possible response.
    await fetch(`${baseUrl}/`, { method: 'GET' });
    console.log(`Server at ${baseUrl} is awake.`);
  } catch (error) {
    console.warn(`Wake-up ping failed for ${baseUrl}, proceeding anyway.`);
  }
}

/**
 * Makes a POST request to the appropriate backend service to verify a signature.
 */
export async function verifySignature(
  modelId: string,
  image: File,
  referenceImage?: File
): Promise<VerificationResult> {
  const baseUrl = API_URL_MAP[modelId as keyof typeof API_URL_MAP];
  if (!baseUrl || baseUrl.includes('your-')) {
    throw new Error(`Backend URL for model '${modelId}' is not configured in api.ts.`);
  }
  
  // First, send a quick request to wake up the server if it's sleeping.
  await wakeUpServer(baseUrl);

  const formData = new FormData();
  formData.append('image', image);

  if (modelId === 'siamesenet' && referenceImage) {
    formData.append('reference_image', referenceImage);
  }

  // --- NEW: Add a timeout controller for the main request ---
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, 60000); // 60 seconds timeout

  const startTime = Date.now();
  try {
    const response = await fetch(`${baseUrl}/verify/`, {
      method: 'POST',
      body: formData,
      signal: controller.signal, // Link the controller to the fetch request
    });

    // Clear the timeout if the request completes in time
    clearTimeout(timeoutId);

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
    clearTimeout(timeoutId); // Also clear timeout on error
    console.error(`API call failed for model ${modelId}:`, error);

    if (error.name === 'AbortError') {
      throw new Error('The server is taking too long to respond. Please try again.');
    }
    if (error.message.includes('Failed to fetch')) {
         throw new Error('Connection to the server failed. Please ensure the backend is running and reachable.');
    }
    throw new Error(error.message || 'An unknown error occurred during verification.');
  }
}