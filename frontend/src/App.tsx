// App.tsx
import React, { useState } from 'react';
import { Fingerprint } from 'lucide-react';
import ModelSelector from './components/ModelSelector';
import SignatureUpload from './components/SignatureUpload';
import VerificationButton from './components/VerificationButton';
import ResultDisplay from './components/ResultDisplay';
// --- MODIFIED: Import the real API function and the utility ---
import { verifySignature, type VerificationResult } from './utils/api';
import { getModelType } from './utils/modelUtils';
// --- End of Modification ---

function App() {
  const [selectedModel, setSelectedModel] = useState('mobilenetv2');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [referenceImage, setReferenceImage] = useState<File | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [result, setResult] = useState<VerificationResult | null>(null);

  const modelType = getModelType(selectedModel);

  const handleImageSelect = (file: File, type: 'test' | 'reference' = 'test') => {
    if (type === 'test') {
      setSelectedImage(file);
    } else {
      setReferenceImage(file);
    }
    setResult(null); // Clear previous results
  };

  const handleClearImage = (type: 'test' | 'reference' = 'test') => {
    if (type === 'test') {
      setSelectedImage(null);
    } else {
      setReferenceImage(null);
    }
    setResult(null);
  };

  const handleModelSelect = (model: string) => {
    setSelectedModel(model);
    setResult(null);
    // Clear reference image if switching away from Siamese network
    if (getModelType(model) === 'single') {
      setReferenceImage(null);
    }
  };

  const handleVerification = async () => {
    if (!selectedImage) return;
    
    // Check if Siamese network requires reference image
    if (modelType === 'dual' && !referenceImage) {
      alert('Please upload both reference and test signatures for the Siamese Networks model.');
      return;
    }

    setIsVerifying(true);
    setResult(null);

    try {
      // --- MODIFIED: Call the real API function ---
      const verificationResult = await verifySignature(
        selectedModel, 
        selectedImage, 
        referenceImage || undefined
      );
      setResult(verificationResult);
    } catch (error: any) {
      console.error('Verification failed:', error);
      alert(`Verification failed: ${error.message}`);
    } finally {
      setIsVerifying(false);
    }
  };

  const isVerificationDisabled = !selectedImage || (modelType === 'dual' && !referenceImage);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-6">
            <div className="relative">
              <Fingerprint className="w-16 h-16 text-neon-blue animate-pulse-glow" />
              <div className="absolute inset-0 w-16 h-16 text-neon-blue opacity-30 animate-ping"></div>
            </div>
            <div>
              <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-neon-blue via-purple-400 to-neon-green bg-clip-text text-transparent">
                SignatureAI
              </h1>
              <div className="h-1 w-32 bg-gradient-to-r from-neon-blue to-neon-green rounded-full mx-auto mt-2"></div>
            </div>
          </div>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Advanced signature verification using state-of-the-art deep learning models with 
            <span className="text-neon-green font-semibold"> up to 97.92% accuracy</span>
          </p>
          
          <div className="flex justify-center gap-6 mt-6 text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-blue rounded-full"></div>
              <span>Real-time Analysis</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-neon-green rounded-full"></div>
              <span>Multiple AI Models</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span>Comparison Technology</span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Left Column - Model Selection */}
            <div className="lg:col-span-1">
              <ModelSelector
                selectedModel={selectedModel}
                onModelSelect={handleModelSelect}
              />
            </div>

            {/* Right Column - Upload and Results */}
            <div className="lg:col-span-2 space-y-6">
              <SignatureUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                referenceImage={referenceImage}
                onClearImage={handleClearImage}
                modelType={modelType}
              />

              <VerificationButton
                onClick={handleVerification}
                isLoading={isVerifying}
                disabled={isVerificationDisabled}
                modelType={modelType}
                selectedModel={selectedModel}
              />

              <ResultDisplay result={result} />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-20">
          <div className="max-w-4xl mx-auto">
            <div className="grid md:grid-cols-3 gap-8 mb-8">
              <div className="text-center">
                <h3 className="text-neon-blue font-semibold mb-2">MobileNetV2</h3>
                <p className="text-sm text-gray-400">Fast & efficient mobile-optimized architecture</p>
              </div>
              <div className="text-center">
                <h3 className="text-neon-green font-semibold mb-2">SimpleCNN</h3>
                <p className="text-sm text-gray-400">Reliable convolutional neural network</p>
              </div>
              <div className="text-center">
                <h3 className="text-purple-400 font-semibold mb-2">Siamese Networks</h3>
                <p className="text-sm text-gray-400">Advanced comparison-based verification</p>
              </div>
            </div>
            <p className="text-gray-500">
              Powered by advanced neural networks and computer vision technology
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;