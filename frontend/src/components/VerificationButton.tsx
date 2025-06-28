import React from 'react';
import { Shield, Loader2, Zap, Users } from 'lucide-react';

interface VerificationButtonProps {
  onClick: () => void;
  isLoading: boolean;
  disabled: boolean;
  modelType: 'single' | 'dual';
  selectedModel: string;
}

export default function VerificationButton({ 
  onClick, 
  isLoading, 
  disabled,
  modelType,
  selectedModel
}: VerificationButtonProps) {
  const getButtonContent = () => {
    if (isLoading) {
      return (
        <>
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Analyzing Signature...</span>
          <div className="flex gap-1">
            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '0ms' }}></div>
            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '200ms' }}></div>
            <div className="w-1 h-1 bg-white rounded-full animate-pulse" style={{ animationDelay: '400ms' }}></div>
          </div>
        </>
      );
    }

    const icon = modelType === 'dual' ? Users : selectedModel === 'mobilenetv2' ? Zap : Shield;
    const IconComponent = icon;
    
    return (
      <>
        <IconComponent className="w-5 h-5" />
        <span>
          {modelType === 'dual' ? 'Compare Signatures' : 'Verify Signature'}
        </span>
      </>
    );
  };

  return (
    <div className="space-y-3">
      <button
        onClick={onClick}
        disabled={disabled || isLoading}
        className={`
          w-full py-4 px-6 rounded-xl font-semibold text-lg transition-all duration-300
          flex items-center justify-center gap-3 relative overflow-hidden
          ${disabled || isLoading
            ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-neon-blue via-purple-500 to-neon-green hover:shadow-xl hover:shadow-neon-blue/30 text-white transform hover:scale-105'
          }
        `}
      >
        {!disabled && !isLoading && (
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 animate-shimmer"></div>
        )}
        {getButtonContent()}
      </button>
      
      {!disabled && !isLoading && (
        <div className="text-center">
          <p className="text-xs text-gray-400">
            {modelType === 'dual' 
              ? 'AI will compare both signatures for similarity analysis'
              : 'AI will analyze the signature for authenticity markers'
            }
          </p>
        </div>
      )}
    </div>
  );
}