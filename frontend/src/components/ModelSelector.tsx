import React from 'react';
import { Brain, Cpu, Network, Smartphone } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (model: string) => void;
}

const models = [
  {
    id: 'mobilenetv2',
    name: 'MobileNetV2',
    description: 'Lightweight mobile-optimized architecture',
    icon: Smartphone,
    accuracy: '97.92%',
    inputType: 'single',
    features: ['Fast inference', 'Mobile optimized', 'High accuracy']
  },
  {
    id: 'simplecnn',
    name: 'SimpleCNN',
    description: 'Basic convolutional neural network',
    icon: Brain,
    accuracy: '95.44%',
    inputType: 'single',
    features: ['Quick processing', 'Reliable', 'Efficient']
  },
  {
    id: 'siamesenet',
    name: 'Siamese Networks',
    description: 'Twin network for signature comparison',
    icon: Network,
    accuracy: '96.8%',
    inputType: 'dual',
    features: ['Comparison based', 'Reference required', 'Similarity analysis']
  }
];

export default function ModelSelector({ selectedModel, onModelSelect }: ModelSelectorProps) {
  return (
    <div className="glass-card p-6">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
        <Brain className="w-6 h-6 text-neon-blue" />
        Select Verification Model
      </h2>
      
      <div className="space-y-4">
        {models.map((model) => {
          const IconComponent = model.icon;
          const isSelected = selectedModel === model.id;
          
          return (
            <button
              key={model.id}
              onClick={() => onModelSelect(model.id)}
              className={`
                w-full p-5 rounded-xl border-2 transition-all duration-300 text-left group
                ${isSelected 
                  ? 'border-neon-blue bg-gradient-to-br from-neon-blue/20 to-neon-green/10 shadow-lg shadow-neon-blue/25' 
                  : 'border-gray-700 hover:border-gray-600 bg-gray-800/30 hover:bg-gray-800/50'
                }
              `}
            >
              <div className="flex items-start gap-4">
                <div className={`
                  p-3 rounded-lg transition-all duration-300
                  ${isSelected 
                    ? 'bg-neon-blue/20 text-neon-blue' 
                    : 'bg-gray-700 text-gray-400 group-hover:bg-gray-600'
                  }
                `}>
                  <IconComponent className="w-6 h-6" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className={`text-lg font-semibold ${
                      isSelected ? 'text-neon-blue' : 'text-white'
                    }`}>
                      {model.name}
                    </h3>
                    <div className="flex items-center gap-2">
                      {model.inputType === 'dual' && (
                        <span className="px-2 py-1 text-xs bg-purple-500/20 text-purple-300 rounded-full border border-purple-500/30">
                          2 Images
                        </span>
                      )}
                      <span className="text-lg font-bold text-neon-green">
                        {model.accuracy}
                      </span>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-400 mb-3">
                    {model.description}
                  </p>
                  
                  <div className="flex flex-wrap gap-2">
                    {model.features.map((feature, index) => (
                      <span
                        key={index}
                        className={`
                          px-2 py-1 text-xs rounded-full border
                          ${isSelected 
                            ? 'bg-neon-blue/10 text-neon-blue border-neon-blue/30' 
                            : 'bg-gray-700/50 text-gray-300 border-gray-600'
                          }
                        `}
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
      
      <div className="mt-6 p-4 bg-gray-800/30 rounded-lg border border-gray-700">
        <h4 className="text-sm font-medium text-white mb-2">Model Information</h4>
        <p className="text-xs text-gray-400">
          {selectedModel === 'siamesenet' 
            ? 'Siamese Networks require both a reference signature and the test signature for comparison.'
            : 'Single-image models analyze the uploaded signature independently.'
          }
        </p>
      </div>
    </div>
  );
}