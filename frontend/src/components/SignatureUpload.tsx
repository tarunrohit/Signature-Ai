import React, { useCallback, useState } from 'react';
import { Upload, X, FileImage, Users } from 'lucide-react';

interface SignatureUploadProps {
  onImageSelect: (file: File, type?: 'test' | 'reference') => void;
  selectedImage: File | null;
  referenceImage?: File | null;
  onClearImage: (type?: 'test' | 'reference') => void;
  modelType: 'single' | 'dual';
}

interface ImageUploadAreaProps {
  title: string;
  subtitle: string;
  file: File | null;
  preview: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  dragActive: boolean;
  onDragHandlers: {
    onDragEnter: (e: React.DragEvent) => void;
    onDragLeave: (e: React.DragEvent) => void;
    onDragOver: (e: React.DragEvent) => void;
    onDrop: (e: React.DragEvent) => void;
  };
}

function ImageUploadArea({ 
  title, 
  subtitle, 
  file, 
  preview, 
  onFileSelect, 
  onClear, 
  dragActive,
  onDragHandlers 
}: ImageUploadAreaProps) {
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileSelect(e.target.files[0]);
    }
  };

  if (file && preview) {
    return (
      <div className="relative">
        <h4 className="text-sm font-medium text-white mb-3">{title}</h4>
        <div className="relative rounded-xl overflow-hidden bg-gray-800/50 border border-gray-700">
          <img
            src={preview}
            alt={`${title} preview`}
            className="w-full h-40 object-contain bg-gray-900/50"
          />
          
          <button
            onClick={onClear}
            className="absolute top-3 right-3 p-2 bg-red-500 hover:bg-red-600 rounded-full transition-colors duration-200 shadow-lg"
          >
            <X className="w-4 h-4 text-white" />
          </button>
          
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-gray-900/90 to-transparent p-3">
            <p className="text-xs text-gray-300 truncate">{file.name}</p>
            <p className="text-xs text-gray-400">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h4 className="text-sm font-medium text-white mb-3">{title}</h4>
      <div
        className={`
          relative border-2 border-dashed rounded-xl p-6 text-center transition-all duration-300
          ${dragActive 
            ? 'border-neon-blue bg-neon-blue/10 scale-105' 
            : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/30'
          }
        `}
        {...onDragHandlers}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        
        <Upload className={`w-8 h-8 mx-auto mb-3 transition-colors duration-300 ${
          dragActive ? 'text-neon-blue' : 'text-gray-400'
        }`} />
        <h5 className="text-sm font-medium text-white mb-1">
          {subtitle}
        </h5>
        <p className="text-xs text-gray-400">
          JPG, PNG files
        </p>
      </div>
    </div>
  );
}

export default function SignatureUpload({ 
  onImageSelect, 
  selectedImage, 
  referenceImage,
  onClearImage,
  modelType
}: SignatureUploadProps) {
  const [dragActive, setDragActive] = useState<'test' | 'reference' | null>(null);
  const [testPreview, setTestPreview] = useState<string | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);

  const createDragHandlers = (type: 'test' | 'reference') => ({
    onDragEnter: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(type);
    },
    onDragLeave: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(null);
    },
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    },
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(null);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith('image/')) {
          handleFileSelect(file, type);
        }
      }
    }
  });

  const handleFileSelect = (file: File, type: 'test' | 'reference' = 'test') => {
    onImageSelect(file, type);
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (type === 'test') {
        setTestPreview(result);
      } else {
        setReferencePreview(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleClear = (type: 'test' | 'reference' = 'test') => {
    onClearImage(type);
    if (type === 'test') {
      setTestPreview(null);
    } else {
      setReferencePreview(null);
    }
  };

  return (
    <div className="glass-card p-6">
      <h2 className="text-xl font-semibold text-white mb-6 flex items-center gap-2">
        {modelType === 'dual' ? (
          <Users className="w-6 h-6 text-neon-green" />
        ) : (
          <FileImage className="w-6 h-6 text-neon-green" />
        )}
        Upload Signature{modelType === 'dual' ? 's' : ''}
      </h2>

      {modelType === 'single' ? (
        <ImageUploadArea
          title="Test Signature"
          subtitle="Drop your signature here or click to browse"
          file={selectedImage}
          preview={testPreview}
          onFileSelect={(file) => handleFileSelect(file, 'test')}
          onClear={() => handleClear('test')}
          dragActive={dragActive === 'test'}
          onDragHandlers={createDragHandlers('test')}
        />
      ) : (
        <div className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <ImageUploadArea
              title="Reference Signature (Genuine)"
              subtitle="Upload known genuine signature"
              file={referenceImage || null}
              preview={referencePreview}
              onFileSelect={(file) => handleFileSelect(file, 'reference')}
              onClear={() => handleClear('reference')}
              dragActive={dragActive === 'reference'}
              onDragHandlers={createDragHandlers('reference')}
            />
            
            <ImageUploadArea
              title="Test Signature"
              subtitle="Upload signature to verify"
              file={selectedImage}
              preview={testPreview}
              onFileSelect={(file) => handleFileSelect(file, 'test')}
              onClear={() => handleClear('test')}
              dragActive={dragActive === 'test'}
              onDragHandlers={createDragHandlers('test')}
            />
          </div>
          
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
              <div>
                <h4 className="text-sm font-medium text-blue-300 mb-1">Siamese Network Requirements</h4>
                <p className="text-xs text-blue-200/80">
                  This model compares the test signature against a reference genuine signature. 
                  Both images are required for accurate verification.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}