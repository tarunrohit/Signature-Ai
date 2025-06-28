// src/utils/modelUtils.ts

export const getModelType = (modelId: string): 'single' | 'dual' => {
  if (modelId === 'siamesenet') {
    return 'dual';
  }
  return 'single';
};